"""
robot/franky_hw.py
====================
Real-hardware drop-in replacement for MujocoFrankaEnv, backed by franky
(-> libfranka 0.9.2) instead of MuJoCo. Same interface as
robot/mujoco_env.py: reset(), step(), get_state(), ee_position().

MOTION TYPES USED (IMPORTANT, read before modifying):
  reset() -> franky.JointMotion (Ruckig-planned, velocity/accel/jerk-
             limited, respects relative_dynamics_factor regardless of
             distance). SAFE for large/arbitrary jumps between poses.
  step()  -> franky.JointImpedanceTrackingMotion + set_reference().
             A torque-based spring-damper controller with NO built-in
             trajectory planning -- it applies torque proportional to
             (reference - actual) EVERY cycle, immediately. CONFIRMED BY
             DIRECT HARDWARE TESTING: an earlier version of this file
             called set_reference() directly inside reset() with a
             far-away target, and the robot moved "very rapidly with
             very high force and torque" -- a real safety incident.

  THE RULE THIS FILE ENFORCES: reset() always uses the Ruckig-planned,
  distance-safe JointMotion. JointImpedanceTrackingMotion is only ever
  started AFTER reset() has already placed the arm at the target pose.

STEP-SIZE SAFETY (velocity-based, not a flat threshold):
  A first version of this file used a single flat max_step_delta (e.g.
  0.05, then 0.1 rad) and REFUSED any step() call exceeding it. This
  was found to be too blunt in practice: GPMP2/MPPI's trajectory
  naturally needs LARGER position deltas per control_dt while
  accelerating from rest (e.g. right after reset()), then stabilizes --
  a real, expected pattern, not a danger signal. A flat refuse-outright
  guard kept faulting on legitimate early-cycle acceleration.

  This version instead SOFT-CLAMPS each requested step to Franka's real
  per-joint velocity limits (approximate official values below) scaled
  by control_dt -- i.e. it lets the robot move as fast as it physically
  safely can, but never faster, rather than erroring out on a
  legitimate-but-large request. A HARD fault is still raised only for
  requests that are wildly larger than even the physical velocity
  ceiling (hard_fault_multiplier x the per-joint velocity-derived
  limit), since that indicates a genuine bug (e.g. a bad CBF-QP
  solution, wrong units, stale state) rather than normal acceleration.

SAFETY: every franky call that talks to the robot is wrapped; any
exception sets self.faulted=True and re-raises, so the calling loop
(main.py) stops rather than silently continuing on stale state.
"""
from __future__ import annotations
import time
import numpy as np
import franky

from robot.franka import DOF, Q_MIN, Q_MAX  # reuse existing joint limits

# Approximate official Franka Panda joint velocity limits [rad/s].
# Joints 1-4 (base-side) are slower than joints 5-7 (wrist-side).
# These are conservative published figures -- if you have exact values
# from Franka's own documentation for your firmware, prefer those.
FRANKA_MAX_JOINT_VEL = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])


class FrankyHwEnv:
    def __init__(self, robot_ip: str = "172.16.0.2",
                 control_dt: float = 0.05,
                 dynamics_factor: float = 0.1,   # scalar in (0, 1] -- START LOW.
                 obstacle_center=(0.5, 0.0, 0.4), obstacle_radius=0.08,
                 stiffness: np.ndarray = None, damping: np.ndarray = None,
                 velocity_safety_factor: float = 0.5,
                 hard_fault_multiplier: float = 2.0):
        """
        stiffness/damping: (7,) arrays for the joint impedance controller
        used by step(). If None, franky's own defaults are used.

        velocity_safety_factor: fraction (0,1] of FRANKA_MAX_JOINT_VEL
        that step() is allowed to use, per joint, before soft-clamping.
        0.5 (default) means step() will clamp requested deltas to at
        most half the robot's rated max joint velocity -- conservative,
        raise gradually once you've observed real behavior at this
        setting.

        hard_fault_multiplier: if a requested delta exceeds
        velocity_safety_factor * FRANKA_MAX_JOINT_VEL by MORE than this
        multiplier, step() raises RuntimeError instead of clamping --
        this catches genuinely anomalous requests (bugs, bad solves),
        not normal acceleration.
        """
        self.control_dt = control_dt
        self.obstacle_center = np.array(obstacle_center)
        self.obstacle_radius = obstacle_radius
        self.faulted = False
        self.franka_model = None  # set externally by build_default_system()
        self._tracking_motion = None
        self._stiffness = stiffness
        self._damping = damping
        self._max_delta = velocity_safety_factor * FRANKA_MAX_JOINT_VEL * control_dt
        self._hard_fault_delta = self._max_delta * hard_fault_multiplier

        try:
            self.robot = franky.Robot(
                robot_ip,
                relative_dynamics_factor=franky.RelativeDynamicsFactor(dynamics_factor),
                realtime_config=franky.RealtimeConfig.Enforce,
            )
        except Exception as e:
            self.faulted = True
            raise RuntimeError(f"Failed to connect to robot at {robot_ip}: {e}")

    def reset(self, q0: np.ndarray, qdot0: np.ndarray = None):
        """
        Moves to q0 using a Ruckig-planned, jerk-limited JointMotion --
        SAFE for any distance. Blocks until arrival. After arrival,
        starts the JointImpedanceTrackingMotion used by step(), with its
        FIRST reference set to q0 (the arm's actual current position),
        so there is never a jump when step()-based tracking begins.
        qdot0 accepted for interface parity but ignored.
        """
        q0 = np.asarray(q0, dtype=np.float64)

        if self._tracking_motion is not None:
            try:
                self.robot.stop()
            except Exception:
                pass
            self._tracking_motion = None

        motion = franky.JointMotion(target=franky.JointState(position=q0))
        self._safe_call(self.robot.move, motion)  # blocking, Ruckig-planned

        self._tracking_motion = franky.JointImpedanceTrackingMotion(
            stiffness=self._stiffness, damping=self._damping,
        )
        self._safe_call(self.robot.move, self._tracking_motion, asynchronous=True)
        self._safe_call(self._tracking_motion.set_reference, franky.JointReference(q=q0))

    def step(self, u_star: np.ndarray):
        """
        u_star: (7,) desired joint POSITION, post CBF-QP filtering --
        same convention as MujocoFrankaEnv.step().

        Requested deltas beyond the velocity-derived soft limit
        (self._max_delta) are CLAMPED per-joint to that limit, not
        refused -- lets the robot move as fast as it safely can even
        during legitimate acceleration from rest. Only deltas beyond
        self._hard_fault_delta (a much larger, genuinely-anomalous
        threshold) cause a hard fault.
        """
        if self._tracking_motion is None:
            raise RuntimeError(
                "step() called before reset() -- call reset(q0) first.")

        t0 = time.monotonic()
        u_star = np.clip(np.asarray(u_star, dtype=np.float64), Q_MIN, Q_MAX)

        q_now = self.get_state()[:7]
        delta = u_star - q_now
        abs_delta = np.abs(delta)

        if np.any(abs_delta > self._hard_fault_delta):
            self.faulted = True
            raise RuntimeError(
                f"step(): u_star differs from current position by up to "
                f"{abs_delta.max():.4f} rad, exceeding the HARD fault "
                f"threshold ({self._hard_fault_delta.max():.4f} rad) -- "
                f"this is well beyond normal acceleration and likely "
                f"indicates a bug (bad CBF-QP solution, stale state, "
                f"wrong units, etc). REFUSING. u_star={u_star}, "
                f"q_now={q_now}")

        # Soft-clamp any joint exceeding the velocity-derived limit --
        # move as fast as safely possible instead of refusing outright.
        over_limit = abs_delta > self._max_delta
        if np.any(over_limit):
            direction = np.sign(delta)
            u_star = np.where(over_limit, q_now + direction * self._max_delta, u_star)

        target = franky.JointReference(q=u_star)
        self._safe_call(self._tracking_motion.set_reference, target)

        elapsed = time.monotonic() - t0
        remaining = self.control_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
        return self.get_state()

    def get_state(self):
        """Returns [q_arm (7,), qdot_arm (7,)] -- same shape/order as
        MujocoFrankaEnv.get_state()."""
        state = self._safe_call(lambda: self.robot.current_joint_state)
        q = np.asarray(state.position, dtype=np.float64).reshape(-1)
        qdot = np.asarray(state.velocity, dtype=np.float64).reshape(-1)
        self._sanity_check(q, qdot)
        return np.concatenate([q, qdot])

    def ee_position(self, ee_body_name: str = "hand") -> np.ndarray:
        """Returns the gripper-tip position via FK (matching sim's convention),
        NOT franky's raw current_pose (which reports the flange, not the
        mounted gripper's tip -- confirmed ~0.107m offset via direct testing)."""
        if self.franka_model is not None:
            q = self.get_state()[:7]
            centers, _ = self.franka_model.fk(q)
            return centers[-1]
        pose = self._safe_call(lambda: self.robot.current_pose)
        return np.asarray(pose.end_effector_pose.translation, dtype=np.float64)

    def stop(self):
        """Explicitly stop the tracking motion. Safe to call even if no
        motion was ever started."""
        if self._tracking_motion is not None:
            try:
                self.robot.stop()
            except Exception:
                pass
            self._tracking_motion = None

    # ---- safety helpers ---------------------------------------------------
    def _safe_call(self, fn, *args, **kwargs):
        if self.faulted:
            raise RuntimeError("FrankyHwEnv already faulted -- refusing further commands.")
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self.faulted = True
            raise RuntimeError(f"franky/robot error -- STOPPING: {e}") from e

    def _sanity_check(self, q, qdot):
        if np.any(np.isnan(q)) or np.any(np.isnan(qdot)):
            self.faulted = True
            raise RuntimeError("NaN in robot state -- STOPPING.")
        if np.any(q < Q_MIN - 0.05) or np.any(q > Q_MAX + 0.05):
            self.faulted = True
            raise RuntimeError(f"Joint position outside limits (with margin): {q} -- STOPPING.")
