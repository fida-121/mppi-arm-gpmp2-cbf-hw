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
             (reference - actual) EVERY cycle, immediately. This is
             fast/smooth for SMALL incremental reference changes (which
             is what the CBF-QP loop produces cycle-to-cycle), but is
             DANGEROUS for large jumps: a big instantaneous reference
             error produces a correspondingly large, fast, forceful
             correction. CONFIRMED BY DIRECT HARDWARE TESTING: an
             earlier version of this file called set_reference()
             directly inside reset() with a far-away target, and the
             robot moved "very rapidly with very high force and
             torque" -- a real safety incident, not a theoretical
             concern.

  THE RULE THIS FILE ENFORCES: reset() always uses the Ruckig-planned,
  distance-safe JointMotion. The JointImpedanceTrackingMotion used by
  step() is only ever started AFTER reset() has already placed the arm
  at the target pose -- so its first reference is always a zero (or
  near-zero) distance from the arm's actual position. NEVER call
  set_reference() with a target that could be far from the robot's
  current actual position.

SAFETY: every franky call that talks to the robot is wrapped; any
exception sets self.faulted=True and re-raises, so the calling loop
(main.py) stops rather than silently continuing on stale state.
"""
from __future__ import annotations
import time
import numpy as np
import franky

from robot.franka import DOF, Q_MIN, Q_MAX  # reuse existing joint limits


class FrankyHwEnv:
    def __init__(self, robot_ip: str = "172.16.0.2",
                 control_dt: float = 0.05,
                 dynamics_factor: float = 0.1,   # scalar in (0, 1] -- START LOW.
                 obstacle_center=(0.5, 0.0, 0.4), obstacle_radius=0.08,
                 stiffness: np.ndarray = None, damping: np.ndarray = None,
                 max_step_delta: float = 0.05):
        """
        stiffness/damping: (7,) arrays for the joint impedance controller
        used by step(). If None, franky's own defaults are used.
        max_step_delta: SAFETY LIMIT (rad). step() will raise rather than
        silently execute if u_star differs from the robot's current
        actual position by more than this, on ANY joint. This is a
        defense-in-depth guard against ever feeding
        JointImpedanceTrackingMotion a large jump (see module docstring
        for why that's dangerous) -- e.g. from a bad CBF-QP solution, a
        bug elsewhere in the pipeline, or a stale/wrong q0. Tune this
        down further once you've observed real cycle-to-cycle step
        sizes from the CBF-QP in practice.
        """
        self.control_dt = control_dt
        self.obstacle_center = np.array(obstacle_center)
        self.obstacle_radius = obstacle_radius
        self.faulted = False
        self.franka_model = None  # set externally by build_default_system()
        self._tracking_motion = None  # the persistent JointImpedanceTrackingMotion
        # used by step(); only started inside reset(), after the arm is
        # already at the reset target -- see module docstring.
        self._stiffness = stiffness
        self._damping = damping
        self._max_step_delta = max_step_delta

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
        SAFE for any distance, since it respects relative_dynamics_factor
        regardless of how far q0 is from the current position (unlike
        step()'s JointImpedanceTrackingMotion -- see module docstring).
        Blocks until arrival (JointMotion's own return_when_finished
        default, no polling loop needed here).

        After arrival, starts (or restarts) the JointImpedanceTrackingMotion
        used by step(), with its FIRST reference set to q0 -- i.e. the
        arm's actual current position -- so there is never a jump when
        step()-based tracking begins.

        qdot0 is accepted for interface parity with MujocoFrankaEnv but
        ignored (matches previous version's behavior).
        """
        q0 = np.asarray(q0, dtype=np.float64)

        # Stop any previously-running tracking motion before the Ruckig
        # move -- franky requires the control signal type not to change
        # mid-motion; stopping first avoids any ambiguity about which
        # controller is active.
        if self._tracking_motion is not None:
            try:
                self.robot.stop()
            except Exception:
                pass
            self._tracking_motion = None

        motion = franky.JointMotion(target=franky.JointState(position=q0))
        self._safe_call(self.robot.move, motion)  # blocking, Ruckig-planned

        # Now start the tracking controller used by step(), with its
        # first reference equal to where the arm actually is -- zero
        # distance, so no jump.
        self._tracking_motion = franky.JointImpedanceTrackingMotion(
            stiffness=self._stiffness, damping=self._damping,
        )
        self._safe_call(self.robot.move, self._tracking_motion, asynchronous=True)
        self._safe_call(self._tracking_motion.set_reference, franky.JointReference(q=q0))

    def step(self, u_star: np.ndarray):
        """
        u_star: (7,) desired joint POSITION, post CBF-QP filtering --
        same convention as MujocoFrankaEnv.step().

        Publishes u_star as the new reference for the already-running
        JointImpedanceTrackingMotion (started by reset()). Raises if
        u_star is more than max_step_delta away from the robot's CURRENT
        actual position on any joint -- see __init__ docstring and the
        module-level docstring for why large jumps through this
        controller are dangerous.
        """
        if self._tracking_motion is None:
            raise RuntimeError(
                "step() called before reset() -- the tracking motion was "
                "never started. Call reset(q0) first (this also protects "
                "against feeding a large, unplanned jump into the torque "
                "controller).")

        t0 = time.monotonic()
        u_star = np.clip(np.asarray(u_star, dtype=np.float64), Q_MIN, Q_MAX)

        q_now = self.get_state()[:7]
        delta = np.abs(u_star - q_now)
        if np.any(delta > self._max_step_delta):
            self.faulted = True
            raise RuntimeError(
                f"step(): u_star differs from current position by up to "
                f"{delta.max():.4f} rad (limit {self._max_step_delta}) -- "
                f"REFUSING to send this to the torque-based tracking "
                f"controller (see module docstring: large jumps through "
                f"JointImpedanceTrackingMotion cause fast, forceful "
                f"motion). u_star={u_star}, q_now={q_now}")

        self._safe_call(self._tracking_motion.set_reference, franky.JointReference(q=u_star))

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
        mounted gripper's tip -- confirmed ~0.107m offset via direct testing).
        Requires self.franka_model to be set externally (see
        build_default_system() in main.py); falls back to the raw flange
        pose (documented as inaccurate) if it hasn't been wired in."""
        if self.franka_model is not None:
            q = self.get_state()[:7]
            centers, _ = self.franka_model.fk(q)
            return centers[-1]
        pose = self._safe_call(lambda: self.robot.current_pose)
        return np.asarray(pose.end_effector_pose.translation, dtype=np.float64)

    def stop(self):
        """Explicitly stop the tracking motion. Safe to call even if no
        motion was ever started (e.g. reset() was never called)."""
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
