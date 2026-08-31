"""
robot/franky_hw.py
====================
Real-hardware drop-in replacement for MujocoFrankaEnv, backed by franky
(-> libfranka 0.9.2) instead of MuJoCo. Same interface as
robot/mujoco_env.py: reset(), step(), get_state(), ee_position().

CONTROL MODE / MOTION TYPE: joint position, streamed via
franky.JointImpedanceTrackingMotion + set_reference() each step() call.

WHY NOT JointWaypointMotion (previous version of this file): sending a
NEW JointWaypointMotion every control_dt, each targeting a ZERO-velocity
waypoint, causes visible stop-start/stutter motion -- confirmed by
direct hardware testing. Each waypoint is planned by Ruckig as a
decelerate-to-rest segment; replanning one every ~50ms means the robot
is repeatedly told to brake, then immediately re-accelerate toward a
new nearby target. JointImpedanceTrackingMotion instead starts ONE
torque-based impedance controller that runs continuously and simply
reads the latest reference each control cycle via set_reference() --
much closer to MuJoCo's continuous position-servo actuators in sim,
which is the behavior we're trying to match.

Every franky API call below was verified against `help()` output from
the actual installed franky build in the franky-hw Docker image (not
guessed from docs).

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
                 stiffness: np.ndarray = None, damping: np.ndarray = None):
        """
        stiffness/damping: (7,) arrays for the joint impedance controller.
        If None, franky's own defaults are used (unverified what those
        are -- START by testing with defaults, and only override if the
        default behavior feels too stiff/soft once you can observe real
        motion quality).
        """
        self.control_dt = control_dt
        self.obstacle_center = np.array(obstacle_center)
        self.obstacle_radius = obstacle_radius
        self.faulted = False
        self.franka_model = None  # set externally by build_default_system()
        self._motion = None  # the persistent JointImpedanceTrackingMotion,
        # created lazily on the first reset()/step() call once we know
        # the robot connected successfully.
        self._stiffness = stiffness
        self._damping = damping

        try:
            self.robot = franky.Robot(
                robot_ip,
                relative_dynamics_factor=franky.RelativeDynamicsFactor(dynamics_factor),
                realtime_config=franky.RealtimeConfig.Enforce,
            )
        except Exception as e:
            self.faulted = True
            raise RuntimeError(f"Failed to connect to robot at {robot_ip}: {e}")

    def _ensure_motion_started(self, q_initial: np.ndarray):
        """Starts the persistent JointImpedanceTrackingMotion if it isn't
        already running. Called internally by reset()/step() -- not
        meant to be called directly."""
        if self._motion is not None:
            return
        self._motion = franky.JointImpedanceTrackingMotion(
            stiffness=self._stiffness, damping=self._damping,
        )
        self._safe_call(self.robot.move, self._motion, asynchronous=True)
        # Publish an initial reference immediately so the controller has
        # something valid to track from the first control cycle --
        # get_reference() can return None before any reference is set.
        self._motion.set_reference(franky.JointReference(q=q_initial))

    def reset(self, q0: np.ndarray, qdot0: np.ndarray = None):
        """Moves to q0. Unlike the old JointWaypointMotion version, this
        does NOT block until arrival -- it starts (or reuses) the
        persistent tracking motion and sets q0 as the reference, then
        waits (polling get_state()) until the arm is close enough to q0
        before returning, so callers can still treat this as an
        effectively-blocking call like MujocoFrankaEnv.reset() is.
        qdot0 is accepted for interface parity but ignored (matches
        previous version's behavior)."""
        q0 = np.asarray(q0, dtype=np.float64)
        self._ensure_motion_started(q0)
        self._safe_call(self._motion.set_reference, franky.JointReference(q=q0))

        # Poll until close to target or timeout -- avoids returning
        # control to the caller while the arm is still mid-motion to q0.
        timeout_s = 10.0
        poll_dt = 0.05
        tol = 0.02  # rad, per-joint
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            q_now = self.get_state()[:7]
            if np.all(np.abs(q_now - q0) < tol):
                return
            time.sleep(poll_dt)
        raise RuntimeError(f"reset(): timed out waiting to reach q0={q0}, "
                            f"last q={self.get_state()[:7]}")

    def step(self, u_star: np.ndarray):
        """
        u_star: (7,) desired joint POSITION, post CBF-QP filtering --
        same convention as MujocoFrankaEnv.step().

        Publishes u_star as the new reference for the ALREADY-RUNNING
        JointImpedanceTrackingMotion -- no new motion object, no
        replanning, no forced deceleration each cycle. This is the fix
        for the stop-start stutter seen with the previous
        JointWaypointMotion-per-cycle approach.
        """
        t0 = time.monotonic()
        u_star = np.clip(np.asarray(u_star, dtype=np.float64), Q_MIN, Q_MAX)

        if self._motion is None:
            # step() called before reset() -- start the tracking motion
            # from the robot's current actual position rather than
            # failing outright.
            self._ensure_motion_started(self.get_state()[:7])

        self._safe_call(self._motion.set_reference, franky.JointReference(q=u_star))

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
        """Explicitly stop the tracking motion (e.g. at the end of a run).
        Not called automatically -- caller decides when the session is
        truly over. Safe to call even if no motion was ever started."""
        if self._motion is not None:
            try:
                self.robot.stop()
            except Exception:
                pass  # best-effort; don't mask an earlier real error
            self._motion = None

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
