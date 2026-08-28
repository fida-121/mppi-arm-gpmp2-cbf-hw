"""
robot/franky_hw.py
====================
Real-hardware drop-in replacement for MujocoFrankaEnv, backed by franky
(-> libfranka 0.9.2) instead of MuJoCo. Same interface as
robot/mujoco_env.py: reset(), step(), get_state(), ee_position().

CONTROL MODE: joint position (matches sim's position-servo actuators --
u_star from the CBF-QP is a DESIRED JOINT POSITION, not torque, exactly
as in mujoco_env.py's step() docstring).

Every franky.Robot/franky.JointState/franky.JointWaypointMotion call
below was verified against `help()` output from the actual installed
franky build in the franky-hw Docker image (not guessed from docs) --
see chat history for the raw help() dumps this was checked against.

SAFETY: every franky call that talks to the robot is wrapped; any
exception sets self.faulted=True and re-raises, so the calling loop
(main.py) stops rather than silently continuing on stale state. Wrap
env.step() in main.py's loop in a try/except RuntimeError before
running this on hardware -- see bottom of this file for the exact
patch.
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
                 # Scales velocity/acceleration/jerk together. Do NOT copy
                 # sim's aggressiveness -- raise this only after Stage 1-3
                 # staged testing (state-read, tiny motion, obstacle-present
                 # at reduced speed) have all passed cleanly.
                 obstacle_center=(0.5, 0.0, 0.4), obstacle_radius=0.08):
        self.control_dt = control_dt
        self.obstacle_center = np.array(obstacle_center)
        self.obstacle_radius = obstacle_radius
        self.faulted = False

        try:
            self.robot = franky.Robot(
                robot_ip,
                relative_dynamics_factor=franky.RelativeDynamicsFactor(dynamics_factor),
                realtime_config=franky.RealtimeConfig.Enforce,
                # Enforce (the library default) refuses to connect if RT
                # scheduling isn't actually available -- fail loud, not
                # silent. Confirmed available in your container (chrt -m
                # showed SCHED_FIFO 1/99), so this should succeed.
            )
        except Exception as e:
            self.faulted = True
            raise RuntimeError(f"Failed to connect to robot at {robot_ip}: {e}")

    def reset(self, q0: np.ndarray, qdot0: np.ndarray = None):
        """Blocking move to q0. Only call at a safe, slow start -- never
        inside the fast control loop. qdot0 is accepted for interface
        parity with MujocoFrankaEnv but ignored: you cannot command an
        initial velocity on real hardware the way sim can set qvel
        directly; the robot always starts this motion from rest."""
        target = franky.JointState(position=np.asarray(q0, dtype=np.float64))
        waypoint = franky.JointWaypoint(target=target)
        motion = franky.JointWaypointMotion([waypoint])
        self._safe_call(self.robot.move, motion)  # blocking (asynchronous=False default)

    def step(self, u_star: np.ndarray):
        """
        u_star: (7,) desired joint POSITION, post CBF-QP filtering --
        same convention as MujocoFrankaEnv.step().

        Streams this as a new waypoint each call. franky documents that
        if a motion is already running, the new one "is queued and takes
        over seamlessly" -- so repeated step() calls behave like a
        continuously-updated target, matching u_star being recomputed
        every control_dt in main.py's loop.
        """
        t0 = time.monotonic()
        u_star = np.clip(np.asarray(u_star, dtype=np.float64), Q_MIN, Q_MAX)

        target = franky.JointState(position=u_star)  # zero-velocity target;
        # this is a POSITION command, consistent with MuJoCo's position
        # servos in mujoco_env.py -- we are not commanding a joint
        # velocity here even though JointState *can* take one.
        waypoint = franky.JointWaypoint(target=target)
        motion = franky.JointWaypointMotion([waypoint], return_when_finished=False)
        # return_when_finished=False: keep holding the target rather than
        # ending the motion the instant it's reached -- avoids a gap
        # between "reached" and the next step() call's new command.

        self._safe_call(self.robot.move, motion, asynchronous=True)

        elapsed = time.monotonic() - t0
        remaining = self.control_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
        # NOTE: naive fixed-sleep loop timing, not a true RT scheduler.
        # Fine for first hardware validation at reduced speed; revisit
        # with a proper rate-limiter (e.g. time.monotonic()-based
        # accumulator, or moving the loop into franky's own callback via
        # register_callback) if timing precision becomes a measured
        # problem later.
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
        """NOTE: ee_body_name is accepted for interface parity with
        MujocoFrankaEnv but ignored here -- franky gives the real robot's
        Cartesian pose directly (no MJCF body names on hardware)."""
        pose = self._safe_call(lambda: self.robot.current_pose)
        return np.asarray(pose.end_effector_pose.translation, dtype=np.float64)

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


# ---------------------------------------------------------------------------
# REQUIRED patch to main.py before running this on hardware:
# the closed loop currently has no try/except around env.step(). Wrap it:
#
#     try:
#         env.step(qp_result.u_safe)
#     except RuntimeError as e:
#         print(f"HARDWARE FAULT -- stopping loop: {e}")
#         break
#
# Without this, a FrankyHwEnv fault propagates as an unhandled exception
# mid-loop instead of a clean, logged stop.
# ---------------------------------------------------------------------------
