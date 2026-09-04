"""
demo/threaded_pipeline_hw.py
==============================
Hardware-appropriate replacement for threaded_pipeline.py's
robot_thread_fn -- NOT a drop-in reuse of it.

Why this file exists (read before wiring anything):

sim's robot_thread_fn calls env.substep(u_safe) at a hardcoded 500Hz.
substep() is MujocoFrankaEnv-only: it advances exactly one 0.002s
physics tick, no trajectory smoothing, callable as fast as Python can
loop. FrankyHwEnv has NO substep() method -- calling it would raise
AttributeError immediately.

FrankyHwEnv.step() is a fundamentally different operation: it sends a
franky JointWaypointMotion command (which the robot's own controller
turns into a smooth trajectory with its own jerk/accel/vel limits) and
then self-paces with an internal time.sleep() to hit control_dt
(default 0.05s => ~20Hz). It is not a raw tick you can call as fast as
you want -- 20Hz is the real, native rate this bridge was built and
tested for (see robot/franky_hw.py's own docstring and Stage 1-3
validation in PROJECT_LOG.md).

Two consequences that shape the function below:

1. No hz=500 parameter here. The loop rate is env.control_dt, full
   stop. Passing a different hz would either be ignored (misleading)
   or would try to call env.step() faster than franky can actually
   execute a new waypoint, which is untested and not something to
   attempt casually on real hardware.

2. Do NOT add a second outer time.sleep(period - exec_time) on top of
   this, the way sim's robot_thread_fn does. env.step() ALREADY
   measures its own elapsed time and sleeps to hit control_dt
   internally (see FrankyHwEnv.step()). Adding another sleep at the
   same period on top would just be redundant, not harmful, but there
   is no reason to duplicate pacing logic that already exists and is
   already tested.

Everything else -- the HOCBF/CBF-QP math, feasibility logging,
conflict-factor detection, covariance steering -- is IDENTICAL to
sim's robot_thread_fn, and reuses the exact same cbf.barrier /
cbf.qp_solver calls. Only the "how do I actually move the robot and
how fast can I check again" part changes, because that is the only
part that is genuinely different about real hardware.

GPMP2 and MPPI threads are unchanged -- reuse gpmp2_thread_fn (or the
gpmp2_process_fn / gpmp2_bridge_thread_fn separate-process version)
and mppi_thread_fn from demo/threaded_pipeline.py exactly as they are.
Nothing about GPMP2's or MPPI's own multi-rate behavior depends on
which robot env is underneath them -- they only ever read/write
PipelineSharedState.
"""
from __future__ import annotations
import time
import numpy as np

from cbf.barrier import hocbf_lie_derivatives, closest_clearance
from cbf.qp_solver import detect_unsafe


def robot_thread_fn_hw(shared, dashboard_state, env, franka, barrier, qp, f_fn, g_fn,
                        conflict_mgr, cov_steer, feas_log, dof,
                        goal_threshold: float = 0.15):
    """
    Hardware version of threaded_pipeline.py's robot_thread_fn.

    env : FrankyHwEnv (NOT MujocoFrankaEnv -- this function will call
          env.step(), never env.substep()).

    Runs at env.control_dt's native pace (~20Hz for the default 0.05s),
    reported honestly via dashboard_state.robot_period_s -- no
    fabricated 500Hz number. Every other stage (HOCBF/CBF-QP,
    feasibility logging, conflict detection, covariance steering) is
    the same call, in the same order, as sim's robot_thread_fn.

    A real hardware fault (franky raising, caught by FrankyHwEnv's
    _safe_call and re-raised as RuntimeError) stops the WHOLE pipeline,
    not just this thread -- shared.stop is set so the GPMP2 and MPPI
    threads notice and exit too, matching main.py's fault-handling
    philosophy (a hardware fault is a stop-everything event, not
    something to silently continue past).
    """
    from cbf.barrier import closest_clearance as _closest_clearance  # local alias, mirrors sim file's import style
    period = env.control_dt  # the ONLY source of truth for this loop's rate -- see module docstring
    alpha_gamma = getattr(qp, "alpha_gamma", 100.0)
    step_idx = 0

    dashboard_state.set_thread_status(robot_period_s=period)

    while not shared.stop:
        if not shared.can_execute():
            dashboard_state.set_thread_status(robot_status="WAITING (holding at start pose)")
            time.sleep(period)
            continue

        dashboard_state.set_thread_status(robot_status="RUNNING")

        try:
            state = env.get_state()
        except RuntimeError as e:
            print(f"HARDWARE FAULT (get_state) — stopping pipeline: {e}")
            dashboard_state.set_thread_status(robot_status=f"FAULTED: {e}")
            shared.stop = True
            break

        q, qdot = state[:dof], state[dof:]
        shared.set_robot_state(q, qdot)
        x = np.concatenate([q, qdot])

        ee_centers, _ = franka.fk(q)

        # Same "stop actually moving once close enough" behavior as sim --
        # avoids continuing to stream waypoint commands once the goal is
        # already reached, which would just add wear/noise on real hardware
        # for no benefit.
        goal_err_check = None
        if dashboard_state.q_goal is not None:
            goal_err_check = float(np.max(np.abs(q - dashboard_state.q_goal)))
        if goal_err_check is not None and goal_err_check < goal_threshold:
            dashboard_state.set_thread_status(
                q=q, qdot=qdot, goal_error=goal_err_check, cbf_active=False,
            )
            time.sleep(period)
            continue

        u_mppi = shared.get_u_mppi()
        if u_mppi is not None:
            # ---- identical math to sim's robot_thread_fn from here down ----
            psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
            d_obs = _closest_clearance(franka.fk, franka.sphere_radii, barrier.sdf, q)
            unsafe = detect_unsafe(u_mppi, Lf_psi1, Lg_psi1, psi1, alpha_gamma=alpha_gamma,
                                    d_obstacle=d_obs, h0_physical=h0)
            qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)

            goal_err = None
            if dashboard_state.q_goal is not None:
                goal_err = float(np.max(np.abs(q - dashboard_state.q_goal)))

            dashboard_state.set_thread_status(
                cbf_active=qp_result.intervention_magnitude > 1e-6,
                safety_margin=h0, qp_intervention=qp_result.intervention_magnitude,
                q=q, qdot=qdot, goal_error=goal_err,
            )

            # ---- the one real difference: step(), not substep() ------------
            # This sends the command AND paces the loop to control_dt
            # internally (see FrankyHwEnv.step()) -- no extra sleep needed
            # here, unlike sim's substep()-based loop.
            try:
                env.step(qp_result.u_safe)
            except RuntimeError as e:
                print(f"HARDWARE FAULT (step) — stopping pipeline: {e}")
                dashboard_state.set_thread_status(robot_status=f"FAULTED: {e}")
                shared.stop = True
                try:
                    env.stop()
                except Exception:
                    pass
                break

            dashboard_state.append_ee_path(ee_centers[-1], cbf_active=qp_result.intervention_magnitude > 1e-6)
            feas_log.record(step_idx, unsafe, qp_result)

            ev = conflict_mgr.check_and_record(step_idx, x, qp_result.intervention, h0)
            if ev is not None:
                dashboard_state.push_conflict_marker(ee_centers[-1])
                shared.push_conflict_event(step_idx, ev)

            cov_steer.update_online(qp_result.intervention)
            if step_idx % 20 == 0:
                cov_steer.update_windowed()

            step_idx += 1
        else:
            # No MPPI output published yet -- nothing to execute this
            # iteration. Still pace to control_dt so this doesn't spin.
            time.sleep(period)
