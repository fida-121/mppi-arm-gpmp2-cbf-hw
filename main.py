"""
main.py
========
Stage 10: the full closed loop.

    GPMP2 -> MPPI -> CBF-QP -> Feasibility -> Covariance Steering
    -> Conflict Factors -> iSAM2 -> updated GPMP2 -> repeat

This file wires together every module built in planner/, controller/,
cbf/, robot/ into the single reactive loop described in the
architecture diagram at the top of the specification. It is the
executable entry point: `python main.py --config configs/default.yaml`
(or just `python main.py` for the built-in defaults below).
"""

from __future__ import annotations
import argparse
import numpy as np
import gtsam

from planner.factor_graph import SignedDistanceField
from planner.gpmp2_planner import GPMP2Planner
from planner.conflict_factor import ConflictFactorManager
from planner.isam_update import ISAM2Manager

from controller.mppi import MPPIController
from controller.covariance import CovarianceSteering

from cbf.barrier import (DistanceBarrier, lie_derivatives, alpha_linear, franka_dynamics,
                          closest_clearance, hocbf_lie_derivatives)
from cbf.qp_solver import CBFQPSolver, detect_unsafe
from cbf.feasibility import FeasibilityLog

from robot.franka import FrankaModel, DOF, Q_MIN, Q_MAX, TAU_MAX
from robot.mujoco_env import MujocoFrankaEnv


def build_default_system(mjcf_path: str, obstacle_center=(0.5, 0.0, 0.4),
                          obstacle_radius: float = 0.08, d_safe: float = 0.05,
                          use_hardware: bool = False, robot_ip: str = "172.16.0.2"):
    import mujoco

    # Build the MuJoCo model independently of which env we use -- needed
    # purely for FK (FrankaModel), never for physics stepping when on
    # hardware. Previously this borrowed env.model, which doesn't exist
    # on FrankyHwEnv (no MuJoCo model on real hardware).
    model = mujoco.MjModel.from_xml_path(mjcf_path)

    if use_hardware:
        from robot.franky_hw import FrankyHwEnv
        env = FrankyHwEnv(robot_ip=robot_ip, dynamics_factor=0.05,
                           obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
    else:
        env = MujocoFrankaEnv(mjcf_path=mjcf_path, obstacle_center=obstacle_center,
                               obstacle_radius=obstacle_radius)

    # IMPORTANT: FrankaModel gets its OWN MjData, separate from env.data
    # (when env is sim) or from anything hardware-side (when env is
    # FrankyHwEnv). FrankaModel.fk_batch() (called every MPPI step to
    # score rollout candidates) writes candidate joint positions directly
    # into self.data.qpos -- if that were env.data (shared with the
    # actual simulation), every MPPI cost evaluation would silently
    # overwrite the real robot state with a fake rollout position, and
    # env.step() could then advance physics from the wrong state. This
    # was a real bug in every earlier version of this file.
    franka = FrankaModel(model, mujoco.MjData(model))

    if use_hardware:
        # Wire franka into the hw env so ee_position() can compute the
        # real gripper-tip position via FK, matching sim's convention --
        # franky's own current_pose reports the flange, not the mounted
        # gripper's tip (confirmed ~0.107m offset via direct testing).
        env.franka_model = franka

    sdf = SignedDistanceField(
        obstacle_centers=np.array([env.obstacle_center]),
        obstacle_radii=np.array([env.obstacle_radius]))
    barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii,
                               sdf=sdf, dof=DOF, d_safe=d_safe)
    JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    qp = CBFQPSolver(m=DOF, u_min=JOINT_LOWER, u_max=JOINT_UPPER, alpha_gamma=100.0)
    return env, franka, sdf, barrier, qp

def run_closed_loop(mjcf_path: str, N_horizon: int = 30, dt: float = 0.05,
                     n_mppi_samples: int = 200, n_planning_cycles: int = 20,
                     rng_seed: int = 0, obstacle_center=(0.5, 0.0, 0.4),
                     obstacle_radius: float = 0.08, d_safe: float = 0.05,
                     tau_conflict: float = 0.05, tau_safe: float = 0.2,
                     lambda_cbf: float = 1.0, gpmp2_eps: float = 0.03,
                     gpmp2_sigma_obs: float = 0.02, on_stage=None,
                     use_hardware: bool = False, robot_ip: str = "172.16.0.2"):
    """
    on_stage: optional callable(stage_name: str, info: dict) -> None,
    invoked at each of the 8 pipeline stages with whatever data is
    relevant at that point (env, franka, theta_star, mppi_result,
    qp_result, feas_log, cov_steer, conflict event, etc.). Used by
    demo_mujoco.py to drive the live visualization WITHOUT duplicating
    any algorithm logic -- every call below is the exact same call this
    function always made; the hook only observes, never alters, the
    pipeline. Default None means zero behavior change for every
    existing caller (run_experiments.py, experiments/adaptive_loop.py).
    """
    def _emit(stage, **info):
        if on_stage is not None:
            on_stage(stage, info)

    rng = np.random.default_rng(rng_seed)
    env, franka, sdf, barrier, qp = build_default_system(
        mjcf_path, obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius, d_safe=d_safe,
        use_hardware=use_hardware, robot_ip=robot_ip)

    q0 = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Franka's
    # standard "ready" home pose -- verified safe on this robot via
    # stage_2c_joint.py (Stage 2c multi-waypoint test), not the
    # arbitrary all-zeros configuration.
    q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
    theta0 = np.concatenate([q0, np.zeros(DOF)])
    theta_goal = np.concatenate([q_goal, np.zeros(DOF)])
    if use_hardware:
        print(f"About to move to start pose: {q0}")
        input("Press Enter to confirm hardware start move (Ctrl+C to abort)...")
    env.reset(q0)

    Qc = 0.5 * np.eye(DOF)
    planner = GPMP2Planner(dof=DOF, dt=dt, Qc=Qc, sdf=sdf,
                            fk_fn=franka.fk, sphere_offsets=franka.sphere_radii,
                            eps=gpmp2_eps, sigma_obs=gpmp2_sigma_obs)
    # IMPORTANT: GPMP2Planner's own default eps=0.15 was LARGER than every
    # d_safe we tried (0.05, 0.10) -- meaning GPMP2's long-horizon reference
    # trajectory was already planning to stay further from the obstacle
    # than the CBF-QP required. That's why intervention stayed at 0.0000
    # regardless of lambda_cbf: the planner upstream had already over-
    # satisfied the CBF's requirement before MPPI or the QP ever got
    # involved. For the CBF/conflict-factor layers to have genuine work to
    # do, gpmp2_eps must be SMALLER than d_safe (GPMP2 only loosely avoids;
    # the CBF is the actual hard safety guarantee) -- default lowered to
    # 0.03 accordingly. Pass gpmp2_eps > d_safe to go back to the old
    # (redundant-CBF) behavior if you ever want that instead.

    # ---- Stage 1: initial GPMP2 solve -----------------------------------
    gpmp2_result = planner.plan(theta0, theta_goal, N=N_horizon)
    theta_star = gpmp2_result.theta_star  # (N+1, 2*dof)
    _emit("GPMP2", theta_star=theta_star, iterations=gpmp2_result.iterations,
          final_error=gpmp2_result.final_error, cycle=0)

    isam = ISAM2Manager()
    isam.initialize_from_batch(gpmp2_result.graph, gpmp2_result.values,
                                gpmp2_result.keys, dof2=2 * DOF)

    Sigma0 = 0.002**2 * np.eye(DOF)
    cov_steer = CovarianceSteering(n=DOF, Sigma0=Sigma0, eta=0.05, beta=0.3, W=20)
    conflict_mgr = ConflictFactorManager(tau_conflict=tau_conflict, tau_safe=tau_safe)
    feas_log = FeasibilityLog()

    mppi = MPPIController(lam=1.0, dt=dt, dof=DOF, sdf=sdf, eps_margin=0.15,
                           sigma_obs=0.02, lambda_cbf=lambda_cbf, fk_batch_fn=franka.fk_batch,
                           sphere_radii=franka.sphere_radii)
    # lambda_cbf weights MPPI's own soft obstacle-avoidance cost term
    # (Step 2.2). At the default 1.0, MPPI already avoids risky rollouts
    # on its own, so the CBF-QP rarely has anything left to correct --
    # for a demo scenario that shows genuine CBF/conflict-factor
    # intervention, try a much lower value (e.g. 0.05-0.1) so MPPI's raw
    # proposals are only weakly obstacle-aware and the hard CBF
    # constraint has to do real work.

    def barrier_batch_fn(V):
        N, T, _ = V.shape
        h = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                x = np.concatenate([V[i, t], np.zeros(DOF)])
                h[i, t] = barrier.forward(x)
        return h

    K_inv_diag = np.ones((N_horizon + 1, DOF))  # simplified diagonal precision

    # Real control-affine dynamics f(x), g(x) for the CBF Lie derivatives
    # (Step 3.2). Wired from robot/franka.py's MuJoCo-backed M(q), G(q),
    # C(q,qdot)qdot -- replaces the earlier f=0,g=I placeholder.
    def gravity_fn(q):
        return franka.gravity(q)

    def coriolis_fn(q, qdot):
        return franka.coriolis_times_qdot(q, qdot)

    def M_fn(q):
        return franka.mass_matrix(q)

    def f_fn(xi):
        fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return fi

    def g_fn(xi):
        _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return gi

    history = {"q": [], "u_mppi": [], "u_safe": [], "h": [], "conflicts": [],
               "cost_history": [], "goal_error": [], "dist": [],
               "cycle_end_goal_error": [], "cycle_end_min_clear": [],
               "replan_accepted": [], "replan_new_error": [], "replan_baseline_error": [],
               "rolled_back_to_best": []}

    # Outcome-based checkpoint: tracks the trajectory that produced the
    # BEST real goal_err actually achieved by the robot, as opposed to
    # the GPMP2-internal cost comparison above. GPMP2's own cost only
    # scores its own short lookahead window (smoothness + obstacle
    # avoidance + reaching the goal within N_horizon steps) -- a replan
    # can score "better" by that formula every single cycle while the
    # robot's REAL distance to the true goal still drifts worse over
    # many cycles, because GPMP2 never directly optimizes "get closer to
    # the goal by cycle 20" -- only "look locally reasonable over the
    # next N_horizon steps from wherever the robot currently is". This
    # was confirmed happening: a 30-cycle run had 0/30 replans REJECTED
    # by the cost-based guard, yet goal_err rose from 0.67 (cycle 11) to
    # 1.95 (cycle 28) before partially recovering. The checkpoint below
    # catches that: whenever a cycle ends meaningfully worse than the
    # best real goal_err seen so far, the NEXT replan is warm-started
    # from the best-known trajectory instead of the just-degraded one.
    best_goal_err = float("inf")
    best_theta_star = None
    rollback_tolerance = 0.05  # rad -- how much worse than best before rolling back
    global_t = 0
    for cycle in range(n_planning_cycles):
        theta_q_ref = theta_star[:, :DOF]  # position block used as MPPI mean

        for k in range(N_horizon):
            x = env.get_state()
            q, qdot = x[:DOF], x[DOF:]

            # ---- Stage 2/3: MPPI around GPMP2 prior ----------------------
            mppi_result = mppi.step(theta_q_ref[k:], cov_steer.Sigma_t,
                                      n_mppi_samples, K_inv_diag[k:], barrier_batch_fn, rng)
            _emit("MPPI", mppi_result=mppi_result, sampling_mean=theta_q_ref[k:],
                  Sigma_t=cov_steer.Sigma_t, cycle=cycle, k=k)
            u_mppi_pos = mppi_result.u_mppi[0]  # first control in the tape -- this is a
            # DESIRED JOINT POSITION (MPPI samples V_i = theta_GPMP2 + eps_i, and
            # theta_GPMP2 is a position trajectory), NOT a torque.

            # Computed-torque control: converts a DESIRED ACCELERATION into
            # properly mass-matrix-scaled torque per joint, instead of
            # guessing a Kp/Kd directly in torque-space (which was wrong
            # twice over: flat Kp=80 saturated the 12 Nm wrist joints;
            # capping Kp AT TAU_MAX per joint then made those same joints
            # too WEAK to track against gravity/inertia). Kp tapers from
            # base to wrist joints, roughly matching their torque budget;
            # Kd is critically damped (2*sqrt(Kp)) per joint.
            # u_mppi_pos IS the control now -- MuJoCo's own position-servo
            # actuators handle the torque conversion internally. No
            # computed-torque law needed; that entire approach was based
            # on a wrong assumption about the actuator model, confirmed
            # wrong by direct testing (diagnose_position_control.py).
            JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
            u_mppi = np.clip(u_mppi_pos, JOINT_LOWER, JOINT_UPPER)

            # ---- Stage 4: unsafe detection before QP ----------------------
            # HOCBF fix (see cbf/barrier.py::hocbf_lie_derivatives docstring):
            # h0(x)=d(q)-d_safe has relative degree 2 under second-order
            # dynamics, so the plain Lg_h0 from lie_derivatives() is
            # IDENTICALLY ZERO and the CBF-QP could never correct anything --
            # this was the actual cause of intervention staying at 0.0000
            # for an entire run even while h0 went negative. psi1/Lf_psi1/
            # Lg_psi1 are used for the real QP constraint; h0 is kept
            # separately for honest physical-distance logging.
            psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
            d_obs = closest_clearance(franka.fk, franka.sphere_radii, sdf, q)
            unsafe = detect_unsafe(u_mppi, Lf_psi1, Lg_psi1, psi1, alpha_gamma=100.0,
                                    d_obstacle=d_obs, h0_physical=h0)

            # ---- Stage 5: CBF-QP -------------------------------------------
            qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
            _emit("CBF-QP", qp_result=qp_result, unsafe=unsafe, u_mppi=u_mppi, cycle=cycle, k=k)

            # ---- Stage 6: execute u* only -----------------------------------
            try:
                env.step(qp_result.u_safe)
            except RuntimeError as e:
                print(f"HARDWARE FAULT — stopping loop: {e}")
                if use_hardware:
                    env.stop()
                return history, feas_log, cov_steer
            _emit("Robot Execution", q=q, qdot=qdot, u_safe=qp_result.u_safe,
                  ee_position=env.ee_position(), t=global_t, cycle=cycle, k=k)

            # ---- Stage 7: feasibility extraction -----------------------------
            feas_log.record(global_t, unsafe, qp_result)
            _emit("Feasibility Extraction", sample=feas_log.samples[-1], t=global_t)

            # ---- Stage 8: covariance steering --------------------------------
            cov_steer.update_online(qp_result.intervention)
            if k % cov_steer.W == 0:
                cov_steer.update_windowed()
            _emit("Covariance Steering", Sigma_t=cov_steer.Sigma_t,
                  eigenvalues=np.linalg.eigvalsh(cov_steer.Sigma_t), cycle=cycle, k=k)

            # ---- Stage 9: conflict factor detection --------------------------
            # Uses h0 (the physical barrier value), matching the spec's
            # h_theta(x_t) -- psi1 is a derived HOCBF surrogate, not itself
            # a meaningful "safety margin" for this trigger condition.
            ev = conflict_mgr.check_and_record(global_t, np.concatenate([q, qdot]),
                                                qp_result.intervention, h0)
            _emit("Conflict Factor", event=ev, n_conflicts_so_far=len(conflict_mgr.events),
                  tau_conflict=tau_conflict, tau_safe=tau_safe, h0=h0,
                  intervention=qp_result.intervention_magnitude, cycle=cycle, k=k)
            if ev is not None:
                key_i = gpmp2_result.keys[min(k, N_horizon)]
                cf = conflict_mgr.factor_for_event(key_i, ev)
                isam.add_factors_incremental([cf])
                theta_star = isam.current_trajectory()
                theta_q_ref = theta_star[:, :DOF]  # <-- feed the iSAM2-updated
                # trajectory back into MPPI's sampling mean immediately; without
                # this line the corrected theta_star was computed but MPPI kept
                # sampling around the stale pre-conflict reference for the rest
                # of the cycle.
                history["conflicts"].append(global_t)

            history["q"].append(q.copy())
            history["u_mppi"].append(u_mppi.copy())
            history["u_safe"].append(qp_result.u_safe.copy())
            history["h"].append(h0)
            history["cost_history"].append(float(np.mean(mppi_result.costs)))
            history["goal_error"].append(float(np.linalg.norm(q - q_goal, ord=np.inf)))
            history["dist"].append(d_obs)
            global_t += 1

        # ---- Per-cycle convergence diagnostic: goal error and worst-case
        # clearance seen so far, tracked once per outer GPMP2-replanning
        # cycle (not per control step) -- lets you see whether goal_err is
        # still dropping cycle-over-cycle or has plateaued, and whether
        # min_clear is stuck at the same value across cycles (a sign the
        # arm is stalled near the obstacle rather than still progressing).
        current_goal_err = history["goal_error"][-1]
        history["cycle_end_goal_error"].append(current_goal_err)
        history["cycle_end_min_clear"].append(float(np.min(history["dist"])))

        # ---- Outcome-based checkpoint update ---------------------------------
        # Update the best-known trajectory using the REAL goal_err just
        # achieved (not GPMP2's internal cost). This runs every cycle,
        # independent of whether the upcoming replan gets accepted or
        # rolled back below.
        if current_goal_err < best_goal_err:
            best_goal_err = current_goal_err
            best_theta_star = theta_star.copy()

        # ---- Stage 1 (re-solve at the top of each cycle): fresh GPMP2 ---------
        theta_star_old = theta_star
        theta0_cycle = np.concatenate([env.get_state()[:DOF], env.get_state()[DOF:]])

        # Choose the warm-start BASIS: normally the trajectory just
        # executed, but if this cycle ended meaningfully worse (by real
        # goal_err) than the best trajectory ever achieved, roll back and
        # warm-start from that best-known trajectory instead. This is a
        # SEPARATE, outcome-based guard from the GPMP2-cost-based one
        # below -- that one only checks GPMP2's own short-horizon math
        # and was confirmed to accept every replan (0/30 rejected) even
        # while real goal_err rose from 0.67 to 1.95 over cycles 11-28.
        # theta0 (the actual robot state) is always the true current
        # state regardless -- only the INITIAL GUESS for the rest of the
        # trajectory changes here, giving LM a historically-successful
        # route to refine from instead of continuing to build on a
        # degrading one.
        rolled_back = current_goal_err > best_goal_err + rollback_tolerance
        if rolled_back:
            cov_steer.reset()
        warm_start_basis = best_theta_star if rolled_back else theta_star
        warm_start = np.vstack([warm_start_basis[1:], warm_start_basis[-1:]])
        history["rolled_back_to_best"].append(rolled_back)

        gpmp2_result = planner.plan(theta0_cycle, theta_goal, N=N_horizon,
                                     init_trajectory=warm_start)

        # ---- Replan acceptance guard ----------------------------------------
        # Warm-starting (above) makes route-flipping LESS likely, but does
        # NOT guarantee the new solve is actually better -- LM is a local
        # optimizer and can still converge to a slightly worse point even
        # from a good initial guess. Before accepting the new trajectory,
        # score it against simply CONTINUING the warm-start basis
        # unchanged, using the SAME new graph (same theta0/goal/obstacle
        # factors) so the two numbers are directly comparable. Only
        # replace theta_star if the new solve actually reduces that cost;
        # otherwise keep the warm-start basis and let the next cycle try
        # again from whatever state that leads to.
        baseline_values = gtsam.Values()
        for key, row in zip(gpmp2_result.keys, warm_start):
            baseline_values.insert(key, row)
        baseline_error = gpmp2_result.graph.error(baseline_values)
        new_error = gpmp2_result.final_error
        accepted = new_error <= baseline_error

        if accepted:
            theta_star = gpmp2_result.theta_star
            isam.initialize_from_batch(gpmp2_result.graph, gpmp2_result.values,
                                        gpmp2_result.keys, dof2=2 * DOF)
        else:
            theta_star = warm_start  # reject: keep continuing the warm-start basis
            # isam is deliberately left untouched here -- it still holds a
            # tree consistent with warm_start (either the previous cycle's
            # trajectory shifted, or the best-known one on rollback), so
            # conflict-factor insertion in the upcoming cycle stays valid.

        history["replan_accepted"].append(accepted)
        history["replan_new_error"].append(float(new_error))
        history["replan_baseline_error"].append(float(baseline_error))
        _emit("Update GPMP2", theta_star_old=theta_star_old, theta_star_new=theta_star,
              iterations=gpmp2_result.iterations, final_error=new_error,
              baseline_error=baseline_error, accepted=accepted, rolled_back=rolled_back,
              cycle=cycle + 1)

    # Surface the best trajectory actually achieved during the run (by real
    # goal_err) as a first-class result, not just an internal rollback
    # detail. The last cycle's ending point can be worse than an earlier
    # point purely because the run happened to end mid-swing -- the best
    # checkpoint is at least as legitimate a "result of this run" and is
    # the trajectory the accept/rollback guard would return to anyway.
    history["best_goal_error"] = best_goal_err
    history["best_theta_star"] = best_theta_star
    if use_hardware:
        env.stop()

    return history, feas_log, cov_steer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=str, default="assets/panda.xml")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--robot-ip", type=str, default="172.16.0.2")
    args = parser.parse_args()
    run_closed_loop(mjcf_path=args.mjcf, n_planning_cycles=args.cycles,
                     use_hardware=args.hardware, robot_ip=args.robot_ip)
