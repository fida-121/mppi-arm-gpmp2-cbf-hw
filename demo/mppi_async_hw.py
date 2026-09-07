"""
demo/mppi_async_hw.py
========================
Two-thread architecture, NOT the earlier three-thread
(GPMP2 + MPPI + robot) design in threaded_pipeline_hw.py:

  1. Control thread (~20Hz, matching FrankyHwEnv.control_dt): reads the
     latest published control tape, runs the CBF-QP safety layer
     (~1.4ms), and calls env.step() -- the ONLY thread that ever
     touches `env`. If MPPI hasn't published a fresh tape since the
     last tick, it advances further along the previous tape instead of
     stalling.

  2. Background MPPI thread: infinite loop, no hardware clock. Reads
     the latest published robot state, runs one full MPPI rollout
     optimization, and publishes the resulting tape back. Takes as
     long as it takes -- if that's slower than 20Hz (very likely with
     this codebase's current unvectorized barrier_batch_fn -- see the
     "Known limitation" note below), the control thread simply keeps
     walking the older tape rather than blocking on a new one.

GPMP2 is intentionally NOT a third thread here. It is solved ONCE,
upfront, to produce a fixed reference trajectory theta_star. The MPPI
thread repeatedly finds where the robot currently is along that fixed
reference and samples/perturbs the next N_lookahead points of it. This
matches exactly what was asked for ("separate only MPPI into a
background thread") -- GPMP2 replanning and iSAM2/conflict-factor
online correction from the original 3-stage design are deliberately
out of scope here, not silently dropped. If the robot drifts far
enough from theta_star over a long run that this matters, that's the
signal to add GPMP2 back as a periodic background re-solve -- not
something this file tries to solve now.

Thread safety: everything passed between the two threads (robot state,
control tape, the compute-time EMA used for latency compensation)
lives in SharedMPPIState, behind one threading.Lock. Never read/write
those fields directly -- always go through the publish_*/get_*
methods below.

Known limitation carried over from main.py: barrier_batch_fn as
currently written in this codebase is an unvectorized, per-sample
Python double loop (see the earlier profiling discussion) and is
likely the dominant cost of one MPPI rollout. That's exactly why this
two-thread split helps: however slow that loop is, it no longer blocks
env.step() calls, which now run strictly at their own ~20Hz pace
regardless of MPPI's cadence. It does NOT make MPPI itself faster --
the robot may be following a tape that's a second or more stale. If
that staleness turns out to matter in practice, vectorizing
barrier_batch_fn (not threading) is the actual fix for that part.
"""
from __future__ import annotations
import os
import sys
import threading
import time
import numpy as np

# demo/ is a package only ever imported by root-level scripts before now
# (demo_dashboard.py etc.) -- this file is meant to be runnable directly
# (`python3 demo/mppi_async_hw.py`), which puts demo/'s own directory on
# sys.path[0], NOT the repo root where cbf/, planner/, controller/,
# robot/, main.py actually live. Without this, `python3 demo/mppi_async_hw.py`
# fails with ModuleNotFoundError: No module named 'cbf' even though
# `python3 -m demo.mppi_async_hw` (run from repo root) works fine, since
# -m puts the current working directory on sys.path instead. Adding the
# repo root explicitly here makes both invocations work.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cbf.barrier import hocbf_lie_derivatives, closest_clearance, franka_dynamics
from cbf.qp_solver import detect_unsafe
from cbf.feasibility import FeasibilityLog
from planner.gpmp2_planner import GPMP2Planner
from planner.factor_graph import SignedDistanceField
from controller.mppi import MPPIController

from main import build_default_system, save_results, _build_summary
from robot.franka import DOF


class SharedMPPIState:
    """The only channel between the two threads. Every field access goes
    through a method below that takes self._lock -- no direct attribute
    access from outside this class."""

    def __init__(self, dof: int, initial_tape: np.ndarray, control_dt: float):
        self._lock = threading.Lock()
        self.dof = dof
        self.control_dt = control_dt
        self.tape = initial_tape          # (T, dof) position tape, row 0 = "next command"
        self.tape_time = time.perf_counter()
        self.robot_state = None           # (q, qdot) or None until control thread publishes once
        # Seed the compute-time estimate at one control_dt rather than 0 --
        # an initial guess of "zero delay" would under-compensate for
        # latency on the very first MPPI iteration.
        self.mppi_compute_time_ema = control_dt
        self.last_mppi_compute_time = float("nan")  # no rollout completed yet
        self.last_mppi_cost = float("nan")
        self.stop = False

    def publish_robot_state(self, q: np.ndarray, qdot: np.ndarray):
        with self._lock:
            self.robot_state = (q.copy(), qdot.copy())

    def get_robot_state(self):
        with self._lock:
            if self.robot_state is None:
                return None, None
            q, qdot = self.robot_state
            return q.copy(), qdot.copy()

    def publish_tape(self, tape: np.ndarray, compute_time: float, cost: float):
        with self._lock:
            self.tape = tape
            self.tape_time = time.perf_counter()
            # Exponential moving average -- smooths out one-off slow/fast
            # MPPI iterations so latency compensation doesn't jitter.
            self.mppi_compute_time_ema = 0.7 * self.mppi_compute_time_ema + 0.3 * compute_time
            self.last_mppi_compute_time = compute_time
            self.last_mppi_cost = cost

    def get_tape_index(self):
        """How far along the current tape the control thread should be,
        based on how long ago it was published -- this IS the "shift
        along the previously computed trajectory" behavior when MPPI
        hasn't finished a new one yet."""
        with self._lock:
            age = time.perf_counter() - self.tape_time
            idx = int(age / self.control_dt)
            return self.tape[min(idx, len(self.tape) - 1)]

    def get_expected_delay(self) -> float:
        with self._lock:
            return self.mppi_compute_time_ema

    def get_last_mppi_stats(self):
        """Returns (last_mppi_compute_time, last_mppi_cost) -- the RAW
        duration/cost of the most recently completed rollout, not the
        EMA (which is smoothed and only meant for latency compensation).
        Used by control_thread_fn to fill the t_mppi/cost_history columns
        with "the most recently known value as of this tick" rather than
        a per-tick measurement that doesn't exist in this architecture --
        see the note in control_thread_fn for why that's the honest thing
        to log here."""
        with self._lock:
            return self.last_mppi_compute_time, self.last_mppi_cost


def control_thread_fn(shared: SharedMPPIState, env, franka, barrier, qp, f_fn, g_fn,
                       dof: int, feas_log: FeasibilityLog, history: dict,
                       q_goal: np.ndarray, goal_threshold: float, run_start_time: float):
    """The ONLY thread that touches `env`. Runs at env.control_dt's real
    native pace -- env.step() self-paces internally (see franky_hw.py),
    so no extra time.sleep() is added here."""
    alpha_gamma = getattr(qp, "alpha_gamma", 100.0)
    step_idx = 0

    while not shared.stop:
        try:
            state = env.get_state()
        except RuntimeError as e:
            print(f"HARDWARE FAULT (get_state) — stopping pipeline: {e}")
            shared.stop = True
            break

        q, qdot = state[:dof], state[dof:]
        shared.publish_robot_state(q, qdot)
        x = np.concatenate([q, qdot])

        goal_error = float(np.max(np.abs(q - q_goal)))
        # NOTE: previously this branch did `continue` here -- skipping CBF-QP,
        # env.step(), AND the entire history-logging block below the moment
        # goal_error first dipped under goal_threshold. That meant the run's
        # monitor loop (in run_mppi_async_hardware) only ever saw a single
        # instantaneous crossing, never confirmation the arm actually STAYED
        # near the goal -- and if goal_error ticked back up afterward (as it
        # did repeatedly in testing, 0.06 -> 0.3+ -> ...), nothing kept
        # logging to show that. Removed: every tick now runs the full
        # CBF-QP/step/log sequence regardless of goal_error, so the monitor
        # loop's hysteresis check (added there) has real, continuous data to
        # confirm settling on, and the arm is never left executing a stale
        # last command while unmonitored. The extra QP solve at goal is
        # negligible (t_cbf averages ~5.6ms, well inside the ~50ms budget).

        u_mppi = shared.get_tape_index()

        psi1, Lf_psi1, Lg_psi1, h0 = hocbf_lie_derivatives(barrier, x, f_fn, g_fn, alpha0=1.0)
        d_obs = closest_clearance(franka.fk, franka.sphere_radii, barrier.sdf, q)
        unsafe = detect_unsafe(u_mppi, Lf_psi1, Lg_psi1, psi1, alpha_gamma=alpha_gamma,
                                d_obstacle=d_obs, h0_physical=h0)

        # ---- t_cbf: genuinely per-tick in this architecture, unlike t_mppi
        # below -- CBF-QP runs on THIS thread, once per control tick, so this
        # is a real per-tick measurement, not a broadcast of a stale value.
        t_cbf_start = time.perf_counter()
        qp_result = qp.solve(u_mppi, Lf_psi1, Lg_psi1, psi1, h0_physical=h0)
        t_cbf = time.perf_counter() - t_cbf_start

        # ---- t_robot: also genuinely per-tick -- env.step() runs on THIS
        # thread, once per control tick.
        t_robot_start = time.perf_counter()
        try:
            env.step(qp_result.u_safe)
        except RuntimeError as e:
            print(f"HARDWARE FAULT (step) — stopping pipeline: {e}")
            shared.stop = True
            try:
                env.stop()
            except Exception:
                pass
            break
        t_robot = time.perf_counter() - t_robot_start

        # ---- t_mppi / cost_history: NOT a per-tick measurement in this
        # architecture -- MPPI runs on its own thread at its own decoupled
        # rate (see module docstring). Rather than log NaN (schema-
        # compatible but useless) or nothing (schema-incompatible with
        # main.py's save_results/CSV writer, which expects one entry per
        # timestep), broadcast the most recently completed rollout's
        # actual duration/cost onto every control tick until a newer one
        # replaces it. This is honestly what it is: "how expensive/costly
        # was the tape currently being followed, when it was computed" --
        # not "the MPPI call for this tick", because no such thing exists
        # here.
        t_mppi, mppi_cost = shared.get_last_mppi_stats()

        feas_log.record(step_idx, unsafe, qp_result)
        history["q"].append(q.copy())
        history["u_mppi"].append(u_mppi.copy())
        history["u_safe"].append(qp_result.u_safe.copy())
        history["h"].append(h0)
        history["goal_error"].append(goal_error)
        history["dist"].append(d_obs)
        history["timestep"].append(step_idx)
        history["real_time"].append(time.perf_counter() - run_start_time)
        history["cost_history"].append(mppi_cost)
        history["t_mppi"].append(t_mppi)
        history["t_cbf"].append(t_cbf)
        history["t_robot"].append(t_robot)
        history["intervention_magnitude"].append(float(qp_result.intervention_magnitude))
        step_idx += 1


def mppi_thread_fn(shared: SharedMPPIState, mppi: MPPIController, theta_ref_full: np.ndarray,
                    dof: int, n_lookahead: int, n_mppi_samples: int, barrier_batch_fn,
                    rng: np.random.Generator, history: dict, cov_steer=None):
    """No hardware clock. Loops as fast as one MPPI rollout takes,
    which -- see module docstring -- is likely to be well slower than
    20Hz with this codebase's current barrier_batch_fn. That's fine:
    it just means the control thread walks a somewhat stale tape
    between updates, which is exactly the tradeoff this architecture
    is meant to make explicit rather than hide."""
    last_idx = 0  # monotonic progress marker along theta_ref_full -- see the
    # note at its use below for why this can only increase, never decrease.
    while not shared.stop:
        q, qdot = shared.get_robot_state()
        if q is None:
            time.sleep(0.01)
            continue

        t0 = time.perf_counter()

        # ---- latency compensation -------------------------------------
        # By the time this rollout finishes and gets published, the real
        # robot will have moved roughly `expected_delay` further along.
        # Roll the starting state forward by that much (simple constant-
        # velocity extrapolation) before picking the reference window, so
        # the tape MPPI hands back is planned from where the robot will
        # actually BE, not where it was when this iteration started.
        expected_delay = shared.get_expected_delay()
        q_predicted = q + qdot * expected_delay

        # Find where the (predicted) current state sits along the fixed
        # GPMP2 reference, and sample around the next n_lookahead points
        # of it -- this replaces per-cycle GPMP2 replanning, which is out
        # of scope for this file (see module docstring).
        #
        # IMPORTANT: search only from last_idx onward, never the whole
        # path. Searching globally (the original version) let idx snap
        # BACKWARD near the goal -- if a noisy/predicted state happened to
        # sit closer (in raw joint-space distance) to an earlier waypoint
        # than to the true next one, the next tape would pull the arm back
        # along its own path instead of holding position. In testing this
        # produced repeated goal_error overshoot-then-retreat cycles
        # (0.06 -> 0.3+ -> 0.06 -> ...) that never settled. Restricting the
        # search to theta_ref_full[last_idx:] makes idx monotonically
        # non-decreasing -- the reference can only move forward or hold,
        # never backward.
        local_dists = np.linalg.norm(theta_ref_full[last_idx:] - q_predicted, axis=1)
        idx = last_idx + int(np.argmin(local_dists))
        last_idx = idx
        ref_window = theta_ref_full[idx: idx + n_lookahead]
        if len(ref_window) < 2:
            # At/near the end of the reference -- hold the final pose so
            # MPPI still has a valid window to sample around.
            ref_window = np.tile(theta_ref_full[-1], (n_lookahead, 1))

        sigma = cov_steer.Sigma_t if cov_steer is not None else (0.05 ** 2) * np.eye(dof)
        K_inv_diag = np.ones((len(ref_window), dof))

        mppi_result = mppi.step(ref_window, sigma, n_mppi_samples, K_inv_diag,
                                 barrier_batch_fn, rng)
        compute_time = time.perf_counter() - t0
        mean_cost = float(np.mean(mppi_result.costs))

        shared.publish_tape(mppi_result.u_mppi, compute_time, mean_cost)
        # Separate, genuine per-rollout log (one entry per MPPI iteration,
        # NOT per control tick) -- this is the one to use if you want the
        # real MPPI-rate timing/cost picture rather than the per-tick
        # broadcast in history["t_mppi"]/history["cost_history"].
        history.setdefault("mppi_cost_history", []).append(mean_cost)
        history.setdefault("mppi_compute_time", []).append(compute_time)

        if cov_steer is not None:
            cov_steer.update_online(np.zeros(dof))  # no per-tick intervention signal available
            # here (CBF-QP runs on the control thread, not this one) --
            # covariance steering's online update is effectively inert in
            # this configuration. Left in only so cov_steer's windowed
            # update doesn't error; the intervention-driven shrink/grow
            # behavior described in controller/covariance.py does not
            # apply in this two-thread architecture without additionally
            # publishing intervention magnitude back from the control
            # thread. Flagging this explicitly rather than silently
            # pretending covariance steering is doing its original job.


def run_mppi_async_hardware(mjcf_path: str, robot_ip: str = "172.16.0.2",
                             obstacle_center=(0.5, 0.0, 0.4), obstacle_radius: float = 0.08,
                             d_safe: float = 0.10, n_lookahead: int = 10,
                             n_mppi_samples: int = 200, gpmp2_N: int = 40,
                             lambda_cbf: float = 1.0,
                             goal_threshold: float = 0.05, max_duration_s: float = 60.0,
                             rng_seed: int = 0):
    """Entry point wiring the two threads together. GPMP2 is solved once,
    here, before either thread starts -- see module docstring for why
    this isn't itself a third thread.

    lambda_cbf: weights MPPI's own soft obstacle-avoidance cost term (same
    meaning as in main.py's run_closed_loop) -- lower values leave MPPI's
    raw proposals only weakly obstacle-aware, so the CBF-QP on the control
    thread has real work to do. Now an actual parameter here (previously
    hardcoded to 1.0 inside this function)."""
    rng = np.random.default_rng(rng_seed)
    env, franka, sdf, barrier, qp = build_default_system(
        mjcf_path, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius,
        d_safe=d_safe, use_hardware=True, robot_ip=robot_ip)

    q0 = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])
    theta0 = np.concatenate([q0, np.zeros(DOF)])
    theta_goal = np.concatenate([q_goal, np.zeros(DOF)])

    print(f"About to move to start pose: {q0}")
    input("Press Enter to confirm hardware start move (Ctrl+C to abort)...")
    env.reset(q0)

    # ---- one-time GPMP2 solve -- the fixed reference for the whole run ----
    Qc = 0.5 * np.eye(DOF)
    planner = GPMP2Planner(dof=DOF, dt=0.05, Qc=Qc, sdf=sdf, fk_fn=franka.fk,
                            sphere_offsets=franka.sphere_radii, eps=0.03, sigma_obs=0.02)
    t_gpmp2_start = time.perf_counter()
    gpmp2_result = planner.plan(theta0, theta_goal, N=gpmp2_N)
    t_gpmp2_initial = time.perf_counter() - t_gpmp2_start
    theta_ref_full = gpmp2_result.theta_star[:, :DOF]

    def gravity_fn(q): return franka.gravity(q)
    def coriolis_fn(q, qdot): return franka.coriolis_times_qdot(q, qdot)
    def M_fn(q): return franka.mass_matrix(q)
    def f_fn(xi):
        fi, _ = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return fi
    def g_fn(xi):
        _, gi = franka_dynamics(xi, DOF, gravity_fn, coriolis_fn, M_fn)
        return gi

    def barrier_batch_fn(V):
        N, T, _ = V.shape
        h = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                x = np.concatenate([V[i, t], np.zeros(DOF)])
                h[i, t] = barrier.forward(x)
        return h
    # ^ same unvectorized function as main.py -- see module docstring's
    # "Known limitation" note. Not fixed here; out of scope for this change.

    mppi = MPPIController(lam=1.0, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                           sigma_obs=0.02, lambda_cbf=lambda_cbf, fk_batch_fn=franka.fk_batch_vectorized,
                           sphere_radii=franka.sphere_radii)
    # ^ matches main.py's current build_default_system/run_closed_loop, which
    # now uses fk_batch_vectorized rather than the older fk_batch.

    feas_log = FeasibilityLog()
    history = {"q": [], "u_mppi": [], "u_safe": [], "h": [], "goal_error": [], "dist": [],
               "cost_history": [], "timestep": [], "real_time": [], "conflicts": [],
               # --- timing instrumentation, matching main.py's schema ---------
               "t_mppi": [], "t_cbf": [], "t_robot": [],
               "t_gpmp2_per_cycle": [t_gpmp2_initial],  # single one-time solve -- see module docstring
               "intervention_magnitude": []}
    # "conflicts" kept as an empty list for save_results()/CSV compatibility --
    # this architecture has no conflict-factor detection (see module docstring).

    shared = SharedMPPIState(dof=DOF, initial_tape=theta_ref_full[:n_lookahead], control_dt=env.control_dt)
    shared.publish_robot_state(q0, np.zeros(DOF))

    run_start_time = time.perf_counter()

    ctrl_thread = threading.Thread(
        target=control_thread_fn,
        args=(shared, env, franka, barrier, qp, f_fn, g_fn, DOF, feas_log, history,
              q_goal, goal_threshold, run_start_time),
        daemon=True)
    mppi_thread = threading.Thread(
        target=mppi_thread_fn,
        args=(shared, mppi, theta_ref_full, DOF, n_lookahead, n_mppi_samples,
              barrier_batch_fn, rng, history, None),
        daemon=True)

    ctrl_thread.start()
    mppi_thread.start()

    try:
        start = time.perf_counter()
        # Require SETTLE_TICKS consecutive readings under goal_threshold,
        # not just the single most-recent one -- the previous single-reading
        # check flagged "goal reached" on the first instantaneous dip below
        # threshold, but goal_error was observed oscillating back up above
        # it repeatedly afterward (see the fix in control_thread_fn/
        # mppi_thread_fn above for why). This check now confirms the arm has
        # actually settled, not just briefly passed through the target.
        settle_ticks = 10  # ~0.5s of consecutive good readings at 20Hz
        while time.perf_counter() - start < max_duration_s and not shared.stop:
            time.sleep(0.5)
            recent = history["goal_error"][-settle_ticks:]
            if len(recent) >= settle_ticks and all(e < goal_threshold for e in recent):
                print(f"Goal reached and settled (goal_error={recent[-1]:.4f}, "
                      f"last {settle_ticks} ticks all under {goal_threshold}).")
                break
    except KeyboardInterrupt:
        print("Interrupted -- stopping threads.")
    finally:
        shared.stop = True
        # Stop the ROBOT as soon as the thread that actually talks to it
        # (ctrl_thread) has exited -- do NOT wait on mppi_thread first.
        # ctrl_thread checks shared.stop every ~control_dt (~50ms), so it
        # should join almost immediately. mppi_thread can be mid-rollout
        # (with the current unvectorized barrier_batch_fn, a single
        # rollout can easily exceed a couple of seconds) and only checks
        # shared.stop between rollouts -- but since it never touches
        # `env`, there is no safety reason to block the hardware-stop on
        # it. Gating env.stop() on both joins (the previous order) meant
        # the actual stop-the-robot command could be delayed by however
        # long MPPI's in-flight rollout took, which is the wrong
        # priority for a safety-relevant stop.
        ctrl_thread.join(timeout=2.0)
        if ctrl_thread.is_alive():
            print("WARNING: control thread did not exit within 2s -- "
                  "stopping the robot anyway.")
        try:
            env.stop()
        except Exception:
            pass
        # mppi_thread is daemon=True and touches no hardware -- let it
        # finish its current rollout in the background rather than
        # blocking shutdown on it. A short join here is just to give it
        # a chance to exit cleanly for a tidy log; it is NOT a
        # correctness or safety requirement the way ctrl_thread's is.
        mppi_thread.join(timeout=1.0)
        if mppi_thread.is_alive():
            print("Note: MPPI thread is still finishing its last rollout "
                  "in the background (harmless -- it's a daemon thread "
                  "and touches no hardware).")

    history["summary"] = _build_summary(history, d_safe, obstacle_center, q_goal, q0, goal_threshold)
    history["summary"]["target_control_loop_hz"] = 1.0 / env.control_dt
    save_results(history)
    return history, feas_log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=str, default="assets/panda.xml")
    parser.add_argument("--robot-ip", type=str, default="172.16.0.2")
    parser.add_argument("--duration", type=float, default=60.0)

    # --- mapped 1:1 from run_mppi_async_hardware's existing parameters ----
    parser.add_argument("--n-mppi-samples", type=int, default=50)
    parser.add_argument("--obstacle-center", type=float, nargs=3, default=[0.536, 0.139, 0.356],
                         metavar=("X", "Y", "Z"), help="midpoint of home->final ee path by default")
    parser.add_argument("--obstacle-radius", type=float, default=0.08)
    parser.add_argument("--d-safe", type=float, default=0.10)
    parser.add_argument("--lambda-cbf", type=float, default=1.0)

    # --- N_horizon: no 1:1 equivalent -- see note below --------------------
    # main.py's run_closed_loop re-solves GPMP2 every N_horizon steps, so
    # N_horizon there is "how many control steps between replans". This
    # file solves GPMP2 exactly ONCE, up front (see module docstring), so
    # there is no per-cycle replanning to give N_horizon a step count for.
    # The closest equivalent is gpmp2_N: how many waypoints the ONE fixed
    # reference trajectory has. Mapped here rather than silently dropped.
    parser.add_argument("--gpmp2-n", type=int, default=10,
                         help="GPMP2 waypoint count for the one-time reference solve "
                              "(closest equivalent to run_closed_loop's N_horizon -- "
                              "see comment above this flag in the source)")

    # --- n_planning_cycles, use_hardware: genuinely not applicable here -----
    # n_planning_cycles doesn't exist in this architecture: there IS no
    # replanning loop to count cycles of (GPMP2 solves once; see module
    # docstring). use_hardware also isn't a real choice here -- this file
    # has no MuJoCo/simulation path at all; build_default_system() is
    # always called with use_hardware=True below. Neither flag is added --
    # doing so would either be a no-op that silently does nothing, or
    # imply a sim mode that doesn't exist in this file.
    args = parser.parse_args()

    history, feas_log = run_mppi_async_hardware(
        mjcf_path=args.mjcf, robot_ip=args.robot_ip, max_duration_s=args.duration,
        n_mppi_samples=args.n_mppi_samples, obstacle_center=tuple(args.obstacle_center),
        obstacle_radius=args.obstacle_radius, d_safe=args.d_safe,
        lambda_cbf=args.lambda_cbf, gpmp2_N=args.gpmp2_n)
    print(history["summary"])
