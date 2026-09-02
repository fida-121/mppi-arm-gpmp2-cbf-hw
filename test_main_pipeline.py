"""
test_main_pipeline.py
=======================
Runs the full GPMP2 -> MPPI -> CBF-QP closed loop (main.py's
run_closed_loop) on REAL HARDWARE, at small/safe scale, using the
validated baseline robot/franky_hw.py (commit 1fbb256 -- the version
that produced the one fully clean hardware run so far: goal_error=0.022,
min_clearance=0.16, 0 conflicts, 0 faults).

This is intentionally conservative by default -- small N_horizon and
few planning cycles -- so you can quickly confirm the pipeline still
runs cleanly after any code changes, before committing to a longer run.

Run from inside the franky-project container:
    python3 test_main_pipeline.py

SAFETY:
  - Workspace must be clear of people/obstacles.
  - Keep a hand near the e-stop throughout.
  - This uses q0 = the Franka "ready" home pose and the sim's original
    q_goal, BOTH already hardware-verified as safe/reachable via
    standalone waypoint tests earlier in this project.
  - No physical obstacle is required for this script (matches the CBF
    "avoiding" a phantom coordinate, as in earlier dry runs) -- if you
    HAVE placed a real, measured obstacle matching obstacle_center/
    obstacle_radius below, update those values to match reality.
"""
import argparse
import numpy as np
from main import run_closed_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", type=str, default="172.16.0.2")
    parser.add_argument("--cycles", type=int, default=2,
                         help="n_planning_cycles -- keep small for a quick check")
    parser.add_argument("--horizon", type=int, default=10,
                         help="N_horizon -- keep small for a quick check")
    parser.add_argument("--samples", type=int, default=200,
                         help="n_mppi_samples")
    parser.add_argument("--sim", action="store_true",
                         help="Run in SIMULATION instead of hardware "
                              "(safe, no robot needed -- useful to sanity "
                              "check the pipeline itself before touching "
                              "real hardware).")
    args = parser.parse_args()

    use_hardware = not args.sim
    mode_str = "HARDWARE" if use_hardware else "SIMULATION"

    print("=" * 60)
    print(f"Running main.py's closed loop in {mode_str} mode")
    print(f"  n_planning_cycles = {args.cycles}")
    print(f"  N_horizon         = {args.horizon}")
    print(f"  n_mppi_samples    = {args.samples}")
    if use_hardware:
        print(f"  robot_ip          = {args.robot_ip}")
        print()
        print("!! REAL HARDWARE RUN -- confirm workspace is clear and")
        print("!! you have a hand near the e-stop before continuing. !!")
    print("=" * 60)

    if use_hardware:
        confirm = input("\nType 'yes' to proceed with a REAL hardware run: ")
        if confirm.strip().lower() != "yes":
            print("Aborted -- did not type 'yes'.")
            return

    history, feas_log, cov_steer = run_closed_loop(
        mjcf_path="assets/panda.xml",
        N_horizon=args.horizon,
        n_planning_cycles=args.cycles,
        n_mppi_samples=args.samples,
        use_hardware=use_hardware,
        robot_ip=args.robot_ip,
    )

    print()
    print("=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print("Best goal error:  ",
          history.get("best_goal_error", "N/A (fault before final summary)"))
    print("Min clearance seen:",
          min(history["dist"]) if history["dist"] else "N/A")
    print("Steps completed:   ", len(history["q"]))
    print("Conflicts:         ", len(history["conflicts"]))
    if history["dist"]:
        print("Clearance range:   ",
              f"[{min(history['dist']):.4f}, {max(history['dist']):.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
