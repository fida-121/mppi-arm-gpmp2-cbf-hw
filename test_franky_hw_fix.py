"""
test_franky_hw_fix.py
=======================
Isolated verification of the fixed FrankyHwEnv, BEFORE re-running the
full closed-loop pipeline. Tests, in order:

  1. reset() to the known-safe home pose -- should move smoothly at
     normal (Ruckig-limited) speed, NOT rapidly/violently.
  2. step() with small deltas -- should track smoothly, no stutter
     (this re-verifies the earlier sine-sweep fix still works).
  3. step() with an intentionally large delta -- should be REFUSED
     (RuntimeError) by the max_step_delta safety guard, NOT executed.

Run this from inside the franky-project container:
    python3 test_franky_hw_fix.py

SAFETY: workspace must be clear, hand near e-stop, as with every
hardware test in this project.
"""
import numpy as np
import time
from robot.franky_hw import FrankyHwEnv

ROBOT_IP = "172.16.0.2"  # confirm this matches your robot's actual FCI IP

# Franka's standard "ready" home pose -- verified safe/reachable on this
# robot in earlier stage tests (stage_2c_joint.py).
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def main():
    print("=" * 60)
    print("TEST 1: reset() -- should move smoothly, NOT rapidly")
    print("=" * 60)
    env = FrankyHwEnv(robot_ip=ROBOT_IP, dynamics_factor=0.05)

    q_before = env.get_state()[:7]
    print("Position before reset:", np.round(q_before, 3))
    print(f"Target (home):         {np.round(Q_HOME, 3)}")
    input("Press Enter to run reset() -- WATCH the motion speed/force "
          "closely (Ctrl+C to abort)...")

    t0 = time.monotonic()
    env.reset(Q_HOME)
    elapsed = time.monotonic() - t0

    q_after = env.get_state()[:7]
    print(f"reset() took {elapsed:.2f}s")
    print("Position after reset:", np.round(q_after, 3))
    print("Error from target:   ", np.round(q_after - Q_HOME, 4))
    input("\nDid the motion feel smooth and appropriately slow (not a "
          "violent snap)? Press Enter to continue to Test 2, "
          "or Ctrl+C to stop here if anything felt wrong...")

    print()
    print("=" * 60)
    print("TEST 2: step() with SMALL deltas -- should track smoothly")
    print("=" * 60)
    q_start = env.get_state()[:7]
    target = q_start.copy()
    print("Sweeping joint 6 smoothly via small step() deltas...")
    input("Press Enter to run the sweep (Ctrl+C to abort)...")

    for i in range(40):
        target[5] = q_start[5] + 0.1 * np.sin(i / 40 * np.pi)  # smooth, small
        state = env.step(target)
        print(f"[{i:2d}] q6={state[5]:.4f}")
    print("Sweep complete -- check the printed values above for a smooth "
          "curve (no plateaus/jumps), and recall how the motion looked.")

    print()
    print("=" * 60)
    print("TEST 3: step() with a LARGE delta -- should be REFUSED")
    print("=" * 60)
    q_now = env.get_state()[:7]
    bad_target = q_now.copy()
    bad_target[0] += 0.5  # intentionally large -- exceeds default
    # max_step_delta=0.05, should be rejected before any motion happens.
    print("Attempting a deliberately large step() delta (should be "
          "refused, not executed)...")
    try:
        env.step(bad_target)
        print("!!! UNEXPECTED: step() did NOT refuse the large delta. "
              "This means the safety guard did not trigger -- "
              "investigate before any further hardware testing.")
    except RuntimeError as e:
        print(f"Correctly refused, as expected:\n  {e}")

    print()
    env.stop()
    print("All tests complete. env.stop() called.")


if __name__ == "__main__":
    main()
