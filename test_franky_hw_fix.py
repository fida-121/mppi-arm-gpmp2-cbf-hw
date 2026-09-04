"""
test_franky_hw_fix.py
=======================
Isolated verification of FrankyHwEnv v6 (Ruckig-planned step() with
position-delta clamping + kinematically-consistent velocity, fixed
control_dt=0.05s). Tests, in order:

  1. reset() to the known-safe home pose -- should move smoothly at
     normal (Ruckig-limited) speed, NOT rapidly/violently.
  2. step() with small deltas -- should track smoothly, no stutter.
  3. step() with a delta LARGER than achievable in one control_dt --
     should move at max safe speed toward the target (clamped), closing
     the rest of the gap over the next few step() calls, WITHOUT a
     reflex fault and WITHOUT violent motion.

Run this from inside the franky-project container:
    python3 test_franky_hw_fix.py

SAFETY: workspace must be clear, hand near e-stop, as with every
hardware test in this project.
"""
import numpy as np
import time
from robot.franky_hw import FrankyHwEnv, FRANKA_MAX_JOINT_VEL, VELOCITY_SAFETY_FACTOR

ROBOT_IP = "172.16.0.2"  # confirm this matches your robot's actual FCI IP

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
        target[5] = q_start[5] + 0.1 * np.sin(i / 40 * np.pi)
        state = env.step(target)
        print(f"[{i:2d}] q6={state[5]:.4f}")
    print("Sweep complete -- check the printed values above for a smooth "
          "curve (no plateaus/jumps), and recall how the motion looked.")

    print()
    print("=" * 60)
    print("TEST 3: step() with a delta LARGER than one cycle's reach")
    print("=" * 60)
    max_delta_j0 = VELOCITY_SAFETY_FACTOR * FRANKA_MAX_JOINT_VEL[0] * env.control_dt
    print(f"Max achievable delta on joint 1 in one control_dt: "
          f"~{max_delta_j0:.4f} rad")
    q_now = env.get_state()[:7]
    bad_target = q_now.copy()
    bad_target[0] += 3 * max_delta_j0  # deliberately ~3x the achievable
    # distance -- should be gracefully clamped, not hard-faulted (that's
    # well under HARD_FAULT_MULTIPLIER=5x) and not violent.
    print(f"Requesting a delta of {3*max_delta_j0:.4f} rad on joint 1 "
          f"(~3x achievable per cycle).")
    print("This should move at a CONTROLLED max-safe speed toward the "
          "target -- NOT a violent snap, and NOT a reflex fault.")
    input("Press Enter to send this (Ctrl+C to abort)...")

    for i in range(10):
        state = env.step(bad_target)
        print(f"[{i}] q0={state[0]:.4f}  (target={bad_target[0]:.4f}, "
              f"remaining gap={bad_target[0]-state[0]:.4f})")
    print("\nGap should be shrinking each cycle without any fault above.")
    input("Did this feel controlled/bounded rather than violent, and did "
          "it run without a reflex fault? Press Enter to finish "
          "(Ctrl+C to stop here)...")

    print()
    env.stop()
    print("All tests complete. env.stop() called.")


if __name__ == "__main__":
    main()
