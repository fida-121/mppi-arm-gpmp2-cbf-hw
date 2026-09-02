"""
verify_fk_matches_hw.py
=========================
Verifies that FrankyHwEnv.ee_position() correctly uses FK (via the
wired-in franka_model) to report the real gripper-tip position, rather
than franky's raw current_pose (which reports the FLANGE, not the
mounted Franka Hand gripper's tip -- a confirmed ~0.107m offset along Z,
matching the MJCF's "hand" body pos="0 0 0.107" gripper-mount geometry).

Uses main.py's build_default_system() so franka_model gets wired into
env exactly as it does in a real run_closed_loop() call.

Run from inside the franky-project container:
    python3 verify_fk_matches_hw.py
"""
import numpy as np
from main import build_default_system

ROBOT_IP = "172.16.0.2"

env, franka, sdf, barrier, qp = build_default_system(
    mjcf_path="assets/panda.xml", use_hardware=True, robot_ip=ROBOT_IP)

print("franka_model wired in?", env.franka_model is not None)

ee_via_fk = env.ee_position()
pose = env.robot.current_pose
ee_via_flange = np.array(pose.end_effector_pose.translation)

print("ee_position() [FK-based, gripper-tip]:", np.round(ee_via_fk, 4))
print("raw flange pose (for comparison):     ", np.round(ee_via_flange, 4))
diff = ee_via_fk - ee_via_flange
print("Difference (X,Y should be ~0, Z should be ~0.107, the known "
      "gripper-mount offset):", np.round(diff, 4))

if abs(diff[2] - 0.107) < 0.03:
    print("\nOK: Z offset is consistent with the known gripper-mount "
          "geometry -- ee_position() is correctly using FK.")
else:
    print("\nWARNING: Z offset does not match the expected ~0.107m "
          "gripper-mount offset -- investigate before trusting "
          "ee_position() for logging/visualization.")
