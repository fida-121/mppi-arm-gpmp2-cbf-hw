import numpy as np
import franky
from robot.franky_hw import FrankyHwEnv
from robot.franka import Q_MIN, Q_MAX


ROBOT_IP = "172.16.0.2"
env = FrankyHwEnv(robot_ip=ROBOT_IP, dynamics_factor=0.05)

print("Current position:", np.round(env.get_state()[:7], 3))

# Absolute joint targets -- YOUR chosen values, in radians, one array
# per waypoint. These are exact configurations, not offsets.
waypoints_q = [
    np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]) #home
    #np.array([0.53, -0.599, -1.346, -1.504, -0.531, 1.521, -0.005]) #new home
    #np.array([ 0.4, -0.3,  0.2, -1.8, 0.1,  1.6,  0.5]),  # example: current + wrist tweak
    #np.array([ 0.012, -0.587,  0.022, -2.655, -0.003,  2.073,  0.921]),  # example: another pose
]

print("Absolute target waypoints:")
for i, wq in enumerate(waypoints_q):
    print(f"  wp{i}: {np.round(wq, 3)}")
input("Press Enter to execute (Ctrl+C to abort)...")

for i, wq in enumerate(waypoints_q):
    assert np.all(wq >= Q_MIN) and np.all(wq <= Q_MAX), f"wp{i} out of joint limits!"

waypoints = [
    franky.JointWaypoint(
        target=franky.JointState(position=wq),
        reference_type=franky.ReferenceType.Absolute,  # explicit, though it's the default
    )
    for wq in waypoints_q
]
motion = franky.JointWaypointMotion(waypoints)
env._safe_call(env.robot.move, motion)

print("\nFinal position:", np.round(env.get_state()[:7], 3))
