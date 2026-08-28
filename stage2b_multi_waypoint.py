
# stage2b_multi_waypoint.py
import numpy as np
import franky
from robot.franky_hw import FrankyHwEnv

ROBOT_IP = "172.16.0.2"

env = FrankyHwEnv(robot_ip=ROBOT_IP, dynamics_factor=0.05)
q_start = env.get_state()[:7]
print("Start position:", np.round(q_start, 3))

# Small, deliberate multi-point sequence -- each waypoint a modest step
# from the last, well within joint limits. Adjust deltas if any joint
# would approach a limit given YOUR robot's current pose.
deltas = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0,  0.1, 0.0]),   # wrist +0.1 rad
    np.array([0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.15]),  # +joint7 +0.15 rad
    np.array([0.0, 0.05, 0.0, 0.0, 0.0, -0.1, -0.15]),# joint2 +0.05, undo wrist/j7
]
waypoints_q = []
q = q_start.copy()
for d in deltas:
    q = q + d
    waypoints_q.append(q.copy())

print("Planned waypoints:")
for i, wq in enumerate(waypoints_q):
    print(f"  wp{i}: {np.round(wq, 3)}")
input("Press Enter to execute this multi-waypoint motion (Ctrl+C to abort)...")

waypoints = [franky.JointWaypoint(target=franky.JointState(position=wq)) for wq in waypoints_q]
motion = franky.JointWaypointMotion(waypoints)  # blocking by default (asynchronous=False)

env._safe_call(env.robot.move, motion)  # reuse bridge's fault-safe wrapper

final_q = env.get_state()[:7]
print("\nFinal position:", np.round(final_q, 3))
print("Target (last waypoint):", np.round(waypoints_q[-1], 3))
print("Error:", np.round(final_q - waypoints_q[-1], 4))
print("Stage 2b complete.")
