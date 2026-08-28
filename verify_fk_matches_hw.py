# verify_fk_matches_hw.py
import numpy as np
import mujoco
from robot.franka import FrankaModel
from robot.franky_hw import FrankyHwEnv

ROBOT_IP = "172.16.0.2"
MJCF_PATH = "assets/panda.xml"

env = FrankyHwEnv(robot_ip=ROBOT_IP, dynamics_factor=0.05)
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
franka = FrankaModel(model, mujoco.MjData(model))

state = env.get_state()
q = state[:7]

centers, jacs = franka.fk(q)   # <-- unpack the tuple correctly
print("Number of spheres:", centers.shape[0])
ee_from_fk = centers[-1]       # last sphere center, closest to end-effector

pose = env.robot.current_pose
ee_from_franky = np.array(pose.end_effector_pose.translation)

print("Last sphere center (from franka.fk): ", np.round(ee_from_fk, 4))
print("EE from franky pose:                  ", np.round(ee_from_franky, 4))
print("Difference (should be small):         ", np.round(ee_from_fk - ee_from_franky, 4))
print(env.robot.current_pose)
# also check if there's a separate flange pose accessor:
help(franky.Robot.current_pose)  # or check franky docs for "flange" vs "end_effector"
