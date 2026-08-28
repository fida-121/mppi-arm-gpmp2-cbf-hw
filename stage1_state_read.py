
import numpy as np
from robot.franky_hw import FrankyHwEnv

import time

ROBOT_IP = "172.16.0.2"  # <-- REPLACE with your robot's real FCI IP

print(f"Connecting to robot at {ROBOT_IP}...")
env = FrankyHwEnv(robot_ip=ROBOT_IP, dynamics_factor=0.05)
print("Connected. Reading state 20 times, arm should remain stationary.\n")

for i in range(20):
    state = env.get_state()
    q, qdot = state[:7], state[7:]
    print(f"[{i:2d}] q={np.round(q,3)}  qdot={np.round(qdot,4)}")
    time.sleep(0.2)

print("\nStage 1 complete — no exceptions, state looks sane.")
