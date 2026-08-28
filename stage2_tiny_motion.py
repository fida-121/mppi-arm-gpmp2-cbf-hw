# stage2_tiny_motion.py
import numpy as np
import time
from robot.franky_hw import FrankyHwEnv

ROBOT_IP = "172.16.0.2"  # confirmed working from Stage 1

env = FrankyHwEnv(robot_ip=ROBOT_IP, dynamics_factor=0.05)

state = env.get_state()
q_start = state[:7]
print("Start position:", np.round(q_start, 3))

# Move joint 6 (wrist) by a SMALL amount only -- +0.1 rad (~5.7 degrees)
q_target = q_start.copy()
q_target[5] += 0.1

print("Target position:", np.round(q_target, 3))
input("Press Enter to command this small motion (Ctrl+C to abort)...")

# Stream toward target over ~2 seconds using step(), matching main.py's
# real usage pattern -- repeated calls with the SAME target, since u_star
# there is recomputed each cycle by the CBF-QP; here we just hold steady.
for i in range(40):  # 40 * control_dt(0.05) = 2s
    state = env.step(q_target)
    print(f"[{i:2d}] q6={state[5]:.4f}")

print("\nFinal position:", np.round(env.get_state()[:7], 3))
print("Stage 2 complete.")
