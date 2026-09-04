# test_smooth_streaming.py
import numpy as np
import time
from robot.franky_hw import FrankyHwEnv

env = FrankyHwEnv(robot_ip="172.16.0.2", dynamics_factor=0.05)
q_start = env.get_state()[:7]
print("Start:", np.round(q_start, 3))

env.reset(q_start)  # starts the tracking motion, holds current position

# Sweep one joint smoothly over ~2 seconds via repeated step() calls --
# should now feel continuous, not stop-start.
target = q_start.copy()
for i in range(40):
    target[5] = q_start[5] + 0.1 * np.sin(i / 40 * np.pi)  # smooth sine sweep
    state = env.step(target)
    print(f"[{i:2d}] q6={state[5]:.4f}")

env.stop()
print("Done.")
