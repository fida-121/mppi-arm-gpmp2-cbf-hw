import numpy as np
from main import build_default_system, DOF
from cbf.barrier import closest_clearance

q0 = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
q_goal = np.array([0.4, -0.3, 0.2, -1.8, 0.1, 1.6, 0.5])

center = (0.536, 0.139, 0.356)
env, franka, sdf, barrier, qp = build_default_system(
    mjcf_path="assets/panda.xml", obstacle_center=center,
    obstacle_radius=0.08, d_safe=0.10, use_hardware=False)

d0 = closest_clearance(franka.fk, franka.sphere_radii, sdf, q0)
dg = closest_clearance(franka.fk, franka.sphere_radii, sdf, q_goal)
print(f"clearance@q0={d0:.3f}, clearance@goal={dg:.3f}")

# Check a few interpolated joint-space midpoints too, as a rough proxy
# for "along the path" (not exact, since the real trajectory is curved,
# but useful as a sanity check)
for alpha in [0.25, 0.5, 0.75]:
    q_mid = (1-alpha)*q0 + alpha*q_goal
    d_mid = closest_clearance(franka.fk, franka.sphere_radii, sdf, q_mid)
    print(f"alpha={alpha}: clearance={d_mid:.3f}")
