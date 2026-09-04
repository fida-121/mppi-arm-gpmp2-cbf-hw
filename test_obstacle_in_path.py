# test_obstacle_in_path.py
from main import run_closed_loop

history, feas_log, cov_steer = run_closed_loop(
    mjcf_path="assets/panda.xml",
    N_horizon=10,
    n_planning_cycles=2,
    n_mppi_samples=200,
    obstacle_center=(0.536, 0.139, 0.356),  # midpoint of home->final ee path
    obstacle_radius=0.08,
    d_safe=0.10,
    lambda_cbf=1.0,
    use_hardware=True,
    robot_ip="172.16.0.2",
)
print("Best goal error:", history.get("best_goal_error", "N/A"))
print("Min clearance seen:", min(history["dist"]) if history["dist"] else "N/A")
print("Conflicts:", len(history["conflicts"]))
