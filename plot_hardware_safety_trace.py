"""
plot_hardware_safety_trace.py
Reads the hardware team's CSV format (timestep, real_time, h,
goal_error, dist, cost_history, conflict, q_0..q_6, u_mppi_0..6,
u_safe_0..6) and produces a safety-over-time figure.
"""
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace(".csv", "_safety_trace.png")

t, h_vals, goal_err, conflicts, intervention = [], [], [], [], []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        t.append(float(row["real_time"]))
        h_vals.append(float(row["h"]))
        goal_err.append(float(row["goal_error"]))
        conflicts.append(int(row["conflict"]))
        u_mppi = np.array([float(row[f"u_mppi_{i}"]) for i in range(7)])
        u_safe = np.array([float(row[f"u_safe_{i}"]) for i in range(7)])
        intervention.append(float(np.linalg.norm(u_safe - u_mppi)))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

ax1.plot(t, h_vals, color="#1f6feb", linewidth=1.2)
ax1.axhline(0, color="#da3633", linestyle="--", linewidth=1, label="h(x) = 0")
conflict_times = [ti for ti, c in zip(t, conflicts) if c]
conflict_h = [hi for hi, c in zip(h_vals, conflicts) if c]
ax1.scatter(conflict_times, conflict_h, color="#e8590c", zorder=5, s=30, label="conflict event")
ax1.set_ylabel("h(x) [m]")
ax1.set_title("Real Hardware: Safety Margin, Goal Convergence, CBF-QP Activity")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax2.plot(t, goal_err, color="#2ea043", linewidth=1.2)
ax2.set_ylabel("goal error [rad]")
ax2.grid(alpha=0.3)

ax3.plot(t, intervention, color="#da3633", linewidth=1.0)
ax3.set_xlabel("time [s]")
ax3.set_ylabel("||u_safe - u_mppi||")
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")

n = len(h_vals)
n_neg = sum(1 for h in h_vals if h < 0)
print(f"\nSteps: {n} | Duration: {t[-1]:.2f}s | Worst h(x): {min(h_vals):.4f}")
print(f"Steps with h(x) < 0: {n_neg} ({100*n_neg/n:.2f}%)")
print(f"Conflict events: {sum(conflicts)}")
print(f"Mean intervention: {np.mean(intervention):.5f}  Max intervention: {max(intervention):.5f}")
