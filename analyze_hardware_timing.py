"""
analyze_hardware_timing.py
Computes real control loop timing statistics from a hardware trace
CSV -- overall loop rate, and (if the CSV has t_mppi_s/t_qp_s columns)
per-stage timing breakdowns.

Usage:
    python3 analyze_hardware_timing.py results/run_20260904_104324_trace.csv
"""
import sys
import csv
import numpy as np

csv_path = sys.argv[1]

real_times = []
t_mppi, t_qp = [], []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        real_times.append(float(row["real_time"]))
        if "t_mppi_s" in fieldnames:
            t_mppi.append(float(row["t_mppi_s"]))
        if "t_qp_s" in fieldnames:
            t_qp.append(float(row["t_qp_s"]))

# Overall control loop rate: time BETWEEN consecutive steps
step_durations = np.diff(real_times)
print(f"=== Overall control loop timing (from real_time column) ===")
print(f"Total steps: {len(real_times)}")
print(f"Total duration: {real_times[-1]:.2f}s")
print(f"Mean step duration: {np.mean(step_durations):.4f}s  "
      f"({1/np.mean(step_durations):.2f} Hz)")
print(f"Min step duration:  {np.min(step_durations):.4f}s  "
      f"({1/np.min(step_durations):.2f} Hz)")
print(f"Max step duration:  {np.max(step_durations):.4f}s  "
      f"({1/np.max(step_durations):.2f} Hz)")
print(f"Std step duration:  {np.std(step_durations):.4f}s")

if t_mppi:
    print(f"\n=== MPPI compute time ===")
    print(f"Mean: {np.mean(t_mppi)*1000:.2f} ms   Max: {np.max(t_mppi)*1000:.2f} ms")
if t_qp:
    print(f"\n=== CBF-QP solve time ===")
    print(f"Mean: {np.mean(t_qp)*1000:.2f} ms   Max: {np.max(t_qp)*1000:.2f} ms")

if not t_mppi and not t_qp:
    print(f"\nNote: this CSV does not have t_mppi_s/t_qp_s columns -- "
          f"only overall step timing is available from 'real_time'. "
          f"To get the per-stage breakdown (MPPI vs CBF-QP vs GPMP2 "
          f"separately), the hardware logger needs those columns added.")
