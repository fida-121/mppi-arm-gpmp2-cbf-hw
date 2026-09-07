"""
validate_fk_batch.py
======================
Standalone script to guarantee the vectorized forward kinematics 
perfectly match MuJoCo's internal sequential kinematics across 
the full joint range.
"""

import time
import numpy as np
import mujoco

# Import your model and joint limits
from robot.franka import FrankaModel, Q_MIN, Q_MAX

def main():
    # 1. Load the actual MuJoCo model you are using
    # (Update "assets/panda.xml" to your actual MJCF path if different)
    mjcf_path = "assets/panda.xml" 
    try:
        model = mujoco.MjModel.from_xml_path(mjcf_path)
    except ValueError as e:
        print(f"Error loading model: {e}")
        print("Please update the 'mjcf_path' variable in this script to point to your Panda XML.")
        return

    data = mujoco.MjData(model)
    franka = FrankaModel(model, data)
    
    print(f"Successfully loaded model with {franka.num_spheres()} collision spheres per configuration.")

    # 2. Generate random valid configurations encompassing maximum joint swings
    # Shape: (N=50 trajectories, T=10 timesteps, DOF=7)
    rng = np.random.default_rng(42)
    Q_random = rng.uniform(Q_MIN, Q_MAX, size=(50, 10, 7))
    
    # 3. Benchmark and collect output from the Original Sequential FK
    print("\nBenchmarking Original MuJoCo fk_batch...")
    t0 = time.perf_counter()
    out_old = franka.fk_batch(Q_random)
    t1 = time.perf_counter()
    time_old = (t1 - t0) * 1000
    print(f"Original Time: {time_old:.2f} ms")
    
    # 4. Benchmark and collect output from the New Vectorized FK
    print("\nBenchmarking Vectorized fk_batch...")
    t0 = time.perf_counter()
    out_new = franka.fk_batch_vectorized(Q_random)
    t1 = time.perf_counter()
    time_new = (t1 - t0) * 1000
    print(f"Vectorized Time: {time_new:.2f} ms")
    print(f"Speedup: {time_old / time_new:.2f}x faster")
    
    # 5. Strict mathematical validation
    # out_old and out_new both have shape (50, 10, M, 3)
    max_err = np.max(np.abs(out_old - out_new))
    print(f"\nMaximum discrepancy across {50 * 10 * franka.num_spheres()} sphere positions: {max_err:.8e} meters")
    
    if max_err < 1e-6:
        print("\n✅ SUCCESS: Vectorized FK is analytically perfect.")
        print("You are now safe to change fk_batch_fn=franka.fk_batch_vectorized in main.py!")
    else:
        print("\n❌ MISMATCH: Do not use the vectorized version yet. The math diverges from MuJoCo.")

if __name__ == "__main__":
    main()
