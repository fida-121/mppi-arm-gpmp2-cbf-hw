# Panda Hardware Bridge — Project Log
### mppi-arm-gpmp2-cbf sim-to-real setup

Robot: Franka Panda, firmware **4.2.2** → libfranka **0.9.2** (confirmed compatible pair)
Host: RT-patched kernel (`6.8.0-rt8`), Ubuntu, Docker

---

## 1. Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Control interface | **Direct libfranka**, not ROS2 | Safety-critical CBF loop needs deterministic timing; ROS2/DDS adds scheduling variance we don't want between "unsafe detected" and "correction applied". Project is Python-heavy (GTSAM/OSQP/MuJoCo) so ROS2's ecosystem advantages don't apply much here. |
| Python↔libfranka binding | **franky** (not panda-py) | panda-py tracks newer libfranka; franky's Docker build explicitly supports pinning `LIBFRANKA_VERSION=0.9.2`, which matches our firmware. franky also raises real Python exceptions instead of printing to stdout — important for a safety-critical loop. |
| Isolation | **Docker**, `--privileged --network=host` | Keeps libfranka's exact version pinned and reproducible, separate from host Python. RT scheduling and FCI networking both need to pass through to the container explicitly. |
| Sim rehearsal | **Skipped** (see problem log) | franky-sim only emulates FR3 protocol, not Panda/libfranka-0.9.2 — incompatible with our robot generation. Fell back to extra-careful staged real-hardware testing instead. |
| Control mode | **Joint position** | Matches sim (`mujoco_env.py` uses position-servo actuators; CBF-QP outputs a desired position, not torque). |

---

## 2. Problems faced & how they were resolved

1. **`docker compose build franky-run --build-arg ...` → `unknown flag: --build-arg`**
   Cause: flag placement — `--build-arg` must come right after `build`, before the service name.
   Also revealed: `docker compose` (v2, no hyphen) wasn't installed at all.
   Fix: installed legacy `docker-compose` (v1.29.2, hyphenated) via `apt-get install docker-compose`, since `docker-compose-plugin` wasn't in the repo. Used hyphenated syntax from then on.

2. **`franky-run` container built successfully but `import franky` → `ModuleNotFoundError`**
   Cause: `docker/run/Dockerfile` only builds+installs the **C++ library** (`libfranky.so`) via `make install`, then deletes the whole build dir (`rm -rf franky`) — the compiled Python extension module never survives. `franky-run` is a C++-only runtime image by design.
   Fix: used the separate `franky-build` service (manylinux-based) to produce actual Python wheels, then layered a wheel-install step on top of `franky-run` in a new custom Dockerfile.

3. **`docker-compose build franky-build` failed mid-pull: `connection reset by peer` (IPv6 address in error)**
   Cause: transient network flakiness, likely IPv6-related.
   Fix: simple retry — Docker resumed from cached layers and succeeded.

4. **Wheel filename guessed wrong** (`manylinux_2_34_x86_64` vs actual `linux_x86_64`)
   Fix: always `ls build/wheels/` and copy the *exact* filename before writing `COPY`/`pip install` lines — don't assume naming conventions.

5. **`franky-sim` → `franky._franky.ProtocolException: incorrect object size`**
   Cause: franky-sim implements the **FR3** libfranka network protocol, not the older Panda/0.9.2 protocol our robot uses. Wire-level struct sizes differ between robot generations — not a config option, a hard incompatibility.
   Fix: abandoned franky-sim as a pre-hardware rehearsal. Fell back to: (a) validate GPMP2/MPPI/CBF logic in isolation using the existing MuJoCo sim, (b) treat Stage 1 (state-read-only) and Stage 2 (tiny motion) on **real hardware** as the de facto first protocol validation, done with extra care/slowness.

6. **`nano` not available inside the minimal container**
   Fix: write scripts on the host inside the volume-mounted project folder instead of editing in-container.

7. **`franky.Robot(...).relative_dynamics_factor = ...` — wrong usage pattern**
   Cause: assumed it was a settable attribute; it's actually a **constructor argument** (`franky.Robot(ip, relative_dynamics_factor=franky.RelativeDynamicsFactor(value), ...)`).
   Fix: confirmed via `help(franky.Robot)`, `help(franky.RelativeDynamicsFactor)` — used the single-scalar constructor form (0,1] for simplicity, applied at connect time.

8. **`franky.JointWaypoint(target=...)` — target isn't a raw list**
   Cause: `target` must be a `franky.JointState` object, not a plain array.
   Fix: confirmed via `help(franky.JointState)` — `franky.JointState(position=array)` for a position-only target (zero velocity).

9. **`GTSAM` + `numpy 2.2.6` → Segmentation fault** (real, not a false alarm)
   Cause: GTSAM's compiled Python extension has an ABI incompatibility with numpy 2.x. `import gtsam` succeeds silently; the crash only appears when GTSAM functions actually exchange arrays with numpy (e.g. `Pose3.transformFrom`, `Values.insert`).
   Fix: pinned `numpy<2.0` (specifically `1.26.4`) — verified franky *also* still works fine under 1.26.4, so no conflict. Pinned in **two places** for robustness: `requirements.txt` (`numpy>=1.24,<2.0`) AND an explicit `RUN pip3 install "numpy<2.0" --force-reinstall` line in the project Dockerfile (since franky's own dependency chain independently wants numpy 2.x, and could silently re-pull it otherwise).
   **Lesson: a clean `import` does NOT prove a compiled-extension package is compatible with your numpy version — exercise real cross-boundary calls before trusting it.**

10. **Project's own `venv/` folder inside `mppi-arm-gpmp2-cbf/`**
    Risk: copying a host-built venv into the container is wasteful and can contain host-specific broken paths/binaries.
    Fix: added `.dockerignore` excluding `venv/`, `__pycache__/`, `*.pyc`, `plots/`.

---

## 3. Docker images — what each one is for

```
~/franky/
├── docker-compose.yml       (defines franky-build, franky-run services)
├── docker/
│   ├── run/Dockerfile       → builds franky_franky-run:latest
│   ├── build/Dockerfile     → builds franky_franky-build:latest
│   ├── hw/Dockerfile        → builds franky-hw:latest
│   └── project/Dockerfile   → builds franky-project:latest
├── build/wheels/            (Python wheels produced by franky-build)
└── mppi-arm-gpmp2-cbf/      (your project, volume-mounted at runtime)
```

| Image | Built from | Purpose |
|---|---|---|
| `franky_franky-run` | `docker/run/Dockerfile` (compose service `franky-run`) | **C++-only** runtime: libfranka 0.9.2 + franky C++ lib, RT + host networking configured. No Python bindings. Base for `franky-hw`. |
| `franky_franky-build` | `docker/build/Dockerfile` (compose service `franky-build`) | manylinux-based **wheel builder**. Not a runtime image — run once (or when franky/libfranka version changes) to produce `.whl` files into `./build/wheels/`. |
| `franky-hw` | `docker/hw/Dockerfile`, `FROM franky_franky-run:latest` | Adds python3/pip + installs the `cp310` wheel from `build/wheels/`. This is the first image where `import franky` actually works. |
| `franky-project` | `docker/project/Dockerfile`, `FROM franky-hw:latest` | Adds project's `requirements.txt` (GTSAM/OSQP/MuJoCo/etc.) + pins `numpy<2.0` + bakes in project code. **This is the image you actually run day-to-day.** |

---

## 4. Commands reference

**Build the C++ runtime (once, or when changing libfranka version):**
```bash
cd ~/franky
docker-compose build --build-arg LIBFRANKA_VERSION=0.9.2 franky-run
```

**Build the wheel builder + produce wheels (once, or when franky/libfranka version changes):**
```bash
docker-compose build --build-arg LIBFRANKA_VERSION=0.9.2 franky-build
docker-compose run --rm franky-build build-wheels
ls -la build/wheels/     # verify wheel filenames before next step
```

**Build the hw image (after wheels exist):**
```bash
docker build -f docker/hw/Dockerfile -t franky-hw:latest .
```

**Build the project image (after editing requirements.txt or Dockerfile):**
```bash
docker build -f docker/project/Dockerfile -t franky-project:latest .
```

**Run the project container (day-to-day command):**
```bash
cd ~/franky
docker run -it --rm --privileged --network=host \
  -v $(pwd)/mppi-arm-gpmp2-cbf:/workspace/mppi-arm-gpmp2-cbf \
  franky-project:latest
```

**Inside the container — sanity checks:**
```bash
chrt -m                                     # confirm SCHED_FIFO 1/99 available
python3 -c "import franky; print(franky.__file__)"
pip3 show numpy | grep Version              # should read 1.26.4
python3 -c "import gtsam, numpy as np; print(gtsam.Pose3().transformFrom(np.array([1.,2.,3.])))"
```

---

## 5. Staged hardware validation — status

1. ✅ **State-read only** (`stage1_state_read.py`) — connected at `172.16.0.2`, 20x `get_state()` reads, stable q, near-zero qdot noise, no exceptions.
2. ✅ **Tiny single-joint motion** (`stage2_tiny_motion.py`) — streamed `step()` toward a +0.1 rad wrist target via `FrankyHwEnv`, `dynamics_factor=0.05`.
3. ✅ **Multi-waypoint trajectories** — both relative-delta and absolute-position `JointWaypointMotion` sequences via native franky waypoint API (not yet through the bridge's `step()` loop).
4. ⏳ **Not yet done:** obstacle-present run of the full GPMP2→MPPI→CBF-QP pipeline against real hardware (this is "Stage 3" in earlier discussion).
5. ⏳ **Not yet done:** full-speed, closer-margin runs.

---

## 6. Next steps (in order)

1. **Streamed sequential targets via the bridge** — run a scripted sequence of targets through `FrankyHwEnv.step()` repeatedly (not native `JointWaypointMotion`), since this is closer to how `main.py`'s real loop will call it every `control_dt`.
2. **Patch `main.py`** — wrap `env.step(qp_result.u_safe)` in try/except so a `FrankyHwEnv` fault (`RuntimeError`) stops the loop cleanly instead of crashing raw:
   ```python
   try:
       env.step(qp_result.u_safe)
   except RuntimeError as e:
       print(f"HARDWARE FAULT — stopping loop: {e}")
       break
   ```
3. **Swap `MujocoFrankaEnv` → `FrankyHwEnv`** inside `build_default_system()` in `main.py`.
4. **Measure and place a real physical obstacle** at a known, fixed position; update `obstacle_center`/`obstacle_radius` to match reality (not the sim's virtual sphere).
5. **Increase `d_safe`** above sim's value to buffer against real-world loop timing/latency (recall sim's own h(x) dip of -0.001 to -0.004 under discrete-time enforcement — real hardware timing jitter could make this worse).
6. **Re-tune parameters against real dynamics** — `d_safe`, `alpha`/`gamma`, sample counts, thread rates were all calibrated in sim (`sweep_alpha_gamma.py` logic).
7. **Stage 3: full pipeline at reduced speed**, obstacle present, watching `h(x)` live.
8. **Stage 4: full speed, closer margins**, e-stop within reach throughout.

---

## 7. Standing safety rules (apply to every hardware run from here on)

- Workspace clear of people/obstacles before any motion command.
- E-stop within physical reach at all times.
- Never raise `dynamics_factor` without having validated the previous, lower value first.
- Any new script gets a `Q_MIN`/`Q_MAX` bounds check on target positions before execution.
- `FrankyHwEnv.faulted` latches — once tripped, the object refuses further commands; reconnect fresh rather than trying to reuse a faulted instance.
