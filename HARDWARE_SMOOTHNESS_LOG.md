# Hardware Bridge — Motion Smoothness Investigation & Revert Log
### mppi-arm-gpmp2-cbf — supplement to PROJECT_LOG.md

This log covers everything since the first successful Stage 3 dry run
(commit `1fbb256`): an attempt to fix a cosmetic motion stutter, several
real hardware safety incidents encountered along the way, and the
decision to revert to the original validated baseline. Read this
alongside `PROJECT_LOG.md` (the original setup log).

---

## 1. Starting point

First fully successful hardware run of the complete closed loop
(GPMP2 → MPPI → CBF-QP → real robot), commit `1fbb256`:
- `best_goal_error = 0.0224` rad
- `min_clearance = 0.1598`
- `0` conflicts, `0` faults

**Only issue observed:** the robot's motion had a visible **stop-start
stutter** — not dangerous, purely cosmetic, but not smooth.

---

## 2. Root cause of the stutter

`step()` sent a **new** `franky.JointWaypointMotion` every `control_dt`
(0.05s), each one targeting a **zero-velocity** waypoint
(`JointState(position=u_star)`, no velocity given). Ruckig plans every
waypoint as a full decelerate-to-rest segment; replanning one every
~50ms meant the robot was repeatedly told to brake, then re-accelerate
toward a new nearby target.

---

## 3. Fix attempts, in order, and what went wrong with each

### v2 — `JointImpedanceTrackingMotion` + `set_reference()`
Switched to a persistent torque-based impedance controller, updated via
`set_reference()` each cycle instead of creating new motions.
**Result:** fixed the stutter (verified via a smooth sine-sweep test).
**BUT:** `reset()` also called `set_reference()` directly with a
potentially far-away target.

**REAL SAFETY INCIDENT:** the robot moved "very rapidly with very
high force and torque" when `reset()` commanded a large jump through
this controller. Root cause: `JointImpedanceTrackingMotion` applies
torque proportional to `(reference - actual)` immediately, with no
distance-aware trajectory planning -- unlike Ruckig-planned motions,
which respect velocity/acceleration/jerk limits regardless of distance.

### v3 — Ruckig-planned `reset()` + manual velocity-clamp guard on `step()`
Fixed `reset()` to use `franky.JointMotion` (Ruckig-planned, safe for
any distance). Kept `JointImpedanceTrackingMotion` for `step()`, added
a manual guard (`max_step_delta`) that refused any `step()` call whose
delta from `q_now` exceeded a threshold.
**Result:** `test_franky_hw_fix.py` passed (smooth reset, smooth
tracking, large-delta correctly refused).
**BUT:** in the real closed-loop run, the guard's measured delta grew
cycle-over-cycle (0.0649 rad -> 0.1069 rad -> 0.1486 rad) and
eventually hard-faulted, even though nothing dangerous was happening.

**Root cause:** the torque-based tracking controller has real physical
lag between "reference updated" and "arm actually arrives" -- unlike
MuJoCo sim's very stiff position-servo actuators (kp up to 4500),
which track almost instantly. Comparing the next planned target
against the arm's real, lagging current position made the apparent
gap compound every cycle.

**Interim mitigation tried:** raised `max_step_delta` from 0.05 to 0.1.
Delayed the fault by one cycle but did not fix the underlying lag issue.

### v4 -- Velocity-based soft-clamp (still on `JointImpedanceTrackingMotion`)
Replaced the flat refuse-guard with a soft clamp against real Franka
joint velocity limits (`FRANKA_MAX_JOINT_VEL`), only hard-faulting at
2x that limit.
**Result:** `test_franky_hw_fix.py` passed. **BUT** the underlying
tracking-lag problem (v3's root cause) was still present -- this version
addressed the symptom (fault threshold) more gracefully but not the
cause.

### v5 -- Back to Ruckig-planned `step()`, with finite-difference velocity
Switched `step()` back to `franky.JointWaypointMotion` (Ruckig-planned,
like `reset()`), now giving each waypoint a non-zero target velocity
estimated by finite-differencing consecutive `u_star` calls, clipped to
`FRANKA_MAX_JOINT_VEL`.
**Result:** fixes the original v1 stutter cause directly (no more
zero-velocity "stop here" waypoints).
**BUT**, tested with a deliberately large artificial delta (0.3 rad):

**REAL HARDWARE FAULT:** `franky._franky.ControlException: libfranka:
Move command aborted: motion aborted by reflex!
["joint_motion_generator_velocity_discontinuity",
"joint_motion_generator_acceleration_discontinuity"]`

**Important:** this was caught by libfranka's own built-in reflex
safety system -- no motion occurred, robot required re-enabling via
Desk web interface, no hardware was harmed. Root cause: clipping only
the velocity estimate to the max, without also constraining the
position delta consistently, produced a kinematically inconsistent
(position, velocity) pair for a fixed `control_dt=0.05s` window (e.g.
0.3 rad in 0.05s implies 6 rad/s average -- clipping velocity alone to
~2.2 rad/s doesn't make covering that distance in that time
achievable).

### v6 -- Position-delta clamp (from actual current position) + matching consistent velocity
Kept `control_dt` fixed at 0.05s (required for the paper -- never
changed). Clamped the position delta itself (measured from the
arm's actual current position via `get_state()`, not any
previously-commanded target -- avoiding v3's lag-compounding problem) to
what's achievable within `control_dt` at a safety-margined velocity
(`VELOCITY_SAFETY_FACTOR=0.6`), then derived the target velocity from
that same clamped delta -- guaranteeing kinematic consistency. Hard
fault reserved for requests >=5x the achievable distance.
**Result:** `test_franky_hw_fix.py` Tests 1-2 passed cleanly.
**BUT**, on Test 3 (small-delta sine sweep, NOT even the large-delta
test):

**NEW HARDWARE FAULT:** `RuntimeError: Motion planner failed with
error code -100` (a generic Ruckig planner failure), occurring on the
third call of an ordinary small sine-sweep sequence -- not an
artificially large request.

**Assessment:** repeatedly calling `robot.move()` with a brand-new
async `JointWaypointMotion` every ~50ms is inherently fragile -- Ruckig
expects well-defined boundary conditions for each point-to-point
motion, and the actual robot velocity at the exact instant a new
motion is queued can differ subtly from what we assumed when reading
`get_state()` moments earlier, producing an infeasible boundary
condition. This is a genuine, subtle real-time control problem, not
something safely resolved by further guessing on live hardware.

---

## 4. Decision: revert to the validated baseline

After four attempted smoothness fixes produced three separate
real hardware fault/incident events (violent motion, reflex abort,
planner failure) while chasing a purely cosmetic stutter, the
decision was made to stop and revert.

**Restored:** `robot/franky_hw.py` to the exact state at commit
`1fbb256` -- the original zero-velocity `JointWaypointMotion` approach,
which is the only version with a clean, fully validated hardware
safety record (the Stage 3 dry run success described in Section 1).

```bash
git checkout 1fbb256 -- robot/franky_hw.py
git add robot/franky_hw.py
git commit -m "Revert franky_hw.py to the exact version used in the successful Stage 3 dry run (commit 1fbb256)..."
git push
```

**One post-revert fix needed:** the restored file was missing a
`stop()` method (added in later versions; `main.py`'s `run_closed_loop`
calls `env.stop()` at the end). Added `stop()` back in -- purely
additive, does not touch `reset()`/`step()`:
```python
def stop(self):
    """Explicitly stop any running motion. Safe to call at any time,
    including if no motion was ever started."""
    try:
        self.robot.stop()
    except Exception:
        pass
```

**Post-revert re-validation:** ran `test_main_pipeline.py` twice on real
hardware. Both runs clean, consistent with the original success:

| Metric | Original success (1fbb256) | Re-validation run 1 |
|---|---|---|
| Best goal error | 0.0224 rad | 0.0275 rad |
| Min clearance | 0.1598 | 0.1597 |
| Clearance range | -- | [0.1597, 0.3039] |
| Conflicts | 0 | 0 |
| Faults | 0 | 0 |

Also re-confirmed `ee_position()`'s FK-based gripper-tip fix survived
the revert (Z-offset from raw flange pose matches known ~0.107m
gripper-mount geometry -- see `verify_fk_matches_hw.py`).

**Status of the stutter:** unresolved, deferred. Documented here as a
known, cosmetic sim-to-real limitation -- motion is safe and
accurate but not smooth. Worth stating plainly in the paper's hardware
section rather than hidden. Future investigation should start from
franky's actual example source code (not just docs) for the officially
intended real-time streaming boundary-condition handling, or raise the
specific `error code -100` / reflex issue with the franky maintainer,
rather than iterating further live on hardware under time pressure.

---

## 5. New/updated files from this investigation

| File | Purpose | Status |
|---|---|---|
| `robot/franky_hw.py` | Hardware bridge | Reverted to validated baseline (commit `1fbb256` content) + `stop()` added back |
| `test_franky_hw_fix.py` | Isolated reset()/step() verification | Exists in repo history in multiple versions (v1 through v6 test variants) -- current committed version corresponds to the reverted baseline; smoothness-specific test variants are in git history only |
| `test_main_pipeline.py` | NEW -- reusable wrapper for running `main.py`'s full closed loop, with `--sim`/hardware modes, confirmation prompt, safe summary printing | Added this session |
| `verify_fk_matches_hw.py` | NEW (recreated) -- confirms `ee_position()` uses FK (gripper-tip), not franky's raw flange pose | Added this session, re-run post-revert to confirm fix survived |

---

## 6. Commands used this session

**Restore the validated baseline:**
```bash
cd ~/franky/mppi-arm-gpmp2-cbf
git log --oneline -- robot/franky_hw.py          # find the right commit
git show 1fbb256:robot/franky_hw.py | head -50    # confirm before restoring
git checkout 1fbb256 -- robot/franky_hw.py
git diff --stat
python3 -c "import ast; ast.parse(open('robot/franky_hw.py').read())" && echo "SYNTAX OK"
git add robot/franky_hw.py
git commit -m "Revert franky_hw.py to the exact version used in the successful Stage 3 dry run..."
git push
```

**Add back the missing `stop()` method (manual edit via nano, then):**
```bash
python3 -c "import ast; ast.parse(open('robot/franky_hw.py').read())" && echo "SYNTAX OK"
grep -n "def stop" robot/franky_hw.py
git add robot/franky_hw.py
git commit -m "Add back stop() method..."
git push
```

**Run the full pipeline test (small/safe scale, sim or hardware):**
```bash
python3 test_main_pipeline.py --sim              # sim sanity check, no robot needed
python3 test_main_pipeline.py                    # hardware, default small scale
python3 test_main_pipeline.py --cycles 5 --horizon 20   # larger, once confident
```

**Verify `ee_position()`'s FK fix:**
```bash
python3 verify_fk_matches_hw.py
```

**General container launch (unchanged from before):**
```bash
cd ~/franky
docker run -it --rm --privileged --network=host \
  -v $(pwd)/mppi-arm-gpmp2-cbf:/workspace/mppi-arm-gpmp2-cbf \
  franky-project:latest
```

---

## 7. Key lessons for the paper / future work

1. **Sim's actuator model does not transfer directly to hardware.**
   MuJoCo's stiff position-servo actuators (kp up to 4500) track a
   commanded position almost instantly; no real hardware controller
   available through franky behaves this way. Any "smoothness" fix must
   account for real actuator dynamics (lag, velocity/acceleration/jerk
   limits, reflex safety systems), not just eliminate visible stutter.

2. **A working-but-imperfect version is safer than an
   untested-but-smoother one.** Every attempt to improve smoothness
   introduced a new failure mode not present in the original. The
   original stutter was a UX/quality issue; the fix attempts introduced
   actual safety incidents. When time/safety pressure is high, prefer
   the validated baseline over an appealing but under-tested
   improvement.

3. **`control_dt=0.05s` is a genuinely tight window** given Franka's
   real joint velocity limits (~2.175-2.61 rad/s) -- max achievable
   distance per cycle is roughly 0.109-0.131 rad at full speed, and
   real pipeline deltas (0.06-0.15 rad observed) sometimes approach or
   exceed that. This is a legitimate sim-to-real discrepancy worth
   discussing in the paper, not a bug to silently patch around.

4. **libfranka's reflex system is a real, working safety net** -- it
   caught a genuinely inconsistent motion request before any hardware
   motion occurred. This is reassuring evidence for the paper's safety
   discussion: even bridge-layer bugs are caught by a lower-level
   hardware safety system, not just software-side guards.
