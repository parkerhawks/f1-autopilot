# Game day: first session with F1 25

Ordered so each step gates the next. If one fails, stop and fix it — the later
steps assume the earlier ones.

## Before launching

```powershell
.\.venv\Scripts\python.exe -m pip install vgamepad
```

Installs the ViGEmBus kernel driver and raises a UAC prompt. Nothing in the
control path works without it.

**Always install through `.\.venv\Scripts\python.exe -m pip`, never bare
`pip`.** Bare `pip` resolves to the system Python, and the package lands
somewhere the project cannot see — the import fails while `pip list` cheerfully
shows it installed. Verify with:

```powershell
.\.venv\Scripts\python.exe -c "import vgamepad as vg; vg.VX360Gamepad(); print('ok')"
```

That also proves the driver is live, not just the Python package.

## Step 0 — first launch, no tools

Let it finish **shader compilation**. It runs for several minutes on first boot
and will wreck any latency or contention measurement taken during it.

Then set:

| setting | value | why |
|---|---|---|
| Display mode | **borderless windowed** | exclusive fullscreen blocks both capture and the overlay |
| Resolution | 1920×1080 | halves preprocessing vs 1440p (measured: 3.9 ms → ~2 ms) |
| Frame cap | 60 | leaves GPU headroom for inference |
| V-Sync | off | adds input latency |
| Motion blur | **off** | blur corrupts training frames |
| Preset | medium/low | simpler visuals help the CNN and cut contention |
| UDP Telemetry | on | |
| UDP IP / Port | 127.0.0.1 / 20777 | |
| UDP send rate | 60 Hz | |
| UDP format | **2025** | `packets.py` assumes this |

Assists on to start (TC, ABS, racing line). Remove them one at a time later.

Load a **Time Trial** at Monza — no opponents, consistent conditions, quick
restarts.

## Step 1 — does telemetry flow, and is our parser right?

```bash
python tools/probe_telemetry.py
```

Compare the byte sizes against `EXPECTED_SIZES` in `packets.py`. A mismatch
means EA's layout differs from the published spec — **fix the layouts, do not
loosen the assertion.** A wrong offset produces subtly corrupt training data
rather than a crash.

```bash
python tools/probe_telemetry.py --decode
```

Drive. Speed, gear, steering and world position must track what you are doing.

```bash
python tools/probe_telemetry.py --scan-floats 2 --scan-seconds 150
```

Complete a full lap during this. It locates `lapDistance` empirically, by
finding the float that climbs monotonically **and resets** at the line. Write
the result into `LAP_DATA_STRIDE` / `LAP_DISTANCE_OFFSET`.

## Step 2 — can we drive it?

```bash
python tools/test_gamepad.py
```

Game focused, car parked. The wheel should sweep lock to lock.

```bash
python tools/measure_latency.py
```

**Gate: median under ~80 ms.** Above that, fix v-sync / frame cap / window mode
before continuing. Whatever the number, record it — the recorder shifts action
labels by that many frames to compensate.

## Step 3 — the item that decides feasibility

```bash
python tools/probe_reset.py --manual --trials 5
```

Crash the car, then reset it however you like — **try a different method each
time**. Restart lap, return to garage, any reset-to-track option. The tool
times each from telemetry and tells you what that reset cost implies for an
overnight run.

Then automate the fastest one:

```bash
python tools/probe_reset.py --test "start:0.6,a:0.8,a:1.2" --trials 5
```

Adjust the button sequence until it recovers reliably, then put it into
`ResetSequence.time_trial_restart()` in `f1ai/rl/f1_backend.py`.

**This gates everything else.** Under 6 s and reset is a non-issue; over 15 s
and it dominates the training budget.

## Step 4 — the real capture cost

```bash
python tools/benchmark_capture.py --seconds 10 --save
```

With the game on screen. Then open `docs/capture_sample.png` and check the crop
actually contains road — not sky, not your own bodywork. Adjust `crop_top` /
`crop_frac` in `f1_backend.py` if not.

```bash
python tools/benchmark_inference.py
```

Re-run with the game running. The gap against the idle numbers (1.0–1.5 ms) is
the GPU contention cost, and that is the one that counts.

## Step 5 — the overlay

```bash
python -m f1ai.hud.overlay
```

Confirm `click-through: on` in its startup output. If the car stops responding
to the pad when the overlay appears, the window took focus — F1 25 drops
gamepad input the instant it loses focus.

---

## What to report back

- packet sizes from step 1, and whether `--decode` tracked correctly
- median latency from step 2
- **the fastest reliable reset method and its time** — the most important number
- capture + inference timings with the game running

## What is still unbuilt

- **Async actor/learner.** The loop is synchronous because the mock outruns
  real time. F1 25 will not pause for a gradient step.
- **The recorder** for behavioural-cloning warm start from your own laps.
- `f1_backend.py` has never run. Expect the first failures there.
