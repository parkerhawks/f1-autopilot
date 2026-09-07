# f1-autopilot

Teach a neural network to drive a lap of F1 25 — from screen pixels and the
game's own telemetry, with no access to the game's internals.

It learns any circuit the same way: you drive a handful of laps, it imitates
them, then reinforcement learning takes over and tries to beat you.

```
    your laps  ->  behavioural cloning  ->  SAC  ->  faster laps
                        (imitate)         (improve)
```

**Results on Monza** (RTX 4070, ~14 h total training): complete laps at
**85.7 s** against the author's own 82.4 s — within about 4%. Sixty-plus
completed laps, average speed 197 km/h, driving the full circuit including both
chicanes, the Lesmos, Ascari and Parabolica.

> **Status: working, not finished.** The agent drives complete laps reliably.
> Track-limits enforcement is newer than the lap times above, so the headline
> figure was set before limits were strictly policed — treat it as indicative.
> See [Known limitations](#known-limitations).

---

## What makes this hard

F1 25 is a closed commercial game. There is no `reset()`, no frame stepping, no
time acceleration. Everything happens at 1x wall clock, and a crashed car has to
be recovered through the pause menu like a human would. Almost every design
decision here follows from that.

| constraint | consequence |
|---|---|
| No simulator API | Screen capture in, virtual gamepad out |
| Real time only | 30 Hz control loop that must never stall |
| No `reset()` | Menu navigation, driven by telemetry state |
| GPU shared with the game | The learner backs off whenever the actor misses a deadline |
| Samples cost wall-clock | Behavioural cloning warm start; every transition reused |

---

## How it works

**Perception.** DXGI desktop duplication grabs the game window at 30 Hz. A
cropped band of road is downsampled to 192x96 greyscale and stacked three deep.

**Proprioception.** The game's documented UDP telemetry gives speed, gear,
g-forces, and — crucially — the controls it actually *applied*, which differ
from what was commanded because F1 25 ramps gamepad steering over ~82 ms.

**The map.** Built from your own recorded laps: curvature and a reference speed
profile ahead, plus the racing line in world coordinates. The map says what the
road ahead does; vision says where on it the car actually is.

**The policy.** SAC with a convolutional encoder. Twin critics, automatic
entropy tuning, and an actor that reads detached features so the policy cannot
reshape the representation its own value estimate depends on.

**Control.** A virtual Xbox pad through ViGEmBus. Analog, because quantising
steering to left/right/none makes the problem unlearnable.

There is also a **simulator** (`f1ai/sim/`) — a bicycle model on a synthetic
circuit that emits real-format UDP packets and renders a driver's-eye view. The
whole pipeline is testable against it without owning the game, and it caught
bugs that would have been far more expensive to find at 1x realtime.

Full tour: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Requirements

- **Windows 10/11**, NVIDIA GPU with ~8 GB VRAM (developed on an RTX 4070)
- **Python 3.11**
- **F1 25** — the telemetry format is set by `UDP Format: 2025`. Other years
  need the byte offsets re-verified; `tools/probe_telemetry.py` does that.
- 16 GB RAM (a 150k-transition replay buffer is ~2.8 GB)

```powershell
git clone https://github.com/<you>/f1-autopilot
cd f1-autopilot
python -m venv .venv
```

**Install PyTorch with CUDA first.** A plain `pip install torch` can give you a
CPU-only build on Windows, and training is unusably slow on CPU — a gradient
step goes from ~30 ms to several seconds. Install from PyTorch's own index:

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then the rest:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
.venv\Scripts\python.exe -m pip install vgamepad
```

`vgamepad` is installed separately and deliberately — it sets up the ViGEmBus
kernel driver and raises a UAC prompt. It is only needed to *control* the game;
recording laps, the simulator and the whole test suite work without it.

**Always install through `.venv\Scripts\python.exe -m pip`, never bare `pip`.**
Bare `pip` resolves to the system Python and the package lands somewhere the
project cannot see, while `pip list` cheerfully shows it installed.

Verify with no game running:

```powershell
.venv\Scripts\python.exe tools\run_tests.py
```

14 suites, ~100 checks, about 30 seconds. All of it runs against the simulator.

---

## Quickstart: any track

Everything below works for any circuit — swap the `--track` name.

### 1. Game settings

| setting | value | why |
|---|---|---|
| Display | **borderless windowed** | exclusive fullscreen blocks capture |
| Resolution | 1080p | halves preprocessing cost |
| Frame cap | 30-60 | frames above your control rate just steal GPU from the learner |
| Motion blur | **off** | blur corrupts training frames |
| Preset | low | simpler visuals help the CNN |
| Camera | **TV Pod** or similar | a cockpit view spends half its pixels on the halo |
| UDP telemetry | on, `127.0.0.1:20777`, 60 Hz, **format 2025** | |

Play **offline, single-player Time Trial only**. See [Fair use](#fair-use).

### 2. Verify the game talks to us

```powershell
.venv\Scripts\python.exe tools\probe_telemetry.py
```

```powershell
.venv\Scripts\python.exe tools\probe_telemetry.py --decode
```

```powershell
.venv\Scripts\python.exe tools\probe_telemetry.py --scan-floats 2 --scan-seconds 150
```

The last one locates `lapDistance` empirically — drive a full lap during it.
Write the reported stride and offset into `f1ai/telemetry/packets.py`.

Byte offsets are the one thing worth measuring rather than trusting. If
anything reads wrong, `--watch-bytes 2` reports which bytes change and when.

### 3. Check what the network sees

```powershell
.venv\Scripts\python.exe tools\tune_crop.py --delay 12 --samples 5
```

Open `docs/crop_tuning.png`. The crop must contain road with both edges visible
and the vanishing point inside it — no wheel, no HUD, nothing outside the game
window. **Do not skip this.** No amount of training fixes a camera pointed at
the wrong thing.

### 4. Record laps and build the map

```powershell
.venv\Scripts\python.exe tools\record_laps.py --laps 5
```

```powershell
.venv\Scripts\python.exe tools\build_map.py demos\<session> --name silverstone
```

That writes `maps\silverstone.npz`, which every later command finds via
`--track silverstone`.

Drive cleanly. The map becomes the definition of where the track is, so a lap
that cuts a corner teaches the agent that cutting is correct.

### 5. Warm start, then reinforce

```powershell
.venv\Scripts\python.exe tools\train_bc.py demos\<session> --track silverstone
```

```powershell
.venv\Scripts\python.exe tools\check_bc.py runs\bc\bc.pt demos\<session>\<lap>.npz
```

```powershell
.venv\Scripts\python.exe tools\train_sac.py --backend f1 --track silverstone --init-from runs\bc\bc.pt --steps 600000 --batch 64 --name run1
```

### 6. Watch it drive

```powershell
.venv\Scripts\python.exe tools\drive.py runs\run1\latest.pt --track silverstone --hud
```

```powershell
.venv\Scripts\python.exe -m f1ai.hud.overlay
```

---

## Tooling

| tool | purpose |
|---|---|
| `run_tests.py` | 14 suites, no game required |
| `probe_telemetry.py` | verify packet layouts; find fields empirically |
| `probe_backend.py` | capture + telemetry + pad, together |
| `probe_reset.py` | discover and time the reset path |
| `tune_crop.py` | see exactly what the CNN receives |
| `record_laps.py` | capture human demonstrations |
| `build_map.py` | curvature, speed profile, racing line |
| `train_bc.py` / `check_bc.py` | imitation warm start, and whether it learned |
| `train_sac.py` | the training loop |
| `drive.py` | run a checkpoint, report lap times **and track limits** |
| `plot_training.py` / `dashboard.py` | diagnostics |

---

## Known limitations

- **Track limits are newly enforced.** The 85.7 s figure predates strict
  enforcement. Expect the first strictly-legal laps to be slower.
- **One track per map.** No cross-track generalisation is attempted; the policy
  learns a circuit.
- **The game crashes.** Roughly every 5-6 hours in testing. Training suspends
  and resumes automatically, and checkpoints every 20k steps.
- **Reset sequences are game-version specific.** The button paths in
  `f1ai/rl/f1_backend.py` are for F1 25 Time Trial. Verify with
  `probe_reset.py` before a long run.
- **Windows only**, because of ViGEmBus and DXGI.

---

## Fair use

This reads the game's **own documented UDP telemetry output** and sends input
through a **virtual controller**. It does not modify game files, read process
memory, or interfere with the game.

Even so:

- **Play offline.** Time Trial submits to global leaderboards, and an automated
  lap has no business there.
- **Never multiplayer or ranked.** Not once, not to test.
- **Check the publisher's terms yourself.** Most prohibit bots and automation;
  keeping this to private, offline, single-player use is the point.

Everything above was developed offline in single-player with no leaderboard
submission.

---

## Acknowledgements

- Bojarski et al., *End to End Learning for Self-Driving Cars* (2016) — PilotNet
- Haarnoja et al., *Soft Actor-Critic* (2018)
- Yarats et al., *Improving Sample Efficiency in Model-Free RL from Images*
  (2019) — the detached-encoder arrangement
- Wurman et al., *Outracing champion Gran Turismo drivers with deep
  reinforcement learning*, Nature (2022)

## License

MIT — see [LICENSE](LICENSE).
