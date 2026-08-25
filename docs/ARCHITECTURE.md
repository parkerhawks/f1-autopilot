# Architecture

~5,300 lines across 22 modules and 20 tools. This is the reading order, and the
reasoning behind the boundaries.

## The one idea that shapes everything

F1 25 is a closed commercial game: no `reset()`, no frame stepping, no time
acceleration, everything at 1x wall clock. So the project is built around a
**simulated stand-in** that speaks the game's exact wire format, and a hard line
between "code that knows about the game" and "code that does not".

```
f1ai/telemetry/   speaks F1 25's UDP protocol      \  swapped when the
f1ai/control/     speaks to F1 25 via a virtual pad /  real game arrives
f1ai/sim/         a fake F1 25 that emits the same packets

f1ai/rl/          environment, replay buffer, SAC   \  never changes
f1ai/hud/         live overlay                      /  between the two
```

Everything in the second group is developed and tested against the mock. When
the game arrives, only the first group is swapped. That is why the mock exists
and why it reproduces the game's quirks rather than being convenient.

---

## Reading order

### 1. Start here: the simulated world

| file | what it is |
|---|---|
| `f1ai/sim/track.py` | Closed circuit as a polar curve. Centreline, arc length, curvature lookahead. |
| `f1ai/sim/vehicle.py` | Kinematic bicycle model. Deliberately not good physics — just causally correct. |
| `f1ai/sim/render.py` | Driver's-eye camera. This is what makes it a vision project. |
| `f1ai/sim/expert.py` | Pure-pursuit controller. The baseline and the stand-in demonstrator. |
| `f1ai/sim/mock_game.py` | Wraps all of the above and emits real-format UDP on port 20777. |

Run `python tools/preview_render.py` and look at `docs/render_preview.png`
before anything else. If you do not understand what the CNN sees, nothing
downstream will make sense.

**Non-obvious:** the camera sits 6 m up, not at cockpit height. Ground-plane
position projects as `h/depth`, so a 1.2 m camera compresses everything past
~45 m into under two pixels and the network gets no signal about upcoming
curvature. See the comment block in `render.py`.

### 2. The learning problem

| file | what it is |
|---|---|
| `f1ai/rl/env.py` | reset / step / reward / termination. Backend-agnostic. |
| `f1ai/rl/buffer.py` | Replay memory. Stores single frames, reconstructs stacks. |
| `f1ai/rl/sac.py` | Encoder, actor, twin critics, entropy tuning. |

**The central asymmetry**, in `env.py`:

- The **observation** contains only what a real car knows about itself: pixels
  plus speed, yaw rate, g-forces, and its own last action.
- The **reward** may use privileged track knowledge — lateral offset, lap
  distance — because it is only evaluated during training and never ships.

If lateral offset leaked into the observation, the policy could ignore the
camera entirely and the CNN would be decorative. `selftest_env.py` correlates
every state dimension against lateral offset and fails if anything leaks.

**Non-obvious:** the buffer stores one frame per step (18 KB) and rebuilds
3-frame stacks from neighbouring indices, 6x cheaper than storing stacks
(2.8 GB vs 16 GB at 150k). The risk is splicing frames across an episode reset,
which trains the policy on transitions that never happened and raises nothing.
Every slot carries an episode id, and `selftest_buffer.py` compares every
reconstructed stack byte-for-byte against what the environment emitted.

### 3. Training and evaluation

| file | what it is |
|---|---|
| `tools/train_sac.py` | The loop. Writes `runs/<name>/metrics.csv` every episode. |
| `tools/plot_training.py` | Six diagnostic panels. |
| `tools/eval_policy.py` | Lap times **and track limits**. Never trust one without the other. |

### 4. The game-facing layer (untested against the real game)

| file | what it is |
|---|---|
| `f1ai/telemetry/packets.py` | Byte layouts for F1 25's UDP packets, format 2025. |
| `f1ai/telemetry/listener.py` | Threaded socket, keeps newest packet per type. |
| `f1ai/control/gamepad.py` | Analog virtual Xbox pad via the ViGEmBus driver. |

**Partially verified against F1 25 (2026-08-17).** Confirmed live at Monza:
`packetFormat` 2025, the 29-byte header (the player car index resolved to the
right car), CarTelemetry at a 60-byte stride, and Motion at a 60-byte stride —
speed, gear, steering, throttle, brake and world coordinates all decoded to
coherent values at 339 km/h in 8th gear on the main straight.

**Still unverified:** LapData (packet 2). Its stride and `lapDistance` offset
must be found empirically with `--scan-floats 2`, and note that **LapData is
only transmitted during an active session** — in a menu or the garage, packets
0 and 6 keep flowing while 2 does not, which looks like a parser failure and
is not one.

`f1ai/control/gamepad.py` has not been exercised against the game at all.

### 5. The overlay

| file | what it is |
|---|---|
| `f1ai/hud/bus.py` | Fire-and-forget UDP between training and display. |
| `f1ai/hud/panel.py` | Renders the HUD to an image. No GUI dependency. |
| `f1ai/hud/overlay.py` | Click-through always-on-top window. |

The overlay runs in **its own process**. A GUI redraw or an outright crash in
the display must never add latency to the 30 Hz control loop.

---

## What is deliberately absent

- **No async actor/learner yet.** The loop is synchronous because the mock runs
  faster than real time. F1 25 will not wait for a gradient step, so the learner
  moves to its own thread before the game integration. `learn_ratio` in
  `train_sac.py` exists to keep that budget honest.
- **No behavioural cloning.** run2 learned from scratch. BC warm start matters
  for the real game, where samples cost wall-clock time.
- **No automated reset.** The single highest-risk item in the project and it
  cannot be built until the game exists.
- **`f1ai/model/policy.py` is unused by SAC.** It is the supervised
  behavioural-cloning architecture and the subject of
  `tools/benchmark_inference.py`. SAC uses `f1ai/rl/sac.py` instead.

---

## Testing

```bash
python tools/run_tests.py
```

Seven suites, 56 checks, ~11 seconds. Suites are ordered from most fundamental
to most dependent — **when several fail, fix the first one**, because a broken
parser or buffer makes everything downstream look broken too.

| suite | protects against |
|---|---|
| `selftest_packets` | wrong byte offsets silently corrupting every label |
| `selftest_reward` | a reward whose optimal policy is one you do not want |
| `selftest_hud` | an overlay that freezes or strobes during a live demo |
| `selftest_sim` | a demonstrator that looks healthy while driving off the map |
| `selftest_buffer` | replay stacks spliced across episode boundaries |
| `selftest_env` | a reward that does not rank driving quality |
| `selftest_sac` | log-prob maths, encoder isolation, target-network lag |

Each is a standalone script; run it directly for the detail behind a failure.
