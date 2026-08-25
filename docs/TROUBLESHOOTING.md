# Troubleshooting

Every entry below is a bug this project actually hit, with the signal that
eventually gave it away. They are grouped by what you observe, because that is
what you have when something goes wrong.

## The two rules that matter most

**1. When several tests fail, fix the first one.** Suites are ordered from most
fundamental to most dependent. A broken parser or replay buffer makes
everything downstream look broken too.

**2. A metric that looks wrong at the very start or very end of a series is
probably the plotting, not the model.** See "return collapses late in training".

---

## Training looks healthy but the policy is bad

### Lap times are great and the policy is cheating

**Seen in run1:** 27.50s laps against a 35.60s expert. Return climbing, Q
tracking its target, entropy exactly on target. Everything looked excellent.

It was spending 2.8% of each lap past the track boundary. The reward paid ~2.8
per step for progress and charged only 0.6/m for leaving the surface, so
running wide was *strictly profitable*. SAC did not malfunction — it solved the
problem as posed.

**Nothing in the training curves showed this.** Only a track-limits evaluation
did.

```bash
python tools/eval_policy.py runs/<name>/final.pt --expert
```

Look at `outside surface`. Above ~1% means the lap times are not legitimate.

**Fix:** do not raise the penalty — a fast enough policy just pays it. Remove
the incentive by denying progress *credit* off the surface (`OFFTRACK_FADE` in
`env.py`). Then confirm the economics before retraining:

```bash
python tools/selftest_reward.py
```

### The agent completes laps but never gets faster

Distinguish two cases in `plot_training.py`:

- **eval distance plateaus at the episode cap, lap time flat** — it has learned
  to survive, not to race. Usually the reward pays too much for staying alive
  relative to progress.
- **eval distance still climbing** — it has not finished learning to complete a
  lap. Not a plateau; give it more steps.

### Q climbs while actual return stays flat

The `critic honesty` panel. The critic is hallucinating value and the policy is
optimising a fiction. Suspect the target network or the reward scale — not
"needs more steps". Q sitting slightly *below* actual return is healthy;
conservative is the safe direction.

---

## Metrics that lie

### Return collapses in the last few thousand steps

**Almost certainly the smoothing.** `np.convolve(..., mode="same")` zero-pads
the edges and drags both ends of the curve toward zero. This project's run2
appeared to degrade from 14,000 to 7,600 while the raw episodes were flat at
14,400.

Check the CSV directly before believing any edge effect:

```bash
python -c "import csv;[print(r['step'],r['episode_return']) for r in list(csv.DictReader(open('runs/run2/metrics.csv')))[-10:]]"
```

### Entropy reads 0.00 on alternate rows

The actor and alpha update on a slower clock than the critic (`actor_every`),
so their metrics are absent on most steps. If the logger fills a missing key
with `0.0`, it looks exactly like entropy collapse. `train_sac.py` merges
metrics rather than replacing them.

### `log_std` is rising, so it must be exploring more

**No.** For a tanh-squashed Gaussian the support is bounded, so a large
pre-squash std piles mass against ±1 and spikes the density there. Entropy is
**non-monotonic in std** — it peaks near `std ≈ 1` and falls again. Measured in
this codebase: entropy `+2.00` at `log_std=0`, but `-13.80` at `log_std=-6`.

Watch the `entropy` metric directly. Never infer it from `log_std`.

### Lap times are suspiciously fast

Episodes start at a random point on the lap, so the first line crossing
completes a *partial* lap. Recording it deflates the headline metric — this
showed 13.53s against a true 35.63s. `env.py` uses the first crossing only to
start the clock (`_lap_valid`).

---

## The mock game

### The expert completes laps but the car is nowhere near the track

**Seen during the build:** the car orbited at 684 m radius on a circuit
spanning 280–520 m, reporting entirely healthy telemetry.

Centreline points are sampled uniformly in the *polar parameter*, not in arc
length — spacing varies by the full wobble amplitude. Converting a lap distance
to an index with `distance / length * N` is wrong by tens of metres exactly in
the corners. Always go through `Track.index_at_distance`.

```bash
python tools/selftest_sim.py
```

Watch `cross-track error`. Anything above the track half-width means the
demonstrator is not usable, no matter how good the lap times look.

### The car understeers out of every corner

The controller divided by the *static* maximum steering angle while the vehicle
model reduces steering authority with speed. That is a ~5x understeer above
200 km/h, and it reads as a tracking bug rather than a units bug. Controllers
must call `effective_max_steer(v)`.

### The expert never brakes

The track's corners are too fast to require it. A circuit that can be taken flat
makes Phase 2 meaningless, because a constant-full-throttle policy scores as a
success. Defaults are tuned for ~130 km/h corners against 330 km/h straights.

---

## The overlay

### The window freezes on its first frame

The subscriber must read **non-blocking**. A socket timeout means "wait for a
gap in the stream", and at 30 Hz there is never a gap — the drain loop only
returns when the publisher dies. `selftest_hud.py` asserts `latest()` returns
in under 100 ms against a live flood.

### Nothing arrives, but the publisher reports success

On Windows, sending UDP to a port with no listener produces ICMP Port
Unreachable, which comes back on the **sending** socket as `WSAECONNRESET` and
poisons it — every later datagram fails. Training normally starts before the
overlay is opened, so this is the common case, not the edge case.
`SIO_UDP_CONNRESET` suppresses it. Check `pub.dropped`.

### The camera view strobes

Frames ride on only every Nth snapshot to keep datagrams small, so most arrive
frameless. The subscriber carries the last frame forward.

### The overlay is invisible over the game

F1 25 must run in **borderless windowed** mode. Exclusive fullscreen owns the
display surface and nothing can draw over it.

### The car stops responding when the overlay appears

The overlay took focus. F1 25 stops accepting gamepad input the instant it
loses focus, which silently kills the agent's control. `WS_EX_NOACTIVATE`
prevents it — check `click-through: on` in the overlay's startup output.

---

## Against the real game (untested)

The telemetry and gamepad layers have never seen F1 25. Expect the first
failures here.

| symptom | first thing to check |
|---|---|
| no packets at all | telemetry settings, UDP format set to 2025, firewall |
| packet sizes differ from `EXPECTED_SIZES` | EA changed the layout; fix `packets.py`, do not loosen the assert |
| decoded values are nonsense | wrong `packetFormat`, or the player is not car index 0 |
| gamepad does nothing | ViGEmBus not installed, or the game window lacks focus |
| latency above ~80 ms | v-sync on, frame cap missing, or exclusive fullscreen |

```bash
python tools/probe_telemetry.py            # rates and sizes
python tools/probe_telemetry.py --decode   # does it track what you do?
python tools/measure_latency.py            # the gate
```
