"""Drive F1 25 with a trained checkpoint. No learning, no exploration noise.

    python tools/drive.py runs/bc/bc.pt
    python tools/drive.py runs/f1-run1/latest.pt --hud --minutes 10

This is how a policy gets judged. Training curves and validation losses are
proxies; lap times and where the car ends up are the thing itself.

Reports what actually matters and nothing else: laps completed, lap times, how
much of each lap was spent off the racing surface, and how far it got before
it stopped making progress.

Safety: the pad is released on every exit path, including Ctrl-C. Leaving a
virtual controller holding full throttle is a genuinely bad time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import (  # noqa: E402
    CONTROL_HZ, FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM, RacingEnv,
)
from f1ai.rl.sac import SACAgent, SACConfig  # noqa: E402
from f1ai.rl.track_map import TrackMap  # noqa: E402


def _load_map(path: str | None):
    """Load the track map, loudly. A silently-absent map zeroes the preview
    features, and the policy then drives with a blindfold it was never trained
    to wear -- which looks like the policy failing, not the plumbing."""
    if not path:
        print("no track map requested -- preview features will be zero")
        return None
    from f1ai.rl.track_map import map_path
    p = map_path(path)
    if not p.exists():
        print(f"WARNING: no map at {p}; preview features will be zero.")
        print("If the policy was TRAINED with a map, it will drive badly.")
        return None
    tmap = TrackMap.load(p)
    print(f"track map {p.name}: {tmap.length:.0f} m")
    return tmap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--crop-top", type=float, default=0.38,
                    help="top of the crop as a fraction of frame height; "
                         "find it with tools/tune_crop.py")
    ap.add_argument("--crop-frac", type=float, default=0.26,
                    help="crop height as a fraction of frame height")
    ap.add_argument("--region", default=None)
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--track", dest="map", default="track",
                    help="track name, or an explicit path to a .npz map")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of using the mean; usually "
                         "you want the mean, which is the policy's best guess")
    ap.add_argument("--no-reset", action="store_true",
                    help="do not auto-reset after a crash; stop instead")
    a = ap.parse_args()

    from f1ai.rl.f1_backend import F1Backend

    region = tuple(int(v) for v in a.region.split(",")) if a.region else None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                     SACConfig(), device=str(dev))
    agent.load_state_dict(torch.load(a.checkpoint, map_location=dev,
                                     weights_only=False))
    agent.encoder.eval()
    agent.actor.eval()
    print(f"loaded {a.checkpoint}")

    backend = F1Backend(capture_region=region,
                         crop_top=a.crop_top, crop_frac=a.crop_frac)
    tmap = _load_map(a.map)
    env = RacingEnv(backend, track_map=tmap)

    pub = None
    if a.hud:
        from f1ai.hud.bus import HudPublisher, HudSnapshot
        pub = HudPublisher()

    print("\nFocus the game. The car will start driving in 5s.")
    print("Ctrl-C to stop -- the pad is always released on exit.\n")
    time.sleep(5)

    dt = 1.0 / CONTROL_HZ
    frames, state = env.reset()
    t0 = time.perf_counter()
    next_tick = t0
    step = 0
    late = 0
    episodes = 0
    offtrack_steps = 0
    all_laps: list[float] = []
    last_report = t0
    act_log: list[np.ndarray] = []

    try:
        while (time.perf_counter() - t0) < a.minutes * 60:
            raw = agent.act(frames, state, deterministic=not a.stochastic)
            action = SACAgent.to_env_action(raw)
            res = env.step(action)
            frames, state = res.frames, res.state
            step += 1
            act_log.append(action.copy())

            if res.info.get("offtrack_frac", 0.0) > 0:
                pass  # cumulative; read at the end
            obs_wheels = getattr(backend, "_last_wheels", 0)

            if pub is not None and step % 3 == 0:
                laps = [t for t in env.lap_times if t > 0]
                snap = HudSnapshot(
                    mode="EVAL",
                    speed_kph=res.info["speed_kph"],
                    gear=int(state[4] * 8),
                    steer=float(action[0]), throttle=float(action[1]),
                    brake=float(action[2]),
                    lap_times=laps + all_laps,
                    best_lap_s=min(laps + all_laps) if (laps or all_laps) else None,
                    last_lap_s=(laps + all_laps)[-1] if (laps or all_laps) else None,
                    env_steps=step, episodes=episodes,
                )
                snap.attach_frame(frames[-1])
                pub.publish(snap)

            now = time.perf_counter()
            if now - last_report > 10.0:
                laps = [t for t in env.lap_times if t > 0]
                best = min(laps + all_laps) if (laps or all_laps) else None
                print(f"  {now - t0:5.0f}s  step {step:>6}  "
                      f"{res.info['speed_kph']:5.0f} km/h  "
                      f"laps {len(all_laps) + len(laps):>2}  "
                      f"best {f'{best:.2f}s' if best else '--':>8}  "
                      f"offtrack {100 * res.info['offtrack_frac']:4.1f}%  "
                      f"late {100 * late / max(1, step):4.1f}%")
                last_report = now

            if res.terminated or res.truncated:
                all_laps += [t for t in env.lap_times if t > 0]
                offtrack_steps += int(res.info["offtrack_frac"] * step)
                episodes += 1
                reason = res.info.get("term_reason") or (
                    "time limit" if res.truncated else "unknown")
                print(f"  episode {episodes}: {reason} after "
                      f"{res.info['distance']:.0f} m at "
                      f"{res.info['speed_kph']:.0f} km/h, "
                      f"{res.info['wheels_off']} wheels off")
                if a.no_reset:
                    break
                frames, state = env.reset()

            if backend.stale_seconds() > 20.0:
                print("\n!! telemetry frozen -- the game has crashed or paused")
                break

            next_tick += dt
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                late += 1
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nstopped by user.")
    finally:
        # Always release the controls, on every path out of the loop.
        backend.apply(0.0, 0.0, 0.0)
        all_laps += [t for t in env.lap_times if t > 0]
        backend.close()
        if pub is not None:
            pub.close()

    elapsed = time.perf_counter() - t0
    print(f"\n{step:,} steps in {elapsed / 60:.1f} min "
          f"({step / max(1e-9, elapsed):.1f} Hz, {100 * late / max(1, step):.1f}% late)")
    print(f"episodes: {episodes}")

    # What the policy actually DID. A driver that never brakes and never turns
    # is a different problem from one that turns the wrong way, and the two are
    # indistinguishable from lap times alone.
    if act_log:
        acts = np.asarray(act_log)
        print("\npolicy output over the whole session")
        print(f"{'channel':<10} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} "
              f"{'% active':>9}")
        for k, name in enumerate(("steer", "throttle", "brake")):
            col = acts[:, k]
            active = float((np.abs(col) > 0.05).mean())
            print(f"{name:<10} {col.mean():>8.3f} {col.std():>8.3f} "
                  f"{col.min():>8.3f} {col.max():>8.3f} {100 * active:>8.1f}%")

        if acts[:, 2].max() < 0.05:
            print("\n  THE POLICY NEVER BRAKES. At Monza the first chicane")
            print("  needs braking from ~340 to ~80 km/h, so it will always")
            print("  arrive too fast however good the steering is.")
        if acts[:, 0].std() < 0.02:
            print("\n  THE POLICY BARELY STEERS. Output variance is far below")
            print("  a human's (~0.19); it is driving in a straight line.")
    if all_laps:
        print(f"laps: {len(all_laps)}  best {min(all_laps):.2f}s  "
              f"mean {sum(all_laps) / len(all_laps):.2f}s")
        print(f"  {['%.2f' % t for t in all_laps]}")
    else:
        print("laps: none completed")
        print("\nFor a behavioural-cloning warm start this is expected: a")
        print("single demonstration lap contains no recovery data, so the")
        print("first small deviation leads somewhere the policy has never")
        print("seen. Distance before the first crash is the number that")
        print("matters -- it is what RL has to improve on.")


if __name__ == "__main__":
    main()
