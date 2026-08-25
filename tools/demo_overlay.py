"""Drive the mock environment and publish live HUD snapshots.

This is the display half of the system end to end -- environment, renderer,
telemetry bus, overlay -- with the expert standing in for a trained policy.

The lap times it shows are REAL expert laps, not a scripted improvement curve.
They will sit near-flat around 35.6s, because a pure-pursuit controller does
not learn. That is the honest baseline: when SAC replaces the expert here, any
downward slope in the trace is genuine.

Run this alongside:  python -m f1ai.hud.overlay
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.hud.bus import HudPublisher, HudSnapshot  # noqa: E402
from f1ai.rl.env import CONTROL_HZ, MockBackend, RacingEnv  # noqa: E402
from f1ai.sim.expert import PurePursuitExpert  # noqa: E402

FRAME_EVERY = 3          # publish the camera image at 10 Hz, numbers at 30 Hz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to run (default: forever)")
    ap.add_argument("--realtime", action="store_true", default=True)
    ap.add_argument("--fast", dest="realtime", action="store_false",
                    help="run as fast as possible instead of at 30 Hz")
    a = ap.parse_args()

    backend = MockBackend(seed=1)
    env = RacingEnv(backend)
    expert = PurePursuitExpert(backend.track)
    pub = HudPublisher()

    frames, state = env.reset()
    dt = 1.0 / CONTROL_HZ
    t0 = time.perf_counter()
    next_tick = t0
    k = 0
    episode_return = 0.0
    lap_clock = 0
    laps_seen = 0

    print(f"publishing to udp://127.0.0.1:20779 at {CONTROL_HZ:g} Hz")
    print("start the overlay in another terminal:  python -m f1ai.hud.overlay")
    print("Ctrl-C to stop.\n")

    try:
        while a.duration is None or (time.perf_counter() - t0) < a.duration:
            steer, throttle, brake = expert.act(backend.veh)
            res = env.step(np.array([steer, throttle, brake], np.float32))
            episode_return += res.reward
            lap_clock += 1
            k += 1

            # Restart the on-screen lap clock the moment a lap is recorded.
            if len(env.lap_times) > laps_seen:
                laps_seen = len(env.lap_times)
                lap_clock = 0

            snap = HudSnapshot(
                mode="DEMO",
                speed_kph=res.info["speed_kph"],
                gear=backend.veh.gear,
                steer=steer, throttle=throttle, brake=brake,
                current_lap_s=lap_clock / CONTROL_HZ,
                last_lap_s=env.lap_times[-1] if env.lap_times else None,
                best_lap_s=min(env.lap_times) if env.lap_times else None,
                lap_times=list(env.lap_times),
                step_reward=res.reward,
                episode_return=episode_return,
                env_steps=k,
                episodes=1,
            )
            if k % FRAME_EVERY == 0:
                snap.attach_frame(res.frames[-1])
            pub.publish(snap)

            if res.terminated or res.truncated:
                frames, state = env.reset()
                episode_return = 0.0
                lap_clock = 0
                laps_seen = 0

            if a.realtime:
                next_tick += dt
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        print(f"snapshots: {pub.sent} sent, {pub.dropped} dropped "
              f"over {k} steps")
        pub.close()
        print(f"{len(env.lap_times)} laps completed")
        if env.lap_times:
            print(f"lap times: {['%.2f' % t for t in env.lap_times]}")


if __name__ == "__main__":
    main()
