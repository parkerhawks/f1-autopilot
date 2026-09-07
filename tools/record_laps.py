"""Record your own laps, to warm-start the policy before RL touches it.

    python tools/record_laps.py --laps 5

Drive normally. Every lap is saved separately with its time and a cleanliness
figure, so you can keep the good ones and throw the rest away afterwards --
one scruffy lap in a small demonstration set does real damage.

**No virtual pad is created**, so your own controller keeps working.

The supervision signal is free: F1 25's telemetry reports the steering,
throttle and brake it applied, so there is no need to read your controller.

RAW data is stored -- frames plus the telemetry fields -- rather than
finished observation vectors. Observation design is still changing, and a
recording that bakes in today's 17-dimensional state vector would be worthless
the next time that changes. `train_bc.py` builds the observations at training
time from whatever the current definition is.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import CONTROL_HZ, FRAME_H, FRAME_W  # noqa: E402

DEMOS = Path(__file__).resolve().parent.parent / "demos"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--laps", type=int, default=5, help="laps to record")
    ap.add_argument("--name", default=None, help="session name")
    ap.add_argument("--crop-top", type=float, default=0.38,
                    help="top of the crop as a fraction of frame height; "
                         "find it with tools/tune_crop.py")
    ap.add_argument("--crop-frac", type=float, default=0.26,
                    help="crop height as a fraction of frame height")
    ap.add_argument("--region", default=None)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    a = ap.parse_args()

    from f1ai.rl.f1_backend import F1Backend

    region = tuple(int(v) for v in a.region.split(",")) if a.region else None
    session = a.name or time.strftime("laps-%Y%m%d-%H%M%S")
    out = DEMOS / session
    out.mkdir(parents=True, exist_ok=True)

    print("opening capture + telemetry (no virtual pad -- you are driving)")
    backend = F1Backend(capture_region=region, create_pad=False,
                         crop_top=a.crop_top, crop_frac=a.crop_frac)
    print("ready.\n")
    print("Drive normally. Recording starts when you cross the line.")
    print("Ctrl-C to stop early.\n")

    dt = 1.0 / CONTROL_HZ
    buf: dict[str, list] = _empty()
    lap_index = 0
    prev_s = None
    lap_start = None
    saved = []
    t_start = time.perf_counter()
    next_tick = t_start

    try:
        while (lap_index < a.laps
               and (time.perf_counter() - t_start) < a.max_minutes * 60):
            obs = backend.observe()

            # A large backwards jump in lap distance is the start/finish line.
            crossed = (prev_s is not None
                       and obs.lap_distance < prev_s - 100.0)
            prev_s = obs.lap_distance

            if crossed:
                if lap_start is not None and len(buf["frame"]) > 60:
                    lap_time = (time.perf_counter() - lap_start)
                    info = _save(out, lap_index, buf, lap_time)
                    saved.append(info)
                    print(f"  lap {lap_index + 1}: {lap_time:6.2f}s   "
                          f"{info['steps']} steps   "
                          f"{info['clean']:.1%} clean   "
                          f"-> {info['path'].name}")
                    lap_index += 1
                buf = _empty()
                lap_start = time.perf_counter()

            if lap_start is not None:
                buf["frame"].append(obs.frame.copy())
                buf["speed"].append(obs.speed_mps)
                buf["yaw_rate"].append(obs.yaw_rate)
                buf["g_lat"].append(obs.g_lat)
                buf["g_lon"].append(obs.g_lon)
                buf["gear"].append(obs.gear)
                buf["steer"].append(obs.applied_steer)
                buf["throttle"].append(obs.applied_throttle)
                buf["brake"].append(obs.applied_brake)
                buf["lap_distance"].append(obs.lap_distance)
                buf["wheels_off"].append(obs.wheels_off)
                # World position, so build_map can lay down a real racing line.
                # Without it the map has curvature but no geometry, and a
                # tarmac run-off is indistinguishable from the track.
                buf["world_x"].append(obs.world_x)
                buf["world_z"].append(obs.world_z)

            next_tick += dt
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        backend.close()

    if not saved:
        print("\nNo complete laps recorded -- you need to cross the line twice")
        print("(once to start the recording, once to finish the lap).")
        return

    print(f"\n{len(saved)} laps in {out}")
    best = min(saved, key=lambda s: s["lap_time"])
    print(f"best: {best['lap_time']:.2f}s ({best['path'].name})\n")
    print("Keep the clean, representative laps and DELETE the rest. A few")
    print("good laps beat many mediocre ones -- behavioural cloning copies")
    print("whatever it is shown, mistakes included.")
    print(f"\nThen:  python tools/train_bc.py {out}")


def _empty() -> dict[str, list]:
    return {k: [] for k in
            ("frame", "speed", "yaw_rate", "g_lat", "g_lon", "gear",
             "steer", "throttle", "brake", "lap_distance", "wheels_off",
             "world_x", "world_z")}


def _save(out: Path, index: int, buf: dict, lap_time: float) -> dict:
    frames = np.asarray(buf["frame"], np.uint8)
    wheels = np.asarray(buf["wheels_off"], np.int8)
    clean = float((wheels == 0).mean())
    path = out / f"lap{index:02d}_{lap_time:.2f}s.npz"
    np.savez_compressed(
        path,
        frame=frames,
        speed=np.asarray(buf["speed"], np.float32),
        yaw_rate=np.asarray(buf["yaw_rate"], np.float32),
        g_lat=np.asarray(buf["g_lat"], np.float32),
        g_lon=np.asarray(buf["g_lon"], np.float32),
        gear=np.asarray(buf["gear"], np.int8),
        steer=np.asarray(buf["steer"], np.float32),
        throttle=np.asarray(buf["throttle"], np.float32),
        brake=np.asarray(buf["brake"], np.float32),
        lap_distance=np.asarray(buf["lap_distance"], np.float32),
        wheels_off=wheels,
        world_x=np.asarray(buf["world_x"], np.float32),
        world_z=np.asarray(buf["world_z"], np.float32),
        lap_time=np.float32(lap_time),
        control_hz=np.float32(CONTROL_HZ),
    )
    return {"path": path, "lap_time": lap_time, "steps": len(frames),
            "clean": clean, "mb": path.stat().st_size / 1e6}


if __name__ == "__main__":
    main()
