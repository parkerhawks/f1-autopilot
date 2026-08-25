"""Pre-flight check for F1Backend: capture + telemetry + pad, all at once.

Every piece has been verified in isolation. Nothing has yet run them together
against the live game, and that is where the interesting failures live -- a
crop that contains sky instead of road, a control loop that cannot hold 30 Hz
with the game rendering, telemetry that stalls the moment capture starts.

Finding any of those forty minutes into a training run wastes the evening.
This takes ninety seconds.

    python tools/probe_backend.py              # observe only, safe
    python tools/probe_backend.py --drive      # also applies throttle/steering

Run it in Time Trial with the car ON TRACK. With --drive, the game must have
focus, and the car will move.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import CONTROL_HZ, FRAME_H, FRAME_W  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "f1_capture_sample.png"
BUDGET_MS = 1000.0 / CONTROL_HZ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--drive", action="store_true",
                    help="apply throttle and steering (game must have focus)")
    ap.add_argument("--region", default=None,
                    help="left,top,right,bottom to capture a window region")
    a = ap.parse_args()

    from f1ai.rl.f1_backend import F1Backend

    region = tuple(int(v) for v in a.region.split(",")) if a.region else None

    print("opening capture + telemetry + virtual pad...")
    try:
        backend = F1Backend(capture_region=region)
    except RuntimeError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
    print("all three opened.\n")

    checks: list[tuple[str, bool, str]] = []
    obs_times: list[float] = []
    speeds: list[float] = []
    lap_d: list[float] = []
    wheels: list[int] = []
    invalid: list[bool] = []
    frames_seen = 0
    last_frame = None

    if a.drive:
        print("DRIVING for a few seconds -- focus the game now. 5s...")
        time.sleep(5)

    t0 = time.perf_counter()
    step = 0
    while time.perf_counter() - t0 < a.seconds:
        loop_start = time.perf_counter()

        if a.drive:
            # A gentle sine on the steering, light throttle. Enough to prove
            # the loop closes without putting the car in a wall.
            t = time.perf_counter() - t0
            backend.apply(0.25 * np.sin(t * 1.2), 0.35, 0.0)

        obs = backend.observe()
        obs_times.append((time.perf_counter() - loop_start) * 1000.0)

        speeds.append(obs.speed_mps * 3.6)
        lap_d.append(obs.lap_distance)
        wheels.append(obs.wheels_off)
        invalid.append(obs.lap_invalid)
        # Live line so the lap-invalid flag can be verified by deliberately
        # running wide and watching it flip -- its byte offset is inferred,
        # not measured, and an overnight run depends on it being right.
        if step % 10 == 0:
            print(f"  {obs.speed_mps * 3.6:5.0f} km/h  "
                  f"lapDist {obs.lap_distance:7.1f}  "
                  f"wheels_off {obs.wheels_off}  "
                  f"lap_invalid {'YES' if obs.lap_invalid else 'no '}  "
                  f"pit {obs.pit_status}  "
                  f"driver {backend.driver_status()}",
                  end="\r", flush=True)
        if last_frame is None or not np.array_equal(obs.frame, last_frame):
            frames_seen += 1
            last_frame = obs.frame.copy()

        step += 1
        slack = BUDGET_MS / 1000.0 - (time.perf_counter() - loop_start)
        if slack > 0:
            time.sleep(slack)

    if a.drive:
        backend.apply(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    med_obs = statistics.median(obs_times)
    p95_obs = sorted(obs_times)[int(len(obs_times) * 0.95)]

    print(f"{step} steps in {a.seconds:.0f}s\n")
    print("timing")
    print(f"  observe()      median {med_obs:5.2f} ms   p95 {p95_obs:5.2f} ms")
    print(f"  budget at {CONTROL_HZ:g} Hz  {BUDGET_MS:.1f} ms  "
          f"({100 * med_obs / BUDGET_MS:.0f}% used)")

    print("\ntelemetry")
    print(f"  speed          {min(speeds):.0f} - {max(speeds):.0f} km/h")
    print(f"  lapDistance    {min(lap_d):.1f} - {max(lap_d):.1f} m")
    print(f"  wheels off     {max(wheels)} max, "
          f"{100 * sum(1 for w in wheels if w) / len(wheels):.0f}% of steps")
    n_inv = sum(1 for v in invalid if v)
    print(f"  lap invalid    {'YES for ' + str(n_inv) + ' steps' if n_inv else 'never'}")
    if max(wheels) >= 2 and n_inv == 0:
        print("    ^ you had 2+ wheels off but the flag never set --")
        print("      LAP_INVALID_OFFSET in packets.py is probably wrong.")
    print(f"  new frames     {frames_seen} of {step}")

    checks.append(("observe() fits the control budget", med_obs < BUDGET_MS * 0.6,
                   f"{med_obs:.1f} ms of {BUDGET_MS:.1f} ms"))
    checks.append(("capture produces new frames", frames_seen > step * 0.5,
                   f"{frames_seen}/{step} distinct"))
    checks.append(("frame is not blank",
                   last_frame is not None and int(last_frame.std()) > 5,
                   f"std {last_frame.std():.1f}" if last_frame is not None else ""))
    checks.append(("lapDistance is calibrated and moving",
                   max(lap_d) - min(lap_d) > 1.0,
                   f"moved {max(lap_d) - min(lap_d):.1f} m"))
    checks.append(("telemetry did not stall", backend.stale_seconds() < 2.0,
                   f"{backend.stale_seconds():.1f}s stale"))
    if a.drive:
        checks.append(("the car actually moved", max(speeds) > 20.0,
                       f"reached {max(speeds):.0f} km/h"))

    # Save the exact frame the network would receive.
    if last_frame is not None:
        import cv2
        OUT.parent.mkdir(parents=True, exist_ok=True)
        vis = cv2.resize(last_frame, (FRAME_W * 3, FRAME_H * 3),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT), vis)

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<36} {detail}")
        failed += not ok

    backend.close()

    print(f"\nwrote {OUT}")
    print("LOOK AT THAT IMAGE before training. It is exactly what the CNN")
    print("receives. It must show road with visible edges -- not sky, not")
    print("bodywork, not HUD. Adjust crop_top / crop_frac in f1_backend.py")
    print("if the framing is wrong; no amount of training fixes a bad crop.")

    if failed:
        print(f"\n{failed}/{len(checks)} FAILED -- do not start a training run")
        sys.exit(1)
    print(f"\nall {len(checks)} checks pass -- the backend is ready")


if __name__ == "__main__":
    main()
