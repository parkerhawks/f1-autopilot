"""Measure the screen-capture half of the control-loop budget.

At 30 Hz the whole loop -- capture, preprocess, infer, write to the pad -- has
33.3 ms. `benchmark_inference.py` showed the network costs 1-1.5 ms, so the
question this answers is what the pixels cost to get hold of in the first
place.

Three separate numbers, because they fail for different reasons:

  grab        DXGI Desktop Duplication handing over a frame
  preprocess  crop, greyscale, resize down to the network's input
  total       what the loop actually pays per step

Run it twice: once on the desktop, once with F1 25 running in borderless
windowed mode. The second is the real number -- Desktop Duplication only
produces a frame when the screen changes, so a static desktop measures the
"nothing happened" path rather than the live one.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGET_H, TARGET_W = 96, 192       # matches f1ai/rl/env.py
BUDGET_MS = 1000.0 / 30.0


def preprocess(frame: np.ndarray, crop_frac: float = 0.55) -> np.ndarray:
    """Full-colour capture -> the single greyscale frame the CNN stacks.

    The crop keeps the middle band of the screen: the sky above and the car's
    own bodywork below carry no steering information, and dropping them raises
    the share of pixels spent on road. INTER_AREA is the right filter for
    heavy downscaling -- INTER_LINEAR aliases the track edges into a shimmer
    that changes with speed, which is exactly the wrong thing to hand a policy.
    """
    h = frame.shape[0]
    top = int(h * 0.28)
    bottom = top + int(h * crop_frac)
    band = frame[top:bottom]
    grey = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def summarise(name: str, times: list[float]) -> float:
    if not times:
        print(f"  {name:<12} no samples")
        return 0.0
    times.sort()
    med = statistics.median(times)
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
    print(f"  {name:<12} median {med:6.2f} ms   p95 {p95:6.2f} ms   "
          f"max {times[-1]:6.2f} ms")
    return med


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--region", default=None,
                    help="left,top,right,bottom to capture a window instead "
                         "of the whole screen")
    ap.add_argument("--save", action="store_true",
                    help="write a sample preprocessed frame to docs/")
    a = ap.parse_args()

    try:
        import dxcam
    except ImportError:
        print("dxcam is not installed:  pip install dxcam")
        sys.exit(1)

    region = None
    if a.region:
        region = tuple(int(v) for v in a.region.split(","))
        if len(region) != 4:
            print("--region needs exactly left,top,right,bottom")
            sys.exit(2)

    cam = dxcam.create(output_idx=0, output_color="BGR")
    if cam is None:
        print("dxcam could not open the display. On laptops with hybrid")
        print("graphics, try output_idx for the other adapter.")
        sys.exit(1)

    probe = cam.grab(region=region)
    for _ in range(40):
        if probe is not None:
            break
        time.sleep(0.05)
        probe = cam.grab(region=region)

    if probe is None:
        print("No frame arrived in 2s. Desktop Duplication only emits a frame")
        print("when the screen CHANGES -- move a window, or run this with the")
        print("game on screen. Falling back to preprocessing-only timings.\n")
        probe = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
        live = False
    else:
        live = True

    print(f"capture source: {probe.shape[1]}x{probe.shape[0]}"
          f"{' (region)' if region else ' (full screen)'}")
    print(f"network input:  {TARGET_W}x{TARGET_H} greyscale")
    print(f"budget at 30 Hz: {BUDGET_MS:.1f} ms total\n")

    grab_times: list[float] = []
    prep_times: list[float] = []
    misses = 0
    last = probe

    deadline = time.perf_counter() + a.seconds
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        frame = cam.grab(region=region) if live else None
        t1 = time.perf_counter()

        if frame is None:
            misses += 1
            frame = last            # reuse: the screen simply did not change
        else:
            last = frame
            grab_times.append((t1 - t0) * 1000.0)

        t2 = time.perf_counter()
        out = preprocess(frame)
        prep_times.append((time.perf_counter() - t2) * 1000.0)

    attempts = len(grab_times) + misses
    print("timings")
    med_grab = summarise("grab", grab_times)
    med_prep = summarise("preprocess", prep_times)
    total = med_grab + med_prep

    print(f"\n  new frames    {len(grab_times)} of {attempts} attempts"
          f"  ({100 * len(grab_times) / max(1, attempts):.0f}%)")
    if not live:
        print("  (grab not measured -- no live frames; preprocess is real)")

    print(f"\n  capture+preprocess {total:6.2f} ms  "
          f"= {100 * total / BUDGET_MS:.0f}% of the 30 Hz budget")
    print(f"  inference (measured separately)  ~1.5 ms   = ~4%")
    headroom = BUDGET_MS - total - 1.5
    print(f"  headroom           {headroom:6.2f} ms")

    if headroom < 5:
        print("\n  TIGHT. Capture a window region instead of the full screen,")
        print("  or drop the game to 1080p.")
    elif live:
        print("\n  Comfortable for a 30 Hz control loop.")

    if a.save:
        out_path = Path(__file__).resolve().parent.parent / "docs" / "capture_sample.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vis = cv2.resize(out, (TARGET_W * 3, TARGET_H * 3),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_path), vis)
        print(f"\nwrote {out_path}")
        print("Check the crop actually contains road, not sky or bodywork.")

    del cam


if __name__ == "__main__":
    main()
