"""Show the full game capture with the crop box drawn on it, so the framing
that reaches the CNN can be checked and adjusted by eye.

    python tools/tune_crop.py
    python tools/tune_crop.py --crop-top 0.34 --crop-frac 0.42

Writes docs/crop_tuning.png: the full captured frame with the crop outlined,
and beside it exactly what the network receives.

WHAT GOOD LOOKS LIKE
  * road filling most of the crop, with both track edges visible
  * the vanishing point inside the crop, not above it
  * no steering wheel, halo, mirrors or HUD
  * nothing from outside the game window

Camera choice matters more than the numbers. A cockpit view spends much of the
frame on the halo and the wheel; a chase or T-cam view shows far more road, and
the road is the only thing here worth any pixels.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.control.window import describe, find_game_region, list_windows  # noqa: E402
from f1ai.rl.env import FRAME_H, FRAME_W  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "crop_tuning.png"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop-top", type=float, default=0.28)
    ap.add_argument("--crop-frac", type=float, default=0.55)
    ap.add_argument("--region", default=None)
    ap.add_argument("--full-screen", action="store_true",
                    help="ignore window detection and grab everything")
    ap.add_argument("--list-windows", action="store_true")
    ap.add_argument("--samples", type=int, default=1,
                    help="capture this many frames spread over --window "
                         "seconds and show them as a grid; one frame is a "
                         "poor basis for judging a crop that has to work "
                         "everywhere on the lap")
    ap.add_argument("--window", type=float, default=12.0,
                    help="seconds over which to spread the samples")
    ap.add_argument("--colour", action="store_true",
                    help="preview what a colour input would look like; at "
                         "Monza the track edge is a white line on grey "
                         "tarmac, which greyscale can flatten away")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to wait before capturing, so you can tab "
                         "back to the game and DRIVE -- a paused frame is "
                         "mostly menu and tells you nothing about the crop")
    a = ap.parse_args()

    if a.list_windows:
        print("visible windows:")
        for t in list_windows():
            print(f"  {t}")
        return

    from f1ai.control.capture import ScreenCapture

    print(describe())
    region = None
    if a.region:
        region = tuple(int(v) for v in a.region.split(","))
    elif not a.full_screen:
        region = find_game_region()
        if region is None:
            print("no F1 window found -- falling back to full screen.")
            print("Run with --list-windows to see what is open.")

    cam = ScreenCapture(region=region)
    if a.delay > 0:
        print(f"\ncapturing in {a.delay:.0f}s -- tab to the game and drive")
        for remaining in range(int(a.delay), 0, -1):
            print(f"  {remaining}...", end="\r", flush=True)
            time.sleep(1.0)
        print("  capturing now ")

    frame = None
    for _ in range(60):
        frame = cam.grab()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        print("\nNo frame captured. Desktop Duplication only produces a frame")
        print("when the screen changes -- make sure the game is running and")
        print("visible, not minimised.")
        sys.exit(1)

    h, w = frame.shape[:2]
    top = int(h * a.crop_top)
    bottom = min(top + int(h * a.crop_frac), h)

    def to_net(src: np.ndarray) -> np.ndarray:
        band = src[top:bottom]
        if a.colour:
            return cv2.resize(band, (FRAME_W, FRAME_H),
                              interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        return cv2.resize(grey, (FRAME_W, FRAME_H),
                          interpolation=cv2.INTER_AREA)

    net_input = to_net(frame)

    # Gather more samples across the lap, so the crop is judged on a corner
    # and a straight rather than on whichever instant we happened to catch.
    extra = []
    if a.samples > 1:
        interval = a.window / a.samples
        print(f"sampling {a.samples} frames over {a.window:.0f}s -- keep driving")
        for i in range(a.samples - 1):
            end = time.perf_counter() + interval
            while time.perf_counter() < end:
                got = cam.grab()
                if got is not None:
                    frame = got
                time.sleep(0.01)
            extra.append(to_net(frame))
            print(f"  {i + 2}/{a.samples}", end="\r", flush=True)
        print()

    # Left: the full capture with the crop outlined.
    annotated = frame.copy()
    cv2.rectangle(annotated, (0, top), (w - 1, bottom - 1), (0, 255, 255), 3)
    cv2.putText(annotated, f"crop_top={a.crop_top:.2f}", (12, top - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated, f"crop_frac={a.crop_frac:.2f}", (12, bottom + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    scale = 720 / max(1, h)
    left_panel = cv2.resize(annotated, (int(w * scale), 720))

    def upscale(img: np.ndarray) -> np.ndarray:
        big = cv2.resize(img, (FRAME_W * 3, FRAME_H * 3),
                         interpolation=cv2.INTER_NEAREST)
        return big if big.ndim == 3 else cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    tiles = [upscale(net_input)] + [upscale(e) for e in extra]
    label = "what the CNN sees" + (" (COLOUR preview)" if a.colour else "")
    cv2.putText(tiles[0], label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1)
    right = np.vstack([np.vstack([t, np.zeros((8, t.shape[1], 3), np.uint8)])
                       for t in tiles])[:-8]

    height = max(720, right.shape[0])
    canvas = np.zeros((height, left_panel.shape[1] + right.shape[1] + 24, 3),
                      np.uint8)
    canvas[:left_panel.shape[0], :left_panel.shape[1]] = left_panel
    y = (height - right.shape[0]) // 2
    canvas[y:y + right.shape[0], left_panel.shape[1] + 24:] = right

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), canvas)
    cam.close()

    print(f"\ncaptured {w}x{h}"
          f"{' (game window)' if region else ' (FULL SCREEN)'}")
    print(f"crop rows {top}-{bottom} -> {FRAME_W}x{FRAME_H} greyscale")
    print(f"detail    {(bottom - top) / FRAME_H:.1f} source rows per output row")
    print(f"\nwrote {OUT}")
    print("\nOpen it. The crop must contain road and track edges -- not the")
    print("wheel, not the halo, not the HUD, and nothing outside the game.")


if __name__ == "__main__":
    main()
