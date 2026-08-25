"""Render sample frames from the mock game's camera into a contact sheet.

A vision pipeline that is never looked at is a vision pipeline that silently
trains on garbage. This dumps frames from a lap -- straights, both corner
directions, and off-track positions -- so the input can be eyeballed before any
training run consumes hours on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.sim.render import TrackRenderer  # noqa: E402
from f1ai.sim.track import make_track  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "render_preview.png"
COLS, SCALE = 4, 3


def main() -> None:
    track = make_track()
    r = TrackRenderer(track)

    shots = []
    # Eight points spread around the lap, on the centreline and facing along it.
    for frac in np.linspace(0.0, 1.0, 8, endpoint=False):
        i = track.index_at_distance(frac * track.length)
        shots.append((f"s={frac * track.length:.0f}m",
                      float(track.x[i]), float(track.z[i]),
                      track.heading_at(i)))

    # Plus deliberately bad states the policy must learn to recover from.
    i = track.index_at_distance(track.length * 0.12)
    h = track.heading_at(i)
    shots.append(("offset +5m", float(track.x[i]) + 5.0, float(track.z[i]), h))
    shots.append(("yaw +25deg", float(track.x[i]), float(track.z[i]), h + 0.44))
    shots.append(("yaw -25deg", float(track.x[i]), float(track.z[i]), h - 0.44))
    shots.append(("off track", float(track.x[i]) * 1.12,
                  float(track.z[i]) * 1.12, h))

    tiles = []
    for label, x, z, yaw in shots:
        img = r.render(x, z, yaw)
        big = cv2.resize(img, (img.shape[1] * SCALE, img.shape[0] * SCALE),
                         interpolation=cv2.INTER_NEAREST)
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.putText(big, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1),
                      (60, 60, 60), 1)
        tiles.append(big)

    rows = []
    for k in range(0, len(tiles), COLS):
        row = tiles[k:k + COLS]
        while len(row) < COLS:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), sheet)

    from f1ai.sim.render import COL_GRASS, COL_ROAD, COL_SKY  # noqa: E402

    stats = r.render(*shots[0][1:])
    print(f"frame shape {stats.shape}, dtype {stats.dtype}")
    # Match the palette exactly: sky and road are both dark, so a naive
    # brightness threshold counts the sky as road.
    for name, col in (("sky", COL_SKY), ("road", COL_ROAD), ("grass", COL_GRASS)):
        frac = 100.0 * (stats == col).mean()
        print(f"  {name:<6} {frac:>5.1f}%")

    # How many image rows the road actually spans is the number that decides
    # whether upcoming curvature is visible at all.
    rows = np.where((stats == COL_ROAD).any(axis=1))[0]
    if len(rows):
        print(f"  road spans rows {rows.min()}-{rows.max()} "
              f"({len(rows)} of {stats.shape[0]})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
