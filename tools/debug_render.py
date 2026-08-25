"""Print the projected geometry for one frame, to find where it goes wrong."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.sim.render import TrackRenderer  # noqa: E402
from f1ai.sim.track import make_track  # noqa: E402

track = make_track()
r = TrackRenderer(track)

i = track.index_at_distance(0.0)
cx_, cz_, yaw = float(track.x[i]), float(track.z[i]), track.heading_at(i)
print(f"car at ({cx_:.1f}, {cz_:.1f}) yaw={yaw:.3f}")
print(f"frame {r.width}x{r.height}, focal={r.focal:.1f}, "
      f"cx={r.cx:.1f}, cy={r.cy:.1f}\n")

n = len(track.x)
spacing = track.length / n
span = int(220.0 / spacing) + 2
idx = np.arange(i, i + span) % n
print(f"walking {span} centreline points ({spacing:.2f} m apart)\n")

left = r._project_many(r._edges_l[idx], cx_, cz_, yaw)
right = r._project_many(r._edges_r[idx], cx_, cz_, yaw)

ok = left[:, 2].astype(bool) & right[:, 2].astype(bool)
print(f"visible points: {ok.sum()} / {len(ok)}")
run = r._longest_run(ok)
print(f"longest run: {run}\n")

# Depth of each sample, to confirm ordering along the ray.
dx = r._edges_l[idx][:, 0] - cx_
dz = r._edges_l[idx][:, 1] - cz_
import math  # noqa: E402
s, c = math.sin(yaw), math.cos(yaw)
depth = dx * s + dz * c

print(f"{'k':>5} {'depth':>9} {'uL':>9} {'vL':>8} {'uR':>9} {'vR':>8} {'vis':>4}")
print("-" * 56)
for k in list(range(0, 12)) + list(range(60, 70)) + [span - 3, span - 2]:
    if k >= len(idx):
        continue
    print(f"{k:>5} {depth[k]:>9.1f} {left[k,0]:>9.1f} {left[k,1]:>8.1f} "
          f"{right[k,0]:>9.1f} {right[k,1]:>8.1f} {int(ok[k]):>4}")

if run:
    a, b = run
    print(f"\npolygon spans k={a}..{b}")
    print(f"  near end: uL={left[a,0]:.1f} uR={right[a,0]:.1f} "
          f"v={left[a,1]:.1f}")
    print(f"  far end:  uL={left[b-1,0]:.1f} uR={right[b-1,0]:.1f} "
          f"v={left[b-1,1]:.1f}")
    print(f"  depth at near={depth[a]:.1f}m  far={depth[b-1]:.1f}m")
