"""Prove the racing-line lookup survives an untrustworthy lapDistance.

The failure this catches killed a 221k-step run in three hours.

`lateral_offset` searches a window around the reported lap distance, which is
right -- Monza passes close to itself, and a global nearest-point search will
match the far side of the circuit and report a tidy offset for a car that is
nowhere near. But the window is only as good as the value it is centred on, and
after a Restart Lap `lapDistance` can be NEGATIVE: the car is on an out-lap,
before the start line. `-500 % 5797` wraps to 5297 m, so the window lands half
a kilometre away, every position reads as wildly off-line, and the episode is
terminated within twenty steps of every reset. A reset loop, from a value that
was never wrong -- only misinterpreted.

Checked here against the real Monza map:

  * a point ON the line reads near zero however its lap distance is reported
  * a genuinely displaced point still reads displaced
  * a negative lap distance does not manufacture a false off-track verdict
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.track_map import TrackMap, find_any_map  # noqa: E402


def main() -> None:
    # Any built map will do -- nothing here is specific to a circuit.
    mp = find_any_map()
    if mp is None:
        print("no map in maps/ -- build one with tools/build_map.py")
        print("(this suite needs a real track map and is skipped without one)")
        sys.exit(0)
    m = TrackMap.load(mp)
    print(f"using {mp.name} ({m.length:.0f} m)")
    if not m.has_geometry:
        print("map has no racing line -- rebuild from recordings with world "
              "position")
        sys.exit(1)

    checks: list[tuple[str, bool, str]] = []
    n = len(m.s)

    # Pick a STRAIGHT, not a chicane. At a chicane the two legs sit close
    # together in space, so a displaced point can legitimately be near the
    # returning track and no distance metric can separate them -- that is a
    # property of the circuit, not a bug, and testing there measures nothing.
    straight = int(np.argmin(np.abs(m.curvature)))
    px, pz, ps = float(m.x[straight]), float(m.z[straight]), float(m.s[straight])
    print(f"testing at lapDistance {ps:.0f} m "
          f"(curvature {m.curvature[straight]:.5f}, a straight)")

    honest = m.lateral_offset(px, pz, ps)
    checks.append(("a point on the line reads ~0 with a correct lap distance",
                   honest < 1.0, f"{honest:.2f} m"))

    # The same point, with lapDistance lying in the ways it actually lies.
    for label, bad_s in (("negative (out-lap)", -500.0),
                         ("wrapped past the line", m.length + 300.0),
                         ("stale by 2 km", ps + 2000.0)):
        off = m.lateral_offset(px, pz, bad_s)
        checks.append((f"survives lapDistance {label}", off < 2.0,
                       f"{off:.2f} m (would have been hundreds)"))

    # A genuinely displaced car must still be reported as displaced, or the
    # global fallback has simply disabled enforcement.
    for d in (15.0, 30.0, 60.0):
        off = m.lateral_offset(px + d, pz, ps)
        ok = abs(off - d) < max(4.0, d * 0.25)
        checks.append((f"a real {d:.0f} m displacement is still detected", ok,
                       f"reported {off:.1f} m"))

    # And every demonstration point must stay well inside the threshold.
    laps = sorted(Path("demos").glob("laps-*/*.npz"))
    if laps:
        with np.load(laps[-1]) as z:
            has_world = "world_x" in z.files
            if has_world:
                wx, wz, ld = z["world_x"], z["world_z"], z["lap_distance"]
        if has_world:
            offs = np.array([
                m.lateral_offset(float(wx[k]), float(wz[k]), float(ld[k]))
                for k in range(0, len(wx), 15)])
            checks.append(("demonstrations stay inside half_width",
                           offs.max() < m.half_width,
                           f"max {offs.max():.1f} m vs {m.half_width:.0f} m"))

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")
        failed += not ok
    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- resets would loop again")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- the lookup tolerates a bad "
          f"lapDistance")


if __name__ == "__main__":
    main()
