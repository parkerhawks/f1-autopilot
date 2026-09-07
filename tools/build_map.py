"""Build a curvature map of the circuit from recorded laps, and plot it.

    python tools/build_map.py demos/laps-20260817-151654

Writes maps/<track>.npz plus a JSON summary and a plot. Look at the plot: the
curvature trace should show recognisable corners with straights between them,
and the speed profile should dip exactly where curvature peaks. If they do not
line up, lapDistance is mis-calibrated and every downstream feature is wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.track_map import build_from_laps  # noqa: E402

MAPS = Path(__file__).resolve().parent.parent / "maps"
BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
CYAN, GREEN, AMBER = "#56d3e3", "#3fb950", "#d29922"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", help="demos/ directory or .npz files")
    ap.add_argument("--name", default="track",
                    help="track name; the map is written to maps/<name>.npz")
    ap.add_argument("--bin-metres", type=float, default=5.0)
    ap.add_argument("--half-width", type=float, default=10.0,
                    help="metres from the racing line still counted as on "
                         "track; raise it if legal lines are being penalised")
    a = ap.parse_args()

    path = Path(a.session)
    files = [path] if path.suffix == ".npz" else sorted(path.glob("*.npz"))
    if not files:
        raise SystemExit(f"no recordings in {path}")

    print(f"building from {len(files)} lap(s)")
    tmap = build_from_laps(files, bin_metres=a.bin_metres)
    tmap.half_width = a.half_width

    MAPS.mkdir(parents=True, exist_ok=True)
    out = MAPS / f"{a.name}.npz"
    tmap.save(out)

    radius = 1.0 / np.maximum(np.abs(tmap.curvature), 1e-6)
    print(f"\ntrack length   {tmap.length:.0f} m")
    print(f"tightest corner {radius.min():.0f} m radius")
    print(f"speed profile   {tmap.ref_speed.min() * 3.6:.0f} - "
          f"{tmap.ref_speed.max() * 3.6:.0f} km/h")

    if tmap.has_geometry:
        # How far the demonstrations themselves stray from the averaged line.
        # If this approaches half_width, legal driving will be penalised and
        # the threshold needs raising.
        import numpy as _np
        spread = []
        for p in files:
            with _np.load(p) as z:
                if "world_x" not in z.files:
                    continue
                wx, wz, ld = z["world_x"], z["world_z"], z["lap_distance"]
            # Measure the SAME points the map was built from. Including the
            # out-lap here reported 35.7 m of deviation on laps that were
            # actually within 7 m of each other -- the pit lane really is that
            # far from the racing line, and it was never part of the map.
            keep = ld >= 0.0
            wx, wz, ld = wx[keep], wz[keep], ld[keep]
            spread += [tmap.lateral_offset(float(wx[i]), float(wz[i]),
                                           float(ld[i]))
                       for i in range(0, len(wx), 25)]
        if spread:
            arr = _np.asarray(spread)
            print(f"racing line     demonstrations sit {arr.mean():.1f} m mean, "
                  f"{arr.max():.1f} m max from it")
            print(f"half-width      {tmap.half_width:.1f} m "
                  f"({tmap.half_width - arr.max():.1f} m of headroom)")
            if arr.max() > tmap.half_width * 0.85:
                print("  WARNING: little headroom -- legal lines may be")
                print("  penalised. Raise --half-width.")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), facecolor=BG, sharex=True)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
        ax.tick_params(colors=FG, labelsize=9)
        for s in ax.spines.values():
            s.set_color(GRID)

    axes[0].plot(tmap.s, tmap.curvature * 1000.0, color=CYAN, linewidth=1.6)
    axes[0].axhline(0, color=GRID, linewidth=1)
    axes[0].set_ylabel("curvature (1/km)", color=FG, fontsize=10)
    axes[0].set_title(f"{a.name}: curvature and demonstrated speed",
                      color=FG, fontsize=12, loc="left")

    axes[1].plot(tmap.s, tmap.ref_speed * 3.6, color=GREEN, linewidth=1.6)
    axes[1].set_ylabel("speed (km/h)", color=FG, fontsize=10)
    axes[1].set_xlabel("lap distance (m)", color=FG, fontsize=10)

    # Mark the tightest corners; these should coincide with the speed minima.
    peaks = np.argsort(-np.abs(tmap.curvature))[:6]
    for p in peaks:
        for ax in axes:
            ax.axvline(tmap.s[p], color=AMBER, alpha=0.35, linewidth=1)

    fig.tight_layout()
    png = MAPS / f"{a.name}.png"
    fig.savefig(png, dpi=110, facecolor=BG)

    print(f"\nwrote {out}")
    print(f"wrote {png}")
    print("\nCHECK THE PLOT: speed minima must line up with curvature peaks")
    print("(the amber lines). If they do not, lapDistance is mis-calibrated")
    print("and every map feature derived from it will be wrong.")


if __name__ == "__main__":
    main()
