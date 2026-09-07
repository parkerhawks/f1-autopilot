"""Compose the README's hero image from a recorded lap.

    python tools/make_hero.py demos/<session>/<lap>.npz
    python tools/make_hero.py demos/<session>/<lap>.npz --live

Three panels, because three things have to be believed at once: that the
network really is driving from pictures, that the telemetry is real, and that
the map is a genuine circuit rather than a drawing.

Everything is rendered from an actual recording -- the frames are the exact
arrays the CNN was fed, and the traces are the bytes the game sent. Nothing
here is illustrative.

With `--live` and the game running, it also captures a full-resolution
screenshot and marks the crop the network actually receives, which is the
clearest way to show how little of the screen the policy sees.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.track_map import TrackMap, find_any_map  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "hero.png"

BG = "#0c0f15"
PANEL = "#131720"
LINE = "#252b37"
INK = "#e3e7ec"
MUTED = "#8b93a2"
ACCENT = "#a87dff"
GREEN = "#42c079"
RED = "#e9584c"
AMBER = "#d6a03c"


def _style(ax, title: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=LINE, linewidth=0.6, alpha=0.6)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lap", help="a recorded .npz lap")
    ap.add_argument("--live", action="store_true",
                    help="also capture the game window, if it is running")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with np.load(Path(a.lap)) as z:
        d = {k: z[k] for k in z.files}

    frames = d["frame"]
    dist = d["lap_distance"].astype(float)
    speed = d["speed"].astype(float) * 3.6
    thr = d["throttle"].astype(float)
    brk = d["brake"].astype(float)
    steer = d["steer"].astype(float)
    lap_time = float(d.get("lap_time", 0.0))

    tmap = None
    mp = find_any_map()
    if mp is not None:
        tmap = TrackMap.load(mp)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    # Top margin has to clear BOTH the title block and the first subplot's own
    # title, or the two overlap; and the right margin has to leave room for the
    # map panel's title, which is the longest on the figure.
    gs = gridspec.GridSpec(
        3, 2, figure=fig, width_ratios=[1.55, 1.0],
        height_ratios=[0.72, 1.0, 1.0], hspace=0.45, wspace=0.14,
        left=0.045, right=0.955, top=0.855, bottom=0.07)

    fig.text(0.045, 0.955, "f1-autopilot", color=INK, fontsize=24)
    sub = "one recorded lap: what the network sees, what it reads, what it maps"
    if lap_time:
        sub = f"{sub}   ·   {lap_time:.2f}s"
    fig.text(0.045, 0.918, sub, color=MUTED, fontsize=11)

    # -- what the network sees ------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    picks = [int(len(frames) * f) for f in (0.04, 0.20, 0.37, 0.54, 0.71, 0.88)]
    strip = np.hstack([np.hstack([frames[i],
                                  np.full((frames.shape[1], 6), 20, np.uint8)])
                       for i in picks])[:, :-6]
    ax.imshow(strip, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.set_title(f"what the network sees  ·  {frames.shape[2]}x{frames.shape[1]}"
                 f" greyscale, 3 stacked  ·  no HUD, no minimap, no track name",
                 color=INK, fontsize=11, loc="left", pad=8)
    for k, i in enumerate(picks):
        ax.text((k + 0.5) * strip.shape[1] / len(picks), frames.shape[1] - 6,
                f"{dist[i]:.0f} m", color=AMBER, fontsize=8.5, ha="center")

    # -- telemetry: speed ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(dist, speed, color=ACCENT, linewidth=1.5)
    ax.fill_between(dist, 0, speed, color=ACCENT, alpha=0.12)
    _style(ax, "telemetry the policy reads  ·  speed", "km/h")
    ax.set_xlim(dist.min(), dist.max())
    ax.set_ylim(0, speed.max() * 1.12)
    ax.text(0.985, 0.88, f"peak {speed.max():.0f} km/h", color=ACCENT,
            fontsize=9, ha="right", transform=ax.transAxes)

    # -- telemetry: pedals and steering ----------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.fill_between(dist, 0, thr, color=GREEN, alpha=0.55, label="throttle")
    ax.fill_between(dist, 0, -brk, color=RED, alpha=0.55, label="brake")
    ax.plot(dist, steer, color=INK, linewidth=0.9, alpha=0.85, label="steering")
    ax.axhline(0, color=LINE, linewidth=1)
    _style(ax, "throttle, brake, steering  ·  the labels for imitation learning",
           "applied")
    ax.set_xlim(dist.min(), dist.max())
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("lap distance (m)", color=MUTED, fontsize=9)
    leg = ax.legend(loc="lower right", fontsize=8, facecolor=PANEL,
                    edgecolor=LINE, labelcolor=INK, ncol=3)
    leg.get_frame().set_alpha(0.9)

    # -- the map it built -------------------------------------------------
    ax = fig.add_subplot(gs[1:, 1])
    ax.set_facecolor(PANEL)
    if tmap is not None and tmap.has_geometry:
        pts = np.column_stack([tmap.x, tmap.z]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        v = tmap.ref_speed * 3.6
        lc = LineCollection(segs, cmap="viridis", linewidth=3.4)
        lc.set_array(v[:-1])
        ax.add_collection(lc)
        ax.autoscale_view()
        cb = fig.colorbar(lc, ax=ax, fraction=0.036, pad=0.02)
        cb.set_label("km/h", color=MUTED, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(LINE)
        slow = int(np.argmin(tmap.ref_speed))
        ax.plot(tmap.x[slow], tmap.z[slow], "o", color=RED, markersize=7)
        ax.annotate(f"slowest {v[slow]:.0f} km/h",
                    (tmap.x[slow], tmap.z[slow]),
                    textcoords="offset points", xytext=(10, 8),
                    color=RED, fontsize=9)
        title = (f"the map it built from your laps  ·  "
                 f"{tmap.length / 1000:.2f} km")
    else:
        ax.text(0.5, 0.5, "no map with geometry\n(build one with build_map.py)",
                color=MUTED, ha="center", va="center", fontsize=11,
                transform=ax.transAxes)
        title = "the racing line it built"
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)

    out = Path(a.out) if a.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=BG)
    print(f"wrote {out}")

    if a.live:
        _capture_live(out.with_name("hero_capture.png"))


def _capture_live(out: Path) -> None:
    """Full-resolution game window, with the network's crop marked on it."""
    import cv2

    from f1ai.control.capture import ScreenCapture
    from f1ai.control.window import find_game_region
    from f1ai.rl.env import FRAME_H, FRAME_W

    region = find_game_region()
    if region is None:
        print("no game window found -- skipping the live capture")
        return

    cam = ScreenCapture(region=region)
    frame = None
    for _ in range(60):
        frame = cam.grab()
        if frame is not None:
            break
    cam.close()
    if frame is None:
        print("no frame captured")
        return

    from f1ai.rl.f1_backend import F1Backend
    h = frame.shape[0]
    top = int(h * F1Backend.__init__.__defaults__[2])   # crop_top default
    band = int(h * F1Backend.__init__.__defaults__[1])  # crop_frac default
    cv2.rectangle(frame, (0, top), (frame.shape[1] - 1, top + band),
                  (120, 255, 255), 3)
    cv2.putText(frame, "the only part the network sees", (18, top - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 255), 2)
    cv2.imwrite(str(out), frame)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
