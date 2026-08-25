"""Renders the HUD to an RGBA image.

Kept entirely separate from the window that displays it, for two reasons: the
layout can be regression-tested by writing PNGs to disk with no GUI involved,
and the same renderer can produce frames for a screen recording or for the
README without a live game running.

Everything is drawn on a magenta key colour, which the overlay window makes
transparent. Win32 layered windows can key out exactly one colour, so nothing
in the design may use pure magenta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bus import HudSnapshot

KEY = (255, 0, 255)          # keyed to transparent by the overlay window

BG = (13, 17, 23)
BG_SOFT = (22, 27, 34)
BORDER = (48, 54, 61)
TEXT = (230, 237, 243)
MUTED = (125, 133, 144)
CYAN = (86, 211, 227)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
AMBER = (210, 153, 34)

W = 430
PAD = 12
FRAME_W, FRAME_H = 384, 192
SPARK_H = 62        # the lap-time trace is the headline visual; give it room
AXIS_GUTTER = 40    # left margin inside the plot, reserved for axis labels

_FONTS = Path("C:/Windows/Fonts")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(_FONTS / name), size)
    except OSError:
        return ImageFont.load_default()


F_TINY = _font("consola.ttf", 11)
F_SMALL = _font("consola.ttf", 13)
F_BOLD = _font("consolab.ttf", 13)
F_HEAD = _font("consolab.ttf", 15)
F_TIMER = _font("consolab.ttf", 34)


def _fmt_lap(seconds: float | None) -> str:
    if seconds is None:
        return "--:--.---"
    m, s = divmod(max(0.0, seconds), 60.0)
    return f"{int(m)}:{s:06.3f}"


class HudPanel:
    """Stateless renderer: one snapshot in, one RGBA image out."""

    def __init__(self, width: int = W):
        self.width = width

    def render(self, s: HudSnapshot) -> Image.Image:
        h = self._height(s)
        img = Image.new("RGB", (self.width, h), KEY)
        d = ImageDraw.Draw(img)

        # Card background with a 1px border, inset so the key colour forms a
        # clean edge rather than bleeding into the game behind it.
        d.rounded_rectangle([2, 2, self.width - 3, h - 3], radius=8,
                            fill=BG, outline=BORDER, width=1)

        y = PAD
        y = self._header(d, y, s)
        y = self._vision(img, d, y, s)
        y = self._lap(d, y, s)
        y = self._sparkline(d, y, s)
        y = self._actions(d, y, s)
        if s.mode == "TRAIN":
            y = self._training(d, y, s)
        return img.convert("RGBA")

    # -- layout ------------------------------------------------------------

    def _height(self, s: HudSnapshot) -> int:
        #   header + vision(label+frame) + lap block + sparkline + actions
        h = PAD + 26 + (15 + FRAME_H + 14) + 74 + (14 + SPARK_H + 12) + 78 + PAD
        if s.mode == "TRAIN":
            h += 90     # three stat rows plus the buffer bar
        return h

    def _header(self, d: ImageDraw.ImageDraw, y: int, s: HudSnapshot) -> int:
        d.text((PAD, y), "F1 AUTOPILOT", font=F_HEAD, fill=TEXT)

        colour = {"TRAIN": AMBER, "EVAL": CYAN}.get(s.mode, GREEN)
        label = s.mode
        tw = d.textlength(label, font=F_BOLD)
        x1 = self.width - PAD
        d.rounded_rectangle([x1 - tw - 14, y - 1, x1, y + 17], radius=4,
                            fill=BG_SOFT, outline=colour, width=1)
        d.text((x1 - tw - 7, y + 2), label, font=F_BOLD, fill=colour)
        return y + 26

    def _vision(self, img: Image.Image, d: ImageDraw.ImageDraw, y: int,
                s: HudSnapshot) -> int:
        d.text((PAD, y), "WHAT THE NETWORK SEES", font=F_TINY, fill=MUTED)
        y += 15

        box = [PAD, y, PAD + FRAME_W + 1, y + FRAME_H + 1]
        frame = s.decode_frame()
        if frame is not None:
            vis = Image.fromarray(frame, mode="L").convert("RGB")
            vis = vis.resize((FRAME_W, FRAME_H), Image.NEAREST)
            img.paste(vis, (PAD + 1, y + 1))
        else:
            d.rectangle(box, fill=BG_SOFT)
            d.text((PAD + FRAME_W // 2 - 40, y + FRAME_H // 2 - 6),
                   "no signal", font=F_SMALL, fill=MUTED)
        d.rectangle(box, outline=BORDER, width=1)
        return y + FRAME_H + 14

    def _lap(self, d: ImageDraw.ImageDraw, y: int, s: HudSnapshot) -> int:
        d.text((PAD, y), "CURRENT", font=F_TINY, fill=MUTED)
        d.text((PAD, y + 13), _fmt_lap(s.current_lap_s), font=F_TIMER, fill=TEXT)

        rx = self.width - PAD
        d.text((rx - 118, y), "BEST", font=F_TINY, fill=MUTED)
        d.text((rx - 118, y + 14), _fmt_lap(s.best_lap_s), font=F_BOLD,
               fill=CYAN)
        d.text((rx - 118, y + 32), "LAST", font=F_TINY, fill=MUTED)
        d.text((rx - 118, y + 46), _fmt_lap(s.last_lap_s), font=F_BOLD,
               fill=TEXT)

        # Delta of the most recent lap against the best before it. This is the
        # number that shows improvement, so it gets the colour.
        if s.last_lap_s is not None and s.best_lap_s is not None:
            delta = s.last_lap_s - s.best_lap_s
            if abs(delta) > 1e-6:
                txt = f"{delta:+.3f}"
                col = GREEN if delta < 0 else RED
                d.text((rx - 46, y + 46), txt, font=F_BOLD, fill=col)
        return y + 74

    def _sparkline(self, d: ImageDraw.ImageDraw, y: int,
                   s: HudSnapshot) -> int:
        d.text((PAD, y), f"LAP TIMES  ({len(s.lap_times)} laps)",
               font=F_TINY, fill=MUTED)
        # Net gain goes in the header row, not the plot: on a rising (i.e.
        # improving) trace the top-right corner is exactly where the latest
        # point sits, and the two collide.
        valid = [t for t in s.lap_times if t and t > 0]
        if len(valid) >= 2:
            gain = valid[0] - min(valid)
            if gain > 0.01:
                txt = f"-{gain:.2f}s total"
                tw = d.textlength(txt, font=F_BOLD)
                d.text((self.width - PAD - tw, y - 1), txt, font=F_BOLD,
                       fill=GREEN)
        y += 14
        x0, x1 = PAD, self.width - PAD
        y0, y1 = y, y + SPARK_H
        d.rectangle([x0, y0, x1, y1], fill=BG_SOFT, outline=BORDER, width=1)

        laps = [t for t in s.lap_times if t and t > 0]
        if len(laps) < 2:
            d.text((x0 + 8, y0 + SPARK_H // 2 - 6), "collecting laps...",
                   font=F_TINY, fill=MUTED)
            return y1 + 12

        lo, hi = min(laps), max(laps)
        span = max(1e-6, hi - lo)
        n = len(laps)

        # Reserve a gutter for the axis labels; drawing them over the plot
        # made the first laps unreadable, which is where the biggest gains are.
        px0 = x0 + AXIS_GUTTER
        px1 = x1 - 6
        top, bot = y0 + 6, y1 - 6

        def to_xy(k: int, t: float) -> tuple[float, float]:
            # Faster laps plot higher, so a learning curve reads as a rising
            # line rather than a falling one.
            return (px0 + (px1 - px0) * k / max(1, n - 1),
                    bot - (bot - top) * (hi - t) / span)

        pts = [to_xy(k, t) for k, t in enumerate(laps)]

        # Best-lap reference line, drawn under the trace.
        d.line([(px0, top), (px1, top)], fill=(38, 70, 58), width=1)

        for k in range(1, len(pts)):
            improving = laps[k] <= laps[k - 1]
            d.line([pts[k - 1], pts[k]], fill=GREEN if improving else RED,
                   width=2)
        lx, ly = pts[-1]
        d.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=CYAN)

        d.text((x0 + 5, top - 4), f"{lo:.1f}s", font=F_TINY, fill=MUTED)
        d.text((x0 + 5, bot - 8), f"{hi:.1f}s", font=F_TINY, fill=MUTED)
        return y1 + 12

    def _actions(self, d: ImageDraw.ImageDraw, y: int,
                 s: HudSnapshot) -> int:
        d.text((PAD, y), "POLICY OUTPUT", font=F_TINY, fill=MUTED)
        d.text((self.width - PAD - 96, y),
               f"{s.speed_kph:5.0f} km/h  G{s.gear}", font=F_TINY, fill=TEXT)
        y += 15

        bx0, bx1 = PAD + 54, self.width - PAD
        mid = (bx0 + bx1) / 2

        # Steering is bidirectional, so it fills outward from the centre.
        d.text((PAD, y), "STEER", font=F_TINY, fill=MUTED)
        d.rectangle([bx0, y, bx1, y + 12], fill=BG_SOFT, outline=BORDER)
        sx = mid + (bx1 - mid) * float(np.clip(s.steer, -1, 1))
        d.rectangle([min(mid, sx), y + 1, max(mid, sx), y + 11], fill=CYAN)
        d.line([(mid, y), (mid, y + 12)], fill=BORDER)
        y += 18

        for label, val, col in (("THROTTLE", s.throttle, GREEN),
                                ("BRAKE", s.brake, RED)):
            d.text((PAD, y), label, font=F_TINY, fill=MUTED)
            d.rectangle([bx0, y, bx1, y + 12], fill=BG_SOFT, outline=BORDER)
            w = (bx1 - bx0) * float(np.clip(val, 0, 1))
            if w > 1:
                d.rectangle([bx0 + 1, y + 1, bx0 + w, y + 11], fill=col)
            y += 18
        return y + 8

    def _training(self, d: ImageDraw.ImageDraw, y: int,
                  s: HudSnapshot) -> int:
        d.text((PAD, y), "TRAINING", font=F_TINY, fill=MUTED)
        y += 15

        def cell(cx: int, cy: int, label: str, value: str,
                 col=TEXT) -> None:
            d.text((cx, cy), label, font=F_TINY, fill=MUTED)
            d.text((cx, cy + 13), value, font=F_BOLD, fill=col)

        col_w = (self.width - 2 * PAD) // 3
        cell(PAD, y, "ENV STEPS", f"{s.env_steps:,}")
        cell(PAD + col_w, y, "GRAD STEPS", f"{s.grad_steps:,}")
        cell(PAD + 2 * col_w, y, "EPISODES", f"{s.episodes:,}")
        y += 32

        cell(PAD, y, "CRITIC",
             "--" if s.critic_loss is None else f"{s.critic_loss:.3f}")
        cell(PAD + col_w, y, "ACTOR",
             "--" if s.actor_loss is None else f"{s.actor_loss:.3f}")
        cell(PAD + 2 * col_w, y, "ALPHA",
             "--" if s.alpha is None else f"{s.alpha:.3f}", CYAN)
        y += 30

        # Replay buffer fill.
        d.rectangle([PAD, y, self.width - PAD, y + 6], fill=BG_SOFT,
                    outline=BORDER)
        w = (self.width - 2 * PAD) * float(np.clip(s.buffer_fill, 0, 1))
        if w > 1:
            d.rectangle([PAD + 1, y + 1, PAD + w, y + 5], fill=AMBER)
        return y + 14
