"""Render the HUD to a PNG with synthetic data, so the layout can be checked.

The overlay is the thing that demonstrates this project to anyone who is not
going to read the code, so its layout gets tested like anything else -- without
needing a game, a GPU or a trained policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.hud.bus import HudSnapshot  # noqa: E402
from f1ai.hud.panel import HudPanel  # noqa: E402
from f1ai.sim.render import TrackRenderer  # noqa: E402
from f1ai.sim.track import make_track  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "hud_preview.png"


def learning_curve(n: int = 26) -> list[float]:
    """A plausible SAC lap-time trace: fast early gains, then a noisy plateau."""
    rng = np.random.default_rng(7)
    out = []
    for k in range(n):
        base = 34.0 + 22.0 * np.exp(-k / 6.0)
        out.append(float(base + rng.normal(0, 0.55)))
    return out


def main() -> None:
    track = make_track()
    r = TrackRenderer(track, 192, 96)
    i = track.index_at_distance(track.length * 0.11)
    frame = r.render(float(track.x[i]), float(track.z[i]), track.heading_at(i))

    laps = learning_curve()
    panel = HudPanel()

    # Two states: mid-training, and a clean evaluation run.
    train = HudSnapshot(
        mode="TRAIN",
        speed_kph=248.6, gear=6,
        steer=-0.34, throttle=0.82, brake=0.0,
        current_lap_s=21.418, last_lap_s=laps[-1], best_lap_s=min(laps),
        lap_times=laps,
        step_reward=2.41, episode_return=5820.0,
        env_steps=412_800, grad_steps=396_100, episodes=1_284,
        buffer_fill=0.83, actor_loss=-14.203, critic_loss=0.847, alpha=0.118,
    )
    train.attach_frame(frame)

    ev = HudSnapshot(
        mode="EVAL",
        speed_kph=312.0, gear=8,
        steer=0.06, throttle=1.0, brake=0.0,
        current_lap_s=8.902, last_lap_s=min(laps), best_lap_s=min(laps),
        lap_times=laps[-8:],
        step_reward=2.9, episode_return=1204.0,
    )
    ev.attach_frame(frame)

    a, b = panel.render(train), panel.render(ev)
    gap = 16
    sheet = Image.new("RGB", (a.width + b.width + gap * 3,
                              max(a.height, b.height) + gap * 2), (55, 58, 64))
    sheet.paste(a, (gap, gap), a)
    sheet.paste(b, (a.width + gap * 2, gap), b)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)

    print(f"panel size: {a.width}x{a.height} (TRAIN), {b.width}x{b.height} (EVAL)")
    print(f"snapshot payload with frame: "
          f"{len(str(train.frame_png)) / 1024:.1f} KB base64")
    print(f"lap curve: {laps[0]:.2f}s -> {min(laps):.2f}s over {len(laps)} laps")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
