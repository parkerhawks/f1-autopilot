"""Prove that cutting the track cannot pay, under the current reward.

The first reward function taxed off-track distance at a flat rate while paying
full progress credit, so a policy fast enough to outrun the tax profited from
running wide -- and SAC found that within 130k steps. The training curves gave
no hint; it took a track-limits evaluation to catch.

This checks the economics directly instead of waiting for a training run:
a fast line that leaves the surface must score BELOW a slower line that stays
on it. If that inequality does not hold, no amount of training will produce a
legal lap, because the illegal one is genuinely optimal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import (  # noqa: E402
    OFFTRACK_TERMINATE, W_OFFTRACK, W_PROGRESS,
    MockBackend, RacingEnv,
)


def score_line(offset_m: float, speed_gain: float, steps: int = 600) -> dict:
    """Reward for holding a fixed cross-track offset while gaining speed.

    Models the trade the policy actually faces: running `offset_m` from the
    centreline buys `speed_gain` extra metres of progress per step.
    """
    env = RacingEnv(MockBackend(seed=0))
    width = env.backend.track.width
    excess = max(0.0, abs(offset_m) - width)

    from f1ai.rl.env import OFFTRACK_FADE
    credit = max(0.0, 1.0 - excess / OFFTRACK_FADE)

    base_ds = 2.4                       # metres per step at racing speed
    ds = base_ds * (1.0 + speed_gain)
    per_step = W_PROGRESS * ds * credit - W_OFFTRACK * excess
    terminates = excess > OFFTRACK_TERMINATE
    return {
        "offset": offset_m, "excess": excess, "ds": ds,
        "per_step": per_step, "total": per_step * steps,
        "terminates": terminates,
    }


def main() -> None:
    env = RacingEnv(MockBackend(seed=0))
    width = env.backend.track.width
    print(f"track half-width {width:.1f} m, "
          f"W_PROGRESS={W_PROGRESS}, W_OFFTRACK={W_OFFTRACK}, "
          f"terminate beyond {OFFTRACK_TERMINATE:.1f} m\n")

    # A clean line on the surface, versus increasingly greedy cuts that buy
    # progressively more speed for going wider.
    lines = {
        "centre, no gain":       score_line(0.0, 0.00),
        "edge, +10% speed":      score_line(6.8, 0.10),
        "1 m over, +20%":        score_line(8.0, 0.20),
        "2 m over, +30%":        score_line(9.0, 0.30),
        "3 m over, +40%":        score_line(10.0, 0.40),
    }

    print(f"{'line':<22} {'excess':>7} {'ds/step':>8} {'reward/step':>12} "
          f"{'600-step':>10} {'ends ep':>8}")
    print("-" * 72)
    for name, r in lines.items():
        print(f"{name:<22} {r['excess']:>7.2f} {r['ds']:>8.2f} "
              f"{r['per_step']:>12.3f} {r['total']:>10.0f} "
              f"{str(r['terminates']):>8}")

    clean = lines["edge, +10% speed"]["per_step"]
    checks = [
        ("staying on surface beats 1 m over",
         clean > lines["1 m over, +20%"]["per_step"],
         f"{clean:.2f} vs {lines['1 m over, +20%']['per_step']:.2f} per step"),
        ("staying on surface beats 2 m over",
         clean > lines["2 m over, +30%"]["per_step"],
         f"{clean:.2f} vs {lines['2 m over, +30%']['per_step']:.2f} per step"),
        ("staying on surface beats 3 m over",
         clean > lines["3 m over, +40%"]["per_step"],
         f"{clean:.2f} vs {lines['3 m over, +40%']['per_step']:.2f} per step"),
        ("deep cuts end the episode",
         lines["3 m over, +40%"]["terminates"], ""),
        ("using full track width still pays",
         lines["edge, +10% speed"]["per_step"]
         > lines["centre, no gain"]["per_step"],
         "racing line must remain worthwhile"),
    ]

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {detail}")
        failed += not ok

    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- the optimal policy is illegal;")
        print("training will faithfully learn to cheat. Fix the reward first.")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- cutting is strictly unprofitable")


if __name__ == "__main__":
    main()
