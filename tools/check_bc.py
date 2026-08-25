"""Sanity-check a behavioural-cloning checkpoint against its own training laps.

A small MSE proves nothing on its own. If the human holds roughly the same
control most of the lap, a network that ignores the road entirely and emits the
dataset mean also scores a small MSE -- and it will drive straight into the
first corner.

So the loss is compared against that exact baseline. R2 above zero means the
policy beats mean-prediction; at or below zero it has learned nothing useful,
whatever the loss looked like.

Also reports how strongly the predicted steering tracks the human's, since a
policy that outputs a plausible RANGE of values while being uncorrelated with
the corners is the other way this fails quietly.

    python tools/check_bc.py runs/bc/bc.pt demos/<session>/lap04_83.42s.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import (  # noqa: E402
    CONTROL_HZ, FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM,
)
from f1ai.rl.sac import SACAgent, SACConfig  # noqa: E402
from tools.train_bc import build_dataset  # noqa: E402

NAMES = ("steer", "throttle", "brake")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("laps", nargs="+")
    ap.add_argument("--delay-ms", type=float, default=100.0)
    a = ap.parse_args()

    files = [Path(p) for p in a.laps]
    delay_steps = int(round(a.delay_ms / 1000.0 * CONTROL_HZ))
    frames, states, actions = build_dataset(files, delay_steps, min_clean=0.0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                     SACConfig(), device=str(dev))
    agent.load_state_dict(torch.load(a.checkpoint, map_location=dev,
                                     weights_only=False))
    agent.encoder.eval()
    agent.actor.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(frames), 256):
            f = torch.as_tensor(frames[i:i + 256], device=dev).float()
            s = torch.as_tensor(states[i:i + 256], device=dev)
            mu, _ = agent.actor(agent.encoder(f), s, deterministic=True)
            preds.append(mu.cpu().numpy())
    pred = np.concatenate(preds)

    print(f"\n{len(pred):,} samples\n")
    print(f"{'channel':<10} {'MSE':>9} {'mean-MSE':>10} {'R2':>8} "
          f"{'corr':>7} {'pred std':>9} {'human std':>10}")
    print("-" * 68)

    checks = []
    for k, name in enumerate(NAMES):
        y, p = actions[:, k], pred[:, k]
        mse = float(np.mean((p - y) ** 2))
        # The baseline every honest report needs: predict the mean, always.
        base = float(np.mean((y.mean() - y) ** 2))
        r2 = 1.0 - mse / base if base > 1e-12 else 0.0
        corr = (float(np.corrcoef(p, y)[0, 1])
                if p.std() > 1e-9 and y.std() > 1e-9 else 0.0)
        print(f"{name:<10} {mse:>9.5f} {base:>10.5f} {r2:>8.3f} "
              f"{corr:>7.3f} {p.std():>9.4f} {y.std():>10.4f}")
        checks.append((name, r2, corr, p.std(), y.std()))

    print()
    failed = 0
    for name, r2, corr, pstd, ystd in checks:
        ok = r2 > 0.3 and corr > 0.6
        print(f"  [{'PASS' if ok else 'WARN'}] {name:<9} "
              f"R2 {r2:+.2f}, correlation {corr:+.2f}")
        failed += not ok

    # A policy whose outputs barely vary has collapsed to the mean regardless
    # of what its loss says.
    collapsed = [n for n, _, _, p, y in checks if p < 0.25 * y]
    if collapsed:
        print(f"\n  COLLAPSED toward the mean on: {', '.join(collapsed)}")
        print("  Output variance is far below the human's. More data, more")
        print("  epochs, or a larger encoder.")

    print()
    if failed:
        print(f"{failed}/{len(checks)} channels are weak. The policy may still")
        print("be a usable warm start -- steering matters most -- but do not")
        print("expect it to complete a lap unaided.")
    else:
        print("All channels track the human. This is a sound warm start.")
    print("\nThe real test is still driving it in-game.")


if __name__ == "__main__":
    main()
