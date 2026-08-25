"""Break down the cost of one SAC gradient step, to find what to cut.

Online, the learner managed 0.10 gradient steps per environment step because
each update took 302 ms while the game held the GPU. Raising that ratio is the
single biggest quality lever available without leaving the game, and the way to
raise it is to know which part of the update is expensive rather than guessing.

Run this with the game CLOSED for a clean baseline, then again with the game
running to see the contention cost.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.buffer import ReplayBuffer  # noqa: E402
from f1ai.rl.env import FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM  # noqa: E402
from f1ai.rl.sac import SACAgent, SACConfig, random_shift  # noqa: E402


def timed(fn, n: int = 30) -> float:
    """Median milliseconds, CUDA-synchronised."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", default="64,128,256")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs CUDA")
        sys.exit(1)

    dev = torch.device("cuda")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name(0)}")
    print(f"vram {free_b / 1e9:.2f} GB free of {total_b / 1e9:.1f} GB "
          f"-- run once with the game closed and once with it open\n")

    buf = ReplayBuffer(2000, (FRAME_H, FRAME_W), STATE_DIM, 3,
                       stack=FRAME_STACK, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(6):
        buf.start_episode()
        for _ in range(300):
            buf.add(rng.integers(0, 255, (FRAME_H, FRAME_W), dtype=np.uint8),
                    rng.normal(0, 1, STATE_DIM).astype(np.float32),
                    rng.uniform(-1, 1, 3).astype(np.float32), 1.0, False)

    print(f"{'batch':>6} {'sample':>9} {'to gpu':>9} {'shift':>9} "
          f"{'encoder':>9} {'update':>9} {'no-shift':>10}")
    print("-" * 70)

    for bs in (int(v) for v in a.batches.split(",")):
        cfg = SACConfig(batch_size=bs)
        agent = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3, cfg,
                         device="cuda")

        t_sample = timed(lambda: buf.sample(bs))
        batch = buf.sample(bs)

        def to_gpu():
            torch.as_tensor(batch["frames"], device=dev).float()

        t_gpu = timed(to_gpu)

        frames = torch.as_tensor(batch["frames"], device=dev).float()
        t_shift = timed(lambda: random_shift(frames, cfg.shift_pad))
        t_enc = timed(lambda: agent.encoder(frames))
        t_update = timed(lambda: agent.update(batch), n=20)

        # The same update with augmentation switched off, which is the
        # cheapest thing to drop if the shift turns out to dominate.
        cfg_ns = SACConfig(batch_size=bs, shift_pad=0)
        agent_ns = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                            cfg_ns, device="cuda")
        t_noshift = timed(lambda: agent_ns.update(batch), n=20)

        print(f"{bs:>6} {t_sample:>8.2f}m {t_gpu:>8.2f}m {t_shift:>8.2f}m "
              f"{t_enc:>8.2f}m {t_update:>8.2f}m {t_noshift:>9.2f}m")

        del agent, agent_ns, frames
        torch.cuda.empty_cache()

    print("\nAt 30 Hz the actor needs ~8 ms of every 33 ms slot, so the learner")
    print("has roughly 25 ms per environment step to work with. Update cost")
    print("divided into that budget is the update-to-data ratio you can expect.")


if __name__ == "__main__":
    main()
