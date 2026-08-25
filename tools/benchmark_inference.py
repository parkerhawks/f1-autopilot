"""Measure the policy's inference latency against the control-loop budget.

At 30 Hz the whole loop -- capture, preprocess, infer, write to the pad -- has
33.3 ms. This measures the inference slice at several input resolutions so the
Phase 2 resolution choice is made against real numbers on this GPU, while F1 25
is competing for it.

Run it twice: once on an idle GPU, once with the game running. The gap between
those two numbers is the contention cost, and it is the one that counts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.model.policy import DrivingPolicy  # noqa: E402

STATE_DIM = 24          # placeholder; the real vector is built in Phase 1
WARMUP = 30
ITERS = 200

CONFIGS = [
    ("PilotNet   3x66x200", 3, (66, 200)),
    ("small      3x96x192", 3, (96, 192)),
    ("stacked    4x96x192", 4, (96, 192)),
    ("medium     3x120x240", 3, (120, 240)),
    ("large      4x144x256", 4, (144, 256)),
]


def bench(model: DrivingPolicy, frames, state, amp: bool) -> float:
    """Median milliseconds per forward pass at batch 1."""
    with torch.no_grad():
        for _ in range(WARMUP):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                model(frames, state)
        torch.cuda.synchronize()

        times = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                model(frames, state)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available -- this benchmark needs the GPU")
        sys.exit(1)

    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    free, total = torch.cuda.mem_get_info()
    print(f"vram:   {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB\n")

    print(f"{'config':<22} {'params':>9} {'fp32':>8} {'fp16':>8} {'@30Hz':>8}")
    print("-" * 60)

    for name, ch, hw in CONFIGS:
        model = DrivingPolicy(
            state_dim=STATE_DIM, frame_channels=ch, frame_hw=hw
        ).to(dev).eval()

        frames = torch.randn(1, ch, *hw, device=dev)
        state = torch.randn(1, STATE_DIM, device=dev)

        ms32 = bench(model, frames, state, amp=False)
        ms16 = bench(model, frames, state, amp=True)
        budget = 100.0 * min(ms32, ms16) / 33.3

        print(f"{name:<22} {model.num_parameters():>9,} "
              f"{ms32:>7.2f}m {ms16:>7.2f}m {budget:>7.1f}%")

        del model, frames, state
        torch.cuda.empty_cache()

    print()
    print("'@30Hz' is the share of a 33.3 ms control loop spent on inference,")
    print("at the faster of the two precisions. Capture and preprocessing are")
    print("measured separately by tools/benchmark_capture.py.")


if __name__ == "__main__":
    main()
