"""Train on a saved replay buffer with the GPU to ourselves.

    python tools/train_offline.py runs/f1-night2/buffer.npz \
        --init-from runs/f1-night2/final.pt --steps 300000

WHY THIS IS THE BIGGEST AVAILABLE WIN
-------------------------------------
Online, the actor is pinned to 30 Hz and the learner has to yield to it or the
control loop falls apart. Measured over a 5.7-hour run: 38,281 gradient steps
against 387,600 environment steps -- an update-to-data ratio of 0.10, with the
learner backing off 2.2 s after every update and each update taking 302 ms
while the game had the GPU.

Pixel-based SAC normally wants an order of magnitude more gradient steps than
that. The policy was not short of experience; it was short of training on the
experience it already had.

Offline there is no game, no 30 Hz deadline and no contention. The same
gradient step costs roughly 57 ms instead of 302, and nothing has to yield --
so an hour here is worth something like fifteen hours of online learning, on
data already collected.

WHAT IT CANNOT DO
-----------------
Only what the buffer contains. The policy will drift toward actions it has
never tried, and the critic will confidently value states nobody visited --
the standard offline-RL extrapolation problem. So alternate: collect online,
train offline, collect again with the improved policy. Each online session
gathers data around the new behaviour, which is what keeps the estimates
honest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.buffer import ReplayBuffer  # noqa: E402
from f1ai.rl.env import FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM  # noqa: E402
from f1ai.rl.sac import SACAgent, SACConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("buffer")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to continue training (strongly advised: "
                         "starting cold wastes the online run entirely)")
    ap.add_argument("--steps", type=int, default=200_000,
                    help="gradient steps")
    ap.add_argument("--batch", type=int, default=256,
                    help="larger than online, since nothing else wants the GPU")
    ap.add_argument("--anchor", action="store_true",
                    help="hold the policy near the loaded checkpoint; useful "
                         "when the buffer is small relative to the step count")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {a.buffer} ...")
    t0 = time.perf_counter()
    buf = ReplayBuffer.load(Path(a.buffer), seed=0)
    print(f"  {buf.size:,} transitions, {buf.nbytes() / 1e9:.1f} GB, "
          f"loaded in {time.perf_counter() - t0:.0f}s")

    cfg = SACConfig(batch_size=a.batch)
    agent = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3, cfg,
                     device=dev)

    if a.init_from:
        sd = torch.load(a.init_from, map_location=dev, weights_only=False)
        agent.load_state_dict(sd)
        print(f"continuing from {a.init_from} "
              f"({agent.grad_steps:,} prior gradient steps)")
        if a.anchor:
            agent.anchor_to_bc()
            print("  anchored to the loaded policy, decaying over "
                  f"{cfg.bc_anchor_decay:,} steps")
    else:
        print("WARNING: starting from random weights, discarding the online "
              "run. Pass --init-from unless that is deliberate.")

    out = Path(a.out) if a.out else Path(a.buffer).parent / "offline"
    out.mkdir(parents=True, exist_ok=True)
    fh = (out / "metrics.csv").open("w", newline="")
    cols = ["grad_step", "critic_loss", "actor_loss", "q_mean", "target_q",
            "alpha", "entropy", "sps"]
    writer = csv.DictWriter(fh, fieldnames=cols)
    writer.writeheader()

    print(f"\n{a.steps:,} gradient steps at batch {a.batch}")
    print(f"{'step':>9} {'critic':>9} {'actor':>10} {'q':>9} {'alpha':>7} "
          f"{'H':>7} {'steps/s':>8} {'eta':>7}")
    print("-" * 74)

    start = time.perf_counter()
    last = start
    last_step = 0
    metrics: dict[str, float] = {}

    try:
        for i in range(1, a.steps + 1):
            metrics.update(agent.update(buf.sample(a.batch)))

            now = time.perf_counter()
            if now - last > 30.0:
                sps = (i - last_step) / (now - last)
                eta = (a.steps - i) / max(1e-9, sps) / 60.0
                print(f"{i:>9,} {metrics.get('critic_loss', 0):>9.3f} "
                      f"{metrics.get('actor_loss', 0):>10.2f} "
                      f"{metrics.get('q_mean', 0):>9.2f} "
                      f"{metrics.get('alpha', 0):>7.4f} "
                      f"{metrics.get('entropy', 0):>7.2f} "
                      f"{sps:>8.1f} {eta:>6.0f}m")
                writer.writerow({"grad_step": i, "sps": round(sps, 2),
                                 **{k: round(v, 5) for k, v in metrics.items()
                                    if k in cols}})
                fh.flush()
                last, last_step = now, i

            if i % 25_000 == 0:
                torch.save(agent.state_dict(), out / "latest.pt")
    except KeyboardInterrupt:
        print("\nstopped early -- saving.")

    torch.save(agent.state_dict(), out / "final.pt")
    fh.close()
    elapsed = time.perf_counter() - start
    print(f"\ndone: {agent.grad_steps:,} total gradient steps "
          f"in {elapsed / 60:.1f} min ({a.steps / elapsed:.1f}/s)")
    (out / "config.json").write_text(json.dumps({
        "buffer": str(a.buffer), "init_from": a.init_from,
        "steps": a.steps, "batch": a.batch, "anchor": a.anchor,
        "sac": vars(cfg),
    }, indent=2))
    print(f"wrote {out / 'final.pt'}")
    print("\nEvaluate it in the game before trusting it -- offline training "
          "can drift\ntoward actions the buffer never contained:")
    print(f"  python tools/drive.py {out / 'final.pt'} --minutes 5")


if __name__ == "__main__":
    main()
