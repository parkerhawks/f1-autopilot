"""Prove a behavioural-cloning warm start survives contact with SAC.

Loading a BC actor beside a randomly-initialised critic is the obvious way to
warm-start, and on its own it does not work: the first actor updates follow Q
values that are still noise, and the imitation is gone within a few hundred
gradient steps. The training curves show nothing unusual, because from SAC's
point of view nothing unusual happened.

Two guards are checked here:

  * the actor does not move at all during critic warmup
  * afterwards the imitation anchor holds it near the demonstrated policy,
    and that anchor decays to zero so the final policy is free to beat it

The last point matters as much as the first. An anchor that never releases
caps the agent at human performance, which defeats the purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.buffer import ReplayBuffer  # noqa: E402
from f1ai.rl.env import FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM  # noqa: E402
from f1ai.rl.sac import SACAgent, SACConfig  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(n: int = 64) -> dict:
    buf = ReplayBuffer(600, (FRAME_H, FRAME_W), STATE_DIM, 3,
                       stack=FRAME_STACK, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(4):
        buf.start_episode()
        for _ in range(90):
            buf.add(rng.integers(0, 255, (FRAME_H, FRAME_W), dtype=np.uint8),
                    rng.normal(0, 1, STATE_DIM).astype(np.float32),
                    rng.uniform(-1, 1, 3).astype(np.float32),
                    float(rng.normal(1.0, 0.5)), False)
    return buf.sample(n)


def actor_snapshot(agent: SACAgent) -> list[torch.Tensor]:
    return [p.detach().clone() for p in agent.actor.parameters()]


def max_delta(agent: SACAgent, before: list[torch.Tensor]) -> float:
    return max((p.detach() - b).abs().max().item()
               for p, b in zip(agent.actor.parameters(), before))


def main() -> None:
    torch.manual_seed(0)
    batch = make_batch()
    checks: list[tuple[str, bool, str]] = []

    cfg = SACConfig(batch_size=64, critic_warmup=50, bc_anchor=2.0,
                    bc_anchor_decay=500)
    agent = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3, cfg,
                     device=DEV)
    agent.anchor_to_bc()

    print(f"gamma {cfg.gamma}  ->  horizon "
          f"{1.0 / (1.0 - cfg.gamma):.0f} steps = "
          f"{1.0 / (1.0 - cfg.gamma) / 30.0:.1f} s at 30 Hz\n")

    # -- the actor must be perfectly still during warmup ------------------
    before = actor_snapshot(agent)
    for _ in range(cfg.critic_warmup - 2):
        agent.update(batch)
    frozen_delta = max_delta(agent, before)
    checks.append(("actor frozen during critic warmup", frozen_delta == 0.0,
                   f"max weight change {frozen_delta:.2e} over "
                   f"{cfg.critic_warmup - 2} steps"))

    # -- and must start moving once warmup ends ---------------------------
    before = actor_snapshot(agent)
    for _ in range(30):
        agent.update(batch)
    moved = max_delta(agent, before)
    checks.append(("actor moves after warmup", moved > 0.0,
                   f"max weight change {moved:.2e}"))

    # -- the anchor must decay to exactly zero ----------------------------
    w_start = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3, cfg,
                       device=DEV)
    w_start.anchor_to_bc()
    w0 = w_start.bc_weight()
    w_start.grad_steps = cfg.bc_anchor_decay // 2
    w_half = w_start.bc_weight()
    w_start.grad_steps = cfg.bc_anchor_decay + 10
    w_end = w_start.bc_weight()
    checks.append(("anchor starts at full strength", abs(w0 - cfg.bc_anchor) < 1e-6,
                   f"{w0:.2f}"))
    checks.append(("anchor decays", w_half < w0, f"{w0:.2f} -> {w_half:.2f}"))
    checks.append(("anchor reaches zero", w_end == 0.0,
                   f"{w_end:.3f} after {cfg.bc_anchor_decay:,} steps"))

    # -- no anchor without a BC checkpoint --------------------------------
    plain = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3, cfg,
                     device=DEV)
    before = actor_snapshot(plain)
    for _ in range(6):
        plain.update(batch)
    checks.append(("from-scratch runs have no warmup or anchor",
                   plain.bc_weight() == 0.0 and max_delta(plain, before) > 0.0,
                   "actor trains immediately"))

    # -- the anchor actually constrains the policy ------------------------
    anchored = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                        SACConfig(batch_size=64, critic_warmup=0,
                                  bc_anchor=50.0, bc_anchor_decay=100_000),
                        device=DEV)
    anchored.anchor_to_bc()
    free = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                    SACConfig(batch_size=64, critic_warmup=0, bc_anchor=0.0),
                    device=DEV)
    free.load_state_dict(anchored.state_dict())

    for _ in range(60):
        anchored.update(batch)
        free.update(batch)

    def divergence(agent: SACAgent, reference: SACAgent) -> float:
        with torch.no_grad():
            f = torch.as_tensor(batch["frames"], dtype=torch.float32,
                                device=agent.device)
            s = torch.as_tensor(batch["state"], dtype=torch.float32,
                                device=agent.device)
            a, _ = agent.actor(agent.encoder(f), s, deterministic=True)
            b, _ = reference._bc_actor(reference._bc_encoder(f), s,
                                       deterministic=True)
            return float((a - b).pow(2).mean())

    d_anchored = divergence(anchored, anchored)
    d_free = divergence(free, anchored)
    checks.append(("anchored policy stays nearer the demonstration",
                   d_anchored < d_free,
                   f"anchored {d_anchored:.4f} vs unanchored {d_free:.4f}"))

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")
        failed += not ok

    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- a BC warm start would be lost")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- the warm start is protected")


if __name__ == "__main__":
    main()
