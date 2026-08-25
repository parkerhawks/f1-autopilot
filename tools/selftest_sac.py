"""Check the SAC internals against properties that must hold exactly.

Every one of these fails silently in a full training run -- the losses still
move, the agent still drives, it just never gets good. Checking them takes
seconds and removes them from the list of suspects when training disappoints.

  * the tanh log-prob matches a brute-force change-of-variables computation
  * actions stay strictly inside (-1, 1)
  * the twin critics really are independent
  * the actor's gradients never reach the encoder
  * target networks track slowly, not instantly
  * alpha moves toward the entropy target from both directions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.buffer import ReplayBuffer  # noqa: E402
from f1ai.rl.env import FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM  # noqa: E402
from f1ai.rl.sac import SACAgent, SACConfig, random_shift  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _entropy_of(agent: SACAgent, batch: dict) -> float:
    """Mean policy entropy on a batch, in nats -- what alpha is chasing."""
    with torch.no_grad():
        frames = torch.as_tensor(batch["frames"], dtype=torch.float32,
                                 device=agent.device)
        state = torch.as_tensor(batch["state"], dtype=torch.float32,
                                device=agent.device)
        _, logp = agent.actor(agent.encoder(frames), state)
        return float(-logp.mean())


def make_batch(buf: ReplayBuffer, n: int) -> dict:
    rng = np.random.default_rng(0)
    for ep in range(6):
        buf.start_episode()
        for _ in range(80):
            buf.add(rng.integers(0, 255, (FRAME_H, FRAME_W), dtype=np.uint8),
                    rng.normal(0, 1, STATE_DIM).astype(np.float32),
                    rng.uniform(-1, 1, 3).astype(np.float32),
                    float(rng.normal(1.0, 0.5)), False)
    return buf.sample(n)


def main() -> None:
    torch.manual_seed(0)
    checks: list[tuple[str, bool, str]] = []

    agent = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                     SACConfig(batch_size=64), device=DEV)
    n_params = sum(p.numel() for p in agent.encoder.parameters()) \
        + sum(p.numel() for p in agent.actor.parameters()) \
        + sum(p.numel() for p in agent.critic.parameters())
    print(f"device {DEV}, {n_params:,} parameters "
          f"(encoder flatten = {agent.encoder.n_flat})\n")

    # -- tanh log-prob correctness ---------------------------------------
    # Compare against an independent computation using torch.distributions.
    feat = torch.randn(512, agent.cfg.feature_dim, device=DEV)
    st = torch.randn(512, STATE_DIM, device=DEV)
    torch.manual_seed(1)
    a, logp = agent.actor(feat, st)

    h = agent.actor.net(torch.cat([feat, st], dim=1))
    mu, log_std = agent.actor.mu(h), torch.clamp(
        agent.actor.log_std(h), -10.0, 2.0)
    u = torch.atanh(a.clamp(-0.999999, 0.999999))
    ref = torch.distributions.Normal(mu, log_std.exp()).log_prob(u)
    ref = ref - torch.log(1.0 - a.pow(2) + 1e-6)
    ref = ref.sum(dim=1, keepdim=True)
    err = (logp - ref).abs().max().item()
    checks.append(("tanh log-prob matches reference", err < 2e-2,
                   f"max abs error {err:.2e}"))

    checks.append(("actions inside (-1, 1)",
                   bool((a.abs() < 1.0).all()), f"max |a| {a.abs().max():.6f}"))
    checks.append(("log-prob finite",
                   bool(torch.isfinite(logp).all()), ""))

    # -- twin critics are independent -------------------------------------
    q1, q2 = agent.critic(feat, st, a)
    diff = (q1 - q2).abs().mean().item()
    checks.append(("twin critics differ", diff > 1e-4,
                   f"mean |Q1-Q2| = {diff:.4f}"))

    # -- random shift preserves shape and changes content -----------------
    x = torch.randint(0, 255, (16, FRAME_STACK, FRAME_H, FRAME_W),
                      device=DEV).float()
    xs = random_shift(x, 4)
    changed = (xs != x).any(dim=(1, 2, 3)).float().mean().item()
    checks.append(("random shift keeps shape", xs.shape == x.shape,
                   str(tuple(xs.shape))))
    checks.append(("random shift perturbs most frames", changed > 0.8,
                   f"{changed:.0%} of frames altered"))

    # -- the actor must not touch the encoder -----------------------------
    buf = ReplayBuffer(1000, (FRAME_H, FRAME_W), STATE_DIM, 3,
                       stack=FRAME_STACK, seed=0)
    batch = make_batch(buf, 64)

    enc_before = [p.detach().clone() for p in agent.encoder.parameters()]
    # Freeze the critic optimiser so only the actor path can move weights.
    for g in agent.opt_critic.param_groups:
        g["lr"] = 0.0
    agent.update(batch)
    for g in agent.opt_critic.param_groups:
        g["lr"] = agent.cfg.critic_lr
    enc_moved = max(
        (p.detach() - b).abs().max().item()
        for p, b in zip(agent.encoder.parameters(), enc_before))
    checks.append(("actor gradients never reach encoder", enc_moved == 0.0,
                   f"max weight delta {enc_moved:.2e}"))

    # -- target networks move slowly --------------------------------------
    agent2 = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                      SACConfig(batch_size=64, tau=0.01), device=DEV)
    tgt_before = [p.detach().clone()
                  for p in agent2.critic_target.parameters()]
    for _ in range(4):
        agent2.update(batch)
    online = list(agent2.critic.parameters())
    tgt_after = list(agent2.critic_target.parameters())
    moved = max((a_.detach() - b).abs().max().item()
                for a_, b in zip(tgt_after, tgt_before))
    gap = max((o.detach() - t.detach()).abs().max().item()
              for o, t in zip(online, tgt_after))
    checks.append(("targets move at all", moved > 0.0, f"delta {moved:.2e}"))
    checks.append(("targets lag the online net", gap > 0.0,
                   f"online-target gap {gap:.2e}"))

    # -- alpha chases the entropy target ----------------------------------
    # Force a low-entropy policy: alpha should be pushed UP to restore it.
    a3 = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                  SACConfig(batch_size=64, alpha_lr=3e-2, init_alpha=0.2),
                  device=DEV)
    with torch.no_grad():
        a3.actor.log_std.bias.fill_(-6.0)   # near-deterministic policy
    start = float(a3.alpha)
    ent_lo = _entropy_of(a3, batch)
    for _ in range(30):
        a3.update(batch)
    rose = float(a3.alpha)
    checks.append(("alpha rises when entropy too low", rose > start,
                   f"alpha {start:.4f} -> {rose:.4f} at entropy {ent_lo:+.2f} "
                   f"(target {a3.target_entropy:+.0f})"))

    # NOTE: entropy is NOT monotonic in log_std for a tanh-squashed Gaussian.
    # The support is bounded, so a large pre-squash std piles mass against
    # +/-1 and spikes the density there -- log_std=1.5 gives entropy near -8.5,
    # far BELOW the -3 target. Entropy peaks around std=1. Reading a rising
    # log_std in the logs as "exploring more" is therefore wrong past that
    # point, and it is an easy way to misdiagnose a stuck policy.
    a4 = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                  SACConfig(batch_size=64, alpha_lr=3e-2, init_alpha=0.9),
                  device=DEV)
    with torch.no_grad():
        a4.actor.log_std.bias.fill_(0.0)    # near maximum entropy (~+2)
    start4 = float(a4.alpha)
    for _ in range(30):
        a4.update(batch)
    fell = float(a4.alpha)
    ent_hi = _entropy_of(a4, batch)
    checks.append(("alpha falls when entropy too high", fell < start4,
                   f"alpha {start4:.4f} -> {fell:.4f} at entropy {ent_hi:+.2f} "
                   f"(target {a4.target_entropy:+.0f})"))

    # -- one update produces finite metrics -------------------------------
    m = agent.update(batch)
    finite = all(np.isfinite(v) for v in m.values())
    checks.append(("update metrics finite", finite,
                   ", ".join(f"{k}={v:.3f}" for k, v in m.items())))

    # -- action mapping ---------------------------------------------------
    env_a = SACAgent.to_env_action(np.array([-1.0, -1.0, 1.0], np.float32))
    ok = (abs(env_a[0] + 1) < 1e-6 and abs(env_a[1]) < 1e-6
          and abs(env_a[2] - 1.0) < 1e-6)
    checks.append(("action mapping to env ranges", ok, str(env_a)))

    failed = 0
    for name, okk, detail in checks:
        print(f"  [{'PASS' if okk else 'FAIL'}] {name:<38} {detail}")
        failed += not okk
    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- SAC internals are sound")


if __name__ == "__main__":
    main()
