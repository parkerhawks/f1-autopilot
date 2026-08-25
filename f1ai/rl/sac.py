"""Soft Actor-Critic over pixels plus proprioception.

Design choices that matter, and why:

**The critic owns the encoder.** Both actor and critic read the same CNN, but
only the critic's gradients update it; the actor sees detached features. Letting
the actor backprop into a shared encoder is a known instability -- the policy
gradient reshapes the representation to make its current actions look good,
which corrupts the value estimate that the policy is being scored against
(Yarats et al., SAC-AE). It fails slowly and looks like "RL is just noisy".

**Twin critics.** A single Q network trained against its own bootstrapped
target systematically overestimates: any upward noise gets selected for by the
max. Taking the minimum of two independently initialised critics cancels most
of it. If Q ever runs far above the actual episode return, suspect the target
network, not the reward.

**Entropy is tuned, not fixed.** Alpha trades exploration against exploitation
and the right value changes over training, so it is learned against a target
entropy of -dim(A). This is the single most informative number to watch: stuck
high means the policy never commits; collapsed early means it stopped exploring
before it found the racing line.

**Random-shift augmentation.** Pixel SAC is sample-starved, and padding then
randomly cropping each frame is close to free while cutting the sample cost
substantially (Kostrikov et al., DrQ). It also stops the CNN memorising exact
pixel positions, which is the wrong invariance for a camera that moves.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0


@dataclass
class SACConfig:
    lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    # Effective planning horizon is 1/(1-gamma) steps: at 30 Hz, 0.99 gives
    # only 100 steps = 3.3 s. That is shorter than a single braking zone --
    # the agent cannot value arriving at Ascari correctly if it cannot see
    # that far. 0.997 gives ~333 steps = 11 s, enough to connect a braking
    # decision to the corner exit that pays for it, without pushing variance
    # so high that the critic stops converging.
    gamma: float = 0.997
    tau: float = 0.01               # target network soft-update rate
    batch_size: int = 256
    feature_dim: int = 128
    hidden: int = 512
    init_alpha: float = 0.5
    actor_every: int = 2            # delay actor vs critic, as in TD3/DrQ
    target_every: int = 2
    shift_pad: int = 4              # random-shift augmentation, in pixels

    # -- protecting a behavioural-cloning warm start ----------------------
    # Loading a BC actor next to a randomly-initialised critic destroys the
    # imitation almost immediately: the first actor updates chase Q values
    # that are still noise. Two guards, both standard practice:
    #
    #   critic_warmup   the actor is frozen while the critic learns to say
    #                   something meaningful about the demonstrated states
    #   bc_anchor       the actor is pulled toward a FROZEN copy of the BC
    #                   policy, with the pull decaying as the critic matures
    #                   (the TD3+BC idea: keep the policy near data you trust
    #                   until value estimates deserve trust instead)
    critic_warmup: int = 5_000      # gradient steps before the actor moves
    bc_anchor: float = 2.0          # initial weight on the imitation term
    bc_anchor_decay: int = 50_000   # gradient steps for it to reach zero


def random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Pad by `pad` px (replicate) then take a random crop of the original size."""
    if pad <= 0:
        return x
    n, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    top = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    left = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = torch.arange(h, device=x.device)
    cols = torch.arange(w, device=x.device)
    idx_r = (top[:, None] + rows[None, :])
    idx_c = (left[:, None] + cols[None, :])
    batch = torch.arange(n, device=x.device)[:, None, None]
    return x[batch, :, idx_r[:, :, None], idx_c[:, None, :]].permute(0, 3, 1, 2)


class Encoder(nn.Module):
    """Convolutional trunk shared by actor and critic."""

    def __init__(self, in_ch: int, hw: tuple[int, int], feature_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_ch, *hw)).numel()
        self.n_flat = n_flat
        # LayerNorm + tanh keeps the feature scale bounded; without it the
        # critic's gradients drive the encoder output to grow without limit.
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flat, feature_dim),
            nn.LayerNorm(feature_dim), nn.Tanh(),
        )

    def forward(self, frames_u8: torch.Tensor) -> torch.Tensor:
        x = frames_u8.float() / 255.0 - 0.5
        return self.proj(self.conv(x))


class Actor(nn.Module):
    """Tanh-squashed diagonal Gaussian over (steer, throttle, brake)."""

    def __init__(self, feature_dim: int, state_dim: int, action_dim: int,
                 hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + state_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, feat: torch.Tensor, state: torch.Tensor,
                deterministic: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([feat, state], dim=1))
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)

        if deterministic:
            return torch.tanh(mu), torch.zeros(mu.shape[0], 1, device=mu.device)

        std = log_std.exp()
        noise = torch.randn_like(mu)
        u = mu + std * noise                       # reparameterised sample
        a = torch.tanh(u)

        # log pi(a|s) with the tanh change-of-variables correction, in the
        # numerically stable form: log(1 - tanh(u)^2) = 2(log2 - u - softplus(-2u))
        log_prob = (-0.5 * noise.pow(2) - log_std - 0.5 * np.log(2 * np.pi))
        log_prob = log_prob - 2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))
        return a, log_prob.sum(dim=1, keepdim=True)


class Critic(nn.Module):
    """Twin Q networks evaluated on the same features."""

    def __init__(self, feature_dim: int, state_dim: int, action_dim: int,
                 hidden: int):
        super().__init__()

        def q() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim + state_dim + action_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )

        # Separate calls, so the two critics get independent initialisations.
        # Sharing weights here would defeat the whole point of the twin trick.
        self.q1, self.q2 = q(), q()

    def forward(self, feat: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([feat, state, action], dim=1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, frame_shape: tuple[int, int, int], state_dim: int,
                 action_dim: int = 3, cfg: SACConfig | None = None,
                 device: str = "cuda"):
        self.cfg = cfg or SACConfig()
        self.device = torch.device(device)
        c, h, w = frame_shape
        k = self.cfg

        self.encoder = Encoder(c, (h, w), k.feature_dim).to(self.device)
        self.actor = Actor(k.feature_dim, state_dim, action_dim,
                           k.hidden).to(self.device)
        self.critic = Critic(k.feature_dim, state_dim, action_dim,
                             k.hidden).to(self.device)
        # Targets cover the encoder too, so the bootstrap target does not move
        # every time the representation shifts.
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.encoder_target = copy.deepcopy(self.encoder).to(self.device)
        for p in list(self.critic_target.parameters()) + \
                list(self.encoder_target.parameters()):
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(
            np.log(k.init_alpha), dtype=torch.float32,
            device=self.device, requires_grad=True)
        # Standard heuristic: one nat of entropy budget per action dimension.
        self.target_entropy = -float(action_dim)

        self.opt_critic = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.encoder.parameters()),
            lr=k.critic_lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=k.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=k.alpha_lr)

        self.grad_steps = 0

        # Frozen snapshot of the demonstrated policy, set by anchor_to_bc().
        self._bc_encoder: Encoder | None = None
        self._bc_actor: Actor | None = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def anchor_to_bc(self) -> None:
        """Freeze the CURRENT actor as the imitation reference.

        Call immediately after loading a behavioural-cloning checkpoint and
        before any gradient step. The snapshot never trains, so it keeps
        representing the human's driving even as the live policy moves away
        from it.
        """
        self._bc_encoder = copy.deepcopy(self.encoder).eval()
        self._bc_actor = copy.deepcopy(self.actor).eval()
        for p in list(self._bc_encoder.parameters()) + \
                list(self._bc_actor.parameters()):
            p.requires_grad_(False)

    def bc_weight(self) -> float:
        """Anchor strength now: full at the start, zero once the critic is
        trusted, so the final policy is free to beat the demonstration."""
        if self._bc_actor is None or self.cfg.bc_anchor <= 0:
            return 0.0
        frac = 1.0 - self.grad_steps / max(1, self.cfg.bc_anchor_decay)
        return self.cfg.bc_anchor * max(0.0, frac)

    # -- acting ------------------------------------------------------------

    @torch.no_grad()
    def act(self, frames: np.ndarray, state: np.ndarray,
            deterministic: bool = False) -> np.ndarray:
        f = torch.as_tensor(frames, device=self.device).unsqueeze(0)
        s = torch.as_tensor(state, device=self.device).unsqueeze(0)
        feat = self.encoder(f)
        a, _ = self.actor(feat, s, deterministic=deterministic)
        return a.squeeze(0).cpu().numpy()

    @staticmethod
    def to_env_action(a: np.ndarray) -> np.ndarray:
        """Network output in [-1,1]^3 -> (steer, throttle, brake).

        Keeping the network on a symmetric range lets the tanh-Gaussian maths
        stay standard; the pedals are remapped at the boundary instead.
        """
        return np.array([a[0], (a[1] + 1.0) * 0.5, (a[2] + 1.0) * 0.5],
                        dtype=np.float32)

    # -- learning ----------------------------------------------------------

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        k = self.cfg
        dev = self.device
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)

        frames = t(batch["frames"], torch.uint8)
        next_frames = t(batch["next_frames"], torch.uint8)
        state = t(batch["state"])
        next_state = t(batch["next_state"])
        action = t(batch["action"])
        reward = t(batch["reward"]).unsqueeze(1)
        not_done = 1.0 - t(batch["terminal"]).unsqueeze(1)

        frames = random_shift(frames.float(), k.shift_pad)
        next_frames = random_shift(next_frames.float(), k.shift_pad)

        # -- critic --------------------------------------------------------
        with torch.no_grad():
            next_feat = self.encoder_target(next_frames)
            next_action, next_logp = self.actor(next_feat, next_state)
            q1_t, q2_t = self.critic_target(next_feat, next_state, next_action)
            target_v = torch.min(q1_t, q2_t) - self.alpha.detach() * next_logp
            target_q = reward + not_done * k.gamma * target_v

        feat = self.encoder(frames)
        q1, q2 = self.critic(feat, state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.encoder.parameters()),
            10.0)
        self.opt_critic.step()

        metrics = {
            "critic_loss": float(critic_loss.detach()),
            "q_mean": float(q1.mean().detach()),
            "target_q": float(target_q.mean()),
            "alpha": float(self.alpha.detach()),
        }

        # -- actor and alpha, on a slower clock ----------------------------
        # While the critic is still warming up its Q values are noise, and an
        # actor that chases them will throw away a behavioural-cloning warm
        # start within a few hundred steps. Hold the actor still until the
        # critic has something to say.
        warming_up = (self._bc_actor is not None
                      and self.grad_steps < k.critic_warmup)
        metrics["critic_warmup"] = float(warming_up)

        if self.grad_steps % k.actor_every == 0 and not warming_up:
            # Detached features: the policy must not reshape the encoder.
            feat_d = feat.detach()
            new_action, logp = self.actor(feat_d, state)
            q1_pi, q2_pi = self.critic(feat_d, state, new_action)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * logp - q_pi).mean()

            # Imitation anchor, decaying to nothing.
            w = self.bc_weight()
            if w > 0.0:
                with torch.no_grad():
                    bc_action, _ = self._bc_actor(
                        self._bc_encoder(frames), state, deterministic=True)
                bc_loss = F.mse_loss(new_action, bc_action)
                actor_loss = actor_loss + w * bc_loss
                metrics["bc_weight"] = w
                metrics["bc_divergence"] = float(bc_loss.detach())

            self.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.opt_actor.step()

            alpha_loss = -(self.log_alpha
                           * (logp.detach() + self.target_entropy)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()

            metrics["actor_loss"] = float(actor_loss.detach())
            metrics["entropy"] = float(-logp.mean().detach())

        if self.grad_steps % k.target_every == 0:
            self._soft_update()

        self.grad_steps += 1
        return metrics

    @torch.no_grad()
    def _soft_update(self) -> None:
        tau = self.cfg.tau
        for net, tgt in ((self.critic, self.critic_target),
                         (self.encoder, self.encoder_target)):
            for p, pt in zip(net.parameters(), tgt.parameters()):
                pt.mul_(1.0 - tau).add_(tau * p)

    # -- checkpointing -----------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "grad_steps": self.grad_steps,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.encoder.load_state_dict(sd["encoder"])
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.encoder_target = copy.deepcopy(self.encoder)
        self.critic_target = copy.deepcopy(self.critic)
        for p in list(self.critic_target.parameters()) + \
                list(self.encoder_target.parameters()):
            p.requires_grad_(False)
        self.grad_steps = sd.get("grad_steps", 0)
