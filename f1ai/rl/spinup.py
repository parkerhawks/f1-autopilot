"""Bring the car up to racing speed after a mid-lap reset.

Two problems, one fix.

**The car does not start.** "Reset to Track" leaves it stationary, and a policy
trained almost entirely on 200-330 km/h states has no useful behaviour at
0 km/h. It dawdles, the stuck timer fires three seconds later, and the run
collapses into a reset loop that generates no learning at all.

**The speed distribution is wrong.** More subtly damaging: restarting
stationary at Roggia teaches "at Roggia, accelerate from zero", when the
situation the policy must actually handle is "arrive at Roggia at 300 km/h and
brake". Every reset position would be learned with the wrong speed attached,
and the resulting policy would be confidently wrong exactly where it resets
most often.

So after each reset the pedals are driven directly until the car reaches the
speed the demonstration says it should be doing at that point on the circuit,
while STEERING STAYS WITH THE POLICY -- it still has to keep the car on the
road, and those steering decisions are real experience worth keeping.

The transitions are stored as normal. SAC is off-policy, so acting from a
different distribution during spin-up is not merely tolerable, it is the
mechanism that lets these states be learned at all.
"""

from __future__ import annotations

import numpy as np

# Fraction of the reference speed at which control is handed back. Full
# reference is not required: the last few km/h take a long time, and the policy
# is perfectly capable of the final approach.
HANDOVER_FRAC = 0.85

# Ceiling on spin-up length. At full throttle the car reaches 300 km/h in about
# 6 s; anything beyond this means it is stuck against a barrier, and the
# episode should be allowed to terminate normally rather than spinning forever.
MAX_SPINUP_STEPS = 240


class SpinUp:
    """Tracks the post-reset acceleration phase for one episode."""

    def __init__(self, env, enabled: bool = True):
        self.env = env
        self.enabled = enabled
        self.active = False
        self.steps = 0
        self.target_mps = 0.0
        self.total_steps = 0
        self.episodes = 0

    def begin(self, lap_distance: float) -> None:
        if not self.enabled:
            return
        self.target_mps = self.env.arrival_speed_mps(lap_distance)
        self.active = True
        self.steps = 0
        self.episodes += 1

    def action(self, policy_action: np.ndarray, speed_mps: float) -> np.ndarray:
        """Policy steering, forced pedals, until up to speed.

        Returns the policy's own action unchanged once the phase is over, so
        callers can pass everything through here without branching.
        """
        if not self.active:
            return policy_action

        self.steps += 1
        self.total_steps += 1
        if (speed_mps >= self.target_mps * HANDOVER_FRAC
                or self.steps >= MAX_SPINUP_STEPS):
            self.active = False
            return policy_action

        # Ease off as the target approaches so the car settles at racing speed
        # rather than arriving at the next corner already over it.
        deficit = 1.0 - speed_mps / max(1.0, self.target_mps)
        throttle = float(np.clip(deficit * 2.0, 0.35, 1.0))
        return np.array([policy_action[0], throttle, 0.0], np.float32)

    @property
    def mean_steps(self) -> float:
        return self.total_steps / max(1, self.episodes)
