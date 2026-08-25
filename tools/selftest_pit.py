"""Prove pit detection fires on the FIRST flagged step, and only once.

Two properties, both of which cost a run when they were missing.

**Immediacy.** Pit detection used to wait out the 30-step invalid-lap grace
period -- a full second, which at pit-entry speed is deep inside the pit lane.
By then F1 25 has taken control and put up its own prompt, and a reset driven
into that prompt wedges the game in a menu it never leaves. Caught on the first
flagged step the car is still on the racing surface and the ordinary pause menu
still works.

**Edge-triggering.** If an episode begins in the pit area, terminating on the
level would end it instantly, trigger another reset, and loop -- the same shape
of failure as the invalid-lap loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import MockBackend, RacingEnv  # noqa: E402
from f1ai.sim.expert import PurePursuitExpert  # noqa: E402

_expert = None


class PitBackend(MockBackend):
    """Mock whose pitStatus can be driven from the test."""

    def __init__(self, pit: int = 0, **kw):
        self.pit = pit
        super().__init__(**kw)

    def observe(self):
        obs = super().observe()
        obs.pit_status = self.pit
        return obs


def drive(env) -> np.ndarray:
    s, t, b = _expert.act(env.backend.veh)
    return np.array([s, t, b], np.float32)


def main() -> None:
    global _expert
    checks: list[tuple[str, bool, str]] = []

    # -- detection must be immediate, not after a grace period ------------
    env = RacingEnv(PitBackend(pit=0, seed=0))
    _expert = PurePursuitExpert(env.backend.track)
    env.reset()
    for _ in range(12):
        env.step(drive(env))

    env.backend.pit = 2                     # the car crosses the pit entry
    res = env.step(drive(env))
    checks.append(("pit entry terminates on the first flagged step",
                   res.terminated and res.info["term_reason"] == "pit lane",
                   f"reason: {res.info.get('term_reason')}"))
    checks.append(("pit status is reported for diagnosis",
                   res.info.get("pit_status") == 2,
                   f"pit_status={res.info.get('pit_status')}"))

    # -- and it must not depend on how far into the episode we are --------
    env2 = RacingEnv(PitBackend(pit=0, seed=0))
    _expert = PurePursuitExpert(env2.backend.track)
    env2.reset()
    env2.backend.pit = 1                    # flagged on the very next step
    res2 = env2.step(drive(env2))
    checks.append(("fires even immediately after a reset",
                   res2.terminated and res2.info["term_reason"] == "pit lane",
                   f"at step 1, reason: {res2.info.get('term_reason')}"))

    # -- an episode that BEGINS in the pit area must not loop -------------
    env3 = RacingEnv(PitBackend(pit=2, seed=0))
    _expert = PurePursuitExpert(env3.backend.track)
    env3.reset()
    early = [env3.step(drive(env3)) for _ in range(40)]
    from_pit = [r for r in early if r.info.get("term_reason") == "pit lane"]
    checks.append(("starting in the pit area does not loop",
                   len(from_pit) == 0,
                   f"{len(from_pit)} pit terminations in 40 steps"))

    # -- leaving and re-entering must fire again --------------------------
    env4 = RacingEnv(PitBackend(pit=2, seed=0))
    _expert = PurePursuitExpert(env4.backend.track)
    env4.reset()
    env4.backend.pit = 0                    # rejoins the track
    for _ in range(10):
        env4.step(drive(env4))
    env4.backend.pit = 1                    # and enters the pits again
    res4 = env4.step(drive(env4))
    checks.append(("re-entering the pits fires again",
                   res4.terminated and res4.info["term_reason"] == "pit lane",
                   f"reason: {res4.info.get('term_reason')}"))

    # -- driving down the pit lane must earn NOTHING ----------------------
    # The reason the agent kept going there. lapDistance climbs in the pit
    # lane, so an unguarded progress reward pays for it at the same rate as
    # the racing line -- and a terminal penalty cannot claw back metres that
    # were already banked on the way in.
    def progress_reward(pit: int) -> float:
        global _expert
        e = RacingEnv(PitBackend(pit=0, seed=0))
        _expert = PurePursuitExpert(e.backend.track)
        e.reset()
        for _ in range(20):
            e.step(drive(e))
        e.backend.pit = pit
        # Read the reward on the step where the flag is already established,
        # so the one-off termination penalty is not what is being measured.
        e._was_in_pit = pit != 0
        return e.step(drive(e)).reward

    on_track = progress_reward(0)
    in_pit = progress_reward(2)
    checks.append(("progress on track pays", on_track > 0.5,
                   f"{on_track:+.2f} per step"))
    checks.append(("progress in the pit lane pays nothing",
                   in_pit <= 0.01,
                   f"{in_pit:+.2f} per step vs {on_track:+.2f} on track"))

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} {detail}")
        failed += not ok
    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- the game will wedge in a menu")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- pit entry is caught immediately")


if __name__ == "__main__":
    main()
