"""Prove that resuming onto an already-invalid lap does not loop forever.

The failure this guards against, in order:

  1. the car runs wide and the game invalidates the lap
  2. the episode ends, correctly
  3. "Reset to Track" puts the car back -- and leaves the lap invalidated
  4. the new episode starts on an invalid lap
  5. the invalid flag reads as a fresh rising edge and ends the episode
  6. go to 3

Nothing raises. Steps accumulate, resets fire, the counter climbs and the run
looks busy all night while learning precisely nothing.

The fix is to seed the edge detector from what the game currently reports at
reset rather than from False, plus a short grace period for a flag that lags
the reset by a packet or two.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import (  # noqa: E402
    INVALID_GRACE_STEPS, MockBackend, RacingEnv,
)
from f1ai.sim.expert import PurePursuitExpert  # noqa: E402

_expert: PurePursuitExpert | None = None


class InvalidLapBackend(MockBackend):
    """Mock whose lap is permanently invalid, as after a Reset to Track."""

    def __init__(self, invalid: bool, **kw):
        self._invalid = invalid
        super().__init__(**kw)

    def observe(self):
        obs = super().observe()
        obs.lap_invalid = self._invalid
        return obs


def drive(env: RacingEnv) -> np.ndarray:
    """Expert action, so the car stays on the road.

    Steering straight sends the mock off the track within twenty steps, and
    `offtrack` then fires long before the invalid-lap logic is reached -- which
    tests nothing about the loop this file exists to catch.
    """
    steer, throttle, brake = _expert.act(env.backend.veh)
    return np.array([steer, throttle, brake], np.float32)


def run_episodes(env: RacingEnv, n_steps: int) -> tuple[list[int], list[str]]:
    """Episode lengths and termination reasons over a fixed number of steps."""
    env.reset()
    lengths, reasons, current = [], [], 0
    for _ in range(n_steps):
        res = env.step(drive(env))
        current += 1
        if res.terminated or res.truncated:
            lengths.append(current)
            reasons.append(res.info.get("term_reason") or "truncated")
            current = 0
            env.reset()
    return lengths, reasons


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    # -- an already-invalid lap must NOT terminate at all -----------------
    env = RacingEnv(InvalidLapBackend(invalid=True, seed=0))
    global _expert
    _expert = PurePursuitExpert(env.backend.track)
    lengths, reasons = run_episodes(env, 1200)
    from_invalid = [r for r in reasons if r == "lap invalidated"]
    checks.append(("resuming onto an invalid lap never re-fires",
                   len(from_invalid) == 0,
                   f"{len(lengths)} episode ends, "
                   f"{len(from_invalid)} blamed on invalidation"))
    checks.append(("no instant terminations",
                   all(n > INVALID_GRACE_STEPS for n in lengths) if lengths
                   else True,
                   f"shortest {min(lengths) if lengths else '-'} steps"))

    # -- but a genuine mid-lap invalidation still terminates --------------
    env2 = RacingEnv(InvalidLapBackend(invalid=False, seed=0))
    _expert = PurePursuitExpert(env2.backend.track)
    env2.reset()
    for _ in range(INVALID_GRACE_STEPS + 10):
        env2.step(drive(env2))
    env2.backend._invalid = True            # the car runs wide, mid-lap
    res = env2.step(drive(env2))
    checks.append(("a genuine mid-lap invalidation terminates",
                   res.terminated
                   and res.info.get("term_reason") == "lap invalidated",
                   f"reason: {res.info.get('term_reason')}"))

    # -- the grace period must protect the opening steps ------------------
    env3 = RacingEnv(InvalidLapBackend(invalid=False, seed=0))
    _expert = PurePursuitExpert(env3.backend.track)
    env3.reset()
    env3.backend._invalid = True            # flag arrives just after the reset
    early = [env3.step(drive(env3)) for _ in range(INVALID_GRACE_STEPS - 2)]
    checks.append(("a lagging flag cannot end the opening steps",
                   not any(r.info.get("term_reason") == "lap invalidated"
                           for r in early),
                   f"survived {len(early)} steps inside the grace window"))

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} {detail}")
        failed += not ok
    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- an overnight run would loop")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- no reset loop")


if __name__ == "__main__":
    main()
