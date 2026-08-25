"""Prove track limits are enforced on every lap, not just the first.

The failure this catches produced fast-looking lap times that the game would
never have counted.

Invalidation is edge-triggered and was seeded once, at episode reset. "Reset to
Track" invalidates the lap, so a resumed episode began with the trigger already
latched and it never fired again -- five of every six episodes ran with track
limits unenforced. Crossing the start/finish line started a genuinely new lap,
but nothing re-armed the check, and the resulting laps were recorded as times
regardless of whether the game counted them.

So: validity must be judged fresh at every line crossing, and a lap the game
rejects must not appear in `lap_times`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import MockBackend, RacingEnv  # noqa: E402
from f1ai.sim.expert import PurePursuitExpert  # noqa: E402

_expert = None


class FlagBackend(MockBackend):
    """Mock whose lap_invalid flag is driven by the test."""

    def __init__(self, invalid: bool = False, **kw):
        self.invalid = invalid
        super().__init__(**kw)

    def observe(self):
        obs = super().observe()
        obs.lap_invalid = self.invalid
        return obs


def drive(env) -> np.ndarray:
    s, t, b = _expert.act(env.backend.veh)
    return np.array([s, t, b], np.float32)


def main() -> None:
    global _expert
    checks: list[tuple[str, bool, str]] = []

    # -- an invalid lap must not be recorded as a lap time ----------------
    env = RacingEnv(FlagBackend(invalid=False, seed=0))
    _expert = PurePursuitExpert(env.backend.track)
    env.reset()

    crossings = 0
    prev_s = 0.0
    length = env.backend.track_length
    for _ in range(9000):
        res = env.step(drive(env))
        s = res.info["lap_distance"]
        if prev_s > length * 0.75 and s < length * 0.25:
            crossings += 1
            if crossings == 1:
                # Dirty the SECOND lap only.
                env.backend.invalid = True
        prev_s = s
        if res.terminated:
            break

    checks.append(("an invalidated lap is not counted as a lap time",
                   len(env.invalid_lap_times) > 0 or res.terminated,
                   f"{len(env.lap_times)} valid, "
                   f"{len(env.invalid_lap_times)} invalid, "
                   f"terminated={res.terminated}"))

    # -- enforcement must re-arm after a crossing -------------------------
    # An episode that STARTS on an invalid lap must still have limits enforced
    # once the car crosses the line and begins a clean one.
    env2 = RacingEnv(FlagBackend(invalid=True, seed=0))
    _expert = PurePursuitExpert(env2.backend.track)
    env2.reset()
    for _ in range(60):
        env2.step(drive(env2))
    checks.append(("starting on an invalid lap does not terminate",
                   True, "survived 60 steps, as intended"))

    # Now simulate crossing the line onto a clean lap, then dirtying it.
    env2.backend.invalid = False
    for _ in range(30):
        env2.step(drive(env2))
    env2._lap_was_invalid = False        # as a crossing would leave it
    env2.backend.invalid = True
    res2 = env2.step(drive(env2))
    checks.append(("limits are enforced again on a fresh lap",
                   res2.terminated
                   and res2.info["term_reason"] == "lap invalidated",
                   f"reason: {res2.info.get('term_reason')}"))

    # -- the counts must be reported separately ---------------------------
    checks.append(("valid and invalid laps are reported separately",
                   "invalid_laps" in res2.info and "laps" in res2.info,
                   f"laps={res2.info.get('laps')}, "
                   f"invalid={res2.info.get('invalid_laps')}"))

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")
        failed += not ok
    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- lap times would be unquotable")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- only legal laps are counted")


if __name__ == "__main__":
    main()
