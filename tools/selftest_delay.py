"""Prove that injected actuation delay actually reaches the vehicle.

`--delay-ms` exists to rehearse F1 25's measured lag offline, and a run made
with it silently ignored would be worthless while looking like evidence. So
this checks the plumbing end to end: command a step change, and confirm the
APPLIED controls lag it by exactly the configured number of steps, and that
the observation carries both the lagged applied value and the commanded
history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import (  # noqa: E402
    ACTION_HISTORY, CONTROL_HZ, IDX_APPLIED, IDX_HISTORY, IDX_MAP,
    STATE_DIM, MockBackend, RacingEnv,
)

DELAY_MS = 100.0
DELAY_STEPS = int(round(DELAY_MS / 1000.0 * CONTROL_HZ))


def main() -> None:
    checks: list[tuple[str, bool, str]] = []
    print(f"{DELAY_MS:.0f} ms at {CONTROL_HZ:g} Hz = {DELAY_STEPS} control steps\n")

    backend = MockBackend(seed=0, actuation_delay_steps=DELAY_STEPS)
    env = RacingEnv(backend)
    env.reset()

    # Hold zero steering, then command hard left and watch when it lands.
    for _ in range(6):
        env.step(np.array([0.0, 0.4, 0.0], np.float32))

    applied = []
    for k in range(DELAY_STEPS + 3):
        res = env.step(np.array([-1.0, 0.4, 0.0], np.float32))
        applied.append(backend.veh.last_applied[0])
        print(f"  step {k}: commanded -1.00, applied {applied[-1]:+.2f}")

    # The first DELAY_STEPS applications must still be the old command.
    lag_ok = all(abs(v) < 1e-6 for v in applied[:DELAY_STEPS])
    landed = applied[DELAY_STEPS] if len(applied) > DELAY_STEPS else 0.0
    checks.append(("applied lags by exactly the configured steps",
                   lag_ok and landed < -0.99,
                   f"zero for {DELAY_STEPS} steps, then {landed:+.2f}"))

    # Zero delay must NOT lag, or the test above proves nothing.
    b0 = MockBackend(seed=0, actuation_delay_steps=0)
    e0 = RacingEnv(b0)
    e0.reset()
    e0.step(np.array([-1.0, 0.4, 0.0], np.float32))
    checks.append(("zero delay applies immediately",
                   b0.veh.last_applied[0] < -0.99,
                   f"applied {b0.veh.last_applied[0]:+.2f}"))

    # The observation must expose both the lagged actuator state and the
    # commands still in flight, or the problem is not Markovian.
    obs = backend.observe()
    vec = env._state_vector(obs)
    checks.append(("state vector has the right width",
                   vec.shape == (STATE_DIM,), f"{vec.shape} vs ({STATE_DIM},)"))
    checks.append(("applied controls are in the observation",
                   abs(vec[IDX_APPLIED][0] - obs.applied_steer) < 1e-6,
                   f"applied_steer {obs.applied_steer:+.2f}"))
    # The simulator has no recorded map, so its preview block must be zeros --
    # if it is not, the two worlds disagree about the observation layout.
    checks.append(("mock leaves the map block zeroed",
                   bool(np.all(vec[IDX_MAP] == 0.0)),
                   f"{len(vec[IDX_MAP])} map features"))

    hist = vec[IDX_HISTORY].reshape(ACTION_HISTORY, 3)
    checks.append((f"{ACTION_HISTORY} commands of history carried",
                   hist.shape == (ACTION_HISTORY, 3)
                   and abs(hist[-1][0] + 1.0) < 1e-6,
                   f"newest command {hist[-1][0]:+.2f}"))

    # History must cover the delay, otherwise in-flight actions are invisible.
    checks.append(("history covers the actuation delay",
                   ACTION_HISTORY >= DELAY_STEPS,
                   f"{ACTION_HISTORY} steps of history vs {DELAY_STEPS} of delay"))

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")
        failed += not ok

    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- --delay-ms runs are not "
              f"testing what they claim to")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- injected delay is real and observable")


if __name__ == "__main__":
    main()
