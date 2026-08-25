"""Prove the post-reset spin-up gets the car moving, at the right speed.

Both failure modes it guards against are silent.

If the car never starts, the stuck timer fires three seconds after every reset
and training becomes an endless reset loop -- steps accumulate, wall clock
burns, and nothing is learned. The step counter still rises, so it looks busy.

If it starts but at the wrong speed, the damage is worse and slower to notice:
the policy learns "at Roggia, accelerate from zero" instead of "arrive at
Roggia at 300 km/h and brake". That trains a confidently wrong response at
exactly the positions the car resets at most often.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import MockBackend, RacingEnv  # noqa: E402
from f1ai.rl.spinup import HANDOVER_FRAC, MAX_SPINUP_STEPS, SpinUp  # noqa: E402
from f1ai.rl.track_map import TrackMap  # noqa: E402


def main() -> None:
    checks: list[tuple[str, bool, str]] = []
    from f1ai.rl.track_map import find_any_map
    mp = find_any_map()
    if mp is None:
        print("no map in maps/ -- build one with tools/build_map.py")
        print("(this suite needs a real track map and is skipped without one)")
        sys.exit(0)

    tmap = TrackMap.load(mp)
    env = RacingEnv(MockBackend(seed=0), track_map=tmap)
    print(f"using {mp.name} ({tmap.length:.0f} m)")

    # -- the target must match the demonstrated profile -------------------
    # Positions are taken from the map itself rather than from a list of named
    # corners, so this works on whatever circuit was recorded.
    fastest = int(np.argmax(tmap.ref_speed))
    slowest = int(np.argmin(tmap.ref_speed))
    straight = env.arrival_speed_mps(float(tmap.s[fastest]))
    corner = env.arrival_speed_mps(float(tmap.s[slowest]))
    print(f"  fastest point {straight * 3.6:6.1f} km/h "
          f"at {tmap.s[fastest]:.0f} m")
    print(f"  slowest point {corner * 3.6:6.1f} km/h "
          f"at {tmap.s[slowest]:.0f} m")

    for idx in (fastest, slowest, len(tmap.s) // 2):
        s = float(tmap.s[idx])
        if abs(env.arrival_speed_mps(s)
               - float(np.interp(s, tmap.s, tmap.ref_speed))) > 1e-3:
            checks.append((f"target matches profile at {s:.0f} m", False, ""))

    checks.append(("target is position-dependent", straight > corner * 1.5,
                   f"{straight * 3.6:.0f} km/h at the fastest point vs "
                   f"{corner * 3.6:.0f} at the slowest"))

    # -- a stationary car gets throttle -----------------------------------
    spin = SpinUp(env)
    spin.begin(float(tmap.s[fastest]))
    idle_policy = np.array([0.1, 0.0, 0.0], np.float32)   # policy asks for none
    out = spin.action(idle_policy, speed_mps=0.0)
    checks.append(("stationary car is given throttle", out[1] > 0.3,
                   f"throttle {out[1]:.2f} where the policy asked {idle_policy[1]:.2f}"))
    checks.append(("brake is released during spin-up", out[2] == 0.0, ""))
    checks.append(("policy keeps the steering", abs(out[0] - 0.1) < 1e-6,
                   "steering passed through unchanged"))

    # -- and control is handed back once up to speed ----------------------
    spin2 = SpinUp(env)
    spin2.begin(float(tmap.s[fastest]))
    target = env.arrival_speed_mps(float(tmap.s[fastest]))
    handed = None
    for k in range(MAX_SPINUP_STEPS + 5):
        speed = min(target, 5.0 + k * 2.0)
        act = spin2.action(idle_policy, speed)
        if not spin2.active:
            handed = (k, speed)
            break
    checks.append(("hands control back at speed", handed is not None,
                   f"after {handed[0]} steps at {handed[1] * 3.6:.0f} km/h"
                   if handed else "never handed back"))
    if handed:
        checks.append(("handover near the reference speed",
                       handed[1] >= target * HANDOVER_FRAC * 0.98,
                       f"{handed[1] * 3.6:.0f} vs target {target * 3.6:.0f} km/h"))

    # -- a car that cannot accelerate must not spin forever ---------------
    spin3 = SpinUp(env)
    spin3.begin(float(tmap.s[fastest]))
    for _ in range(MAX_SPINUP_STEPS + 10):
        spin3.action(idle_policy, 0.0)      # wedged against a barrier
    checks.append(("gives up when the car cannot move", not spin3.active,
                   f"stopped after {MAX_SPINUP_STEPS} steps "
                   f"({MAX_SPINUP_STEPS / 30:.0f}s)"))

    # -- disabled on the mock, which resets at speed already --------------
    off = SpinUp(env, enabled=False)
    off.begin(float(tmap.s[fastest]))
    checks.append(("no-op when disabled",
                   not off.active
                   and np.array_equal(off.action(idle_policy, 0.0), idle_policy),
                   ""))

    # -- the spin-up target must come from AFTER the reset -----------------
    # It used to be taken from where the previous episode ended. With Restart
    # Lap the car goes back to the line, so a crash at the chicane set a
    # 74 km/h target for a car sitting on the main straight -- it handed over
    # at a crawl, trundled to the chicane and crashed again. Average speed sat
    # at 114 km/h and every evaluation died at the same 1000 m.
    env_r = RacingEnv(MockBackend(seed=5), track_map=tmap)
    env_r.reset()
    checks.append(("env reports the post-reset position",
                   hasattr(env_r, "reset_lap_distance"),
                   f"reset_lap_distance = {getattr(env_r, 'reset_lap_distance', None)}"))

    # -- speed reward: faster on track must pay more ----------------------
    def reward_at(speed_mps: float) -> float:
        e = RacingEnv(MockBackend(seed=5), track_map=tmap)
        e.reset()
        e.backend.veh.state.v = speed_mps
        return e.step(np.array([0.0, 0.5, 0.0], np.float32)).reward

    slow, fast = reward_at(25.0), reward_at(75.0)
    checks.append(("faster on track pays more", fast > slow,
                   f"{slow:+.2f} at 90 km/h vs {fast:+.2f} at 270 km/h"))

    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")
        failed += not ok
    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- resets will loop or teach "
              f"the wrong speeds")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- resets resume at racing speed")


if __name__ == "__main__":
    main()
