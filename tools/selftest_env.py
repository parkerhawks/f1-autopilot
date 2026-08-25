"""Validate the environment and, more importantly, the reward function.

Before any training run, the reward must demonstrably rank driving quality:
expert > cautious > random. A reward that fails this ordering will be optimised
faithfully by SAC into a policy that does something useless, and the training
curve will look perfectly healthy while it happens. This is the cheapest test
in the project and it catches the most expensive class of mistake.

Also checks the observation contract: frames are the right shape, and the state
vector carries no privileged track information.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import (  # noqa: E402
    CONTROL_HZ, FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM,
    MockBackend, RacingEnv,
)
from f1ai.sim.expert import PurePursuitExpert  # noqa: E402

EPISODE_STEPS = int(90 * CONTROL_HZ)


def run(env: RacingEnv, policy, steps: int, seed: int) -> dict:
    frames, state = env.reset()
    total, n, crashed = 0.0, 0, False
    for _ in range(steps):
        a = policy(env, frames, state)
        r = env.step(a)
        total += r.reward
        n += 1
        frames, state = r.frames, r.state
        if r.terminated:
            crashed = True
            frames, state = env.reset()
    return {
        "return": total,
        "return_per_step": total / max(1, n),
        "distance": r.info["distance"],
        "laps": len(env.lap_times),
        "best_lap": min(env.lap_times) if env.lap_times else None,
        "crashed": crashed,
    }


def expert_policy(env, frames, state):
    """Privileged controller, used only as a scoring reference."""
    e = expert_policy.expert
    steer, throttle, brake = e.act(env.backend.veh)
    return np.array([steer, throttle, brake], np.float32)


def cautious_policy(env, frames, state):
    """Drives straight at a modest speed -- survives briefly, makes no progress."""
    return np.array([0.0, 0.25, 0.0], np.float32)


def make_random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def policy(env, frames, state):
        return np.array([rng.uniform(-1, 1), rng.uniform(0, 1),
                         rng.uniform(0, 0.3)], np.float32)
    return policy


def main() -> None:
    backend = MockBackend(seed=0)
    env = RacingEnv(backend)
    expert_policy.expert = PurePursuitExpert(backend.track)

    frames, state = env.reset()
    print("observation contract")
    print(f"  frames {frames.shape} {frames.dtype}  "
          f"expected ({FRAME_STACK}, {FRAME_H}, {FRAME_W}) uint8")
    print(f"  state  {state.shape} {state.dtype}  expected ({STATE_DIM},) float32")
    shape_ok = (frames.shape == (FRAME_STACK, FRAME_H, FRAME_W)
                and frames.dtype == np.uint8
                and state.shape == (STATE_DIM,))

    # The state vector must not encode lateral offset. Sample states at very
    # different offsets and confirm the vector does not track it.
    obs_a = backend.observe()
    offsets, vectors = [], []
    for _ in range(60):
        backend.reset()
        o = backend.observe()
        offsets.append(o.lateral_offset)
        vectors.append(env._state_vector(o))
    # Constant columns have zero variance and yield NaN correlations; those
    # carry no offset information by definition, so treat them as 0.
    vecs = np.array(vectors)
    corr = 0.0
    for k in range(STATE_DIM):
        col = vecs[:, k]
        if np.std(col) < 1e-9:
            continue
        corr = max(corr, abs(np.corrcoef(offsets, col)[0, 1]))
    leak_ok = corr < 0.5
    print(f"  max |corr| between state vector and lateral offset: {corr:.3f}")
    print(f"  -> {'no privileged leak' if leak_ok else 'LEAK: offset is observable'}\n")

    print(f"scoring three policies over {EPISODE_STEPS} steps "
          f"({EPISODE_STEPS / CONTROL_HZ:.0f}s)\n")
    results = {}
    for name, pol in (
        ("expert", expert_policy),
        ("cautious", cautious_policy),
        ("random", make_random_policy(1)),
    ):
        env2 = RacingEnv(MockBackend(seed=0))
        expert_policy.expert = PurePursuitExpert(env2.backend.track)
        results[name] = run(env2, pol, EPISODE_STEPS, seed=0)

    print(f"{'policy':<10} {'return':>10} {'per step':>10} {'distance':>10} "
          f"{'laps':>5} {'best lap':>9} {'crashed':>8}")
    print("-" * 70)
    for name, r in results.items():
        bl = f"{r['best_lap']:.2f}s" if r["best_lap"] else "-"
        print(f"{name:<10} {r['return']:>10.0f} {r['return_per_step']:>10.3f} "
              f"{r['distance']:>10.0f} {r['laps']:>5} {bl:>9} "
              f"{str(r['crashed']):>8}")

    print()
    checks = [
        ("observation shapes", shape_ok, ""),
        ("no privileged leak", leak_ok, f"max corr {corr:.3f}"),
        ("expert beats cautious",
         results["expert"]["return"] > results["cautious"]["return"],
         f"{results['expert']['return']:.0f} vs {results['cautious']['return']:.0f}"),
        ("expert beats random",
         results["expert"]["return"] > results["random"]["return"],
         f"{results['expert']['return']:.0f} vs {results['random']['return']:.0f}"),
        ("cautious beats random",
         results["cautious"]["return"] > results["random"]["return"],
         f"{results['cautious']['return']:.0f} vs {results['random']['return']:.0f}"),
        ("expert completes laps", results["expert"]["laps"] >= 2,
         f"{results['expert']['laps']} laps"),
        ("expert never crashes", not results["expert"]["crashed"], ""),
        ("random does crash", results["random"]["crashed"], ""),
    ]

    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<24} {detail}")
        failed += not ok

    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- do not start training on this reward")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- reward ranks driving quality correctly")


if __name__ == "__main__":
    main()
