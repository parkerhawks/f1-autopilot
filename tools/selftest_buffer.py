"""Prove the replay buffer reconstructs exactly what the environment emitted.

The buffer stores one frame per step and rebuilds stacks from neighbouring
indices to save 6x memory. If that reconstruction is wrong -- most likely by
splicing frames across an episode reset -- the agent trains on observations
that never occurred. Nothing raises, nothing looks unusual in the loss curves,
and the policy just quietly fails to learn.

So every stack the buffer returns is compared byte-for-byte against the stack
the environment produced at that same step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.buffer import ReplayBuffer  # noqa: E402
from f1ai.rl.env import (  # noqa: E402
    FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM, MockBackend, RacingEnv,
)

CAPACITY = 4000
STEPS = 2600


def main() -> None:
    env = RacingEnv(MockBackend(seed=3))
    buf = ReplayBuffer(CAPACITY, (FRAME_H, FRAME_W), STATE_DIM, 3,
                       stack=FRAME_STACK, seed=0)
    rng = np.random.default_rng(0)

    # Ground truth: what the environment actually showed, per buffer slot.
    truth: dict[int, np.ndarray] = {}
    episodes = 0

    frames, state = env.reset()
    buf.start_episode()
    episodes += 1

    for _ in range(STEPS):
        action = np.array([rng.uniform(-1, 1), rng.uniform(0.2, 1.0),
                           rng.uniform(0, 0.4)], np.float32)
        slot = buf.ptr
        buf.add(frames[-1], state, action, 0.0, False)
        truth[slot] = frames.copy()

        res = env.step(action)
        frames, state = res.frames, res.state

        if res.terminated or res.truncated:
            slot = buf.ptr
            buf.add_final(frames[-1], state)
            truth[slot] = frames.copy()
            frames, state = env.reset()
            buf.start_episode()
            episodes += 1

    # -- reconstruction fidelity -----------------------------------------
    checked = mismatched = 0
    worst = None
    for slot, expected in truth.items():
        if buf.ep_id[slot] < 0:
            continue                      # overwritten by wraparound
        got = buf.stack_at(slot)
        checked += 1
        if not np.array_equal(got, expected):
            mismatched += 1
            if worst is None:
                worst = slot

    # -- episode boundaries are never crossed ----------------------------
    idx = buf.sample_indices(2048)
    nxt = (idx + 1) % buf.capacity
    same_episode = bool(np.all(buf.ep_id[idx] == buf.ep_id[nxt]))

    stacks = buf._stack_indices(idx)
    stack_eps = buf.ep_id[stacks]
    stacks_clean = bool(np.all(stack_eps == buf.ep_id[idx][:, None]))

    batch = buf.sample(256)
    shapes_ok = (
        batch["frames"].shape == (256, FRAME_STACK, FRAME_H, FRAME_W)
        and batch["next_frames"].shape == (256, FRAME_STACK, FRAME_H, FRAME_W)
        and batch["state"].shape == (256, STATE_DIM)
        and batch["action"].shape == (256, 3)
    )

    # next_obs must be the successor stack, not a copy of obs.
    shifted = np.mean([
        not np.array_equal(batch["frames"][k], batch["next_frames"][k])
        for k in range(256)
    ])

    print(f"buffer: {buf.size} slots, {episodes} episodes, "
          f"{buf.nbytes() / 1e6:.0f} MB for {CAPACITY} capacity")
    print(f"projected at 150k capacity: "
          f"{buf.nbytes() / CAPACITY * 150_000 / 1e9:.2f} GB\n")

    checks = [
        ("stacks match environment", mismatched == 0,
         f"{checked} checked, {mismatched} wrong"
         + (f" (first at slot {worst})" if worst is not None else "")),
        ("transitions stay in one episode", same_episode, ""),
        ("stacks never cross a reset", stacks_clean, ""),
        ("batch shapes correct", shapes_ok, ""),
        ("next_obs differs from obs", shifted > 0.95,
         f"{shifted:.1%} of sampled pairs differ"),
    ]

    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} {detail}")
        failed += not ok

    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- do not train on this buffer")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- reconstruction is faithful")


if __name__ == "__main__":
    main()
