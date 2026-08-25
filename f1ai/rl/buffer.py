"""Replay buffer for pixel observations.

Memory is the binding constraint. A single stacked observation is
3 x 96 x 192 = 55 KB, so storing obs and next_obs as full stacks costs 110 KB
per transition -- 16 GB for a 150k buffer, on a machine with 32 GB total.

Instead only the newest frame is stored per step (18 KB) and stacks are
reconstructed from neighbouring indices, which is 6x cheaper: ~2.8 GB at 150k.
The catch is that reconstruction must respect episode boundaries, or a stack
will silently splice together frames from either side of a reset -- the policy
then trains on transitions that never happened, and nothing crashes. Every slot
carries an episode id so `tools/selftest_buffer.py` can prove reconstruction
matches what the environment actually emitted.
"""

from __future__ import annotations

import threading

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, frame_hw: tuple[int, int],
                 state_dim: int, action_dim: int, stack: int = 3,
                 seed: int | None = None):
        h, w = frame_hw
        self.capacity = capacity
        self.stack = stack

        self.frames = np.zeros((capacity, h, w), np.uint8)
        self.states = np.zeros((capacity, state_dim), np.float32)
        self.actions = np.zeros((capacity, action_dim), np.float32)
        self.rewards = np.zeros(capacity, np.float32)
        self.terminals = np.zeros(capacity, np.bool_)
        # -1 marks a slot that has never been written.
        self.ep_id = np.full(capacity, -1, np.int64)

        self.ptr = 0
        self.size = 0
        self._episode = 0
        self.rng = np.random.default_rng(seed)

        # Against the real game the actor writes on one thread while the
        # learner samples on another. The array element writes are harmless,
        # but ptr and size are not atomic: a sampler that reads a half-updated
        # size can select a slot the writer has not filled yet and train on
        # uninitialised memory. The lock is only held for bookkeeping, never
        # across the frame copies, so it costs the 30 Hz loop nothing.
        self._lock = threading.Lock()

    # -- writing -----------------------------------------------------------

    def start_episode(self) -> None:
        self._episode += 1

    def add(self, frame: np.ndarray, state: np.ndarray, action: np.ndarray,
            reward: float, terminal: bool) -> None:
        """Record the observation BEFORE the action, plus what it produced."""
        i = self.ptr
        self.frames[i] = frame
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminals[i] = terminal
        self.ep_id[i] = self._episode
        # Publish the slot only once it is fully written.
        with self._lock:
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def add_final(self, frame: np.ndarray, state: np.ndarray) -> None:
        """Store an episode's last observation.

        Without this the final transition has no next_obs and must be dropped,
        which throws away exactly the crash and lap-completion states that
        carry the most learning signal.
        """
        self.add(frame, state, np.zeros_like(self.actions[0]), 0.0, False)

    # -- reading -----------------------------------------------------------

    def _stack_indices(self, idx: np.ndarray) -> np.ndarray:
        """(B, stack) frame indices, clamped at episode starts.

        Walking backwards, any index belonging to a different episode (or an
        unwritten slot) repeats the oldest valid frame instead -- the same
        padding the environment does when it fills the stack on reset.
        """
        b = len(idx)
        out = np.empty((b, self.stack), np.int64)
        out[:, -1] = idx
        for k in range(self.stack - 2, -1, -1):
            prev = (out[:, k + 1] - 1) % self.capacity
            same = (self.ep_id[prev] == self.ep_id[idx]) & (self.ep_id[prev] >= 0)
            out[:, k] = np.where(same, prev, out[:, k + 1])
        return out

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Indices whose successor exists and is in the same episode."""
        with self._lock:
            size, ptr = self.size, self.ptr
        if size < self.stack + 2:
            raise ValueError("buffer too small to sample")
        # The slot at ptr-1 is the newest write; its successor is not there yet.
        candidates = np.arange(size)
        nxt = (candidates + 1) % self.capacity
        ok = (self.ep_id[candidates] >= 0) & (self.ep_id[nxt] == self.ep_id[candidates])
        ok &= candidates != (ptr - 1) % self.capacity
        valid = candidates[ok]
        if len(valid) == 0:
            raise ValueError("no valid transitions yet")
        return self.rng.choice(valid, size=batch_size, replace=True)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.sample_indices(batch_size)
        nxt = (idx + 1) % self.capacity
        return {
            "frames": self.frames[self._stack_indices(idx)],
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "terminal": self.terminals[idx],
            "next_frames": self.frames[self._stack_indices(nxt)],
            "next_state": self.states[nxt],
        }

    def stack_at(self, index: int) -> np.ndarray:
        """The stacked observation at one index -- used by the tests."""
        return self.frames[self._stack_indices(np.array([index]))][0]

    # -- persistence -------------------------------------------------------

    def save(self, path) -> None:
        """Write the buffer so it can be trained on offline.

        Online collection is capped at 30 Hz and shares the GPU with the game,
        which held the update-to-data ratio down to about 0.10 -- roughly 38k
        gradient steps for 388k transitions, where pixel-based SAC normally
        wants an order of magnitude more. Persisting the buffer decouples the
        two: experience is gathered at real-time pace, then learned from at
        whatever pace the hardware allows, with nothing else on the GPU.

        Only the written portion is saved, so an early-terminated run does not
        store a hundred thousand blank frames.
        """
        import numpy as _np
        n = self.size
        _np.savez(
            path,
            frames=self.frames[:n], states=self.states[:n],
            actions=self.actions[:n], rewards=self.rewards[:n],
            terminals=self.terminals[:n], ep_id=self.ep_id[:n],
            ptr=_np.int64(self.ptr), size=_np.int64(n),
            episode=_np.int64(self._episode), stack=_np.int64(self.stack),
        )

    @classmethod
    def load(cls, path, capacity: int | None = None,
             seed: int | None = None) -> "ReplayBuffer":
        d = np.load(path)
        n = int(d["size"])
        cap = capacity or n
        h, w = d["frames"].shape[1:]
        buf = cls(cap, (h, w), d["states"].shape[1], d["actions"].shape[1],
                  stack=int(d["stack"]), seed=seed)
        m = min(n, cap)
        buf.frames[:m] = d["frames"][:m]
        buf.states[:m] = d["states"][:m]
        buf.actions[:m] = d["actions"][:m]
        buf.rewards[:m] = d["rewards"][:m]
        buf.terminals[:m] = d["terminals"][:m]
        buf.ep_id[:m] = d["ep_id"][:m]
        buf.size = m
        buf.ptr = m % cap
        buf._episode = int(d["episode"])
        return buf

    @property
    def fill(self) -> float:
        return self.size / self.capacity

    def nbytes(self) -> int:
        return sum(a.nbytes for a in
                   (self.frames, self.states, self.actions, self.rewards,
                    self.terminals, self.ep_id))
