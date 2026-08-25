"""A learner that runs on its own thread, so the actor can hold 30 Hz.

Against the mock this is unnecessary: the simulator steps faster than real time,
so one loop can act and learn in turn. F1 25 cannot be paused for a gradient
step. The car keeps moving whether or not the optimiser has finished, and a
control loop that stalls for 50 ms has simply stopped driving for a metre and a
half at racing speed.

So the two run independently:

    actor thread    fixed 30 Hz: observe -> act -> write to the pad -> store
    learner thread  as fast as the GPU allows: sample -> update

Measured budget on an RTX 4070 with the game rendering:

    actor    ~8 ms of work inside a 33.3 ms slot   (capture 6.4 + inference 1.5)
    learner  ~32 ms per gradient step idle, 50-65 ms contended

which puts the achievable update-to-data ratio near 0.5-0.65 rather than the
1.0 the mock ran at. `updates_per_env_step` reports what actually happened, and
that is the number to trust -- not the one you asked for.

**On sharing weights without a lock.** The actor reads the same networks the
learner is writing, with no synchronisation. This is deliberate. Locking would
push optimiser latency into the control loop, which is exactly what this design
exists to avoid. The cost is that the actor occasionally sees a mix of
pre- and post-update parameters -- equivalent to acting on slightly stale
weights, which is inherent to asynchronous RL anyway and is benign here.

The replay buffer is a different matter and *is* locked: a torn read of its
size/pointer can hand out a slot that was never written, which is silent
corruption rather than harmless staleness.
"""

from __future__ import annotations

import threading
import time

from .buffer import ReplayBuffer
from .sac import SACAgent


class AsyncLearner(threading.Thread):
    """Runs `agent.update()` continuously against a shared replay buffer."""

    def __init__(self, agent: SACAgent, buffer: ReplayBuffer,
                 min_buffer: int = 5000, max_ratio: float = 1.0,
                 target_late_frac: float = 0.02,
                 name: str = "sac-learner"):
        super().__init__(name=name, daemon=True)
        self.agent = agent
        self.buffer = buffer
        self.min_buffer = min_buffer
        # Ceiling on gradient steps per environment step. Without it the
        # learner will happily replay a small buffer hundreds of times per
        # transition early on and overfit to the first few seconds of driving.
        self.max_ratio = max_ratio

        # THE ACTOR ALWAYS WINS.
        #
        # First in-game run: the actor held a clean 30 Hz alone, then dropped to
        # 3-10 Hz the moment the learner started, with 81% of control steps
        # overrunning their slot. Gradient steps had ballooned from 32 ms idle
        # to 243 ms while competing with the game for the GPU, and they were
        # issued back to back.
        #
        # That is not merely slow, it is fatal: at 3-10 Hz the timestep varies
        # threefold between transitions, so the same action has different
        # consequences depending on scheduling. The MDP stops being stationary
        # and the critic diverges trying to fit it -- which is exactly what the
        # loss did.
        #
        # So the learner watches how often the actor misses its deadline and
        # backs off until it stops. Fewer, timely gradient steps beat more
        # gradient steps on corrupted data.
        self.target_late_frac = target_late_frac
        self._late_frac = 0.0
        self._backoff = 0.0        # seconds of sleep after each update

        # NOT `_stop`: threading.Thread has a private _stop() that join() calls
        # internally, and shadowing it with an Event makes join() raise
        # "'Event' object is not callable" at shutdown.
        self._stop_event = threading.Event()
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float] = {}
        self._env_steps = 0
        self.grad_steps = 0
        self.idle_seconds = 0.0
        self.update_ms: float = 0.0

    # -- called from the actor thread --------------------------------------

    def note_env_step(self, total: int) -> None:
        """Tell the learner how much data exists, for the ratio ceiling."""
        self._env_steps = total

    def note_actor_late(self, late_frac: float) -> None:
        """Report the actor's recent deadline-miss rate, 0..1.

        Drives the backoff controller. Called from the actor thread; a plain
        float assignment is atomic enough for a signal that is only ever used
        as a hint.
        """
        self._late_frac = late_frac

    @property
    def backoff_ms(self) -> float:
        return self._backoff * 1000.0

    @property
    def metrics(self) -> dict[str, float]:
        with self._metrics_lock:
            return dict(self._metrics)

    @property
    def updates_per_env_step(self) -> float:
        return self.grad_steps / max(1, self._env_steps)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)

    # -- learner thread ----------------------------------------------------

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self.buffer.size < self.min_buffer:
                time.sleep(0.05)
                continue

            # Respect the ratio ceiling rather than spinning on a hot buffer.
            allowed = (self._env_steps - self.min_buffer) * self.max_ratio
            if self.grad_steps >= allowed:
                t0 = time.perf_counter()
                time.sleep(0.002)
                self.idle_seconds += time.perf_counter() - t0
                continue

            try:
                batch = self.buffer.sample(self.agent.cfg.batch_size)
            except ValueError:
                time.sleep(0.02)
                continue

            t0 = time.perf_counter()
            m = self.agent.update(batch)
            dt = (time.perf_counter() - t0) * 1000.0

            self.grad_steps += 1
            # Exponential average: one slow step during a menu animation should
            # not dominate the reported cost.
            self.update_ms = (0.9 * self.update_ms + 0.1 * dt
                              if self.update_ms else dt)
            with self._metrics_lock:
                self._metrics.update(m)

            self._adjust_backoff(dt / 1000.0)
            if self._backoff > 0.0:
                time.sleep(self._backoff)

    def _adjust_backoff(self, update_seconds: float) -> None:
        """Proportional controller on the actor's deadline-miss rate.

        Backoff is expressed as a multiple of one update's duration, so it
        adapts automatically when the GPU gets busier -- a slower gradient step
        earns proportionally more yielding rather than needing a retuned
        constant.
        """
        error = self._late_frac - self.target_late_frac
        if error > 0:
            # Missing deadlines: yield more, quickly.
            self._backoff = min(self._backoff + 0.5 * update_seconds,
                                8.0 * update_seconds)
        elif self._backoff > 0.0:
            # Comfortable: creep back toward full speed, slowly, so the
            # controller does not oscillate between starving and idling.
            self._backoff = max(0.0, self._backoff - 0.1 * update_seconds)
