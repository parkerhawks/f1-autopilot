"""Kinematic bicycle model -- the car inside the mock game.

This is deliberately *not* a good F1 physics model.  Its job is to produce
plausible, causally-correct telemetry so the data pipeline, the control
abstraction and the training loop can all be built and tested before the real
game is involved.  Nothing learned against this model is expected to transfer;
what transfers is the code that surrounds it.

The one property that is faithful on purpose is the actuation delay, because
that is the thing the dataset construction has to compensate for.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

# Roughly F1-shaped numbers. Not accurate, just not absurd.
WHEELBASE = 3.6          # m
MAX_STEER_RAD = 0.36     # ~20 deg at the wheel
MAX_ACCEL = 14.0         # m/s^2, traction-limited launch
MAX_BRAKE = 45.0         # m/s^2, F1 cars really do brake this hard
DRAG_K = 0.0016          # v^2 drag coefficient lumped
ROLL_RESIST = 0.6        # m/s^2
V_MAX = 92.0             # m/s, ~330 km/h
STEER_FALLOFF_V = 45.0   # m/s at which steering authority is halved


def effective_max_steer(v: float) -> float:
    """Steering angle available at speed v, in radians.

    Authority falls off with speed so the model cannot pivot on the spot at
    300 km/h. Any controller must divide by *this*, not by MAX_STEER_RAD --
    using the static limit understeers by ~5x on the straights, which looks
    like a tracking bug and is really a units bug.
    """
    return MAX_STEER_RAD / (1.0 + (v / STEER_FALLOFF_V) ** 2)


@dataclass
class VehicleState:
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0         # radians, 0 = +Z, matching F1's convention
    v: float = 0.0           # m/s along the body axis
    yaw_rate: float = 0.0
    accel_lon: float = 0.0
    accel_lat: float = 0.0


@dataclass
class Vehicle:
    """Bicycle model with a FIFO actuation delay on the control inputs."""

    state: VehicleState = field(default_factory=VehicleState)
    delay_steps: int = 0
    last_applied: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _queue: deque = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        for _ in range(self.delay_steps):
            self._queue.append((0.0, 0.0, 0.0))

    def step(self, steer: float, throttle: float, brake: float, dt: float) -> None:
        # Actuation delay: what the tyres feel now is what was commanded
        # `delay_steps` ticks ago.  Setting this non-zero is how you reproduce
        # the real game's input latency offline and check that the label
        # shifting in the recorder actually cancels it.
        if self.delay_steps > 0:
            self._queue.append((steer, throttle, brake))
            steer, throttle, brake = self._queue.popleft()
        # Recorded so the environment can report the APPLIED controls, the way
        # F1 25's telemetry does. Without this the delay is hidden state.
        self.last_applied = (steer, throttle, brake)

        s = self.state
        steer = max(-1.0, min(1.0, steer))
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))

        # Longitudinal: engine, brakes, drag, rolling resistance.
        a = throttle * MAX_ACCEL - brake * MAX_BRAKE
        a -= DRAG_K * s.v * s.v
        a -= ROLL_RESIST if s.v > 0.1 else 0.0
        v_new = max(0.0, min(V_MAX, s.v + a * dt))

        # Lateral: steering authority falls off with speed, which is what
        # stops the model from pivoting on the spot at 300 km/h.
        delta = steer * effective_max_steer(v_new)
        yaw_rate = (v_new / WHEELBASE) * math.tan(delta)

        s.yaw = (s.yaw + yaw_rate * dt) % (2.0 * math.pi)
        s.x += v_new * math.sin(s.yaw) * dt
        s.z += v_new * math.cos(s.yaw) * dt

        s.accel_lon = (v_new - s.v) / dt if dt > 0 else 0.0
        s.accel_lat = v_new * yaw_rate
        s.yaw_rate = yaw_rate
        s.v = v_new

    # -- derived quantities the telemetry packet needs ---------------------

    @property
    def speed_kph(self) -> int:
        return int(self.state.v * 3.6)

    @property
    def gear(self) -> int:
        """Eight ratios spread over the speed range."""
        if self.state.v < 0.5:
            return 0
        return max(1, min(8, int(self.state.v / (V_MAX / 8.0)) + 1))

    @property
    def engine_rpm(self) -> int:
        g = self.gear
        if g == 0:
            return 4000
        band = V_MAX / 8.0
        frac = (self.state.v - (g - 1) * band) / band
        return int(6000 + 9000 * max(0.0, min(1.0, frac)))
