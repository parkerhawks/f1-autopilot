"""A scripted driver for the mock game -- the stand-in demonstrator.

Pure pursuit for steering, curvature-limited target speed for the pedals.  This
is a classical controller with no learning in it at all, and that is the point:
it plays the role the human will play once the real game exists, so the whole
imitation-learning pipeline (record -> train -> evaluate -> DAgger) can be
built and validated against a demonstrator whose behaviour is known exactly.

Having a perfect, reproducible expert also gives Phase 2 a ceiling to measure
against.  If behavioral cloning cannot match a pure-pursuit controller on a
synthetic track, the bug is in the pipeline, not in the difficulty of driving.
"""

from __future__ import annotations

import math

from .track import Track
from .vehicle import MAX_BRAKE, V_MAX, WHEELBASE, Vehicle, effective_max_steer

# Lookahead grows with speed: too short and the car weaves, too long and it
# cuts corners.  The linear-in-speed form is the standard pure-pursuit result.
LOOKAHEAD_BASE = 12.0     # m
LOOKAHEAD_GAIN = 0.55     # m per (m/s)

LAT_ACCEL_LIMIT = 38.0    # m/s^2 the model will hold before it runs wide
BRAKE_MARGIN = 1.35       # brake this much earlier than the ideal point


class PurePursuitExpert:
    """Deterministic demonstrator. Same track position -> same controls."""

    def __init__(self, track: Track) -> None:
        self.track = track

    def act(self, veh: Vehicle) -> tuple[float, float, float]:
        s = veh.state
        i, _, _ = self.track.nearest(s.x, s.z)

        # -- steering: aim at a point ahead on the centreline ---------------
        ld = LOOKAHEAD_BASE + LOOKAHEAD_GAIN * s.v
        k = self.track.index_at_distance(self.track.s[i] + ld)
        tx, tz = self.track.x[k], self.track.z[k]

        # Transform the target into the car's frame. F1 yaw is measured from
        # +Z, so the rotation below is (sin, cos) rather than the usual pair.
        dx, dz = tx - s.x, tz - s.z
        local_x = dx * math.cos(s.yaw) - dz * math.sin(s.yaw)
        local_z = dx * math.sin(s.yaw) + dz * math.cos(s.yaw)

        # Pure pursuit: the arc through the target has curvature 2x/d^2, and
        # the bicycle model needs delta = atan(L * kappa) to follow it. Convert
        # to the [-1,1] control range using the authority actually available at
        # this speed -- dividing by the static MAX_STEER_RAD instead understeers
        # by ~5x above 200 km/h.
        dist = math.hypot(local_x, local_z) or 1.0
        curvature = 2.0 * local_x / (dist * dist)
        delta_required = math.atan(WHEELBASE * curvature)
        steer = max(-1.0, min(1.0, delta_required / effective_max_steer(s.v)))

        # -- pedals: slow to what the upcoming corner will allow ------------
        v_target = self._speed_limit_ahead(i, s.v)
        err = v_target - s.v
        if err > 0.5:
            throttle, brake = min(1.0, err / 8.0), 0.0
        elif err < -0.5:
            throttle, brake = 0.0, min(1.0, -err / 12.0)
        else:
            throttle, brake = 0.35, 0.0

        # Never ask for full throttle while cranking on steering lock.
        throttle *= 1.0 - 0.55 * abs(steer)
        return steer, throttle, brake

    def _speed_limit_ahead(self, i: int, v: float) -> float:
        """Lowest speed demanded by any corner inside the braking distance."""
        horizon = BRAKE_MARGIN * v * v / (2.0 * MAX_BRAKE) + 25.0
        samples = tuple(d for d in (10.0, 25.0, 50.0, 90.0, 140.0, 200.0)
                        if d <= max(horizon, 30.0))
        if not samples:
            samples = (10.0,)

        limit = V_MAX
        for d, kappa in zip(samples, self.track.curvature_ahead(i, samples)):
            if abs(kappa) < 1e-6:
                continue
            v_corner = math.sqrt(LAT_ACCEL_LIMIT / abs(kappa))
            # Speed we can still be at now and shed in time for that corner.
            v_allowed = math.sqrt(
                max(0.0, v_corner ** 2 + 2.0 * (MAX_BRAKE / BRAKE_MARGIN) * d)
            )
            limit = min(limit, v_allowed)
        return min(limit, V_MAX)
