"""A closed synthetic circuit, used by the mock game and by offline tests.

The shape is a polar curve r(theta) = R * (1 + a*sin(3*theta)), which gives a
closed loop with three fast sections and three slow corners of genuinely
different radii.  That variety matters: a constant-radius oval lets a policy
succeed with a single fixed steering angle, which would make the Phase 2
behavioral-cloning results meaningless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Track:
    """Centreline sampled at uniform parameter, with cumulative arc length."""

    x: np.ndarray          # (N,) world X of centreline points
    z: np.ndarray          # (N,) world Z
    s: np.ndarray          # (N,) cumulative distance from start/finish
    width: float           # half-width of the drivable surface, metres

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def nearest(self, px: float, pz: float) -> tuple[int, float, float]:
        """Closest centreline index, its lap distance, and signed offset.

        The signed offset is positive to the left of the direction of travel.
        This is the cross-track error every lateral controller needs, and it is
        also the cleanest 'am I still on the road' signal for a reward or for
        an automatic episode reset.
        """
        d2 = (self.x - px) ** 2 + (self.z - pz) ** 2
        i = int(np.argmin(d2))

        # Tangent from the neighbouring points, then cross product for the side.
        j = (i + 1) % len(self.x)
        tx, tz = self.x[j] - self.x[i], self.z[j] - self.z[i]
        n = math.hypot(tx, tz) or 1.0
        tx, tz = tx / n, tz / n
        dx, dz = px - self.x[i], pz - self.z[i]
        signed = dx * (-tz) + dz * tx

        return i, float(self.s[i]), float(signed)

    def heading_at(self, i: int) -> float:
        """Track direction at a centreline index, in the same frame as yaw."""
        j = (i + 1) % len(self.x)
        return math.atan2(self.x[j] - self.x[i], self.z[j] - self.z[i])

    def index_at_distance(self, distance: float) -> int:
        """Centreline index at a given lap distance, wrapping past the line.

        Points are sampled uniformly in the polar parameter, NOT in arc length
        -- spacing varies by the full wobble amplitude. Interpolating an index
        as `distance / length * N` is therefore wrong by tens of metres in the
        corners, which is exactly where a lookahead point matters most. Always
        come through here.
        """
        target = distance % self.length
        return int(np.searchsorted(self.s, target)) % len(self.x)

    def curvature_ahead(self, i: int, distances: tuple[float, ...]) -> list[float]:
        """Signed curvature at several lookahead distances.

        This is the single most useful hand-built feature for a racing policy:
        it is what lets the car brake *before* a corner rather than reacting to
        it.  In the real game this comes from a track map built by driving a
        recorded lap; here it is exact.
        """
        out = []
        n = len(self.x)
        for d in distances:
            k = self.index_at_distance(self.s[i] + d)
            a, b, c = self.x[(k - 1) % n], self.x[k], self.x[(k + 1) % n]
            az, bz, cz = self.z[(k - 1) % n], self.z[k], self.z[(k + 1) % n]
            out.append(_menger_curvature(a, az, b, bz, c, cz))
        return out


def _menger_curvature(
    ax: float, az: float, bx: float, bz: float, cx: float, cz: float
) -> float:
    """Signed curvature of the circle through three points (1/radius)."""
    # Twice the signed area of the triangle.
    area2 = (bx - ax) * (cz - az) - (bz - az) * (cx - ax)
    la = math.hypot(bx - ax, bz - az)
    lb = math.hypot(cx - bx, cz - bz)
    lc = math.hypot(cx - ax, cz - az)
    denom = la * lb * lc
    if denom < 1e-9:
        return 0.0
    return 2.0 * area2 / denom


def make_track(
    radius: float = 300.0,
    wobble: float = 0.40,
    lobes: int = 3,
    n_points: int = 4000,
    width: float = 7.0,
) -> Track:
    """Build the default synthetic circuit: ~2.45 km, tightest radius ~36 m.

    These defaults are chosen so the corners actually demand braking -- roughly
    130 km/h through the slowest, against 330 km/h on the straights. A track a
    car can take flat is useless for Phase 2, because a policy that outputs
    constant full throttle would score as a success.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    r = radius * (1.0 + wobble * np.sin(lobes * theta))
    x = r * np.cos(theta)
    z = r * np.sin(theta)

    seg = np.hypot(np.diff(x, append=x[0]), np.diff(z, append=z[0]))
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])

    return Track(x=x, z=z, s=s, width=width)
