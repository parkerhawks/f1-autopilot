"""Renders a driver's-eye view of the synthetic track, so the CNN sees pixels.

Without this the mock game can only exercise the state branch, and the vision
half of the policy -- the half that makes this an autonomous-driving project
rather than a control project -- would go untested until the real game arrives.

The projection is a pinhole camera on the ground plane: for a point at depth z
and lateral offset x, with the camera h metres up,

    u = cx + f * x / z
    v = cy + f * h / z

so the horizon sits at v = cy and nearer ground projects lower in the frame.
That is the same geometry a real chase/cockpit camera obeys, which is what
makes the learned features stand a chance of surviving the swap to real frames:
the road still narrows to a vanishing point at the same rate.

Deliberately low-fidelity. The point is not to look like F1 25 -- it is to
present the same *task* (infer curvature and lateral offset from a perspective
view of a road) at the same input resolution.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .track import Track

# A RAISED chase camera, not a cockpit one. This is not a stylistic choice.
# Vertical position on the ground plane goes as h/depth, so a 1.2 m cockpit
# camera compresses everything beyond ~45 m into under two pixels -- the CNN
# would have almost no signal about upcoming curvature, which is precisely what
# it needs to brake early. Lifting the camera to 6 m spreads 20-150 m of road
# across ~25 rows instead. F1 25's default chase camera sits in this range for
# the same reason.
CAM_HEIGHT = 6.0       # m above the road
CAM_PITCH = 0.0        # rad; the h/depth term already tilts the view down
FOCAL = 95.0           # px, at the reference 192x96 frame
NEAR_CLIP = 6.0        # m; nearer than this projects below the frame anyway
FAR_CLIP = 160.0       # m; beyond this the road is sub-pixel

# 8-bit greys. Road darker than grass gives the CNN a clean edge to latch onto,
# which is also true of the real game.
COL_SKY = 40
COL_GRASS = 105
COL_ROAD = 25
COL_EDGE = 235


class TrackRenderer:
    """Rasterises the track ahead of the car into a single-channel image."""

    def __init__(self, track: Track, width: int = 192, height: int = 96):
        self.track = track
        self.width = width
        self.height = height
        # Scale intrinsics with resolution so the FOV is resolution-independent.
        self.focal = FOCAL * (width / 192.0)
        self.cx = width * 0.5
        # Horizon high in the frame, so the lower two-thirds is road.
        self.cy = height * 0.30

        # Precompute track edges once; they never move.
        self._edges_l, self._edges_r = self._build_edges()

    def _build_edges(self) -> tuple[np.ndarray, np.ndarray]:
        t = self.track
        n = len(t.x)
        nxt = (np.arange(n) + 1) % n
        tx, tz = t.x[nxt] - t.x, t.z[nxt] - t.z
        norm = np.hypot(tx, tz)
        norm[norm < 1e-9] = 1.0
        tx, tz = tx / norm, tz / norm
        # Left normal of a (x, z) tangent is (-tz, tx).
        left = np.stack([t.x - tz * t.width, t.z + tx * t.width], axis=1)
        right = np.stack([t.x + tz * t.width, t.z - tx * t.width], axis=1)
        return left, right

    def render(self, car_x: float, car_z: float, yaw: float) -> np.ndarray:
        """One greyscale frame, shape (height, width), dtype uint8."""
        img = np.full((self.height, self.width), COL_GRASS, np.uint8)
        # Sky above the horizon.
        horizon = int(self.cy)
        if horizon > 0:
            img[:horizon, :] = COL_SKY

        # Which centreline points are near enough to matter. Walking forward
        # from the nearest index keeps this O(window) rather than O(track).
        i, _, _ = self.track.nearest(car_x, car_z)
        n = len(self.track.x)
        span = int(FAR_CLIP / max(1e-6, self.track.length / n)) + 2
        idx = (np.arange(i, i + span) % n)

        left = self._project_many(self._edges_l[idx], car_x, car_z, yaw)
        right = self._project_many(self._edges_r[idx], car_x, car_z, yaw)

        # Keep only the run where BOTH edges are visible, so the road polygon
        # never gets stitched across a clipped gap.
        ok = left[:, 2].astype(bool) & right[:, 2].astype(bool)
        if ok.sum() >= 2:
            run = self._longest_run(ok)
            if run is not None:
                a, b = run
                lp = left[a:b, :2]
                rp = right[a:b, :2]
                poly = np.concatenate([lp, rp[::-1]], axis=0).astype(np.int32)
                cv2.fillPoly(img, [poly], COL_ROAD, lineType=cv2.LINE_AA)
                cv2.polylines(img, [lp.astype(np.int32)], False, COL_EDGE, 1,
                              cv2.LINE_AA)
                cv2.polylines(img, [rp.astype(np.int32)], False, COL_EDGE, 1,
                              cv2.LINE_AA)
        return img

    @staticmethod
    def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
        """Longest contiguous True span, as [start, end)."""
        best = cur = None
        best_len = 0
        for k, v in enumerate(mask):
            if v:
                cur = k if cur is None else cur
                if k - cur + 1 > best_len:
                    best_len, best = k - cur + 1, (cur, k + 1)
            else:
                cur = None
        return best if best_len >= 2 else None

    def _project_many(
        self, pts: np.ndarray, car_x: float, car_z: float, yaw: float
    ) -> np.ndarray:
        """World (N,2) -> image (N,3) of [u, v, visible]."""
        dx = pts[:, 0] - car_x
        dz = pts[:, 1] - car_z

        # Into the car frame: forward is +Z, right is +X, matching the yaw
        # convention used by the vehicle model and by F1's Motion packet.
        s, c = math.sin(yaw), math.cos(yaw)
        depth = dx * s + dz * c
        lateral = dx * c - dz * s

        # Pitch tilts the ray; a small-angle shift of the horizon is enough.
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.focal * lateral / depth
            v = self.cy + self.focal * (CAM_HEIGHT / depth + CAM_PITCH)

        visible = (depth > NEAR_CLIP) & (depth < FAR_CLIP)
        visible &= np.isfinite(u) & np.isfinite(v)
        # Allow generous overshoot so polygons still clip correctly at edges.
        visible &= (u > -4 * self.width) & (u < 5 * self.width)

        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return np.stack([u, v, visible.astype(np.float64)], axis=1)
