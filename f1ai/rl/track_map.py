"""A curvature map of the real circuit, built from recorded laps.

WHY THIS EXISTS
---------------
The observation originally carried no track knowledge at all, so the CNN had to
infer everything about the road from pixels. That was too purist for the data
available: with a couple of thousand demonstration frames, learning "a corner
is coming, start braking" from images alone is close to hopeless, and the first
in-game policy duly held full throttle into Monza's first chicane every time.

Real autonomy stacks do not work that way either. Waymo and Cruise both drive
on prior HD maps; the map supplies road geometry, perception supplies
localisation and everything dynamic. This is that split:

    the MAP says   what the road ahead does
    VISION says    where on it we actually are

So the features here are strictly forward-looking curvature and a reference
speed profile. Lateral offset and heading error are deliberately NOT provided,
because those are exactly what the camera is for -- handing them over would
leave nothing for the CNN to do and reduce the whole thing to control on a
known trajectory.

HOW IT IS BUILT
---------------
No world coordinates are needed. For a car following a path,

    curvature = yaw_rate / speed

and both are in the telemetry, indexed by lapDistance. Averaging over several
recorded laps and smoothing gives a stable kappa(s) for the circuit.

The recorded speed profile comes along for free and is worth as much: it says
where a competent human was willing to be at full speed and where they braked,
which is a far better prior than anything the agent could infer early on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Metres ahead at which curvature is sampled. Roughly: turn-in, mid-corner
# planning, braking-zone warning, and long-range "is a chicane coming".
# At 300 km/h the car covers 83 m/s, so 150 m is under two seconds of warning.
LOOKAHEADS = (20.0, 50.0, 90.0, 150.0)
# Where the reference speed profile is sampled, in metres ahead.
SPEED_LOOKAHEADS = (40.0, 110.0)

MAP_FEATURES = len(LOOKAHEADS) + len(SPEED_LOOKAHEADS)

MAPS_DIR = Path(__file__).resolve().parent.parent.parent / "maps"


def map_path(track: str) -> Path:
    """Resolve a track NAME or an explicit path to a map file.

    Nothing here is specific to any circuit -- the map is built from whatever
    laps you record, so the same commands work for Monza, Silverstone or Spa
    by changing one argument.
    """
    p = Path(track)
    return p if p.suffix == ".npz" else MAPS_DIR / f"{track}.npz"


def find_any_map() -> Path | None:
    """The first available map, for tools that only need geometry to exist."""
    if not MAPS_DIR.exists():
        return None
    return next(iter(sorted(MAPS_DIR.glob("*.npz"))), None)

# Curvature is tiny in absolute terms (1/radius, so ~0.01 for a 100 m corner).
# Scaling brings it into the range the rest of the state vector lives in.
KAPPA_SCALE = 60.0


@dataclass
class TrackMap:
    s: np.ndarray            # (N,) lap distance, metres, ascending
    curvature: np.ndarray    # (N,) signed 1/m; positive is one direction
    ref_speed: np.ndarray    # (N,) m/s, the demonstrated speed profile
    length: float
    source_laps: tuple[str, ...] = ()

    # World-coordinate racing line, when the recordings carried position.
    #
    # This is what makes a tarmac run-off detectable. Monza's Rettifilo escape
    # road keeps all four wheels on tarmac, so surface type reports nothing
    # wrong and the progress reward pays in full for driving straight across
    # the chicane. Only distance from the demonstrated line reveals it.
    x: np.ndarray | None = None
    z: np.ndarray | None = None

    # Metres from the RACING LINE that count as still on the track.
    #
    # Not the track's half-width. The reference is a racing line, which hugs
    # the inside through corners, so a car legitimately using the far side of
    # the road can sit ~10 m away from it. Measured on two recorded laps, the
    # driven line itself varies up to 5.6 m from the average, so a tight
    # threshold would punish perfectly legal driving.
    #
    # Detection does not need tightness: the Rettifilo run-off cut puts the car
    # 20 m+ off line, so 10 m separates "using the width of the track" from
    # "driving down the escape road" with room to spare.
    half_width: float = 10.0

    @property
    def has_geometry(self) -> bool:
        return self.x is not None and self.z is not None

    # How far along the track to search either side of the reported position.
    #
    # MUST BE NARROW. A wide window spans both legs of a chicane, and a car
    # displaced sideways then matches the returning track instead of the leg
    # it is actually on -- measured at Roggia with a 120 m window, a genuine
    # 15 m displacement reported 0.8 m and a 60 m displacement reported 3.3 m.
    # Enforcement would have been silently disabled at exactly the corners it
    # exists to police. Monza's chicane legs are ~50 m apart along the track,
    # so 20 m keeps them separate while still tolerating a stale packet or two.
    SEARCH_SPAN_M = 20.0

    # Beyond this, a windowed result is treated as a lookup failure rather
    # than as a car that is genuinely that far from the road.
    WINDOW_SANITY_M = 80.0

    def lateral_offset(self, world_x: float, world_z: float,
                       lap_distance: float) -> float:
        """Metres from the demonstrated racing line, unsigned.

        Searched in a window around the reported lap distance rather than
        globally, because Monza crosses close to itself and a global
        nearest-point search will happily match a point on the other side of
        the circuit and report a tidy offset for a car that is nowhere near.

        But the window is only as good as `lapDistance`, and that value is not
        always trustworthy: after a Restart Lap it can be NEGATIVE while the
        car sits on an out-lap before the start line, and `-500 % 5797` wraps
        to 5297 m -- half a kilometre from where the car actually is. The
        window then centres on the wrong part of the track, every position
        reads as wildly off-line, and the episode is terminated within twenty
        steps of every reset. That is a reset loop, and it burned a run.

        So a windowed answer that looks absurd is treated as a failed lookup,
        and the search widens globally. If the global answer is also large the
        car really is off the road; if it is small, `lapDistance` was simply
        lying.
        """
        if not self.has_geometry:
            return 0.0

        n = len(self.s)
        centre = int(np.searchsorted(self.s, lap_distance % self.length)) % n
        span = max(4, int(self.SEARCH_SPAN_M / max(1e-6, self.length / n)))
        idx = (np.arange(centre - span, centre + span) % n)
        dx = self.x[idx] - world_x
        dz = self.z[idx] - world_z
        windowed = float(np.sqrt(np.min(dx * dx + dz * dz)))

        if windowed <= self.WINDOW_SANITY_M:
            return windowed

        dx = self.x - world_x
        dz = self.z - world_z
        return float(min(windowed, np.sqrt(np.min(dx * dx + dz * dz))))

    # -- lookup -----------------------------------------------------------

    def _at(self, values: np.ndarray, distance: float) -> float:
        return float(np.interp(distance % self.length, self.s, values))

    def features(self, lap_distance: float, speed_mps: float) -> np.ndarray:
        """The map's contribution to the observation.

        Curvature is signed, so the policy can tell left from right without
        the camera; magnitude tells it how hard. Reference speed is expressed
        as a RATIO to current speed rather than an absolute, because "you are
        going 1.6x faster than this corner allows" is the actionable quantity
        and it stays meaningful at any speed.
        """
        feats = [
            np.clip(self._at(self.curvature, lap_distance + d) * KAPPA_SCALE,
                    -3.0, 3.0)
            for d in LOOKAHEADS
        ]
        v = max(speed_mps, 5.0)
        for d in SPEED_LOOKAHEADS:
            ref = self._at(self.ref_speed, lap_distance + d)
            feats.append(np.clip(ref / v, 0.0, 3.0) - 1.0)
        return np.asarray(feats, np.float32)

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        arrays = dict(s=self.s, curvature=self.curvature,
                      ref_speed=self.ref_speed,
                      length=np.float32(self.length),
                      half_width=np.float32(self.half_width))
        if self.has_geometry:
            arrays["x"] = self.x
            arrays["z"] = self.z
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps({
            "length_m": round(self.length, 1),
            "samples": len(self.s),
            "source_laps": list(self.source_laps),
            "lookaheads_m": list(LOOKAHEADS),
            "speed_lookaheads_m": list(SPEED_LOOKAHEADS),
            "max_curvature": float(np.abs(self.curvature).max()),
            "min_radius_m": float(1.0 / max(1e-6, np.abs(self.curvature).max())),
            "ref_speed_kph": [round(float(self.ref_speed.min() * 3.6), 1),
                              round(float(self.ref_speed.max() * 3.6), 1)],
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TrackMap":
        d = np.load(path)
        return cls(s=d["s"], curvature=d["curvature"],
                   ref_speed=d["ref_speed"], length=float(d["length"]),
                   x=d["x"] if "x" in d.files else None,
                   z=d["z"] if "z" in d.files else None,
                   half_width=float(d["half_width"])
                   if "half_width" in d.files else 7.0)


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Circular moving average -- the track is a loop, so the ends join."""
    if window < 2:
        return y
    k = np.ones(window) / window
    padded = np.concatenate([y[-window:], y, y[:window]])
    return np.convolve(padded, k, mode="same")[window:-window]


def build_from_laps(lap_files: list[Path], bin_metres: float = 5.0,
                    smooth_bins: int = 5) -> TrackMap:
    """Average curvature and speed across laps, binned by lap distance."""
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    world: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    names = []

    for path in sorted(lap_files):
        with np.load(path) as npz:
            d = {k: npz[k] for k in npz.files}
        speed = d["speed"].astype(np.float64)
        yaw_rate = d["yaw_rate"].astype(np.float64)
        dist = d["lap_distance"].astype(np.float64)

        # Below walking pace the curvature estimate explodes and means nothing.
        ok = speed > 8.0
        if ok.sum() < 100:
            print(f"  skip {path.name}: too little motion")
            continue
        kappa = np.zeros_like(speed)
        kappa[ok] = yaw_rate[ok] / speed[ok]
        samples.append((dist[ok], kappa[ok], speed[ok]))
        if "world_x" in d and "world_z" in d:
            world.append((dist[ok], d["world_x"].astype(np.float64)[ok],
                          d["world_z"].astype(np.float64)[ok]))
        names.append(path.name)
        print(f"  use  {path.name}: {ok.sum()} samples, "
              f"lapDistance {dist[ok].min():.0f}-{dist[ok].max():.0f} m")

    if not samples:
        raise SystemExit("no usable laps")

    length = max(float(d.max()) for d, _, _ in samples)
    n_bins = max(64, int(length / bin_metres))
    edges = np.linspace(0.0, length, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    kappa_sum = np.zeros(n_bins)
    speed_sum = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for dist, kappa, speed in samples:
        idx = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
        np.add.at(kappa_sum, idx, kappa)
        np.add.at(speed_sum, idx, speed)
        np.add.at(counts, idx, 1.0)

    # Bins nobody drove through (pit lane, a missed section) get filled by
    # interpolation rather than left as zero -- a spurious "straight" in the
    # middle of a corner is worse than a slightly wrong estimate.
    filled = counts > 0
    if filled.sum() < n_bins * 0.5:
        raise SystemExit(
            f"only {filled.sum()}/{n_bins} of the track was covered -- "
            "record a full clean lap")
    kappa = np.interp(centres, centres[filled],
                      (kappa_sum[filled] / counts[filled]))
    speed = np.interp(centres, centres[filled],
                      (speed_sum[filled] / counts[filled]))

    # Average the racing line itself, when the recordings carried position.
    line_x = line_z = None
    if world:
        xs = np.zeros(n_bins)
        zs = np.zeros(n_bins)
        wc = np.zeros(n_bins)
        for dist, wx, wz in world:
            idx = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
            np.add.at(xs, idx, wx)
            np.add.at(zs, idx, wz)
            np.add.at(wc, idx, 1.0)
        filled_w = wc > 0
        line_x = np.interp(centres, centres[filled_w], xs[filled_w] / wc[filled_w])
        line_z = np.interp(centres, centres[filled_w], zs[filled_w] / wc[filled_w])
        # No smoothing: this is a position, and averaging neighbouring bins
        # would cut corners exactly where precision matters most.
        print(f"  racing line from {len(world)} lap(s) with world position")
    else:
        print("  NO WORLD POSITION in these recordings -- the map will have no")
        print("  geometry, and a tarmac run-off will be undetectable.")

    return TrackMap(
        s=centres,
        curvature=_smooth(kappa, smooth_bins),
        ref_speed=_smooth(speed, smooth_bins),
        length=length,
        source_laps=tuple(names),
        x=line_x, z=line_z,
    )
