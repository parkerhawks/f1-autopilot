"""Binary layout of the F1 25 UDP telemetry packets.

The struct formats below are transcribed from the EA/Codemasters F1 25 UDP
specification (packet format 2025).  They are *asserted* at import time and
re-validated against every packet at runtime, because EA revises these layouts
between titles and a silently-wrong byte offset produces training data that is
subtly corrupt rather than obviously broken.

If `tools/probe_telemetry.py` reports a size mismatch, the game's layout differs
from what is encoded here -- fix it here, do not loosen the check.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

# Every packet is little-endian and unpadded; '<' enforces both.
_LE = "<"


class PacketId(IntEnum):
    MOTION = 0
    SESSION = 1
    LAP_DATA = 2
    EVENT = 3
    PARTICIPANTS = 4
    CAR_SETUPS = 5
    CAR_TELEMETRY = 6
    CAR_STATUS = 7
    FINAL_CLASSIFICATION = 8
    LOBBY_INFO = 9
    CAR_DAMAGE = 10
    SESSION_HISTORY = 11
    TYRE_SETS = 12
    MOTION_EX = 13
    TIME_TRIAL = 14
    LAP_POSITIONS = 15


# --------------------------------------------------------------------------
# Packet header -- prefixes every packet
# --------------------------------------------------------------------------
# uint16 packetFormat      (2025)
# uint8  gameYear          (25)
# uint8  gameMajorVersion
# uint8  gameMinorVersion
# uint8  packetVersion
# uint8  packetId
# uint64 sessionUID
# float  sessionTime
# uint32 frameIdentifier
# uint32 overallFrameIdentifier
# uint8  playerCarIndex
# uint8  secondaryPlayerCarIndex
HEADER_FMT = _LE + "HBBBBBQfIIBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 29, HEADER_SIZE

_header_unpack = struct.Struct(HEADER_FMT).unpack_from


@dataclass(frozen=True, slots=True)
class Header:
    packet_format: int
    game_year: int
    game_major: int
    game_minor: int
    packet_version: int
    packet_id: int
    session_uid: int
    session_time: float
    frame_id: int
    overall_frame_id: int
    player_car_index: int
    secondary_player_car_index: int


def parse_header(buf: bytes) -> Header:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"runt packet: {len(buf)} bytes")
    return Header(*_header_unpack(buf))


# --------------------------------------------------------------------------
# Packet 0: Motion -- one CarMotionData per car (22 cars)
# --------------------------------------------------------------------------
# float worldPosition   X, Y, Z
# float worldVelocity   X, Y, Z
# int16 worldForwardDir X, Y, Z   (normalised, /32767.0)
# int16 worldRightDir   X, Y, Z   (normalised, /32767.0)
# float gForce          lateral, longitudinal, vertical
# float yaw, pitch, roll          (radians)
CAR_MOTION_FMT = _LE + "6f6h6f"
CAR_MOTION_SIZE = struct.calcsize(CAR_MOTION_FMT)
assert CAR_MOTION_SIZE == 60, CAR_MOTION_SIZE

NUM_CARS = 22

_car_motion_unpack = struct.Struct(CAR_MOTION_FMT).unpack_from


@dataclass(frozen=True, slots=True)
class CarMotion:
    world_x: float
    world_y: float
    world_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    fwd_x: float
    fwd_y: float
    fwd_z: float
    right_x: float
    right_y: float
    right_z: float
    g_lat: float
    g_lon: float
    g_vert: float
    yaw: float
    pitch: float
    roll: float


def parse_car_motion(buf: bytes, car_index: int) -> CarMotion:
    """Parse a single car's motion block without decoding the other 21."""
    off = HEADER_SIZE + car_index * CAR_MOTION_SIZE
    raw = _car_motion_unpack(buf, off)
    # Direction vectors are int16-normalised; scale them back to unit range.
    scaled = (
        *raw[0:6],
        *(v / 32767.0 for v in raw[6:12]),
        *raw[12:18],
    )
    return CarMotion(*scaled)


# --------------------------------------------------------------------------
# Packet 6: Car Telemetry -- one CarTelemetryData per car
# --------------------------------------------------------------------------
# uint16 speed                  (km/h)
# float  throttle               0.0 .. 1.0
# float  steer                  -1.0 (full left) .. 1.0 (full right)
# float  brake                  0.0 .. 1.0
# uint8  clutch                 0 .. 100
# int8   gear                   -1 reverse, 0 neutral, 1..8
# uint16 engineRPM
# uint8  drs                    0 off, 1 on
# uint8  revLightsPercent
# uint16 revLightsBitValue
# uint16 brakesTemperature[4]
# uint8  tyresSurfaceTemperature[4]
# uint8  tyresInnerTemperature[4]
# uint16 engineTemperature
# float  tyresPressure[4]
# uint8  surfaceType[4]
CAR_TELEMETRY_FMT = _LE + "HfffBbHBBH4H4B4BH4f4B"
CAR_TELEMETRY_SIZE = struct.calcsize(CAR_TELEMETRY_FMT)
assert CAR_TELEMETRY_SIZE == 60, CAR_TELEMETRY_SIZE

_car_telemetry_unpack = struct.Struct(CAR_TELEMETRY_FMT).unpack_from


@dataclass(frozen=True, slots=True)
class CarTelemetry:
    speed_kph: int
    throttle: float
    steer: float
    brake: float
    clutch: int
    gear: int
    engine_rpm: int
    drs: int
    rev_lights_pct: int
    rev_lights_bits: int
    brake_temp: tuple[int, int, int, int]
    tyre_surface_temp: tuple[int, int, int, int]
    tyre_inner_temp: tuple[int, int, int, int]
    engine_temp: int
    tyre_pressure: tuple[float, float, float, float]
    surface_type: tuple[int, int, int, int]


def parse_car_telemetry(buf: bytes, car_index: int) -> CarTelemetry:
    off = HEADER_SIZE + car_index * CAR_TELEMETRY_SIZE
    r = _car_telemetry_unpack(buf, off)
    return CarTelemetry(
        speed_kph=r[0],
        throttle=r[1],
        steer=r[2],
        brake=r[3],
        clutch=r[4],
        gear=r[5],
        engine_rpm=r[6],
        drs=r[7],
        rev_lights_pct=r[8],
        rev_lights_bits=r[9],
        brake_temp=r[10:14],
        tyre_surface_temp=r[14:18],
        tyre_inner_temp=r[18:22],
        engine_temp=r[22],
        tyre_pressure=r[23:27],
        surface_type=r[27:31],
    )


# --------------------------------------------------------------------------
# Packet 2: Lap Data
# --------------------------------------------------------------------------
# The tail of LapData has churned across F1 22/23/24/25 (delta fields and
# pit-stop predictions were inserted mid-struct).  Only the leading fields are
# decoded here; `lapDistance` is the one we actually need for progress reward
# and it is located empirically by tools/probe_telemetry.py --scan-floats
# rather than trusted from a hardcoded offset.
#
# Confirm the offset against your build, then set it here.
# CALIBRATED against F1 25 at Monza, 2026-08-17, packet format 2025.
#
# Found empirically with `probe_telemetry.py --scan-floats 2`, and confirmed
# three independent ways rather than by divisibility alone:
#   * the field runs 37 -> 5798 m and resets at the line; Monza is 5793 m
#   * totalDistance sits 4 bytes later and never resets
#   * the same fields reappear in other cars' blocks at exactly 228-byte
#     (4 x 57) intervals -- 248, 252, 429, 657 in the scan output
LAP_DATA_STRIDE: int | None = 57        # bytes per car
LAP_DISTANCE_OFFSET: int | None = 20    # bytes into a car's block


# Offset of currentLapInvalid (uint8, 1 = invalid) within a car's LapData
# block. Derived from the same struct layout whose lapDistance (+20) and
# totalDistance (+24) were both confirmed empirically against F1 25, so the
# intervening fields are almost certainly right too -- but "almost certainly"
# is why tools/probe_lap_flags.py exists to check it by running wide on purpose.
LAP_INVALID_OFFSET = 37

# pitStatus: 0 none, 1 pitting, 2 in the pit area.
#
# MEASURED USELESS IN TIME TRIAL. A byte-level watch across a session that
# included a pit entry showed this offset never changing at all. Time Trial
# does not send the car down a pit lane -- it drops straight to the garage
# screen, so nothing is ever "pitting". Kept for completeness and for other
# session types; do not rely on it here. Use DRIVER_STATUS_OFFSET instead.
PIT_STATUS_OFFSET = 34

# driverStatus within a car's LapData block:
#   0 in garage, 1 flying lap, 2 in lap, 3 out lap, 4 on track
#
# CONFIRMED EMPIRICALLY against F1 25 at Monza with
# `probe_telemetry.py --watch-bytes 2`: this byte toggled between 0 and 1
# exactly when the car left the garage and when it was returned to it. In Time
# Trial only those two values appear -- 0 in the garage, 1 on a flying lap --
# so `!= 1` is a reliable "not driving" test.
#
# This is the signal that detects a pit entry, because the pit entry IS a
# return to the garage. pitStatus (+34) never moves and was the wrong thing to
# watch for entirely.
DRIVER_STATUS_OFFSET = 44
DRIVER_IN_GARAGE = 0
DRIVER_FLYING_LAP = 1


def parse_driver_status(buf: bytes, car_index: int) -> int:
    """Where the game thinks the driver is.

    This is what makes menu recovery reliable. Blind button sequences have to
    guess whether a prompt has appeared yet; this says outright when the car
    has landed back in the garage, so the escape can press B until that is
    true and then start a lap, instead of pressing a fixed number of times and
    hoping.
    """
    if LAP_DATA_STRIDE is None:
        raise RuntimeError("LapData offsets not calibrated")
    off = HEADER_SIZE + car_index * LAP_DATA_STRIDE + DRIVER_STATUS_OFFSET
    if off >= len(buf):
        return -1
    return int(buf[off])


def parse_pit_status(buf: bytes, car_index: int) -> int:
    """0 = on track, 1 = pitting, 2 = in the pit area.

    Needed because the pit lane is invisible to every other signal we have:
    it is tarmac, so no wheels read as off-surface; it does not invalidate the
    lap; and the pit speed limiter keeps the car moving just fast enough to
    stay above the stuck threshold. An agent that wanders in simply crawls
    there until the episode times out, learning nothing for a hundred seconds.
    """
    if LAP_DATA_STRIDE is None:
        raise RuntimeError("LapData offsets not calibrated")
    off = HEADER_SIZE + car_index * LAP_DATA_STRIDE + PIT_STATUS_OFFSET
    if off >= len(buf):
        return 0
    return int(buf[off])


def parse_lap_invalid(buf: bytes, car_index: int) -> bool:
    """True when the game itself has invalidated the current lap.

    This is the game's own track-limits verdict, which is strictly better than
    any rule we could infer: it already knows about kerbs, the difference
    between a legal and illegal cut, and every local exception at this circuit.
    """
    if LAP_DATA_STRIDE is None:
        raise RuntimeError("LapData offsets not calibrated")
    off = HEADER_SIZE + car_index * LAP_DATA_STRIDE + LAP_INVALID_OFFSET
    if off >= len(buf):
        return False
    return buf[off] != 0


def parse_lap_distance(buf: bytes, car_index: int) -> float:
    """Distance around the current lap in metres (negative before the line)."""
    if LAP_DATA_STRIDE is None or LAP_DISTANCE_OFFSET is None:
        raise RuntimeError(
            "LapData offsets not calibrated -- run:\n"
            "  python tools/probe_telemetry.py --scan-floats 2\n"
            "and fill in LAP_DATA_STRIDE / LAP_DISTANCE_OFFSET in packets.py"
        )
    off = HEADER_SIZE + car_index * LAP_DATA_STRIDE + LAP_DISTANCE_OFFSET
    return struct.unpack_from(_LE + "f", buf, off)[0]


# --------------------------------------------------------------------------
# Expected total packet sizes, used to detect spec drift at runtime.
# A value of None means "not yet pinned" -- the probe will report the real one.
# --------------------------------------------------------------------------
EXPECTED_SIZES: dict[int, int | None] = {
    PacketId.MOTION: HEADER_SIZE + NUM_CARS * CAR_MOTION_SIZE,
    PacketId.CAR_TELEMETRY: None,  # has a short trailer; pinned by probe
    # 29 + 22*57 + 2 trailing bytes, measured on F1 25 (2026-08-17).
    PacketId.LAP_DATA: HEADER_SIZE + NUM_CARS * 57 + 2,
}
