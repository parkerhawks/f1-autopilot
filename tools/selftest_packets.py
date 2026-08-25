"""Offline validation of the packet layouts -- no game required.

Builds synthetic packets from the same struct formats the parser uses and
checks the values survive the round trip at the right offsets.  This cannot
prove the layouts match F1 25 (only tools/probe_telemetry.py can do that), but
it does prove the parser is self-consistent and that per-car indexing lands on
the right block -- which is the bug that would otherwise silently give you
another car's telemetry as training labels.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.telemetry import packets as P  # noqa: E402

PLAYER = 3


def test_sizes() -> None:
    assert P.HEADER_SIZE == 29, P.HEADER_SIZE
    assert P.CAR_MOTION_SIZE == 60, P.CAR_MOTION_SIZE
    assert P.CAR_TELEMETRY_SIZE == 60, P.CAR_TELEMETRY_SIZE
    print(f"  header {P.HEADER_SIZE}B  motion {P.CAR_MOTION_SIZE}B  "
          f"telemetry {P.CAR_TELEMETRY_SIZE}B")


def _header(packet_id: int) -> bytes:
    return struct.pack(
        P.HEADER_FMT,
        2025, 25, 1, 0, 1, packet_id,
        0xDEADBEEF, 7.5, 100, 100, PLAYER, 255,
    )


def test_header() -> None:
    h = P.parse_header(_header(0))
    assert h.packet_format == 2025
    assert h.game_year == 25
    assert h.packet_id == 0
    assert h.player_car_index == PLAYER
    assert abs(h.session_time - 7.5) < 1e-6
    print(f"  format={h.packet_format} year={h.game_year} "
          f"playerIdx={h.player_car_index} t={h.session_time}")


def test_motion_indexing() -> None:
    """Each car gets a distinct worldX so mis-indexing is impossible to miss."""
    buf = _header(0)
    for i in range(P.NUM_CARS):
        buf += struct.pack(
            P.CAR_MOTION_FMT,
            1000.0 + i, 2.0, -300.0,   # world pos -- X encodes the car index
            50.0, 0.0, 1.0,            # world velocity
            32767, 0, 0,               # forward dir -> (1, 0, 0)
            0, 0, 32767,               # right dir   -> (0, 0, 1)
            0.5, -1.2, 1.0,            # g force
            0.75, 0.01, -0.02,         # yaw pitch roll
        )
    assert len(buf) == P.HEADER_SIZE + P.NUM_CARS * P.CAR_MOTION_SIZE

    for i in (0, PLAYER, P.NUM_CARS - 1):
        m = P.parse_car_motion(buf, i)
        assert abs(m.world_x - (1000.0 + i)) < 1e-4, (i, m.world_x)

    m = P.parse_car_motion(buf, PLAYER)
    assert abs(m.fwd_x - 1.0) < 1e-4, m.fwd_x      # int16 -> unit range
    assert abs(m.right_z - 1.0) < 1e-4, m.right_z
    assert abs(m.yaw - 0.75) < 1e-6
    print(f"  car {PLAYER}: worldX={m.world_x:.1f} fwd=({m.fwd_x:.3f},"
          f"{m.fwd_y:.3f},{m.fwd_z:.3f}) yaw={m.yaw}")


def test_telemetry_indexing() -> None:
    buf = _header(6)
    for i in range(P.NUM_CARS):
        buf += struct.pack(
            P.CAR_TELEMETRY_FMT,
            200 + i,                    # speed encodes the car index
            1.0, -0.42, 0.0,            # throttle, steer, brake
            0, 7, 11500, 1, 80, 0b1111,
            500, 500, 450, 450,         # brake temps
            95, 95, 90, 90,             # tyre surface temps
            100, 100, 95, 95,           # tyre inner temps
            110,                        # engine temp
            22.5, 22.5, 21.0, 21.0,     # tyre pressures
            0, 0, 0, 0,                 # surface types
        )

    for i in (0, PLAYER, P.NUM_CARS - 1):
        t = P.parse_car_telemetry(buf, i)
        assert t.speed_kph == 200 + i, (i, t.speed_kph)

    t = P.parse_car_telemetry(buf, PLAYER)
    assert abs(t.steer + 0.42) < 1e-6, t.steer
    assert t.gear == 7 and t.engine_rpm == 11500 and t.drs == 1
    assert t.tyre_pressure == (22.5, 22.5, 21.0, 21.0)
    print(f"  car {PLAYER}: {t.speed_kph} km/h  gear {t.gear}  "
          f"steer {t.steer:+.2f}  rpm {t.engine_rpm}")


def test_negative_gear() -> None:
    """gear is int8 -- reverse must decode as -1, not 255."""
    buf = _header(6) + struct.pack(
        P.CAR_TELEMETRY_FMT,
        5, 0.0, 0.0, 0.0, 0, -1, 3000, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90,
        21.0, 21.0, 21.0, 21.0, 0, 0, 0, 0,
    ) * P.NUM_CARS
    assert P.parse_car_telemetry(buf, 0).gear == -1
    print("  reverse gear decodes as -1 (int8, not uint8)")


def test_lap_distance_calibrated() -> None:
    """The offsets measured against F1 25 must decode per-car correctly.

    Builds a LapData packet with a distinct lapDistance per car so a wrong
    stride reads someone else's progress -- which would poison the reward with
    another driver's lap and raise nothing.
    """
    assert P.LAP_DATA_STRIDE == 57, P.LAP_DATA_STRIDE
    assert P.LAP_DISTANCE_OFFSET == 20, P.LAP_DISTANCE_OFFSET

    buf = bytearray(_header(2))
    for i in range(P.NUM_CARS):
        block = bytearray(P.LAP_DATA_STRIDE)
        # lapDistance at offset 20, totalDistance right behind it at 24.
        struct.pack_into("<f", block, P.LAP_DISTANCE_OFFSET, 1000.0 + i)
        struct.pack_into("<f", block, P.LAP_DISTANCE_OFFSET + 4, 50000.0 + i)
        buf += block
    buf += b"\xff\xff"                      # the two trailing car-index bytes

    expected = P.EXPECTED_SIZES[P.PacketId.LAP_DATA]
    assert len(buf) == expected, (len(buf), expected)

    for i in (0, PLAYER, P.NUM_CARS - 1):
        d = P.parse_lap_distance(bytes(buf), i)
        assert abs(d - (1000.0 + i)) < 1e-3, (i, d)

    print(f"  packet {expected} B, stride {P.LAP_DATA_STRIDE}, "
          f"lapDistance at +{P.LAP_DISTANCE_OFFSET}; car {PLAYER} reads "
          f"{P.parse_lap_distance(bytes(buf), PLAYER):.1f} m")


def main() -> None:
    tests = [
        ("struct sizes", test_sizes),
        ("packet header", test_header),
        ("motion per-car indexing", test_motion_indexing),
        ("telemetry per-car indexing", test_telemetry_indexing),
        ("signed gear field", test_negative_gear),
        ("lapDistance calibration", test_lap_distance_calibrated),
    ]
    failed = 0
    for name, fn in tests:
        try:
            print(f"[ {name} ]")
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"all {len(tests)} checks pass")
    print("\nSelf-consistency proven. Layout vs. the real game is still unverified --")
    print("run tools/probe_telemetry.py with F1 25 running to confirm that.")


if __name__ == "__main__":
    main()
