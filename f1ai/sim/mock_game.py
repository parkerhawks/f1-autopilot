"""A fake F1 25: emits real-format UDP telemetry from a simulated car.

Everything downstream of the socket -- parser, recorder, feature extraction,
training, evaluation -- can be built and regression-tested against this without
owning the game.  When the real game arrives, only the control backend and the
frame source change.

  emitted:   packet 0 (Motion), 2 (LapData), 6 (CarTelemetry) at 60 Hz
  control:   UDP on port 20778, three little-endian float32s
             (steer, throttle, brake)

The LapData layout below mirrors the real F1 24/25 struct closely, *including*
its two-byte trailer after the 22 car blocks.  That trailer is why a naive
(packet_size - 29) / 22 does not divide evenly, and reproducing it here means
tools/probe_telemetry.py gets tested against the same trap the real game sets.
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import time

from ..telemetry.packets import (
    CAR_MOTION_FMT,
    CAR_TELEMETRY_FMT,
    HEADER_FMT,
    NUM_CARS,
    PacketId,
)
from .expert import PurePursuitExpert
from .track import make_track
from .vehicle import Vehicle, VehicleState

TELEMETRY_PORT = 20777
CONTROL_PORT = 20778
PLAYER_INDEX = 0

# -- mock LapData layout ---------------------------------------------------
# 57 bytes per car; lapDistance sits at offset 20 within each car's block.
# Deliberately NOT exported from packets.py: the whole point of the Phase 0
# workflow is that this offset gets *discovered* by the probe, not assumed.
LAP_FMT = "<IIHBHBHBHBfff" + "B" * 15 + "HHBfB"
LAP_STRIDE = struct.calcsize(LAP_FMT)
assert LAP_STRIDE == 57, LAP_STRIDE
LAP_TRAILER = 2  # timeTrialPBCarIdx, timeTrialRivalCarIdx


def _header(packet_id: int, session_time: float, frame: int, uid: int) -> bytes:
    return struct.pack(
        HEADER_FMT,
        2025, 25, 1, 0, 1, int(packet_id),
        uid, session_time, frame, frame, PLAYER_INDEX, 255,
    )


def _motion_packet(veh: Vehicle, hdr: bytes) -> bytes:
    s = veh.state
    fwd_x, fwd_z = math.sin(s.yaw), math.cos(s.yaw)
    right_x, right_z = math.cos(s.yaw), -math.sin(s.yaw)

    def q(v: float) -> int:
        return max(-32767, min(32767, int(v * 32767)))

    player = struct.pack(
        CAR_MOTION_FMT,
        s.x, 0.0, s.z,
        s.v * fwd_x, 0.0, s.v * fwd_z,
        q(fwd_x), 0, q(fwd_z),
        q(right_x), 0, q(right_z),
        s.accel_lat / 9.81, s.accel_lon / 9.81, 1.0,
        s.yaw, 0.0, 0.0,
    )
    # Park the other 21 cars far away rather than omitting them: the packet
    # must be the size the parser expects.
    idle = struct.pack(
        CAR_MOTION_FMT,
        -9999.0, 0.0, -9999.0, 0.0, 0.0, 0.0,
        0, 0, 32767, 32767, 0, 0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    )
    return hdr + player + idle * (NUM_CARS - 1)


def _telemetry_packet(
    veh: Vehicle, controls: tuple[float, float, float], hdr: bytes
) -> bytes:
    steer, throttle, brake = controls
    body = struct.pack(
        CAR_TELEMETRY_FMT,
        veh.speed_kph, throttle, steer, brake,
        0, veh.gear, veh.engine_rpm, 0,
        int(100 * veh.engine_rpm / 15000), 0,
        350, 350, 320, 320,
        90, 90, 88, 88,
        95, 95, 92, 92,
        105,
        22.5, 22.5, 21.0, 21.0,
        0, 0, 0, 0,
    )
    idle = struct.pack(
        CAR_TELEMETRY_FMT,
        0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0,
    )
    # The real packet carries a short trailer (MFD panel state, suggested gear).
    trailer = struct.pack("<BBBb", 0, 0, 0, 0)
    return hdr + body + idle * (NUM_CARS - 1) + trailer


def _lap_packet(
    lap_distance: float, total_distance: float, lap_num: int,
    lap_time_ms: int, last_lap_ms: int, hdr: bytes,
) -> bytes:
    body = struct.pack(
        LAP_FMT,
        last_lap_ms, lap_time_ms,
        0, 0, 0, 0, 0, 0, 0, 0,
        lap_distance, total_distance, 0.0,
        1, lap_num, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0,
        0, 0, 0,
        0.0, 0,
    )
    idle = struct.pack(
        LAP_FMT,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0, 0.0, 0.0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
        0, 0, 0,
        0.0, 0,
    )
    return (hdr + body + idle * (NUM_CARS - 1)
            + struct.pack("<BB", 255, 255))


def run(
    host: str = "127.0.0.1",
    rate_hz: float = 60.0,
    autopilot: bool = True,
    delay_ms: float = 0.0,
    duration: float | None = None,
    verbose: bool = True,
) -> None:
    track = make_track()
    dt = 1.0 / rate_hz
    delay_steps = int(round(delay_ms / 1000.0 / dt))
    veh = Vehicle(delay_steps=delay_steps)

    # Place the car on the start line, pointing along the track.
    veh.state = VehicleState(x=float(track.x[0]), z=float(track.z[0]),
                             yaw=track.heading_at(0), v=30.0)

    expert = PurePursuitExpert(track) if autopilot else None

    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctl.bind((host, CONTROL_PORT))
    ctl.setblocking(False)

    addr = (host, TELEMETRY_PORT)
    uid = int(time.time()) & 0xFFFFFFFF
    controls = (0.0, 0.0, 0.0)
    frame = 0
    lap_num = 1
    lap_start = time.perf_counter()
    last_lap_ms = 0
    prev_s = 0.0
    total_distance = 0.0

    if verbose:
        print(f"mock F1 25 -> udp://{host}:{TELEMETRY_PORT}  @{rate_hz:g} Hz")
        print(f"  track {track.length:.0f} m   control on udp://{host}:{CONTROL_PORT}")
        print(f"  driver: {'built-in pure-pursuit expert' if autopilot else 'external'}"
              f"   actuation delay: {delay_ms:g} ms ({delay_steps} ticks)")
        print("  Ctrl-C to stop.\n")

    t0 = time.perf_counter()
    next_tick = t0
    try:
        while duration is None or (time.perf_counter() - t0) < duration:
            # Drain the control socket; keep only the newest command.
            while True:
                try:
                    data, _ = ctl.recvfrom(64)
                except BlockingIOError:
                    break
                if len(data) >= 12:
                    controls = struct.unpack_from("<fff", data, 0)

            if expert is not None:
                controls = expert.act(veh)

            veh.step(*controls, dt)

            i, s_here, _ = track.nearest(veh.state.x, veh.state.z)
            step_dist = veh.state.v * dt
            total_distance += step_dist
            # Crossing the start/finish line shows up as s wrapping to ~0.
            if prev_s > track.length * 0.75 and s_here < track.length * 0.25:
                now = time.perf_counter()
                last_lap_ms = int((now - lap_start) * 1000)
                lap_start = now
                lap_num += 1
                if verbose:
                    print(f"  lap {lap_num - 1:>3} complete: "
                          f"{last_lap_ms / 1000:.3f} s")
            prev_s = s_here

            now = time.perf_counter()
            st = now - t0
            lap_ms = int((now - lap_start) * 1000)

            out.sendto(_motion_packet(
                veh, _header(PacketId.MOTION, st, frame, uid)), addr)
            out.sendto(_telemetry_packet(
                veh, controls, _header(PacketId.CAR_TELEMETRY, st, frame, uid)), addr)
            out.sendto(_lap_packet(
                s_here, total_distance, lap_num, lap_ms, last_lap_ms,
                _header(PacketId.LAP_DATA, st, frame, uid)), addr)

            frame += 1
            next_tick += dt
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()  # we fell behind; resync
    except KeyboardInterrupt:
        if verbose:
            print("\nstopped.")
    finally:
        out.close()
        ctl.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--rate", type=float, default=60.0, help="packet rate (Hz)")
    ap.add_argument("--external-control", action="store_true",
                    help="disable the built-in expert; drive it over UDP")
    ap.add_argument("--delay-ms", type=float, default=0.0,
                    help="simulated actuation delay, to mimic the real game")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds (default: run forever)")
    a = ap.parse_args()
    run(host=a.host, rate_hz=a.rate, autopilot=not a.external_control,
        delay_ms=a.delay_ms, duration=a.duration)


if __name__ == "__main__":
    main()
