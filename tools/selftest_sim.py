"""Offline validation of the mock game's driving loop -- no sockets, no waiting.

Runs the expert around the synthetic track at 60 Hz in tight loop and asserts
the things that actually have to hold for it to be a useful stand-in:

  * it completes laps
  * it stays inside the track width (the previous bug had it orbiting in open
    space at a radius the circuit never reaches, while reporting healthy-looking
    telemetry -- the exact failure a naive smoke test misses)
  * it uses the brakes, so the corners are genuinely demanding
  * it produces a real speed range rather than pinned full throttle
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.sim.expert import PurePursuitExpert  # noqa: E402
from f1ai.sim.track import make_track  # noqa: E402
from f1ai.sim.vehicle import Vehicle, VehicleState  # noqa: E402

DT = 1.0 / 60.0
SIM_SECONDS = 240.0


def main() -> None:
    track = make_track()
    expert = PurePursuitExpert(track)
    veh = Vehicle()
    veh.state = VehicleState(x=float(track.x[0]), z=float(track.z[0]),
                             yaw=track.heading_at(0), v=30.0)

    print(f"track: {track.length:.0f} m, half-width {track.width:.1f} m")
    print(f"simulating {SIM_SECONDS:.0f}s at {1 / DT:.0f} Hz\n")

    laps, lap_times, offsets, speeds = 0, [], [], []
    brake_ticks = 0
    prev_s = 0.0
    lap_start = 0.0
    t = 0.0

    for _ in range(int(SIM_SECONDS / DT)):
        steer, throttle, brake = expert.act(veh)
        veh.step(steer, throttle, brake, DT)
        t += DT

        _, s_here, offset = track.nearest(veh.state.x, veh.state.z)
        offsets.append(abs(offset))
        speeds.append(veh.state.v * 3.6)
        if brake > 0.01:
            brake_ticks += 1

        if prev_s > track.length * 0.75 and s_here < track.length * 0.25:
            laps += 1
            lap_times.append(t - lap_start)
            lap_start = t
        prev_s = s_here

    worst = max(offsets)
    mean_off = sum(offsets) / len(offsets)
    brake_pct = 100.0 * brake_ticks / len(speeds)

    print(f"  laps completed     {laps}")
    if lap_times[1:]:
        print(f"  lap time           {min(lap_times[1:]):.2f} s best, "
              f"{sum(lap_times[1:]) / len(lap_times[1:]):.2f} s mean")
    print(f"  cross-track error  {mean_off:.2f} m mean, {worst:.2f} m worst")
    print(f"  speed              {min(speeds):.0f} - {max(speeds):.0f} km/h")
    print(f"  braking            {brake_pct:.1f}% of ticks")
    print()

    checks = [
        ("completes laps", laps >= 2, f"{laps} laps"),
        ("stays on track", worst < track.width,
         f"worst offset {worst:.2f} m vs half-width {track.width:.1f} m"),
        ("uses the brakes", brake_pct > 3.0, f"{brake_pct:.1f}% of ticks"),
        ("real speed range", max(speeds) - min(speeds) > 80.0,
         f"{min(speeds):.0f}-{max(speeds):.0f} km/h"),
        ("does not stall", min(speeds) > 20.0, f"min {min(speeds):.0f} km/h"),
    ]

    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<20} {detail}")
        failed += not ok

    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED -- the expert is not a usable demonstrator")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- expert is a usable demonstrator")


if __name__ == "__main__":
    main()
