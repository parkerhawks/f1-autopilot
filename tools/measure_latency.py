"""Phase 0 gate: measure closed-loop control latency.

Sends a step change in steering through the virtual pad and times how long
until the game reports it back in CarTelemetryData.  That round trip is the
irreducible delay your policy operates under -- vgamepad -> ViGEm -> game input
sampling -> physics tick -> UDP send -> your socket.

Why this gate matters: if the loop is ~50 ms, the car travels ~4 m at Monza
speeds between deciding and acting.  A policy trained on frames labelled with
*simultaneous* actions will then be systematically late, and it will understeer
out of every corner.  Knowing the number lets you shift the action labels to
compensate during dataset construction, which is far cheaper than discovering
the bias after training.

Run this parked on track, engine running, in Time Trial.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.control.gamepad import VirtualPad  # noqa: E402
from f1ai.telemetry.listener import TelemetryListener  # noqa: E402
from f1ai.telemetry.packets import PacketId, parse_car_telemetry  # noqa: E402

STEP = 0.6          # steering magnitude of the test input
# Two thresholds, because the single number they replace was ambiguous.
#
# F1 25 ramps gamepad steering rather than applying it instantly (steering
# linearity / saturation settings), so time-to-0.15 measures transport latency
# PLUS the game's input smoothing. FIRST_MOVE catches the earliest moment the
# game acknowledges the input at all, which is transport alone -- and only
# transport can be improved by changing graphics settings. The gap between them
# is the ramp, which has to be compensated for instead.
FIRST_MOVE = 0.01   # |steer| that means "the input has been received"
DETECT = 0.15       # |steer| that means "the car is actually turning"
TIMEOUT = 0.5       # give up on a trial after this long
TRIALS = 20


def _read_steer(tl: TelemetryListener) -> tuple[float, int] | None:
    pkt = tl.latest(PacketId.CAR_TELEMETRY)
    if pkt is None:
        return None
    hdr, buf = pkt
    return parse_car_telemetry(buf, tl.player_car_index).steer, hdr.frame_id


def main() -> None:
    print("Closed-loop latency test.")
    print("Park the car on track in Time Trial, then leave it alone.\n")

    with TelemetryListener() as tl:
        if not tl.wait_for_data(timeout=30.0):
            print("no telemetry -- run tools/probe_telemetry.py first")
            return

        with VirtualPad() as pad:
            samples: list[float] = []
            transport: list[float] = []
            for i in range(TRIALS):
                pad.set(0.0, 0.0, 0.0)
                time.sleep(0.35)  # let the reported value settle at centre

                base = _read_steer(tl)
                if base is None or abs(base[0]) > FIRST_MOVE:
                    continue

                direction = 1.0 if i % 2 == 0 else -1.0
                t0 = time.perf_counter()
                pad.set(STEP * direction, 0.0, 0.0)

                first_ms = None
                deadline = t0 + TIMEOUT
                while time.perf_counter() < deadline:
                    cur = _read_steer(tl)
                    if cur is None:
                        time.sleep(0.001)
                        continue
                    mag = abs(cur[0])
                    if first_ms is None and mag > FIRST_MOVE:
                        first_ms = (time.perf_counter() - t0) * 1000.0
                        transport.append(first_ms)
                    if mag > DETECT:
                        dt = (time.perf_counter() - t0) * 1000.0
                        samples.append(dt)
                        ramp = dt - (first_ms if first_ms is not None else dt)
                        print(f"  trial {i + 1:>2}: {dt:6.1f} ms "
                              f"(transport {first_ms or 0:5.1f}, ramp {ramp:5.1f})")
                        break
                    time.sleep(0.001)
                else:
                    print(f"  trial {i + 1:>2}: no response (is the game focused?)")

            pad.neutral()

    if not samples:
        print("\nNo measurements. The game window must have focus to accept pad input.")
        return

    samples.sort()
    print(f"\n  n        {len(samples)}")
    print(f"  median   {statistics.median(samples):6.1f} ms")
    print(f"  mean     {statistics.mean(samples):6.1f} ms")
    print(f"  p90      {samples[int(len(samples) * 0.9) - 1]:6.1f} ms")
    print(f"  min/max  {samples[0]:.1f} / {samples[-1]:.1f} ms")

    med = statistics.median(samples)
    med_t = statistics.median(transport) if transport else None
    if med_t is not None:
        print(f"\n  transport {med_t:5.1f} ms   ramp {med - med_t:5.1f} ms")
        print("  transport is what graphics settings can improve;")
        print("  ramp is the game smoothing gamepad steering and is not a bug.")

    print()
    n_steps = max(1, round(med / 33.3))
    if med < 40:
        print("  Good. Comfortable for a 30 Hz control loop.")
    elif med < 80:
        print(f"  Workable. Include {n_steps} steps of action history in the")
        print("  observation so the delayed problem stays Markovian.")
    else:
        print(f"  High: {med:.0f} ms is {n_steps} control steps at 30 Hz.")
        print("  Try first, in order of expected effect:")
        print("    - V-Sync OFF, and NVIDIA Reflex / low-latency mode ON")
        print("    - raise the frame cap (120+); more frames = faster input")
        print("      sampling and telemetry, and a 4070 has the headroom")
        print("    - steering linearity/smoothing to 0, deadzone to 0")
        print("    - confirm UDP send rate is 60 Hz, not 20")
        print(f"  Whatever remains, carry {n_steps} steps of action history in")
        print("  the observation -- a delayed MDP without it is not Markovian,")
        print("  and the policy oscillates trying to correct what it cannot see.")


if __name__ == "__main__":
    main()
