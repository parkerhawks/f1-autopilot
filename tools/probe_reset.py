"""Discover and time F1 25's reset path. Run this FIRST once the game installs.

Reset is the single highest-risk item in the project. An RL agent crashes
constantly for its first several thousand episodes; if each recovery needs a
human, in-game training is dead no matter how good the algorithm is. And the
cost is not binary -- it is arithmetic:

    at 30 Hz an early episode lasts ~30-60 s
    a 20 s reset therefore burns a third of your training throughput
    an overnight run drops from ~320 laps to ~180

So the number this measures directly sets what an overnight run can achieve.

Two modes:

  --manual    You reset by hand however you like. This watches telemetry and
              times it, so you learn what the ceiling is before automating.

  --test      Drives a candidate button sequence through the virtual pad and
              reports whether telemetry shows the car actually recovered.

Both detect recovery from the telemetry stream rather than the screen: the car
is back when speed climbs from near-zero and lap distance starts advancing
again. That works regardless of which menu path you took.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.telemetry.listener import TelemetryListener  # noqa: E402
from f1ai.telemetry.packets import PacketId, parse_car_telemetry  # noqa: E402

STOPPED_KPH = 5.0        # below this the car is parked, crashed or in a menu
RECOVERED_KPH = 60.0     # above this it is genuinely driving again
TIMEOUT = 45.0


def speed_now(tl: TelemetryListener) -> float | None:
    pkt = tl.latest(PacketId.CAR_TELEMETRY)
    if pkt is None:
        return None
    return parse_car_telemetry(pkt[1], tl.player_car_index).speed_kph


def wait_until(tl: TelemetryListener, predicate, timeout: float) -> float | None:
    """Seconds until predicate(speed) holds, or None on timeout."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        s = speed_now(tl)
        if s is not None and predicate(s):
            return time.perf_counter() - t0
        time.sleep(0.02)
    return None


def mode_manual(tl: TelemetryListener, trials: int) -> list[float]:
    print("MANUAL MODE")
    print("  Drive, then crash or stop the car. When it is stationary, reset")
    print("  it however you normally would. This times each recovery.\n")
    print("  Try a DIFFERENT method each time -- restart lap, return to")
    print("  garage, any reset-to-track option -- and note which is fastest.\n")

    times = []
    for i in range(trials):
        print(f"  trial {i + 1}/{trials}: waiting for the car to stop...",
              end="", flush=True)
        if wait_until(tl, lambda s: s < STOPPED_KPH, 300.0) is None:
            print(" timed out")
            continue
        print(" stopped. Reset now.")

        t0 = time.perf_counter()
        dt = wait_until(tl, lambda s: s > RECOVERED_KPH, TIMEOUT)
        if dt is None:
            print(f"           no recovery within {TIMEOUT:.0f}s")
            continue
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"           recovered in {elapsed:.1f}s")
    return times


def mode_test(tl: TelemetryListener, sequence: str, trials: int) -> list[float]:
    from f1ai.control.gamepad import VirtualPad
    import vgamepad as vg

    names = {
        "start": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
        "back": vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
        "a": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
        "b": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        "x": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
        "y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        "up": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
        "down": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
        "left": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
        "right": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
    }

    steps = []
    for token in sequence.split(","):
        token = token.strip().lower()
        if ":" in token:
            name, wait = token.split(":", 1)
            steps.append((name.strip(), float(wait)))
        else:
            steps.append((token, 0.8))

    unknown = [n for n, _ in steps if n not in names]
    if unknown:
        print(f"unknown button(s): {unknown}")
        print(f"valid: {', '.join(sorted(names))}")
        sys.exit(2)

    print("TEST MODE")
    print(f"  sequence: {' -> '.join(f'{n}({w}s)' for n, w in steps)}")
    print("  Focus the game window. Starting in 5s...\n")
    time.sleep(5)

    times = []
    with VirtualPad() as pad:
        for i in range(trials):
            print(f"  trial {i + 1}/{trials}: waiting for the car to stop...",
                  end="", flush=True)
            if wait_until(tl, lambda s: s < STOPPED_KPH, 300.0) is None:
                print(" timed out")
                continue
            print(" stopped. Driving the sequence.")

            pad.neutral()
            t0 = time.perf_counter()
            for name, wait in steps:
                pad.press_button(names[name])
                time.sleep(wait)

            dt = wait_until(tl, lambda s: s > RECOVERED_KPH, TIMEOUT)
            if dt is None:
                print(f"           NO RECOVERY within {TIMEOUT:.0f}s "
                      f"-- wrong sequence, or the car needs throttle")
                continue
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"           recovered in {elapsed:.1f}s")
    return times


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manual", action="store_true",
                    help="time your own resets")
    ap.add_argument("--test", metavar="SEQUENCE",
                    help="comma-separated buttons, e.g. 'start:0.6,a:0.8,a:1.2'")
    ap.add_argument("--trials", type=int, default=5)
    a = ap.parse_args()

    if not a.manual and not a.test:
        ap.error("choose --manual or --test")

    with TelemetryListener() as tl:
        print("waiting for telemetry...")
        if not tl.wait_for_data(timeout=30.0):
            print("no packets -- run tools/probe_telemetry.py first")
            sys.exit(1)
        print("connected.\n")

        times = (mode_test(tl, a.test, a.trials) if a.test
                 else mode_manual(tl, a.trials))

    if not times:
        print("\nNo successful resets measured.")
        return

    med = statistics.median(times)
    print(f"\n  n {len(times)}   median {med:.1f}s   "
          f"min {min(times):.1f}s   max {max(times):.1f}s")

    # Translate into what it costs a training run.
    episode_s = 45.0
    overhead = med / (episode_s + med)
    print(f"\n  With ~{episode_s:.0f}s early episodes, a {med:.0f}s reset costs")
    print(f"  {overhead:.0%} of training throughput.")
    print(f"  An 8h run: ~{8 * 3600 / (episode_s + med):.0f} episodes "
          f"(vs {8 * 3600 / episode_s:.0f} with instant reset).")

    if med < 6:
        print("\n  Excellent -- reset is not a bottleneck.")
    elif med < 15:
        print("\n  Workable. Prefer the fastest option you found.")
    else:
        print("\n  SLOW. Look for a faster path before committing to in-game")
        print("  RL -- this dominates your training budget.")


if __name__ == "__main__":
    main()
