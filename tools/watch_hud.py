"""Headless subscriber: prints what the overlay would be showing.

Useful for confirming the publisher is alive without opening a window, and for
checking the overlay's data over SSH or in CI.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.hud.bus import HudSubscriber  # noqa: E402
from f1ai.hud.panel import HudPanel  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "hud_live.png"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--save", action="store_true",
                    help="write the last frame to docs/hud_live.png")
    a = ap.parse_args()

    sub = HudSubscriber()
    t0 = time.perf_counter()
    seen = 0
    last = None

    print(f"{'t':>6} {'mode':>6} {'speed':>7} {'gear':>5} {'steer':>7} "
          f"{'thr':>5} {'brk':>5} {'lap':>7} {'laps':>5} {'best':>8}")
    print("-" * 72)
    while time.perf_counter() - t0 < a.seconds:
        s = sub.latest()
        if s is None:
            time.sleep(0.1)
            continue
        seen += 1
        last = s
        if seen % 10 == 0 or seen == 1:
            best = f"{s.best_lap_s:.2f}s" if s.best_lap_s else "-"
            print(f"{time.perf_counter() - t0:>6.1f} {s.mode:>6} "
                  f"{s.speed_kph:>7.1f} {s.gear:>5} {s.steer:>+7.3f} "
                  f"{s.throttle:>5.2f} {s.brake:>5.2f} "
                  f"{s.current_lap_s:>7.2f} {len(s.lap_times):>5} {best:>8}")
        time.sleep(0.4)

    print(f"\nreceived {seen} snapshots")
    if last is not None:
        has_frame = last.decode_frame() is not None
        print(f"last snapshot: frame={'yes' if has_frame else 'no'}, "
              f"laps={last.lap_times}")
        if a.save:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            HudPanel().render(last).convert("RGB").save(OUT)
            print(f"wrote {OUT}")
    sub.close()


if __name__ == "__main__":
    main()
