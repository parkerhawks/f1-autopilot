"""Phase 0 smoke test: prove the virtual pad reaches the game.

Run this with F1 25 focused and the car sitting in Time Trial.  It sweeps the
steering lock to lock and blips the throttle.  Watch the wheel move on screen.

If nothing happens, the problem is one of:
  - ViGEmBus driver not installed (reinstall vgamepad)
  - the game window does not have focus
  - F1 25 has bound to a different controller; unplug real pads and retry
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.control.gamepad import VirtualPad  # noqa: E402


def main() -> None:
    print("Focus the F1 25 window. Starting in 5s...")
    time.sleep(5)

    with VirtualPad() as pad:
        print("steering sweep (6s)")
        t0 = time.perf_counter()
        while (t := time.perf_counter() - t0) < 6.0:
            pad.set(math.sin(t * 1.5), 0.0, 0.0)
            time.sleep(1 / 60)

        print("throttle blips (3s)")
        pad.set(0.0, 0.0, 0.0)
        for _ in range(3):
            pad.set(0.0, 0.5, 0.0)
            time.sleep(0.4)
            pad.set(0.0, 0.0, 0.0)
            time.sleep(0.6)

        print("brake (1s)")
        pad.set(0.0, 0.0, 1.0)
        time.sleep(1.0)

    print("done -- pad released to neutral.")


if __name__ == "__main__":
    main()
