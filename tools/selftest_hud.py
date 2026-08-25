"""Validate the HUD chain without opening a window.

Covers the failure modes that would otherwise only show up as a blank or
frozen overlay during a live demo:

  * snapshots survive the JSON round trip, including the encoded camera frame
  * datagrams stay inside the UDP payload limit even with a frame attached
  * the publisher never raises when nothing is listening
  * the panel renders when fields are missing, empty, or degenerate
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.hud.bus import HudPublisher, HudSnapshot, HudSubscriber  # noqa: E402
from f1ai.hud.panel import HudPanel  # noqa: E402

TEST_PORT = 20999


def main() -> None:
    checks: list[tuple[str, bool, str]] = []
    panel = HudPanel()

    # -- publisher with no subscriber must be silent ----------------------
    lonely = HudPublisher(port=21001)
    try:
        lonely.publish(HudSnapshot(mode="TRAIN"))
        checks.append(("publish with no listener", lonely.dropped == 0, ""))
    except Exception as e:
        checks.append(("publish with no listener", False, repr(e)))
    lonely.close()

    # -- round trip -------------------------------------------------------
    sub = HudSubscriber(port=TEST_PORT)
    pub = HudPublisher(port=TEST_PORT)

    frame = (np.random.default_rng(0)
             .integers(0, 255, (96, 192), dtype=np.uint8))
    sent = HudSnapshot(
        mode="TRAIN", speed_kph=287.4, gear=7,
        steer=-0.42, throttle=0.93, brake=0.0,
        current_lap_s=12.5, last_lap_s=35.6, best_lap_s=34.1,
        lap_times=[56.0, 44.2, 38.9, 35.6, 34.1],
        env_steps=120_000, grad_steps=118_000, episodes=430,
        buffer_fill=0.62, actor_loss=-11.2, critic_loss=0.94, alpha=0.2,
    )
    sent.attach_frame(frame)
    pub.publish(sent)
    time.sleep(0.15)
    got = sub.latest()

    checks.append(("snapshot received", got is not None, ""))
    if got is not None:
        checks.append(("scalars survive round trip",
                       abs(got.speed_kph - 287.4) < 1e-6
                       and got.gear == 7 and got.env_steps == 120_000, ""))
        checks.append(("lap list survives",
                       got.lap_times == sent.lap_times,
                       f"{len(got.lap_times)} laps"))
        back = got.decode_frame()
        checks.append(("frame survives round trip",
                       back is not None and np.array_equal(back, frame),
                       f"{back.shape if back is not None else None}"))

    # -- worst-case datagram size ----------------------------------------
    # Random noise is the least compressible image PNG will ever see, so this
    # bounds the real payload rather than flattering it with a smooth render.
    import json
    from dataclasses import asdict
    size = len(json.dumps(asdict(sent)).encode())
    checks.append(("datagram under UDP limit", size < 60000,
                   f"{size / 1024:.1f} KB worst case"))

    # -- degenerate panel inputs -----------------------------------------
    cases = {
        "empty": HudSnapshot(),
        "one lap": HudSnapshot(mode="TRAIN", lap_times=[35.6]),
        "identical laps": HudSnapshot(mode="EVAL", lap_times=[34.0] * 6,
                                      best_lap_s=34.0, last_lap_s=34.0),
        "extreme actions": HudSnapshot(mode="TRAIN", steer=-1.0, throttle=1.0,
                                       brake=1.0, lap_times=[40.0, 39.0],
                                       speed_kph=340.0, gear=8),
        "no frame + train": HudSnapshot(mode="TRAIN", env_steps=1,
                                        lap_times=[50.0, 45.0, 41.0]),
    }
    for name, snap in cases.items():
        try:
            img = panel.render(snap)
            ok = img.width == 430 and img.height > 200
            checks.append((f"renders: {name}", ok, f"{img.width}x{img.height}"))
        except Exception as e:
            checks.append((f"renders: {name}", False, repr(e)))

    # -- latest() must not hang under a continuous publisher --------------
    # Regression: with a socket timeout instead of non-blocking reads, the
    # drain loop was always fed by the next datagram and never returned, so
    # the overlay froze on its first frame and showed one snapshot per run.
    flood = HudPublisher(port=TEST_PORT)
    for _ in range(400):
        flood.publish(HudSnapshot(mode="TRAIN", env_steps=1))
    t_start = time.perf_counter()
    _ = sub.latest()
    drain_ms = (time.perf_counter() - t_start) * 1000.0
    checks.append(("latest() returns promptly under flood", drain_ms < 100.0,
                   f"{drain_ms:.1f} ms to drain a 400-packet backlog"))

    # And it must stay prompt when packets keep arriving during the call.
    import threading
    stop = threading.Event()

    def keep_sending() -> None:
        while not stop.is_set():
            flood.publish(HudSnapshot(mode="TRAIN", env_steps=2))
            time.sleep(0.002)

    th = threading.Thread(target=keep_sending, daemon=True)
    th.start()
    time.sleep(0.1)
    worst = 0.0
    for _ in range(10):
        t_start = time.perf_counter()
        sub.latest()
        worst = max(worst, (time.perf_counter() - t_start) * 1000.0)
        time.sleep(0.02)
    stop.set()
    th.join(timeout=1.0)
    flood.close()
    checks.append(("latest() bounded with live publisher", worst < 100.0,
                   f"worst call {worst:.1f} ms"))

    # -- schema drift tolerance ------------------------------------------
    import socket
    raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    raw.sendto(b'{"mode":"TRAIN","unknown_field":1}', ("127.0.0.1", TEST_PORT))
    raw.sendto(b"not json at all", ("127.0.0.1", TEST_PORT))
    time.sleep(0.15)
    try:
        sub.latest()
        checks.append(("survives malformed datagrams", True, ""))
    except Exception as e:
        checks.append(("survives malformed datagrams", False, repr(e)))
    raw.close()

    pub.close()
    sub.close()

    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} {detail}")
        failed += not ok
    print()
    if failed:
        print(f"{failed}/{len(checks)} FAILED")
        sys.exit(1)
    print(f"all {len(checks)} checks pass -- HUD chain is sound")


if __name__ == "__main__":
    main()
