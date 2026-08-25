"""Phase 0 verification: prove the packet layout against the running game.

Three modes:

  python tools/probe_telemetry.py
      Live table of every packet type, its arrival rate and byte size.
      Confirms UDP is flowing and pins the real packet sizes.

  python tools/probe_telemetry.py --decode
      Decodes speed/steer/throttle/brake and world position for the player car.
      Drive around: if these track what you're doing, the offsets are correct.

  python tools/probe_telemetry.py --scan-floats 2
      Dumps every plausible float32 in a packet and watches which ones change.
      This is how you locate lapDistance in LapData without trusting a spec.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.telemetry.listener import TelemetryListener  # noqa: E402
from f1ai.telemetry.packets import (  # noqa: E402
    HEADER_SIZE,
    NUM_CARS,
    PacketId,
    parse_car_motion,
    parse_car_telemetry,
)

_NAMES = {int(p): p.name for p in PacketId}


def _banner(tl: TelemetryListener) -> None:
    print(f"listening on {tl._addr[0]}:{tl._addr[1]}")
    print("waiting for packets -- load into a session (Time Trial works)...")
    if not tl.wait_for_data(timeout=30.0):
        print("\nNo packets received. Check, in F1 25:")
        print("  Settings > Telemetry Settings")
        print("    UDP Telemetry ....... On")
        print("    UDP Broadcast Mode .. Off")
        print("    UDP IP .............. 127.0.0.1")
        print("    UDP Port ............ 20777")
        print("    UDP Format .......... 2025")
        print("  and that Windows Firewall is not blocking Python.")
        sys.exit(1)
    fmt = tl.packet_format
    print(f"\npacket format reported by game: {fmt}")
    if fmt != 2025:
        print(f"  !! expected 2025 -- set 'UDP Format' to 2025, or the")
        print(f"     offsets in packets.py will not apply to this stream.")
    print()


def mode_rates(tl: TelemetryListener) -> None:
    """Live per-packet arrival rate and size."""
    prev: dict[int, int] = {}
    last = time.perf_counter()
    while True:
        time.sleep(1.0)
        now = time.perf_counter()
        dt = now - last
        last = now
        stats = tl.stats()

        print("\033[2J\033[H", end="")  # clear
        print(f"player car index: {tl.player_car_index}   format: {tl.packet_format}")
        print(f"{'id':>3}  {'name':<22} {'Hz':>7}  {'bytes':>7}  {'sizes seen':<20}")
        print("-" * 68)
        for pid in sorted(stats):
            st = stats[pid]
            hz = (st.count - prev.get(pid, st.count)) / dt
            prev[pid] = st.count
            sizes = ",".join(str(s) for s in sorted(st.sizes_seen))
            print(
                f"{pid:>3}  {_NAMES.get(pid, '?'):<22} {hz:>7.1f}  "
                f"{st.last_size:>7}  {sizes:<20}"
            )
        print("\nCtrl-C to stop.  Copy the 'bytes' column into EXPECTED_SIZES.")


def mode_decode(tl: TelemetryListener) -> None:
    """Decode the fields we actually depend on, so you can eyeball them."""
    print(f"{'speed':>7} {'gear':>5} {'steer':>7} {'thr':>6} {'brk':>6} "
          f"{'  worldX':>10} {'  worldZ':>10} {'yaw':>7}")
    print("-" * 70)
    while True:
        time.sleep(0.1)
        idx = tl.player_car_index

        tel = tl.latest(PacketId.CAR_TELEMETRY)
        mot = tl.latest(PacketId.MOTION)
        if tel is None or mot is None:
            continue

        try:
            t = parse_car_telemetry(tel[1], idx)
            m = parse_car_motion(mot[1], idx)
        except struct.error as e:
            print(f"decode failed -- layout mismatch: {e}")
            continue

        print(
            f"\r{t.speed_kph:>7} {t.gear:>5} {t.steer:>+7.3f} {t.throttle:>6.2f} "
            f"{t.brake:>6.2f} {m.world_x:>10.1f} {m.world_z:>10.1f} {m.yaw:>+7.3f}",
            end="",
            flush=True,
        )


def mode_watch_bytes(tl: TelemetryListener, packet_id: int,
                     seconds: float = 60.0) -> None:
    """Report which individual bytes change, and when.

    Written because guessing offsets from a transcribed struct stopped working.
    `pitStatus` was assumed at +34 and never fired, because in Time Trial
    entering the pits does not put the car in a pit lane at all -- it drops you
    straight to the garage screen, which is a different event with a different
    flag.

    So stop guessing. Drive normally, then trigger the event, and see exactly
    which byte moved. The offsets that change are the flags; everything else is
    noise, timers, or another car's block.
    """
    pkt = None
    deadline = time.perf_counter() + 15.0
    while time.perf_counter() < deadline:
        pkt = tl.latest(PacketId(packet_id))
        if pkt is not None:
            break
        time.sleep(0.1)
    if pkt is None:
        print(f"no packets of id {packet_id} in 15s")
        return

    size = len(pkt[1])
    print(f"packet {packet_id} ({_NAMES.get(packet_id, '?')}) is {size} bytes")
    print(f"\nWatching every byte for {seconds:.0f}s.")
    print("Drive normally for the first half, then TRIGGER THE EVENT")
    print("(enter the pits, go off track -- whatever you are hunting).\n")

    history: dict[int, list[tuple[float, int]]] = {}
    t0 = time.perf_counter()
    prev: bytes | None = None

    while time.perf_counter() - t0 < seconds:
        time.sleep(0.1)
        cur = tl.latest(PacketId(packet_id))
        if cur is None:
            continue
        buf = cur[1]
        now = time.perf_counter() - t0
        if prev is not None and len(buf) == len(prev):
            for off in range(HEADER_SIZE, min(len(buf), HEADER_SIZE + 240)):
                if buf[off] != prev[off]:
                    history.setdefault(off, []).append((now, buf[off]))
        prev = buf
        if int(now) % 10 == 0:
            print(f"  {now:5.0f}s / {seconds:.0f}s", end="\r", flush=True)

    print(f"\n\n{'offset':>7} {'rel':>5} {'changes':>8} {'values seen':<28} "
          f"{'first change':>13}")
    print("-" * 70)

    rows = []
    for off, hist in history.items():
        values = sorted({v for _, v in hist})
        # Flags take a handful of distinct values; counters and timers churn
        # through hundreds and are never what we are looking for.
        if len(values) > 12:
            continue
        rows.append((len(hist), off, values, hist[0][0]))

    rows.sort(key=lambda r: (len(r[2]), r[0]))
    for n, off, values, first in rows[:30]:
        vs = ",".join(str(v) for v in values[:10])
        print(f"{off:>7} {off - HEADER_SIZE:>5} {n:>8} {vs:<28} {first:>12.1f}s")

    print("\nA flag you triggered once shows 2 values and a first-change time")
    print("matching when you did it. Cross-check the 'rel' column against the")
    print("offsets already confirmed: lapDistance +20, totalDistance +24.")


def mode_scan_floats(tl: TelemetryListener, packet_id: int,
                     seconds: float = 150.0) -> None:
    """Find a field empirically by watching which float32s behave sensibly.

    Drive a slow lap while this runs.  lapDistance is the offset whose value
    climbs monotonically from ~0 to the track length, then resets.
    """
    # Wait for this specific packet type rather than checking once. The banner
    # only waits for the FIRST packet of any kind, and at 60 Hz the different
    # types are interleaved -- a single check almost always loses the race.
    pkt = None
    deadline = time.perf_counter() + 15.0
    while time.perf_counter() < deadline:
        pkt = tl.latest(PacketId(packet_id))
        if pkt is not None:
            break
        time.sleep(0.1)

    if pkt is None:
        seen = sorted(tl.stats())
        print(f"no packets of id {packet_id} "
              f"({_NAMES.get(packet_id, '?')}) in 15s\n")
        if seen:
            print("packet types that ARE arriving:")
            for pid in seen:
                print(f"  {pid:>3}  {_NAMES.get(pid, '?')}")
            print("\nLapData is only sent during an ACTIVE session. If you are")
            print("in a menu, the garage, or a replay, drive out onto the")
            print("track first and re-run.")
        else:
            print("no packets of any type -- check the telemetry settings.")
        return
    size = len(pkt[1])
    print(f"packet {packet_id} ({_NAMES.get(packet_id)}) is {size} bytes")
    print(f"header is {HEADER_SIZE} bytes; scanning float32 at every 4-byte offset\n")
    print(f"Drive for {seconds:.0f}s and COMPLETE AT LEAST ONE FULL LAP -- crossing")
    print("the line is what distinguishes lapDistance from totalDistance.\n")

    offsets = list(range(HEADER_SIZE, size - 3, 4))
    history: dict[int, list[float]] = {o: [] for o in offsets}

    samples = int(seconds / 0.5)
    try:
        for n in range(samples):
            time.sleep(0.5)
            if n % 20 == 0:
                print(f"  {n * 0.5:>5.0f}s / {seconds:.0f}s", end="\r", flush=True)
            cur = tl.latest(PacketId(packet_id))
            if cur is None:
                continue
            buf = cur[1]
            for o in offsets:
                (v,) = struct.unpack_from("<f", buf, o)
                history[o].append(v)
    except KeyboardInterrupt:
        pass

    # A plausible lapDistance: finite, mostly increasing, spans a few hundred
    # metres or more.  Rank candidates so the real one floats to the top.
    print(f"\n{'offset':>7} {'rel':>5} {'min':>12} {'max':>12} "
          f"{'rise':>7} {'monotonic':>10} {'resets':>7}")
    print("-" * 68)
    scored = []
    for o, vals in history.items():
        vals = [v for v in vals if v == v and abs(v) < 1e9]  # drop NaN/garbage
        if len(vals) < 5:
            continue
        deltas = [b - a for a, b in zip(vals, vals[1:])]
        rises = sum(1 for d in deltas if d > 0)
        frac = rises / len(deltas)
        span = max(vals) - min(vals)
        if span < 1.0:
            continue
        # A large backwards jump is a lap rollover. This is what separates
        # lapDistance (resets at the line) from totalDistance (never does),
        # and both are ~100% monotonic otherwise.
        resets = sum(1 for d in deltas if d < -0.25 * span)
        scored.append((frac, span, o, min(vals), max(vals), resets))

    # Rank resetting fields first: that is the signature we are looking for.
    scored.sort(key=lambda r: (-(r[5] > 0), -r[0], -r[1]))
    for frac, span, o, lo, hi, resets in scored[:25]:
        print(f"{o:>7} {o - HEADER_SIZE:>5} {lo:>12.2f} {hi:>12.2f} "
              f"{span:>7.1f} {frac:>10.0%} {resets:>7}")

    if scored and not any(r[5] for r in scored):
        print("\n  note: no field reset during the scan -- it was too short to")
        print("  cross the start/finish line. Re-run and complete a full lap,")
        print("  otherwise lapDistance and totalDistance look identical.")

    print("\nThe lapDistance candidate is the one spanning hundreds/thousands of")
    print("metres at ~100% monotonic. 'rel' is its offset past the header, which")
    print("is LAP_DISTANCE_OFFSET provided the player is car 0.\n")

    # The stride is NOT simply (size - header) / 22: these packets carry a few
    # trailing bytes after the last car block (LapData ends with two car-index
    # fields). Solve for the stride across plausible trailer lengths instead.
    print("candidate strides, allowing for a trailer after the last car block:")
    body = size - HEADER_SIZE
    found = False
    for trailer in range(0, 9):
        if (body - trailer) % NUM_CARS == 0:
            stride = (body - trailer) // NUM_CARS
            if stride < 8:
                continue
            print(f"  trailer {trailer} byte(s) -> LAP_DATA_STRIDE = {stride}")
            found = True
    if not found:
        print(f"  none for a {size}-byte packet -- is this really a per-car packet?")

    print("\nDisambiguate by re-running with the player in a different car slot,")
    print("or by checking that offset + stride lands on another sane float.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=20777)
    ap.add_argument("--decode", action="store_true",
                    help="decode player car telemetry + motion")
    ap.add_argument("--scan-floats", type=int, metavar="PACKET_ID",
                    help="locate a float field empirically")
    ap.add_argument("--scan-seconds", type=float, default=150.0,
                    help="scan duration; must span a full lap (default 150)")
    ap.add_argument("--watch-bytes", type=int, metavar="PACKET_ID",
                    help="report which single bytes change while you drive; "
                         "use it to find a flag empirically instead of "
                         "guessing its offset")
    ap.add_argument("--watch-seconds", type=float, default=60.0)
    args = ap.parse_args()

    with TelemetryListener(args.host, args.port) as tl:
        _banner(tl)
        try:
            if args.watch_bytes is not None:
                mode_watch_bytes(tl, args.watch_bytes, args.watch_seconds)
            elif args.scan_floats is not None:
                mode_scan_floats(tl, args.scan_floats, args.scan_seconds)
            elif args.decode:
                mode_decode(tl)
            else:
                mode_rates(tl)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
