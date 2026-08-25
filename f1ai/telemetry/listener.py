"""Threaded UDP listener for F1 25 telemetry.

The game fires packets at up to 60 Hz and will happily overrun a slow consumer.
This listener owns a socket on its own thread and keeps only the *latest* packet
of each type, because a control loop wants current state, not a backlog.  Any
consumer that needs history should record it downstream.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .packets import HEADER_SIZE, Header, PacketId, parse_header

# F1's largest packet comfortably fits; 4096 avoids any chance of truncation.
_RECV_SIZE = 4096


@dataclass
class PacketStats:
    count: int = 0
    last_size: int = 0
    last_recv_monotonic: float = 0.0
    sizes_seen: set[int] = field(default_factory=set)


class TelemetryListener:
    """Receives F1 25 UDP packets and exposes the most recent of each type.

    Usage:
        with TelemetryListener() as tl:
            pkt = tl.latest(PacketId.CAR_TELEMETRY)
    """

    # Loopback by default: the game sends to 127.0.0.1 and this is the only
    # consumer, so there is no reason to accept datagrams from the network.
    #
    # Pass host="0.0.0.0" only if you ever enable F1 25's "UDP Broadcast Mode"
    # to run a second telemetry consumer alongside this one -- broadcast
    # datagrams are addressed to the subnet, not to loopback, and a socket
    # bound to 127.0.0.1 receives none of them.
    def __init__(self, host: str = "127.0.0.1", port: int = 20777) -> None:
        self._addr = (host, port)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Guards _latest and _stats.  Held only for dict assignment, never
        # across a recv, so the reader never blocks the network thread.
        self._lock = threading.Lock()
        self._latest: dict[int, tuple[Header, bytes]] = {}
        self._stats: dict[int, PacketStats] = defaultdict(PacketStats)

        self.packet_format: int | None = None
        self.player_car_index: int = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "TelemetryListener":
        if self._thread is not None:
            raise RuntimeError("listener already started")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # A large receive buffer absorbs scheduling hiccups without dropping.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # SO_REUSEADDR lets a second consumer share the port. It is not a
        # substitute for broadcast mode -- with a directed send, delivery to
        # two bound sockets is not guaranteed on Windows and typically only one
        # receives. Turn broadcast mode on in the game if you need both.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(self._addr)
        self._sock.settimeout(0.5)  # so the thread can observe _stop
        self._thread = threading.Thread(
            target=self._run, name="f1-telemetry", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "TelemetryListener":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- receive loop ------------------------------------------------------

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                buf = self._sock.recv(_RECV_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed during shutdown

            now = time.perf_counter()
            if len(buf) < HEADER_SIZE:
                continue
            try:
                hdr = parse_header(buf)
            except (ValueError, Exception):
                continue

            if self.packet_format is None:
                self.packet_format = hdr.packet_format
            self.player_car_index = hdr.player_car_index

            with self._lock:
                self._latest[hdr.packet_id] = (hdr, buf)
                st = self._stats[hdr.packet_id]
                st.count += 1
                st.last_size = len(buf)
                st.last_recv_monotonic = now
                st.sizes_seen.add(len(buf))

    # -- access ------------------------------------------------------------

    def latest(self, packet_id: PacketId) -> tuple[Header, bytes] | None:
        """Most recent (header, raw bytes) for a packet type, or None."""
        with self._lock:
            return self._latest.get(int(packet_id))

    def stats(self) -> dict[int, PacketStats]:
        with self._lock:
            return dict(self._stats)

    def wait_for_data(self, timeout: float = 10.0) -> bool:
        """Block until any packet arrives.  False if none did."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._latest:
                    return True
            time.sleep(0.05)
        return False
