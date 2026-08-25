"""Run every self-test and summarise. This is the entry point for "is it broken".

    python tools/run_tests.py            # everything, ~11 seconds
    python tools/run_tests.py --quick    # skip the GPU suite
    python tools/run_tests.py buffer sac # only suites matching these names

Each suite is a standalone script that prints its own checks and exits non-zero
on failure, so any of them can also be run directly when one starts failing --
the summary here is deliberately thin, because the detail lives in the suite.

Suites are ordered from most fundamental to most dependent. When several fail
at once, fix the FIRST one: a broken packet parser or a broken buffer will make
everything downstream look broken too.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (module, what it protects, needs a GPU / slower)
# The whole suite runs in about 11 seconds, so run it constantly -- there is no
# reason to batch up changes before testing.
SUITES = [
    ("selftest_packets", "F1 25 UDP byte layouts", False),
    ("selftest_reward",  "cutting the track cannot pay", False),
    ("selftest_hud",     "overlay telemetry chain", False),
    ("selftest_sim",     "mock game drives clean laps", False),
    ("selftest_buffer",  "replay stacks match the env", False),
    ("selftest_env",     "reward ranks driving quality", False),
    ("selftest_delay",   "injected delay is real and observable", False),
    ("selftest_spinup",  "resets resume at racing speed", False),
    ("selftest_resetloop", "no infinite reset loop on invalid laps", False),
    ("selftest_pit",      "pit entry is caught immediately", False),
    ("selftest_lapvalid", "only legal laps are counted", False),
    ("selftest_geometry", "racing-line lookup survives bad lapDistance", False),
    ("selftest_sac",     "SAC maths and gradient isolation", True),
    ("selftest_warmstart", "BC warm start survives SAC", True),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("filters", nargs="*",
                    help="only run suites whose name contains one of these")
    ap.add_argument("--quick", action="store_true",
                    help="skip suites that take more than ~30s")
    a = ap.parse_args()

    selected = [
        s for s in SUITES
        if (not a.quick or not s[2])
        and (not a.filters or any(f.lower() in s[0].lower() for f in a.filters))
    ]
    if not selected:
        print("no suites matched")
        sys.exit(2)

    print(f"running {len(selected)} suite(s)\n")
    results = []
    t_all = time.perf_counter()

    for name, protects, slow in selected:
        script = ROOT / "tools" / f"{name}.py"
        print(f"  {name:<20} {protects:<38} ", end="", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, cwd=ROOT,
        )
        dt = time.perf_counter() - t0
        ok = proc.returncode == 0

        # Pull the suite's own summary line rather than re-counting here.
        tail = [ln for ln in proc.stdout.splitlines()
                if "checks pass" in ln or "FAILED" in ln]
        summary = tail[-1].strip() if tail else (
            "no summary" if ok else "crashed")
        print(f"{'PASS' if ok else 'FAIL'}  {dt:5.1f}s")
        results.append((name, ok, dt, summary, proc))

    total = time.perf_counter() - t_all
    failed = [r for r in results if not r[1]]

    print()
    for name, ok, _dt, summary, _p in results:
        mark = " " if ok else "!"
        print(f" {mark} {name:<20} {summary}")

    if failed:
        print(f"\n{'=' * 70}")
        for name, _ok, _dt, _s, proc in failed:
            print(f"\n--- {name} ---")
            out = (proc.stdout or "").strip().splitlines()
            for ln in out[-25:]:
                print(f"  {ln}")
            if proc.stderr.strip():
                print("  stderr:")
                for ln in proc.stderr.strip().splitlines()[-12:]:
                    print(f"    {ln}")

    print(f"\n{len(results) - len(failed)}/{len(results)} suites passed "
          f"in {total:.0f}s")
    if failed:
        print(f"first failure: {failed[0][0]} -- fix this one before the rest")
        sys.exit(1)


if __name__ == "__main__":
    main()
