"""Evaluate a trained policy against track limits, not just the clock.

A lap time alone cannot tell you whether a policy is fast or merely cheating.
The reward pays roughly 2.7 per step for progress and charges only 0.6 per
metre for leaving the racing surface, so cutting a corner can be strictly
profitable -- and a reward-hacking policy produces a beautiful lap-time curve
while driving through the scenery.

This measures what the reward does not enforce:

  * peak and mean cross-track error against the track half-width
  * fraction of the lap spent outside the surface
  * whether any lap would survive a real track-limits rule
  * the driven line, plotted against the circuit

Compare against the pure-pursuit expert on the same track with --expert.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import (  # noqa: E402
    CONTROL_HZ, FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM,
    MockBackend, RacingEnv,
)  # noqa: F401
from f1ai.rl.sac import SACAgent  # noqa: E402
from f1ai.sim.expert import PurePursuitExpert  # noqa: E402

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
CYAN, GREEN, RED, AMBER = "#56d3e3", "#3fb950", "#f85149", "#d29922"


def rollout(env: RacingEnv, policy, steps: int) -> dict:
    frames, state = env.reset()
    xs, zs, offs, spds, acts = [], [], [], [], []
    total = 0.0
    for _ in range(steps):
        a = policy(frames, state, env)
        res = env.step(a)
        total += res.reward
        veh = env.backend.veh.state
        xs.append(veh.x)
        zs.append(veh.z)
        offs.append(abs(res.info["offset"]))
        spds.append(res.info["speed_kph"])
        acts.append(a.copy())
        frames, state = res.frames, res.state
        if res.terminated:
            break
    return {
        "x": np.array(xs), "z": np.array(zs),
        "offset": np.array(offs), "speed": np.array(spds),
        "actions": np.array(acts), "return": total,
        "laps": [t for t in env.lap_times if t > 0],
        "crashed": res.terminated,
        "distance": res.info["distance"],
    }


def report(name: str, r: dict, width: float) -> dict:
    off = r["offset"]
    outside = off > width
    frac = float(outside.mean())
    worst = float(off.max())
    laps = r["laps"]
    print(f"\n{name}")
    print(f"  laps completed     {len(laps)}")
    if laps:
        print(f"  best / mean lap    {min(laps):.2f}s / {np.mean(laps):.2f}s")
    print(f"  distance           {r['distance']:.0f} m")
    print(f"  speed              {r['speed'].min():.0f} - {r['speed'].max():.0f}"
          f" km/h (mean {r['speed'].mean():.0f})")
    print(f"  cross-track error  mean {off.mean():.2f} m, worst {worst:.2f} m"
          f"  (half-width {width:.1f} m)")
    print(f"  outside surface    {frac:6.2%} of steps"
          f"   {'<-- CUTTING' if frac > 0.02 else ''}")
    print(f"  crashed            {r['crashed']}")
    return {"frac_out": frac, "worst": worst,
            "best_lap": min(laps) if laps else None, "laps": len(laps)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", nargs="?", default=None,
                    help="path to a .pt checkpoint")
    ap.add_argument("--expert", action="store_true",
                    help="also evaluate the pure-pursuit expert")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--delay-ms", type=float, default=None,
                    help="actuation delay to evaluate under; defaults to "
                         "whatever the run was TRAINED with, read from "
                         "config.json next to the checkpoint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    results = {}
    track = None

    # Evaluate under the SAME dynamics the policy was trained in. Evaluating a
    # delay-trained policy in a delay-free world flatters it, and the resulting
    # lap time describes an environment that does not exist.
    delay_ms = a.delay_ms
    if delay_ms is None and a.checkpoint:
        cfg_path = Path(a.checkpoint).parent / "config.json"
        if cfg_path.exists():
            try:
                delay_ms = json.loads(cfg_path.read_text()).get("delay_ms", 0.0)
            except ValueError:
                delay_ms = 0.0
    delay_steps = int(round((delay_ms or 0.0) / 1000.0 * CONTROL_HZ))
    if delay_steps:
        print(f"evaluating with {delay_ms:.0f} ms actuation delay "
              f"({delay_steps} steps), as trained")

    if a.checkpoint:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        agent = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                         device=dev)
        agent.load_state_dict(torch.load(a.checkpoint, map_location=dev,
                                         weights_only=False))
        env = RacingEnv(MockBackend(seed=a.seed,
                                    actuation_delay_steps=delay_steps))
        track = env.backend.track

        def pol(frames, state, _env):
            return SACAgent.to_env_action(
                agent.act(frames, state, deterministic=True))

        results["SAC policy"] = rollout(env, pol, a.steps)

    if a.expert or not a.checkpoint:
        # The expert gets the same delay, so the comparison is like for like.
        env = RacingEnv(MockBackend(seed=a.seed,
                                    actuation_delay_steps=delay_steps))
        track = track or env.backend.track
        exp = PurePursuitExpert(env.backend.track)

        def pol(frames, state, e):
            return np.array(exp.act(e.backend.veh), np.float32)

        results["pure-pursuit expert"] = rollout(env, pol, a.steps)

    summ = {n: report(n, r, track.width) for n, r in results.items()}

    # -- verdict ----------------------------------------------------------
    print()
    if "SAC policy" in summ:
        s = summ["SAC policy"]
        if s["frac_out"] > 0.02:
            print("VERDICT: lap times are NOT legitimate -- the policy spends")
            print(f"  {s['frac_out']:.1%} of the lap off the racing surface.")
            print("  Raise W_OFFTRACK or terminate sooner in env.py, retrain.")
        elif s["best_lap"] is None:
            print("VERDICT: no completed laps.")
        else:
            print(f"VERDICT: clean. {s['best_lap']:.2f}s best lap with"
                  f" {s['frac_out']:.2%} of steps off surface.")
            if "pure-pursuit expert" in summ and summ["pure-pursuit expert"]["best_lap"]:
                e = summ["pure-pursuit expert"]["best_lap"]
                d = e - s["best_lap"]
                print(f"  vs expert {e:.2f}s: {d:+.2f}s "
                      f"({100 * d / e:+.1f}%)")

    # -- plot -------------------------------------------------------------
    out = Path(a.out) if a.out else Path("runs") / "eval_lines.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 6),
                             facecolor=BG)
    axes = np.atleast_1d(axes)

    th = np.arange(len(track.x))
    nxt = (th + 1) % len(track.x)
    tx, tz = track.x[nxt] - track.x, track.z[nxt] - track.z
    nrm = np.hypot(tx, tz)
    nrm[nrm < 1e-9] = 1.0
    lx, lz = track.x - tz / nrm * track.width, track.z + tx / nrm * track.width
    rx, rz = track.x + tz / nrm * track.width, track.z - tx / nrm * track.width

    for ax, (name, r) in zip(axes, results.items()):
        ax.set_facecolor(BG)
        ax.plot(lx, lz, color=GRID, linewidth=1)
        ax.plot(rx, rz, color=GRID, linewidth=1)
        off = r["offset"]
        bad = off > track.width
        ax.scatter(r["x"][~bad], r["z"][~bad], c=r["speed"][~bad], s=1.5,
                   cmap="viridis")
        if bad.any():
            ax.scatter(r["x"][bad], r["z"][bad], color=RED, s=4,
                       label=f"off surface ({bad.mean():.1%})")
            ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
        best = min(r["laps"]) if r["laps"] else None
        ax.set_title(f"{name}\n{best:.2f}s best" if best else f"{name}\nno lap",
                     color=FG, fontsize=11)
        ax.set_aspect("equal")
        ax.tick_params(colors=FG, labelsize=8)
        for s_ in ax.spines.values():
            s_.set_color(GRID)

    ax = axes[-1]
    ax.set_facecolor(BG)
    for (name, r), col in zip(results.items(), (CYAN, AMBER)):
        ax.hist(r["offset"], bins=60, alpha=0.6, color=col, label=name)
    ax.axvline(track.width, color=RED, linestyle="--",
               label=f"track edge ({track.width:.0f} m)")
    ax.set_title("cross-track error distribution", color=FG, fontsize=11)
    ax.set_xlabel("distance from centreline (m)", color=FG)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    ax.tick_params(colors=FG, labelsize=8)
    for s_ in ax.spines.values():
        s_.set_color(GRID)

    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
