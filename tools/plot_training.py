"""Plot a training run's diagnostics, arranged so failure modes are legible.

Six panels, each answering one question:

  return      is it getting better at the actual task?
  eval laps   can it complete a lap without exploration noise helping or hurting?
  lap time    the headline claim -- is it getting FASTER, not just surviving?
  critic loss is the value function learning, or diverging?
  Q vs return is the critic honest, or overestimating?
  alpha       is it still exploring, or has it committed too early?

The Q-vs-return panel is the one people skip and the one that catches the most
bugs. In a healthy run the critic's Q roughly tracks the discounted return the
policy actually achieves. If Q climbs while return stays flat, the critic is
hallucinating value and the policy is optimising a fiction -- that is a target
network or reward-scale problem, not a "needs more steps" problem.

  python tools/plot_training.py runs/run1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#56d3e3"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"


def load(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"{path} is empty -- has training produced an episode yet?")
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            v = r.get(k, "")
            try:
                vals.append(float(v) if v not in ("", "None") else np.nan)
            except ValueError:
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def smooth(y: np.ndarray, k: int = 15) -> np.ndarray:
    """Moving average that does not fabricate edge effects.

    np.convolve(..., mode="same") implicitly zero-pads, which drags both ends
    of the smoothed curve toward zero. On a return curve that renders as a
    dramatic collapse in the final steps -- run2 appeared to degrade from
    14,000 to 7,600 while the raw episodes were in fact flat at 14,400. A
    diagnostic that invents a failure is worse than no diagnostic, so the
    kernel is normalised by its actual coverage at each point instead.
    """
    m = ~np.isnan(y)
    if m.sum() < 3:
        return y
    k = max(1, min(k, m.sum() // 3))
    out = np.full_like(y, np.nan, dtype=float)
    vals, idx = y[m], np.where(m)[0]
    kern = np.ones(k)
    total = np.convolve(vals, kern, mode="same")
    coverage = np.convolve(np.ones_like(vals), kern, mode="same")
    out[idx] = total / coverage
    return out


def style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=FG, fontsize=9)
    ax.tick_params(colors=FG, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    for s in ax.spines.values():
        s.set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="run directory, e.g. runs/run1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = Path(a.run)
    d = load(run / "metrics.csv")
    step = d["step"]
    out = Path(a.out) if a.out else run / "training.png"

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor=BG)
    fig.suptitle(f"SAC training diagnostics — {run.name}", color=FG,
                 fontsize=13, x=0.02, ha="left")

    # 1. episode return
    ax = axes[0, 0]
    ax.plot(step, d["episode_return"], color=CYAN, alpha=0.25, linewidth=0.8)
    ax.plot(step, smooth(d["episode_return"]), color=CYAN, linewidth=1.8)
    style(ax, "episode return", "return")

    # 2. evaluation: distance and laps
    ax = axes[0, 1]
    m = ~np.isnan(d.get("eval_distance", np.array([np.nan])))
    if m.any():
        ax.plot(step[m], d["eval_distance"][m], "o-", color=GREEN,
                linewidth=1.6, markersize=4)
        ax.axhline(2457, color=AMBER, linestyle="--", linewidth=1,
                   label="one lap (2457 m)")
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    style(ax, "eval distance (greedy policy)", "metres")

    # 3. lap time -- the headline
    ax = axes[0, 2]
    for key, col, lab in (("eval_best_lap", GREEN, "eval (greedy)"),
                          ("train_best_lap", CYAN, "train (with noise)")):
        if key in d:
            m = ~np.isnan(d[key]) & (d[key] > 0)
            if m.any():
                ax.plot(step[m], d[key][m], "o-", color=col, linewidth=1.6,
                        markersize=3, label=lab)
    if ax.has_data():
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no completed laps yet", color=FG, alpha=0.5,
                ha="center", transform=ax.transAxes, fontsize=10)
    style(ax, "best lap time", "seconds")

    # 4. critic loss
    ax = axes[1, 0]
    ax.plot(step, d["critic_loss"], color=RED, alpha=0.25, linewidth=0.8)
    ax.plot(step, smooth(d["critic_loss"]), color=RED, linewidth=1.8)
    ax.set_yscale("log")
    style(ax, "critic loss (log scale)", "MSE")

    # 5. Q vs actual return -- honesty check
    ax = axes[1, 1]
    ax.plot(step, smooth(d["q_mean"]), color=AMBER, linewidth=1.8,
            label="critic Q")
    # Discounted return the policy actually gets, for a like-for-like scale.
    gamma = 0.99
    horizon = 1.0 / (1.0 - gamma)
    per_step = d["episode_return"] / np.maximum(d["episode_len"], 1)
    ax.plot(step, smooth(per_step * horizon), color=CYAN, linewidth=1.8,
            label="actual (discounted)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    style(ax, "critic honesty: Q vs achieved return", "value")

    # 6. alpha and entropy
    ax = axes[1, 2]
    ax.plot(step, smooth(d["alpha"]), color=GREEN, linewidth=1.8, label="alpha")
    ax.set_yscale("log")
    style(ax, "entropy coefficient", "alpha")
    if "entropy" in d and (~np.isnan(d["entropy"])).any():
        ax2 = ax.twinx()
        ax2.plot(step, smooth(d["entropy"]), color=CYAN, linewidth=1.4,
                 alpha=0.8)
        ax2.axhline(-3.0, color=AMBER, linestyle="--", linewidth=1)
        ax2.set_ylabel("entropy (dashed = target)", color=CYAN, fontsize=9)
        ax2.tick_params(colors=CYAN, labelsize=8)
        for s in ax2.spines.values():
            s.set_color(GRID)

    for ax in axes[1]:
        ax.set_xlabel("environment steps", color=FG, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110, facecolor=BG)
    print(f"wrote {out}")

    # A short text summary, for when a picture is not to hand.
    fin = lambda k: d[k][~np.isnan(d[k])] if k in d else np.array([])
    r, ed = fin("episode_return"), fin("eval_distance")
    bl = fin("eval_best_lap")
    print(f"\nepisodes: {len(d['step'])}   steps: {int(step[-1]):,}")
    if len(r) > 20:
        print(f"return:   first20 {r[:20].mean():8.1f}   "
              f"last20 {r[-20:].mean():8.1f}")
    if len(ed):
        print(f"eval dist: first {ed[0]:8.0f} m   best {ed.max():8.0f} m")
    if len(bl):
        print(f"eval lap:  first {bl[0]:8.2f} s   best {bl.min():8.2f} s")
    else:
        print("eval lap:  no completed laps yet")


if __name__ == "__main__":
    main()
