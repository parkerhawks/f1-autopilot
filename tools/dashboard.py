"""Build a self-contained HTML dashboard from one or more training runs.

    python tools/dashboard.py runs/run2
    python tools/dashboard.py runs/run1 runs/run2 --open

Charts are pre-rendered as inline SVG and the whole page is a single file with
no external requests, so it works offline, survives being emailed, and can be
published as an artifact unchanged. Re-run it mid-training -- `train_sac.py`
flushes its CSV every episode, so the dashboard is never more than one episode
behind.

Deliberately reads the same `metrics.csv` that `plot_training.py` does. That
file is the single source of truth for a run; anything that disagrees with it
is wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import webbrowser
from pathlib import Path

ACCENT = "var(--accent)"
PASS = "var(--pass)"
WARN = "var(--warn)"
INK3 = "var(--ink-3)"


# ---------------------------------------------------------------- data ----

def load(run: Path) -> dict:
    path = run / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"no metrics.csv in {run}")
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"{path} is empty")

    def col(name: str) -> list[tuple[float, float]]:
        """(step, value) pairs, skipping blanks. Never interpolates."""
        out = []
        for r in rows:
            v = r.get(name, "")
            if v in ("", "None"):
                continue
            try:
                out.append((float(r["step"]), float(v)))
            except (ValueError, KeyError):
                continue
        return out

    cfg = {}
    cfg_path = run / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except ValueError:
            pass

    return {
        "name": run.name,
        "config": cfg,
        "return": col("episode_return"),
        "ep_len": col("episode_len"),
        "eval_lap": col("eval_best_lap"),
        "train_lap": col("train_best_lap"),
        "eval_dist": col("eval_distance"),
        "eval_laps": col("eval_laps"),
        "critic": col("critic_loss"),
        "q": col("q_mean"),
        "alpha": col("alpha"),
        "entropy": col("entropy"),
        "steps": max((float(r["step"]) for r in rows if r.get("step")), default=0),
    }


def smooth(pts: list[tuple[float, float]], k: int = 12) -> list[tuple[float, float]]:
    """Moving average normalised by actual coverage.

    Not mode="same" convolution: zero-padding drags both ends toward zero and
    invents collapses that never happened. That bug once made a flat 14,400
    return look like it fell to 7,600.
    """
    if len(pts) < 4:
        return pts
    k = max(1, min(k, len(pts) // 3))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - k // 2), min(len(ys), i + k // 2 + 1)
        window = ys[lo:hi]
        out.append((xs[i], sum(window) / len(window)))
    return out


# --------------------------------------------------------------- charts ----

def line_chart(series: list[dict], width: int = 640, height: int = 200,
               y_label: str = "", invert: bool = False,
               y_floor: float | None = None, marker: bool = False) -> str:
    """Inline SVG line chart. `series` entries: {pts, color, label, dashed}."""
    pad_l, pad_r, pad_t, pad_b = 52, 12, 14, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_pts = [p for s in series for p in s["pts"]]
    if len(all_pts) < 2:
        return (f'<svg viewBox="0 0 {width} {height}" role="img">'
                f'<text x="{width/2}" y="{height/2}" text-anchor="middle" '
                f'class="empty">not enough data yet</text></svg>')

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y_floor is not None:
        y0 = min(y0, y_floor)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1.0
    span_y = (y1 - y0) * 1.12
    y0 -= (y1 - y0) * 0.06

    def sx(x: float) -> float:
        return pad_l + plot_w * (x - x0) / max(1e-9, x1 - x0)

    def sy(y: float) -> float:
        t = (y - y0) / span_y
        return pad_t + plot_h * (t if invert else 1.0 - t)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'preserveAspectRatio="none">']

    for frac in (0.0, 0.5, 1.0):
        gy = pad_t + plot_h * frac
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" '
                     f'y2="{gy:.1f}" class="grid"/>')
        val = y0 + span_y * (frac if invert else 1.0 - frac)
        parts.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" '
                     f'text-anchor="end" class="tick">{fmt(val)}</text>')

    parts.append(f'<text x="{pad_l}" y="{height - 6}" class="tick">0</text>')
    parts.append(f'<text x="{width - pad_r}" y="{height - 6}" '
                 f'text-anchor="end" class="tick">{fmt(x1)} steps</text>')

    for s in series:
        pts = s["pts"]
        if len(pts) < 2:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                     for i, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="4 3"' if s.get("dashed") else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" '
                     f'stroke-width="{s.get("w", 2)}" stroke-linejoin="round"'
                     f'{dash} opacity="{s.get("opacity", 1)}"/>')
        if marker:
            for x, y in pts:
                parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" '
                             f'r="2.6" fill="{s["color"]}"/>')

    if y_label:
        parts.append(f'<text x="4" y="12" class="axis">{y_label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def fmt(v: float) -> str:
    a = abs(v)
    if a >= 1_000_000:
        return f"{v / 1e6:.1f}M"
    if a >= 10_000:
        return f"{v / 1000:.0f}k"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.3f}"


def lap_str(s: float) -> str:
    m, sec = divmod(s, 60.0)
    return f"{int(m)}:{sec:06.3f}" if m else f"{sec:.2f}s"


# ----------------------------------------------------------------- page ----

def stat(label: str, value: str, sub: str = "", tone: str = "") -> str:
    cls = f" {tone}" if tone else ""
    return (f'<div class="stat{cls}"><p class="k">{label}</p>'
            f'<p class="v">{value}</p>'
            + (f'<p class="s">{sub}</p>' if sub else "") + "</div>")


def panel(title: str, note: str, svg: str, legend: str = "") -> str:
    return (f'<figure class="panel"><figcaption><h3>{title}</h3>'
            f'<p>{note}</p>{legend}</figcaption>'
            f'<div class="chart">{svg}</div></figure>')


def swatch(color: str, label: str) -> str:
    return (f'<span class="sw"><i style="background:{color}"></i>{label}</span>')


def build(runs: list[dict]) -> str:
    primary = runs[-1]

    laps = [v for _, v in primary["eval_lap"]]
    best = min(laps) if laps else None
    first = laps[0] if laps else None
    gain = (first - best) if (laps and first is not None) else None

    ret = [v for _, v in primary["return"]]
    ret_last = sum(ret[-10:]) / len(ret[-10:]) if len(ret) >= 10 else (
        ret[-1] if ret else 0)

    cfg = primary["config"]
    steps = int(primary["steps"])
    est_min = steps / 30.0 / 60.0

    stats = [
        stat("best lap", lap_str(best) if best else "—",
             "greedy evaluation", "good" if best else ""),
        stat("improvement", f"−{gain:.2f}s" if gain and gain > 0 else "—",
             "first eval to best", "good" if gain and gain > 0 else ""),
        stat("env steps", f"{steps:,}",
             f"≈{est_min:.0f} min at 30 Hz in-game"),
        stat("mean return", fmt(ret_last), "last 10 episodes"),
    ]

    # --- lap time, the headline ---
    lap_series = []
    if primary["eval_lap"]:
        lap_series.append({"pts": primary["eval_lap"], "color": PASS,
                           "label": "eval", "w": 2.4})
    if primary["train_lap"]:
        lap_series.append({"pts": smooth(primary["train_lap"]), "color": ACCENT,
                           "label": "train", "w": 1.8, "opacity": 0.75})
    lap_legend = ('<div class="legend">'
                  + swatch(PASS, "greedy evaluation")
                  + swatch(ACCENT, "training (with exploration noise)")
                  + "</div>") if lap_series else ""

    body = [
        panel("Lap time",
              "The only number that ultimately counts. Evaluation runs with "
              "exploration switched off, so it is the honest pace.",
              line_chart(lap_series, height=230, marker=True), lap_legend),
        panel("Episode return",
              "Total reward per episode. Rises as the agent learns to make "
              "progress without leaving the surface.",
              line_chart([{"pts": smooth(primary["return"]), "color": ACCENT}],
                         y_floor=0)),
        panel("Evaluation distance",
              "How far the greedy policy gets before crashing. Flat at the "
              "episode cap means it no longer crashes at all.",
              line_chart([{"pts": primary["eval_dist"], "color": PASS}],
                         marker=True, y_floor=0)),
        panel("Critic honesty",
              "Q should track the return actually achieved. Q climbing while "
              "return stays flat means the critic is hallucinating value.",
              line_chart([
                  {"pts": smooth(primary["q"]), "color": WARN, "label": "Q"},
              ]),
              '<div class="legend">' + swatch(WARN, "critic Q estimate")
              + "</div>"),
        panel("Exploration",
              "Alpha weighs exploration against exploitation and is tuned "
              "automatically. Stuck high means the policy never commits; "
              "collapsing early means it stopped exploring too soon.",
              line_chart([{"pts": smooth(primary["alpha"]), "color": PASS}])),
        panel("Critic loss",
              "Should rise early as the agent finds higher-reward regions, "
              "then settle. Exploding means Q-divergence.",
              line_chart([{"pts": smooth(primary["critic"]), "color": WARN}],
                         y_floor=0)),
    ]

    # --- run comparison ---
    compare = ""
    if len(runs) > 1:
        rows = []
        for r in runs:
            rl = [v for _, v in r["eval_lap"]]
            rows.append(
                f"<tr><td class='mono'>{r['name']}</td>"
                f"<td class='num'>{int(r['steps']):,}</td>"
                f"<td class='num'>{lap_str(min(rl)) if rl else '—'}</td>"
                f"<td class='num'>{len(rl)}</td></tr>")
        compare = (
            '<section><p class="eyebrow">Comparison</p><h2>Runs</h2>'
            '<div class="scroll"><table><tr><th>run</th><th>steps</th>'
            '<th>best lap</th><th>evals</th></tr>'
            + "".join(rows) + "</table></div></section>")

    cfg_bits = ""
    if cfg:
        sac = cfg.get("sac", {})
        items = [
            ("batch", sac.get("batch_size")), ("gamma", sac.get("gamma")),
            ("tau", sac.get("tau")), ("actor lr", sac.get("lr")),
            ("warmup", cfg.get("warmup")), ("buffer", cfg.get("capacity")),
        ]
        cfg_bits = "".join(f'<span class="chip">{k} {v}</span>'
                           for k, v in items if v is not None)

    return TEMPLATE.format(
        name=primary["name"],
        stats="".join(stats),
        panels="".join(body),
        compare=compare,
        config=cfg_bits,
    )


TEMPLATE = """<title>f1-autopilot — {name}</title>
<style>
:root{{
  --ground:#F6F7F9; --surface:#FFFFFF; --surface-2:#EFF1F4; --line:#DDE0E6;
  --ink:#13161C; --ink-2:#535964; --ink-3:#868C99;
  --accent:#6832CE; --accent-soft:#EEE8FC;
  --pass:#1A7A48; --warn:#9C6400; --fail:#BC332A;
  --mono:ui-monospace,"Cascadia Mono",Consolas,monospace;
  --sans:-apple-system,"Segoe UI Variable Text","Segoe UI",Roboto,sans-serif;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    --ground:#0C0F15; --surface:#131720; --surface-2:#1A1F29; --line:#252B37;
    --ink:#E3E7EC; --ink-2:#99A1B0; --ink-3:#6B7382;
    --accent:#A87DFF; --accent-soft:#1E1733;
    --pass:#42C079; --warn:#D6A03C; --fail:#E9584C;
  }}
}}
:root[data-theme="dark"]{{
  --ground:#0C0F15; --surface:#131720; --surface-2:#1A1F29; --line:#252B37;
  --ink:#E3E7EC; --ink-2:#99A1B0; --ink-3:#6B7382;
  --accent:#A87DFF; --accent-soft:#1E1733;
  --pass:#42C079; --warn:#D6A03C; --fail:#E9584C;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font:15.5px/1.6 var(--sans);-webkit-font-smoothing:antialiased}}
.wrap{{max-width:960px;margin:0 auto;padding:48px 24px 88px;
  display:flex;flex-direction:column;gap:44px}}
.eyebrow{{font:11.5px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 10px}}
h1{{font-size:32px;line-height:1.14;letter-spacing:-.02em;font-weight:640;margin:0}}
h2{{font-size:20px;font-weight:620;letter-spacing:-.012em;margin:0 0 14px}}
h3{{font-size:15px;font-weight:640;margin:0 0 4px}}
p{{margin:0}}
header{{border-bottom:2px solid var(--ink);padding-bottom:20px}}
.chips{{display:flex;flex-wrap:wrap;gap:7px;margin-top:14px}}
.chip{{font:11.5px/1 var(--mono);padding:6px 10px;border-radius:20px;
  border:1px solid var(--line);color:var(--ink-2);background:var(--surface)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.stat{{background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:16px 18px}}
.stat.good{{border-color:var(--pass)}}
.stat .k{{font:11px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:9px}}
.stat .v{{font:25px/1.1 var(--mono);font-variant-numeric:tabular-nums;
  letter-spacing:-.02em}}
.stat.good .v{{color:var(--pass)}}
.stat .s{{font-size:12.5px;color:var(--ink-3);margin-top:6px}}
.panels{{display:flex;flex-direction:column;gap:16px}}
.panel{{margin:0;background:var(--surface);border:1px solid var(--line);
  border-radius:8px;padding:18px 20px;display:flex;flex-direction:column;gap:14px}}
.panel figcaption p{{color:var(--ink-2);font-size:13.5px;max-width:64ch}}
.chart{{overflow-x:auto}}
svg{{display:block;width:100%;height:auto;min-width:420px}}
.grid{{stroke:var(--line);stroke-width:1}}
.tick{{font:10.5px var(--mono);fill:var(--ink-3)}}
.axis{{font:10.5px var(--mono);fill:var(--ink-3);letter-spacing:.08em}}
.empty{{font:13px var(--sans);fill:var(--ink-3)}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;margin-top:8px}}
.sw{{display:inline-flex;align-items:center;gap:6px;font:11.5px var(--mono);
  color:var(--ink-2)}}
.sw i{{width:11px;height:3px;border-radius:2px;display:inline-block}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;
  background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:14px;min-width:420px}}
th{{font:11px var(--mono);letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);text-align:left;padding:11px 16px;font-weight:400;
  border-bottom:1px solid var(--line)}}
td{{padding:11px 16px;border-bottom:1px solid var(--line);color:var(--ink-2)}}
tr:last-child td{{border-bottom:0}}
td.mono{{font:13px var(--mono);color:var(--ink)}}
td.num{{font-variant-numeric:tabular-nums;text-align:right}}
@media (max-width:560px){{.wrap{{padding:32px 16px 60px;gap:32px}}h1{{font-size:26px}}}}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">f1-autopilot / training run</p>
  <h1>{name}</h1>
  <div class="chips">{config}</div>
</header>
<section class="stats">{stats}</section>
<section class="panels">{panels}</section>
{compare}
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--open", action="store_true",
                    help="open in the default browser when done")
    a = ap.parse_args()

    runs = [load(Path(r)) for r in a.runs]
    html = build(runs)

    out = Path(a.out) if a.out else Path(runs[-1]["name"]).with_suffix(".html")
    if not a.out:
        out = Path("runs") / f"{runs[-1]['name']}-dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB, self-contained)")
    for r in runs:
        laps = [v for _, v in r["eval_lap"]]
        print(f"  {r['name']:<10} {int(r['steps']):>8,} steps   "
              f"best {lap_str(min(laps)) if laps else '—'}")

    if a.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
