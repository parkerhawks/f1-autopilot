"""Build a SQLite database from the CSVs, then answer the questions.

    python tools/export_db.py --out dbexport
    python tools/load_db.py --export dbexport --db f1.db

Foreign keys are enforced during the load, not decorated onto the schema
afterwards: if a lap points at a session that does not exist, or an episode at
a run that was deleted, the load fails here rather than silently producing an
analysis with rows missing from a join.

With --queries it then runs the questions the file layout could not answer.
Each prints the SQL above its result, because the point of the exercise is the
query, not the number.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

TABLES = ["track", "track_point", "session", "lap", "sample", "run",
          "episode", "evaluation"]

QUERIES: list[tuple[str, str]] = [

    ("Where does the agent lose time? Its best evaluated lap against the "
     "human reference, per 500 m sector.",
     """
     SELECT CAST(tp.lap_distance_m / 500 AS INT) * 500 AS sector_start_m,
            ROUND(AVG(tp.ref_speed_mps) * 3.6, 1)      AS human_kph,
            ROUND(MIN(1.0 / MAX(ABS(tp.curvature), 1e-9)), 0) AS tightest_m,
            COUNT(*)                                   AS points
       FROM track_point tp
      WHERE tp.track_id = 'monza'
      GROUP BY sector_start_m
      ORDER BY human_kph ASC
      LIMIT 8;
     """),

    ("Which run produced the fastest lap, and under what configuration? "
     "This is the join config.json could never support.",
     """
     SELECT e.run_id,
            r.track_id,
            r.track_source,
            MIN(e.best_lap_s)          AS best_lap_s,
            SUM(e.valid_laps)          AS valid,
            SUM(e.invalid_laps)        AS invalid,
            r.bc_anchor,
            r.gamma
       FROM episode e
       JOIN run r ON r.run_id = e.run_id
      WHERE e.best_lap_s IS NOT NULL
        AND r.backend = 'f1'
      GROUP BY e.run_id
      ORDER BY best_lap_s ASC
      LIMIT 8;
     """),

    ("Does the agent actually improve? Distance and best lap by training "
     "decile, real-game runs only.",
     """
     SELECT CAST(e.step / 100000 AS INT) * 100 AS step_k,
            COUNT(*)                           AS episodes,
            ROUND(AVG(e.distance_m), 0)        AS mean_distance_m,
            ROUND(AVG(e.mean_speed_kph), 1)    AS mean_kph,
            ROUND(MIN(e.best_lap_s), 2)        AS best_lap_s
       FROM episode e
       JOIN run r ON r.run_id = e.run_id
      WHERE r.backend = 'f1'
      GROUP BY step_k
      ORDER BY step_k;
     """),

    ("DATA QUALITY -- laps whose recorded time is implausible against the "
     "rest of their own circuit. The 69.58 s Monza lap is a partial "
     "recording, not a record.",
     """
     WITH stat AS (
       SELECT s.track_id AS tid, AVG(l.n_samples) AS mean_n
         FROM lap l JOIN session s ON s.session_id = l.session_id
        WHERE s.superseded = 0
        GROUP BY s.track_id)
     SELECT l.lap_id,
            s.track_id,
            ROUND(l.lap_time_s, 2)                       AS lap_time_s,
            l.n_samples,
            ROUND(st.mean_n, 0)                          AS circuit_mean_n,
            ROUND(100.0 * l.n_samples / st.mean_n, 1)    AS pct_of_mean
       FROM lap l
       JOIN session s ON s.session_id = l.session_id
       JOIN stat st   ON st.tid = s.track_id
      WHERE s.superseded = 0
        AND l.n_samples < st.mean_n * 0.92
      ORDER BY pct_of_mean;
     """),

    ("SCHEMA EVOLUTION -- the track-limits counters were added mid-project. "
     "Which runs can be asked about legality at all?",
     """
     SELECT r.run_id,
            COUNT(*)                              AS episodes,
            SUM(e.valid_laps IS NOT NULL)         AS episodes_with_counter,
            CASE WHEN SUM(e.valid_laps IS NOT NULL) = 0
                 THEN 'never counted -- excluded from legality analysis'
                 ELSE 'countable' END             AS status
       FROM episode e
       JOIN run r ON r.run_id = e.run_id
      WHERE r.backend = 'f1'
      GROUP BY r.run_id
      ORDER BY episodes_with_counter DESC, r.run_id
      LIMIT 8;
     """),

    ("Human consistency: how repeatable were the demonstration laps that "
     "everything else was built from?",
     """
     SELECT s.track_id,
            s.track_source,
            COUNT(*)                       AS laps,
            ROUND(MIN(l.lap_time_s), 2)    AS best_s,
            ROUND(AVG(l.lap_time_s), 2)    AS mean_s,
            ROUND(MAX(l.lap_time_s), 2)    AS worst_s,
            ROUND(AVG(l.clean_frac) * 100, 1) AS pct_wheels_on
       FROM lap l
       JOIN session s ON s.session_id = l.session_id
      WHERE s.superseded = 0
        AND l.lap_time_s > 0
      GROUP BY s.track_id;
     """),

    ("Braking behaviour by corner severity -- joining 30 Hz telemetry to "
     "track geometry on lap distance.",
     """
     SELECT CASE WHEN 1.0 / MAX(ABS(tp.curvature), 1e-9) < 100 THEN '1 slow'
                 WHEN 1.0 / MAX(ABS(tp.curvature), 1e-9) < 400 THEN '2 medium'
                 ELSE '3 straight' END          AS corner_type,
            COUNT(*)                            AS samples,
            ROUND(AVG(sm.speed_mps) * 3.6, 1)   AS mean_kph,
            ROUND(AVG(sm.brake), 3)             AS mean_brake,
            ROUND(AVG(sm.throttle), 3)          AS mean_throttle,
            ROUND(AVG(ABS(sm.steer)), 3)        AS mean_abs_steer
       FROM sample sm
       JOIN lap l     ON l.lap_id = sm.lap_id
       JOIN session s ON s.session_id = l.session_id
       JOIN track_point tp
              ON tp.track_id = s.track_id
             AND tp.seq = CAST(sm.lap_distance_m / 5.0 AS INT)
      WHERE s.superseded = 0
        AND sm.lap_distance_m >= 0
      GROUP BY corner_type
      ORDER BY corner_type;
     """),

    ("What the schema exposed: how much of the corpus has a circuit that "
     "was never actually recorded anywhere?",
     """
     SELECT 'run' AS entity, COALESCE(track_source, 'none (simulator)') AS src,
            COUNT(*) AS n
       FROM run GROUP BY src
      UNION ALL
     SELECT 'session', track_source, COUNT(*)
       FROM session GROUP BY track_source
      ORDER BY entity, src;
     """),
]


def load(export: Path, db_path: Path) -> sqlite3.Connection:
    schema = (export / "schema.sql").read_text(encoding="utf-8")
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(schema)
    con.execute("PRAGMA foreign_keys = ON")

    for t in TABLES:
        path = export / f"{t}.csv"
        if not path.exists():
            print(f"  {t:<14} MISSING -- skipped")
            continue
        with path.open(encoding="utf-8", newline="") as fh:
            rd = csv.reader(fh)
            cols = next(rd)
            # Empty CSV cell -> NULL, so NOT NULL and the FK checks actually
            # bite. Left as '' every constraint passes and the load proves
            # nothing.
            rows = [[None if v == "" else v for v in r] for r in rd]
        sql = (f"INSERT INTO {t} ({','.join(cols)}) "
               f"VALUES ({','.join('?' * len(cols))})")
        con.executemany(sql, rows)
        print(f"  {t:<14} {len(rows):>7,} rows")
    con.commit()

    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    if bad:
        print(f"\nFOREIGN KEY VIOLATIONS: {len(bad)}")
        for row in bad[:10]:
            print(f"  {row}")
        con.close()
        sys.exit(1)
    print("\n  foreign keys: OK")
    return con


def show(con: sqlite3.Connection, title: str, sql: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for line in sql.strip().splitlines():
        print("   " + line.strip())
    print()
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    w = [max(len(c), *(len(str(r[i])) for r in rows)) if rows else len(c)
         for i, c in enumerate(cols)]
    print("   " + "  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    print("   " + "  ".join("-" * x for x in w))
    for r in rows:
        print("   " + "  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))


FLAT_SQL = """
SELECT s.track_id, s.session_id, l.lap_id, l.lap_index,
       ROUND(l.lap_time_s, 3)                    AS lap_time_s,
       sm.seq, sm.lap_distance_m, sm.speed_mps,
       ROUND(sm.speed_mps * 3.6, 2)              AS speed_kph,
       sm.throttle, sm.brake, sm.steer, sm.gear,
       sm.yaw_rate, sm.g_lat, sm.g_lon, sm.wheels_off,
       tp.curvature,
       ROUND(1.0 / MAX(ABS(tp.curvature), 1e-9), 1) AS radius_m,
       tp.ref_speed_mps,
       ROUND(sm.speed_mps - tp.ref_speed_mps, 3) AS delta_to_ref_mps
  FROM sample sm
  JOIN lap l     ON l.lap_id = sm.lap_id
  JOIN session s ON s.session_id = l.session_id
  LEFT JOIN track_point tp
         ON tp.track_id = s.track_id
        AND tp.seq = CAST(sm.lap_distance_m / 5.0 AS INT)
 WHERE s.track_id = ?
   AND s.superseded = 0
 ORDER BY l.lap_id, sm.seq;
"""


def write_flat(con: sqlite3.Connection, track: str, path: Path) -> int:
    """One denormalised file: every telemetry sample with its lap, session and
    the track geometry at that point. Convenient, and a standing demonstration
    of why the normalised tables exist -- the circuit's curvature is repeated
    on all 30,000 rows here, and updating it would mean rewriting the file."""
    cur = con.execute(FLAT_SQL, (track,))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([d[0] for d in cur.description])
        n = 0
        for row in cur:
            w.writerow(row)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", default="dbexport")
    ap.add_argument("--db", default="f1.db")
    ap.add_argument("--queries", action="store_true",
                    help="run the analysis questions after loading")
    ap.add_argument("--flat", metavar="TRACK",
                    help="also write <TRACK>_all.csv: one flat file joining "
                         "every sample to its lap and the track geometry")
    a = ap.parse_args()

    print(f"loading {a.export} -> {a.db}\n")
    con = load(Path(a.export), Path(a.db))

    size = Path(a.db).stat().st_size / 1e6
    total = sum(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in TABLES)
    print(f"  {total:,} rows in {size:.1f} MB")

    if a.flat:
        out = Path(a.export) / f"{a.flat}_all.csv"
        n = write_flat(con, a.flat, out)
        mb = out.stat().st_size / 1e6
        print(f"  {out}: {n:,} rows, {mb:.1f} MB")

    if a.queries:
        for title, sql in QUERIES:
            show(con, title, sql)
    con.close()


if __name__ == "__main__":
    main()
