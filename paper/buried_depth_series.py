#!/usr/bin/env python3
"""Regolith depth-dose curve from buried-habitat FLOOD runs (replaces regolith_sweep).

Why. lunarsim.regolith_sweep lights a regolith slab with a 22 cm beam spot. The
2026-11-24 spot probe (paper/regolith_spot_probe.py) showed that spot has no
lateral equilibrium at depth: H/He skin dose rises x1.7 (0.3 m) and >=x3.4 (1 m)
from 22 -> 180 cm and is still climbing, and the slab illumination guard forbids
larger spots. Its depth numbers are retired. This harness rebuilds the curve with
the production flood geometry instead: the full 900 cm sky source (rings x azimuth
parallel discs, solid-angle ring weights), the real ground, the outer-gauge
normalisation and CAL exactly as every habitat result uses them.

Design series. The shipped buried habitat (geometry._buried; same as
paper/weighted_source_fom.designs()["buried150"]): inner radius 300 cm, 1 cm Al
ceiling (2.7 g/cm2), regolith overburden of depth d, side berm max(200 cm, d).
Only the overburden changes; the TOTAL overhead areal density is the series
coordinate (Al ceiling + regolith, 1.78 g/cm3). Note what this curve is and is not:
  * it is a BURIED-HABITAT curve, not a semi-infinite slab: the side berm is
    >= 356 g/cm2, so at thin overburden the grazing sky is already cut off and
    the dose falls more slowly than a slab would predict at low depth;
  * above d = 200 cm (overhead > ~359 g/cm2) the berm thickens with d too;
  * the default series stops at 500 g/cm2, the deepest point whose enclosing
    corner stays 'ok' inside the 900 cm footprint (dosimetry.beam_footprint_flag).
55.6 and 193.7 g/cm2 are the regolith_v2 0.3 m / 1 m column depths, so the old
sweep and the probe can be read against this curve directly.

Runs: plain Beam source, continuous sampling, full composition, converge_on both,
fixed rounds (target_rel_err 0, min = max = rounds) via weighted_source_fom.run_one,
so every point has the same statistics budget. Seeds: round k uses seed
k+1+seed_offset; repeat with a new --seed-offset for an independent sample, and
the report pools samples per depth (inverse variance, dose-equivalent errors).

Report: per depth, phantom and skin annual E (mSv/yr) and absorbed D (mGy/d) with
errors, E/E(first depth), and the local attenuation length between neighbouring
points, lambda = dx / ln(E1/E2) (g/cm2), with its 1-sigma.

TARGET-BLIND: no external dose enters the run or the report. The one external
number printed (Blunier 2025, ICRU sphere ~213 mSv/yr at 45 g/cm2) is labelled
orientation only, not a gate. CAL 0.94932 untouched.

Usage
-----
  python3 paper/buried_depth_series.py run --dry-run
  # PC, in tmux (one line; ~a few hours per depth at the full tier):
  .venv/bin/python paper/buried_depth_series.py run | tee runs/buried_depth_s0.log
  .venv/bin/python paper/buried_depth_series.py report
Rows are appended after every depth; rerun with --depths to fill gaps or with a
new --seed-offset for another sample. Last row per (depth, seed_offset) wins.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "paper"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import weighted_source_fom as wsf          # noqa: E402  (designs, run_one, pool, _norm)

CSV_DEFAULT = _REPO / "paper" / "buried_depth_series.csv"
DEPTHS_GCM2 = (25.0, 55.6, 100.0, 150.0, 193.7, 300.0, 500.0)
FIELDS = ["design", "overhead_gcm2", "burial_depth_cm", "side_gcm2", "footprint",
          "areal_gcm2", "tier", "rounds", "wall_s", "total_primaries",
          "phantom_msv", "phantom_rel", "phantom_eq_rel", "skin_msv", "skin_rel",
          "skin_eq_rel", "phantom_mgy_d", "skin_mgy_d", "fluence_out", "fluence_in",
          "seed_offset", "provenance", "utc", "phantom_species"]
ORIENTATION = ("orientation only (not a gate): Blunier 2025 ICRU sphere ~213 mSv/yr "
               "at 45 g/cm2; retired regolith_v2 (22 cm spot) skin 73.7 at 55.6 and "
               "25.0 at 193.7 g/cm2.")


def _base():
    return wsf.designs()["buried150"]


def _ceiling_gcm2() -> float:
    from lunarsim.spec import MATERIALS
    b = _base()
    return sum(w.thickness_cm * MATERIALS[w.material]["density"] for w in b.walls)


def buried_spec(overhead_gcm2: float):
    """The buried150 design with its regolith overburden set so the TOTAL overhead
    areal density (Al ceiling + regolith) equals overhead_gcm2."""
    from lunarsim.spec import MATERIALS
    ceil = _ceiling_gcm2()
    if overhead_gcm2 <= ceil:
        raise ValueError(f"overhead {overhead_gcm2:g} g/cm2 must exceed the "
                         f"{ceil:g} g/cm2 Al ceiling")
    depth = (overhead_gcm2 - ceil) / MATERIALS["regolith"]["density"]
    s = dataclasses.replace(_base(), name=f"wdb_buried{overhead_gcm2:g}",
                            burial_depth_cm=depth)
    s.validate()
    return s


def describe(overhead_gcm2: float) -> dict:
    from lunarsim import dosimetry
    from lunarsim.geometry import _buried_side_cm
    from lunarsim.spec import MATERIALS
    s = buried_spec(overhead_gcm2)
    return {"design": f"bur{overhead_gcm2:g}", "overhead_gcm2": overhead_gcm2,
            "burial_depth_cm": round(s.burial_depth_cm, 2),
            "side_gcm2": round(_buried_side_cm(s) * MATERIALS["regolith"]["density"], 1),
            "footprint": dosimetry.beam_footprint_flag(s)}


def check_depths(depths) -> list:
    """Problems that would make a depth's result untrustworthy (empty = fine)."""
    out = []
    for g in depths:
        try:
            f = describe(g)["footprint"]
        except ValueError as e:
            out.append(str(e))
            continue
        if f != "ok":
            out.append(f"{g:g} g/cm2: enclosing corner '{f}' against the 900 cm "
                       "source footprint (dose would read low)")
    return out


def append_row(path: Path, row: dict) -> None:
    new =not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_run(args) -> None:
    from lunarsim.bridge import QUICK_LOOK, FULL_RUN
    tier = {"quick": QUICK_LOOK, "full": FULL_RUN}[args.tier]
    depths = sorted(args.depths)
    bad = check_depths(depths)
    if bad and not args.allow_marginal:
        raise SystemExit("refusing:\n  " + "\n  ".join(bad)
                         + "\n(--allow-marginal to run anyway; result is a lower bound)")
    print(f"tier {tier.name} ({tier.rings}x{tier.azimuth}x{tier.histories}), rounds "
          f"{args.rounds}, seed offset {args.seed_offset}, csv {args.csv}")
    for g in depths:
        d = describe(g)
        print(f"  {d['design']:9s} overhead {g:6.1f} g/cm2  regolith {d['burial_depth_cm']:6.1f} cm"
              f"  side {d['side_gcm2']:6.1f} g/cm2  footprint {d['footprint']}")
    if args.dry_run:
        return
    for g in depths:
        d = describe(g)
        print(f"== {d['design']}", flush=True)
        row = wsf.run_one(buried_spec(g), tier, "beam", args.rounds, 0.0, args.seed_offset)
        row.update(d)
        append_row(args.csv, row)
        print(f"   phantom {row['phantom_msv']:.1f} mSv/yr +/- "
              f"{100 * float(row['phantom_eq_rel'] or row['phantom_rel']):.1f}%  "
              f"skin {row['skin_msv']:.1f} +/- "
              f"{100 * float(row['skin_eq_rel'] or row['skin_rel']):.1f}%  "
              f"wall {row['wall_s']:.0f} s", flush=True)
    report(args.csv)


def atten_length(x1, e1, r1, x2, e2, r2):
    """lambda = (x2-x1)/ln(e1/e2) and its 1-sigma (independent relative errors
    r1, r2). None when the dose does not fall (ln <= 0)."""
    lr = math.log(e1 / e2)
    if lr <= 0:
        return None, None
    lam = (x2 - x1) / lr
    return lam, lam * math.hypot(r1, r2) / lr


def pooled_series(path: Path) -> list:
    """One pooled entry per depth, sorted by overhead: last row per
    (overhead, seed_offset), inverse-variance pooled over samples."""
    last = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            last[(float(r["overhead_gcm2"]), r["seed_offset"])] = r
    out = []
    for g in sorted({k[0] for k in last}):
        rows = [r for (gg, _), r in sorted(last.items()) if gg == g]
        p = wsf.pool(rows)
        p.update(overhead_gcm2=g, n=len(rows), side_gcm2=float(rows[-1]["side_gcm2"]),
                 footprint=rows[-1]["footprint"],
                 stamps=sorted({r["provenance"] for r in rows}))
        out.append(p)
    return out


def report(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"no results yet: {path}")
    s = pooled_series(path)
    print("\nburied-habitat depth series (flood, 900 cm sky source); E = annual dose "
          "equivalent, errors = dose-equivalent batch SEM, n = pooled seed samples")
    print(f"{'g/cm2':>7} {'side':>6} {'n':>2}  {'phantom E':>17} {'skin E':>17} "
          f"{'ph D mGy/d':>16} {'E/E0':>6} {'lambda_ph':>14} {'lambda_sk':>14}")
    for i, p in enumerate(s):
        e0 = s[0]["phantom_msv"]
        lp = ls = ""
        if i:
            q = s[i - 1]
            a, sa = atten_length(q["overhead_gcm2"], q["phantom_msv"], q["e_ph"],
                                 p["overhead_gcm2"], p["phantom_msv"], p["e_ph"])
            b, sb = atten_length(q["overhead_gcm2"], q["skin_msv"], q["e_sk"],
                                 p["overhead_gcm2"], p["skin_msv"], p["e_sk"])
            lp = "rising" if a is None else f"{a:6.0f}+/-{sa:<5.0f}"
            ls = "rising" if b is None else f"{b:6.0f}+/-{sb:<5.0f}"
        flag = "" if p["footprint"] == "ok" else f"  [{p['footprint']}: lower bound]"
        if not p["eq_ok"]:
            flag += "  [absorbed-err only]"
        print(f"{p['overhead_gcm2']:7.1f} {p['side_gcm2']:6.0f} {p['n']:2d}  "
              f"{p['phantom_msv']:8.1f} +/-{100 * p['e_ph']:4.1f}% "
              f"{p['skin_msv']:8.1f} +/-{100 * p['e_sk']:4.1f}% "
              f"{p['phantom_mgy_d']:9.4f} +/-{100 * p['phantom_rel']:3.1f}% "
              f"{p['phantom_msv'] / e0:6.3f} {lp:>14} {ls:>14}{flag}")
    # The stamp ends in a run timestamp; only the model fields must agree.
    stamps = sorted({re.sub(r"\s*date=\S+", "", x) for p in s for x in p["stamps"]})
    if len(stamps) > 1:
        print(f"\nWARNING: {len(stamps)} different provenance stamps in the series; "
              "points may not share one model:\n  " + "\n  ".join(stamps))
    print("\n" + ORIENTATION)
    return s


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--depths", nargs="*", type=float, default=list(DEPTHS_GCM2),
                   help="total overhead g/cm2 (Al ceiling + regolith)")
    r.add_argument("--rounds", type=int, default=12)
    r.add_argument("--tier", choices=("quick", "full"), default="full")
    r.add_argument("--seed-offset", type=int, default=0)
    r.add_argument("--allow-marginal", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    q = sub.add_parser("report")
    q.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    a = p.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
    else:
        report(a.csv)


if __name__ == "__main__":
    main()
