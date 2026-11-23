#!/usr/bin/env python3
"""Beam-spot sweep for the regolith dose-vs-depth slab (PC).

Why this exists
---------------
regolith_v2 (2026-11-23) matches Blunier 2025 at the surface (322.7 vs 323 mSv/yr)
but falls far too fast with depth: 73.7 at 55.6 g/cm^2 (Blunier ~213 at 45) and
25.0 at 194 g/cm^2. The leading suspect is the kernel's NARROW 22 cm beam spot.
Behind 30 cm to 3 m of regolith the cascade spreads laterally much wider than
22 cm, so particles scattered out of the phantom are not replaced by in-scatter
from an unlit plane. A fully lit (broad-field) result is spot-independent,
because phi_ff = N/(pi*spot^2) follows the spot.

The test: the same sweep at 0.3 m and 1.0 m with the spot at 22/45/90/180 cm.
  * Dose RISES with the spot and levels off: truncation confirmed. The plateau
    value is the broad-field answer, and the size of the rise says how much of the
    deep deficit it explains.
  * Dose FLAT with the spot (within the error bars): truncation ruled out, so look
    at the slab slant and the missing floor next.
180 cm is about the largest spot the grazing ring allows at 1.0 m (the disc must
stay above the slab top); generate refuses a spot that does not fit.

Histories scale as (spot/22)^EXP. EXP 2 keeps the phantom hits, and so the error
bar, equal at every spot, but the 180 cm sweep then costs 67x the 22 cm one.
The default EXP 1 costs ~15x one 22 cm sweep in total; the 180 cm error bar is then
~2.9x the 22 cm one, which is plenty to see a 2-3x effect.

TARGET-BLIND: no CAL, no physics change. The tool's numbers are only compared
across spots here; the literature values above are printed for orientation only.

Usage (PC)
----------
    .venv/bin/python paper/regolith_spot_probe.py generate runs/regolith_spot
    bash runs/regolith_spot/run_all.sh | tee runs/regolith_spot.log      (in tmux)
    .venv/bin/python paper/regolith_spot_probe.py compare runs/regolith_spot
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lunarsim import regolith_sweep as rs  # noqa: E402

DEFAULT_SPOTS = [22.0, 45.0, 90.0, 180.0]
DEFAULT_DEPTHS = [0.3, 1.0]
REF_SPOT = rs.BEAM_SPOT_CM


def spot_dir(root: Path, spot: float) -> Path:
    return root / f"spot{spot:g}"


def histories_for(spot: float, base: int, exponent: float) -> int:
    return max(1, int(round(base * (spot / REF_SPOT) ** exponent)))


def check_spots(spots, depths) -> list[str]:
    return [f"spot {s:g} cm at {d:g} m: {p}" for s in spots for d in depths
            for p in rs.illumination_problems(d, s)]


def generate(root: Path, spots, depths, base: int, exponent: float, seeds: int,
             threads: int, phi: float) -> None:
    bad = check_spots(spots, depths)
    if bad:
        raise SystemExit("spots that do not fit:\n  " + "\n  ".join(bad))
    root.mkdir(parents=True, exist_ok=True)
    plan = []
    for s in spots:
        h = histories_for(s, base, exponent)
        print(f"\n== spot {s:g} cm: {h} histories/source ==")
        rs.generate(spot_dir(root, s), histories=h, seeds=seeds, threads=threads,
                    phi_mv=phi, depths_m=depths, beam_spot_cm=s)
        plan.append({"spot_cm": s, "histories": h, "dir": spot_dir(root, s).name})
    (root / "probe.json").write_text(json.dumps(
        {"spots": plan, "depths_m": list(depths), "seeds": seeds,
         "hist_exponent": exponent, "base_histories": base, "phi_MV": phi}, indent=2))
    lines = ["#!/usr/bin/env bash",
             "# Beam-spot probe: runs each spot's resumable sweep in turn, smallest first.",
             ': "${TOPAS_G4_DATA_DIR:?set it first, e.g. export TOPAS_G4_DATA_DIR=~/G4Data}"',
             'cd "$(dirname "$0")"', "rc=0"]
    lines += [f'bash {p["dir"]}/run.sh || rc=1' for p in plan]
    lines += ['exit "$rc"']
    (root / "run_all.sh").write_text("\n".join(lines) + "\n")
    cost = sum(p["histories"] for p in plan) / base
    print(f"\nwrote {root/'run_all.sh'}: {len(spots)} spots x {len(depths)} depths x "
          f"{seeds} seed(s); total histories = {cost:.1f}x one {REF_SPOT:g} cm sweep "
          "at these depths")


def compare_rows(by_spot: dict) -> list[dict]:
    """by_spot: {spot: [collect rows]} -> one row per depth x spot, with the ratio
    to the reference (smallest) spot and its combined error."""
    spots = sorted(by_spot)
    ref = spots[0]
    out = []
    for i, r_ref in enumerate(by_spot[ref]):
        for s in spots:
            r = by_spot[s][i]
            if r["depth_m"] != r_ref["depth_m"]:
                raise ValueError(f"depth mismatch spot {s:g}: {r['depth_m']} vs "
                                 f"{r_ref['depth_m']}")
            row = {"depth_m": r["depth_m"], "areal_gcm2": r["areal_gcm2"],
                   "spot_cm": s, "usable": r["usable"]}
            for o in rs._REPORT_ORGANS:
                v, v0 = r[f"{o}_doseeq_mSv_yr"], r_ref[f"{o}_doseeq_mSv_yr"]
                e, e0 = r.get(f"{o}_doseeq_relerr", ""), r_ref.get(f"{o}_doseeq_relerr", "")
                row[f"{o}_mSv_yr"] = v
                row[f"{o}_relerr"] = e
                row[f"{o}_over_ref"] = round(v / v0, 4) if v0 else ""
                row[f"{o}_ratio_relerr"] = (round(math.hypot(e, e0), 4)
                                            if e != "" and e0 != "" and s != ref else "")
            out.append(row)
    return out


def _fmt(v, f):
    return "--" if v in ("", None) else format(v, f)


def compare(root: Path, legacy: bool = False) -> int:
    probe = json.loads((root / "probe.json").read_text())
    by_spot = {}
    for p in probe["spots"]:
        d = root / p["dir"]
        print(f"collecting {d} ...", flush=True)
        by_spot[p["spot_cm"]] = rs.collect(d, legacy=legacy)["rows"]
    rows = compare_rows(by_spot)
    print(f"\n{'depth':>6} {'g/cm2':>6} {'spot':>5} {'skin':>8} {'+-':>6} {'/ref':>6} "
          f"{'core':>8} {'+-':>6} {'/ref':>6}")
    for r in rows:
        print(f"{r['depth_m']:6g} {r['areal_gcm2']:6.1f} {r['spot_cm']:5g} "
              f"{r['skin_mSv_yr']:8.1f} {_fmt(r['skin_relerr'], '6.1%')} "
              f"{_fmt(r['skin_over_ref'], '6.2f')} {r['core_mSv_yr']:8.1f} "
              f"{_fmt(r['core_relerr'], '6.1%')} {_fmt(r['core_over_ref'], '6.2f')}"
              + ("" if r["usable"] else "  (INCOMPLETE)"))
    print("\norientation only (not a gate): Blunier 2025 ICRU sphere ~213 mSv/yr at 45 "
          "g/cm^2; regolith_v2 22 cm gave 73.7 skin at 55.6 and 25.0 at 193.7.")
    out = root / "spot_compare.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("root", type=Path)
    g.add_argument("--spots", type=float, nargs="+", default=DEFAULT_SPOTS)
    g.add_argument("--depths", type=float, nargs="+", default=DEFAULT_DEPTHS)
    g.add_argument("--histories", type=int, default=2000,
                   help=f"histories/source at the {REF_SPOT:g} cm spot (default 2000)")
    g.add_argument("--hist-exponent", type=float, default=1.0,
                   help="histories scale as (spot/22)^EXP; 2 = equal error bars")
    g.add_argument("--seeds", type=int, default=2,
                   help="seeds per run (>=2 gives error bars; default 2)")
    g.add_argument("--threads", type=int, default=0)
    g.add_argument("--phi", type=float, default=rs.PHI_MV)
    c = sub.add_parser("compare")
    c.add_argument("root", type=Path)
    c.add_argument("--legacy", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        generate(a.root, a.spots, a.depths, a.histories, a.hist_exponent, a.seeds,
                 a.threads, a.phi)
        return 0
    return compare(a.root, a.legacy)


if __name__ == "__main__":
    sys.exit(main())
