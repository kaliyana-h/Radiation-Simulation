#!/usr/bin/env python3
"""GCR ring weighting A/B: equal histories per ring vs solid-angle shares.

Question: the GCR sky is rings x azimuth parallel discs, each disc perpendicular
to its own beam. Fluence from a direction is then N/area, so an ISOTROPIC sky
needs N_ring ∝ ΔΩ_ring (Dobynde & Guo 2021, Eq. 1-2). The legacy scheme (default
until 2026-11-15, when "solidangle" was adopted) gives
every ring EQUAL histories (sin²-band "Lambert" argument), i.e. a cosθ-weighted
sky: steep rays ~2x over-, grazing rays ~2.5x under-represented. Total primaries
(hence the outer-gauge normalisation and CAL 0.94932) are identical in both arms.

Laptop ray-trace prediction (normal-incidence D(x)=exp(-x/λ)):
  al19 / al54 dome : solidangle/equal ≈ 1.00 (wall is equally thick every way)
  buried150        : solidangle/equal ≈ 0.91 (λ400) / 0.83 (λ200) / 0.71 (λ100)
                     -- grazing rays cross the 2 m side berm (~370-390 g/cm2)
The MC adds what the ray-trace cannot: secondaries, albedo, the 17° discs that
start partly below grade.

Arms (both plain TOPAS Beam discs, continuous sampling, full composition, same
tier/rounds; only LUNARSIM_GCR_RING_WEIGHTS differs -- make_source reads it):
  equal       legacy pre-2026-11-15 default (byte-identical to it except comments)
  solidangle  ring share = ΔΩ of its sin²-band, exact total by largest remainder

Pass/flag per design (pooled over seed samples, dose-equivalent batch errors):
  ratio R = E_solidangle / E_equal for phantom and skin, with z = (R-1)/σ_R.
  Dome control expects |z| < 2. buried150 expects R well below 1.

Seeds: round k uses seed k+1+seed_offset; both arms of one sample use the same
offset but sample differently, so they are treated as independent.

TARGET-BLIND (geometry/direction only); CAL untouched. The user adopted
"solidangle" as the make_source default on 2026-11-15; "equal" is the legacy arm.

Usage
-----
  python3 paper/ring_weighting_ab.py run --dry-run
  # PC, in tmux, one line:
  .venv/bin/python paper/ring_weighting_ab.py run --designs buried150 --seed-offset 15000 | tee runs/ringw_buried150_s15000.log
  .venv/bin/python paper/ring_weighting_ab.py report
Rows are appended after every (design, arm). Last row per (design, sample, arm) wins.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "paper"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import weighted_source_fom as wsf          # noqa: E402  (designs, run_one, pool)

CSV_DEFAULT = _REPO / "paper" / "ring_weighting_ab.csv"
ENV = "LUNARSIM_GCR_RING_WEIGHTS"
ARMS = ("equal", "solidangle")
FIELDS = ["design", "areal_gcm2", "arm", "tier", "rounds", "wall_s", "total_primaries",
          "phantom_msv", "phantom_rel", "phantom_eq_rel", "skin_msv", "skin_rel",
          "skin_eq_rel", "phantom_mgy_d", "skin_mgy_d", "fluence_out", "fluence_in",
          "seed_offset", "provenance", "utc", "phantom_species"]
PREDICT = {"al19": "≈1.00", "al54": "≈1.00", "buried150": "≈0.83 (0.71-0.91)"}


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_run(args) -> None:
    from lunarsim.bridge import QUICK_LOOK, FULL_RUN
    tier = {"quick": QUICK_LOOK, "full": FULL_RUN}[args.tier]
    ds = wsf.designs()
    names = args.designs or ["buried150"]
    for n in names:
        if n not in ds:
            raise SystemExit(f"unknown design {n!r}; choose from {list(ds)}")
    print(f"tier {tier.name} ({tier.rings}x{tier.azimuth}x{tier.histories}), rounds "
          f"{args.rounds}, seed offset {args.seed_offset}, arms {args.arms}, csv {args.csv}")
    if args.dry_run:
        ms = __import__("make_source")
        for a in args.arms:
            per_ring = [sum(h) for h in ms.ring_histories(tier.rings, tier.azimuth,
                                                          tier.histories, a)]
            print(f"  {a:10s} histories/ring {per_ring}  total {sum(per_ring)}")
        for n in names:
            print(f"  {n:10s} {ds[n].shape:6s} {ds[n].areal_density_gcm2():7.2f} g/cm2  "
                  f"predicted solidangle/equal {PREDICT.get(n, '?')}")
        return
    saved = os.environ.get(ENV)
    try:
        for n in names:
            for a in args.arms:
                os.environ[ENV] = a          # bridge passes os.environ to make_source
                print(f"== {n} {a}", flush=True)
                row = wsf.run_one(ds[n], tier, "beam", args.rounds, 0.0, args.seed_offset)
                row.update(design=n, arm=a)
                append_row(args.csv, row)
                print(f"   phantom {row['phantom_msv']:.1f} mSv/yr +/- "
                      f"{100 * float(row['phantom_eq_rel'] or row['phantom_rel']):.1f}%  "
                      f"skin {row['skin_msv']:.1f} +/- "
                      f"{100 * float(row['skin_eq_rel'] or row['skin_rel']):.1f}%  "
                      f"wall {row['wall_s']:.0f} s", flush=True)
    finally:
        if saved is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = saved
    report(args.csv)


def _ratio(a: dict, b: dict, key: str, ekey: str):
    """R = b/a and its 1-sigma, independent samples."""
    r = b[key] / a[key]
    s = r * math.hypot(a[ekey], b[ekey])
    return r, s, (r - 1.0) / s if s > 0 else float("nan")


def report(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"no results yet: {path}")
    last = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            last[(r["design"], r["seed_offset"], r["arm"])] = r
    for d in sorted({k[0] for k in last}):
        print(f"\n== {d}   predicted solidangle/equal {PREDICT.get(d, '?')}")
        pooled = {}
        for a in ARMS:
            rows = [r for (dd, _, aa), r in sorted(last.items()) if dd == d and aa == a]
            for r in rows:
                n = wsf._norm(r)
                print(f"  {a:10s} s{r['seed_offset']:>6s}  phantom {n['phantom_msv']:7.1f} "
                      f"+/- {100 * n['e_ph']:4.1f}%   skin {n['skin_msv']:7.1f} "
                      f"+/- {100 * n['e_sk']:4.1f}%   wall {n['wall_s']:.0f} s")
            if rows:
                pooled[a] = wsf.pool(rows)
                pooled[a]["n"] = len(rows)
        if len(pooled) < 2:
            print("  (need both arms for a ratio)")
            continue
        eq, sa = pooled["equal"], pooled["solidangle"]
        print(f"  pooled n={eq['n']}/{sa['n']}")
        for lab, key, ek in (("phantom E", "phantom_msv", "e_ph"), ("skin E", "skin_msv", "e_sk"),
                             ("phantom D", "phantom_mgy_d", "phantom_rel"),
                             ("skin D", "skin_mgy_d", "skin_rel")):
            r, s, z = _ratio(eq, sa, key, ek)
            print(f"  {lab:10s} solidangle/equal = {r:.3f} +/- {s:.3f}   z(R-1) {z:+.2f}")
        print(f"  outer fluence gauge ratio {sa['fluence_out'] / eq['fluence_out']:.3f} "
              f"(expect 1: same total primaries)" if eq["fluence_out"] else "")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--designs", nargs="*", help="subset of al19 al54 buried150 (default buried150)")
    r.add_argument("--arms", nargs="*", choices=ARMS, default=list(ARMS))
    r.add_argument("--rounds", type=int, default=12)
    r.add_argument("--tier", choices=("quick", "full"), default="full")
    r.add_argument("--seed-offset", type=int, default=0)
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
