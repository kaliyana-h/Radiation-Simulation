#!/usr/bin/env python3
"""Continuous vs energy-stratified flood: paired PC comparison.

Question: does stratifying the primary energy (jobs.run_composition(
sampling="stratified")) cut the phantom's statistical error at the same cost,
without moving the answer?

Each design is run twice, once per sampling mode, for the SAME number of rounds
(target_rel_err=0, min=max=rounds, so neither stops early). Because the two modes
spend different wall time, efficiency is compared by the figure of merit
    FOM = 1 / (rel_err^2 * wall_seconds)
which is the matched-walltime comparison: rel_err at the continuous run's wall
is rel_err_strat * sqrt(wall_strat / wall_cont).

Designs: 3 m dome, Al at 19 / 30 / 54 g/cm2, plus a buried habitat at ~150 g/cm2
(regolith overburden on a thin Al ceiling).

Pass, per design:
  agree   : |z| < 2 for the phantom AND skin annual E, z = diff / combined sigma
  better  : phantom FOM(strat) / FOM(cont) >= --min-gain (default 2, i.e. the
            matched-wall phantom rel_err drops by >= ~30%)

TARGET-BLIND: no dose value or limit enters the physics or the sampling.
OUTER_GAUGE_ANCHOR_CAL is untouched. The production default stays continuous;
flipping it is the user's decision.

Usage
-----
  python3 paper/stratified_flood_compare.py run --dry-run          # laptop: plan only
  # PC (TOPAS), in tmux:
  .venv/bin/python paper/stratified_flood_compare.py run \\
      | tee runs/strat_compare.log
  .venv/bin/python paper/stratified_flood_compare.py report
Rows are appended to the CSV after every (design, mode), so an interrupted run
keeps what finished; rerun with --designs to fill in the rest.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CSV_DEFAULT = _REPO / "paper" / "stratified_flood_compare.csv"
FIELDS = ["design", "areal_gcm2", "sampling", "strata", "tier", "rounds",
          "wall_s", "total_primaries",
          "phantom_msv", "phantom_rel", "skin_msv", "skin_rel",
          "phantom_mgy_d", "skin_mgy_d", "band_histories", "provenance", "utc"]


# ----------------------------------------------------------------------------
# designs
# ----------------------------------------------------------------------------
def designs() -> dict:
    """name -> HabitatSpec. Thicknesses use the library densities so the labelled
    areal density is exact."""
    from lunarsim import HabitatSpec, WallLayer
    from lunarsim.spec import MATERIALS
    rho_al = MATERIALS["aluminium"]["density"]
    rho_reg = MATERIALS["regolith"]["density"]
    out = {}
    for g in (19.0, 30.0, 54.0):
        out[f"al{g:g}"] = HabitatSpec(name=f"strat_al{g:g}", shape="dome",
                                      inner_radius_cm=300.0,
                                      walls=[WallLayer("aluminium", g / rho_al)])
    ceiling = 1.0                                    # cm Al = 2.7 g/cm2
    depth = (150.0 - rho_al * ceiling) / rho_reg
    out["buried150"] = HabitatSpec(name="strat_buried150", shape="buried",
                                   inner_radius_cm=300.0,
                                   walls=[WallLayer("aluminium", ceiling)],
                                   burial_depth_cm=depth)
    return out


# ----------------------------------------------------------------------------
# PC run
# ----------------------------------------------------------------------------
def run_one(spec, tier, sampling: str, strata: int, rounds: int) -> dict:
    from lunarsim import dosimetry, jobs

    def progress(done, total, rel):
        r = f"{rel:.3f}" if rel is not None else "-"
        print(f"    [{sampling}] {done}/{total} batches  rel {r}", flush=True)

    t0 = time.time()
    comp = jobs.run_composition(spec, tier, target_rel_err=0.0, min_batches=rounds,
                                max_batches=rounds, converge_on="both",
                                sampling=sampling, strata=strata, progress_cb=progress)
    if not comp.ok:
        raise RuntimeError(f"{spec.name} {sampling}: run failed rc={comp.returncode}\n"
                           f"{comp.log_tail[-2000:]}")
    a_ph = dosimetry.assess_composition(comp.species_results, phi_MV=tier.phi_mv,
                                        skin=False, model=comp.model)
    a_sk = dosimetry.assess_composition(comp.species_results, phi_MV=tier.phi_mv,
                                        skin=True, model=comp.model)
    return {
        "design": "", "areal_gcm2": round(spec.areal_density_gcm2(), 3),
        "sampling": sampling, "strata": comp.strata, "tier": tier.name,
        "rounds": min(r.n_batches for _, r in comp.species_results),
        "wall_s": round(comp.wall_seconds or (time.time() - t0), 1),
        "total_primaries": comp.total_primaries,
        "phantom_msv": a_ph.annual_msv, "phantom_rel": a_ph.rel_err,
        "skin_msv": a_sk.annual_msv, "skin_rel": a_sk.rel_err,
        "phantom_mgy_d": a_ph.dose_rate_gy_s * 86400e3,
        "skin_mgy_d": a_sk.dose_rate_gy_s * 86400e3,
        "band_histories": json.dumps(comp.band_histories or {}, separators=(",", ":")),
        "provenance": dosimetry.provenance_str(a_sk.provenance),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_run(args) -> None:
    from lunarsim.bridge import QUICK_LOOK, FULL_RUN
    tier = {"quick": QUICK_LOOK, "full": FULL_RUN}[args.tier]
    ds = designs()
    names = args.designs or list(ds)
    for n in names:
        if n not in ds:
            raise SystemExit(f"unknown design {n!r}; choose from {list(ds)}")
    modes = ["continuous", "stratified"]
    if args.stratified_first:
        modes.reverse()
    print(f"tier {tier.name} ({tier.total_primaries} primaries/run/species), "
          f"{args.rounds} rounds, K={args.strata}, csv {args.csv}")
    for n in names:
        print(f"  {n:10s} {ds[n].shape:6s} {ds[n].areal_density_gcm2():7.2f} g/cm2")
    if args.dry_run:
        return
    for n in names:
        for mode in modes:
            print(f"== {n} {mode}", flush=True)
            row = run_one(ds[n], tier, mode, args.strata if mode == "stratified" else 1,
                          args.rounds)
            row["design"] = n
            append_row(args.csv, row)
            print(f"   phantom {row['phantom_msv']:.1f} mSv/yr +/- "
                  f"{100 * row['phantom_rel']:.1f}%  skin {row['skin_msv']:.1f} +/- "
                  f"{100 * row['skin_rel']:.1f}%  wall {row['wall_s']:.0f} s", flush=True)
    report(args.csv, args.min_gain)


# ----------------------------------------------------------------------------
# report (laptop-safe)
# ----------------------------------------------------------------------------
def _z(m1, r1, m2, r2):
    s = math.hypot(m1 * r1, m2 * r2)
    return (m2 - m1) / s if s > 0 else float("nan")


def _fom(rel, wall):
    return 1.0 / (rel * rel * wall) if rel and wall else float("nan")


def compare(cont: dict, strat: dict, min_gain: float) -> dict:
    f = {k: float(cont[k]) for k in ("phantom_msv", "phantom_rel", "skin_msv",
                                     "skin_rel", "wall_s")}
    g = {k: float(strat[k]) for k in f}
    z_ph = _z(f["phantom_msv"], f["phantom_rel"], g["phantom_msv"], g["phantom_rel"])
    z_sk = _z(f["skin_msv"], f["skin_rel"], g["skin_msv"], g["skin_rel"])
    gain_ph = _fom(g["phantom_rel"], g["wall_s"]) / _fom(f["phantom_rel"], f["wall_s"])
    gain_sk = _fom(g["skin_rel"], g["wall_s"]) / _fom(f["skin_rel"], f["wall_s"])
    matched = g["phantom_rel"] * math.sqrt(g["wall_s"] / f["wall_s"])
    agree = abs(z_ph) < 2.0 and abs(z_sk) < 2.0
    better = gain_ph >= min_gain
    return {"z_phantom": z_ph, "z_skin": z_sk, "fom_gain_phantom": gain_ph,
            "fom_gain_skin": gain_sk, "phantom_rel_matched": matched,
            "agree": agree, "better": better, "pass": agree and better}


def report(path: Path, min_gain: float = 2.0) -> list:
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    rows = list(csv.DictReader(path.open()))
    latest = {}
    for r in rows:                                  # last row per (design, mode) wins
        latest[(r["design"], r["sampling"])] = r
    out = []
    print(f"\n{'design':10s} {'g/cm2':>6s} | {'E_ph cont':>16s} {'E_ph strat':>16s} "
          f"{'z':>5s} | {'E_sk z':>6s} | {'rel ph c/s(@wall)':>20s} | "
          f"{'FOM x ph':>8s} {'sk':>5s} | verdict")
    for d in dict.fromkeys(r["design"] for r in rows):
        c, s = latest.get((d, "continuous")), latest.get((d, "stratified"))
        if not (c and s):
            print(f"{d:10s} incomplete (have {[m for (dd, m) in latest if dd == d]})")
            continue
        v = compare(c, s, min_gain)
        v.update(design=d, areal_gcm2=float(c["areal_gcm2"]))
        out.append(v)
        verdict = "PASS" if v["pass"] else (
            "FAIL-disagree" if not v["agree"] else "no gain")
        print(f"{d:10s} {float(c['areal_gcm2']):6.1f} | "
              f"{float(c['phantom_msv']):7.1f}+/-{100 * float(c['phantom_rel']):4.1f}% "
              f"{float(s['phantom_msv']):7.1f}+/-{100 * float(s['phantom_rel']):4.1f}% "
              f"{v['z_phantom']:+5.1f} | {v['z_skin']:+6.1f} | "
              f"{100 * float(c['phantom_rel']):5.1f}/{100 * v['phantom_rel_matched']:5.1f}% "
              f"{'':7s}| {v['fom_gain_phantom']:8.2f} {v['fom_gain_skin']:5.2f} | {verdict}")
        if c["provenance"].split(" engine")[0] != s["provenance"].split(" engine")[0]:
            print(f"{'':10s} WARNING: provenance differs between modes")
    if out:
        n = sum(v["pass"] for v in out)
        print(f"\n{n}/{len(out)} designs PASS (|z|<2 on phantom+skin E, phantom FOM "
              f"gain >= {min_gain:g}). Default flip remains the user's decision.")
    return out


def cmd_report(args) -> None:
    report(args.csv, args.min_gain)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="PC: paired flood runs (TOPAS)")
    r.add_argument("--designs", nargs="*", help="subset of al19 al30 al54 buried150")
    r.add_argument("--rounds", type=int, default=12)
    r.add_argument("--strata", type=int, default=4)
    r.add_argument("--tier", choices=("quick", "full"), default="full")
    r.add_argument("--stratified-first", action="store_true",
                   help="run the stratified mode first (checks for order/thermal drift)")
    r.add_argument("--dry-run", action="store_true", help="print the plan only")
    for q in (r, sub.add_parser("report", help="pair rows and print verdicts")):
        q.add_argument("--csv", type=Path, default=CSV_DEFAULT)
        q.add_argument("--min-gain", type=float, default=2.0)
    args = p.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
