#!/usr/bin/env python3
"""Per-species NASA/ICRP breakdown of the flood dose equivalent vs Al depth (PC).

Why this exists
---------------
The 2026-11-19/20 flood sweep (paper/accuracy_flood_sweep.py) found the NASA/ICRP
skin ratio rising smoothly with Al depth, 1.10 (5) -> 1.35 (54 g/cm^2), while ICRP
skin is flat at ~205 from 25 up. That sweep ran its batches in temp dirs
(run_design keep=False), so no per-species numbers survive, and this script re-runs
the same flood MC. It converges on the SKIN only: skin reaches 0.5% in ~2 rounds,
and the phantom is not needed to answer the question.

For each depth and each species it reports:
  * H_ICRP and H_NASA on the skin lining, in mSv/yr, both read from the SAME
    transport, so only Q differs;
  * the species' own Q_ICRP, Q_NASA and NASA/ICRP ratio;
  * its share of the total NASA-minus-ICRP excess;
  * its skin neutron-lineage dose-equivalent fraction f_n (ICRP basis, from the
    DoseEquivalent_ICRP_Neutron twin).

How to read it
--------------
  * Excess carried by H/He AND growing with depth, alongside a rising f_n: the
    wall-bred neutron/recoil component is the driver (the leading hypothesis).
  * Excess carried by C/Si/Fe: the heavy primaries/fragments are the driver.
  * Per-species NASA/ICRP flat with depth while the species MIX shifts: a
    composition effect, not a per-species Q effect.
The phantom columns are printed too, but at 2 skin-converged rounds they are noise.

TARGET-BLIND: this only reads what the tool returns. No CAL or physics change.
Seeds replay the sweep's (1..N), so with the same round count the skin matches
paper/accuracy_al_flood.csv.

Usage (PC)
----------
    export TOPAS_G4_DATA_DIR=~/G4Data
    cd ~/topas
    .venv/bin/python paper/nasa_q_breakdown.py --only 5 19 54 | tee runs/nasaq.log
Writes paper/nasa_q_breakdown.csv (one row per depth x species, plus a TOTAL row).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_DEPTHS = [5.0, 10.0, 19.0, 30.0, 54.0]
CSV_PATH = HERE / "nasa_q_breakdown.csv"
CSV_FIELDS = ["areal_gcm2", "target", "species", "dose_share",
              "h_icrp_mSv_yr", "h_nasa_mSv_yr", "q_icrp", "q_nasa", "nasa_over_icrp",
              "excess_share", "f_neutron_skin", "rel_err_abs", "provenance", "date"]


def breakdown(a_icrp, a_nasa) -> list[dict]:
    """Per-species rows from two assess_composition results on the SAME species
    set and target, differing only in qf. Each species' H is its share of the
    summed dose-equivalent rate times the total annual value, so no unit
    constants are repeated here. The last row is TOTAL."""
    ci = {c["species"]: c for c in a_icrp.contributions}
    cn = {c["species"]: c for c in a_nasa.contributions}
    if set(ci) != set(cn):
        raise ValueError(f"species sets differ: {sorted(ci)} vs {sorted(cn)}")
    eq_i = sum(c["doseeq_rate_sv_s"] or 0.0 for c in ci.values())
    eq_n = sum(c["doseeq_rate_sv_s"] or 0.0 for c in cn.values())
    if eq_i <= 0 or eq_n <= 0:
        raise ValueError("no dose-equivalent scorer data (doseeq_rate_sv_s missing)")
    tot_i, tot_n = a_icrp.annual_msv, a_nasa.annual_msv
    excess = tot_n - tot_i
    rows = []
    for name in ci:                       # composition order
        hi = tot_i * (ci[name]["doseeq_rate_sv_s"] or 0.0) / eq_i
        hn = tot_n * (cn[name]["doseeq_rate_sv_s"] or 0.0) / eq_n
        rows.append({
            "species": name,
            "dose_share": ci[name]["dose_fraction"],
            "h_icrp": hi, "h_nasa": hn,
            "q_icrp": ci[name]["quality_factor"], "q_nasa": cn[name]["quality_factor"],
            "ratio": hn / hi if hi > 0 else None,
            "excess_share": (hn - hi) / excess if excess else None,
            "rel_err_abs": ci[name]["rel_err"],
        })
    rows.append({"species": "TOTAL", "dose_share": 1.0, "h_icrp": tot_i,
                 "h_nasa": tot_n, "q_icrp": a_icrp.quality_factor,
                 "q_nasa": a_nasa.quality_factor,
                 "ratio": tot_n / tot_i if tot_i > 0 else None,
                 "excess_share": 1.0 if excess else None,
                 "rel_err_abs": a_icrp.rel_err})
    return rows


def _f(x, fmt):
    return "--" if x is None else format(x, fmt)


def print_table(gcm2: float, target: str, rows: list[dict], fn: dict) -> None:
    print(f"\n  {gcm2:g} g/cm^2  [{target}]")
    print(f"    {'species':8} {'D share':>8} {'H_ICRP':>8} {'H_NASA':>8} {'Q_I':>6} "
          f"{'Q_N':>6} {'N/I':>6} {'excess':>7} {'f_n':>6} {'D err':>6}")
    for r in rows:
        print(f"    {r['species']:8} {_f(r['dose_share'], '8.3f')} {r['h_icrp']:8.1f} "
              f"{r['h_nasa']:8.1f} {_f(r['q_icrp'], '6.2f')} {_f(r['q_nasa'], '6.2f')} "
              f"{_f(r['ratio'], '6.3f')} {_f(r['excess_share'], '7.2f')} "
              f"{_f(fn.get(r['species']), '6.3f')} {_f(r['rel_err_abs'], '6.3f')}",
              flush=True)


def neutron_fractions(species_results, a_skin_icrp) -> dict:
    """Skin neutron-lineage H fraction per species, plus the dose-eq-weighted TOTAL."""
    out = {sp[0]: getattr(res, "neutron_doseeq_fraction", None)
           for sp, res in species_results}
    out["TOTAL"] = a_skin_icrp.neutron_fraction
    return out


def csv_rows(gcm2, target, rows, fn, prov, date) -> list[dict]:
    return [{"areal_gcm2": f"{gcm2:g}", "target": target, "species": r["species"],
             "dose_share": _f(r["dose_share"], ".4f"),
             "h_icrp_mSv_yr": f"{r['h_icrp']:.2f}", "h_nasa_mSv_yr": f"{r['h_nasa']:.2f}",
             "q_icrp": _f(r["q_icrp"], ".3f"), "q_nasa": _f(r["q_nasa"], ".3f"),
             "nasa_over_icrp": _f(r["ratio"], ".4f"),
             "excess_share": _f(r["excess_share"], ".3f"),
             "f_neutron_skin": _f(fn.get(r["species"]), ".4f") if target == "skin" else "",
             "rel_err_abs": _f(r["rel_err_abs"], ".4f"),
             "provenance": prov, "date": date} for r in rows]


def _run_depth(gcm2: float, args) -> list[dict]:
    from lunarsim import bridge, dosimetry, jobs
    from lunarsim.spec import HabitatSpec, WallLayer
    import accuracy_flood_sweep as afs

    t_cm = gcm2 / afs.AL_RHO_G_CM3
    spec = HabitatSpec(name=f"nasaq_al{gcm2:g}", shape="dome",
                       inner_radius_cm=afs.DOME_INNER_R_CM,
                       walls=[WallLayer("aluminium", t_cm)])
    tier = bridge.FULL_RUN
    print(f"[{gcm2:g} g/cm^2 = {t_cm:.2f} cm Al] full-comp flood, skin-converged "
          f"to {args.rel_err:.3f}, <= {args.max_batches} rounds ...", flush=True)
    comp = jobs.run_composition(spec, tier=tier, converge_on="skin",
                                target_rel_err=args.rel_err,
                                min_batches=2, max_batches=args.max_batches)
    if comp.returncode != 0 or not comp.species_results:
        raise SystemExit(f"[{gcm2:g}] composition MC failed (rc={comp.returncode})")
    sr = comp.species_results
    prov = dosimetry.provenance_str(comp.provenance)
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    out = []
    fn = None
    for target, is_skin in (("skin", True), ("phantom", False)):
        a_i = dosimetry.assess_composition(sr, phi_MV=tier.phi_mv, skin=is_skin,
                                           qf="icrp", model=comp.model)
        a_n = dosimetry.assess_composition(sr, phi_MV=tier.phi_mv, skin=is_skin,
                                           qf="nasa", model=comp.model)
        if a_i is None or a_n is None:
            print(f"    [{target}] no usable dose -- skipped", flush=True)
            continue
        if fn is None:
            fn = neutron_fractions(sr, a_i)
        rows = breakdown(a_i, a_n)
        print_table(gcm2, target, rows, fn if is_skin else {})
        out += csv_rows(gcm2, target, rows, fn, prov, date)
    print(f"    wall {comp.wall_seconds/60:.1f} min, {comp.n_batches} batches", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=float, nargs="+", metavar="GCM2",
                    default=DEFAULT_DEPTHS)
    ap.add_argument("--rel-err", type=float, default=0.02,
                    help="skin combined rel_err target (default 0.02)")
    ap.add_argument("--max-batches", type=int, default=4)
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    args = ap.parse_args(argv)
    from lunarsim import dosimetry
    # run_composition scopes the flood model itself (energy LIS, flood composition)
    print(f"flood model: {dosimetry.current_model().flood().label()}")
    all_rows = []
    for g in args.only:
        all_rows += _run_depth(g, args)
        with open(args.csv, "w", newline="") as fh:     # checkpoint after each depth
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(all_rows)
    print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
