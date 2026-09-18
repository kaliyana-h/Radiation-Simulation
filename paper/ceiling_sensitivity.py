#!/usr/bin/env python3
"""Energy-ceiling sensitivity: is the deep-shield flood dropoff a truncation artifact?

Why this exists
---------------
The tool's dose-vs-Al curve DROPS steeply with depth (kernel 1177->147, flood
skin 685->445 over 19->30 g/cm^2) while OLTARIS is FLAT-to-RISING (effective
228->238, absorbed 0.267->0.302 mGy/day @ 19->50). GCR dose is tail-dominated:
the penetrating multi-GeV/nuc tail is what keeps deep dose flat. The source
ceiling is pinned at 20 GeV/nuc (make_source.GCR_EMAX_PER_NUC) for Fe compute
cost -- and its "only 5.6% under-report" was measured at ~22 g/cm^2 (5 cm Al +
30 cm regolith), NOT at these deep pure-Al depths. If the ceiling truncation is
the cause, the surviving deep dose is disproportionately carried by the
>20 GeV/nuc tail TOPAS never samples, so RAISING the ceiling should lift the
deep dose toward OLTARIS.

This runs ONE depth (default 50 g/cm^2 pure Al, 750 cm dome -- IDENTICAL geometry
to accuracy_flood_sweep so it is like-for-like with the flood column) at WHATEVER
ceiling is in the environment, and appends the result to a comparison CSV. Run it
once per ceiling:

    export TOPAS_G4_DATA_DIR=~/G4Data
    cd ~/topas
    LUNARSIM_GCR_EMAX_PER_NUC=2.0e4 python3 paper/ceiling_sensitivity.py runs/ceil20   # 20 GeV/nuc (baseline)
    LUNARSIM_GCR_EMAX_PER_NUC=1.0e5 python3 paper/ceiling_sensitivity.py runs/ceil100  # 100 GeV/nuc

CRITICAL: the ceiling MUST be set in the SHELL before the process starts (as
above), NOT switched inside one process. make_source.GCR_EMAX_PER_NUC and the
dosimetry flux normalisation both read it at IMPORT; a fresh process per ceiling
is the only way the sampled spectrum and its normalisation stay on the SAME
ceiling. The script prints the ceiling actually in effect so you can verify.

At 100 GeV/nuc, H stays cheap (100 GeV total) but Fe reaches 5.6 TeV/primary and
its FTFP_BERT_HP showers are slow -- expect this run to take substantially longer
than the 20 GeV/nuc baseline. That cost is the point: it buys the answer to
"is the deep dropoff physical, or a truncation artifact?".

TARGET-BLIND: nothing here feeds a dose value into the physics. The ceiling is a
source-sampling knob; OLTARIS is only a post-hoc yardstick for the trend.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AL_RHO_G_CM3 = 2.70
DOME_INNER_R_CM = 750.0                     # matches accuracy_flood_sweep / kernel column
CSV_PATH = HERE / "ceiling_sensitivity.csv"
CSV_FIELDS = ["emax_per_nuc_mev", "emax_gev_per_nuc", "areal_gcm2", "al_cm",
              "skin_mgy_day", "skin_mSv_yr", "phantom_mSv_yr",
              "skin_rel_err_pct", "phantom_rel_err_pct", "wall_min", "note"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir", help="run scratch dir")
    p.add_argument("--areal", type=float, default=50.0,
                   help="pure-Al areal density g/cm^2 (default 50, the deep-shield point)")
    p.add_argument("--rel-err", type=float, default=0.05,
                   help="per-species target relative error (default 0.05)")
    p.add_argument("--max-batches", type=int, default=12,
                   help="max convergence batches per species (default 12)")
    args = p.parse_args(argv)

    # Import AFTER argparse but the ceiling is already locked from the shell env at
    # make_source import time. Pin LIS to the flood (energy) basis -- the ceiling
    # test must ride on the same spectral form the flood column was validated on.
    from lunarsim.spec import HabitatSpec, WallLayer
    from lunarsim import bridge, dosimetry, jobs
    from lunarsim import make_source as ms
    dosimetry.set_lis_form(dosimetry.FLOOD_LIS_FORM)

    emax = ms.GCR_EMAX_PER_NUC                       # the ceiling ACTUALLY in effect
    env_raw = os.environ.get("LUNARSIM_GCR_EMAX_PER_NUC", "<unset -> module default>")
    print(f"  ENERGY CEILING IN EFFECT: {emax:.4g} MeV/nuc = {emax/1000:.4g} GeV/nuc "
          f"(env LUNARSIM_GCR_EMAX_PER_NUC={env_raw})")
    print(f"  LIS form: {dosimetry.FLOOD_LIS_FORM}  |  geometry: {DOME_INNER_R_CM:.0f} cm dome, pure Al\n")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tier = bridge.FULL_RUN

    t_cm = args.areal / AL_RHO_G_CM3
    spec = HabitatSpec(name=f"ceil_al{args.areal:g}", shape="dome",
                       inner_radius_cm=DOME_INNER_R_CM,
                       walls=[WallLayer("aluminium", t_cm)])
    print(f"  [{args.areal:g} g/cm^2 = {t_cm:.2f} cm Al] full-composition flood MC "
          f"(5 species, converge_on=both, tier={tier.name}) ...", flush=True)
    t0 = time.time()
    comp = jobs.run_composition(spec, tier=tier, converge_on="both",
                                target_rel_err=args.rel_err, min_batches=2,
                                max_batches=args.max_batches,
                                lis=dosimetry.FLOOD_LIS_FORM)
    if comp.returncode != 0 or not comp.species_results:
        raise SystemExit(f"composition MC failed (rc={comp.returncode}); see {outdir}")
    a_phan = dosimetry.assess_composition(comp.species_results, phi_MV=tier.phi_mv, skin=False)
    a_skin = dosimetry.assess_composition(comp.species_results, phi_MV=tier.phi_mv, skin=True)
    wall_min = (time.time() - t0) / 60.0

    skin_mgy_day = a_skin.dose_rate_ugy_day / 1000.0
    skin_eff = a_skin.annual_msv
    phan_eff = a_phan.annual_msv
    print(f"\n      -> skin absorbed {skin_mgy_day:.4f} mGy/day | skin {skin_eff:7.1f} mSv/yr "
          f"(rel_err {(a_skin.rel_err or 0)*100:4.1f}%) | phantom(eff) {phan_eff:7.1f} mSv/yr "
          f"(rel_err {(a_phan.rel_err or 0)*100:4.1f}%) | wall {wall_min:.1f} min")

    row = {
        "emax_per_nuc_mev": f"{emax:.6g}", "emax_gev_per_nuc": f"{emax/1000:.4g}",
        "areal_gcm2": f"{args.areal:g}", "al_cm": f"{t_cm:.2f}",
        "skin_mgy_day": f"{skin_mgy_day:.4f}", "skin_mSv_yr": f"{skin_eff:.1f}",
        "phantom_mSv_yr": f"{phan_eff:.1f}",
        "skin_rel_err_pct": f"{(a_skin.rel_err or 0)*100:.1f}",
        "phantom_rel_err_pct": f"{(a_phan.rel_err or 0)*100:.1f}",
        "wall_min": f"{wall_min:.1f}",
        "note": f"ceiling sensitivity @ {args.areal:g} g/cm^2 Al, 750cm dome, energy LIS",
    }

    # Accumulate rows keyed by (ceiling, areal) so both ceilings coexist for compare.
    rows: dict[tuple, dict] = {}
    if CSV_PATH.exists():
        for r in csv.DictReader(open(CSV_PATH, newline="")):
            try:
                rows[(round(float(r["emax_per_nuc_mev"]), 1), round(float(r["areal_gcm2"]), 1))] = r
            except (ValueError, KeyError):
                continue
    rows[(round(emax, 1), round(args.areal, 1))] = row
    ordered = sorted(rows.values(),
                     key=lambda r: (float(r["areal_gcm2"]), float(r["emax_per_nuc_mev"])))
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"\n  wrote {CSV_PATH}")

    # If both a low and a high ceiling exist at this areal, print the comparison.
    same_areal = [r for r in ordered if round(float(r["areal_gcm2"]), 1) == round(args.areal, 1)]
    if len(same_areal) >= 2:
        print(f"\n  === ceiling comparison @ {args.areal:g} g/cm^2 pure Al ===")
        print("   ceiling(GeV/nuc)  skin(mGy/d)  skin(mSv/yr)  phantom(mSv/yr)")
        for r in sorted(same_areal, key=lambda r: float(r["emax_per_nuc_mev"])):
            print(f"   {float(r['emax_gev_per_nuc']):>15.4g}  {r['skin_mgy_day']:>11}  "
                  f"{r['skin_mSv_yr']:>12}  {r['phantom_mSv_yr']:>14}")
        lo = min(same_areal, key=lambda r: float(r["emax_per_nuc_mev"]))
        hi = max(same_areal, key=lambda r: float(r["emax_per_nuc_mev"]))
        if float(hi["emax_per_nuc_mev"]) > float(lo["emax_per_nuc_mev"]):
            rs = float(hi["skin_mgy_day"]) / float(lo["skin_mgy_day"])
            print(f"\n   skin absorbed ratio (high/low ceiling) = {rs:.3f}")
            print("   >1 by more than the ~1.06 seen at 22 g/cm^2 => deep dropoff is a")
            print("   truncation artifact (the missing >ceiling tail grows with depth).")


if __name__ == "__main__":
    main()
