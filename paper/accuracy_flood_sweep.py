#!/usr/bin/env python3
"""Fill the FLOOD column of the accuracy-vs-depth map (PC-driven, heavy MC).

Why this exists
---------------
`paper/accuracy_vs_depth.py` reads three columns: the OLTARIS-Al reference
(hand-entered), the thin-wall KERNEL (computed live, no MC), and the FLOOD MC
column -- which until now had to be filled by hand, one PC run per depth. This
driver sweeps the map's flood-regime grid (>= the 19 g/cm^2 gate) in one
command and writes `paper/accuracy_al_flood.csv` directly.

For each aluminium areal density it runs the FULL GCR composition
(H+He+C+Si+Fe via jobs.run_composition) -- the SAME path production/GUI uses --
and records two effective-dose-equivalent numbers the map compares to OLTARIS:

  * flood_eff_mSv_yr   = crew-phantom effective dose equivalent (skin=False).
                         This is the PRODUCTION value accuracy_vs_depth reads
                         above the gate (the 259.1 phantom-ICRP basis @ 50).
  * flood_skin_mSv_yr  = skin dose equivalent (skin=True). The independently
                         OLTARIS-validated 287.7-@-50 basis (crossover memory).

GEOMETRY is matched EXACTLY to the map's kernel column so the two tool curves
are like-for-like: dome, inner_r = 750 cm (accuracy_vs_depth.DOME_INNER_R_CM),
pure aluminium. The outer fluence gauge is size-independent at this radius
(memory: outer-fluence-gauge-fix), so gauge_corr ~ 1.

Convergence: the crew phantom is Bragg-noisy at depth, so this converges on
BOTH skin and phantom (converge_on="both"). That is the expensive part; the
sweep is resumable -- depths already present in the CSV (non-blank flood_eff)
are skipped unless --force.

TARGET-BLIND: nothing here feeds a dose value back into the physics. OLTARIS is
a post-hoc reference; this only measures what the tool already returns.

Prerequisite: the full-composition anchor must first be confirmed with
`python3 -m lunarsim.anchor_crosscheck runs/xcheck54` (expect skin ~0.31
mGy/day, ratio ~1.0). Only run this sweep once that closes.

Usage (PC)
----------
    export TOPAS_G4_DATA_DIR=~/G4Data
    cd ~/topas
    python3 paper/accuracy_flood_sweep.py runs/floodsweep              # all gate+ depths
    python3 paper/accuracy_flood_sweep.py runs/floodsweep --only 50 54  # just these
    python3 paper/accuracy_flood_sweep.py runs/floodsweep --force       # recompute all
HEAVY -- one full-composition (5-species) flood MC per depth, both-converged.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Geometry matched to the map's kernel column (accuracy_vs_depth.py) so the two
# tool curves are like-for-like.
AL_RHO_G_CM3 = 2.70
DOME_INNER_R_CM = 750.0
GATE_GCM2 = 19.0
# The map's flood-regime grid (>= gate). Kept in sync with accuracy_vs_depth.GRID.
FLOOD_GRID_GCM2 = [19.0, 25.0, 30.0, 40.0, 50.0, 54.0]
CSV_PATH = HERE / "accuracy_al_flood.csv"
CSV_FIELDS = ["areal_gcm2", "al_cm", "flood_eff_mSv_yr", "flood_skin_mSv_yr", "note"]


def _load_csv() -> dict[float, dict]:
    """Existing rows keyed by rounded areal density, so we can update in place."""
    rows: dict[float, dict] = {}
    if CSV_PATH.exists():
        for r in csv.DictReader(open(CSV_PATH, newline="")):
            try:
                rows[round(float(r["areal_gcm2"]), 1)] = r
            except (ValueError, KeyError):
                continue
    return rows


def _write_csv(rows: dict[float, dict]) -> None:
    ordered = sorted(rows.values(), key=lambda r: float(r["areal_gcm2"]))
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def _run_depth(gcm2: float, run_dir: Path, tier, args) -> dict:
    """One full-composition flood MC at this Al areal density; effective doses."""
    from lunarsim.spec import HabitatSpec, WallLayer
    from lunarsim import dosimetry, jobs

    t_cm = gcm2 / AL_RHO_G_CM3
    spec = HabitatSpec(name=f"floodsweep_al{gcm2:g}", shape="dome",
                       inner_radius_cm=DOME_INNER_R_CM,
                       walls=[WallLayer("aluminium", t_cm)])
    print(f"  [{gcm2:g} g/cm^2 = {t_cm:.2f} cm Al] full-composition flood MC "
          f"(5 species, converge_on=both, tier={tier.name}) ...", flush=True)
    comp = jobs.run_composition(spec, tier=tier, converge_on="both",
                                target_rel_err=args.rel_err,
                                min_batches=2, max_batches=args.max_batches)
    if comp.returncode != 0 or not comp.species_results:
        raise SystemExit(f"[{gcm2:g}] composition MC failed "
                         f"(rc={comp.returncode}); see {run_dir}")
    a_phan = dosimetry.assess_composition(comp.species_results,
                                          phi_MV=tier.phi_mv, skin=False)
    a_skin = dosimetry.assess_composition(comp.species_results,
                                          phi_MV=tier.phi_mv, skin=True)
    eff, skin = a_phan.annual_msv, a_skin.annual_msv
    print(f"      -> phantom(eff) {eff:7.1f} mSv/yr (rel_err "
          f"{(a_phan.rel_err or 0)*100:4.1f}%),  skin {skin:7.1f} mSv/yr "
          f"(rel_err {(a_skin.rel_err or 0)*100:4.1f}%),  "
          f"wall {comp.wall_seconds/60:.1f} min", flush=True)
    note = (f"2026-09-16 full-comp flood sweep (750cm dome, both-converged); "
            f"phantom rel_err {(a_phan.rel_err or 0)*100:.1f}%, "
            f"skin rel_err {(a_skin.rel_err or 0)*100:.1f}%")
    return {"areal_gcm2": f"{gcm2:g}", "al_cm": f"{t_cm:.2f}",
            "flood_eff_mSv_yr": f"{eff:.1f}", "flood_skin_mSv_yr": f"{skin:.1f}",
            "note": note}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir", help="run scratch dir (per-depth subdirs created under it)")
    p.add_argument("--only", type=float, nargs="+", metavar="GCM2",
                   help="restrict to these areal densities (default: whole flood grid)")
    p.add_argument("--force", action="store_true",
                   help="recompute depths even if already filled in the CSV")
    p.add_argument("--rel-err", type=float, default=0.05,
                   help="per-species target relative error (default 0.05)")
    p.add_argument("--max-batches", type=int, default=12,
                   help="max convergence batches per species (default 12)")
    args = p.parse_args(argv)

    from lunarsim import bridge
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tier = bridge.FULL_RUN

    targets = args.only if args.only else FLOOD_GRID_GCM2
    below = [g for g in targets if g < GATE_GCM2]
    if below:
        print(f"  NOTE: {', '.join('%g' % g for g in below)} g/cm^2 are BELOW the "
              f"{GATE_GCM2:g} gate (kernel regime); flooding them anyway as requested.")

    rows = _load_csv()
    print(f"Flood-column sweep: {len(targets)} depth(s), 750 cm dome, pure Al, "
          f"full GCR composition\n  writing -> {CSV_PATH}\n")

    for g in targets:
        key = round(g, 1)
        existing = rows.get(key)
        if existing and (existing.get("flood_eff_mSv_yr") or "").strip() and not args.force:
            print(f"  [{g:g}] already filled (flood_eff="
                  f"{existing['flood_eff_mSv_yr']}); skip (use --force to redo).")
            continue
        row = _run_depth(g, outdir / f"al{g:g}", tier, args)
        # preserve any pre-existing hand note by appending
        if existing and (existing.get("note") or "").strip():
            row["note"] = existing["note"].strip() + " | " + row["note"]
        rows[key] = row
        _write_csv(rows)            # checkpoint after every depth (resumable)
        print(f"  [{g:g}] written.\n")

    print(f"Done. Regenerate the map with:\n"
          f"    python3 paper/accuracy_vs_depth.py            # table\n"
          f"    python3 paper/accuracy_vs_depth.py --plot     # + F13 figure")


if __name__ == "__main__":
    main()
