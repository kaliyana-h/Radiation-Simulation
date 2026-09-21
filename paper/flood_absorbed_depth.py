#!/usr/bin/env python3
"""Same-physics GROUND TRUTH for the kernel's deep absorbed-dose plunge (PC MC).

Why this exists
---------------
The thin-wall KERNEL's effective-dose plunge with depth (805->147 mSv/yr over
10->50 g/cm^2 Al, where OLTARIS-Al is flat-to-rising) was split into absorbed
dose x quality factor on the laptop (2026-09-21, no MC). The result pinned the
defect to ABSORBED dose, not Q:

  * kernel Q-bar @50 (2.18) already MATCHES OLTARIS (2.16) -- the deep field's
    quality is right.
  * kernel absorbed @50 (0.184 mGy/day) is only 0.61x OLTARIS (0.302) and 0.60x
    the OLTARIS-validated FLOOD absorbed at the same depth (0.3065). The kernel's
    absorbed-dose depth-SLOPE is inverted: it FALLS where truth RISES.

Because the FLOOD engine runs the IDENTICAL TOPAS/FTFP_BERT_HP physics list yet
gets the deep absorbed dose right, "Geant4-vs-HZETRN cross-code fragmentation" is
ruled out -- the defect is kernel_gen's transport GEOMETRY (flat slab + a single
constant flux-weighted slant <1/cos theta> ~ 1.70x, config.py:83-97; the true
obliquity correction actually GROWS 1.32->3.21 with thickness).

This driver measures the same-physics FLOOD absorbed-dose-vs-depth curve across
the exact 19->50 g/cm^2 band where the kernel plunges. That flood curve is the
GROUND TRUTH the kernel R["D"] table must reproduce; the per-depth flood/kernel
ratio it prints IS the per-anchor over-attenuation factor the fix needs (whether
that fix is a per-anchor slant re-derivation or a true-shell kernel_gen regen).

What it records (per Al areal density, pure Al, 750 cm dome -- geometry matched
EXACTLY to the kernel column):
  * flood_skin_mgy_day  -- skin ABSORBED dose (Gy), the fast-converging channel
                           (skin rel_err ~0.5-1.6% in ~50 min; the crew phantom
                           is Bragg-noisy at depth so we do NOT chase it here).
  * flood_skin_mSv_yr, flood_skin_Q  -- context (the eff/Q the absorbed folds to).
  * kernel_dabs_mgy_day -- the kernel's absorbed dose at the SAME depth, computed
                           live (deterministic fold, no MC), for the ratio.
  * kernel_over_flood    -- kernel_dabs / flood_dabs. <1 = kernel under-reads
                           absorbed (the over-attenuation). If this FALLS with
                           depth, the slant over-attenuation is depth-growing --
                           confirmed against same-physics ground truth.

WHY converge_on="skin" (not "both"): skin absorbed is the OLTARIS-validated,
fast channel at ALL depths; the phantom is under-converged noise (rel_err 9-17%,
swings 227<->694 mSv/yr) and irrelevant to the absorbed-transport question here.

WHY the default depths are 19/30/40 (not the thin 10): the flood absorbed dose is
OLTARIS-validated at THICK (0.3065 ~ 0.302 @50) but over-reads in the thin,
un-self-shielded regime, so it is only a trustworthy GROUND TRUTH at/above the
19 g/cm^2 gate. 50 g/cm^2 is already measured (0.3065); 19/30/40 densify the
plunge region within the flood's good band. Add thin points with --depths if you
want them, but read them as flood-thin (less anchored), not ground truth.

TARGET-BLIND: nothing here feeds a dose value back into the physics. The flood is
a same-physics MC yardstick; OLTARIS is a post-hoc reference. This only measures
what the tool already returns.

Prerequisite: confirm the full-composition anchor first --
    python3 -m lunarsim.anchor_crosscheck runs/xcheck54   # expect skin ~0.31 mGy/day, ratio ~1.0
Only run this sweep once that closes.

Usage (PC)
----------
    export TOPAS_G4_DATA_DIR=~/G4Data
    cd ~/topas
    python3 paper/flood_absorbed_depth.py runs/floodabs                  # 19, 30, 40 g/cm^2
    python3 paper/flood_absorbed_depth.py runs/floodabs --depths 30 40   # just these
    python3 paper/flood_absorbed_depth.py runs/floodabs --depths 10 19 30 50 --force
HEAVY -- one full-composition (5-species) flood MC per depth, skin-converged
(~50 min each). Resumable: depths already filled in the CSV are skipped.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Geometry matched to the kernel column (accuracy_vs_depth / accuracy_flood_sweep).
AL_RHO_G_CM3 = 2.70
DOME_INNER_R_CM = 750.0
GATE_GCM2 = 19.0
DEFAULT_DEPTHS_GCM2 = [19.0, 30.0, 40.0]     # 50 already measured (0.3065); thick band only
# Hand-entered OLTARIS-Al absorbed dose (mGy/day) where known, for context only.
OLTARIS_DABS_MGY_DAY = {19.0: 0.267, 50.0: 0.302}
CSV_PATH = HERE / "flood_absorbed_depth.csv"
CSV_FIELDS = ["areal_gcm2", "al_cm", "flood_skin_mgy_day", "flood_skin_mSv_yr",
              "flood_skin_Q", "flood_skin_rel_err_pct", "kernel_dabs_mgy_day",
              "kernel_over_flood", "oltaris_dabs_mgy_day", "wall_min", "note"]


def _load_csv() -> dict[float, dict]:
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


def _spec(gcm2: float):
    from lunarsim.spec import HabitatSpec, WallLayer
    return HabitatSpec(name=f"floodabs_al{gcm2:g}", shape="dome",
                       inner_radius_cm=DOME_INNER_R_CM,
                       walls=[WallLayer("aluminium", gcm2 / AL_RHO_G_CM3)])


def _kernel_dabs(gcm2: float, phi_mv: float) -> float:
    """Kernel absorbed dose (mGy/day) at this depth -- deterministic fold, no MC.

    Pins the KERNEL LIS form (rigidity); the caller MUST re-pin the flood form
    before running any flood MC (assess_gcr_thinwall's fold flips the env LIS)."""
    from lunarsim import dosimetry
    dosimetry.set_lis_form(dosimetry.KERNEL_LIS_FORM)
    k = dosimetry.assess_gcr_thinwall(_spec(gcm2), phi_MV=phi_mv)
    return k.dose_rate_ugy_day / 1000.0


def _run_depth(gcm2: float, run_dir: Path, tier, args, kernel_dabs: float) -> dict:
    """One full-composition flood MC at this Al areal density; SKIN absorbed dose."""
    from lunarsim import dosimetry, jobs

    spec = _spec(gcm2)
    t_cm = gcm2 / AL_RHO_G_CM3
    print(f"  [{gcm2:g} g/cm^2 = {t_cm:.2f} cm Al] full-composition flood MC "
          f"(5 species, converge_on=skin, tier={tier.name}) ...", flush=True)
    t0 = time.time()
    comp = jobs.run_composition(spec, tier=tier, converge_on="skin",
                                target_rel_err=args.rel_err, min_batches=2,
                                max_batches=args.max_batches,
                                lis=dosimetry.FLOOD_LIS_FORM)
    if comp.returncode != 0 or not comp.species_results:
        raise SystemExit(f"[{gcm2:g}] composition MC failed "
                         f"(rc={comp.returncode}); see {run_dir}")
    a_skin = dosimetry.assess_composition(comp.species_results,
                                          phi_MV=tier.phi_mv, skin=True)
    wall_min = (time.time() - t0) / 60.0

    flood_dabs = a_skin.dose_rate_ugy_day / 1000.0          # mGy/day
    ratio = kernel_dabs / flood_dabs if flood_dabs else float("nan")
    olt = OLTARIS_DABS_MGY_DAY.get(round(gcm2, 1))
    print(f"      -> flood skin ABSORBED {flood_dabs:.4f} mGy/day "
          f"(Q {a_skin.quality_factor:.2f}, {a_skin.annual_msv:.1f} mSv/yr, "
          f"rel_err {(a_skin.rel_err or 0)*100:.1f}%) | wall {wall_min:.1f} min\n"
          f"         kernel absorbed {kernel_dabs:.4f} mGy/day  ->  "
          f"kernel/flood = {ratio:.3f}"
          + (f"  (OLTARIS {olt:.3f})" if olt else ""), flush=True)

    note = (f"2026-09-21 flood absorbed-depth (750cm dome, skin-converged); "
            f"skin rel_err {(a_skin.rel_err or 0)*100:.1f}%")
    return {"areal_gcm2": f"{gcm2:g}", "al_cm": f"{t_cm:.2f}",
            "flood_skin_mgy_day": f"{flood_dabs:.4f}",
            "flood_skin_mSv_yr": f"{a_skin.annual_msv:.1f}",
            "flood_skin_Q": f"{a_skin.quality_factor:.2f}",
            "flood_skin_rel_err_pct": f"{(a_skin.rel_err or 0)*100:.1f}",
            "kernel_dabs_mgy_day": f"{kernel_dabs:.4f}",
            "kernel_over_flood": f"{ratio:.3f}",
            "oltaris_dabs_mgy_day": f"{olt:.3f}" if olt else "",
            "wall_min": f"{wall_min:.1f}", "note": note}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir", help="run scratch dir (per-depth subdirs created under it)")
    p.add_argument("--depths", type=float, nargs="+", metavar="GCM2",
                   help=f"Al areal densities g/cm^2 (default {DEFAULT_DEPTHS_GCM2})")
    p.add_argument("--force", action="store_true",
                   help="recompute depths even if already filled in the CSV")
    p.add_argument("--rel-err", type=float, default=0.05,
                   help="per-species target relative error (default 0.05)")
    p.add_argument("--max-batches", type=int, default=12,
                   help="max convergence batches per species (default 12)")
    args = p.parse_args(argv)

    from lunarsim import bridge, dosimetry
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tier = bridge.FULL_RUN
    depths = args.depths if args.depths else list(DEFAULT_DEPTHS_GCM2)

    # Kernel absorbed is a free deterministic fold: compute ALL depths up front
    # (this pins the KERNEL LIS form), THEN pin the flood form once for the whole
    # MC loop -- avoids per-depth cache flips between the two engines.
    print("  kernel absorbed (deterministic fold, no MC):")
    kern = {}
    for g in depths:
        kern[round(g, 1)] = _kernel_dabs(g, tier.phi_mv)
        print(f"    {g:>5g} g/cm^2 -> {kern[round(g, 1)]:.4f} mGy/day")
    dosimetry.set_lis_form(dosimetry.FLOOD_LIS_FORM)
    print(f"\n  LIS form pinned for flood MC: {dosimetry.FLOOD_LIS_FORM} "
          f"(flood CAL basis)\n  writing -> {CSV_PATH}")

    thin = [g for g in depths if g < GATE_GCM2]
    if thin:
        print(f"  NOTE: {', '.join('%g' % g for g in thin)} g/cm^2 are BELOW the "
              f"{GATE_GCM2:g} gate -- flood absorbed there is thin-regime (over-reads, "
              f"less anchored), NOT ground truth. Reading them for trend only.")
    print()

    rows = _load_csv()
    for g in depths:
        key = round(g, 1)
        existing = rows.get(key)
        if existing and (existing.get("flood_skin_mgy_day") or "").strip() and not args.force:
            print(f"  [{g:g}] already filled (flood_skin="
                  f"{existing['flood_skin_mgy_day']}); skip (use --force to redo).")
            continue
        row = _run_depth(g, outdir / f"al{g:g}", tier, args, kern[key])
        if existing and (existing.get("note") or "").strip():
            row["note"] = existing["note"].strip() + " | " + row["note"]
        rows[key] = row
        _write_csv(rows)             # checkpoint after every depth (resumable)
        print(f"  [{g:g}] written.\n")

    # Trend summary: does kernel/flood FALL with depth (depth-growing over-attenuation)?
    filled = [r for r in sorted(rows.values(), key=lambda r: float(r["areal_gcm2"]))
              if (r.get("flood_skin_mgy_day") or "").strip()]
    if filled:
        print("  === absorbed-dose ground truth (kernel must reproduce flood) ===")
        print(f"  {'g/cm2':>6} {'flood_Dabs':>10} {'kernel_Dabs':>11} {'kern/flood':>10} "
              f"{'OLTARIS':>8}")
        for r in filled:
            print(f"  {float(r['areal_gcm2']):>6g} {r['flood_skin_mgy_day']:>10} "
                  f"{r['kernel_dabs_mgy_day']:>11} {r['kernel_over_flood']:>10} "
                  f"{(r.get('oltaris_dabs_mgy_day') or '--'):>8}")
        print("\n  kern/flood FALLING with depth => kernel_gen slant over-attenuation is\n"
              "  depth-growing (the fix target). FLAT => a scale offset, not a slope defect.")


if __name__ == "__main__":
    main()
