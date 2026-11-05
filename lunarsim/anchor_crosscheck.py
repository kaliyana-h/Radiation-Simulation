"""Absorbed-dose cross-check: is the flood pipeline still self-consistent?

Why this exists
---------------
`rerun_lis_fix.py anchor` re-anchors OUTER_GAUGE_ANCHOR_CAL so the *rigidity*
LIS reproduces the Chang'E-4 surface anchor (13.2 uGy/h absorbed at 22 g/cm^2,
5 cm Al + 5 cm regolith). But that run revealed that the *legacy* (energy) form
-- the basis the current CAL was pinned on -- reads only ~5.3 uGy/h there, i.e.
~2.5x BELOW 13.2. If the legacy form no longer reproduces its own anchor, the
re-anchored CAL is double-counting a hidden drift, and pushing it into
dosimetry.py would multiply EVERY dose by ~5.7x and break the OLTARIS-validated
thick-shield agreement (skin 287.7 -> ~1630 mSv/yr at 50 g/cm^2).

Before touching CAL we must know whether the drift is:
  (A) config-specific  -- only the 22 g/cm^2 Al+regolith dome under-reads
      (regolith albedo / heavy-ion surface tail), OR
  (B) global           -- the whole flood pipeline now reads low.

This harness measures ABSORBED dose at the *other*, independent absorbed anchor:
54 g/cm^2 PURE aluminium, compared post-hoc against OLTARIS lunar-surface TOTAL
0.296 mGy/day (memory: oltaris-absorbed-dose-crosscheck). It runs the FULL GCR
composition (H+He+C+Si+Fe via jobs.run_composition + dosimetry.assess_composition)
-- the SAME path production/GUI uses -- under BOTH LIS forms, with the CURRENT
CAL applied.

UPDATED 2026-11-02: the reference used to be the tool's own earlier 0.310 mGy/day.
That number was produced with the hand-set GCR heavy abundances (C/Si/Fe 4.7-7.1x
Matthiae) and the 2.0 /cm2/s proton pin, both since removed (memory flood-heavy-
abundance-overweight), so it was a stale self-reference, not a measurement. The
comparand is now the external OLTARIS-Total value; the Matthiae-absolute flood
measured 0.260 mGy/day on the PC (2026-10, also in paper/lnd_absolute_recheck.py)
= 0.88x, inside the cross-code band.

ROOT CAUSE FOUND (2026-09-15): the earlier "~2.75x global drift" was a HARNESS
BUG, not a pipeline regression. The old version called bare bridge.run_design +
dosimetry.assess, which default to PROTONS ONLY (bridge.py:556). It measured H
alone -- ~0.36x of the full-field dose (that IS the proton dose fraction) -- and
compared it against the full-composition 0.310 / 13.2 anchors. Apples-to-oranges.
rerun_lis_fix.py anchor has the identical flaw. Fixing the harness to run the
full composition removes the apparent drift; CAL is untouched and correct.

Interpreting the result (energy form, current CAL applied):
  * skin 0.8-1.25x OLTARIS-Total      -> no global drift; the flood pipeline is
    consistent with an independent code. Do NOT touch CAL.
  * skin < 0.6x                       -> a genuine drift would remain; then find
    the root cause (composition/flux/normalisation), NOT via a CAL rescale.

Target-blind: nothing here feeds a dose value into the physics. This is a
post-hoc consistency check against two independently-measured absorbed anchors.

Usage (PC)
----------
    export TOPAS_G4_DATA_DIR=~/G4Data
    python3 -m lunarsim.anchor_crosscheck runs/xcheck54          # 54 g/cm^2 pure Al
    python3 -m lunarsim.anchor_crosscheck runs/xcheck54 --mult 20 --areal 54
    python3 -m lunarsim.anchor_crosscheck runs/xcheck54 --forms energy   # flood form only
HEAVY -- one flood MC per LIS form. --mult mirrors the re-anchor statistics.
The summary JSON carries the provenance stamp (git, GCR model, CAL) of each form's
assessment, so a re-verification is tied to the code that produced it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The independent absorbed anchor: 54 g/cm^2 PURE Al on the same 3 m dome the CAL
# is pinned to (so gauge_corr == 1, like-for-like with OUTER_GAUGE_ANCHOR_CAL).
XCHECK_SHAPE = "dome"
XCHECK_INNER_R_CM = 300.0
AL_RHO_G_CM3 = 2.70
DEFAULT_AREAL_GCM2 = 54.0
# Post-hoc external comparand at 54 g/cm^2 Al (memory: oltaris-absorbed-dose-crosscheck):
# OLTARIS lunar-surface TOTAL, not the "Freespace GCR effects only" component.
OLTARIS_TOTAL_MGY_DAY = 0.296      # ~= 12.3 uGy/h
LIS_FORMS = ("energy", "rigidity")


def _run_one(lis_var: str, areal_gcm2: float, run_dir: Path, tier) -> dict:
    """One pure-Al flood MC under the chosen LIS form; absorbed rates only.

    Runs the FULL GCR composition (H+He+C+Si+Fe), one MC per species, summed by
    dosimetry.assess_composition -- IDENTICAL to the production path (jobs.run_
    composition -> gui). This is what the OLTARIS-Total comparand is: full-field
    absorbed dose. (The earlier proton-only run_design +
    assess measured H alone, ~0.36x of the total -- the proton dose fraction, not
    a pipeline drift; that comparison was apples-to-oranges.)"""
    from lunarsim.spec import HabitatSpec, WallLayer
    from lunarsim import bridge, dosimetry, jobs

    # run_composition is handed lis=lis_var so it runs (beam AND normalisation)
    # under this form instead of the flood default (energy); the result carries
    # that model, and the assessments below are normalised under the SAME model,
    # so each form's number uses its own consistent flux normalisation.
    t_cm = areal_gcm2 / AL_RHO_G_CM3
    spec = HabitatSpec(name=f"xcheck_{lis_var}", shape=XCHECK_SHAPE,
                       inner_radius_cm=XCHECK_INNER_R_CM,
                       walls=[WallLayer("aluminium", t_cm)])
    print(f"  [{lis_var}] full-composition flood MC pure Al "
          f"{spec.areal_density_gcm2():.1f} g/cm^2 ({t_cm:.2f} cm), 5 species, "
          f"tier={tier.name} per species/batch ...", flush=True)
    # Skin converges fast at all depths (crossover-discontinuity memory); the
    # crew phantom is Bragg-noisy, so converge on skin and report both.
    comp = jobs.run_composition(spec, tier=tier, converge_on="skin",
                                target_rel_err=0.03, min_batches=2, max_batches=8,
                                lis=lis_var)
    if comp.returncode != 0 or not comp.species_results:
        raise SystemExit(f"[{lis_var}] composition MC failed (rc={comp.returncode}); "
                         f"see {run_dir}")
    a_skin = dosimetry.assess_composition(comp.species_results,
                                          phi_MV=tier.phi_mv, skin=True,
                                          model=comp.model)
    a_phan = dosimetry.assess_composition(comp.species_results,
                                          phi_MV=tier.phi_mv, skin=False,
                                          model=comp.model)
    # skin dose equivalent / absorbed = the lining's mean Q (ICRP-60), same run
    qbar = a_skin.quality_factor
    # per-species absorbed skin dose fraction, so the proton fraction is visible
    # (assess_composition already builds this: dose_rate_gy_s + dose_fraction per row)
    per_species = {c["species"]: {"frac": c.get("dose_fraction"),
                                  "ugy_h": c["dose_rate_gy_s"] * 1e6 * 3600.0}
                   for c in (a_skin.contributions or [])}
    return {
        "skin_ugy_h": a_skin.dose_rate_ugy_day / 24.0,
        "phantom_ugy_h": a_phan.dose_rate_ugy_day / 24.0,
        "skin_mgy_day": a_skin.dose_rate_ugy_day / 1000.0,
        "phantom_mgy_day": a_phan.dose_rate_ugy_day / 1000.0,
        "skin_rel_err": a_skin.rel_err,
        "skin_msv_yr": a_skin.annual_msv,
        "skin_qbar": qbar,
        "provenance": dosimetry.provenance_str(a_skin.provenance),
        "per_species_skin_ugy_h": per_species,
        "wall_seconds": comp.wall_seconds,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir")
    p.add_argument("--areal", type=float, default=DEFAULT_AREAL_GCM2,
                   help="pure-Al areal density in g/cm^2 (default 54, the validated anchor)")
    p.add_argument("--mult", type=int, default=1,
                   help="per-species per-batch size multiplier on FULL_RUN "
                        "(statistics now come from run_composition's convergence "
                        "rounds over 5 species, so 1 is usually enough)")
    p.add_argument("--forms", default=",".join(LIS_FORMS),
                   help="comma list of LIS forms to run (default energy,rigidity). "
                        "The flood engine is pinned to energy, so '--forms energy' "
                        "is the production re-verification at half the cost.")
    args = p.parse_args(argv)
    forms = tuple(f.strip() for f in args.forms.split(",") if f.strip())
    bad = [f for f in forms if f not in LIS_FORMS]
    if bad or "energy" not in forms:
        p.error(f"--forms must be drawn from {LIS_FORMS} and include energy (got {args.forms})")

    from lunarsim import bridge, dosimetry
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cal_old = dosimetry.OUTER_GAUGE_ANCHOR_CAL

    mult = max(1, int(args.mult))
    base = bridge.FULL_RUN
    tier = dataclasses.replace(base, name=f"xcheck{mult}x",
                               histories=base.histories * mult)

    print(f"Absorbed-dose cross-check: {args.areal:g} g/cm^2 pure Al, "
          f"{XCHECK_SHAPE} inner_r={XCHECK_INNER_R_CM:.0f} cm")
    print(f"  post-hoc comparand (OLTARIS-Total): {OLTARIS_TOTAL_MGY_DAY:.3f} mGy/day "
          f"(= {OLTARIS_TOTAL_MGY_DAY*1000/24:.2f} uGy/h)")
    print(f"  statistics: {mult}x FULL_RUN = {tier.total_primaries} primaries/form")
    print(f"  current OUTER_GAUGE_ANCHOR_CAL = {cal_old:.5f}\n")

    results = {v: _run_one(v, args.areal, outdir / f"lis_{v}", tier)
               for v in forms}

    print("\n  FULL-COMPOSITION absorbed dose, CAL_old applied:")
    print("  form        skin(uGy/h)  skin(mGy/d)   phantom(uGy/h)  phantom(mGy/d)"
          "  skin(mSv/yr)  skin Qbar")
    for v in forms:
        r = results[v]
        q = r.get("skin_qbar")
        print(f"  {v:<9} {r['skin_ugy_h']:>11.3f}  {r['skin_mgy_day']:>11.4f}   "
              f"{r['phantom_ugy_h']:>13.3f}  {r['phantom_mgy_day']:>13.4f}"
              f"  {r['skin_msv_yr']:>12.1f}  {('%.3f' % q) if q else '--':>9}")
        print(f"    provenance: {r['provenance']}")

    # per-species breakdown -- the proton fraction is the key number: the old
    # proton-only harness read ~this fraction of the full-composition anchor.
    for v in forms:
        ps = results[v].get("per_species_skin_ugy_h") or {}
        if ps:
            print(f"\n  [{v}] per-species skin absorbed dose (uGy/h, dose fraction):")
            for name in ("H", "He", "C", "Si", "Fe"):
                if name in ps:
                    d = ps[name]
                    fr = d.get("frac")
                    print(f"      {name:<3} {d['ugy_h']:>9.3f}   "
                          f"{('%.3f' % fr) if fr is not None else '--':>6}")

    e = results["energy"]
    ratio = e["skin_mgy_day"] / OLTARIS_TOTAL_MGY_DAY
    print(f"\n  energy-form FULL-composition skin / OLTARIS-Total = {e['skin_mgy_day']:.4f} / "
          f"{OLTARIS_TOTAL_MGY_DAY:.3f} = {ratio:.3f}")
    if 0.8 <= ratio <= 1.25:
        verdict = ("PIPELINE OK. The full-composition 54 g/cm^2 absorbed dose is "
                   f"{ratio:.2f}x OLTARIS-Total, within the cross-code band -- NO "
                   "global drift. Do NOT rescale CAL.")
    elif ratio < 0.6:
        verdict = ("STILL LOW even at full composition -- a genuine drift remains. "
                   "Find the root cause (composition/flux/normalisation), NOT via "
                   "a CAL rescale. Inspect the per-species breakdown and convergence.")
    else:
        verdict = ("AMBIGUOUS (0.6-0.8x). Inspect per-species and convergence "
                   "before any CAL decision.")
    print(f"\n  VERDICT: {verdict}")

    summary = {"areal_gcm2": args.areal, "cal_old": cal_old,
               "oltaris_total_mgy_day": OLTARIS_TOTAL_MGY_DAY, "mult": mult,
               "forms": list(forms),
               "composition": "full GCR (H+He+C+Si+Fe), converge_on=skin",
               "results": results, "energy_skin_over_oltaris": ratio}
    (outdir / "xcheck_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {outdir / 'xcheck_summary.json'}")


if __name__ == "__main__":
    main()
