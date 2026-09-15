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
54 g/cm^2 PURE aluminium, the design validated at 0.310 mGy/day (= 12.9 uGy/h)
against OLTARIS-Total / Chang'E-4 (memory: oltaris-absorbed-dose-crosscheck). It
runs the identical bridge.run_design + dosimetry.assess path as the re-anchor,
under BOTH LIS forms, with the CURRENT CAL_old applied -- so it is directly
comparable to both the re-anchor output and the 0.310 mGy/day validation.

Interpreting the result (energy form, CAL_old applied):
  * skin ~0.31 mGy/day (~12.9 uGy/h)  -> pipeline is FINE globally; the 22 g/cm^2
    anchor under-reads for a config reason (regolith/albedo). Do NOT touch the
    global CAL; the LIS re-anchor must be reworked on a like-for-like basis.
  * skin ~0.12-0.13 mGy/day (~2.5x low) -> GLOBAL regression since CAL was set;
    fix the root cause (composition/flux/normalisation), NOT via a CAL rescale.

Target-blind: nothing here feeds a dose value into the physics. This is a
post-hoc consistency check against two independently-measured absorbed anchors.

Usage (PC)
----------
    export TOPAS_G4_DATA_DIR=~/G4Data
    python3 -m lunarsim.anchor_crosscheck runs/xcheck54          # 54 g/cm^2 pure Al
    python3 -m lunarsim.anchor_crosscheck runs/xcheck54 --mult 20 --areal 54
HEAVY -- one flood MC per LIS form. --mult mirrors the re-anchor statistics.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The independent absorbed anchor: 54 g/cm^2 PURE Al on the same 3 m dome the CAL
# is pinned to (so gauge_corr == 1, like-for-like with OUTER_GAUGE_ANCHOR_CAL).
XCHECK_SHAPE = "dome"
XCHECK_INNER_R_CM = 300.0
AL_RHO_G_CM3 = 2.70
DEFAULT_AREAL_GCM2 = 54.0
# Validated absorbed dose at 54 g/cm^2 Al (memory: oltaris-absorbed-dose-crosscheck):
VALIDATED_MGY_DAY = 0.310          # ~= 12.9 uGy/h; OLTARIS-Total / Chang'E-4 cross-check
LIS_FORMS = ("energy", "rigidity")


def _run_one(lis_var: str, areal_gcm2: float, run_dir: Path, tier) -> dict:
    """One pure-Al flood MC under the chosen LIS form; absorbed rates only."""
    from lunarsim.spec import HabitatSpec, WallLayer
    from lunarsim import bridge, dosimetry

    os.environ["LUNARSIM_GCR_LIS"] = lis_var
    dosimetry._load_make_source.cache_clear()
    t_cm = areal_gcm2 / AL_RHO_G_CM3
    spec = HabitatSpec(name=f"xcheck_{lis_var}", shape=XCHECK_SHAPE,
                       inner_radius_cm=XCHECK_INNER_R_CM,
                       walls=[WallLayer("aluminium", t_cm)])
    print(f"  [{lis_var}] flood MC pure Al {spec.areal_density_gcm2():.1f} g/cm^2 "
          f"({t_cm:.2f} cm), tier={tier.name}, {tier.total_primaries} primaries ...",
          flush=True)
    res = bridge.run_design(spec, tier=tier, run_dir=run_dir, keep=True)
    if not res.ok:
        raise SystemExit(f"[{lis_var}] MC failed (rc={res.returncode}); "
                         f"see {run_dir}/topas_stdout.log")
    a_skin = dosimetry.assess(res, skin=True)
    a_phan = dosimetry.assess(res, skin=False)
    return {
        "skin_ugy_h": a_skin.dose_rate_ugy_day / 24.0,
        "phantom_ugy_h": a_phan.dose_rate_ugy_day / 24.0,
        "skin_mgy_day": a_skin.dose_rate_ugy_day / 1000.0,
        "phantom_mgy_day": a_phan.dose_rate_ugy_day / 1000.0,
        "wall_seconds": res.wall_seconds,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir")
    p.add_argument("--areal", type=float, default=DEFAULT_AREAL_GCM2,
                   help="pure-Al areal density in g/cm^2 (default 54, the validated anchor)")
    p.add_argument("--mult", type=int, default=20,
                   help="statistics multiplier on FULL_RUN (mirror the re-anchor's 20x)")
    args = p.parse_args(argv)

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
    print(f"  validated target (energy/legacy basis): {VALIDATED_MGY_DAY:.3f} mGy/day "
          f"(= {VALIDATED_MGY_DAY*1000/24:.2f} uGy/h)")
    print(f"  statistics: {mult}x FULL_RUN = {tier.total_primaries} primaries/form")
    print(f"  current OUTER_GAUGE_ANCHOR_CAL = {cal_old:.5f}\n")

    results = {v: _run_one(v, args.areal, outdir / f"lis_{v}", tier)
               for v in LIS_FORMS}

    print("\n  absorbed dose, CAL_old applied:")
    print("  form        skin(uGy/h)  skin(mGy/d)   phantom(uGy/h)  phantom(mGy/d)")
    for v in LIS_FORMS:
        r = results[v]
        print(f"  {v:<9} {r['skin_ugy_h']:>11.3f}  {r['skin_mgy_day']:>11.4f}   "
              f"{r['phantom_ugy_h']:>13.3f}  {r['phantom_mgy_day']:>13.4f}")

    e = results["energy"]
    ratio = e["skin_mgy_day"] / VALIDATED_MGY_DAY
    print(f"\n  energy-form skin / validated = {e['skin_mgy_day']:.4f} / "
          f"{VALIDATED_MGY_DAY:.3f} = {ratio:.3f}")
    if 0.8 <= ratio <= 1.25:
        verdict = ("PIPELINE OK GLOBALLY. The 54 g/cm^2 anchor still reproduces "
                   "0.31 mGy/day, so the 22 g/cm^2 Al+regolith under-read is "
                   "config-specific (regolith albedo / heavy-ion tail). Do NOT "
                   "rescale the global CAL; rework the LIS re-anchor like-for-like.")
    elif ratio < 0.6:
        verdict = ("GLOBAL DRIFT. The 54 g/cm^2 anchor is also low, so the flood "
                   "pipeline regressed since CAL was set. Find the root cause "
                   "(composition/flux/normalisation) -- do NOT paper over it with CAL.")
    else:
        verdict = ("AMBIGUOUS (0.6-0.8x). Partial drift; inspect per-species and "
                   "convergence before any CAL decision.")
    print(f"\n  VERDICT: {verdict}")

    summary = {"areal_gcm2": args.areal, "cal_old": cal_old,
               "validated_mgy_day": VALIDATED_MGY_DAY, "mult": mult,
               "primaries_per_form": tier.total_primaries,
               "results": results, "energy_skin_over_validated": ratio}
    (outdir / "xcheck_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {outdir / 'xcheck_summary.json'}")


if __name__ == "__main__":
    main()
