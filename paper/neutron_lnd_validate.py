#!/usr/bin/env python3
"""Step 4 -- target-blind validation of the secondary-neutron dose-equivalent
fraction diagnostic against Chang'E-4 LND.

Unlike the Step 3 kernel check (paper/perelem_validate.py, pure arithmetic), this
harness RUNS the flood MC: it needs TOPAS + G4Data, so it runs on the PC, not the
laptop. It drives a small wall-thickness sweep of a SURFACE habitat over regolith
through the flood path -- the only path that carries the CrewSkin neutron-lineage
twin scorer (DoseEquivalent_ICRP_Neutron) -- and reads back

    f = H_neutron / H_total   (dose-equivalent-weighted, tissue CrewSkin lining)

the fraction of behind-shield crew dose-equivalent carried by wall/regolith-bred
(albedo) secondary neutrons. f is a RATIO of two dose-equivalents scored in the
SAME run, so flux/gauge normalisation cancels: it is dimensionless, carries NO
calibration, and CAL is NOT touched anywhere in this script.

It is TARGET-BLIND. Every Chang'E-4 / LND number below is a POST-HOC reference,
never a fit target; nothing here is tuned to reproduce them.

WHY A SURFACE SWEEP. LND sat on the lunar surface: open sky above, regolith below,
minimal overhead. The tool's albedo source is the regolith floor (always present
via lunar_environment.txt) plus the engineered wall. A THIN-wall surface dome is
therefore the LND analogue -- regolith-floor albedo dominant, little wall. Thicker
walls add wall-bred neutrons AND attenuate the primary (non-neutron) dose, so f
must RISE with wall areal density: that monotone trend is a falsifiable physics
check that needs no reference at all.

THE DEFINITION GAP (read before judging the numbers). The tool's f is a
NEUTRON-LINEAGE dose-equivalent fraction in TISSUE. LND measured a NEUTRAL-PARTICLE
absorbed-dose fraction in SILICON. Two reasons the tool f is EXPECTED to sit AT or
ABOVE the LND silicon numbers, not on top of them:
  (1) tissue is more neutron-sensitive than silicon (the LND team's own pre-flight
      model puts the albedo contribution at ~20% in Si but ~25% in tissue);
  (2) "neutron lineage" is broader than "neutral-particle dose" -- it counts the
      recoil protons and every charged descendant of an albedo neutron, which a
      neutral-only tally excludes.
So the primary comparand is the ~0.25 TISSUE model value, and a thin-wall f landing
in ~0.25-0.40 is CONSISTENT, not a miss. (LND's raw measured neutral share also
folds in lander/LND-structure neutron multiplication the tool does not model -- a
third reason to compare to the tissue model, not the raw 0.23.)

Usage (run from repo root ~/topas, on the PC with TOPAS_G4_DATA_DIR set):
    python3 -m paper.neutron_lnd_validate
    python3 -m paper.neutron_lnd_validate --walls 0.5,2,5 --radius 300
    python3 -m paper.neutron_lnd_validate --walls 0.5 --max-batches 12   # one LND point

Default sweep: 0.5 / 2.0 / 5.0 cm Al surface dome (~1.35 / 5.4 / 13.5 g/cm2).
The thinnest point is the LND comparand; the trio tests the monotone rise.
"""
import argparse
import sys

from lunarsim.spec import HabitatSpec, WallLayer, MATERIALS
from lunarsim import jobs
from lunarsim import dosimetry as D

# ---------------------------------------------------------------------------
# POST-HOC reference values (target-blind: references, never fit targets).
# ---------------------------------------------------------------------------

# Measured neutral-particle share of the total absorbed dose, Chang'E-4 LND,
# SILICON detector, lunar surface. Zhang et al., Sci. Adv. 6, eaaz1334 (2020):
# 3.1 +- 0.5 uGy/h neutral / 13.2 +- 1 uGy/h total = 0.23. The 2-yr dataset
# (Chin. J. Space Sci. 2021) gives 2.67/12.66 = 0.21. Raw (pre-background)
# 4.7/13.7 ~ 0.34 is an upper bracket.
LND_NEUTRAL_SI_MEASURED = 0.23
LND_NEUTRAL_SI_2YR = 0.21

# Pre-flight model estimate of the TOTAL albedo contribution to the dose rate,
# Wimmer-Schweingruber et al., Space Sci. Rev. 216:104 (2020): ~20% in silicon,
# ~25% in tissue. The TISSUE value is the primary comparand for the tool's
# tissue-lining f (same material, model-to-model), and it already shows the
# tissue > silicon direction the tool must also show.
LND_ALBEDO_SI_MODEL = 0.20
LND_ALBEDO_TISSUE_MODEL = 0.25          # <-- primary comparand

# Context only (not a pass/fail gate): albedo protons are ~40% of the ~70 MeV
# surface proton population, Xu et al. (2022) -- the charged side of the same
# regolith-albedo field the neutron fraction samples.
XU_ALBEDO_PROTON_FRAC = 0.40

# Upper cap: the tool's own earlier PROTON-ONLY thick-shield dome measured
# f ~ 0.69 (memory: neutron-fraction-diagnostic). A full-GCR fraction must sit
# BELOW it: the HZE ions carry non-neutron dose-equivalent that dilutes the
# neutron share. Not an LND number -- an internal sanity ceiling.
PROTON_ONLY_THICK_F = 0.69

AL_DENSITY = MATERIALS["aluminium"]["density"]      # 2.70 g/cm^3


def build_spec(wall_cm: float, radius_cm: float) -> HabitatSpec:
    """A SURFACE dome (not buried) over regolith with a single Al wall of the
    given thickness -- the LND-analogue flood geometry. The regolith half-space
    and FTFP_BERT_HP neutron transport come from lunar_environment.txt, which the
    flood path always includes; the engineered wall is the only overhead shield,
    so a thin wall approximates LND's near-bare surface exposure."""
    return HabitatSpec(
        name=f"lnd_neutron_{wall_cm:g}cm",
        shape="dome",
        inner_radius_cm=radius_cm,
        walls=[WallLayer(material="aluminium", thickness_cm=wall_cm)],
    )


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if x is not None else "  n/a"


def run_point(wall_cm: float, radius_cm: float, args):
    """Flood-run one surface spec and return (areal_gcm2, f, rel_err)."""
    spec = build_spec(wall_cm, radius_cm)
    areal = spec.areal_density_gcm2()
    print(f"  -> running {spec.name}  (wall {wall_cm:g} cm Al = {areal:.2f} g/cm2) ...",
          flush=True)
    cc = jobs.run_composition(
        spec, tier=jobs.QUICK_LOOK,
        target_rel_err=args.rel_err,
        min_batches=args.min_batches, max_batches=args.max_batches,
        converge_on="skin",                # f lives on the skin lining; it converges fast
    )
    if not cc.ok:
        print(f"     RUN FAILED (returncode={cc.returncode}). log tail:\n{cc.log_tail}")
        return areal, None, None
    da = D.assess_composition(cc.species_results, phi_MV=jobs.QUICK_LOOK.phi_mv, skin=True)
    if da is None:
        print("     assess_composition returned None (no usable species dose).")
        return areal, None, None
    if da.neutron_fraction is None:
        print("     f is None -- the CrewSkin neutron twin (DoseEquivalent_ICRP_Neutron)\n"
              "     did not report. Confirm the scorer is built into the flood path and\n"
              "     that skin_doseeq_neutron.csv was produced.")
        return areal, None, da.rel_err
    return areal, da.neutron_fraction, da.rel_err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walls", default="0.5,2.0,5.0",
                    help="comma-separated Al wall thicknesses in cm (default 0.5,2.0,5.0)")
    ap.add_argument("--radius", type=float, default=300.0,
                    help="dome inner radius in cm (default 300)")
    ap.add_argument("--rel-err", type=float, default=0.05,
                    help="skin dose convergence target (default 0.05)")
    ap.add_argument("--min-batches", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=8)
    args = ap.parse_args(argv)

    try:
        walls = [float(w) for w in args.walls.split(",") if w.strip()]
    except ValueError:
        ap.error(f"could not parse --walls {args.walls!r}")
    if not walls:
        ap.error("no wall thicknesses given")

    print("=" * 74)
    print("STEP 4 -- secondary-neutron dose-equivalent fraction vs Chang'E-4 LND")
    print("=" * 74)
    print("TARGET-BLIND: LND numbers are POST-HOC references, never fit targets.")
    print("f = H_neutron/H_total is a ratio of two dose-equivalents in the SAME run")
    print("(tissue CrewSkin lining, neutron-LINEAGE filtered) -> no calibration, CAL")
    print("untouched. This RUNS the flood MC -- needs TOPAS + G4Data (PC only).")
    print()

    rows = []
    for w in walls:
        areal, f, rel = run_point(w, args.radius, args)
        rows.append((w, areal, f, rel))

    # ---- results table -----------------------------------------------------
    print()
    print(f"{'wall(cm Al)':>12}  {'areal(g/cm2)':>12}  {'f(tissue,lineage)':>18}  {'rel_err':>8}")
    print(f"{'-'*12}  {'-'*12}  {'-'*18}  {'-'*8}")
    for w, areal, f, rel in rows:
        print(f"{w:>12g}  {areal:>12.2f}  {_fmt(f):>18}  {_fmt(rel):>8}")

    # ---- post-hoc references ----------------------------------------------
    print()
    print("POST-HOC references (Chang'E-4 LND, lunar surface):")
    print(f"  measured neutral share, Si       {LND_NEUTRAL_SI_MEASURED:.2f}   "
          "Zhang+ 2020, Sci.Adv. 6 eaaz1334 (3.1/13.2 uGy/h)")
    print(f"  2-yr update, Si                  {LND_NEUTRAL_SI_2YR:.2f}   "
          "Chin.J.Space Sci. 2021 (2.67/12.66)")
    print(f"  pre-flight model albedo, Si      {LND_ALBEDO_SI_MODEL:.2f}   "
          "Wimmer-Schweingruber+ 2020, SSR 216:104")
    print(f"  pre-flight model albedo, TISSUE  {LND_ALBEDO_TISSUE_MODEL:.2f}   "
          "SSR 216:104  <-- PRIMARY comparand (same material)")
    print(f"  context: albedo protons ~{XU_ALBEDO_PROTON_FRAC:.0%} of 70 MeV surface protons  (Xu+ 2022)")
    print(f"  internal cap: proton-only THICK-shield dome f ~ {PROTON_ONLY_THICK_F:.2f}  (tool, prior)")

    # ---- falsifiable physics checks ---------------------------------------
    fs = [(areal, f) for _, areal, f, _ in rows if f is not None]
    print()
    print("PHYSICS checks (falsifiable, no fit):")

    if len(fs) >= 2:
        mono = all(fs[i + 1][1] >= fs[i][1] - 1e-9 for i in range(len(fs) - 1))
        verdict = "PASS" if mono else "FAIL"
        print(f"  [1] f rises with wall areal density ................... {verdict}")
        if not mono:
            print("      (non-monotone -- check statistics before reading it as physics)")
    else:
        print("  [1] f rises with wall areal density ................... SKIP (need >=2 points)")

    if fs:
        f_thin = fs[0][1]                       # thinnest wall = LND analogue
        if f_thin >= LND_ALBEDO_TISSUE_MODEL - 0.02:
            v = "PASS"
        elif f_thin >= LND_NEUTRAL_SI_2YR - 0.02:
            v = "LOW (between Si and tissue model -- watch, do NOT re-fit)"
        else:
            v = "UNDERSHOOT (below LND Si -- investigate geometry/scorer, do NOT re-fit)"
        print(f"  [2] thin-wall f >= 0.25 tissue-albedo model .......... {v}")
        print(f"      thin-wall f = {f_thin:.3f}  vs tissue model {LND_ALBEDO_TISSUE_MODEL:.2f} "
              f"/ Si measured {LND_NEUTRAL_SI_MEASURED:.2f}")

        f_max = max(f for _, f in fs)
        v3 = "PASS" if f_max < PROTON_ONLY_THICK_F else "FAIL (exceeds proton-only cap -- unphysical)"
        print(f"  [3] f < {PROTON_ONLY_THICK_F:.2f} proton-only thick cap .............. {v3}")
    else:
        print("  [2] thin-wall f >= tissue model ...................... SKIP (no f reported)")
        print("  [3] f < proton-only cap .............................. SKIP (no f reported)")

    # ---- interpretation ----------------------------------------------------
    print()
    print("INTERPRETATION / CAVEATS:")
    print("  * Definition gap: tool f is a NEUTRON-LINEAGE dose-equivalent fraction in")
    print("    TISSUE; LND measured a NEUTRAL-PARTICLE absorbed-dose fraction in SILICON.")
    print("    Tissue is more neutron-sensitive than Si and lineage is broader than")
    print("    neutral-only, so the tool f is EXPECTED at/above the 0.25 tissue model;")
    print("    a thin-wall f in ~0.25-0.40 is CONSISTENT, not a miss.")
    print("  * LND's raw measured 0.23 folds in lander/LND-structure neutron")
    print("    multiplication the tool does not model (its shield is regolith+Al only) --")
    print("    a second reason the floor comparand is the 0.25 tissue model, not 0.23.")
    print("  * f is dimensionless and normalisation-invariant (ratio within one run):")
    print("    it carries NO calibration and CAL is untouched here.")
    print()
    print("git add owed (provide to the user, do NOT stage):")
    print("  git add paper/neutron_lnd_validate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
