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
    python3 -m paper.neutron_lnd_validate --walls 0.5 --radii 30,100,300  # dome-size curve
    python3 -m paper.neutron_lnd_validate --walls 0.5 --radius 30         # tight LND analogue

Default sweep: 0.5 / 2.0 / 5.0 cm Al surface dome (~1.35 / 5.4 / 13.5 g/cm2).
The thinnest wall at the smallest radius is the LND comparand; the wall trio
tests the monotone rise, and --radii sweeps the regolith solid angle the
CrewSkin lining sees (a geometry control, not a fit -- see build_spec).
"""
import argparse
import sys

from lunarsim.spec import HabitatSpec, WallLayer, MATERIALS
from lunarsim import jobs
from lunarsim.bridge import QUICK_LOOK, FULL_RUN   # FULL_RUN not re-exported via jobs
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

# Above this skin-dose rel_err, f is too noisy to read as a physics result: a
# small dome's thin lining shell gets few hits, so variance stays high unless
# given many more batches. Points above it are flagged NOISY in the table and
# excluded from the pass/fail verdicts (reported, not silently trusted).
REL_ERR_NOISY = 0.15


def _fit_phantom(radius_cm: float, crew_h: float | None,
                 phantom_r: float | None) -> tuple[float, float]:
    """Return (crew_height_cm, phantom_radius_cm) that satisfy spec.validate()
    for a dome of the given inner radius. The phantom is a legality formality
    here -- f is scored on the CrewSkin lining, not the phantom -- so for a tight
    LND-analogue dome we just shrink it to fit. A standing-adult phantom
    (100/20 cm) needs a >=120 cm dome; below that we scale both down (crew at
    0.5R, phantom at 0.2R => sum 0.7R < R and crew >= phantom, always legal).
    Explicit --crew-height / --phantom-radius override the auto rule."""
    if phantom_r is None:
        phantom_r = min(20.0, 0.2 * radius_cm)
    if crew_h is None:
        crew_h = min(100.0, 0.5 * radius_cm)
    return crew_h, phantom_r


def build_spec(wall_cm: float, radius_cm: float,
               crew_h: float | None = None,
               phantom_r: float | None = None) -> HabitatSpec:
    """A SURFACE dome (not buried) over regolith with a single Al wall of the
    given thickness -- the LND-analogue flood geometry. The regolith half-space
    and FTFP_BERT_HP neutron transport come from lunar_environment.txt, which the
    flood path always includes; the engineered wall is the only overhead shield,
    so a thin wall approximates LND's near-bare surface exposure.

    A SMALL inner radius pulls the CrewSkin lining down toward the regolith
    interface -- the geometry LND actually sampled (detector ~at the surface,
    ~2pi regolith below). A large (3 m) dome's lining mostly faces open sky, so
    it dilutes the regolith-albedo neutron fraction. Sweeping radius therefore
    sweeps the regolith solid angle the lining sees -- not a fit knob, a geometry
    control."""
    crew_h, phantom_r = _fit_phantom(radius_cm, crew_h, phantom_r)
    return HabitatSpec(
        name=f"lnd_neutron_{wall_cm:g}cm_r{radius_cm:g}",
        shape="dome",
        inner_radius_cm=radius_cm,
        walls=[WallLayer(material="aluminium", thickness_cm=wall_cm)],
        crew_height_cm=crew_h,
        phantom_radius_cm=phantom_r,
    )


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if x is not None else "  n/a"


def run_point(wall_cm: float, radius_cm: float, args):
    """Flood-run one surface spec and return (areal_gcm2, f, rel_err)."""
    spec = build_spec(wall_cm, radius_cm,
                      crew_h=args.crew_height, phantom_r=args.phantom_radius)
    areal = spec.areal_density_gcm2()
    print(f"  -> running {spec.name}  (wall {wall_cm:g} cm Al = {areal:.2f} g/cm2, "
          f"R={radius_cm:g} cm) ...", flush=True)
    tier = FULL_RUN if args.tier == "full" else QUICK_LOOK
    cc = jobs.run_composition(
        spec, tier=tier,
        target_rel_err=args.rel_err,
        min_batches=args.min_batches, max_batches=args.max_batches,
        converge_on="skin",                # f lives on the skin lining; it converges fast
    )
    if not cc.ok:
        print(f"     RUN FAILED (returncode={cc.returncode}). log tail:\n{cc.log_tail}")
        return areal, None, None
    da = D.assess_composition(cc.species_results, phi_MV=tier.phi_mv, skin=True)
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
                    help="dome inner radius in cm (default 300). Ignored if --radii given.")
    ap.add_argument("--radii", default=None,
                    help="comma-separated dome inner radii in cm to sweep (e.g. 30,100,300). "
                         "Pulls the CrewSkin lining toward (small) or away from (large) the "
                         "regolith interface -- a geometry control, not a fit knob.")
    ap.add_argument("--crew-height", type=float, default=None,
                    help="phantom-centre height above floor in cm (default auto-fit to radius; "
                         "phantom does not enter f, only legality)")
    ap.add_argument("--phantom-radius", type=float, default=None,
                    help="crew phantom radius in cm (default auto-fit to radius)")
    ap.add_argument("--tier", choices=("quick", "full"), default="quick",
                    help="MC tier (default quick ~1e3 primaries/batch; 'full' ~1e4/batch "
                         "-- use for small domes whose thin lining shell converges slowly)")
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

    if args.radii:
        try:
            radii = [float(r) for r in args.radii.split(",") if r.strip()]
        except ValueError:
            ap.error(f"could not parse --radii {args.radii!r}")
        if not radii:
            ap.error("no radii given")
    else:
        radii = [args.radius]

    print("=" * 74)
    print("STEP 4 -- secondary-neutron dose-equivalent fraction vs Chang'E-4 LND")
    print("=" * 74)
    print("TARGET-BLIND: LND numbers are POST-HOC references, never fit targets.")
    print("f = H_neutron/H_total is a ratio of two dose-equivalents in the SAME run")
    print("(tissue CrewSkin lining, neutron-LINEAGE filtered) -> no calibration, CAL")
    print("untouched. This RUNS the flood MC -- needs TOPAS + G4Data (PC only).")
    print()

    rows = []
    for r in radii:
        for w in walls:
            areal, f, rel = run_point(w, r, args)
            rows.append((w, r, areal, f, rel))

    # ---- results table -----------------------------------------------------
    print()
    print(f"{'wall(cm Al)':>12}  {'R(cm)':>7}  {'areal(g/cm2)':>12}  "
          f"{'f(tissue,lineage)':>18}  {'rel_err':>8}")
    print(f"{'-'*12}  {'-'*7}  {'-'*12}  {'-'*18}  {'-'*8}")
    for w, r, areal, f, rel in rows:
        flag = "  NOISY" if (rel is not None and rel > REL_ERR_NOISY) else ""
        print(f"{w:>12g}  {r:>7g}  {areal:>12.2f}  {_fmt(f):>18}  {_fmt(rel):>8}{flag}")

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
    # Keep only points whose f actually converged: a NOISY f is reported in the
    # table but never drives a verdict (it would fire false pass/fail).
    good = [(w, r, areal, f) for w, r, areal, f, rel in rows
            if f is not None and (rel is None or rel <= REL_ERR_NOISY)]
    dropped = [(w, r, rel) for w, r, areal, f, rel in rows
               if f is not None and rel is not None and rel > REL_ERR_NOISY]
    print()
    print("PHYSICS checks (falsifiable, no fit; NOISY points excluded):")
    if dropped:
        for w, r, rel in dropped:
            print(f"  (excluded: wall {w:g} cm, R={r:g} cm -- rel_err {rel:.3f} > "
                  f"{REL_ERR_NOISY:.2f}; f not converged, re-run with more batches)")

    # [1] f must rise with wall areal density -- checked WITHIN each radius group
    # (mixing radii would confound the wall trend with the geometry trend).
    wall_groups = {}
    for w, r, areal, f in good:
        wall_groups.setdefault(r, []).append((areal, f))
    checkable = [g for g in wall_groups.values() if len(g) >= 2]
    if checkable:
        mono_all = True
        for g in checkable:
            g.sort()
            if not all(g[i + 1][1] >= g[i][1] - 1e-9 for i in range(len(g) - 1)):
                mono_all = False
        verdict = "PASS" if mono_all else "FAIL"
        print(f"  [1] f rises with wall areal density (per radius) ..... {verdict}")
        if not mono_all:
            print("      (non-monotone -- check statistics before reading it as physics)")
    else:
        print("  [1] f rises with wall areal density .................. SKIP (need >=2 walls at a radius)")

    # [2] LND analogue = thinnest wall at the SMALLEST radius (lining closest to
    # the regolith interface, the geometry LND actually sampled).
    fs = sorted(good, key=lambda t: (t[1], t[2]))   # by (radius, areal)
    # If a SMALLER radius than any converged point was attempted but dropped for
    # noise, the true LND analogue is still pending -- do not read the undershoot
    # off a larger (sky-dominated) dome.
    smallest_dropped_r = min((r for _, r, _ in dropped), default=None)
    if fs:
        f_thin = fs[0][3]
        r_thin = fs[0][1]
        analogue_pending = smallest_dropped_r is not None and smallest_dropped_r < r_thin
        if analogue_pending:
            print(f"  [2] thin-wall f >= 0.25 tissue-albedo model .......... "
                  "INCONCLUSIVE (analogue R unconverged)")
            print(f"      smallest converged point is R={r_thin:g} cm (f={f_thin:.3f}); the tighter")
            print(f"      R={smallest_dropped_r:g} cm analogue was NOISY -- re-run it with more batches.")
        else:
            if f_thin >= LND_ALBEDO_TISSUE_MODEL - 0.02:
                v = "PASS"
            elif f_thin >= LND_NEUTRAL_SI_2YR - 0.02:
                v = "LOW (between Si and tissue model -- watch, do NOT re-fit)"
            else:
                v = "UNDERSHOOT (below LND Si -- investigate geometry/scorer, do NOT re-fit)"
            print(f"  [2] thin-wall f >= 0.25 tissue-albedo model .......... {v}")
            print(f"      LND analogue = smallest R={r_thin:g} cm, thinnest wall: f = {f_thin:.3f}")
            print(f"      vs tissue model {LND_ALBEDO_TISSUE_MODEL:.2f} "
                  f"/ Si measured {LND_NEUTRAL_SI_MEASURED:.2f}")

        f_max = max(f for _, _, _, f in fs)
        v3 = "PASS" if f_max < PROTON_ONLY_THICK_F else "FAIL (exceeds proton-only cap -- unphysical)"
        print(f"  [3] f < {PROTON_ONLY_THICK_F:.2f} proton-only thick cap .............. {v3}")
    else:
        print("  [2] thin-wall f >= tissue model ...................... "
              + ("INCONCLUSIVE (only NOISY points -- re-run with more batches)"
                 if dropped else "SKIP (no f reported)"))
        print("  [3] f < proton-only cap .............................. SKIP (no usable f)")

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
    print("  * GEOMETRY control (not a fit knob): dome inner radius sets how much")
    print("    regolith the CrewSkin lining sees. A 3 m dome's lining mostly faces")
    print("    open sky, so direct GCR dominates and f is DILUTED; a small radius")
    print("    pulls the lining to the regolith interface -- LND's actual geometry")
    print("    (detector ~at the surface, ~2pi regolith below). Sweep --radii to")
    print("    see f climb as the regolith solid angle grows. Compare the LND")
    print("    analogue (smallest R) to the tissue model, NOT the 3 m dome point.")
    print("  * f is dimensionless and normalisation-invariant (ratio within one run):")
    print("    it carries NO calibration and CAL is untouched here.")
    print()
    print("git add owed (provide to the user, do NOT stage):")
    print("  git add paper/neutron_lnd_validate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
