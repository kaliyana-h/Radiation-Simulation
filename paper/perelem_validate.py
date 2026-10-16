#!/usr/bin/env python3
"""Step 3 -- target-blind validation of the per-element (10-ion) thin-wall kernel.

This is the acceptance cross-check that must pass BEFORE the perelement kernel is
copied into lunarsim/data/ and shipped. It is TARGET-BLIND: every external number
below is a POST-HOC reference, never a fit target. CAL is NOT re-fit here; nothing
in this script touches the calibration. It only folds the already-run kernel
(pure arithmetic -- no TOPAS / G4Data) and lays the result beside the published
benchmarks so a human can judge the agreement.

What it checks, per anchor wall [0, 2.025, 10, 50 g/cm2 Al-equiv]:

  (A) ORGAN fold of the shipped/candidate perelement kernel (via perelem_gate.
      fold_anchors): organ-weighted <Q>(depth), organ effective dose (mSv/yr),
      organ effective absorbed dose (uGy/day). These are the tool's deliverables.

  (B) [--let PATH] WATER-POINT <Q>(depth) read straight from a LET-harness collect
      JSON (runs/let_perelem/let_qbar_depth.json). This is the quantity directly
      comparable to the CRaTER / arXiv:2511.09040 measured+INCL arbiter. The organ
      fold in (A) is ~0.61x the water point by geometry -- do NOT compare the organ
      <Q> to the arbiter; compare the water-point <Q> here.

  (C) Absolute-dose anchors printed inline for the human:
        - Chang'E-4 LND: 13.2 uGy/h Si ~ 0.41 mGy/day tissue (converged 22.4 g/cm2
          anchor, <Q> 4.04, SILICON-referenced dose-equivalent).
        - OLTARIS: 0.296 mGy/day TOTAL at 54 g/cm2 Al (lunar surface). The thin-wall
          ORGAN absorbed dose is ~0.5-0.56x the OLTARIS BARE-POINT by self-shielding,
          so a thin-wall anchor that lands ~half the bare point is EXPECTED, not a
          miss. The point-like match to OLTARIS lives on the FLOOD side (>19 g/cm2),
          checked separately by anchor_crosscheck (command printed at the end).

CAVEATS enforced in the output:
  * water-point vs organ geometry (never call the organ-vs-arbiter gap a defect);
  * thin-wall organ absorbed ~0.5-0.56x OLTARIS bare-point (self-shielding);
  * the ~19 g/cm2 crossover: below it the kernel is authoritative, at/above it the
    flood engine is, and the "absorbed dose" label changes flavour across it.

Usage (run from repo root ~/topas):
    python3 -m paper.perelem_validate
    python3 -m paper.perelem_validate runs/perelem/gcr_thinwall_kernel_perelem.json \
        --let runs/let_perelem/let_qbar_depth.json

With no kernel argument it uses lunarsim/data/gcr_thinwall_kernel_perelem.json.
"""
import json
import sys
from pathlib import Path

from lunarsim import dosimetry as D
from paper.perelem_gate import fold_anchors

SPD = D.SECONDS_PER_DAY          # 86400.0
DPY = D.DAYS_PER_YEAR            # 365.0

# ---------------------------------------------------------------------------
# POST-HOC reference anchors (target-blind: references, never fit targets).
# ---------------------------------------------------------------------------

# CRaTER / arXiv:2511.09040 measured + Geant4-INCLXX WATER-POINT <Q> vs Al depth.
# Falls 5.6@0 -> ~3.1@50. Compare ONLY against the --let water-point <Q>, never the
# organ fold. 2026-10-15 watch: the per-element MC now reads a mild deep UNDERSHOOT
# (2.55 vs 3.5 @20) after the lumping fix -- flag it, do not "correct" it.
ARBITER_QBAR = [
    (0.0, 5.6),
    (2.0, 5.8),
    (10.0, 4.3),
    (20.0, 3.5),
    (50.0, 3.1),
]

# OLTARIS lunar-surface TOTAL absorbed dose (compare to the FLOOD top-line, not the
# thin-wall organ fold). 54 g/cm2 Al. Thin-wall organ absorbed ~0.5-0.56x this.
OLTARIS_TOTAL_MGY_DAY = 0.296
OLTARIS_WALL_GCM2 = 54.0
THINWALL_SELF_SHIELD_LO = 0.50   # organ absorbed / bare-point (expected band)
THINWALL_SELF_SHIELD_HI = 0.56

# Chang'E-4 LND converged anchor (SILICON-referenced dose-equivalent).
CHANGE4_UGY_H_SI = 13.2
CHANGE4_MGY_DAY_TISSUE = 0.41
CHANGE4_WALL_GCM2 = 22.4
CHANGE4_QBAR = 4.04


def _interp(curve, x):
    """Piecewise-linear interpolation (clamped at the ends) of a sorted (x,y) list."""
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return curve[-1][1]


def _msv_yr(E_sv_s):
    return E_sv_s * SPD * DPY * 1.0e3


def _ugy_day(D_gy_s):
    return D_gy_s * SPD * 1.0e6


def report_organ(folded):
    """(A) Organ-weighted fold vs references."""
    print("(A) ORGAN fold of the perelement kernel (tool deliverable)")
    print(f"    {'wall':>6} | {'<Q> organ':>10} | {'E mSv/yr':>10} | "
          f"{'D_abs uGy/day':>14}")
    print(f"    {'g/cm2':>6} | {'':>10} | {'':>10} | {'(organ eff.)':>14}")
    print("    " + "-" * 50)
    for w, D_eff, E, Q, _ps in folded:
        print(f"    {w:6.2f} | {Q:10.3f} | {_msv_yr(E):10.2f} | "
              f"{_ugy_day(D_eff):14.4f}")
    print()


def report_waterpoint(let_path):
    """(B) Water-point <Q>(depth) from the LET collect JSON vs the CRaTER arbiter."""
    if not let_path:
        print("(B) WATER-POINT <Q> vs CRaTER arbiter -- SKIPPED (no --let given)")
        print("    Pass --let runs/let_perelem/let_qbar_depth.json to enable it.\n")
        return
    p = Path(let_path)
    if not p.exists():
        print(f"(B) WATER-POINT <Q> vs CRaTER arbiter -- SKIPPED (no {p})")
        print("    Run the LET harness on the PC and point --let at its "
              "let_qbar_depth.json.\n")
        return
    with open(p) as fh:
        let = json.load(fh)
    # Schema (let_harness.collect): points live under materials[<name>]["points"],
    # NOT at the top level. Prefer aluminium (the Al-equiv arbiter axis); else the
    # first material present.
    mats = let.get("materials", {})
    mat = "aluminium" if "aluminium" in mats else (next(iter(mats), None))
    pts = sorted(mats.get(mat, {}).get("points", []) if mat else [],
                 key=lambda q: q["wall_gcm2"])
    print(f"(B) WATER-POINT <Q>(depth) vs CRaTER / arXiv:2511.09040 arbiter"
          f"   [material={mat}]")
    print(f"    {'wall':>6} | {'<Q> water':>10} | {'arbiter':>8} | "
          f"{'ratio':>6} | note")
    print(f"    {'g/cm2':>6} | {'(kernel)':>10} | {'(meas+INCL)':>8} | "
          f"{'':>6} |")
    print("    " + "-" * 60)
    for pt in pts:
        w = pt["wall_gcm2"]
        q = pt.get("qbar_waterpoint_native", pt.get("qbar_waterpoint_ntuple"))
        if q is None:
            continue
        arb = _interp(ARBITER_QBAR, w)
        r = q / arb if arb else float("nan")
        note = ""
        if r < 0.9:
            note = "UNDERSHOOT (watch -- post-lumping-fix; do NOT re-fit)"
        elif r > 1.15:
            note = "over-read (the original thin-Q defect -- should be gone)"
        print(f"    {w:6.2f} | {q:10.3f} | {arb:8.2f} | {r:6.3f} | {note}")
    print("\n    NOTE: this is the WATER-POINT <Q>; the organ fold in (A) is "
          "~0.61x it by\n    geometry. Compare the arbiter ONLY to this table.\n")


def report_absolute(folded):
    """(C) Absolute-dose references and the thin-wall-vs-bare-point expectation."""
    print("(C) ABSOLUTE-DOSE reference anchors (post-hoc; CAL not re-fit)")
    print(f"    Chang'E-4 LND  : {CHANGE4_UGY_H_SI} uGy/h Si "
          f"~ {CHANGE4_MGY_DAY_TISSUE} mGy/day tissue "
          f"(conv. {CHANGE4_WALL_GCM2} g/cm2, <Q> {CHANGE4_QBAR}, Si-ref.)")
    print(f"    OLTARIS total  : {OLTARIS_TOTAL_MGY_DAY} mGy/day "
          f"@ {OLTARIS_WALL_GCM2} g/cm2 Al (lunar surface, TOTAL not GCR-only)")
    bare = OLTARIS_TOTAL_MGY_DAY
    lo = bare * THINWALL_SELF_SHIELD_LO * 1.0e3   # -> uGy/day
    hi = bare * THINWALL_SELF_SHIELD_HI * 1.0e3
    print(f"    => thin-wall ORGAN absorbed is EXPECTED ~{THINWALL_SELF_SHIELD_LO:.2f}"
          f"-{THINWALL_SELF_SHIELD_HI:.2f}x the bare point by self-shielding")
    print(f"       i.e. ~{lo:.0f}-{hi:.0f} uGy/day organ-eff. at depth is "
          f"consistent, NOT a miss.")
    # closest folded anchor to OLTARIS depth, for a sanity sightline only
    if folded:
        near = min(folded, key=lambda t: abs(t[0] - OLTARIS_WALL_GCM2))
        print(f"       (nearest folded anchor: wall={near[0]:g} g/cm2, "
              f"organ D_abs={_ugy_day(near[1]):.1f} uGy/day)")
    print()


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    let_path = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--let" and i + 1 < len(sys.argv[1:]):
            let_path = sys.argv[1:][i + 1]
    # strip the --let VALUE from positional args
    if let_path and let_path in args:
        args.remove(let_path)

    kernel_path = Path(args[0]) if args else Path(
        "lunarsim/data/gcr_thinwall_kernel_perelem.json")
    if not kernel_path.exists():
        print(f"per-element kernel not found: {kernel_path}", file=sys.stderr)
        print("  (run Step 2 on the PC, or pass the runs/ path explicitly)",
              file=sys.stderr)
        return 2

    with open(kernel_path) as fh:
        pe = json.load(fh)
    folded = fold_anchors(pe, "perelement")
    D.set_perelement_model("none")        # leave the process clean

    nspec = len(pe["points"][0]["species"]) if pe.get("points") else 0
    print("=" * 70)
    print("STEP 3  per-element thin-wall kernel -- TARGET-BLIND validation")
    print("=" * 70)
    print(f"kernel : {kernel_path}  ({len(folded)} anchors, {nspec} species)")
    print(f"LIS    : {D.KERNEL_LIS_FORM} (kernel/thin engine)   phi=400 MV")
    print("CAL    : NOT re-fit -- references below are post-hoc only\n")

    report_organ(folded)
    report_waterpoint(let_path)
    report_absolute(folded)

    print("-" * 70)
    print("DEEP STABILITY (>19 g/cm2 flood side -- point-like OLTARIS match):")
    print("  laptop: preserved by construction -- Option A pins the flood engine to")
    print("          the OLTARIS-validated 5-ion baseline under 'perelement'.")
    print("  PC MC : python3 -m lunarsim.anchor_crosscheck runs/xcheck54")
    print("          (54 g/cm2 Al dome, expect skin ratio ~1.0 vs 0.310 mGy/day)")
    print("-" * 70)
    print("CAVEATS: (1) arbiter is WATER-POINT <Q>; organ fold ~0.61x it.")
    print("         (2) thin-wall organ absorbed ~0.50-0.56x OLTARIS bare-point.")
    print("         (3) ~19 g/cm2 crossover flips the 'absorbed dose' flavour;")
    print("             kernel authoritative below, flood above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
