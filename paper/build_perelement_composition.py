#!/usr/bin/env python3
"""Per-element GCR abundance fix harness -- resolve the thin-wall <Q> over-read.

TARGET-BLIND. Laptop-only, NO Monte Carlo. This builds the per-element replacement
for the lumped 5-species GCR_COMPOSITION (H/He/C/Si/Fe) that paper/dose_fraction_audit.py
localized as the root cause of the thin-wall water-point <Q> over-read:

    the heavy-ion ABUNDANCES rel H in GCR_COMPOSITION are ~2-3x too high, and each
    heavy BAND is collapsed onto a single near-top-Z representative (Z=3-9 -> C=6,
    Z=10-20 -> Si=14, Z=21-28 -> Fe=26), so dose ~ a_band * Z_rep^2 overweights the
    band vs the true  SUM_i a_i * Z_i^2.

The fix de-lumps every band to its real elements AND re-normalizes the heavy scale to
measured GCR composition. The audit proved the method: folded through a clean analytic
water LET, He (abundance 0.105) reproduces its measured 18.5% absorbed-dose share
exactly, so the He=0.105 anchor and the LET method are sound; only the Z>=3 inputs move.

WHAT IS CITED, WHAT IS DERIVED (the only physics input here is composition, never dose):
  * H  == 1.0                                   reference
  * He == 0.105 rel H                           KEPT -- audit-validated (He/H dose 0.43
                                                == measured 18.5/42.8); also the bridge.
  * within-heavy SHAPE (Z=3..28):  George et al. 2009, ApJ 698, 1666, Table 1 (ACE/CRIS,
        solar min, 160 MeV/nuc, Si==1000) for the measured elements; minor spallation
        secondaries from standard GCR source tables at the same normalization. Imported
        verbatim as dose_fraction_audit.ABUND.
  * heavy-to-H bridge:  a single measured He/O abundance ratio HE_OVER_O (solar-min,
        ~GeV) sets O rel H = He/(He/O); every other heavy follows the George shape.
        He-anchored so the normalization is internally consistent with the validated
        He=0.105 and uses ZERO dose information.

ACCEPTANCE GATE (post-hoc cross-check, NOT a fit): the derived composition is folded
through the SAME clean analytic water LET the audit used (dose_fraction_audit.dose_integral
-> the tool's own calibrated incident spectrum x Z_eff^2 * PSTAR LET), and the resulting
zero-shield water-point dose split is compared to the measured free-space decomposition
(Hong et al. 2019, J. Space Weather Space Clim. 9 A3: protons 42.8%, He 18.5%, Z>=3 30.1%)
and the arXiv 2511.09040 <Q> arbiter (~5.6 at zero shielding). Because the tool folds every
species through ONE shared modulated-proton LIS shape, this analytic fold is a faithful
preview of the incident/zero-shield numbers the rebuilt composition will produce in the
tool; it does NOT preview in-wall transport (that is the per-element MC the PC must run).

EMITS (ready to paste; nothing here edits the repo):
  * GCR_COMPOSITION           block for lunarsim/dosimetry.py
  * CORRECTED_COMPOSITION     block for lunarsim/kernel_gen/config.py
  * paper/gcr_perelement_composition.json   the derived table + provenance
and prints the species count / MC-regen implications and a dose-coverage-vs-truncation
table so the per-element MC cost can be scoped against lost accuracy.

Usage:
    python3 paper/build_perelement_composition.py                 # full report + emit
    python3 paper/build_perelement_composition.py --he-over-o 50  # sweep the bridge
    python3 paper/build_perelement_composition.py --emit-json     # also (re)write the JSON
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lunarsim import dosimetry                                   # noqa: E402
# reuse the audit's clean-LET physics verbatim -- same LET, same Q, same incident fold
from dose_fraction_audit import (                                # noqa: E402
    dose_integral, ABUND, A_OF, REF_BAND, ARBITER_Q0,
)

# --- kept anchors (see module docstring) -------------------------------------
HE_ABUNDANCE = 0.105           # rel H; audit-validated, unchanged
HE_OVER_O_DEFAULT = 55.0       # measured GCR He/O abundance ratio (solar min, ~GeV)
Z_OF_O = 8                     # oxygen is the heavy-to-H bridge element

# --- element names (Z -> symbol), for readable species names -----------------
SYMBOL = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni",
}
# node-grid group per Z, matching lunarsim/kernel_gen/config.py (_NODES_*). The grid
# NAMES are emitted (not the numbers) so the pasted block stays in sync with config.
#   H, He -> _NODES_LIGHT ; Z=3-9 -> _NODES_C ; Z=10-20 -> _NODES_SI ; Z=21-28 -> _NODES_FE
def grid_name(z):
    if z <= 2:
        return "_NODES_LIGHT"
    if z <= 9:
        return "_NODES_C"
    if z <= 20:
        return "_NODES_SI"
    return "_NODES_FE"

def band_label(z):
    if z <= 9:
        return "CNO + light (Z=3-9)"
    if z <= 20:
        return "Ne-Ca (Z=10-20)"
    return "Fe peak (Z=21-28)"

def particle_of(z, a):
    if z == 1:
        return "proton"
    if z == 2:
        return "alpha"
    return f"GenericIon({z},{a})"


def build_abundances(he_over_o):
    """Per-element abundance rel H for Z=1..28, from the cited anchors + George shape."""
    o_rel_h = HE_ABUNDANCE / he_over_o                 # oxygen rel H via the He/O bridge
    scale = o_rel_h / ABUND[Z_OF_O]                     # George(Si==1000) units -> rel H
    ab = {1: 1.0, 2: HE_ABUNDANCE}
    for z in range(3, 29):
        ab[z] = ABUND[z] * scale
    return ab


def fold(ab, phi_MV, nbins):
    """Per-element clean-LET fold. Returns rows [Z, sym, a, ab, a*Z^2, D, DQ, qbar]
    (D, DQ already multiplied by abundance) sorted by descending dose."""
    rows = []
    for z in sorted(ab):
        a = A_OF[z]
        D1, DQ1, _lowE, _em = dose_integral(z, a, phi_MV, nbins)
        D, DQ = ab[z] * D1, ab[z] * DQ1
        rows.append([z, SYMBOL[z], a, ab[z], ab[z] * z * z, D, DQ,
                     (DQ1 / D1) if D1 > 0 else 0.0])
    rows.sort(key=lambda r: r[5], reverse=True)
    return rows


def rollup(rows):
    Dtot = sum(r[5] for r in rows)
    DQtot = sum(r[6] for r in rows)
    def frac(pred):
        D = sum(r[5] for r in rows if pred(r))
        DQ = sum(r[6] for r in rows if pred(r))
        return 100 * D / Dtot, 100 * DQ / DQtot
    return Dtot, DQtot, DQtot / Dtot, frac


def in_ref(value01, lo_hi):
    return lo_hi[0] <= value01 <= lo_hi[1]


def emit_gcr_composition(rows):
    """GCR_COMPOSITION list block for lunarsim/dosimetry.py."""
    out = ["GCR_COMPOSITION = [",
           "    # name, TOPAS particle,        Z,  A,   abundance (rel. H), group"]
    for z, sym, a, ab, _az2, _D, _DQ, _q in sorted(rows, key=lambda r: r[0]):
        part = particle_of(z, a)
        out.append(f'    ({sym!r:6s}, {part!r:18s}, {z:>2}, {a:>3}, '
                   f'{ab:>10.3e}, {band_label(z)!r}),')
    out.append("]")
    return "\n".join(out)


def emit_corrected_composition(rows):
    """CORRECTED_COMPOSITION dict block for lunarsim/kernel_gen/config.py."""
    out = ["CORRECTED_COMPOSITION = {"]
    for z, sym, a, ab, _az2, _D, _DQ, _q in sorted(rows, key=lambda r: r[0]):
        part = particle_of(z, a)
        out.append(f'    {sym!r:6s}: {{"z": {z:>2}, "a": {a:>3}, '
                   f'"abundance": {ab:>10.3e}, "particle": {part!r},')
        out.append(f'            "group": {band_label(z)!r:28s}, '
                   f'"nodes_pernuc_mev": list({grid_name(z)})}},')
    out.append("}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--he-over-o", type=float, default=HE_OVER_O_DEFAULT,
                    help="measured GCR He/O abundance ratio (heavy-to-H bridge)")
    ap.add_argument("--lis", default=dosimetry.KERNEL_LIS_FORM,
                    choices=[dosimetry.KERNEL_LIS_FORM, dosimetry.FLOOD_LIS_FORM],
                    help="LIS form for the fold (default kernel/rigidity, matches let_harness)")
    ap.add_argument("--phi", type=float, default=400.0, help="modulation phi (MV)")
    ap.add_argument("--nbins", type=int, default=400, help="per-nucleon energy bins")
    ap.add_argument("--emit-json", action="store_true",
                    help="write paper/gcr_perelement_composition.json")
    args = ap.parse_args()

    dosimetry.set_lis_form(args.lis)
    ab = build_abundances(args.he_over_o)
    rows = fold(ab, args.phi, args.nbins)
    Dtot, DQtot, qbar, frac = rollup(rows)

    print("=" * 80)
    print("Per-element GCR abundance fix -- de-lumped + measured-composition rescale")
    print(f"  He=0.105 rel H (kept) | He/O={args.he_over_o:g} (bridge) | O/H="
          f"{HE_ABUNDANCE/args.he_over_o:.3e} | George-2009 heavy shape")
    print(f"  LIS={args.lis!r}  phi={args.phi:g} MV  nbins={args.nbins}  "
          f"LET=Z_eff^2*PSTAR [analytic, no MC]")
    print("=" * 80)

    # ---- per-element table ------------------------------------------------
    print("\n-- Derived per-element composition (sorted by absorbed-dose share) --")
    print(f"  {'Z':>3} {'sp':>3} {'A':>3} {'a/H':>11} {'a*Z^2':>9} "
          f"{'dose%':>7} {'deq%':>7} {'<Q>_s':>7}")
    for z, sym, a, abz, az2, D, DQ, qs in rows:
        print(f"  {z:>3} {sym:>3} {a:>3} {abz:>11.3e} {az2:>9.4f} "
              f"{100*D/Dtot:7.2f} {100*DQ/DQtot:7.2f} {qs:7.2f}")

    # ---- ACCEPTANCE GATE: rolled-up split vs Hong 2019 + arbiter ----------
    print("\n-- ACCEPTANCE GATE: zero-shield water-point split vs measured (Hong 2019) --")
    print(f"  {'group':>14} {'dose%':>7} {'deq%':>7}   {'ref dose%':>12} {'ref deq%':>12} {'':>4}")
    checks = [
        ("H",  lambda r: r[0] == 1),
        ("He", lambda r: r[0] == 2),
        ("Z>=3 (all HZE)", lambda r: r[0] >= 3),
    ]
    all_pass = True
    for label, pred in checks:
        dpct, qpct = frac(pred)
        rb = REF_BAND[label]
        ok = in_ref(dpct / 100.0, rb["dose"])
        all_pass = all_pass and ok
        print(f"  {label:>14} {dpct:7.2f} {qpct:7.2f}   "
              f"{100*rb['dose'][0]:5.0f}-{100*rb['dose'][1]:<5.0f} "
              f"{100*rb['deq'][0]:5.0f}-{100*rb['deq'][1]:<5.0f} "
              f"{'PASS' if ok else 'OUT':>4}")
    print(f"  point <Q> (de-lumped, per-element) = {qbar:.2f}   "
          f"[tool lumped MC = 11.31 ; arbiter = {ARBITER_Q0:.1f}]")
    print(f"  GATE: absorbed-dose split {'WITHIN' if all_pass else 'NOT within'} "
          f"the measured Hong-2019 bands.")

    # ---- truncation / MC-cost scoping ------------------------------------
    print("\n-- Dose coverage vs species count (scope the per-element MC) --")
    print("   keep the top-N dose species as individual MC runs; the rest are trace")
    print(f"  {'N':>3} {'species':>40} {'cum dose%':>10} {'cum deq%':>9}")
    cumD = cumDQ = 0.0
    for n, r in enumerate(rows, 1):
        cumD += 100 * r[5] / Dtot
        cumDQ += 100 * r[6] / DQtot
        if n <= 6 or n in (8, 10, 13, len(rows)) or r[1] in ("Fe", "Ni"):
            names = ", ".join(rr[1] for rr in rows[:n])
            names = names if len(names) <= 40 else names[:37] + "..."
            print(f"  {n:>3} {names:>40} {cumD:10.2f} {cumDQ:9.2f}")

    # ---- emitted blocks ---------------------------------------------------
    print("\n" + "=" * 80)
    print("PASTE 1/2 -> lunarsim/dosimetry.py  (replace GCR_COMPOSITION)")
    print("=" * 80)
    print(emit_gcr_composition(rows))
    print("\n" + "=" * 80)
    print("PASTE 2/2 -> lunarsim/kernel_gen/config.py  (replace CORRECTED_COMPOSITION)")
    print("=" * 80)
    print(emit_corrected_composition(rows))

    if args.emit_json:
        out = {
            "meta": {
                "built_by": "paper/build_perelement_composition.py",
                "he_abundance_rel_h": HE_ABUNDANCE,
                "he_over_o_bridge": args.he_over_o,
                "o_rel_h": HE_ABUNDANCE / args.he_over_o,
                "heavy_shape_source": "George et al. 2009 ApJ 698 1666 Table 1 "
                                      "(ACE/CRIS, 160 MeV/nuc, Si==1000) + GCR secondary tables",
                "acceptance_ref": "Hong et al. 2019 J.Space Weather Space Clim 9 A3 "
                                  "(p 42.8%, He 18.5%, Z>=3 30.1% absorbed)",
                "lis_form": args.lis, "phi_MV": args.phi,
                "note": "abundances are target-blind (composition only); the Hong-2019 "
                        "split is a post-hoc fold check, never a fit. Per-element LIS "
                        "(energy-dependent abundances) remains the deferred refinement.",
            },
            "species": [
                {"name": sym, "z": z, "a": a, "particle": particle_of(z, a),
                 "abundance_rel_h": abz, "group": band_label(z),
                 "dose_frac": D / Dtot, "deq_frac": DQ / DQtot, "qbar": qs}
                for z, sym, a, abz, az2, D, DQ, qs in sorted(rows, key=lambda r: r[0])
            ],
        }
        p = Path(__file__).resolve().parent / "gcr_perelement_composition.json"
        p.write_text(json.dumps(out, indent=2))
        print(f"\n[wrote {p}]")

    # ---- PC regen checklist ----------------------------------------------
    print("\n" + "-" * 80)
    print("WHAT THE PC MUST REGENERATE (abundance is NOT transport -- see config.py notes):")
    print(f"  * the thin-wall kernel adds {len(rows)-5:+d} species vs the current 5 "
          f"(now {len(rows)} total). Each species' response R=D/phi is one MC run per")
    print("    material (Al/EVA/poly) x anchor depth. Abundance itself needs no MC -- it")
    print("    only reweights an existing per-species response -- so species already in")
    print("    the kernel (H/He + the CNO/Ne-Ca/Fe representatives) can be RE-LABELLED,")
    print("    but each genuinely new ion (N,O,Ne,Mg,S,Ar,Ca,Ni,...) is new transport.")
    print("  * UNCHANGED: the shared proton-LIS shape, _calibration_factor, Q-model,")
    print("    geometry, the flood engine's thick-shield headline E (135-200 mSv/yr).")
    print("  * tests referencing GCR_COMPOSITION by index (test_pipeline.py GCR_COMPOSITION[4]")
    print("    = Fe) must move to lookup-by-name once the list grows.")
    print("  * cheapest first step: keep the 5 species but swap in the de-lumped")
    print("    EFFECTIVE abundances (conserve SUM a_i*Z_i^2 per band) -- NO new MC, fixes")
    print("    the dominant abundance error; full per-element MC then firms the residual.")


if __name__ == "__main__":
    main()
