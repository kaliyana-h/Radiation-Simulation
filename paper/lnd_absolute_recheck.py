#!/usr/bin/env python3
"""Absolute-dose recheck against Chang'E-4 LND after the Matthiae absolute
normalisation (memory: flood-heavy-abundance-overweight).

The 2026-10-27 change dropped the legacy 2.0 /cm2/s proton pin for matthia2013 /
perelement, which moves EVERY absolute dose (flood x~1.245 at W=0 relative to the
pin, kernel x1.215; the kernel organ fold confirms the latter exactly with <Q>
unchanged -- run `python3 -m paper.perelem_validate` on the laptop). The 54 g/cm2
anchor_crosscheck and phi_sweep confirmed the deep side (0.260 mGy/d, 218.4
mSv/yr, Q-bar 2.295). This harness re-checks the SURFACE side against the only
in-situ absolute measurement, Chang'E-4 LND.

It RUNS the flood MC (TOPAS + G4Data, PC only), full GCR composition, skin lining,
at two geometries:

  lnd   : 0.5 cm Al surface dome, R = 30 cm -- the LND analogue already used and
          validated for the neutron fraction (neutron_lnd_validate: f = 0.279 at
          this point vs 0.25 tissue model). Lining hugs the regolith, near-bare sky.
  jul22 : 3 m dome, 5 cm Al + 5 cm regolith = 22.4 g/cm2 -- the 2026-07-20
          "converged Chang'E-4 anchor" geometry, for continuity with that number.

TARGET-BLIND. LND values are POST-HOC references, never fit targets; CAL is not
touched. Nothing here feeds back into the physics.

BASIS (read before judging). Zhang et al. 2020 (Sci. Adv. 6 eaaz1334) report
13.2 +- 1 uGy/h TOTAL absorbed in SILICON (charged 10.2, neutral 3.1), and convert
Si -> water with a constant factor 1.30 (CRaTER uses 1.33). Their <Q> = 4.3 and
the headline 1369 uSv/day (= 57.1 uSv/h) are CHARGED-ONLY and WATER-referenced
(10.2 x 1.30 x 4.3). The tool scores absorbed dose in a TISSUE lining, so LND
must be moved to water -- and the factor is spectrum-dependent. Since 2026-10-30
the harness uses the tool's OWN factor, SI_TO_WATER = 1.227 at phi 400: Bethe
mass stopping power folded over the tool's input spectrum (paper.si_water_factor;
target-blind, no dose value enters). Matthiae & Berger 2024 (Space Weather 22
e2024SW004095) model-compute 1.21 for the same field, so Zhang's 1.30 overstates
the water comparand by ~6% and is printed for reference only. The factor is a
charged-particle one; the neutral share converts differently (hydrogen makes
tissue more neutron-sensitive). (The 2026-07-20 anchor divided by 1.5, which
overstated the agreement.)

SILICON DETECTOR (--si-detector UM, 2026-10-30). Removes the conversion entirely:
a thin G4_Si shell (300 um ~ LND / M&B 2024) on the inner wall face, scored with
DoseToMedium + EntryChannelDose "Charged", so the tool's charged dose-to-SILICON
compares with LND's 10.2 directly. The lining moves inward by the detector
thickness (negligible). Harness-only, via LUNARSIM_SI_DETECTOR_UM (dome only).
On by default (300 um) whenever [bare3m] is requested; --si-detector 0 turns it off.
Its z uses the detector's OWN batch-to-batch error (si_charged_rel_err), not the
skin's: the 300 um shell catches far fewer hits than the 2 cm lining.

SILICON NEUTRAL CHANNEL (2026-11-02, reported, NOT a gate). The detector also
carries EntryChannelDose "Neutral" (LND's C1 anticoincidence channel: Si recoils +
low-energy gammas, nothing charged entering) and its "Neutron" / "Gamma" split, so
the tool's neutral dose-to-silicon is scored directly (no total-minus-charged
subtraction). It is printed against LND's 3.1 +- 0.5 AND against Matthiae & Berger
2024's same-environment Geant4 value, 0.073 uGy/h Si (their Table 3): the published
model is ~42x below LND too. LND's neutral is after a 1.7 +- 0.4 RTG/RHU background
measured on the ground (Aug 2018, no neutron multiplication; Zhang fn 20), and
Zhang fn 22 attributes its height to multiplication and lander/LND structure
secondaries -- platform terms no lunar-environment model carries.

SILICON CHARGED <Q> AND LET SPECTRUM (2026-11-02, post-hoc, NOT a gate). The
detector also carries a charged EntryChannelDose with WeightByQ (ICRP-60 Q of each
step's unrestricted electronic LET IN WATER -- G4EmCalculator, G4_WATER) and the
charged dose split into half-decade water-LET bins (geometry.SI_LET_EDGES_KEV_UM,
0.1..1000 keV/um, open-ended first/last). So <Q>_charged = H_Si,ch / D_Si,ch compares
with LND's 4.3 +- 0.7, and the factor-free dose equivalent D_Si,ch x <Q> with
LND's 57.1 / 1.30 = 43.9 uSv/h (Zhang's H is 10.2 Si x 1.30 x 4.3; dividing their
own constant back out leaves Si-basis x <Q>, error +-18.6%). The bins should sum to
the charged Si dose (closure printed). Method differences, stated not corrected:
LND infers LET from a coincidence-telescope dE/dx in Si scaled to water by the
constant 1.3, the tool takes per-step water LET of the stepping particle; the
tool's <Q> weights are Si dose, LND's are water dose (the Si->water stopping
ratio is ~flat in LET, so this is a small bias); delta-ray energy deposited by a
secondary electron is binned at the ELECTRON's LET.

Other known differences: LND sits inside the lander payload compartment (extra
local mass + structure neutrons the tool does not model) and measured
Jan-Feb 2019 near phi ~400 MV; the tool's ceiling is 20 GeV/n (~-7.6% surface).

GATE (2026-10-29; Si primary since 2026-11-02). When the build has the
EntryChannelDose scorer, the verdict is read on the CHARGED channel, in sigma units
combining LND's +-10.8% with the run's own rel_err. With the Si detector (the
default for [bare3m]) the verdict is tool charged dose-to-SILICON vs LND 10.2 +- 1.1
Si directly -- no conversion factor -- and the water-basis ratio (tool charged
tissue vs 10.2 x SI_TO_WATER) is printed as context. Without the detector the
water-basis ratio is the verdict. (2026-10-30 bare3m: Si 0.87x, z -1.4 CONSISTENT;
water basis 0.80x, z -2.3 -- a third of the apparent gap was the factor.) The neutral
channel is not a valid gate -- it depends on lining-to-regolith geometry (the
neutron-fraction radius sweep) and LND's neutral includes lander/RHU secondaries
the tool does not model -- so the TOTAL band below is printed for context only.
Without the scorer, the total band is the verdict as before.

Usage (repo root ~/topas, PC, TOPAS_G4_DATA_DIR set, inside tmux; the per-element
model is the default since 2026-10-28, no export needed):
    python3 -u -m paper.lnd_absolute_recheck
    python3 -u -m paper.lnd_absolute_recheck --points lnd --max-batches 12
    python3 -u -m paper.lnd_absolute_recheck --points bare3m      # Si detector on (300 um)
    python3 -u -m paper.lnd_absolute_recheck --points bare3m --si-detector 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

from lunarsim.spec import HabitatSpec, WallLayer
from lunarsim import jobs
from lunarsim.bridge import FULL_RUN
from lunarsim import dosimetry as D
from lunarsim import geometry as G
from paper.neutron_lnd_validate import build_spec as lnd_analogue_spec
from paper.si_water_factor import si_to_water_factor

# ---------------------------------------------------------------------------
# POST-HOC references, Zhang et al. 2020 Sci. Adv. 6 eaaz1334 (never fit targets)
# ---------------------------------------------------------------------------
LND_TOTAL_UGY_H_SI = 13.2         # +- 1, total absorbed, silicon
LND_TOTAL_ERR_SI = 1.0
LND_NEUTRAL_UGY_H_SI = 3.1        # +- 0.5
LND_NEUTRAL_ERR_SI = 0.5
LND_CHARGED_UGY_H_SI = 10.2       # +- 1.1
LND_CHARGED_ERR_SI = 1.1
LND_NEUTRAL_GROSS_UGY_H_SI = 4.7  # +- 0.1, neutral BEFORE the RTG/RHU subtraction
LND_NEUTRAL_BKG_UGY_H_SI = 1.7    # +- 0.4, RTG + 3 RHU, measured on ground Aug 2018
# Matthiae & Berger 2024 Table 3: their Geant4 neutral (n + gamma) at LND, in Si --
# the published same-environment model, for the cross-code read of the neutral row
MB2024_NEUTRAL_UGY_H_SI = 0.0730
# Si -> water: the tool's own spectrum-folded factor (see BASIS), not a fit.
SI_TO_WATER = si_to_water_factor(FULL_RUN.phi_mv)            # 1.227 at phi 400
SI_TO_WATER_ZHANG = 1.30          # Zhang 2020 constant (CRaTER: 1.33) -- reference only
SI_TO_WATER_MB2024 = 1.21         # Matthiae & Berger 2024 model-computed, charged
LND_TOTAL_UGY_H_WATER = LND_TOTAL_UGY_H_SI * SI_TO_WATER     # 16.2 (17.16 with 1.30)
LND_CHARGED_UGY_H_WATER = LND_CHARGED_UGY_H_SI * SI_TO_WATER  # 12.5 (13.26 with 1.30)
LND_NEUTRAL_FRACTION = LND_NEUTRAL_UGY_H_SI / LND_TOTAL_UGY_H_SI  # 0.235, in Si
LND_QBAR_CHARGED = 4.3            # +- 0.7, charged only, LET in water
LND_HDOT_CHARGED_USV_H = 57.1     # +- 10.6, charged only, water

# ratio band read as CONSISTENT (see BASIS in the docstring): LND +-7.6% stat
# combined with the charged-only Si->water factor applied to a total that is
# ~23% neutral.
CONSISTENT_BAND = (0.85, 1.20)
# Charged-channel gate: |z| <= CHARGED_SIGMA_OK is consistent, where z is the
# charged ratio's distance from 1 in combined sigma (LND +-10.8%, MC rel_err).
LND_CHARGED_REL_ERR = LND_CHARGED_ERR_SI / LND_CHARGED_UGY_H_SI
CHARGED_SIGMA_OK = 2.0
SI_DETECTOR_DEFAULT_UM = 300.0    # ~ LND / M&B 2024 detector thickness; default for [bare3m]


def jul22_spec() -> HabitatSpec:
    """The 2026-07-20 converged-anchor geometry: 3 m dome, 5 cm Al (inner) +
    5 cm regolith (outer) = 13.5 + 8.9 = 22.4 g/cm2."""
    return HabitatSpec(name="lnd_jul22_anchor", shape="dome", inner_radius_cm=300.0,
                       walls=[WallLayer("aluminium", 5.0), WallLayer("regolith", 5.0)])


def bare3m_spec() -> HabitatSpec:
    """Same near-bare 0.5 cm Al wall as [lnd] but on a 3 m dome, so the skin
    lining is large enough to sample the heavy-ion tail. [lnd]'s R=30 cm lining
    was sized for the normalisation-invariant neutron fraction f; its absolute
    dose under-samples (2026-10-28: 8.25 uGy/h at rel_err 0.063, below the
    22.4 g/cm2 point -- the wrong ordering)."""
    return HabitatSpec(name="lnd_bare_3m", shape="dome", inner_radius_cm=300.0,
                       walls=[WallLayer("aluminium", 0.5)])


POINTS = {
    "lnd": lambda: lnd_analogue_spec(0.5, 30.0),
    "jul22": jul22_spec,
    "bare3m": bare3m_spec,
}


def run_point(key: str, args) -> dict:
    spec = POINTS[key]()
    areal = spec.areal_density_gcm2()
    print(f"\n  -> [{key}] {spec.name}: {areal:.2f} g/cm2, R={spec.inner_radius_cm:g} cm, "
          f"full composition, converge on skin ...", flush=True)
    cc = jobs.run_composition(spec, tier=FULL_RUN, target_rel_err=args.rel_err,
                              min_batches=args.min_batches, max_batches=args.max_batches,
                              converge_on="skin")
    if not cc.ok:
        print(f"     RUN FAILED (returncode={cc.returncode}). log tail:\n{cc.log_tail}")
        return {"point": key, "ok": False}
    da = D.assess_composition(cc.species_results, phi_MV=FULL_RUN.phi_mv, skin=True)
    if da is None:
        print("     assess_composition returned None (no usable species dose).")
        return {"point": key, "ok": False}
    tissue_ugy_h = da.dose_rate_ugy_day / 24.0
    row = {
        "point": key, "ok": True, "areal_gcm2": areal,
        "inner_radius_cm": spec.inner_radius_cm,
        "skin_tissue_ugy_h": tissue_ugy_h,
        "skin_si_equiv_ugy_h": tissue_ugy_h / SI_TO_WATER,
        "ratio_to_lnd": tissue_ugy_h / LND_TOTAL_UGY_H_WATER,
        "qbar": da.quality_factor,
        "hdot_usv_h": da.equiv_rate_msv_day * 1000.0 / 24.0,
        "annual_msv": da.annual_msv,
        "neutron_fraction": da.neutron_fraction,
        # LND-style entry-channel split (EntryChannelDose scorer; None on a build
        # without the extension). Charged share = 1 - neutral.
        "neutral_dose_fraction": da.neutral_dose_fraction,
        "neutron_abs_fraction": da.neutron_abs_fraction,
        # thin Si detector (--si-detector): dose to SILICON, no conversion factor
        "si_ugy_h": (None if da.si_to_skin_ratio is None
                     else tissue_ugy_h * da.si_to_skin_ratio),
        "si_charged_ugy_h": (None if da.si_charged_to_skin_ratio is None
                             else tissue_ugy_h * da.si_charged_to_skin_ratio),
        "si_charged_rel_err": da.si_charged_rel_err,
        "si_neutral_ugy_h": (None if da.si_neutral_to_skin_ratio is None
                             else tissue_ugy_h * da.si_neutral_to_skin_ratio),
        "si_neutron_ugy_h": (None if da.si_neutron_to_skin_ratio is None
                             else tissue_ugy_h * da.si_neutron_to_skin_ratio),
        "si_gamma_ugy_h": (None if da.si_gamma_to_skin_ratio is None
                           else tissue_ugy_h * da.si_gamma_to_skin_ratio),
        "si_neutral_rel_err": da.si_neutral_rel_err,
        # charged ICRP-60 dose equivalent in Si (water-LET Q) and its LET spectrum
        "si_doseeq_charged_usv_h": (None if da.si_doseeq_charged_to_skin_ratio is None
                                    else tissue_ugy_h * da.si_doseeq_charged_to_skin_ratio),
        "si_doseeq_charged_rel_err": da.si_doseeq_charged_rel_err,
        "si_let_charged_ugy_h": (None if da.si_charged_let_ratios is None
                                 else [tissue_ugy_h * r for r in da.si_charged_let_ratios]),
        "charged_tissue_ugy_h": (None if da.neutral_dose_fraction is None
                                 else tissue_ugy_h * (1.0 - da.neutral_dose_fraction)),
        "neutral_tissue_ugy_h": (None if da.neutral_dose_fraction is None
                                 else tissue_ugy_h * da.neutral_dose_fraction),
        "rel_err": da.rel_err,
        "per_species": {c["species"]: c.get("dose_fraction")
                        for c in (da.contributions or [])},
    }
    print(f"     skin {tissue_ugy_h:.2f} uGy/h tissue (= {row['skin_si_equiv_ugy_h']:.2f} Si-equiv), "
          f"Q-bar {da.quality_factor:.2f}, rel_err "
          f"{'n/a' if da.rel_err is None else f'{da.rel_err:.3f}'}", flush=True)
    if row["si_ugy_h"] is not None:
        print(f"     Si detector {row['si_ugy_h']:.2f} uGy/h Si "
              f"(charged {_f(row['si_charged_ugy_h'], '.2f')}, rel_err "
              f"{_f(row['si_charged_rel_err'], '.3f')}; neutral "
              f"{_f(row['si_neutral_ugy_h'], '.3f')}, rel_err "
              f"{_f(row['si_neutral_rel_err'], '.3f')})", flush=True)
    return row


def _f(x, fmt):
    return format(x, fmt) if x is not None else "n/a"


def charged_z(row: dict, silicon: bool = False):
    """(ratio, sigma, z) of the tool's charged dose vs LND charged, or None
    without the split: water lining vs LND x SI_TO_WATER, or (silicon=True) the
    thin Si detector vs LND's 10.2 Si directly. sigma propagates LND's relative
    error and the run's rel_err onto the ratio. Water basis: the total skin dose's
    rel_err stands in for the charged channel's, which carries ~75-90% of it.
    Silicon: the detector's own si_charged_rel_err; only when that is missing
    (<2 batches) is the skin's borrowed, and then z is optimistic."""
    ch = row.get("si_charged_ugy_h" if silicon else "charged_tissue_ugy_h")
    if ch is None:
        return None
    ratio = ch / (LND_CHARGED_UGY_H_SI if silicon else LND_CHARGED_UGY_H_WATER)
    mc = row.get("si_charged_rel_err") if silicon else None
    if mc is None:
        mc = row.get("rel_err") or 0.0
    sigma = ratio * (LND_CHARGED_REL_ERR ** 2 + mc ** 2) ** 0.5
    return ratio, sigma, (ratio - 1.0) / sigma


def si_neutral_report(row: dict):
    """The Si detector's neutral channel against LND and M&B 2024, or None without
    the scorer. REPORTED, not a gate (see SILICON NEUTRAL CHANNEL): the ratios are
    context for a platform-contaminated measurement. 'other' is the neutral entry
    that is neither neutron nor gamma (e.g. neutral pions/kaons), by difference."""
    nt = row.get("si_neutral_ugy_h")
    if nt is None:
        return None
    n, g = row.get("si_neutron_ugy_h"), row.get("si_gamma_ugy_h")
    other = nt - n - g if (n is not None and g is not None) else None
    return {"neutral": nt, "neutron": n, "gamma": g, "other": other,
            "rel_err": row.get("si_neutral_rel_err"),
            "to_lnd": nt / LND_NEUTRAL_UGY_H_SI,
            "to_lnd_gross": nt / LND_NEUTRAL_GROSS_UGY_H_SI,
            "to_mb2024": nt / MB2024_NEUTRAL_UGY_H_SI,
            "neutron_share": (n / nt if (n is not None and nt) else None),
            "gamma_share": (g / nt if (g is not None and nt) else None)}


LND_QBAR_CHARGED_ERR = 0.7
LND_HDOT_CHARGED_ERR = 10.6
# factor-free: Zhang's water H with their own constant 1.30 divided back out
LND_HDOT_CHARGED_SI_BASIS = LND_HDOT_CHARGED_USV_H / SI_TO_WATER_ZHANG   # 43.9
LET_HIGH_KEV_UM = 10.0            # "high-LET" split for the printed fraction


def let_bin_edges() -> list:
    """(lo, hi) water-LET edges, keV/um, of each Si charged LET bin; None = open."""
    e = (None,) + tuple(G.SI_LET_EDGES_KEV_UM) + (None,)
    return [(e[k], e[k + 1]) for k in range(G.SI_LET_NBINS)]


def si_charged_q_report(row: dict):
    """The Si detector's charged <Q>, factor-free dose equivalent and water-LET
    spectrum against LND, or None without the dose-eq scorer. Post-hoc, NOT a gate
    (see SILICON CHARGED <Q> AND LET SPECTRUM). The <Q> z carries only LND's +-0.7
    (H and D share the same hits, so the MC part largely cancels); the H z combines
    LND's +-18.6% with the detector's own H error."""
    h, d = row.get("si_doseeq_charged_usv_h"), row.get("si_charged_ugy_h")
    if h is None or not d:
        return None
    q = h / d
    h_ratio = h / LND_HDOT_CHARGED_SI_BASIS
    h_sigma = h_ratio * ((LND_HDOT_CHARGED_ERR / LND_HDOT_CHARGED_USV_H) ** 2
                         + (row.get("si_doseeq_charged_rel_err") or 0.0) ** 2) ** 0.5
    out = {"qbar": q, "q_ratio": q / LND_QBAR_CHARGED,
           "q_z": (q - LND_QBAR_CHARGED) / LND_QBAR_CHARGED_ERR,
           "h_usv_h": h, "h_ratio": h_ratio, "h_sigma": h_sigma,
           "h_z": (h_ratio - 1.0) / h_sigma,
           "let_fractions": None, "let_closure": None, "high_let_fraction": None,
           "high_let_h_share": None}
    bins = row.get("si_let_charged_ugy_h")
    if bins:
        tot = sum(bins)
        if tot > 0:
            fr = [b / tot for b in bins]
            edges = let_bin_edges()
            hi = [k for k, (lo, _) in enumerate(edges)
                  if lo is not None and lo >= LET_HIGH_KEV_UM]
            out["let_fractions"] = fr
            out["let_closure"] = tot / d
            out["high_let_fraction"] = sum(fr[k] for k in hi)
        out["let_edges"] = let_bin_edges()
    return out


def _charged_verdict(z: float) -> str:
    return ("CONSISTENT" if abs(z) <= CHARGED_SIGMA_OK else
            "LOW -- tension (do NOT re-fit CAL)" if z < 0 else
            "HIGH -- tension (do NOT re-fit CAL)")


def resolve_si_detector(arg, keys) -> float:
    """--si-detector value to use: an explicit value wins (0 = off); unset means
    SI_DETECTOR_DEFAULT_UM when [bare3m] -- the trusted comparand -- is requested,
    else off."""
    if arg is not None:
        return arg
    return SI_DETECTOR_DEFAULT_UM if "bare3m" in keys else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default="lnd,jul22",
                    help=f"comma-separated subset of {sorted(POINTS)} (default both)")
    ap.add_argument("--rel-err", type=float, default=0.05,
                    help="skin dose convergence target (default 0.05)")
    ap.add_argument("--min-batches", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=10,
                    help="the R=30 cm lining is small and converges slowly (default 10)")
    ap.add_argument("--si-detector", type=float, default=None, metavar="UM",
                    help="thin Si detector of UM micrometres on the inner wall face; its "
                         "charged dose is compared with LND in Si directly and is the "
                         f"verdict (dome points; default {SI_DETECTOR_DEFAULT_UM:g} when "
                         "bare3m is requested, else off; 0 = off)")
    ap.add_argument("--out", default="runs/lnd_recheck/lnd_recheck.json")
    args = ap.parse_args(argv)

    keys = [k.strip() for k in args.points.split(",") if k.strip()]
    bad = [k for k in keys if k not in POINTS]
    if bad or not keys:
        ap.error(f"unknown point(s) {bad}; choose from {sorted(POINTS)}")
    args.si_detector = resolve_si_detector(args.si_detector, keys)
    if args.si_detector:
        os.environ[G.SI_DETECTOR_ENV] = f"{args.si_detector:g}"
        G.si_detector_um()        # validate the range before any run starts

    model = D.PERELEMENT_MODEL
    print("=" * 74)
    print("Chang'E-4 LND absolute recheck -- Matthiae absolute normalisation")
    print("=" * 74)
    print(f"LUNARSIM_GCR_PERELEMENT = {model}   (flood runs {D.flood_model_for(model)})")
    if model == "none":
        print("WARNING: 'none' = legacy 2.0 pin + hand-set heavies, NOT the new basis.\n"
              "         unset LUNARSIM_GCR_PERELEMENT (default perelement) and re-run.")
    print(f"CAL = {D.OUTER_GAUGE_ANCHOR_CAL:.5f} (not re-fit)   phi = {FULL_RUN.phi_mv:g} MV")
    print(f"Si->water {SI_TO_WATER:.3f} (tool spectrum, paper.si_water_factor; Zhang "
          f"constant {SI_TO_WATER_ZHANG}, M&B 2024 model {SI_TO_WATER_MB2024})")
    print(f"LND total {LND_TOTAL_UGY_H_SI} +- {LND_TOTAL_ERR_SI} uGy/h Si x {SI_TO_WATER:.3f} "
          f"= {LND_TOTAL_UGY_H_WATER:.2f} uGy/h water  (post-hoc reference; "
          f"{LND_TOTAL_UGY_H_SI * SI_TO_WATER_ZHANG:.2f} with 1.30)")
    if args.si_detector:
        print(f"thin Si detector {args.si_detector:g} um on the inner wall face "
              f"({G.SI_DETECTOR_ENV})")

    rows = [run_point(k, args) for k in keys]

    print("\n" + "-" * 74)
    print(f"{'point':>6} {'g/cm2':>6} {'R cm':>5} {'tissue uGy/h':>13} {'Si-eq':>6} "
          f"{'/LND':>6} {'Q-bar':>6} {'f_n':>6} {'mSv/yr':>7} {'rel':>6}")
    for r in rows:
        if not r.get("ok"):
            print(f"{r['point']:>6}  FAILED")
            continue
        print(f"{r['point']:>6} {r['areal_gcm2']:>6.1f} {r['inner_radius_cm']:>5.0f} "
              f"{r['skin_tissue_ugy_h']:>13.2f} {r['skin_si_equiv_ugy_h']:>6.2f} "
              f"{r['ratio_to_lnd']:>6.2f} {r['qbar']:>6.2f} {_f(r['neutron_fraction'], '6.3f')} "
              f"{r['annual_msv']:>7.0f} {_f(r['rel_err'], '6.3f')}")

    split = [r for r in rows if r.get("ok") and r.get("neutral_dose_fraction") is not None]
    if split:
        print(f"\nLND-style charged/neutral split (entry channel; LND charged "
              f"{LND_CHARGED_UGY_H_SI} +- {LND_CHARGED_ERR_SI} Si x {SI_TO_WATER:.3f} = "
              f"{LND_CHARGED_UGY_H_WATER:.2f} water, neutral {LND_NEUTRAL_UGY_H_SI} +- "
              f"{LND_NEUTRAL_ERR_SI} Si = {LND_NEUTRAL_FRACTION:.3f} of total):")
        print(f"{'point':>6} {'charged':>8} {'/LND':>6} {'neutral':>8} {'/LND-Si':>8} "
              f"{'f_neutral':>9} {'f_n,abs':>8}")
        for r in split:
            ch, nt = r["charged_tissue_ugy_h"], r["neutral_tissue_ugy_h"]
            print(f"{r['point']:>6} {ch:>8.2f} {ch / LND_CHARGED_UGY_H_WATER:>6.2f} "
                  f"{nt:>8.2f} {nt / LND_NEUTRAL_UGY_H_SI:>8.2f} "
                  f"{r['neutral_dose_fraction']:>9.3f} {_f(r['neutron_abs_fraction'], '8.3f')}")
        print("  charged is compared in water (the Si->water factor is a charged one);"
              " neutral is tissue vs LND SILICON -- no agreed Si->tissue factor for"
              " neutrals (H-rich tissue reads neutrons HIGHER), so read /LND-Si as an"
              " upper-side ratio. Zhang 2020 call LND's neutral 'remarkably high'"
              " (lander/RHU secondaries the tool does not model).")
    elif any(r.get("ok") for r in rows):
        print("\n(no charged/neutral split: build lacks the EntryChannelDose extension)")

    lo, hi = CONSISTENT_BAND
    if split:
        print(f"\nPOST-HOC verdict on the CHARGED channel (|z| <= {CHARGED_SIGMA_OK:g} "
              f"consistent; LND +-{LND_CHARGED_REL_ERR:.1%} with run rel_err, see GATE):")
        for r in split:
            ratio, sigma, z = charged_z(r)
            r["charged_ratio"], r["charged_sigma"], r["charged_z"] = ratio, sigma, z
            si = charged_z(r, silicon=True)
            if si is None:
                print(f"  [{r['point']}] charged (water basis) {ratio:.2f} +- {sigma:.2f} "
                      f"x LND (z = {z:+.1f})  -> {_charged_verdict(z)}")
                continue
            s_ratio, s_sigma, s_z = si
            r["si_charged_ratio"], r["si_charged_sigma"], r["si_charged_z"] = s_ratio, s_sigma, s_z
            err = (f"Si rel_err {r['si_charged_rel_err']:.3f}"
                   if r.get("si_charged_rel_err") is not None
                   else "skin rel_err BORROWED (<2 batches) -- z optimistic")
            print(f"  [{r['point']}] Si detector charged {r['si_charged_ugy_h']:.2f} uGy/h Si = "
                  f"{s_ratio:.2f} +- {s_sigma:.2f} x LND {LND_CHARGED_UGY_H_SI} Si "
                  f"(z = {s_z:+.1f}; {err}; no conversion factor)  -> {_charged_verdict(s_z)}")
            print(f"         context: water basis {ratio:.2f} +- {sigma:.2f} x LND x "
                  f"{SI_TO_WATER:.3f} (z = {z:+.1f}); tissue/Si "
                  f"{r['charged_tissue_ugy_h'] / r['si_charged_ugy_h']:.3f} is NOT a Si->water"
                  " factor (2 cm tissue lining vs Si at the face)")
        for r in split:
            rep = si_neutral_report(r)
            if rep is None:
                continue
            r["si_neutral"] = rep
            print(f"  [{r['point']}] Si detector NEUTRAL {rep['neutral']:.3f} uGy/h Si "
                  f"(rel_err {_f(rep['rel_err'], '.3f')}; neutron {_f(rep['neutron'], '.3f')}"
                  f" [{_f(rep['neutron_share'], '.0%')}], gamma {_f(rep['gamma'], '.3f')}"
                  f" [{_f(rep['gamma_share'], '.0%')}], other {_f(rep['other'], '.3f')}) "
                  "-- REPORTED, NOT A GATE:")
            print(f"         {rep['to_lnd']:.3f}x LND {LND_NEUTRAL_UGY_H_SI} (after RTG/RHU "
                  f"{LND_NEUTRAL_BKG_UGY_H_SI} subtracted; {rep['to_lnd_gross']:.3f}x the gross "
                  f"{LND_NEUTRAL_GROSS_UGY_H_SI});  {rep['to_mb2024']:.2f}x M&B 2024 Geant4 "
                  f"{MB2024_NEUTRAL_UGY_H_SI} (which is itself "
                  f"{MB2024_NEUTRAL_UGY_H_SI / LND_NEUTRAL_UGY_H_SI:.3f}x LND)")
        for r in split:
            qr = si_charged_q_report(r)
            if qr is None:
                continue
            r["si_charged_q"] = qr
            print(f"  [{r['point']}] Si detector charged <Q> {qr['qbar']:.2f} vs LND "
                  f"{LND_QBAR_CHARGED} +- {LND_QBAR_CHARGED_ERR} = {qr['q_ratio']:.2f}x "
                  f"(z = {qr['q_z']:+.1f}) -- post-hoc, NOT a gate")
            print(f"         H = D_Si,ch x <Q> = {qr['h_usv_h']:.1f} uSv/h vs LND "
                  f"{LND_HDOT_CHARGED_USV_H}/{SI_TO_WATER_ZHANG} = "
                  f"{LND_HDOT_CHARGED_SI_BASIS:.1f} (factor-free) = {qr['h_ratio']:.2f} +- "
                  f"{qr['h_sigma']:.2f}x (z = {qr['h_z']:+.1f}; H rel_err "
                  f"{_f(r.get('si_doseeq_charged_rel_err'), '.3f')})")
            if qr["let_fractions"] is not None:
                print(f"         charged Si dose by water LET (keV/um); bins sum to "
                      f"{qr['let_closure']:.3f}x the charged Si dose (expect 1); "
                      f">= {LET_HIGH_KEV_UM:g} keV/um carries {qr['high_let_fraction']:.1%}:")
                for (lo, hi), fr in zip(qr["let_edges"], qr["let_fractions"]):
                    lab = f"[{_f(lo, '.3g') if lo else '0'}, {_f(hi, '.3g') if hi else 'inf'})"
                    print(f"           {lab:>16} {fr:>7.2%}")
        print(f"  context only -- TOTAL tissue/LND-water vs band {lo}-{hi} (neutral "
              "channel is geometry/lander dependent, not a gate):")
    else:
        print(f"\nPOST-HOC reading (band {lo}-{hi} on tissue/LND-water, see BASIS):")
    for r in rows:
        if not r.get("ok"):
            continue
        x = r["ratio_to_lnd"]
        v = ("CONSISTENT" if lo <= x <= hi else
             "LOW -- investigate (do NOT re-fit CAL)" if x < lo else
             "HIGH -- investigate (do NOT re-fit CAL)")
        print(f"  [{r['point']}] {x:.2f}x LND  -> {v}")
    print(f"  Q-bar column is skin total (incl. neutrons), not LND's charged-only "
          f"{LND_QBAR_CHARGED} +- {LND_QBAR_CHARGED_ERR}; the Si charged <Q> above is the"
          " like-for-like read. Context, not a gate.")
    print("  Trusted absolute comparand is [bare3m] (near-bare wall, 3 m lining samples"
          " the heavy-ion tail), read through its Si detector; [lnd]'s R=30 cm lining is valid for ratios only;"
          " [jul22] is continuity with the 2026-07-20 anchor.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "cal": D.OUTER_GAUGE_ANCHOR_CAL,
                               "phi_mv": FULL_RUN.phi_mv,
                               "si_to_water": SI_TO_WATER,
                               "si_detector_um": args.si_detector,
                               "lnd_water_ugy_h": LND_TOTAL_UGY_H_WATER,
                               "lnd_charged_water_ugy_h": LND_CHARGED_UGY_H_WATER,
                               "lnd_neutral_si_ugy_h": LND_NEUTRAL_UGY_H_SI,
                               "mb2024_neutral_si_ugy_h": MB2024_NEUTRAL_UGY_H_SI,
                               "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
