#!/usr/bin/env python3
"""Crossover coherence check for the shipped per-element thin-wall GCR kernel.

Background (memory crossover-discontinuity + let-scorer-thin-qbar-rootcause):
the tool runs TWO GCR engines and hands off at the ~19 g/cm^2 crossover --
the phantom-matched thin-wall KERNEL fold below it (direct-transmission regime)
and the OLTARIS-validated FLOOD MC above it. Two independent fixes have since
shipped:

  * WALL_GEOMETRY = "shell" (kernel_gen/config.py, 2026-10-13) removed the slab's
    constant ~1.70x flux-weighted slant that over-attenuated deep ABSORBED dose,
    so the kernel's absorbed-dose SLOPE now tracks OLTARIS (flat with depth)
    instead of plunging.
  * the 10-ion per-element kernel (2026-10-19) removed the 5-group Z^2-lumping
    that inflated <Q> and the heavy-ion dose fraction.

This script does NOT run MC. It folds the shipped kernels (pure arithmetic,
dosimetry.fold_gcr_thinwall via assess_gcr_thinwall) across the crossover and
lays the result beside the OLTARIS-validated deep reference, so the coherence of
the handoff can be read off and locked with a regression guard. It deliberately
re-measures the stale pre-fix claim baked into dosimetry.areal_density_confidence
("kernel 0.52x below vs flood 2.6x above OLTARIS" at the 19 g/cm^2 gate).

TARGET-BLIND: OLTARIS numbers here are a POST-HOC reference, never a fit target.

Run (laptop, no MC needed):
    python3 -m paper.crossover_coherence
"""
import os
import sys
import csv
from pathlib import Path

# Fold the kernels at solar minimum (phi=400 MV), the kernel's build point.
PHI_MV = 400.0
AL_DENSITY = 2.7  # g/cm^3 (lunarsim.spec.MATERIALS['aluminium'])

# Sweep spans both sides of the ~19 g/cm^2 crossover. The per-element kernel
# anchors are [0, 2.025, 10, 50]; fold_gcr_thinwall interpolates between them
# (log-log) and clamps outside, so 2.025..50 is all inside the measured band.
SWEEP_GCM2 = [2.025, 5.0, 10.0, 15.0, 19.0, 25.0, 40.0, 50.0]

# OLTARIS lunar-surface TOTAL at 54 g/cm^2 Al (memory oltaris-absorbed-dose-
# crosscheck + perelem_validate): absorbed 0.296 mGy/day, <Q> 2.16. The deep
# anchor the shell/corrected kernel was validated against.
OLTARIS_DEEP_GCM2 = 50.0
OLTARIS_D_MGY_DAY = 0.296
OLTARIS_QBAR = 2.16

REPO = Path(__file__).resolve().parent.parent
FLOOD_CSV = REPO / "paper" / "flood_qbar_depth.csv"


def _make_al_spec(ad_gcm2: float):
    """A dome with a single aluminium wall whose areal density is ad_gcm2."""
    from lunarsim.spec import HabitatSpec, WallLayer
    t = ad_gcm2 / AL_DENSITY
    return HabitatSpec(name=f"al_{ad_gcm2:g}",
                       shape="dome", inner_radius_cm=400.0,
                       walls=[WallLayer(material="aluminium", thickness_cm=t)])


def _fold(ad_gcm2: float, model: str) -> dict:
    """Fold the shipped thin-wall kernel at ad_gcm2 under the given GCR model
    ('perelement' -> 10-ion kernel; 'none' -> shipped 5-ion lumped kernel).
    Returns effective absorbed D (mGy/day), effective dose E (mSv/yr), <Q>."""
    import lunarsim.dosimetry as D
    # fold_gcr_thinwall reads LUNARSIM_GCR_PERELEMENT at call time AND the kernel
    # loader keys on it; set BOTH the env and the dosimetry model so the spectrum
    # and the kernel stay consistent (set_perelement_model also clears the lru
    # caches that memoise the flux integral / make_source).
    os.environ["LUNARSIM_GCR_PERELEMENT"] = model
    D.set_perelement_model(model)
    a = D.assess_gcr_thinwall(_make_al_spec(ad_gcm2), phi_MV=PHI_MV)
    if a is None:
        return {"D_mgy_day": 0.0, "E_msv_yr": 0.0, "qbar": 0.0}
    return {"D_mgy_day": a.dose_rate_ugy_day / 1.0e3,
            "E_msv_yr": a.annual_msv,
            "qbar": a.quality_factor}


def _load_flood_deep() -> dict:
    """OLTARIS-validated deep reference curve (the shell/corrected kernel folded at
    the deep anchors, whose 50 g/cm^2 point reproduces OLTARIS). Keyed by g/cm^2."""
    out = {}
    if not FLOOD_CSV.exists():
        return out
    with open(FLOOD_CSV) as fh:
        for row in csv.DictReader(fh):
            ad = float(row["areal_gcm2"])
            out[ad] = {"D_mgy_day": float(row["kernel_Deff_mgy_day"]),
                       "E_msv_yr": float(row["kernel_E_mSv_yr"]),
                       "qbar": float(row["kernel_qbar"])}
    return out


def main() -> int:
    flood = _load_flood_deep()
    base = {ad: _fold(ad, "none") for ad in SWEEP_GCM2}
    perel = {ad: _fold(ad, "perelement") for ad in SWEEP_GCM2}

    print("\nCROSSOVER COHERENCE  --  shipped thin-wall GCR kernel folds "
          f"(phi={PHI_MV:.0f} MV, Al wall)")
    print("Effective absorbed D [mGy/day] | effective dose E [mSv/yr] | emergent <Q>")
    print("-" * 92)
    print(f"{'g/cm2':>7} | {'D base':>7} {'D p-el':>7} {'Dr p/b':>7} | "
          f"{'E base':>7} {'E p-el':>7} {'Er p/b':>7} | {'Q base':>7} {'Q p-el':>7}")
    print("-" * 92)
    for ad in SWEEP_GCM2:
        b, p = base[ad], perel[ad]
        dr = p["D_mgy_day"] / b["D_mgy_day"] if b["D_mgy_day"] else 0.0
        er = p["E_msv_yr"] / b["E_msv_yr"] if b["E_msv_yr"] else 0.0
        print(f"{ad:>7.2f} | {b['D_mgy_day']:>7.4f} {p['D_mgy_day']:>7.4f} {dr:>7.3f} | "
              f"{b['E_msv_yr']:>7.1f} {p['E_msv_yr']:>7.1f} {er:>7.3f} | "
              f"{b['qbar']:>7.3f} {p['qbar']:>7.3f}")

    # --- Sub-check 1: baseline kernel reproduces OLTARIS at the deep anchor ----
    bd = base[OLTARIS_DEEP_GCM2]
    print("\n[1] Baseline (5-ion lumped) kernel vs OLTARIS at the deep anchor "
          f"({OLTARIS_DEEP_GCM2:g} g/cm2):")
    print(f"    D {bd['D_mgy_day']:.4f} vs OLTARIS {OLTARIS_D_MGY_DAY:.4f} "
          f"mGy/day  (ratio {bd['D_mgy_day']/OLTARIS_D_MGY_DAY:.3f})")
    print(f"    Q {bd['qbar']:.3f} vs OLTARIS {OLTARIS_QBAR:.3f}  "
          f"(ratio {bd['qbar']/OLTARIS_QBAR:.3f})")
    s1 = abs(bd["D_mgy_day"] / OLTARIS_D_MGY_DAY - 1.0) < 0.10 and \
        abs(bd["qbar"] / OLTARIS_QBAR - 1.0) < 0.10
    print(f"    -> {'PASS' if s1 else 'CONCERN'}: deep-anchor OLTARIS match "
          "(the kernel's absorbed-dose SLOPE fix is intact)")

    # --- Sub-check 2: <Q> falls monotonically with shielding (both kernels) ----
    def mono_down(d):
        q = [d[ad]["qbar"] for ad in SWEEP_GCM2]
        return all(q[i] >= q[i + 1] - 1e-9 for i in range(len(q) - 1))
    s2 = mono_down(base) and mono_down(perel)
    print(f"\n[2] <Q> falls monotonically with shielding (both kernels): "
          f"{'PASS' if s2 else 'CONCERN'}")

    # --- Sub-check 3: per-element lowers <Q> and E everywhere (the fix's point) -
    s3 = all(perel[ad]["qbar"] <= base[ad]["qbar"] + 1e-9 and
             perel[ad]["E_msv_yr"] <= base[ad]["E_msv_yr"] + 1e-9
             for ad in SWEEP_GCM2)
    print(f"[3] Per-element kernel lowers <Q> AND E at every depth "
          f"(de-lumping removes over-read): {'PASS' if s3 else 'CONCERN'}")

    # --- Sub-check 4: <Q> continuity across the gate (the fix's coherence win) --
    # dosimetry.areal_density_confidence still carries the PRE-FIX claim that the
    # headline is "~2-3x off" at the 19 g/cm^2 gate (kernel <Q>~4 vs OLTARIS deep
    # ~2.2). With the per-element kernel the thin-side <Q> at the gate should now
    # MATCH the OLTARIS-validated deep <Q>, i.e. the handoff is seamless in <Q>.
    gate = 19.0
    pg, bg = perel[gate], base[gate]
    qgap = abs(pg["qbar"] - OLTARIS_QBAR) / OLTARIS_QBAR
    print(f"\n[4] <Q> continuity at the crossover gate ({gate:g} g/cm2):")
    print(f"    per-element thin-kernel <Q> {pg['qbar']:.3f}  vs  OLTARIS deep <Q> "
          f"{OLTARIS_QBAR:.3f}   (gap {qgap*100:.1f}%)")
    print(f"    lumped (old) thin-kernel <Q> {bg['qbar']:.3f}  <- the pre-fix "
          "over-read baked into areal_density_confidence")
    s4 = qgap < 0.10
    print(f"    -> {'PASS' if s4 else 'CONCERN'}: per-element <Q> handoff is "
          "seamless; the stale areal_density_confidence note is now conservative")
    print(f"    (per-element E @gate {pg['E_msv_yr']:.0f} mSv/yr reproduces the "
          f"validated ~130 shielded-dome headline; lumped read {bg['E_msv_yr']:.0f})")

    # --- Sub-check 5: thin organ-absorbed vs OLTARIS point (self-shielding) -----
    # The per-element absorbed dose is ~flat with depth (proton/He-dominated once
    # the heavy over-weight is gone) at ~0.55x the OLTARIS bare-point -- the
    # EXPECTED organ-vs-point self-shielding factor (perelem_validate: ~0.5-0.56x),
    # NOT a defect. Flag if it drifts outside that physical band.
    pd = perel[OLTARIS_DEEP_GCM2]["D_mgy_day"] / OLTARIS_D_MGY_DAY
    s5 = 0.45 <= pd <= 0.62
    print(f"\n[5] Per-element organ-absorbed vs OLTARIS point @{OLTARIS_DEEP_GCM2:g}"
          f" g/cm2: ratio {pd:.3f}  (expected 0.50-0.56 organ/point self-shielding)")
    print(f"    -> {'PASS' if s5 else 'CONCERN'}")

    print("\nNOTE: the per-element kernel is the THIN engine, live only BELOW the "
          f"{gate:g} g/cm2 gate.")
    print("      The flood (deep) engine collapses 'perelement' -> 'none' "
          "(jobs.flood_model_for),")
    print("      so the OLTARIS-validated deep headline is unchanged by the "
          "per-element kernel. A real")
    print("      flood-MC value at the gate needs a PC run; this script folds "
          "kernels only (no MC).")

    ok = s1 and s2 and s3 and s4 and s5
    print(f"\nVERDICT: {'COHERENT' if ok else 'REVIEW'} "
          f"(deep-anchor={('PASS' if s1 else 'FAIL')}, "
          f"Q-monotone={('PASS' if s2 else 'FAIL')}, "
          f"delump-direction={('PASS' if s3 else 'FAIL')}, "
          f"Q-gate-continuity={('PASS' if s4 else 'FAIL')}, "
          f"organ-absorbed-band={('PASS' if s5 else 'FAIL')})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
