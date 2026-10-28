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
(10.2 x 1.30 x 4.3). The tool scores TOTAL absorbed dose in a TISSUE lining, so
the like-for-like comparand is 13.2 x 1.30 = 17.2 uGy/h water (0.41 mGy/d).
The constant 1.30 is a charged-particle factor; the neutral share converts
differently (hydrogen makes tissue more neutron-sensitive), so a tool/LND ratio
of ~1.0-1.15 is consistent, not an over-read. (The 2026-07-20 anchor divided by
1.5, which overstated the agreement; with 1.30 its 19.79 uGy/h tissue was 1.15x.)

Other known differences: LND sits inside the lander payload compartment (extra
local mass + structure neutrons the tool does not model) and measured
Jan-Feb 2019 near phi ~400 MV; the tool's ceiling is 20 GeV/n (~-7.6% surface).

Usage (repo root ~/topas, PC, TOPAS_G4_DATA_DIR set, inside tmux):
    export LUNARSIM_GCR_PERELEMENT=perelement
    python3 -u -m paper.lnd_absolute_recheck
    python3 -u -m paper.lnd_absolute_recheck --points lnd --max-batches 12
"""
import argparse
import json
import sys
from pathlib import Path

from lunarsim.spec import HabitatSpec, WallLayer
from lunarsim import jobs
from lunarsim.bridge import FULL_RUN
from lunarsim import dosimetry as D
from paper.neutron_lnd_validate import build_spec as lnd_analogue_spec

# ---------------------------------------------------------------------------
# POST-HOC references, Zhang et al. 2020 Sci. Adv. 6 eaaz1334 (never fit targets)
# ---------------------------------------------------------------------------
LND_TOTAL_UGY_H_SI = 13.2         # +- 1, total absorbed, silicon
LND_TOTAL_ERR_SI = 1.0
LND_NEUTRAL_UGY_H_SI = 3.1        # +- 0.5
SI_TO_WATER = 1.30                # Zhang 2020 constant conversion (CRaTER: 1.33)
LND_TOTAL_UGY_H_WATER = LND_TOTAL_UGY_H_SI * SI_TO_WATER     # 17.16
LND_QBAR_CHARGED = 4.3            # +- 0.7, charged only, LET in water
LND_HDOT_CHARGED_USV_H = 57.1     # +- 10.6, charged only, water

# ratio band read as CONSISTENT (see BASIS in the docstring): LND +-7.6% stat
# combined with the charged-only Si->water factor applied to a total that is
# ~23% neutral.
CONSISTENT_BAND = (0.85, 1.20)


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
        "rel_err": da.rel_err,
        "per_species": {c["species"]: c.get("dose_fraction")
                        for c in (da.contributions or [])},
    }
    print(f"     skin {tissue_ugy_h:.2f} uGy/h tissue (= {row['skin_si_equiv_ugy_h']:.2f} Si-equiv), "
          f"Q-bar {da.quality_factor:.2f}, rel_err "
          f"{'n/a' if da.rel_err is None else f'{da.rel_err:.3f}'}", flush=True)
    return row


def _f(x, fmt):
    return format(x, fmt) if x is not None else "n/a"


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
    ap.add_argument("--out", default="runs/lnd_recheck/lnd_recheck.json")
    args = ap.parse_args(argv)

    keys = [k.strip() for k in args.points.split(",") if k.strip()]
    bad = [k for k in keys if k not in POINTS]
    if bad or not keys:
        ap.error(f"unknown point(s) {bad}; choose from {sorted(POINTS)}")

    model = D.PERELEMENT_MODEL
    print("=" * 74)
    print("Chang'E-4 LND absolute recheck -- Matthiae absolute normalisation")
    print("=" * 74)
    print(f"LUNARSIM_GCR_PERELEMENT = {model}   (flood runs {D.flood_model_for(model)})")
    if model == "none":
        print("WARNING: 'none' = legacy 2.0 pin + hand-set heavies, NOT the new basis.\n"
              "         export LUNARSIM_GCR_PERELEMENT=perelement and re-run.")
    print(f"CAL = {D.OUTER_GAUGE_ANCHOR_CAL:.5f} (not re-fit)   phi = {FULL_RUN.phi_mv:g} MV")
    print(f"LND total {LND_TOTAL_UGY_H_SI} +- {LND_TOTAL_ERR_SI} uGy/h Si x {SI_TO_WATER} "
          f"= {LND_TOTAL_UGY_H_WATER:.2f} uGy/h water  (post-hoc reference)")

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

    lo, hi = CONSISTENT_BAND
    print(f"\nPOST-HOC reading (band {lo}-{hi} on tissue/LND-water, see BASIS):")
    for r in rows:
        if not r.get("ok"):
            continue
        x = r["ratio_to_lnd"]
        v = ("CONSISTENT" if lo <= x <= hi else
             "LOW -- investigate (do NOT re-fit CAL)" if x < lo else
             "HIGH -- investigate (do NOT re-fit CAL)")
        print(f"  [{r['point']}] {x:.2f}x LND  -> {v}")
    print(f"  Q-bar is total (incl. neutrons) vs LND charged-only {LND_QBAR_CHARGED} +- 0.7:"
          " context, not a gate.")
    print("  Primary comparand is [lnd] (near-bare, lining at the regolith); [jul22]"
          " is continuity with the 2026-07-20 anchor.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "cal": D.OUTER_GAUGE_ANCHOR_CAL,
                               "phi_mv": FULL_RUN.phi_mv,
                               "lnd_water_ugy_h": LND_TOTAL_UGY_H_WATER,
                               "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
