#!/usr/bin/env python3
"""Does the flood's outer gauge read the habitat's free-field fluence?

Diagnostic for the flat ~6% left by the geometry-residual probe (flood/kernel
after the lining x spot lifts = 0.937 +/- 0.025). The two engines normalise the
incident beam differently:

  kernel : R = D / phi_ff,  phi_ff = N / (pi * spot^2)  -- the true free-field
           scalar fluence at the phantom (every beam axis passes through it).
  flood  : rate = D * flux / fo * OUTER_GAUGE_ANCHOR_CAL * gauge_corr,
           fo = TOPAS Fluence (track length / volume) on the OuterShell
           hemisphere (880-882 cm, z >= 0), filtered to the primary Z/A.

Both are consistent iff fo / OUTER_GAUGE_ANCHOR_CAL == phi_hab, i.e. iff the
physics-expected calibration  CAL_phys = fo / phi_hab  equals the shipped 0.94932.
CAL was never derived from that; it carries a historical re-anchor chain
(dosimetry.py, three stages preserving one 3 m dome dose).

  geometric  -- laptop ray-trace of the exact make_source ring dome (straight
                primaries, ground at z = 0, no interactions). No TOPAS.
  measure    -- PC: real flood runs of the 3 m CAL-anchor dome per species; reads
                fluence_outside and the launched primaries, so CAL_phys includes
                everything the real gauge sees (albedo / same-Z/A secondaries).

If CAL_phys ~ 1.00 the flood is normalised 0.949x low against first principles,
which predicts flood/kernel ~ 0.949 -- the size and sign of the residual.

TARGET-BLIND: no dose enters; fo and phi_hab are source-geometry quantities.
DIAGNOSTIC ONLY: OUTER_GAUGE_ANCHOR_CAL is NOT changed (no refit).

Usage
-----
  python3 paper/gauge_normalisation_check.py geometric
  # PC (TOPAS):
  .venv/bin/python paper/gauge_normalisation_check.py measure --seeds 4 \\
      | tee runs/gauge_check.log
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RESIDUAL = (0.937, 0.025)        # flood/kernel after lining x spot (probe, 36 seeds)


# ----------------------------------------------------------------------------
# geometric ray-trace
# ----------------------------------------------------------------------------
def ring_layout(rings: int, azimuth: int):
    """(theta, phi) exactly as make_source.gcr_block: sin^2 area-midpoint rings."""
    out = []
    for r in range(rings):
        th = math.asin(math.sqrt((r + 0.5) / rings))
        for a in range(azimuth):
            out.append((th, a * 2.0 * math.pi / azimuth))
    return out


def _len_in_ball_above_ground(p, d, R):
    import numpy as np
    b = np.einsum("ij,ij->i", p, d)
    c = np.einsum("ij,ij->i", p, p) - R * R
    disc = b * b - c
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    s0 = np.maximum(-b - sq, 0.0)
    s1 = np.maximum(-b + sq, 0.0)
    dz, pz = d[:, 2], p[:, 2]
    sz = np.where(dz < 0, np.maximum(pz, 0.0) / -dz, np.inf)   # stopped at the ground
    sz = np.where(pz < 0, 0.0, sz)                              # born underground
    return np.where(ok, np.maximum(np.minimum(s1, sz) - np.minimum(s0, sz), 0.0), 0.0)


def geometric_cal(rings: int, azimuth: int, beam_radius: float = 1400.0,
                  spot: float = 900.0, rg: float = 880.0, n: int = 100_000,
                  seed: int = 1) -> dict:
    """fo, phi_hab (per primary, /cm^2) and CAL_phys = fo/phi_hab."""
    import numpy as np
    rng = np.random.default_rng(seed)
    vol = 2.0 / 3.0 * math.pi * ((rg + 2.0) ** 3 - rg ** 3)
    track = 0.0
    born_below = 0
    srcs = ring_layout(rings, azimuth)
    for th, ph in srcs:
        u = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        e1 = np.array([math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), -math.sin(th)])
        e2 = np.cross(u, e1)
        rr = spot * np.sqrt(rng.random(n))
        aa = 2.0 * math.pi * rng.random(n)
        p = beam_radius * u + np.outer(rr * np.cos(aa), e1) + np.outer(rr * np.sin(aa), e2)
        d = np.tile(-u, (n, 1))
        born_below += int((p[:, 2] < 0).sum())
        track += float((_len_in_ball_above_ground(p, d, rg + 2.0)
                        - _len_in_ball_above_ground(p, d, rg)).sum())
    ntot = n * len(srcs)
    fo = track / ntot / vol
    phi = 1.0 / (math.pi * spot ** 2)
    return {"fo": fo, "phi_hab": phi, "cal_phys": fo / phi,
            "born_below": born_below / ntot}


def stage2_check(rings: int, azimuth: int) -> float:
    """Predicted historical stage-2 factor (skin/fo at spot 500 over spot 900, 800
    gauge): skin ~ phi_hab, so the ratio is (phi500/fo500)/(phi900/fo900)."""
    a = geometric_cal(rings, azimuth, spot=500.0, rg=800.0, n=40_000)
    b = geometric_cal(rings, azimuth, spot=900.0, rg=800.0, n=40_000)
    return b["cal_phys"] / a["cal_phys"]


def cmd_geometric(args) -> None:
    from lunarsim import dosimetry
    from lunarsim.bridge import QUICK_LOOK, FULL_RUN
    cal = dosimetry.OUTER_GAUGE_ANCHOR_CAL
    print(f"shipped OUTER_GAUGE_ANCHOR_CAL = {cal:.5f}")
    for t in (QUICK_LOOK, FULL_RUN):
        g = geometric_cal(t.rings, t.azimuth, n=args.n)
        print(f"  {t.name:5s} {t.rings}x{t.azimuth}: fo={g['fo']:.5e}/cm2 "
              f"phi_hab={g['phi_hab']:.5e}/cm2  CAL_phys={g['cal_phys']:.4f}  "
              f"(born below ground {g['born_below']:.1%})")
    s2 = stage2_check(QUICK_LOOK.rings, QUICK_LOOK.azimuth)
    print(f"stage-2 sanity: predicted N500/N900 = {s2:.3f} vs measured 4.4836 "
          f"({s2 / 4.4836 - 1:+.1%})")
    g = geometric_cal(FULL_RUN.rings, FULL_RUN.azimuth, n=args.n)
    _verdict(g["cal_phys"], 0.0, cal, "geometric")


def _verdict(cal_phys: float, err: float, cal: float, tag: str) -> None:
    pred = cal / cal_phys                     # flood/kernel the normalisation alone predicts
    r, re = RESIDUAL
    after = r / pred
    after_err = math.hypot(re / pred, r * err / cal_phys / pred) if err else re / pred
    z = (after - 1.0) / after_err
    print(f"[{tag}] normalisation alone predicts flood/kernel = CAL/CAL_phys = {pred:.3f}")
    print(f"[{tag}] probe residual {r:.3f}+/-{re:.3f} -> after normalisation "
          f"{after:.3f}+/-{after_err:.3f}  (z {z:+.1f} sigma from 1)")


# ----------------------------------------------------------------------------
# PC measurement
# ----------------------------------------------------------------------------
def anchor_spec():
    """The CAL anchor: 3 m dome, 22 g/cm^2 Al (fits the fixed gauge, corr = 1)."""
    from lunarsim import HabitatSpec, WallLayer
    return HabitatSpec(name="gauge_check", shape="dome", inner_radius_cm=300.0,
                       walls=[WallLayer("aluminium", 22.0 / 2.699)])


def cmd_measure(args) -> None:
    from lunarsim import dosimetry
    from lunarsim.bridge import RunTier, run_design, DEFAULT_BEAM_SPOT_CM
    spec = anchor_spec()
    assert abs(dosimetry._gauge_corr(spec) - 1.0) < 1e-12, "anchor must not grow the gauge"
    tier = RunTier("gauge", rings=args.rings, azimuth=args.azimuth,
                   histories=args.histories)
    phi = 1.0 / (math.pi * DEFAULT_BEAM_SPOT_CM ** 2)
    cal = dosimetry.OUTER_GAUGE_ANCHOR_CAL
    g = geometric_cal(args.rings, args.azimuth, n=40_000)
    print(f"anchor {spec.name}: {spec.areal_density_gcm2():.1f} g/cm2, "
          f"{tier.total_primaries} primaries/run, geometric CAL_phys {g['cal_phys']:.4f}")
    species = [("H", "proton", 1, 1), ("He", "alpha", 2, 4),
               ("Fe", "GenericIon(26,56)", 26, 56)][: args.species]
    allv = []
    for name, particle, z, a in species:
        vals = []
        for s in range(1, args.seeds + 1):
            res = run_design(spec, tier, seed=s, keep=False,
                             particle=particle, ion_z=z, ion_a=a)
            if not res.fluence_outside:
                print(f"  {name} seed {s}: no fluence_outside (rc {res.returncode})")
                continue
            fo = res.fluence_outside * 100.0 / tier.total_primaries   # /mm2 -> /cm2, per primary
            vals.append(fo / phi)
            print(f"  {name} seed {s}: fo/primary {fo:.5e}/cm2  CAL_phys {fo / phi:.4f}",
                  flush=True)
        if vals:
            m = statistics.fmean(vals)
            e = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            print(f"{name}: CAL_phys = {m:.4f} +/- {e:.4f}  "
                  f"(vs geometric {g['cal_phys']:.4f}: {m / g['cal_phys'] - 1:+.1%} "
                  f"= albedo/secondaries of the primary Z/A)")
            allv.append((m, e))
    if allv:
        # Fe is the cleanest (no same-Z/A secondaries); H carries albedo protons.
        m, e = allv[-1]
        print(f"shipped CAL {cal:.5f}")
        _verdict(m, e, cal, f"measured {species[len(allv) - 1][0]}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometric", help="laptop ray-trace (no TOPAS)")
    g.add_argument("--n", type=int, default=100_000, help="rays per source")
    m = sub.add_parser("measure", help="PC: real flood gauge per species")
    m.add_argument("--seeds", type=int, default=4)
    m.add_argument("--rings", type=int, default=5)
    m.add_argument("--azimuth", type=int, default=8)
    m.add_argument("--histories", type=int, default=50)
    m.add_argument("--species", type=int, default=3, help="1=H, 2=+He, 3=+Fe")
    args = p.parse_args(argv)
    {"geometric": cmd_geometric, "measure": cmd_measure}[args.cmd](args)


if __name__ == "__main__":
    main()
