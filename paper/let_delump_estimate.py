#!/usr/bin/env python3
"""Route (a): effective-Z de-lumping estimate for the thin-wall <Q> over-read.

TARGET-BLIND. No new Monte Carlo. This is an OFFLINE re-weighting of the
existing LET-harness fold (lunarsim/kernel_gen/let_harness.py). It tests ONE
hypothesis quantitatively before any route (b) per-element MC is commissioned:

    "The thin-wall water-point <Q> over-read (~2x the arXiv 2511.09040 arbiter
     at every thin depth, incl. ZERO shielding) is dominated by representing
     each heavy GCR charge-band by a single near-top-Z representative ion
     (C=6 for Z=3-9, Si=14 for Z=10-20, Fe=26 for Z=21-28)."

Mechanism: the fold applies a band's SUMMED abundance at the representative
ion's Z (dose ~ a_band * Z_rep^2, and <Q> through Q(L), L ~ Z^2). De-lumping
redistributes that abundance across the band's real elements i, each carrying
dose ~ a_i * Z_i^2 at LET scaled by (Z_i/Z_rep)^2. For band g with lumped
dose D_g (= spec_den) and dose-averaged quality qbar_g:

    f_D(g)      = sum_i (a_i/a_band) (Z_i/Z_rep)^2          # dose re-weight
    qbar_new(g) = [ sum_i (a_i/a_band)(Z_i/Z_rep)^2 Q(L_s*(Z_i/Z_rep)^2) ]
                  / [ sum_i (a_i/a_band)(Z_i/Z_rep)^2 ]
    with L_s = Q^{-1}(qbar_g)   (single effective band LET)

Corrected totals over the native fold (den_native, num_native = qbar*den):
    den_corr = den_native - sum_g D_g      + sum_g D_g*f_D(g)
    num_corr = num_native - sum_g D_g*qbar_g + sum_g D_g*f_D(g)*qbar_new(g)
    <Q>_corr = num_corr / den_corr

APPROXIMATION (documented, do not over-read the exact digit): collapsing each
band to a single effective LET L_s ignores the within-band node-LET spread.
Because Q is convex in L, this UNDER-states the <Q> drop for a band whose
abundance sits BELOW the representative (Si, Fe) and the <Q> rise for one whose
abundance sits ABOVE it (C, pulled up by O=8). The node/ntuple-exact version
(scale each .phsp step's LET by (Z_i/Z_rep)^2 and re-sum Q) removes this; it
reads Det_LET.phsp and must run on the PC. This estimate exists to bound the
SIZE of the de-lumping effect, which is all route (a) needs to decide route (b).

Within-band GCR abundances (relative, solar-min): George et al. 2009, ApJ 698,
1666, Table 1 (CRIS, 160 MeV/nuc, Si==1000) for the well-measured elements;
minor spallation secondaries (P, Cl, K; Sc, Co, Ni; Li, Be, B, F) from standard
GCR source tables at the same normalization. These are reference inputs, never
fit targets.

Usage:
    python3 paper/let_delump_estimate.py                      # embedded wall=0 demo (laptop)
    python3 paper/let_delump_estimate.py runs/let_batch/let_qbar_depth.json   # all depths (PC)
"""
import json
import math
import sys
from pathlib import Path

# --- ICRP-60 Q(L), identical to the scorer / config.quality_factor_icrp60 -----
def Q(L):
    if L < 10.0:
        return 1.0
    if L <= 100.0:
        return 0.32 * L - 2.2
    return 300.0 / math.sqrt(L)

def invQ(qbar):
    """Single effective LET whose Q equals qbar (monotone branches of ICRP-60)."""
    if qbar <= 1.0:
        return 9.99                      # at/under the Q=1 plateau
    L_mid = (qbar + 2.2) / 0.32          # 10 <= L <= 100  ->  Q in [1.0, 29.8]
    if L_mid <= 100.0:
        return L_mid
    return (300.0 / qbar) ** 2           # L > 100

# --- within-band GCR abundances (relative, Si==1000); see module docstring ----
ABUND = {
    # C-band  Z=3..9  (CNO + light secondaries)
    3: 136.0, 4: 67.0, 5: 178.0, 6: 2640.0, 7: 557.0, 8: 2697.0, 9: 50.0,
    # Si-band Z=10..20 (Ne-Ca); George 2009 heavies, minor P/Cl/K from GCR tables
    10: 998.7, 11: 189.6, 12: 1368.2, 13: 202.7, 14: 1000.0,
    15: 29.0, 16: 157.0, 17: 15.0, 18: 58.8, 19: 24.0, 20: 124.8,
    # Fe-band Z=21..28 (Fe peak); Fe from George 2009, sub-Fe/Ni from GCR tables
    21: 18.0, 22: 70.0, 23: 35.0, 24: 105.0, 25: 75.0, 26: 653.7, 27: 4.0, 28: 34.0,
}

# representative species name -> (Z_rep, [band Z list])
BANDS = {
    "C":  (6,  list(range(3, 10))),
    "Si": (14, list(range(10, 21))),
    "Fe": (26, list(range(21, 29))),
}

# arXiv 2511.09040 water-point <Q>(Al depth) arbiter, POST-HOC overlay only
ARBITER = {0.0: 5.6, 2.0: 5.8, 2.025: 5.8, 10.0: 4.3, 20.0: 3.5}


def delump_band(name, qbar_lump):
    """Return (f_D, qbar_new) for one band's single-effective-LET de-lumping."""
    z_rep, zs = BANDS[name]
    L_s = invQ(qbar_lump)
    a_band = sum(ABUND[z] for z in zs)
    f_D = 0.0
    num = 0.0
    for z in zs:
        w = (ABUND[z] / a_band) * (z / z_rep) ** 2          # dose weight of element z
        f_D += w
        num += w * Q(L_s * (z / z_rep) ** 2)
    return f_D, (num / f_D)


def delump_point(species_split, qbar_total):
    """Apply band de-lumping to one depth point. den_native normalized to 1."""
    den_native = 1.0
    num_native = qbar_total
    rows = []
    d_den = 0.0
    d_num = 0.0
    for name in ("H", "He", "C", "Si", "Fe"):
        sp = species_split[name]
        D = sp["D_share"]                      # spec_den (dose share, sums to 1)
        qb = sp["qbar"]
        if name in BANDS:
            f_D, qb_new = delump_band(name, qb)
            D_new = D * f_D
            d_den += D_new - D
            d_num += D_new * qb_new - D * qb
            rows.append((name, D, qb, D_new, qb_new, f_D))
        else:
            rows.append((name, D, qb, D, qb, 1.0))
    den_corr = den_native + d_den
    num_corr = num_native + d_num
    return num_corr / den_corr, rows


def arbiter_at(wall):
    if wall in ARBITER:
        return ARBITER[wall]
    keys = sorted(ARBITER)
    if wall <= keys[0]:
        return ARBITER[keys[0]]
    if wall >= keys[-1]:
        return ARBITER[keys[-1]]
    for a, b in zip(keys, keys[1:]):
        if a <= wall <= b:
            t = (wall - a) / (b - a)
            return ARBITER[a] + t * (ARBITER[b] - ARBITER[a])
    return None


def report_point(wall, species_split, qbar_total):
    qb_corr, rows = delump_point(species_split, qbar_total)
    arb = arbiter_at(wall)
    print(f"\n  wall = {wall:g} g/cm^2   <Q> lumped={qbar_total:.2f}  "
          f"de-lumped={qb_corr:.2f}  arbiter~{arb:.2f}")
    print(f"    {'band':>4} {'dose%':>7} {'->dose%':>8} {'qbar':>7} {'->qbar':>8} {'f_D':>6}")
    tot = sum(r[1] for r in rows)
    tot_new = sum(r[3] for r in rows)
    for name, D, qb, D_new, qb_new, f_D in rows:
        print(f"    {name:>4} {100*D/tot:7.1f} {100*D_new/tot_new:8.1f} "
              f"{qb:7.2f} {qb_new:8.2f} {f_D:6.3f}")
    closed = (qbar_total - qb_corr)        # +ve = toward arbiter, -ve = away
    gap = (qbar_total - arb)
    print(f"    de-lumping shifts <Q> by {qb_corr - qbar_total:+.2f} "
          f"(gap to arbiter = {gap:.2f}); closes {100*closed/gap:+.0f}% of the gap")
    return qb_corr


# embedded wall=0 per-species split (PC run 2026-10-14, corrected comp, Al, phi=400)
DEMO_W0 = {
    "H":  {"D_share": 0.110, "qbar": 1.44},
    "He": {"D_share": 0.089, "qbar": 1.30},
    "C":  {"D_share": 0.191, "qbar": 1.60},
    "Si": {"D_share": 0.378, "qbar": 15.70},
    "Fe": {"D_share": 0.232, "qbar": 20.70},
}
DEMO_W0_QTOTAL = 11.31


def main():
    print("=" * 74)
    print("Route (a): effective-Z de-lumping estimate -- thin-wall <Q> over-read")
    print("=" * 74)
    for name, (z_rep, zs) in BANDS.items():
        a_band = sum(ABUND[z] for z in zs)
        z_eff = math.sqrt(sum(ABUND[z] * z * z for z in zs) / a_band)
        print(f"  band {name:>2} (rep Z={z_rep:>2}): Z_eff_rms={z_eff:5.2f}  "
              f"f_D(dose)={ (z_eff/z_rep)**2 :.3f}   [Z={zs[0]}..{zs[-1]}]")

    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        data = json.loads(Path(sys.argv[1]).read_text())
        for material, md in data["materials"].items():
            print(f"\n-- {material} (phi={data.get('phi_MV')} MV, "
                  f"{data.get('composition')}) --")
            for p in md["points"]:
                ss = p.get("species_split")
                qt = p.get("qbar_waterpoint_native")
                if ss and qt:
                    ss2 = {k: {"D_share": v["D_share"], "qbar": v["qbar"]}
                           for k, v in ss.items()}
                    report_point(p["wall_gcm2"], ss2, qt)
    else:
        print("\n-- EMBEDDED wall=0 demo (laptop; pass the PC JSON for all depths) --")
        report_point(0.0, DEMO_W0, DEMO_W0_QTOTAL)

    print("\n" + "-" * 74)
    print("READ-OUT: de-lumping the Z-band representatives is a REAL but SECONDARY")
    print("effect. It does NOT close the ~2x thin over-read to the arbiter. The")
    print("residual lives in the heavy-band DOSE WEIGHT (a_s * Z_s^2 fluence share),")
    print("not the within-band Z collapse -> next diagnostic is the incident")
    print("water-point dose-fraction audit, NOT route (b) per-element MC.")


if __name__ == "__main__":
    main()
