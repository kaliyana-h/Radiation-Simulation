#!/usr/bin/env python3
"""Matthiae-2013 (DLR modified-ISO) per-element GCR fold -- the force-field-
COMPATIBLE physics-derived fix for the thin-wall <Q> over-read (laptop, NO MC).

WHY THIS MODEL (and not BON2020)
--------------------------------
paper/bon2020_lis_fold.py showed that the Badhwar-O'Neill 2020 LIS folded
through the tool's force-field modulation FAILS the gate and is essentially
unchanged from the proton-clone (<Q>(0)=11.56). Root cause: BON2020's delta is
co-fit to its numerical Fokker-Planck transport, so its LIS at low E is a
NON-PHYSICAL heliopause boundary condition (He/H_LIS=2.3 at 100 MeV/n). A
force-field energy-shift cannot substitute for the FP solve => the hybrid is
physically inconsistent, and its failure is a method artifact, not a result.

The tool's own proton LIS is Burger(2000)/Usoskin(2005) -- a RIGIDITY
parametrization DESIGNED to be used with the force-field modulation
(make_source.lis_proton). The whole tool machinery assumes a force-field-
COMPATIBLE LIS. Matthiae et al. 2013 ("A ready-to-use galactic cosmic ray
model", Adv. Space Res. 51(3) 329-338, doi:10.1016/j.asr.2012.09.022) is the
canonical such model: an analytic per-element (Z=1-28) spectrum controlled by a
SINGLE solar-modulation parameter W (0-200), built as the fast analytic
alternative to BON's FP solve and calibrated against BON + measurements. It
drops into the existing machinery exactly like lis_proton does.

MODEL (verbatim from the reference implementation ssc-maire/CosRayModifiedISO,
internalFunctions/pythonModifiedISO.py; the paper's Eq. set):
    x        = 1e-3 * E     [GeV/n]          (E = kinetic energy MeV/n)
    m        = 0.938 (Z=1) else 0.939        [GeV/n, per-nucleon rest mass]
    R        = (A/Z)*sqrt(x(x+2m))           [GV]  rigidity
    beta     = sqrt(x(x+2m))/(x+m)
    R0       = 0.37 + 0.0003*W^1.45
    delta    = 0.02*W + 4.7
    phi      = C * beta^alpha / R^gamma * (R/(R+R0))^delta
    flux     = 1e-4 * phi * (A/Z) * 1e-3 / beta
               [particles cm^-2 s^-1 sr^-1 (MeV/n)^-1]
Per-element constants (A, C, gamma, alpha) for Z=1-28 are the model's own
tabulated values (abundance is baked into C). W is the ONLY knob; R0 and delta
carry all the solar modulation, so this LIS is force-field-native by design.

TARGET-BLIND: the Matthiae constants are fit to GCR flux measurements, never to
any dose. The Hong 2019 split and arXiv 2511.09040 <Q>(0)=5.6 arbiter are used
ONLY as a POST-HOC gate. W is self-calibrated to the tool's own phi=400 MV
proton spectrum (a flux<->flux match), NOT tuned to any dose target.

Reuses the CLEAN analytic water LET + ICRP-60 Q from dose_fraction_audit, so the
ONLY thing that differs from the audit/BON baseline is the incident spectrum.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dose_fraction_audit as audit
from dose_fraction_audit import (Q, let_water_kev_um, beta_of, A_OF,
                                  REF_BAND, ARBITER_Q0)

E_LO = 11.0     # per-nucleon MeV; model validity is E>=10, code floors at 11
E_HI = 20000.0  # 20 GeV/nuc, matches make_source.GCR_EMAX_PER_NUC

ELEMENT_SYM = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni",
}

# --------------------------------------------------------------------------
# Matthiae-2013 per-element constants (Z-ordered, index Z-1), VERBATIM from
# ssc-maire/CosRayModifiedISO internalFunctions/pythonModifiedISO.py.
# --------------------------------------------------------------------------
AI = [1.0, 4.0, 6.9, 9.0, 10.8, 12.0, 14.0, 16.0, 19.0, 20.2,
      23.0, 24.3, 27.0, 28.1, 31.0, 32.1, 35.4, 39.9, 39.1, 40.1,
      44.9, 47.9, 50.9, 52.0, 54.9, 55.8, 58.9, 58.7]
CI = [1.85e4, 3.69e3, 19.5, 17.7, 49.2, 103.0, 36.7, 87.4, 3.19, 16.4,
      4.43, 19.3, 4.17, 13.4, 1.15, 3.06, 1.30, 2.33, 1.87, 2.17,
      0.74, 2.63, 1.23, 2.12, 1.14, 9.32, 0.10, 0.48]
GAMMAI = [2.74, 2.77, 2.82, 3.05, 2.96, 2.76, 2.89, 2.70, 2.82, 2.76,
          2.84, 2.70, 2.77, 2.66, 2.89, 2.71, 3.00, 2.93, 3.05, 2.77,
          2.97, 2.99, 2.94, 2.89, 2.74, 2.63, 2.63, 2.63]
ALPHAI = [2.85, 3.12, 3.41, 4.30, 3.93, 3.18, 3.77, 3.11, 4.05, 3.11,
          3.14, 3.65, 3.46, 3.00, 4.04, 3.30, 4.40, 4.33, 4.49, 2.93,
          3.78, 3.79, 3.50, 3.28, 3.29, 3.01, 4.25, 3.52]


def mattia_flux(z, e_mevn, W):
    """Matthiae-2013 differential flux [cm^-2 s^-1 sr^-1 (MeV/n)^-1] at per-nucleon
    kinetic energy e_mevn (MeV/n), solar modulation W (0-200)."""
    i = z - 1
    A = AI[i]
    m = 0.938 if z == 1 else 0.939          # GeV/n
    c = CI[i]
    alpha = ALPHAI[i]
    gamm = GAMMAI[i]
    x = 1e-3 * e_mevn                        # GeV/n
    s = math.sqrt(x * (x + 2 * m))
    rigidity = (A / z) * s                   # GV
    beta = s / (x + m)
    R0 = 0.37 + 0.0003 * (W ** 1.45)
    delta = 0.02 * W + 4.7
    phi = c * (beta ** alpha) / (rigidity ** gamm) * (rigidity / (rigidity + R0)) ** delta
    return 1e-4 * phi * (A / z) * 1e-3 / beta


def mattia_dose_integral(z, W, nbins):
    """(flux, D, DQ, Emean) for element z folded through the clean water LET."""
    flux = D = DQ = D_logE = 0.0
    for k in range(nbins):
        e0 = E_LO * (E_HI / E_LO) ** (k / nbins)
        e1 = E_LO * (E_HI / E_LO) ** ((k + 1) / nbins)
        sub = 8
        fbin = 0.0
        tp = jp = None
        for s in range(sub + 1):
            t = e0 * (e1 / e0) ** (s / sub)
            j = mattia_flux(z, t, W)
            if tp is not None:
                fbin += 0.5 * (j + jp) * (t - tp)
            tp, jp = t, j
        emid = math.sqrt(e0 * e1)
        let = let_water_kev_um(z, emid)
        d = fbin * let
        flux += fbin
        D += d
        DQ += d * Q(let)
        D_logE += d * math.log(emid)
    return flux, D, DQ, (math.exp(D_logE / D) if D > 0 else 0.0)


def calibrate_W_to_phi(phi_mv):
    """Self-calibrate W by matching the Matthiae proton spectral SHAPE to the
    tool's own Burger/Usoskin + force-field proton spectrum at phi (flux<->flux,
    NOT a dose target). Returns the best-fit W in [0,200]."""
    from lunarsim import dosimetry
    ms = dosimetry._load_make_source()
    dosimetry.set_lis_form(dosimetry.KERNEL_LIS_FORM)
    E0 = 938.272
    es = [100.0 * (10000.0 / 100.0) ** (k / 20) for k in range(21)]  # 100 MeV..10 GeV
    # tool proton differential (normalised to its own mean-log)
    def tool_p(e):
        Tm = e + phi_mv                       # proton Phi=(1/1)*phi
        factor = (e * (e + 2 * E0)) / (Tm * (Tm + 2 * E0))
        return ms.lis_proton(Tm / 1000.0) * factor
    tp = [tool_p(e) for e in es]
    ltp = [math.log(v) for v in tp]
    mtp = sum(ltp) / len(ltp)
    best = (1e99, None)
    W = 0.0
    while W <= 200.0:
        lm = [math.log(mattia_flux(1, e, W)) for e in es]
        mm = sum(lm) / len(lm)
        err = sum(((lm[i] - mm) - (ltp[i] - mtp)) ** 2 for i in range(len(es)))
        if err < best[0]:
            best = (err, W)
        W += 1.0
    return best[1]


def band_of(z):
    return "H" if z == 1 else ("He" if z == 2 else "Z>=3 (all HZE)")


def run_gate(W, nbins, show_table=False):
    rows = []
    for z in range(1, 29):
        flux, D, DQ, Em = mattia_dose_integral(z, W, nbins)
        rows.append({"z": z, "sym": ELEMENT_SYM[z], "flux": flux, "D": D,
                     "DQ": DQ, "Em": Em, "qbar": DQ / D if D > 0 else 0.0})
    flux_H = next(r["flux"] for r in rows if r["z"] == 1)
    for r in rows:
        r["ab_relH"] = r["flux"] / flux_H
    Dtot = sum(r["D"] for r in rows)
    DQtot = sum(r["DQ"] for r in rows)
    qbar0 = DQtot / Dtot
    split = {k: (sum(r["D"] for r in rows if band_of(r["z"]) == k) / Dtot,
                 sum(r["DQ"] for r in rows if band_of(r["z"]) == k) / DQtot)
             for k in ("H", "He", "Z>=3 (all HZE)")}
    He_over_H = (sum(r["D"] for r in rows if r["z"] == 2) /
                 sum(r["D"] for r in rows if r["z"] == 1))

    if show_table:
        print("\n--- PER-ELEMENT MATTHIAE-2013 FOLD (sorted by absorbed dose) ---")
        print(f"  {'Z':>2} {'el':>3} {'C':>8} {'gam':>5} {'alp':>5} "
              f"{'ab/H':>10} {'D%':>6} {'DQ%':>6} {'<Q>':>6} {'Emean':>7}")
        for r in sorted(rows, key=lambda x: -x["D"]):
            i = r["z"] - 1
            print(f"  {r['z']:>2} {r['sym']:>3} {CI[i]:>8.3g} {GAMMAI[i]:>5.2f} "
                  f"{ALPHAI[i]:>5.2f} {r['ab_relH']:>10.3e} {100*r['D']/Dtot:>6.2f} "
                  f"{100*r['DQ']/DQtot:>6.2f} {r['qbar']:>6.2f} {r['Em']:>7.0f}")
    return {"rows": rows, "Dtot": Dtot, "DQtot": DQtot, "qbar0": qbar0,
            "split": split, "He_over_H": He_over_H}


def print_gate(res):
    split, qbar0, He_over_H = res["split"], res["qbar0"], res["He_over_H"]
    print("\n--- ACCEPTANCE GATE (POST-HOC references, NOT fit targets) ---")
    print(f"  {'band':>16} {'Mat D%':>8} {'Hong D%':>12}   {'Mat DQ%':>8} {'Hong DQ%':>12}")
    gate_split = True
    for k in ("H", "He", "Z>=3 (all HZE)"):
        dlo, dhi = REF_BAND[k]["dose"]
        qlo, qhi = REF_BAND[k]["deq"]
        bd, bq = split[k]
        ok = dlo <= bd <= dhi
        gate_split = gate_split and ok
        print(f"  {k:>16} {100*bd:>7.1f}% {100*dlo:>5.0f}-{100*dhi:<4.0f}% "
              f"{'OK' if ok else '--':>2}   {100*bq:>7.1f}% {100*qlo:>5.0f}-{100*qhi:<4.0f}%")
    gate_He = abs(He_over_H - 0.43) <= 0.10
    gate_Q = abs(qbar0 - 5.6) / 5.6 <= 0.20
    print(f"\n  He/H absorbed-dose ratio : Mat {He_over_H:.3f}   vs measured 0.43 (Hong 2019)")
    print(f"  water-point <Q>(0)       : Mat {qbar0:.2f}    vs arbiter 5.6 (arXiv 2511.09040)")
    print(f"  (BON2020+ff hybrid 11.56 / proton-clone 11.31 both FAILED)")
    verdict = "PASS" if (gate_He and gate_Q and gate_split) else "FAIL"
    print(f"\n  GATE: He/H {'ok' if gate_He else 'X'} | <Q>(0) {'ok' if gate_Q else 'X'}"
          f" | split {'ok' if gate_split else 'X'}  =>  {verdict}")
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phi", type=float, default=400.0,
                    help="tool modulation phi (MV) to self-calibrate W against")
    ap.add_argument("--W", type=float, default=None,
                    help="force a specific W (0-200) instead of phi-calibrating")
    ap.add_argument("--nbins", type=int, default=400)
    ap.add_argument("--sweep", action="store_true", help="also sweep W=0..120")
    ap.add_argument("--emit-json", action="store_true")
    args = ap.parse_args()

    if args.W is not None:
        W = args.W
        wsrc = f"forced W={W:.0f}"
    else:
        W = calibrate_W_to_phi(args.phi)
        wsrc = f"W={W:.0f} self-calibrated to tool phi={args.phi:.0f} MV proton spectrum"

    print("=" * 78)
    print("MATTHIAE-2013 (DLR modified-ISO) PER-ELEMENT FOLD")
    print(f"  {wsrc}   nbins={args.nbins}")
    print("  LIS: Matthiae et al. 2013 Adv.Space Res. 51(3) 329 (ssc-maire impl., verbatim)")
    print("  modulation: single W knob (R0,delta) -- force-field-NATIVE by design")
    print("  LET/Q: clean analytic water, NIST PSTAR x Barkas Z_eff^2, ICRP-60 Q")
    print("=" * 78)

    res = run_gate(W, args.nbins, show_table=True)
    verdict = print_gate(res)

    if args.sweep:
        print("\n--- W SWEEP (dose split & <Q>(0) vs solar modulation) ---")
        print(f"  {'W':>4} {'H%':>6} {'He%':>6} {'Z>=3%':>6} {'He/H':>6} {'<Q>0':>6}")
        for Wx in (0, 20, 40, 60, 80, 100, 120):
            r = run_gate(float(Wx), args.nbins)
            print(f"  {Wx:>4} {100*r['split']['H'][0]:>6.1f} "
                  f"{100*r['split']['He'][0]:>6.1f} "
                  f"{100*r['split']['Z>=3 (all HZE)'][0]:>6.1f} "
                  f"{r['He_over_H']:>6.3f} {r['qbar0']:>6.2f}")

    rows = res["rows"]
    c_sum = sum(r["ab_relH"] for r in rows if 3 <= r["z"] <= 9)
    si_sum = sum(r["ab_relH"] for r in rows if 10 <= r["z"] <= 20)
    fe_sum = sum(r["ab_relH"] for r in rows if 21 <= r["z"] <= 28)
    he_ab = next(r["ab_relH"] for r in rows if r["z"] == 2)
    print("\n--- LUMPED abundances rel H (Matthiae flux-integrated) ---")
    print(f"  H  (Z=1)       1.0")
    print(f"  He (Z=2)       {he_ab:.4f}")
    print(f"  C  (Z=3-9)     {c_sum:.4f}   [current tool: 0.033]")
    print(f"  Si (Z=10-20)   {si_sum:.4f}   [current tool: 0.012]")
    print(f"  Fe (Z=21-28)   {fe_sum:.4f}   [current tool: 0.0020]")
    print(f"  Z>=3 total     {c_sum + si_sum + fe_sum:.4f}   [current tool 0.047]")

    if args.emit_json:
        out = {
            "_provenance": {
                "lis_model": "Matthiae et al. 2013 (DLR modified-ISO)",
                "lis_source": "Adv. Space Res. 51(3) 329-338, doi:10.1016/j.asr.2012.09.022; "
                              "impl ssc-maire/CosRayModifiedISO (verbatim)",
                "modulation": "single solar parameter W (0-200), force-field-native",
                "W": W, "W_source": wsrc, "nbins": args.nbins,
                "target_blind": "constants fit to GCR flux; Hong 2019 / arXiv 2511.09040 "
                                "used as POST-HOC gate only; W self-calibrated to tool "
                                "phi proton spectrum (flux<->flux), not to any dose",
                "gate_verdict": verdict,
            },
            "qbar0": res["qbar0"], "He_over_H": res["He_over_H"],
            "dose_split": {k: res["split"][k][0] for k in res["split"]},
            "deq_split": {k: res["split"][k][1] for k in res["split"]},
            "lumped_rel_H": {"H": 1.0, "He": he_ab, "C_3_9": c_sum,
                             "Si_10_20": si_sum, "Fe_21_28": fe_sum},
            "per_element": [
                {"z": r["z"], "sym": r["sym"], "ab_relH": r["ab_relH"],
                 "D_frac": r["D"] / res["Dtot"], "DQ_frac": r["DQ"] / res["DQtot"],
                 "qbar": r["qbar"]} for r in rows],
        }
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mattia2013_composition.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
