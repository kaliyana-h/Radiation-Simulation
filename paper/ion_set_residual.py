#!/usr/bin/env python3
"""Transported-ion-set residual: how much does each engine's ion REPRESENTATION
move the incident water-point dose and <Q>, vs carrying all 28 elements?
(laptop, NO MC, TARGET-BLIND -- tool vs itself only).

WHY
---
The thin-wall kernel transports 10 elements individually (config.PERELEMENT_IONS,
every other Z=1-28 Voronoi-assigned to its nearest carried ion), while the FLOOD
engine (thick walls, the regime most habitats sit in) is pinned by
dosimetry.flood_model_for to the 5-ion proton-clone baseline. Before spending PC
cores on extending either transported set, this sizes the INPUT-representation
error each one carries, at zero shielding where it is transport-free and exact
(the same logic as the PERELEM_RUNBOOK "cheap test first").

REPRESENTATIONS (each is exactly what the named engine feeds Geant4)
  FULL28   every Z=1..28 at its own Z, own Matthiae-2013 spectrum and abundance.
           The reference: the tool's own input model at full Z resolution.
  PEREL<n> carried ions only (default: the kernel's 10); each carries the summed
           abundance of its Voronoi cell, at its OWN Z and OWN single-element
           spectrum -- i.e. the 'perelement' kernel. --ions sets a what-if set.
  GROUP5   5 reps (H/He/C/Si/Fe), band-summed Matthiae spectrum and abundance at
           the rep Z -- the flood's optional 'matthia2013' model.
  CLONE5   5 reps, order-of-magnitude GCR_COMPOSITION abundances, proton-clone
           shape with Phi=(Z/A)*phi, on the FLOOD LIS form -- the shipped flood.

PHYSICS: clean analytic water LET (Z_eff^2 x NIST PSTAR, Barkas charge) and ICRP-60
Q(L), reused verbatim from paper/dose_fraction_audit.py. Every representation is
normalised to the SAME full-range proton flux, which is the tool's own absolute
contract (_calibration_factor pins the proton flux, never a dose). Matthiae W is
self-calibrated to phi on the KERNEL LIS form (as the per-element kernel is).

LIMITATION (do not over-read): zero shielding only. Behind a wall, fragmentation
redistributes heavy-ion dose to lighter fragments and protons/neutrons, so the
input-representation error is expected to SHRINK with depth -- but by how much is a
transport question. On the PC, LUNARSIM_GCR_PERELEMENT=matthia2013 python3 -m
lunarsim.anchor_crosscheck <outdir> measures it for the flood.

    python3 paper/ion_set_residual.py
    python3 paper/ion_set_residual.py --ions 1,2,6,7,8,10,12,13,14,16,20,24,26,28
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dose_fraction_audit import Q, let_water_kev_um          # noqa: E402
from lunarsim import dosimetry                                 # noqa: E402
from lunarsim.kernel_gen import config as kcfg                 # noqa: E402

E_LO = 10.0          # MeV/n -- make_source._MAT_EMIN (Matthiae validity floor)
NBINS = 400
OUT = Path(__file__).resolve().parent / "ion_set_residual.csv"
REP5 = (1, 2, 6, 14, 26)


def _grid(e_hi):
    return [E_LO * (e_hi / E_LO) ** (k / NBINS) for k in range(NBINS + 1)]


def _fold(species, e_hi):
    """species: list of (label, z_transported, flux_fn(e0, e1) -> bin flux in
    proton-full-range units). Returns per-species (D, H) and totals."""
    g = _grid(e_hi)
    rows = []
    for label, z, flux in species:
        D = H = 0.0
        for k in range(NBINS):
            e0, e1 = g[k], g[k + 1]
            let = let_water_kev_um(z, math.sqrt(e0 * e1))
            d = flux(e0, e1) * let
            D += d
            H += d * Q(let)
        rows.append((label, z, D, H))
    return rows


def _bin_integral(f, e0, e1, m=6):
    """Trapezoid of f over [e0, e1] on a small log sub-grid."""
    s = [e0 * (e1 / e0) ** (i / m) for i in range(m + 1)]
    return sum(0.5 * (f(s[i]) + f(s[i + 1])) * (s[i + 1] - s[i]) for i in range(m))


def matthiae_species(ms, phi, ions):
    """FULL28 (ions=None) or a Voronoi PEREL set, Matthiae per-element, on the
    kernel LIS form. Flux in units of the full-range Matthiae proton flux."""
    W = ms.matthia_W_for_phi(phi)
    e_hi = ms.GCR_EMAX_PER_NUC
    p_full = ms._mattia_element_integral(1, W, ms._MAT_EMIN, e_hi, 4000)
    ab = {z: ms.mattia_element_abundance(z, phi) for z in range(1, 29)}

    def shape(z):
        full = ms._mattia_element_integral(z, W, ms._MAT_EMIN, e_hi, 4000)
        return lambda e0, e1: _bin_integral(lambda e: ms.mattia_flux(z, e, W), e0, e1) / full

    if ions is None:
        cells = {z: [z] for z in range(1, 29)}
    else:
        cells = {z: [] for z in ions}
        for zz in range(1, 29):   # same rule as config._perelement_assignment
            cells[min(ions, key=lambda iz: (abs(iz - zz), -iz))].append(zz)
    out = []
    for z, members in cells.items():
        a_cell = sum(ab[m] for m in members)
        sh = shape(z)
        out.append((f"Z{z}" + (f"[{members[0]}-{members[-1]}]" if len(members) > 1 else ""),
                    z, (lambda s, a: lambda e0, e1: a * s(e0, e1))(sh, a_cell)))
    del p_full
    return out, e_hi


def group5_species(ms, phi):
    """GROUP5: band-summed Matthiae shape + abundance at the rep Z."""
    W = ms.matthia_W_for_phi(phi)
    e_hi = ms.GCR_EMAX_PER_NUC
    out = []
    for z in REP5:
        full = ms._mattia_group_integral(z, W, ms._MAT_EMIN, e_hi, 4000)
        a = ms.mattia_group_abundance(z, phi)
        f = (lambda zz, fu, aa: lambda e0, e1: aa * _bin_integral(
            lambda e: ms.mattia_group_flux(zz, e, W), e0, e1) / fu)(z, full, a)
        out.append((f"Z{z}(band)", z, f))
    return out, e_hi


def clone5_species(phi):
    """CLONE5: the shipped flood -- GCR_COMPOSITION, proton-clone, FLOOD LIS form."""
    dosimetry.set_lis_form(dosimetry.FLOOD_LIS_FORM)
    ms = dosimetry._load_make_source()
    e_hi = ms.GCR_EMAX_PER_NUC
    p_full = dosimetry._raw_gcr_integral(1, 1, phi, 10.0, e_hi, n=4000)
    out = []
    for name, _p, z, a, ab, _g in dosimetry.GCR_COMPOSITION:
        full = dosimetry._raw_gcr_integral(z, a, phi, 10.0, e_hi, n=4000)
        f = (lambda zz, aa, ab_, fu: lambda e0, e1: ab_ * dosimetry._raw_gcr_integral(
            zz, aa, phi, e0, e1, n=8) / fu)(z, a, ab, full)
        out.append((name, z, f))
    del p_full
    return out, e_hi


def summarise(name, rows):
    D = sum(r[2] for r in rows)
    H = sum(r[3] for r in rows)
    hze = sum(r[2] for r in rows if r[1] >= 3)
    return {"representation": name, "D_rel": D, "H_rel": H, "Qbar": H / D,
            "hze_dose_frac": hze / D, "n_ions": len(rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phi", type=float, default=400.0)
    ap.add_argument("--ions", default=None,
                    help="comma Z list for a what-if PEREL set (default: kernel's 10)")
    ap.add_argument("--csv", default=str(OUT))
    args = ap.parse_args(argv)

    kern_ions = sorted(z for _, z, _ in kcfg.PERELEMENT_IONS)
    what_if = sorted(int(x) for x in args.ions.split(",")) if args.ions else None
    if what_if and (1 not in what_if or 2 not in what_if):
        ap.error("--ions must include 1 and 2")

    dosimetry.set_perelement_model("none")          # we call the shape fns directly
    dosimetry.set_lis_form(dosimetry.KERNEL_LIS_FORM)
    ms = dosimetry._load_make_source()

    reps = []
    sp, eh = matthiae_species(ms, args.phi, None)
    full = _fold(sp, eh)
    reps.append(("FULL28", full))
    sp, eh = matthiae_species(ms, args.phi, kern_ions)
    reps.append((f"PEREL{len(kern_ions)} (kernel)", _fold(sp, eh)))
    if what_if:
        sp, eh = matthiae_species(ms, args.phi, what_if)
        reps.append((f"PEREL{len(what_if)} (what-if)", _fold(sp, eh)))
    sp, eh = group5_species(ms, args.phi)
    reps.append(("GROUP5 (flood matthia2013)", _fold(sp, eh)))
    sp, eh = clone5_species(args.phi)                # switches to flood LIS form
    reps.append(("CLONE5 (shipped flood)", _fold(sp, eh)))
    dosimetry.set_lis_form(dosimetry.KERNEL_LIS_FORM)

    ref = summarise("FULL28", full)
    print(f"Zero-shield water point, phi={args.phi:g} MV, per unit proton flux "
          f"(transport-free; tool vs itself)\n")
    print(f"{'representation':28s} {'ions':>4s} {'D/D28':>7s} {'H/H28':>7s} "
          f"{'<Q>':>6s} {'<Q>/28':>7s} {'HZE dose%':>9s}")
    summ = []
    for name, rows in reps:
        s = summarise(name, rows)
        s["D_over_full"] = s["D_rel"] / ref["D_rel"]
        s["H_over_full"] = s["H_rel"] / ref["H_rel"]
        s["Q_over_full"] = s["Qbar"] / ref["Qbar"]
        summ.append(s)
        print(f"{name:28s} {s['n_ions']:4d} {s['D_over_full']:7.3f} "
              f"{s['H_over_full']:7.3f} {s['Qbar']:6.2f} {s['Q_over_full']:7.3f} "
              f"{100 * s['hze_dose_frac']:8.1f}%")

    # where the kernel's residual lives: per-cell H error vs the FULL28 members
    print(f"\nKernel Voronoi residual by carried ion (H_cell - H_members) / H28:")
    by_z = {r[1]: r for r in full}
    W_rows = reps[1][1]
    cells = {z: [] for z in kern_ions}
    for zz in range(1, 29):
        cells[min(kern_ions, key=lambda iz: (abs(iz - zz), -iz))].append(zz)
    for label, z, D, H in W_rows:
        Hm = sum(by_z[m][3] for m in cells[z])
        print(f"  {label:14s} {100 * (H - Hm) / ref['H_rel']:+6.2f} %")

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        for s in summ:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()})
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
