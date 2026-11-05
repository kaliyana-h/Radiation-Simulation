#!/usr/bin/env python3
"""Step-3 harness: decompose the kernel/flood gap at the ~19 g/cm2 engine gate.

The shipped shell-geometry thin-wall kernel reads ~0.72x (lining/skin) to ~0.78x
(phantom) of the same-depth Al flood MC at 19-50 g/cm2 (memory
[[crossover-discontinuity]]). The kernel's MEASURED anchors are only 0, 2.025, 10
and 50 g/cm2, so 50 is an anchor-to-flood comparison (not an interpolation
artifact), and the pending flood rows at 5 and 10 g/cm2 give a second anchor and
the thin-side shape.

This script separates the gap into factors, multiplied in order:

  flood / K_shipped  =  [K_lis / K_shipped]        LIS form   (rigidity -> energy)
                      x [K_in / K_lis]             composition (10-ion perelement ->
                                                   5-ion matthia2013, i.e. the flood's
                                                   own input)
                      x [flood / K_in]             RESIDUAL: transport / geometry /
                                                   scoring (needs MC to split further)

K_in is the kernel folded on EXACTLY the flood's input model (energy LIS, the
5-ion matthia2013 group spectra and abundances), so the residual has no
input-spectrum content left. All of this is laptop arithmetic: no MC, no fit, and
target-blind, because no reference dose enters. OLTARIS is not read here.

Two comparator pairings are reported, because the two engines score different
objects:
  phantom : flood = 20 cm solid water sphere, whole-volume (mass-average) H.
            The kernel = mass-average over its 15 cm, 5-shell phantom. The kernel's
            wT-weighted E (the GUI headline) is also printed for reference.
  skin    : flood = CrewSkin lining on the wall's inner face. The kernel = its
            skin shell (14.5-15 cm), which also touches the wall.

How to read the residual against depth (once the 5 and 10 g/cm2 flood rows exist):
  flat     -> a depth-independent factor: normalisation, the regolith-floor
              albedo the kernel lacks (no ground), or phantom size / illumination.
  varying  -> transport-geometry: hugging shell vs a 7.5 m dome (secondary
              build-up, obliquity), or the anchor interpolation.

Usage:
  python3 paper/gate_ratio_decompose.py                 # table + CSV
  python3 paper/gate_ratio_decompose.py --depths 5 10 19 30 50
Writes paper/gate_ratio_decompose.csv (provenance-stamped).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from lunarsim import dosimetry                     # noqa: E402
from lunarsim.spec import HabitatSpec, WallLayer   # noqa: E402

FLOOD_CSV = HERE / "accuracy_al_flood.csv"
OUT_CSV = HERE / "gate_ratio_decompose.csv"
AL_DENSITY = 2.70
DOME_INNER_R_CM = 750.0          # same dome as accuracy_vs_depth / the flood sweep
DEFAULT_DEPTHS = (2.025, 5.0, 10.0, 19.0, 25.0, 30.0, 40.0, 50.0)
SV_S_TO_MSV_YR = 86400.0 * 365.0 * 1.0e3

# The three kernel configurations, in decomposition order.
CONFIGS = (
    ("shipped", "rigidity", "perelement"),     # what the GUI folds today
    ("lis",     "energy",   "perelement"),     # swap LIS form only
    ("input",   "energy",   "matthia2013"),    # flood-input-matched (5-ion kernel)
)


@contextlib.contextmanager
def kernel_lis_form(form: str):
    """Fold the kernel on `form` instead of KERNEL_LIS_FORM. fold_gcr_thinwall
    reads the module constant at call time. This harness is single-threaded, so
    a scoped patch is safe; production code never does this."""
    saved = dosimetry.KERNEL_LIS_FORM
    dosimetry.KERNEL_LIS_FORM = form
    try:
        yield
    finally:
        dosimetry.KERNEL_LIS_FORM = saved


def shell_mass_fractions(shells) -> dict:
    """Mass fraction of each phantom shell (uniform water, so it is the volume fraction)."""
    vol = {k: r1 ** 3 - r0 ** 3 for k, r0, r1 in shells}
    tot = sum(vol.values())
    return {k: v / tot for k, v in vol.items()}


def kernel_point(gcm2: float, lis_form: str, perelement: str) -> dict:
    """Fold the kernel on a 750 cm Al dome of this areal density under one model.
    Returns the effective E, the mass-average phantom H, the skin-shell H (mSv/yr),
    the folded flux and the kernel file used."""
    spec = HabitatSpec(name=f"al{gcm2:g}", shape="dome", inner_radius_cm=DOME_INNER_R_CM,
                       walls=[WallLayer("aluminium", gcm2 / AL_DENSITY)])
    model = dataclasses.replace(dosimetry.current_model(), perelement=perelement)
    with dosimetry.use_model(model), kernel_lis_form(lis_form):
        f = dosimetry.fold_gcr_thinwall(spec)
    K = (dosimetry._load_gcr_thinwall_kernel_perelem()
         if f["kernel_path"] == dosimetry._GCR_THINWALL_KERNEL_PERELEM
         else dosimetry._load_gcr_thinwall_kernel("aluminium"))
    mf = shell_mass_fractions(K["meta"]["phantom_shells_cm"])
    rows = {r["shell"]: r for r in f["rows"]}
    H_mass = sum(mf[k] * rows[k]["doseeq_sv_s"] for k in mf)
    E = f["E_sv_s"]
    return {"E": E * SV_S_TO_MSV_YR,
            "E_rel": (f["sem_E_sv_s"] / E) if E else 0.0,
            "H_mass": H_mass * SV_S_TO_MSV_YR,
            "H_skin": rows["skin"]["doseeq_sv_s"] * SV_S_TO_MSV_YR,
            "flux": f["flux_cm2_s"],
            "kernel": f["kernel_path"].name,
            "model": f["model"]}


_REL = re.compile(r"phantom rel_err\s*([\d.]+)%.*?skin(?: rel_err)?\s*([\d.]+)%")


def load_flood(path: Path = FLOOD_CSV) -> dict:
    """{areal_gcm2: {phantom, skin, phantom_rel, skin_rel, stale}} for filled rows.
    stale lists the provenance mismatches against the current flood model."""
    out = {}
    if not path.exists():
        return out
    want = dosimetry.current_model().flood()
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            eff = (r.get("flood_eff_mSv_yr") or "").strip()
            if not eff:
                continue
            m = _REL.search(r.get("note") or "")
            out[float(r["areal_gcm2"])] = {
                "phantom": float(eff),
                "skin": float(r["flood_skin_mSv_yr"]),
                "phantom_rel": float(m.group(1)) / 100 if m else None,
                "skin_rel": float(m.group(2)) / 100 if m else None,
                "stale": dosimetry.provenance_mismatch(r.get("provenance", ""), want),
            }
    return out


def free_field_flux() -> dict:
    """Upper-hemisphere scalar fluence (protons, and all species) under the kernel's
    shipped model vs the flood model. If the two engines disagree on the incident
    field, this is where it shows."""
    out = {}
    base = dosimetry.current_model()
    for tag, model in (("kernel", dataclasses.replace(base, lis_form=dosimetry.KERNEL_LIS_FORM)),
                       ("flood", base.flood())):
        with dosimetry.use_model(model):
            p = dosimetry.gcr_scalar_fluence_rate(400.0)
            comp = dosimetry.matthia_composition(400.0)
            tot = sum(dosimetry.gcr_species_fluence_rate(z, a, ab, 400.0)
                      for _n, _p, z, a, ab, _g in comp)
        out[tag] = {"proton": p, "all5": tot, "model": model.label()}
    return out


def decompose(depths, flood: dict) -> list:
    rows = []
    for g in depths:
        k = {tag: kernel_point(g, lis, pe) for tag, lis, pe in CONFIGS}
        row = {"areal_gcm2": g,
               "K_E_shipped": k["shipped"]["E"],
               "K_Hmass_shipped": k["shipped"]["H_mass"],
               "K_Hskin_shipped": k["shipped"]["H_skin"],
               "K_Hmass_input": k["input"]["H_mass"],
               "K_Hskin_input": k["input"]["H_skin"],
               "f_lis_phantom": k["lis"]["H_mass"] / k["shipped"]["H_mass"],
               "f_comp_phantom": k["input"]["H_mass"] / k["lis"]["H_mass"],
               "f_lis_skin": k["lis"]["H_skin"] / k["shipped"]["H_skin"],
               "f_comp_skin": k["input"]["H_skin"] / k["lis"]["H_skin"],
               "kernel_shipped": k["shipped"]["kernel"],
               "kernel_input": k["input"]["kernel"]}
        fl = flood.get(g)
        if fl:
            row.update({
                "flood_phantom": fl["phantom"], "flood_skin": fl["skin"],
                "gap_phantom": fl["phantom"] / k["shipped"]["H_mass"],
                "gap_skin": fl["skin"] / k["shipped"]["H_skin"],
                "resid_phantom": fl["phantom"] / k["input"]["H_mass"],
                "resid_skin": fl["skin"] / k["input"]["H_skin"],
                "resid_phantom_err": (fl["phantom_rel"] or 0.0) * fl["phantom"] / k["input"]["H_mass"],
                "resid_skin_err": (fl["skin_rel"] or 0.0) * fl["skin"] / k["input"]["H_skin"],
                "flood_stale": ";".join(fl["stale"])})
        rows.append(row)
    return rows


def log_slope(rows, key: str):
    """d ln(residual) / d ln(depth) by least squares over rows with a residual, with
    the flood rel_err as weights. None if fewer than 2 points. Near 0 = flat."""
    pts = [(math.log(r["areal_gcm2"]), math.log(r[key]),
            max(r[key + "_err"] / r[key], 1e-3)) for r in rows if r.get(key)]
    if len(pts) < 2:
        return None
    w = [1 / s ** 2 for _x, _y, s in pts]
    sw = sum(w)
    mx = sum(wi * x for wi, (x, _y, _s) in zip(w, pts)) / sw
    my = sum(wi * y for wi, (_x, y, _s) in zip(w, pts)) / sw
    sxx = sum(wi * (x - mx) ** 2 for wi, (x, _y, _s) in zip(w, pts))
    sxy = sum(wi * (x - mx) * (y - my) for wi, (x, y, _s) in zip(w, pts))
    if sxx == 0:
        return None
    return sxy / sxx, math.sqrt(1 / sxx)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depths", type=float, nargs="+", default=list(DEFAULT_DEPTHS))
    ap.add_argument("--flood-csv", type=Path, default=FLOOD_CSV)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args(argv)

    flood = load_flood(args.flood_csv)
    depths = sorted(set(args.depths) | set(flood))
    rows = decompose(depths, flood)

    ff = free_field_flux()
    print("Free-field incident flux (phi=400 MV, /cm2/s, upper hemisphere):")
    for tag, v in ff.items():
        print(f"  {tag:6s} protons {v['proton']:.4f}  5-ion total {v['all5']:.4f}   [{v['model']}]")
    print(f"  flood/kernel: protons {ff['flood']['proton']/ff['kernel']['proton']:.3f}, "
          f"total {ff['flood']['all5']/ff['kernel']['all5']:.3f}")

    print("\nKernel (mSv/yr; 750 cm Al dome; anchors 0/2.025/10/50 g/cm2, log-log between):")
    print(f"{'g/cm2':>6} {'E_wT':>7} {'H_mass':>7} {'H_skin':>7} | "
          f"{'LIS ph':>6} {'comp ph':>7} {'LIS sk':>6} {'comp sk':>7}")
    for r in rows:
        print(f"{r['areal_gcm2']:6g} {r['K_E_shipped']:7.1f} {r['K_Hmass_shipped']:7.1f} "
              f"{r['K_Hskin_shipped']:7.1f} | {r['f_lis_phantom']:6.3f} {r['f_comp_phantom']:7.3f} "
              f"{r['f_lis_skin']:6.3f} {r['f_comp_skin']:7.3f}")

    print("\nGap and residual vs flood (flood / kernel):")
    print(f"{'g/cm2':>6} {'flood ph':>8} {'gap ph':>7} {'resid ph':>13} | "
          f"{'flood sk':>8} {'gap sk':>7} {'resid sk':>13}  flood provenance")
    for r in rows:
        if "flood_phantom" not in r:
            print(f"{r['areal_gcm2']:6g}   (no flood row yet)")
            continue
        print(f"{r['areal_gcm2']:6g} {r['flood_phantom']:8.1f} {r['gap_phantom']:7.3f} "
              f"{r['resid_phantom']:6.3f}±{r['resid_phantom_err']:.3f} | "
              f"{r['flood_skin']:8.1f} {r['gap_skin']:7.3f} "
              f"{r['resid_skin']:6.3f}±{r['resid_skin_err']:.3f}  {r['flood_stale'] or 'ok'}")

    print("\nResidual depth-shape (d ln resid / d ln depth; 0 = flat):")
    for key in ("resid_phantom", "resid_skin"):
        s = log_slope(rows, key)
        if s is None:
            print(f"  {key}: need >= 2 flood depths")
        else:
            print(f"  {key}: {s[0]:+.3f} ± {s[1]:.3f}  "
                  f"({'flat within 2 sigma' if abs(s[0]) < 2 * s[1] else 'DEPTH-DEPENDENT'})")
    if any(r.get("flood_stale") for r in rows):
        print("\nNOTE: some flood rows are unstamped or stale. Re-run "
              "accuracy_flood_sweep.py --force before reading the residual as final.")
    print("\nRemaining (needs MC): the residual folds together the regolith-floor albedo "
          "(the kernel has no ground), the hugging shell vs the 7.5 m dome, and phantom "
          "size/position.")

    fields = list(dict.fromkeys(k for r in rows for k in r))
    fields.append("provenance")
    stamp = dosimetry.provenance_str(dosimetry.provenance(engine="gate_decompose"))
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()},
                        "provenance": stamp})
    print(f"\nwrote {args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
