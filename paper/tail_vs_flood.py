#!/usr/bin/env python3
"""Step-4 harness: does the energy-ceiling tail explain the kernel/flood residual?

Background. The shipped per-element thin-wall kernel's top node is 6 GeV/n (H, He)
or 5 GeV/n (C..Fe), and the fold integrates the free-field spectrum only up to that
top node. The flood MC source samples up to GCR_EMAX_PER_NUC = 20 GeV/n. So the two
engines do NOT see the same energy range: the kernel is missing the 5/6-20 GeV/n band
that the flood transports.

The 2026-10-20 tail MC (kernel_gen.tail, --composition perelement_tail) measured
that band: E lift +7.6% @0, +11.6% @10, +21.5% @50 g/cm2. It was kept as a
diagnostic, not promoted, because promoting it would have broken the then-validated
kernel/flood ~0.99 at 50. That validation no longer holds. Under the matthia2013
flood (2026-10-27), step 3 (paper/gate_ratio_decompose.py) measured flood/K_in
residuals of 1.27 -> 1.45 (phantom) and 1.28 -> 1.60 (skin) from 19 to 50 g/cm2.
The residual grows with depth, which is the same direction as the tail lift.

This script re-folds base vs tail under the current model and divides the step-3
residual by the tail lift:

    resid_after = (flood / K_in) / (K_tail / K_base)

The lift is measured on the per-element kernel (the only one with a tail) under the
ENERGY LIS, which is the flood's LIS. It is then applied as a ratio to the
flood-input-matched K_in. The rigidity-LIS lift (what the GUI would fold) is
reported alongside, to show that the lift barely depends on the LIS form.

How to read it (target-blind: tool vs itself, no reference dose enters):
  resid_after flat in depth -> the tail carries the depth dependence of the gap. The
      constant that remains is a geometry factor: no regolith floor in the kernel,
      and a 15 cm hugging shell vs the 7.5 m dome. Promote the tail, then attack
      that constant separately.
  resid_after still rising  -> the tail is not the depth driver. Look at
      transport geometry, i.e. secondary build-up in the dome vs the hugging shell.
The kernel anchors are 0 / 2.025 / 10 / 50 g/cm2 only. Depths between anchors are
log-log interpolations, so the lift at 19-40 is interpolated, not measured.

Laptop column (needs no tail JSON):
  ff_missing = free-field particle flux above the base kernel's top node, as a
               fraction of the 10 MeV/n - 20 GeV/n flux the flood samples.

Usage:
  python3 paper/tail_vs_flood.py        # base = *_pretail.json, tail = shipped (promoted)
  python3 paper/tail_vs_flood.py --tail <json> --depths 2.025 5 10 19 30 50
Writes paper/tail_vs_flood.csv (provenance-stamped).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from lunarsim import dosimetry      # noqa: E402

_spec = importlib.util.spec_from_file_location("gate_ratio_decompose",
                                               HERE / "gate_ratio_decompose.py")
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)

# Since the 2026-11-05 promotion the SHIPPED per-element kernel IS the tail kernel;
# the pre-promotion kernel is kept as *_pretail.json and is the default base here.
DEFAULT_TAIL = REPO_ROOT / "lunarsim" / "data" / "gcr_thinwall_kernel_perelem.json"
DEFAULT_BASE = REPO_ROOT / "lunarsim" / "data" / "gcr_thinwall_kernel_perelem_pretail.json"
OUT_CSV = HERE / "tail_vs_flood.csv"
QUANTITIES = (("E", "E"), ("phantom", "H_mass"), ("skin", "H_skin"))


@contextlib.contextmanager
def perelem_kernel(path: Path):
    """Fold the per-element path on `path` instead of the shipped file. This is a
    scoped, single-threaded harness patch (like gd.kernel_lis_form). Production
    code never does this."""
    path = Path(path).resolve()
    K = json.loads(path.read_text())
    if K.get("meta", {}).get("composition") not in ("perelement", "perelement_tail"):
        raise ValueError(f"{path.name}: meta.composition="
                         f"{K.get('meta', {}).get('composition')!r} is not a "
                         "per-element kernel; folding it on the per-element "
                         "spectrum would double- or under-count bands")
    saved_path = dosimetry._GCR_THINWALL_KERNEL_PERELEM
    saved_load = dosimetry._load_gcr_thinwall_kernel_perelem
    dosimetry._GCR_THINWALL_KERNEL_PERELEM = path
    dosimetry._load_gcr_thinwall_kernel_perelem = lambda: K
    try:
        yield K
    finally:
        dosimetry._GCR_THINWALL_KERNEL_PERELEM = saved_path
        dosimetry._load_gcr_thinwall_kernel_perelem = saved_load


def top_nodes(K: dict) -> dict:
    """{species: top node MeV/n}, checked to be the same at every anchor."""
    tops = {}
    for p in K["points"]:
        for name, sp in p["species"].items():
            t = sp["nodes_pernuc_mev"][-1]
            if tops.setdefault(name, t) != t:
                raise ValueError(f"{name}: top node differs between anchors")
    return tops


def missing_flux_fraction(K: dict, phi_MV: float = 400.0) -> float:
    """Free-field flux above each species' top node / flux over the whole range the
    flood samples (the per-nucleon range from the lowest node's edge to emax). It is
    abundance-weighted and uses the shipped fold model. It is a pure spectrum
    integral, so it needs no tail kernel."""
    pt = K["points"][0]
    emax = dosimetry.current_model().emax_per_nuc
    model = dataclasses.replace(dosimetry.current_model(),
                                          lis_form=dosimetry.KERNEL_LIS_FORM)
    lost = tot = 0.0
    with dosimetry.use_model(model):
        for _name, sp in pt["species"].items():
            z, a, ab = sp["z"], sp["a"], sp["abundance"]
            lo, top = sp["nodes_pernuc_mev"][0], sp["nodes_pernuc_mev"][-1]
            hi = dosimetry._raw_gcr_integral(z, a, phi_MV, top, emax, n=400)
            lost += ab * hi
            tot += ab * (dosimetry._raw_gcr_integral(z, a, phi_MV, lo, top, n=400) + hi)
    return lost / tot if tot else 0.0


def lift_point(gcm2: float, tail: Path, lis_form: str, base_path=None) -> dict:
    """K_tail / K_base for E, mass-average phantom H and skin H at one depth.
    base_path None = the shipped kernel as base."""
    if base_path is None:
        base = gd.kernel_point(gcm2, lis_form, "perelement")
    else:
        with perelem_kernel(base_path):
            base = gd.kernel_point(gcm2, lis_form, "perelement")
    with perelem_kernel(tail):
        ext = gd.kernel_point(gcm2, lis_form, "perelement")
    if ext["kernel"] != Path(tail).name:
        raise RuntimeError(f"tail fold ran on {ext['kernel']}, not {Path(tail).name}")
    out = {f"lift_{q}": ext[k] / base[k] for q, k in QUANTITIES}
    out.update({f"Kbase_{q}": base[k] for q, k in QUANTITIES})
    out.update({f"Ktail_{q}": ext[k] for q, k in QUANTITIES})
    out["Ktail_E_rel"] = ext["E_rel"]
    return out


def compare(depths, tail: Path, flood: dict, base_path=None) -> list:
    """One row per depth: the lift under both LIS forms, plus the step-3 residual
    before and after the energy-LIS lift wherever a flood row exists."""
    resid = {r["areal_gcm2"]: r for r in gd.decompose(depths, flood)}
    rows = []
    for g in depths:
        en = lift_point(g, tail, "energy", base_path)
        rg = lift_point(g, tail, gd.CONFIGS[0][1], base_path)   # shipped LIS form
        row = {"areal_gcm2": g, **en,
               **{f"lift_{q}_shipped_lis": rg[f"lift_{q}"] for q, _k in QUANTITIES}}
        d = resid[g]
        for q in ("phantom", "skin"):
            if d.get(f"resid_{q}"):
                lift = en[f"lift_{q}"]
                row[f"resid_{q}"] = d[f"resid_{q}"]
                row[f"resid_{q}_err"] = d[f"resid_{q}_err"]
                row[f"resid_after_{q}"] = d[f"resid_{q}"] / lift
                row[f"resid_after_{q}_err"] = d[f"resid_{q}_err"] / lift
        if d.get("flood_stale"):
            row["flood_stale"] = d["flood_stale"]
        rows.append(row)
    return rows


def _fmt_slope(s) -> str:
    return "   n/a        " if s is None else f"{s[0]:+.3f}±{s[1]:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tail", type=Path, default=DEFAULT_TAIL)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help="pre-tail kernel (default *_pretail.json)")
    ap.add_argument("--depths", type=float, nargs="+", default=list(gd.DEFAULT_DEPTHS))
    ap.add_argument("--flood-csv", type=Path, default=gd.FLOOD_CSV)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args(argv)

    base_K = json.loads(args.base.read_text())
    print(f"base kernel: {args.base.name}   tail kernel: {args.tail.name}")
    tops = top_nodes(base_K)
    print(f"base kernel top nodes (MeV/n): {tops}")
    print(f"flood source ceiling: {dosimetry.current_model().emax_per_nuc:g} MeV/n")
    print(f"ff_missing (free-field flux the kernel never folds): "
          f"{missing_flux_fraction(base_K):.2%}\n")

    if not args.tail.exists():
        print(f"tail kernel not found: {args.tail}\n"
              "It lives on the PC. Copy it here, or run this script on the PC.")
        return 2
    tail_K = json.loads(args.tail.read_text())
    ttops = top_nodes(tail_K)
    print(f"tail kernel top nodes (MeV/n): {ttops}")
    if any(t < dosimetry.current_model().emax_per_nuc for t in ttops.values()):
        print("  WARNING: tail top node below the flood ceiling; part of the band is "
              "still missing")
    if set(p["wall_gcm2"] for p in tail_K["points"]) != \
            set(p["wall_gcm2"] for p in base_K["points"]):
        print("  WARNING: tail and base anchors differ")

    flood = gd.load_flood(args.flood_csv)
    rows = compare(args.depths, args.tail, flood, args.base)

    print("\nLIFT = K_tail/K_base (energy LIS; shipped-LIS E lift in brackets)")
    print(f"{'g/cm2':>6} | {'E':>14} {'phantom':>8} {'skin':>8} | "
          f"{'resid ph':>9} {'-> after':>9} | {'resid sk':>9} {'-> after':>9}")
    for r in rows:
        def cell(q):
            if f"resid_{q}" not in r:
                return f"{'-':>9} {'-':>9}"
            return f"{r[f'resid_{q}']:9.3f} {r[f'resid_after_{q}']:9.3f}"
        print(f"{r['areal_gcm2']:6g} | {r['lift_E']:6.3f} [{r['lift_E_shipped_lis']:5.3f}] "
              f"{r['lift_phantom']:8.3f} {r['lift_skin']:8.3f} | "
              f"{cell('phantom')} | {cell('skin')}")

    print("\nd ln(resid) / d ln(depth), flood-error weighted (0 = flat):")
    for q in ("phantom", "skin"):
        print(f"  {q:8} before {_fmt_slope(gd.log_slope(rows, f'resid_{q}'))}"
              f"   after tail {_fmt_slope(gd.log_slope(rows, f'resid_after_{q}'))}")
    stale = sorted({r['flood_stale'] for r in rows if r.get('flood_stale')})
    if stale:
        print(f"\n  NOTE: flood rows stale vs current model: {stale}. Rerun the flood sweep "
              "with --force before drawing a verdict.")

    prov = dosimetry.provenance_str(dosimetry.provenance(engine="kernel+tail"))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "areal_gcm2", k))
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["tail_kernel", "provenance"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "tail_kernel": args.tail.name, "provenance": prov})
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
