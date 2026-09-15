#!/usr/bin/env python3
"""Accuracy-vs-depth map: tool vs OLTARIS-Al, matched aluminium areal densities.

Purpose
-------
Turn "accurate at thick, conservative at thin" from a qualitative claim into a
measured tolerance curve for the thesis/paper. For a grid of ALUMINIUM areal
densities it places, on one axis:

  * the OLTARIS-Al reference   (Effective Dose Equivalent vs Al-sphere thickness)
  * the tool's THIN-WALL KERNEL effective dose   (computed live here, no MC)
  * the tool's FLOOD MC effective/skin dose       (filled from PC runs)
  * the PRODUCTION value the tool actually returns (kernel below the ~19 g/cm^2
    gate, flood above it) and its ratio to OLTARIS.

The kernel column is pure-Python (dosimetry.assess_gcr_thinwall) and is filled
every run. The OLTARIS column is entered by hand from the OLTARIS web tool
(accuracy_al_oltaris.csv); the flood column comes from PC MC runs
(accuracy_al_flood.csv). Rows without a reference are still computed and printed
so the kernel curve is visible immediately.

TARGET-BLIND: nothing here feeds a dose value back into the physics. OLTARIS is
a post-hoc external reference for reporting a tolerance, never a fit target.

Usage
-----
    python3 paper/accuracy_vs_depth.py                 # table only
    python3 paper/accuracy_vs_depth.py --plot          # + F13 figure
    python3 paper/accuracy_vs_depth.py --write-kernel   # also dump kernel CSV
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Matched aluminium grid (g/cm^2). Chosen so the OLTARIS Al-sphere thicknesses are
# round-ish numbers and the ~19 g/cm^2 kernel->flood gate is bracketed.
GRID_GCM2 = [2.03, 5.0, 10.0, 15.0, 19.0, 25.0, 30.0, 40.0, 50.0, 54.0]
AL_RHO_G_CM3 = 2.70          # aluminium density
GATE_GCM2 = 19.0             # production kernel->flood crossover
DOME_INNER_R_CM = 750.0      # same 3 m-class dome the kernel anchor uses


def al_cm(gcm2: float) -> float:
    return gcm2 / AL_RHO_G_CM3


def compute_kernel(gcm2: float) -> dict | None:
    """Tool thin-wall kernel effective dose for an Al wall of this areal density.
    Pure-Python fold (no MC); uses the shipped default LIS (rigidity)."""
    from lunarsim.spec import HabitatSpec, WallLayer
    from lunarsim import dosimetry
    t_cm = al_cm(gcm2)
    spec = HabitatSpec(name=f"al{gcm2:g}", shape="dome",
                       inner_radius_cm=DOME_INNER_R_CM,
                       walls=[WallLayer("aluminium", t_cm)])
    try:
        a = dosimetry.assess_gcr_thinwall(spec)
    except Exception as e:                       # noqa: BLE001  (report, don't crash the map)
        print(f"  ! kernel fold failed at {gcm2} g/cm^2: {e}", file=sys.stderr)
        return None
    return {"areal_gcm2": spec.areal_density_gcm2(),
            "eff_mSv_yr": a.annual_msv,
            "eff_mGy_yr": a.annual_mgy,
            "qbar": a.quality_factor}


def load_ref(path: Path, value_col: str) -> dict[float, float]:
    """areal_gcm2 -> value, skipping blank cells. Keyed to 0.1 g/cm^2."""
    out: dict[float, float] = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path, newline="")):
        v = (r.get(value_col) or "").strip()
        if not v:
            continue
        out[round(float(r["areal_gcm2"]), 1)] = float(v)
    return out


def nearest(ref: dict[float, float], g: float, tol: float = 0.6) -> float | None:
    """Look up a reference value at areal density g, tolerating small grid offsets."""
    if not ref:
        return None
    key = min(ref, key=lambda k: abs(k - g))
    return ref[key] if abs(key - g) <= tol else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oltaris", default=str(HERE / "accuracy_al_oltaris.csv"))
    ap.add_argument("--flood", default=str(HERE / "accuracy_al_flood.csv"))
    ap.add_argument("--plot", action="store_true", help="write F13 figure")
    ap.add_argument("--write-kernel", action="store_true",
                    help="dump computed kernel column to accuracy_al_kernel.csv")
    ap.add_argument("-o", "--out", default=str(HERE / "F13_accuracy_vs_depth.png"))
    args = ap.parse_args()

    olt = load_ref(Path(args.oltaris), "oltaris_eff_mSv_yr")
    flood_eff = load_ref(Path(args.flood), "flood_eff_mSv_yr")

    rows = []
    for g in GRID_GCM2:
        k = compute_kernel(g)
        o = nearest(olt, g)
        f = nearest(flood_eff, g)
        prod = (k["eff_mSv_yr"] if k else None) if g < GATE_GCM2 else f
        prod_engine = "kernel" if g < GATE_GCM2 else "flood"
        rows.append({"g": g, "al_cm": al_cm(g), "k": k, "olt": o,
                     "flood": f, "prod": prod, "engine": prod_engine})

    # ---- table ----
    hdr = (f"{'Al g/cm2':>9} {'Al cm':>6} {'OLTARIS':>8} {'kernel':>8} "
           f"{'k/OLT':>6} {'Q_kern':>6} {'flood':>8} {'PROD':>8} "
           f"{'prod/OLT':>8}  engine")
    print("\nTool vs OLTARIS-Al  --  effective dose equivalent (mSv/yr)\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        k = r["k"]
        ke = f"{k['eff_mSv_yr']:8.1f}" if k else f"{'--':>8}"
        kq = f"{k['qbar']:6.2f}" if k else f"{'--':>6}"
        o = f"{r['olt']:8.0f}" if r["olt"] is not None else f"{'--':>8}"
        kr = (f"{k['eff_mSv_yr']/r['olt']:6.2f}"
              if (k and r["olt"]) else f"{'--':>6}")
        fl = f"{r['flood']:8.1f}" if r["flood"] is not None else f"{'--':>8}"
        pr = f"{r['prod']:8.1f}" if r["prod"] is not None else f"{'--':>8}"
        prr = (f"{r['prod']/r['olt']:8.2f}"
               if (r["prod"] is not None and r["olt"]) else f"{'--':>8}")
        print(f"{r['g']:9.2f} {r['al_cm']:6.2f} {o} {ke} {kr} {kq} "
              f"{fl} {pr} {prr}  {r['engine']}")
    print(f"\ngate = {GATE_GCM2} g/cm^2 (kernel below, flood above); "
          "ratio >1 = tool conservative (over-warns), <1 = optimistic.")
    missing_olt = [f"{r['g']:g}" for r in rows if r["olt"] is None]
    missing_fl = [f"{r['g']:g}" for r in rows
                  if r["engine"] == "flood" and r["flood"] is None]
    if missing_olt:
        print(f"  TODO OLTARIS-Al web run at (g/cm^2): {', '.join(missing_olt)} "
              f"-> add to {Path(args.oltaris).name}")
    if missing_fl:
        print(f"  TODO flood MC (PC) at (g/cm^2): {', '.join(missing_fl)} "
              f"-> add to {Path(args.flood).name}")

    if args.write_kernel:
        kp = HERE / "accuracy_al_kernel.csv"
        with open(kp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["areal_gcm2", "al_cm", "eff_mSv_yr", "eff_mGy_yr", "qbar"])
            for r in rows:
                if r["k"]:
                    w.writerow([f"{r['g']:g}", f"{r['al_cm']:.3f}",
                                f"{r['k']['eff_mSv_yr']:.1f}",
                                f"{r['k']['eff_mGy_yr']:.2f}",
                                f"{r['k']['qbar']:.3f}"])
        print(f"wrote {kp}")

    if args.plot:
        make_plot(rows, Path(args.out))


def make_plot(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_OLT, C_KERN, C_FLOOD, C_PROD = "#b8860b", "#2c6fbf", "#c0392b", "#1a1a1a"

    gx = [r["g"] for r in rows]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.6, 7.0), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})

    def series(key, sub=None):
        xs, ys = [], []
        for r in rows:
            v = r[key] if sub is None else (r[key][sub] if r[key] else None)
            if v is not None:
                xs.append(r["g"]); ys.append(v)
        return xs, ys

    ox, oy = series("olt")
    if ox:
        ax.plot(ox, oy, "-o", color=C_OLT, lw=2, ms=6, zorder=6,
                label="OLTARIS-Al (reference)")
    kx, ky = series("k", "eff_mSv_yr")
    ax.plot(kx, ky, "--s", color=C_KERN, ms=5, alpha=0.85, zorder=5,
            label="tool kernel (all depths)")
    fx, fy = series("flood")
    if fx:
        ax.plot(fx, fy, "^", color=C_FLOOD, ms=8, mfc="white", zorder=6,
                label="tool flood MC")
    px, py = series("prod")
    if px:
        ax.plot(px, py, "-", color=C_PROD, lw=2.6, alpha=0.55, zorder=4,
                label="tool PRODUCTION (gated)")
    ax.axvline(GATE_GCM2, color="#888", ls=":", lw=1.3, zorder=2)
    ax.text(GATE_GCM2, ax.get_ylim()[1], " gate 19", color="#666",
            va="top", ha="left", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Effective dose equivalent  (mSv/yr)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.94)
    ax.set_title("Tool vs OLTARIS-Al accuracy map\n"
                 "effective dose equivalent vs Al areal density "
                 "(ICRP-60 Q, solar min)", fontsize=11)

    # ratio panel
    for r in rows:
        if not r["olt"]:
            continue
        if r["k"]:
            axr.plot(r["g"], r["k"]["eff_mSv_yr"] / r["olt"], "s",
                     color=C_KERN, ms=5, alpha=0.85, zorder=5)
        if r["flood"]:
            axr.plot(r["g"], r["flood"] / r["olt"], "^", color=C_FLOOD,
                     ms=8, mfc="white", zorder=6)
        if r["prod"] is not None:
            axr.plot(r["g"], r["prod"] / r["olt"], "o", color=C_PROD,
                     ms=6, zorder=7)
    axr.axhline(1.0, color="#2ca02c", lw=1.4, zorder=3)
    axr.axhspan(0.9, 1.1, color="#2ca02c", alpha=0.10, zorder=1)
    axr.axvline(GATE_GCM2, color="#888", ls=":", lw=1.3, zorder=2)
    axr.set_yscale("log")
    axr.set_ylabel("tool / OLTARIS")
    axr.set_xlabel("aluminium areal density  (g/cm$^2$)")
    axr.grid(True, which="both", alpha=0.22)
    axr.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
