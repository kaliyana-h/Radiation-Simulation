#!/usr/bin/env python3
"""Figure F11 - habitat dose vs regolith overburden depth.

Reads the regolith-sweep product dose_vs_depth.csv (produced by
`python3 -m lunarsim.regolith_sweep collect <sweepdir>`) and renders the
dose-vs-depth figure: skin and core organ points, dose-equivalent (mSv/yr)
on the primary axis and absorbed dose (mGy/yr) on a twin axis, against a
regolith-depth x-axis with an areal-density secondary scale on top.

GCR spectrum: Usoskin et al. (2005) LIS, force-field modulated at
phi = 475 MV (Akisheva/BON-2014). Surface geometry, upper-hemisphere source.

Usage:
    python3 paper/plot_dose_vs_depth.py [dose_vs_depth.csv] [-o F11_dose_vs_depth.png]

Expected columns:
    depth_m, areal_gcm2, usable,
    skin_absorbed_mGy_yr, core_absorbed_mGy_yr,
    skin_doseeq_mSv_yr,   core_doseeq_mSv_yr
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if not rows:
        sys.exit(f"{path}: no data rows")
    rows.sort(key=lambda r: float(r["depth_m"]))
    return rows


def col(rows, name):
    return [float(r[name]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="dose_vs_depth.csv",
                    help="collect() output (default: dose_vs_depth.csv)")
    ap.add_argument("-o", "--out", default="F11_dose_vs_depth.png",
                    help="output image path")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        sys.exit(f"not found: {args.csv} "
                 f"(run: python3 -m lunarsim.regolith_sweep collect <sweepdir>)")

    rows = load(args.csv)
    depth = col(rows, "depth_m")
    areal = col(rows, "areal_gcm2")
    skin_h = col(rows, "skin_doseeq_mSv_yr")
    core_h = col(rows, "core_doseeq_mSv_yr")
    skin_d = col(rows, "skin_absorbed_mGy_yr")
    core_d = col(rows, "core_absorbed_mGy_yr")

    C_SKIN, C_CORE = "#c0392b", "#2c6fbf"

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    axd = ax.twinx()

    # Dose equivalent (primary, solid, filled markers)
    ax.plot(depth, skin_h, "-o", color=C_SKIN, label="Skin  H (mSv/yr)", zorder=5)
    ax.plot(depth, core_h, "-o", color=C_CORE, label="Core  H (mSv/yr)", zorder=5)
    # Absorbed dose (twin, dashed, open markers)
    axd.plot(depth, skin_d, "--s", color=C_SKIN, mfc="white",
             label="Skin  D (mGy/yr)", alpha=0.75)
    axd.plot(depth, core_d, "--s", color=C_CORE, mfc="white",
             label="Core  D (mGy/yr)", alpha=0.75)

    ax.set_xlabel("Regolith overburden depth  (m)")
    ax.set_ylabel("Dose equivalent  H  (mSv / yr)")
    axd.set_ylabel("Absorbed dose  D  (mGy / yr)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    axd.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, zorder=0)

    # Secondary top axis: areal density (g/cm^2), sharing the depth mapping.
    def d2a(d):
        import numpy as np
        return np.interp(d, depth, areal)

    def a2d(a):
        import numpy as np
        return np.interp(a, areal, depth)

    top = ax.secondary_xaxis("top", functions=(d2a, a2d))
    top.set_xlabel("Areal density  (g / cm$^2$)")

    # Merge legends from both axes.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axd.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.92, fontsize=9)

    ax.set_title("Habitat organ dose vs. regolith overburden\n"
                 "GCR, Usoskin LIS, $\\phi$ = 475 MV (solar min)", fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    fig.savefig(Path(args.out).with_suffix(".pdf"))
    print(f"wrote {args.out} and {Path(args.out).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
