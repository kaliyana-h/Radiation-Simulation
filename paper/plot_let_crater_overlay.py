#!/usr/bin/env python3
"""LET / water-point <Q>(depth) validation overlay -- tool vs CRaTER + arXiv 2511.09040.

Companion figure to the thin-wall Q-bar cross-check (memory crater-qfactor-depth-
reference + let-scorer-thin-qbar-rootcause). It renders the output of
kernel_gen.let_harness.collect() -- runs/.../let_qbar_depth.json -- as a two-panel
figure for the NIMB paper:

  Panel 1  water-point dose-averaged <Q> vs Al shielding depth: the tool's native
           (DoseToMedium / DoseEquivalent_ICRP) curve and, when the LETSpectrumWater
           ntuple is present, the ntuple curve, overlaid with the POST-HOC measured
           + modelled arbiter anchors (CRaTER/LRO, arXiv 2511.09040, TGO/Liulin-MO).
           The weighting-robust test is the fall-off SHAPE: ~5.6 at 0 g/cm^2 down to
           a ~3.5 minimum near 10-20 g/cm^2. A too-slow fall is the HZE-over-
           retention signature the per-element fix removed.

  Panel 2  dose-weighted LET spectrum dD/dlnL at each depth anchor (ntuple only) --
           shows WHERE on the LET axis the dose equivalent sits and how the HZE
           (high-LET) shoulder attenuates with depth. Blank with a note if the
           ntuple path was not run (extension not built).

TARGET-BLIND: the arbiter anchors are drawn as a reference overlay only; nothing
here is fit to them. Reads only the harness JSON -- no MC, laptop-only.

Usage (after the PC run + collect; see the recipe in let_harness.emit output):
    python3 paper/plot_let_crater_overlay.py --json runs/let_perelem/let_qbar_depth.json
    python3 paper/plot_let_crater_overlay.py --json <path> -m aluminium -o paper/F13_let_crater_overlay.png
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# Palette (shared with plot_oltaris_overlay.py where it overlaps).
C_NATIVE = "#2c6fbf"   # tool native scalar fold (blue)
C_NTUPLE = "#c0392b"   # tool ntuple fold (red)
C_REF = "#b8860b"      # arbiter anchors (dark goldenrod)
C_REF2 = "#6a3d9a"     # arbiter minimum / TGO (purple)
_DEPTH_CMAP = "viridis"


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(
            f"not found: {path}\n"
            "  This figure renders the LET harness output. Produce it on the PC:\n"
            "    1. build the ntuple extension (once):\n"
            "         cmake -DTOPAS_EXTENSIONS_DIR=~/topas_extensions . && make\n"
            "    2. emit + run + collect:\n"
            "         python3 -m lunarsim.kernel_gen.let_harness emit -o runs/let_perelem \\\n"
            "             --composition perelement -m aluminium\n"
            "         export TOPAS_G4_DATA_DIR=~/G4Data && bash runs/let_perelem/run_all.sh\n"
            "         python3 -m lunarsim.kernel_gen.let_harness collect -o runs/let_perelem\n")
    return json.loads(path.read_text())


def _bin_centers(edges):
    """Geometric-mean centre of each log LET bin."""
    return [math.sqrt(edges[i] * edges[i + 1]) for i in range(len(edges) - 1)]


def _dln_widths(edges):
    """ln-width of each log LET bin (constant for the harness' 10/decade grid)."""
    return [math.log(edges[i + 1] / edges[i]) for i in range(len(edges) - 1)]


def _panel_qbar(ax, mat, ref_depth, ref_point):
    """Panel 1: water-point <Q>(depth), tool curves + arbiter overlay."""
    pts = mat["points"]
    depth = [p["wall_gcm2"] for p in pts]

    nat = [p.get("qbar_waterpoint_native") for p in pts]
    nx = [d for d, q in zip(depth, nat) if q is not None]
    ny = [q for q in nat if q is not None]
    if nx:
        ax.plot(nx, ny, "-o", color=C_NATIVE, lw=2, ms=7, zorder=6,
                label="Tool water point  (native ICRP scalar fold)")

    ntp = [p.get("qbar_waterpoint_ntuple") for p in pts]
    tx = [d for d, q in zip(depth, ntp) if q is not None]
    ty = [q for q in ntp if q is not None]
    if tx:
        ax.plot(tx, ty, "-s", color=C_NTUPLE, lw=2, ms=7, mfc="white", zorder=7,
                label="Tool water point  (LET-ntuple fold)")

    # --- arbiter <Q>(depth) anchors: post-hoc reference, NOT fit targets ---
    # minimum/TGO band vs the shallow CRaTER/arXiv points get distinct colours.
    for d, q, lbl in sorted(ref_depth):
        is_min = ("minimum" in lbl) or ("TGO" in lbl) or ("Liulin" in lbl)
        ax.plot(d, q, marker=("v" if is_min else "*"),
                color=(C_REF2 if is_min else C_REF),
                ms=(11 if is_min else 15), mec="k", mew=0.6, zorder=9,
                linestyle="none", label=f"arbiter: {lbl}")

    ax.set_xlabel("Al shielding depth  (g / cm$^2$)")
    ax.set_ylabel(r"water-point dose-averaged  $\langle Q \rangle$")
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.22, zorder=0)
    ax.axhspan(3.4, 3.6, color=C_REF2, alpha=0.08, zorder=1)
    ax.annotate(r"measured $\sim$3.5 minimum", xy=(0.5, 3.5), fontsize=8,
                color=C_REF2, va="bottom")
    ax.legend(loc="upper right", framealpha=0.94, fontsize=7.6)
    ax.set_title(r"Water-point $\langle Q \rangle$ vs depth: tool vs arbiter"
                 "\n(TARGET-BLIND overlay; CRaTER / arXiv 2511.09040 / TGO)",
                 fontsize=10)


def _panel_spectrum(ax, mat):
    """Panel 2: dose-weighted LET spectrum dD/dlnL per depth anchor (ntuple only)."""
    edges = mat["let_bins_kev_um"]
    centers = _bin_centers(edges)
    dln = _dln_widths(edges)
    pts = [p for p in mat["points"] if p.get("dose_let_spectrum")]
    if not pts:
        ax.text(0.5, 0.5, "no LET-ntuple spectrum\n(build the LETSpectrumWater\n"
                "extension + rerun to populate)",
                ha="center", va="center", fontsize=10, color="#888",
                transform=ax.transAxes)
        ax.set_title("Dose-weighted LET spectrum  (ntuple)", fontsize=10)
        ax.set_xscale("log")
        return

    cmap = matplotlib.colormaps[_DEPTH_CMAP]
    dmax = max(p["wall_gcm2"] for p in pts) or 1.0
    for p in pts:
        spec = p["dose_let_spectrum"]
        tot = sum(spec) or 1.0
        # dD/dlnL, normalised to unit integrated dose so SHAPE (not the per-node
        # primary count) is compared across depths.
        y = [(spec[i] / tot) / dln[i] for i in range(len(spec))]
        col = cmap(0.15 + 0.8 * (p["wall_gcm2"] / dmax))
        ax.plot(centers, y, "-", lw=1.8, color=col,
                label=f"{p['wall_gcm2']:g} g/cm$^2$")

    # ICRP-60 Q kink markers (L=10 and 100 keV/um) for orientation.
    for L in (10.0, 100.0):
        ax.axvline(L, color="#999", ls=":", lw=1, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"LET in water  $L$  (keV / $\mu$m)")
    ax.set_ylabel(r"normalised dose spectrum  $\mathrm{d}D/\mathrm{d}\ln L$")
    ax.grid(True, which="both", alpha=0.22, zorder=0)
    ax.legend(loc="upper left", framealpha=0.94, fontsize=8, title="Al depth")
    ax.set_title("Dose-weighted LET spectrum vs depth\n"
                 r"(high-$L$ shoulder = HZE dose; attenuates with shielding)",
                 fontsize=10)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, type=Path,
                    help="let_qbar_depth.json from kernel_gen.let_harness collect")
    ap.add_argument("-m", "--material", default=None,
                    help="material key to plot (default: first in the file)")
    ap.add_argument("-o", "--out", default=str(HERE / "F13_let_crater_overlay.png"))
    args = ap.parse_args(argv)

    data = _load(args.json)
    materials = data.get("materials", {})
    if not materials:
        sys.exit(f"{args.json}: no materials block")
    mkey = args.material or next(iter(materials))
    if mkey not in materials:
        sys.exit(f"{args.json}: material '{mkey}' not in {list(materials)}")
    mat = materials[mkey]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    _panel_qbar(ax1, mat, data.get("reference_qbar_depth", []),
                data.get("reference_qbar_point", []))
    _panel_spectrum(ax2, mat)
    fig.suptitle(f"LET / water-point validation  --  {mkey}  "
                 f"($\\phi$={data.get('phi_MV', 400):g} MV, "
                 f"{data.get('composition', '?')})", fontsize=12, y=1.02)

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")

    # brief console echo so the figure's headline numbers are inspectable.
    print(f"\n{mkey}: water-point <Q>(depth)  (native | ntuple)")
    for p in mat["points"]:
        qn = p.get("qbar_waterpoint_native")
        qt = p.get("qbar_waterpoint_ntuple")
        hz = p.get("hze_dose_fraction")
        print(f"  {p['wall_gcm2']:>6g} g/cm^2 : "
              f"{(f'{qn:.2f}' if qn is not None else '--'):>6} | "
              f"{(f'{qt:.2f}' if qt is not None else '--'):>6}"
              f"{f'   HZE dose {100*hz:.1f}%' if hz is not None else ''}")


if __name__ == "__main__":
    main()
