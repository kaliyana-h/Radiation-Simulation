#!/usr/bin/env python3
"""Wall-geometry bracket probe: is the kernel's deep-shield absorbed-dose plunge a
WALL-OBLIQUITY defect, and does a dome-faithful concentric SHELL reproduce the
trusted flood/OLTARIS absorbed across depth?

What the first (50 g/cm^2, skin) run established
------------------------------------------------
The committed kernel shields the phantom with a FLAT slab: every upper-hemisphere
ring crosses t/cos(theta), an effective <1/cos>=2.0 obliquity that OVER-attenuates at
depth. A concentric SHELL (zero obliquity) at 50 g/cm^2 folds to a wT-weighted
EFFECTIVE absorbed of 0.302 mGy/day -- reproducing the target-blind OLTARIS deep-shield
absorbed (0.302) to the third digit, vs the slab's 0.184 (0.61x). So the wall geometry
is the dominant, essentially SUFFICIENT lever for the headline; config.py "WHY A SLAB,
NOT A SHELL" rejected the shell by measuring it against the (wrong) slab-tuned kernel,
not against the flood -- a circular test this probe overturns.

The SKIN organ is the one exception and is NOT a defect: the flood scores skin as a
2 cm wall-LINING shell (backed by vacuum, unshielded), while the kernel scores it as the
outer shell of the self-shielded 15 cm central phantom. That definitional gap is the
whole residual ~7% at skin -- and skin carries wT=0.01, ~0.1% of the effective dose.

What THIS multi-anchor run decides
----------------------------------
Before spending a full shell REGEN, the one live risk is config's "shell/slab grows with
thickness": the kernel grid is [0, 2.025, 10, 50] and the plunge is only at 50 (the
interpolated 19 g/cm^2 reads a healthy 1.07x OLTARIS today). If a pure-shell regen also
inflates the THIN anchors (2.025, 10), it would over-read where the kernel is currently
correct. This probe folds the shell at ALL shielded anchors and compares, per anchor:

    slab effective absorbed   (committed kernel, over-attenuates at depth)
    shell effective absorbed  (this run, dome-faithful zero-obliquity)
    OLTARIS / flood reference (target-blind, valid at depth >= the 19 g/cm^2 gate)

Read-out:
  * at 50: shell should sit on OLTARIS 0.302 (confirmed) -- the deep-shield fix.
  * at 2.025 & 10 (thin regime, kernel currently ~right): shell/slab should be SMALL.
    A large thin-anchor ratio => pure-shell regen would break the thin regime, and the
    fix is a THICKNESS-DEPENDENT obliquity (dome curvature), not a flat shell swap.

TARGET-BLIND: nothing here feeds a dose value into the physics. The shell geometry is a
transport bracket; the flood/OLTARIS numbers are only post-hoc yardsticks.

Usage
-----
  # 1) laptop: emit the shell-geometry batch at all shielded anchors (or a subset)
  python3 paper/wall_geometry_probe.py emit runs/wallprobe --seeds 4
  python3 paper/wall_geometry_probe.py emit runs/wallprobe --seeds 4 --anchors 2.025,10

  # 2) PC (after git pull): run the batch
  export TOPAS_G4_DATA_DIR=~/G4Data
  bash runs/wallprobe/run_all.sh

  # 3) either machine: fold and print the depth bracket
  python3 paper/wall_geometry_probe.py report runs/wallprobe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lunarsim import dosimetry                        # noqa: E402
from lunarsim.kernel_gen import config, generate, templates  # noqa: E402

# The three SHIELDED kernel anchors (skip the 0.0 bare-phantom point). The plunge is at
# 50; the thin anchors are where a pure-shell regen must NOT over-read.
ANCHORS_GCM2 = [2.025, 10.0, 50.0]
MATERIAL = "aluminium"
COMPOSITION = "corrected"    # matches the shipped kernel (H/He/C/Si/Fe)
PHI_MV = 400.0               # FULL_RUN solar-min; same as the kernel fold & flood
SEC_PER_DAY = 86400.0

# Target-blind post-hoc yardsticks (never fed into the physics).
# OLTARIS effective absorbed (mGy/day), only where a reference point exists.
OLTARIS_DABS_MGY_DAY = {50.0: 0.302}
# Flood SKIN (2 cm wall-LINING shell) absorbed at 50 -- kept only to reproduce the
# original skin bracket; NOT comparable organ-for-organ to the central-phantom kernel.
FLOOD_SKIN_MGY_DAY = {50.0: 0.3065}


def _fold(point: dict, organs: list) -> dict:
    """Fold one anchor point -> per-organ absorbed (mGy/day) + wT-weighted effective."""
    f = dosimetry._thinwall_fold_point(point, PHI_MV, organs)
    per_organ = {k: f["HD"][k] * SEC_PER_DAY * 1e3 for k, _w in organs}
    eff = sum(wT * per_organ[k] for k, wT in organs)
    return {"per_organ": per_organ, "effective": eff}


def emit(outdir: Path, seeds: int, threads: int, anchors: list) -> None:
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    species = config.species_for(COMPOSITION)
    manifest = {"material": MATERIAL, "seeds": seeds, "threads": threads,
                "composition": COMPOSITION, "grid_gcm2": list(anchors),
                "wall_geometry": "shell", "phi_ff_cm2": config.PHI_FF_CM2,
                "note": "DIAGNOSTIC shell-wall bracket -- NOT a shippable kernel",
                "runs": []}
    n = 0
    for anchor in anchors:
        for sname, s in species.items():
            for node in range(len(s["nodes_pernuc_mev"])):
                for seed in range(1, seeds + 1):
                    rn = generate._run_name(MATERIAL, anchor, sname, node, seed)
                    rdir = runs_dir / rn
                    rdir.mkdir(exist_ok=True)
                    text = templates.build_param_file(
                        MATERIAL, anchor, sname, node, seed, threads,
                        COMPOSITION, wall_geometry="shell")
                    (rdir / "run.txt").write_text(text)
                    manifest["runs"].append({"name": rn, "wall_gcm2": anchor,
                                             "species": sname, "node": node,
                                             "seed": seed})
                    n += 1
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    generate._write_runner(outdir)
    print(f"emitted {n} SHELL-geometry runs at anchors {anchors} g/cm^2 Al "
          f"(composition={COMPOSITION}, seeds={seeds}) into {outdir}")
    print(f"  species={list(species)}  wall_geometry=shell (zero-obliquity bracket)")
    print(f"  next: export TOPAS_G4_DATA_DIR=~/G4Data && bash {outdir}/run_all.sh")


def report(outdir: Path) -> None:
    manifest = json.loads((outdir / "manifest.json").read_text())
    seeds = manifest["seeds"]
    runs_dir = outdir / "runs"
    anchors = sorted(manifest.get("grid_gcm2", ANCHORS_GCM2))

    # The kernel fold was validated on the rigidity LIS form; pin it so the shell and
    # the committed slab are folded against the IDENTICAL free-field flux (the only
    # difference between the two numbers is then the wall geometry).
    dosimetry.set_lis_form(dosimetry.KERNEL_LIS_FORM)

    K = json.loads((REPO_ROOT / "lunarsim" / "data" / "gcr_thinwall_kernel.json").read_text())
    organs = K["meta"]["organs"]

    print(f"\n=== wall-geometry EFFECTIVE absorbed bracket vs depth "
          f"({COMPOSITION} comp, phi={PHI_MV:g} MV) ===")
    print("    slab = committed kernel (<1/cos>=2.0);  shell = this run (zero obliquity)")
    print("    'ref' = target-blind OLTARIS effective absorbed where a point exists\n")
    header = f"  {'g/cm^2':>7}  {'slab':>8}  {'shell':>8}  {'shell/slab':>10}  {'ref':>7}  {'shell/ref':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for anchor in anchors:
        p = next((q for q in K["points"] if abs(q["wall_gcm2"] - anchor) < 1e-6), None)
        if p is None:
            print(f"  {anchor:7.3f}  (no committed kernel point at this anchor -- skipped)")
            continue
        slab = _fold(p, organs)
        shell_pt = generate._collect_point(runs_dir, MATERIAL, anchor, seeds, COMPOSITION)
        if all(v == 0.0 for sp in shell_pt["species"].values()
               for v in sp["R"]["skin"]["D"]):
            print(f"  {anchor:7.3f}  (no shell scorer output at this anchor -- run run_all.sh)")
            continue
        shell = _fold(shell_pt, organs)
        ref = OLTARIS_DABS_MGY_DAY.get(round(anchor, 3))
        ratio = shell["effective"] / slab["effective"]
        refcol = f"{ref:.3f}" if ref else "--"
        srref = f"{shell['effective']/ref:.2f}x" if ref else "--"
        print(f"  {anchor:7.3f}  {slab['effective']:8.4f}  {shell['effective']:8.4f}  "
              f"{ratio:9.2f}x  {refcol:>7}  {srref:>9}")
        rows.append({"anchor": anchor, "slab": slab, "shell": shell, "ref": ref,
                     "ratio": ratio})

    if not rows:
        raise SystemExit("no shell anchors folded -- run run_all.sh first")

    # Per-organ detail at the deepest anchor present (where the plunge lives), plus the
    # skin caveat so the wall-lining-vs-central-sphere scorer gap is never mis-read.
    deep = rows[-1]
    a = deep["anchor"]
    print(f"\n  per-organ absorbed mGy/day at {a:g} g/cm^2  (slab -> shell, ratio):")
    for k, wT in organs:
        s = deep["slab"]["per_organ"][k]
        h = deep["shell"]["per_organ"][k]
        tag = "  <- wall-lining in flood, self-shielded here (wT=0.01)" if k == "skin" else ""
        print(f"    {k:8s} (wT={wT:.2f})  {s:.4f} -> {h:.4f}   ({h/s:.2f}x){tag}")

    print("\n  VERDICT:")
    deep_ref = deep["ref"]
    if deep_ref and abs(deep["shell"]["effective"] / deep_ref - 1.0) <= 0.10:
        print(f"  >> DEEP FIX CONFIRMED: shell effective absorbed sits on OLTARIS at "
              f"{a:g} g/cm^2")
        print("     ({:.4f} vs {:.3f}). The slab's over-attenuation is the plunge; the"
              .format(deep["shell"]["effective"], deep_ref))
        print("     dome-faithful concentric shell is the root-cause fix (NOT a CAL).")
    # thin-anchor safety: does the shell balloon where the kernel is currently right?
    thin = [r for r in rows if r["anchor"] < 19.0]
    if thin:
        worst = max(thin, key=lambda r: r["ratio"])
        if worst["ratio"] <= 1.20:
            print(f"  >> THIN REGIME SAFE: largest thin-anchor shell/slab is "
                  f"{worst['ratio']:.2f}x (@ {worst['anchor']:g} g/cm^2) <= 1.20 --")
            print("     a PURE-SHELL regen preserves the thin regime while fixing depth. GO.")
        else:
            print(f"  >> THIN REGIME AT RISK: shell/slab reaches {worst['ratio']:.2f}x "
                  f"(@ {worst['anchor']:g} g/cm^2) in the thin band where the kernel is")
            print("     currently ~right. A flat shell swap would over-read there -- the "
                  "obliquity is")
            print("     THICKNESS-DEPENDENT (dome curvature); derive a per-anchor obliquity"
                  " instead.")
    else:
        print("  (emit the 2.025 and 10 g/cm^2 anchors too to test the thin regime "
              "before a full regen.)")


def _parse_anchors(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("emit", help="write the shell-geometry batch")
    pe.add_argument("outdir")
    pe.add_argument("--seeds", type=int, default=config.SEEDS)
    pe.add_argument("--threads", type=int, default=0, help="TOPAS threads per run (0=all)")
    pe.add_argument("--anchors", type=_parse_anchors, default=ANCHORS_GCM2,
                    help="comma-separated g/cm^2 anchors (default: 2.025,10,50)")
    pr = sub.add_parser("report", help="fold the batch and print the depth bracket")
    pr.add_argument("outdir")
    args = p.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    if args.cmd == "emit":
        emit(outdir, args.seeds, args.threads, args.anchors)
    else:
        report(outdir)


if __name__ == "__main__":
    main()
