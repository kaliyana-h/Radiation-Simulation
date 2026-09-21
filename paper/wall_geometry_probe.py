#!/usr/bin/env python3
"""Wall-geometry bracket probe: is the kernel's deep-shield absorbed-dose plunge a
WALL-OBLIQUITY defect, and does the trusted flood sit BETWEEN the slab and a
zero-obliquity shell?

Why this exists
---------------
The thin-wall kernel's SKIN absorbed dose at 50 g/cm^2 Al is only ~0.16 mGy/day --
0.52x the OLTARIS-validated FLOOD skin absorbed at the same depth (0.3065 mGy/day,
same 750 cm dome, same physics). The Q-bar is right (kernel 2.18 ~ OLTARIS 2.16) and
the same-physics flood gets deep absorbed right, so the plunge is a TRANSPORT-GEOMETRY
defect in absorbed dose, not physics, not Q, not the source ceiling (all ruled out;
see memory crossover-discontinuity).

kernel_gen shields the phantom with a FLAT slab: every upper-hemisphere ring crosses
t/cos(theta), an effective <1/cos>=2.0 obliquity that OVER-attenuates at depth.
config.py "WHY A SLAB, NOT A SHELL" records the opposite extreme -- a concentric shell
wall (zero obliquity) reads HIGH, growing with thickness (2.025->1.32, 10->1.57,
50->3.21x). The truth (the curvature-bounded flood dome) must sit BETWEEN.

This probe MEASURES that bracket at the single 50 g/cm^2 anchor, skin-to-skin:

    slab skin (committed kernel)   ~0.16   <-- over-attenuates (full <1/cos>=2.0)
    flood skin (trusted)            0.3065  <-- TRUTH (curvature-bounded dome)
    shell skin (this run)           ????    <-- zero-obliquity bracket (expect ~0.5)

If  slab < flood < shell  is confirmed, the wall-obliquity model is a SUFFICIENT lever
for the whole gap: the fix is a geometry-derived obliquity between slab and shell (NOT
a CAL, NOT ceiling nodes). The energy-resolved shell R this run produces is the
zero-obliquity endpoint needed to DERIVE that obliquity per anchor. If the flood is
NOT bracketed (e.g. shell still < flood), the gap is not wall geometry alone and lives
in the scorer/fold/normalisation -- fix that instead, before any full regen.

TARGET-BLIND: nothing here feeds a dose value into the physics. The shell geometry is
a transport bracket; the flood/OLTARIS number is only a post-hoc yardstick.

Usage
-----
  # 1) on the laptop: emit the single-anchor shell-geometry batch
  python3 paper/wall_geometry_probe.py emit runs/wallprobe --seeds 4

  # 2) on the PC (after git pull): run the batch
  export TOPAS_G4_DATA_DIR=~/G4Data
  bash runs/wallprobe/run_all.sh

  # 3) either machine: fold and print the bracket
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

ANCHOR_GCM2 = 50.0            # the deep-shield anchor where the plunge is worst
MATERIAL = "aluminium"
COMPOSITION = "corrected"    # matches the shipped kernel (H/He/C/Si/Fe)
PHI_MV = 400.0               # FULL_RUN solar-min; same as the kernel fold & flood
SEC_PER_DAY = 86400.0
FLOOD_SKIN_MGY_DAY = 0.3065  # OLTARIS-validated flood skin absorbed @50 g/cm^2, 750cm dome


def _skin_absorbed_mgy_day(point: dict, organs: list) -> tuple[float, dict]:
    """Fold one anchor point -> (skin absorbed mGy/day, full HD dict Gy/s)."""
    f = dosimetry._thinwall_fold_point(point, PHI_MV, organs)
    return f["HD"]["skin"] * SEC_PER_DAY * 1e3, f["HD"]


def emit(outdir: Path, seeds: int, threads: int) -> None:
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    species = config.species_for(COMPOSITION)
    manifest = {"material": MATERIAL, "seeds": seeds, "threads": threads,
                "composition": COMPOSITION, "grid_gcm2": [ANCHOR_GCM2],
                "wall_geometry": "shell", "phi_ff_cm2": config.PHI_FF_CM2,
                "note": "DIAGNOSTIC shell-wall bracket @50 g/cm^2 -- NOT a shippable kernel",
                "runs": []}
    n = 0
    for sname, s in species.items():
        for node in range(len(s["nodes_pernuc_mev"])):
            for seed in range(1, seeds + 1):
                rn = generate._run_name(MATERIAL, ANCHOR_GCM2, sname, node, seed)
                rdir = runs_dir / rn
                rdir.mkdir(exist_ok=True)
                text = templates.build_param_file(
                    MATERIAL, ANCHOR_GCM2, sname, node, seed, threads,
                    COMPOSITION, wall_geometry="shell")
                (rdir / "run.txt").write_text(text)
                manifest["runs"].append({"name": rn, "wall_gcm2": ANCHOR_GCM2,
                                         "species": sname, "node": node, "seed": seed})
                n += 1
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    generate._write_runner(outdir)
    print(f"emitted {n} SHELL-geometry runs @ {ANCHOR_GCM2:g} g/cm^2 Al "
          f"(composition={COMPOSITION}, seeds={seeds}) into {outdir}")
    print(f"  species={list(species)}  wall_geometry=shell (zero-obliquity bracket)")
    print(f"  next: export TOPAS_G4_DATA_DIR=~/G4Data && bash {outdir}/run_all.sh")


def report(outdir: Path) -> None:
    manifest = json.loads((outdir / "manifest.json").read_text())
    seeds = manifest["seeds"]
    runs_dir = outdir / "runs"

    # The kernel fold was validated on the rigidity LIS form; pin it so the shell and
    # the committed slab are folded against the IDENTICAL free-field flux (the only
    # difference between the two numbers is then the wall geometry).
    dosimetry.set_lis_form(dosimetry.KERNEL_LIS_FORM)

    K = json.loads((REPO_ROOT / "lunarsim" / "data" / "gcr_thinwall_kernel.json").read_text())
    organs = K["meta"]["organs"]
    p50 = next(p for p in K["points"] if abs(p["wall_gcm2"] - ANCHOR_GCM2) < 1e-9)
    slab_skin, slab_HD = _skin_absorbed_mgy_day(p50, organs)

    shell_pt = generate._collect_point(runs_dir, MATERIAL, ANCHOR_GCM2, seeds, COMPOSITION)
    # Guard: did the runs actually land?
    if all(v == 0.0 for sp in shell_pt["species"].values()
           for v in sp["R"]["skin"]["D"]):
        raise SystemExit(f"no skin scorer output found under {runs_dir} -- run run_all.sh first")
    shell_skin, shell_HD = _skin_absorbed_mgy_day(shell_pt, organs)

    print(f"\n=== wall-geometry SKIN absorbed bracket @ {ANCHOR_GCM2:g} g/cm^2 Al "
          f"(750 cm-equivalent, {COMPOSITION} comp, phi={PHI_MV:g} MV) ===\n")
    print(f"  slab  (committed kernel, <1/cos>=2.0) : {slab_skin:.4f} mGy/day  "
          f"({slab_skin / FLOOD_SKIN_MGY_DAY:.2f}x flood)")
    print(f"  FLOOD (trusted, curvature-bounded)    : {FLOOD_SKIN_MGY_DAY:.4f} mGy/day  (1.00x)")
    print(f"  shell (this run, zero obliquity)      : {shell_skin:.4f} mGy/day  "
          f"({shell_skin / FLOOD_SKIN_MGY_DAY:.2f}x flood)")
    print(f"\n  shell/slab absorbed ratio (skin)      : {shell_skin / slab_skin:.2f}x  "
          f"(config.py records ~3.21x at 50 g/cm^2)")

    # Per-organ, to show the ratio is (or is not) flat across depth as config claims.
    print("\n  per-organ absorbed mGy/day  (slab -> shell, ratio):")
    for k, _wT in organs:
        s = slab_HD[k] * SEC_PER_DAY * 1e3
        h = shell_HD[k] * SEC_PER_DAY * 1e3
        print(f"    {k:8s} {s:.4f} -> {h:.4f}   ({h / s:.2f}x)" if s > 0
              else f"    {k:8s} {s:.4f} -> {h:.4f}")

    print("\n  VERDICT:")
    if slab_skin < FLOOD_SKIN_MGY_DAY < shell_skin:
        print("  >> FLOOD IS BRACKETED  (slab < flood < shell). The wall-obliquity model")
        print("     is a SUFFICIENT lever for the whole deep-shield gap: an effective")
        print("     obliquity BETWEEN the slab's <1/cos>=2.0 and the shell's zero lands on")
        print("     the flood. Fix = geometry-derived per-anchor obliquity (NOT a CAL, NOT")
        print("     ceiling nodes). This run's energy-resolved shell R is the zero-obliquity")
        print("     endpoint to derive it from.")
    elif shell_skin <= FLOOD_SKIN_MGY_DAY:
        print("  >> NOT BRACKETED (shell <= flood). Even zero wall obliquity cannot reach")
        print("     the flood, so the gap is NOT the wall obliquity alone -- it lives in the")
        print("     SCORER geometry (central 15cm phantom skin vs wall-lining skin), the")
        print("     fold, or normalisation. Do NOT regen the wall; chase the scorer/fold.")
    else:  # flood <= slab
        print("  >> flood <= slab: the slab already meets/exceeds the flood at skin; the")
        print("     deep-shield gap is not a slab over-attenuation at skin. Re-examine the")
        print("     depth/organ where the plunge was diagnosed.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("emit", help="write the single-anchor shell-geometry batch")
    pe.add_argument("outdir")
    pe.add_argument("--seeds", type=int, default=config.SEEDS)
    pe.add_argument("--threads", type=int, default=0, help="TOPAS threads per run (0=all)")
    pr = sub.add_parser("report", help="fold the batch and print the bracket")
    pr.add_argument("outdir")
    args = p.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    if args.cmd == "emit":
        emit(outdir, args.seeds, args.threads)
    else:
        report(outdir)


if __name__ == "__main__":
    main()
