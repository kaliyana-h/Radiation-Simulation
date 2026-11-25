#!/usr/bin/env python3
"""Rebuild the shipped per-element kernel's 50 g/cm2 anchor with a 45 cm beam spot.

Why. The shipped kernel illuminates every anchor with a 22 cm disc. The shell's
outer radius is 15 / 15.75 / 18.7 / 33.5 cm at 0 / 2.025 / 10 / 50 g/cm2, so only
the 50 anchor is under-lit: the rim of the wall gets no direct beam and the phantom
loses the lateral in-scatter it would get under broad illumination. The 2026-11-05
geometry probe (paper/geometry_residual_probe.py, 36 seeds) measured the effect as a
skin lift of 1.018 at 10 (null) and 1.213 at 50, with lift_E ~1.

The 0 / 2.025 / 10 anchors are fully covered by the 22 cm spot, so they are copied
through unchanged. Only the 50 anchor is rebuilt:

  base nodes (62)  <- the probe's 36-seed "spot" variant runs (45 cm, 17 histories,
                      skin_mode phantom = the production skin definition)
  tail nodes (20)  <- a new 45 cm batch at 50 g/cm2 only (emit-tail below)

The per-run fluence normalisation is the variant's own phi_ff (45 cm, 17 histories),
so R = D / phi_ff stays on the same scale as the shipped anchors. Species metadata
(z, a, abundance, particle, group) is copied from the shipped kernel, so the fold
model sees exactly what it sees today; only nodes / R / Rsem at 50 change.

TARGET-BLIND: no external dose enters. The acceptance reads are the tool vs itself
(tail.delta) and the kernel/flood residual (paper/tail_vs_flood.py).

The output is a CANDIDATE file. It is not shipped: test_crossover_coherence pins
the shipped numbers, and promotion is the user's decision.

Usage (PC):
  .venv/bin/python paper/kernel_spot45_regen.py emit-tail --out runs/spot45_tail
  export TOPAS_G4_DATA_DIR=~/G4Data
  bash runs/spot45_tail/run_all.sh 2>&1 | tee runs/spot45_tail.log
  .venv/bin/python paper/kernel_spot45_regen.py merge \
      --probe runs/geoprobe36/spot --tail runs/spot45_tail
  .venv/bin/python -m lunarsim.kernel_gen.tail delta \
      --base lunarsim/data/gcr_thinwall_kernel_perelem.json \
      --tail lunarsim/data/gcr_thinwall_kernel_perelem_spot45.json
  .venv/bin/python paper/tail_vs_flood.py \
      --base lunarsim/data/gcr_thinwall_kernel_perelem.json \
      --tail lunarsim/data/gcr_thinwall_kernel_perelem_spot45.json \
      --depths 2.025 5 10 19 30 50 --out paper/spot45_vs_flood.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lunarsim.kernel_gen import config, generate, templates  # noqa: E402

MATERIAL = "aluminium"
ANCHOR = 50.0
SPOT_CM = 45.0
HISTORIES = 17           # ~4*(45/22)^2: same per-run fluence as production
SKIN_MODE = "phantom"    # production skin definition
TAIL_COMPOSITION = "perelement_tail"
TAIL_SEEDS = 8
DATA = REPO_ROOT / "lunarsim" / "data"
SHIPPED = DATA / "gcr_thinwall_kernel_perelem.json"
DEST = DATA / "gcr_thinwall_kernel_perelem_spot45.json"
KNOBS = {"beam_spot_cm": SPOT_CM, "histories": HISTORIES, "skin_mode": SKIN_MODE}


def phi_ff() -> float:
    return templates.phi_ff(SPOT_CM, HISTORIES)


def emit_tail(out: Path, seeds: int = TAIL_SEEDS, threads: int = 1) -> int:
    """Tail nodes at the 50 anchor only, 45 cm spot. Same layout as generate.emit,
    so generate._collect_point and run_all.sh work unchanged."""
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    man = {"material": MATERIAL, "composition": TAIL_COMPOSITION,
           "wall_geometry": "shell", **KNOBS, "phi_ff_cm2": phi_ff(),
           "seeds": seeds, "threads": threads, "grid_gcm2": [ANCHOR], "runs": []}
    for sname, s in config.species_for(TAIL_COMPOSITION).items():
        for j in range(len(s["nodes_pernuc_mev"])):
            for seed in range(1, seeds + 1):
                rn = generate._run_name(MATERIAL, ANCHOR, sname, j, seed)
                (runs / rn).mkdir(exist_ok=True)
                (runs / rn / "run.txt").write_text(templates.build_param_file(
                    MATERIAL, ANCHOR, sname, j, seed, threads, TAIL_COMPOSITION,
                    "shell", **KNOBS))
                man["runs"].append({"name": rn, "wall_gcm2": ANCHOR, "species": sname,
                                    "node": j, "seed": seed})
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    generate._write_runner(out)
    print(f"{len(man['runs'])} tail runs (50 g/cm2, {SPOT_CM:g} cm spot, "
          f"{HISTORIES} histories, {seeds} seeds) -> {out}")
    return len(man["runs"])


def _check_manifest(d: Path, composition: str) -> dict:
    m = json.loads((d / "manifest.json").read_text())
    want = {"material": MATERIAL, "composition": composition, "wall_geometry": "shell",
            **KNOBS}
    bad = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    if bad:
        raise SystemExit(f"{d}: manifest does not match the 45 cm spot setup "
                         f"(got, want): {bad}")
    if ANCHOR not in m["grid_gcm2"]:
        raise SystemExit(f"{d}: no {ANCHOR:g} g/cm2 anchor in {m['grid_gcm2']}")
    return m


def _count_missing(seeds: int, runs_dir: Path, composition: str) -> list:
    f0, _ = templates.scorer_csv_names(config.organs_from_reference()[0][0])
    miss = []
    for sname, s in config.species_for(composition).items():
        for j in range(len(s["nodes_pernuc_mev"])):
            for seed in range(1, seeds + 1):
                rn = generate._run_name(MATERIAL, ANCHOR, sname, j, seed)
                if not (runs_dir / rn / f"{f0}.csv").exists():
                    miss.append(rn)
    return miss


def build_anchor(probe: Path, tail: Path) -> dict:
    """The rebuilt 50 point: probe base nodes + new tail nodes, 45 cm normalisation."""
    mp = _check_manifest(probe, "perelement")
    mt = _check_manifest(tail, TAIL_COMPOSITION)
    for d, m, comp in ((probe, mp, "perelement"), (tail, mt, TAIL_COMPOSITION)):
        miss = _count_missing(m["seeds"], d / "runs", comp)
        if miss:
            raise SystemExit(f"{d}: {len(miss)} runs have no output, e.g. {miss[:3]}")
    pb = generate._collect_point(probe / "runs", MATERIAL, ANCHOR, mp["seeds"],
                                 "perelement", phi_ff())
    pt = generate._collect_point(tail / "runs", MATERIAL, ANCHOR, mt["seeds"],
                                 TAIL_COMPOSITION, phi_ff())
    return {"base": pb, "tail": pt, "seeds_base": mp["seeds"], "seeds_tail": mt["seeds"]}


def merge_into(K: dict, built: dict) -> dict:
    """Replace only the 50 anchor's nodes/R/Rsem; everything else is copied through."""
    K = copy.deepcopy(K)
    organs = [o for o, _ in config.organs_from_reference()]
    (p50,) = [p for p in K["points"] if p["wall_gcm2"] == ANCHOR]
    pb, pt = built["base"]["species"], built["tail"]["species"]
    if set(pb) != set(p50["species"]) or set(pt) != set(p50["species"]):
        raise SystemExit(f"species mismatch: shipped {sorted(p50['species'])}, "
                         f"probe {sorted(pb)}, tail {sorted(pt)}")
    for sname, sp in p50["species"].items():
        nodes = pb[sname]["nodes_pernuc_mev"] + pt[sname]["nodes_pernuc_mev"]
        if [float(x) for x in nodes] != [float(x) for x in sp["nodes_pernuc_mev"]]:
            raise SystemExit(f"{sname}: rebuilt nodes {nodes} != shipped "
                             f"{sp['nodes_pernuc_mev']}")
        for o in organs:
            for q in ("D", "I"):
                sp["R"][o][q] = pb[sname]["R"][o][q] + pt[sname]["R"][o][q]
                sp["Rsem"][o][q] = pb[sname]["Rsem"][o][q] + pt[sname]["Rsem"][o][q]
    p50["seeds"] = built["seeds_base"]
    p50["sem_estimator"] = "sample"
    K["meta"]["anchor50_spot45"] = {
        "beam_spot_cm": SPOT_CM, "histories": HISTORIES, "phi_ff_cm2": phi_ff(),
        "seeds_base_nodes": built["seeds_base"], "seeds_tail_nodes": built["seeds_tail"],
        "provenance": generate._provenance(),
        "note": ("50 g/cm2 anchor rebuilt with a 45 cm beam spot (shell R_out 33.5 cm "
                 "> 22 cm production spot). Other anchors unchanged (fully lit at "
                 "22 cm). Base nodes from paper/geometry_residual_probe.py spot "
                 "variant; tail nodes from paper/kernel_spot45_regen.py emit-tail."),
    }
    return K


def merge(probe: Path, tail: Path, base: Path = SHIPPED, dest: Path = DEST) -> dict:
    K = merge_into(json.loads(base.read_text()), build_anchor(probe, tail))
    dest.write_text(json.dumps(K, indent=2))
    print(f"rebuilt the {ANCHOR:g} g/cm2 anchor of {base.name} -> {dest}")
    return K


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("emit-tail")
    pe.add_argument("--out", type=Path, required=True)
    pe.add_argument("--seeds", type=int, default=TAIL_SEEDS)
    pe.add_argument("--threads", type=int, default=1)
    pm = sub.add_parser("merge")
    pm.add_argument("--probe", type=Path, required=True,
                    help="geometry probe spot-variant dir, e.g. runs/geoprobe36/spot")
    pm.add_argument("--tail", type=Path, required=True)
    pm.add_argument("--base", type=Path, default=SHIPPED)
    pm.add_argument("--dest", type=Path, default=DEST)
    a = ap.parse_args(argv)
    if a.cmd == "emit-tail":
        emit_tail(a.out, a.seeds, a.threads)
    else:
        merge(a.probe, a.tail, a.base, a.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
