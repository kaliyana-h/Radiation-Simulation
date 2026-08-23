#!/usr/bin/env python3
"""kernel_gen.generate -- build/collect a thin-wall GCR response kernel.

Reconstructs the offline Monte-Carlo harness for lunarsim's thin-wall GCR kernel.
The assistant builds this; the USER runs the Monte Carlo on the 24-core PC.

Workflow (three phases, so the slow MC is one clean parallel batch):

  1) EMIT   -- write every TOPAS param file + a manifest + a parallel runner.
       python -m lunarsim.kernel_gen.generate emit --material evasuit --out RUNDIR
       python -m lunarsim.kernel_gen.generate emit --material aluminium --out RUNDIR_AL

  2) RUN    -- on the PC (this is the only slow, TOPAS-dependent step):
       export TOPAS_G4_DATA_DIR=~/G4Data
       bash RUNDIR/run_all.sh          # NPROC parallel single-thread TOPAS runs

  3) COLLECT / VALIDATE -- parse the CSVs into a kernel JSON:
       python -m lunarsim.kernel_gen.generate collect  --out RUNDIR      # -> eva kernel
       python -m lunarsim.kernel_gen.generate validate --out RUNDIR_AL   # diff vs committed

VALIDATE is the correctness gate: regenerate the ALUMINIUM kernel and confirm it
reproduces the committed data/gcr_thinwall_kernel.json R values within statistics
BEFORE trusting any EVA number. Only the phantom-shell radii and wall-shell shape
are reconstructed; if they are right, aluminium reproduces; then the identical
harness with the wall material swapped gives a faithful EVA kernel.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from . import config, templates


# --------------------------------------------------------------------------
# run enumeration
# --------------------------------------------------------------------------
def _grid_for(material: str) -> list:
    return config.AL_GRID_GCM2 if material == "aluminium" else config.EVA_GRID_GCM2


def _run_name(material, wall_gcm2, species, node, seed) -> str:
    return f"{material}_w{wall_gcm2:g}_{species}_n{node}_s{seed}"


def enumerate_runs(material: str, seeds: int):
    """Yield (wall_gcm2, species_name, node_index, seed) for every run."""
    species = config.species_from_reference()
    for wall in _grid_for(material):
        for sname, s in species.items():
            for node in range(len(s["nodes_pernuc_mev"])):
                for seed in range(1, seeds + 1):
                    yield wall, sname, node, seed


# --------------------------------------------------------------------------
# EMIT
# --------------------------------------------------------------------------
def emit(material: str, out: Path, seeds: int, threads: int) -> None:
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"material": material, "seeds": seeds, "threads": threads,
                "grid_gcm2": _grid_for(material), "phi_ff_cm2": config.PHI_FF_CM2,
                "runs": []}
    n = 0
    for wall, sname, node, seed in enumerate_runs(material, seeds):
        rn = _run_name(material, wall, sname, node, seed)
        rdir = runs_dir / rn
        rdir.mkdir(exist_ok=True)
        text = templates.build_param_file(material, wall, sname, node, seed, threads)
        (rdir / "run.txt").write_text(text)
        manifest["runs"].append({"name": rn, "wall_gcm2": wall, "species": sname,
                                 "node": node, "seed": seed})
        n += 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_runner(out)
    print(f"emitted {n} runs for material={material} into {out}")
    print(f"  grid={_grid_for(material)} g/cm^2, seeds={seeds}, threads/run={threads}")
    print(f"  next: export TOPAS_G4_DATA_DIR=~/G4Data && bash {out}/run_all.sh")


def _write_runner(out: Path) -> None:
    script = out / "run_all.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "# Parallel TOPAS batch for the thin-wall kernel. Each run is single-thread;\n"
        "# NPROC runs execute concurrently. Override TOPAS / NPROC via the environment.\n"
        "set -uo pipefail\n"
        ': "${TOPAS_G4_DATA_DIR:?set TOPAS_G4_DATA_DIR to your G4Data dir}"\n'
        'export TOPAS=${TOPAS:-$HOME/topas/bin/topas}\n'
        'NPROC=${NPROC:-$(nproc)}\n'
        'cd "$(dirname "$0")/runs"\n'
        'echo "running $(ls -d */ | wc -l) TOPAS jobs, $NPROC at a time..."\n'
        "ls -d */ | sed 's#/##' | xargs -P \"$NPROC\" -I{} bash -c '\n"
        '  cd "{}" && "'"$TOPAS"'" run.txt > topas.log 2>&1 \\\n'
        '    && echo "ok  {}" || echo "FAIL {} (see runs/{}/topas.log)"\n'
        "'\n"
        'echo "batch complete."\n'
    )
    script.chmod(0o755)


# --------------------------------------------------------------------------
# COLLECT
# --------------------------------------------------------------------------
def _read_scalar_csv(path: Path):
    """Last comma-separated field of the first non-comment row (TOPAS whole-volume
    scorer summary). Returns None if the file is missing/empty."""
    if not path.exists():
        return None
    with path.open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                return float(row[-1])
            except ValueError:
                continue
    return None


def _collect_point(runs_dir: Path, material: str, wall: float, seeds: int) -> dict:
    """Assemble one areal-density anchor: per-species R & Rsem over seeds."""
    species_meta = config.species_from_reference()
    organs = [name for name, _ in config.organs_from_reference()]
    out_species = {}
    for sname, meta in species_meta.items():
        nodes = meta["nodes_pernuc_mev"]
        R = {o: {"D": [0.0] * len(nodes), "I": [0.0] * len(nodes)} for o in organs}
        Rsem = {o: {"D": [0.0] * len(nodes), "I": [0.0] * len(nodes)} for o in organs}
        for j in range(len(nodes)):
            for o in organs:
                fd, fi = templates.scorer_csv_names(o)
                for qkey, fbase in (("D", fd), ("I", fi)):
                    vals = []
                    for seed in range(1, seeds + 1):
                        rn = _run_name(material, wall, sname, j, seed)
                        v = _read_scalar_csv(runs_dir / rn / f"{fbase}.csv")
                        if v is not None:
                            vals.append(v / config.PHI_FF_CM2)   # D -> R = D/phi_ff
                    if vals:
                        R[o][qkey][j] = statistics.fmean(vals)
                        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                        Rsem[o][qkey][j] = sd / math.sqrt(len(vals))
        out_species[sname] = {
            "z": meta["z"], "a": meta["a"], "abundance": meta["abundance"],
            "particle": meta["particle"], "group": meta["group"],
            "nodes_pernuc_mev": nodes, "R": R, "Rsem": Rsem,
        }
    return {"wall_gcm2": wall, "seeds": seeds, "species": out_species}


def collect(out: Path) -> dict:
    manifest = json.loads((out / "manifest.json").read_text())
    material = manifest["material"]
    seeds = manifest["seeds"]
    runs_dir = out / "runs"
    ref_meta = config.load_reference()["meta"]
    points = [_collect_point(runs_dir, material, w, seeds) for w in manifest["grid_gcm2"]]
    kernel = {
        "meta": {
            "description": f"Thin-wall GCR response kernel R=D_organ/Phi_ff, "
                           f"cal_material={material}. Regenerated by lunarsim.kernel_gen.",
            "quantity": ref_meta["quantity"],
            "cal_material": material,
            "phi_ff_cm2": config.PHI_FF_CM2,
            "beam_spot_cm": config.BEAM_SPOT_CM,
            "rings_azimuth": [config.RINGS, config.AZIMUTH],
            "histories": config.HISTORIES,
            "crossover_gcm2": ref_meta["crossover_gcm2"],
            "areal_grid_gcm2": manifest["grid_gcm2"],
            "organs": [list(o) for o in config.organs_from_reference()],
            "note": ref_meta.get("note", ""),
            "phantom_shells_cm": [list(s) for s in config.SHELLS],
            "wall_rmin_cm": config.WALL_RMIN_CM,
        },
        "points": points,
    }
    fname = config.EVA_OUT if material != "aluminium" else config.AL_VALIDATION_OUT
    dest = out / fname
    dest.write_text(json.dumps(kernel, indent=2))
    print(f"collected {material} kernel -> {dest}")
    if material != "aluminium":
        print(f"  to ship: copy to lunarsim/data/{config.EVA_OUT}")
    return kernel


# --------------------------------------------------------------------------
# VALIDATE (aluminium regeneration vs committed kernel)
# --------------------------------------------------------------------------
def validate(out: Path) -> None:
    regen = collect(out)
    ref = config.load_reference()
    if regen["meta"]["cal_material"] != "aluminium":
        print("WARNING: validate expects an aluminium regeneration.")
    organs = [n for n, _ in config.organs_from_reference()]
    ratios = []
    flagged = []
    print("\n=== Al regeneration vs committed kernel (ratio regen/committed) ===")
    for rp, cp in zip(regen["points"], ref["points"]):
        w = cp["wall_gcm2"]
        for sname, cs in cp["species"].items():
            rs = rp["species"][sname]
            for o in organs:
                for q in ("D", "I"):
                    cvals, rvals = cs["R"][o][q], rs["R"][o][q]
                    csem = cs["Rsem"][o][q]
                    for j, (c, r) in enumerate(zip(cvals, rvals)):
                        if c <= 0 and r <= 0:
                            continue
                        if c > 0:
                            ratio = r / c
                            ratios.append(ratio)
                            # z-score against committed value's own seed SEM
                            sem = csem[j] if csem[j] > 0 else abs(c)
                            z = (r - c) / sem
                            if ratio < 0.5 or ratio > 2.0:
                                flagged.append((w, sname, o, q, j, c, r, ratio, z))
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        within2 = sum(1 for x in ratios if 0.5 <= x <= 2.0) / len(ratios)
        gm = math.exp(statistics.fmean(math.log(x) for x in ratios if x > 0))
        print(f"  n compared        : {len(ratios)}")
        print(f"  median ratio      : {med:.3f}")
        print(f"  geo-mean ratio    : {gm:.3f}   (1.0 = unbiased reconstruction)")
        print(f"  within 0.5-2.0x   : {within2*100:.0f}%")
        print(f"  flagged (>2x off) : {len(flagged)}")
        if flagged:
            print("\n  worst offenders (wall,species,organ,q,node: committed -> regen  ratio):")
            for f in sorted(flagged, key=lambda t: abs(math.log(t[7] if t[7] else 1)))[:15]:
                w, sn, o, q, j, c, r, ratio, z = f
                print(f"    {w:>5g} {sn:<3} {o:<8} {q} n{j}: {c:.2e} -> {r:.2e}  x{ratio:.2f}")
        print("\n  Interpretation: a geo-mean ratio near 1.0 with no organ-systematic")
        print("  bias means the phantom-shell radii + wall geometry are reconstructed")
        print("  faithfully. A whole organ skewed high/low => adjust that shell's radii")
        print("  in config.SHELLS and re-run. Only then trust the EVA kernel.")
    else:
        print("  no overlapping non-zero values -- did the runs complete? check topas.log")


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("emit", help="write all TOPAS param files + runner")
    pe.add_argument("--material", required=True, choices=sorted(config.MATERIALS))
    pe.add_argument("--out", required=True, type=Path)
    pe.add_argument("--seeds", type=int, default=config.SEEDS)
    pe.add_argument("--threads", type=int, default=1,
                    help="TOPAS threads PER run (default 1; runs are parallelised across cores)")

    pc = sub.add_parser("collect", help="parse CSVs into a kernel JSON")
    pc.add_argument("--out", required=True, type=Path)

    pv = sub.add_parser("validate", help="collect an Al regen and diff vs committed kernel")
    pv.add_argument("--out", required=True, type=Path)

    pp = sub.add_parser("plan", help="print run counts without writing files")
    pp.add_argument("--material", required=True, choices=sorted(config.MATERIALS))
    pp.add_argument("--seeds", type=int, default=config.SEEDS)

    args = p.parse_args(argv)
    if args.cmd == "emit":
        emit(args.material, args.out, args.seeds, args.threads)
    elif args.cmd == "collect":
        collect(args.out)
    elif args.cmd == "validate":
        validate(args.out)
    elif args.cmd == "plan":
        runs = list(enumerate_runs(args.material, args.seeds))
        nodes = sum(len(s["nodes_pernuc_mev"]) for s in config.species_from_reference().values())
        print(f"material={args.material}  grid={_grid_for(args.material)}")
        print(f"  species-nodes total = {nodes}, seeds={args.seeds}")
        print(f"  total runs = {len(runs)}  (each {config.RINGS*config.AZIMUTH*config.HISTORIES} primaries)")
        print(f"  phi_ff_cm2 = {config.PHI_FF_CM2:.6g}")


if __name__ == "__main__":
    main()
