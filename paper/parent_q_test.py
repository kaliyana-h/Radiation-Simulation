#!/usr/bin/env python3
"""Delta-ray parent-Q scorer twin: paired same-run A/B on the thin-wall kernel slab (PC).

QUESTION (code review 2026-11-15, "delta rays at Q=1")
------------------------------------------------------
DoseEquivalent_ICRP scores each step's edep x Q(L) with L the STEPPING particle's
unrestricted LET, so delta electrons above the production cut (0.05 mm, ~57 keV in
water) are scored at their own Q ~ 1. Under a primary-L_inf convention (CRaTER-style
<Q>) that energy belongs to the ion and carries the ion's Q. How big is the difference?

WHY NOT A CUT TEST: paper/delta_ray_cut_test.py raised the phantom e- cut instead, and
the 2026-11-17 energy balance showed that breaks energy loss in this TOPAS/Geant4_Modular
build (whole-phantom E_dep 1.7-6.3x). That approach is abandoned.

DESIGN (scorer-only, transport identical by construction)
----------------------------------------------------------
The UNTOUCHED production param file (templates.build_param_file, FTFP_BERT_HP, default
cuts) plus ONE extra scorer per organ shell:
    Sc/<shell comp>_P   Quantity "DoseEquivalent_ICRP_ParentQ"   -> <shell>_P.csv
alongside the existing _D (DoseToMedium) and _I (DoseEquivalent_ICRP). The twin
(extensions/DoseEquivalentICRPParentQ.cc) is DoseEquivalent_ICRP except that an e-
made by an ionisation process on a step scored with Q > 1 inherits that Q.
  * D is the same file as production, so absorbed dose is unchanged by definition.
  * P/I per shell and node is a PAIRED same-run ratio: it shares every history, so
    its seed scatter is much smaller than an unpaired ratio's.
  * The folded figure is the production fold run twice, with the I files and then
    with the P files in the dose-equivalent slot.
Limitations make P/I a LOWER bound: deltas born in the wall or air keep Q = 1, and
photon-carried energy is not inherited.

CONVENTION CAVEAT: ICRU's D_L over all charged particles arguably already scores
deltas at their own L (what I does). P/I measures how much the convention matters;
it is not a bug fix. Adopting it is a user decision. TARGET-BLIND: nothing is fitted,
no CAL refit, and CRaTER/OLTARIS are not used here.

BUILD (PC, once; the scorer must be compiled in):
    cp extensions/DoseEquivalent*.cc extensions/DoseEquivalent*.hh ~/topas_extensions/
    cd ~/topas && cmake -DTOPAS_EXTENSIONS_DIR=$HOME/topas_extensions . && make -j
Run:
    export TOPAS_G4_DATA_DIR=~/G4Data
    .venv/bin/python paper/parent_q_test.py runs/parentq54 --areal 54 --species H,Fe --seeds 2
    .venv/bin/python paper/parent_q_test.py runs/parentq54 --areal 54        # full fold
Re-running is safe: a node whose D/I/P CSVs all parse is skipped unless --force.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lunarsim.kernel_gen import config, templates  # noqa: E402
import fragmentation_sensitivity as fs  # noqa: E402  (shared kernel-slab machinery)

TOPAS_BIN = REPO_ROOT / "bin" / "topas"
CSV_PATH = HERE / "parent_q_test.csv"
COMPOSITION = "perelement"          # the SHIPPED 10-ion per-element kernel set
WALL_GEOMETRY = "slab"              # the production kernel geometry
QUANTITY = "DoseEquivalent_ICRP_ParentQ"
CFG = "parentq"                     # run-tree subdirectory
SEC_PER_YEAR = 3.15576e7
EXT_CC = REPO_ROOT / "extensions" / "DoseEquivalentICRPParentQ.cc"


def parent_csv_name(shell_name: str) -> str:
    return f"{shell_name}_P"


def scorer_lines(shell_name: str) -> list[str]:
    comp = templates._shell_component_name(shell_name)
    sc = f"{comp}_P"
    return [f's:Sc/{sc}/Quantity   = "{QUANTITY}"',
            f's:Sc/{sc}/Component  = "{comp}"',
            f's:Sc/{sc}/OutputFile = "{parent_csv_name(shell_name)}"',
            f's:Sc/{sc}/OutputType = "csv"',
            f's:Sc/{sc}/IfOutputFileAlreadyExists = "Overwrite"']


def build_text(material: str, wall: float, species: str, node: int, seed: int,
               threads: int) -> str:
    """The production slab file, with each shell's _P scorer inserted after its _I."""
    text = templates.build_param_file(material, wall, species, node, seed,
                                      threads=threads, composition=COMPOSITION,
                                      wall_geometry=WALL_GEOMETRY)
    lines = text.splitlines()
    for name, _, _ in config.SHELLS:
        anchor = (f's:Sc/{templates._shell_component_name(name)}_I/'
                  f'IfOutputFileAlreadyExists = "Overwrite"')
        idx = [i for i, ln in enumerate(lines) if ln == anchor]
        if len(idx) != 1:
            raise RuntimeError(f"expected one line {anchor!r}, found {len(idx)}; "
                               "templates.py scorer emission changed.")
        lines[idx[0] + 1:idx[0] + 1] = scorer_lines(name)
    return "\n".join(lines) + "\n"


def files_present(rundir: Path, organ_names: list[str]) -> bool:
    return fs._organ_files_present(rundir, organ_names) and all(
        fs._read_scalar_csv(rundir / f"{parent_csv_name(o)}.csv") is not None
        for o in organ_names)


def _run_one(rundir: Path, material: str, wall: float, species: str, node: int,
             seed: int, threads: int, env: dict, organ_names: list[str],
             force: bool) -> bool:
    rundir.mkdir(parents=True, exist_ok=True)
    if not force and files_present(rundir, organ_names):
        return True
    (rundir / "run.txt").write_text(build_text(material, wall, species, node, seed,
                                               threads))
    with (rundir / "topas.log").open("w") as log:
        proc = subprocess.run([str(TOPAS_BIN), "run.txt"], cwd=rundir, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not files_present(rundir, organ_names):
        print(f"      FAIL {species}/n{node}/s{seed} (rc={proc.returncode}); "
              f"see {rundir/'topas.log'}")
        return False
    return True


def collect_point(base: Path, seeds: int, meta: dict, organs: list, eq_file) -> dict:
    """fs._collect_point layout (R = csv / PHI_FF, keys D and I), but with the
    dose-equivalent slot read from eq_file(organ) -> basename."""
    organ_names = [o for o, _ in organs]
    out = {}
    for sname, m in meta.items():
        nodes = m["nodes_pernuc_mev"]
        R = {o: {"D": [0.0] * len(nodes), "I": [0.0] * len(nodes)} for o in organ_names}
        Rsem = {o: {"D": [0.0] * len(nodes), "I": [0.0] * len(nodes)} for o in organ_names}
        for j in range(len(nodes)):
            for o in organ_names:
                for key, fbase in (("D", templates.scorer_csv_names(o)[0]),
                                   ("I", eq_file(o))):
                    vals = [v / config.PHI_FF_CM2 for v in
                            (fs._read_scalar_csv(fs._run_dir(base, CFG, sname, j, s)
                                                 / f"{fbase}.csv")
                             for s in range(1, seeds + 1)) if v is not None]
                    if vals:
                        R[o][key][j] = statistics.fmean(vals)
                        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                        Rsem[o][key][j] = sd / math.sqrt(len(vals))
        out[sname] = {"z": m["z"], "a": m["a"], "abundance": m["abundance"],
                      "particle": m["particle"], "group": m["group"],
                      "nodes_pernuc_mev": nodes, "R": R, "Rsem": Rsem}
    return {"wall_gcm2": None, "seeds": seeds, "species": out}


def paired_ratio(rundirs: list[Path], organ: str) -> tuple[float, float, int]:
    """Mean and SEM over seeds of P/I in the SAME run (paired)."""
    rs = []
    for rd in rundirs:
        i = fs._read_scalar_csv(rd / f"{templates.scorer_csv_names(organ)[1]}.csv")
        p = fs._read_scalar_csv(rd / f"{parent_csv_name(organ)}.csv")
        if i and p is not None and i > 0:
            rs.append(p / i)
    if not rs:
        return float("nan"), float("nan"), 0
    if len(rs) == 1:
        return rs[0], float("nan"), 1
    return statistics.fmean(rs), statistics.stdev(rs) / math.sqrt(len(rs)), len(rs)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir")
    p.add_argument("--material", default="aluminium")
    p.add_argument("--areal", type=float, default=54.0,
                   help="slab areal density g/cm^2 (default 54 = reference design)")
    p.add_argument("--species", default="all",
                   help="comma list or 'all' (a subset gives per-node ratios only)")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--phi-mv", type=float, default=400.0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    if not TOPAS_BIN.exists():
        raise SystemExit(f"TOPAS binary not found at {TOPAS_BIN}")
    env = dict(os.environ)
    if "TOPAS_G4_DATA_DIR" not in env:
        raise SystemExit("TOPAS_G4_DATA_DIR is unset: export TOPAS_G4_DATA_DIR=~/G4Data")

    meta_all = config.species_for(COMPOSITION)
    if args.species == "all":
        meta, subset = meta_all, False
    else:
        want = [s.strip() for s in args.species.split(",") if s.strip()]
        bad = [s for s in want if s not in meta_all]
        if bad:
            raise SystemExit(f"unknown species {bad}; choose from {list(meta_all)}")
        meta, subset = {s: meta_all[s] for s in want}, True
    organs = config.organs_from_reference()
    organ_names = [o for o, _ in organs]
    base = Path(args.outdir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    n_runs = sum(len(m["nodes_pernuc_mev"]) for m in meta.values()) * args.seeds
    print("=" * 78)
    print("DELTA-RAY PARENT-Q SCORER TWIN  (convention size; TARGET-BLIND)")
    print("=" * 78)
    print(f"  anchor  : {args.areal:g} g/cm^2 {args.material}, slab, {COMPOSITION}, "
          "production physics + cuts")
    print(f"  species : {', '.join(meta)}"
          + ("   [SUBSET: folded numbers NOT valid, per-node ratios only]" if subset else ""))
    print(f"  runs    : {n_runs}   seeds/node {args.seeds}   phi {args.phi_mv:g} MV\n")

    t0, ok, fail = time.time(), 0, 0
    for sname, m in meta.items():
        for j in range(len(m["nodes_pernuc_mev"])):
            for seed in range(1, args.seeds + 1):
                if _run_one(fs._run_dir(base, CFG, sname, j, seed), args.material,
                            args.areal, sname, j, seed, args.threads, env,
                            organ_names, args.force):
                    ok += 1
                else:
                    fail += 1
        print(f"    {sname:>3}: done", flush=True)
    wall_min = (time.time() - t0) / 60.0

    pt_i = collect_point(base, args.seeds, meta, organs,
                         lambda o: templates.scorer_csv_names(o)[1])
    pt_p = collect_point(base, args.seeds, meta, organs, parent_csv_name)
    for pt in (pt_i, pt_p):
        pt["wall_gcm2"] = args.areal
    f_i = fs._fold_qbar(pt_i, args.phi_mv, organs)
    f_p = fs._fold_qbar(pt_p, args.phi_mv, organs)

    print("=" * 78)
    print(f"  SUMMARY (wT-weighted, production fold)   ok {ok}  fail {fail}  "
          f"{wall_min:.1f} min")
    print("=" * 78)
    print(f"  {'scorer':<10} {'Q-bar_eff':>9} {'E mSv/yr':>9} {'D_eff mGy/yr':>13}")
    for lab, f in (("I (own L)", f_i), ("P (parent)", f_p)):
        print(f"  {lab:<10} {f['q_eff']:>9.3f} {f['E_sv_s']*SEC_PER_YEAR*1e3:>9.1f} "
              f"{f['Deff_gy_s']*SEC_PER_YEAR*1e3:>13.3f}")
    e_ratio = f_p["E_sv_s"] / f_i["E_sv_s"] if f_i["E_sv_s"] > 0 else float("nan")
    print(f"\n  P / I :  E {e_ratio:.4f}   (D_eff identical by construction: "
          f"{f_p['Deff_gy_s'] / f_i['Deff_gy_s']:.4f})")
    print("  per-organ Q-bar  I -> P : " + "   ".join(
        f"{o} {f_i['per_organ_q'][o]:.2f}->{f_p['per_organ_q'][o]:.2f}" for o in organ_names))
    if subset:
        print("  [SUBSET run: the folded numbers are NOT the kernel answer]")

    print("\n  PER-NODE paired P/I (mean ± SEM over seeds, same histories)")
    print(f"   {'sp':>3} {'E MeV/n':>8}  " + "  ".join(f"{o:>14}" for o in organ_names))
    rows = []
    for sname, m in meta.items():
        for j, e in enumerate(m["nodes_pernuc_mev"]):
            rds = [fs._run_dir(base, CFG, sname, j, s) for s in range(1, args.seeds + 1)]
            cells = []
            for o in organ_names:
                r, se, n = paired_ratio(rds, o)
                cells.append(f"{r:>7.4f}±{se:<6.4f}")
                rows.append({"species": sname, "E_MeV_n": e, "organ": o,
                             "P_over_I": f"{r:.5f}", "sem": f"{se:.5f}", "n_seeds": n})
            print(f"   {sname:>3} {e:>8.0f}  " + "  ".join(cells))

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scorer", "areal_gcm2", "seeds", "phi_mv", "subset", "q_eff",
                    "E_mSv_yr", "Deff_mGy_yr", "runs_ok", "runs_fail"])
        for lab, f in (("I", f_i), ("P", f_p)):
            w.writerow([lab, f"{args.areal:g}", args.seeds, f"{args.phi_mv:g}", subset,
                        f"{f['q_eff']:.4f}", f"{f['E_sv_s']*SEC_PER_YEAR*1e3:.2f}",
                        f"{f['Deff_gy_s']*SEC_PER_YEAR*1e3:.4f}", ok, fail])
        if rows:
            w.writerow([])
            w.writerow(list(rows[0]))
            for r in rows:
                w.writerow(list(r.values()))
    print(f"\n  wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
