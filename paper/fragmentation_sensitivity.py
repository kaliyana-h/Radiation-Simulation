#!/usr/bin/env python3
"""HZE fragmentation-model sensitivity: does QMD reduce the thin-wall Q-bar over-read?

Why this exists
---------------
The thin-shield (kernel, <19 g/cm^2) EFFECTIVE dose over-reads OLTARIS because the
mean quality factor is high: kernel Q-bar ~ 5.8 vs OLTARIS ~ 2.3 at 10 g/cm^2 Al
(see memory thinwall-qbar-fragmentation / effective-dose-option-c). That gap is a
Q-bar effect, NOT an absorbed-dose or a normalisation error: Geant4's default ion
inelastic model in FTFP_BERT_HP (G4IonPhysics = binary/abrasion light-ion +
Fritiof) lets a heavy projectile survive further UN-FRAGMENTED through a thin wall,
so more high-LET, high-Q primary track reaches the phantom than HZETRN/OLTARIS
predicts. The tool currently books that ~1.6x spread as "irreducible cross-code
physics".

Quantum Molecular Dynamics (G4IonQMDPhysics, TOPAS module "g4ion-QMD") reproduces
heavy-ion charge-changing / fragmentation cross-sections closer to accelerator beam
data -- it fragments the projectile MORE, which should push Q-bar DOWN. This harness
measures, at one thin anchor, how far Q-bar actually moves when the ONLY thing that
changes is the ion inelastic model. Two clean outcomes, both publishable:
  * Q-bar drops toward ~2.3  -> part of the "irreducible spread" was a physics-list
    choice; the thin regime can be made genuinely more accurate.
  * Q-bar barely moves       -> the spread is real cross-code disagreement even
    against the accelerator-preferred model; the conservatism is defensible.

TARGET-BLIND. Nothing here feeds a dose value, limit, or OLTARIS number into the
physics. The ion model is selected on its agreement with ACCELERATOR fragmentation
data (a physics-model choice), never fitted to a dose. OLTARIS ~2.3 is a post-hoc
yardstick for the trend only. Q-bar = I/D is a per-track ratio in which the free-
field fluence phi_ff and the species abundance cancel exactly, so the measured shift
is the fragmentation model and nothing else.

What it does
------------
For each physics CONFIG and each species/energy-node of the corrected composition
(H/He/C/Si/Fe) at ONE thin areal-density anchor, it regenerates the PRODUCTION
kernel slab param file (lunarsim.kernel_gen.templates.build_param_file, wall_geometry
="slab", composition="corrected") -- byte-identical geometry, illumination and
scorers to the shipped kernel -- splices in the config's physics block, runs TOPAS,
and reads the per-organ DoseToMedium (D) and DoseEquivalent_ICRP (I) scorers. It
then assembles the exact same one-anchor "point" dict generate._collect_point builds
(R = D / phi_ff) and folds it through the PRODUCTION fold
(dosimetry._thinwall_fold_point) so the reported Q-bar is computed identically to
the shipped kernel's -- only the physics list differs between columns.

Physics configs (edit PHYSICS_CONFIGS below to add/remove):
  ftfp     the SHIPPED reference list  s:Ph/Default/Type = "FTFP_BERT_HP".
           This is exactly what the committed kernel uses, so it reproduces the
           ~5.8 baseline and anchors the comparison.
  qmd      FTFP_BERT_HP rebuilt as a modular list with the ion inelastic swapped
           to "g4ion-QMD" (the experiment).
  modular  the SAME modular decomposition but with the DEFAULT ion module "g4ion"
           -- the controlled baseline. Run this too (--configs ftfp,modular,qmd) to
           confirm ftfp ~= modular, i.e. that re-expressing FTFP_BERT_HP as modules
           is itself neutral, so the modular->qmd delta is purely the ion model.
  inclxx   ion inelastic swapped to "g4ion-inclxx" (INCLXX), the other space-
           radiation-preferred model, as a second bracket.

All module strings were confirmed verbatim against the TOPAS user guide
(topas.readthedocs.io, Physics > Modular physics lists) on 2026-09-22.

Run it (on the PC, per-config; QMD/INCLXX on heavy Fe nodes are SLOW -- that cost
is the point):
    export TOPAS_G4_DATA_DIR=~/G4Data
    cd ~/topas
    python3 paper/fragmentation_sensitivity.py runs/frag10 --areal 10 --configs ftfp,qmd

Re-running is safe: a node whose organ CSVs already parse is skipped unless --force.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lunarsim import dosimetry
from lunarsim.kernel_gen import config, templates

TOPAS_BIN = REPO_ROOT / "bin" / "topas"
CSV_PATH = HERE / "fragmentation_sensitivity.csv"
COMPOSITION = "corrected"            # H/He/C/Si/Fe -- matches the shipped corrected kernel
WALL_GEOMETRY = "slab"              # the PRODUCTION geometry (never "shell" here)
OLTARIS_QBAR_REF = 2.3             # post-hoc yardstick ONLY (thin-shield effective Q)

# --- Physics configs ------------------------------------------------------
# FTFP_BERT_HP rebuilt from modules, differing ONLY in the ion inelastic module,
# so modular->{qmd,inclxx} is a controlled single-variable swap. Module strings
# verified against topas.readthedocs.io (Modular physics lists) 2026-09-22.
def _modular(ion_module: str) -> list[str]:
    mods = ["g4em-standard_opt0", "g4h-phy_FTFP_BERT_HP", "g4h-elastic_HP",
            "g4decay", ion_module, "g4stopping", "g4radioactivedecay"]
    joined = " ".join(f'"{m}"' for m in mods)
    return ['s:Ph/Default/Type = "Geant4_Modular"',
            f"sv:Ph/Default/Modules = {len(mods)} {joined}"]


PHYSICS_CONFIGS = {
    "ftfp":    [f's:Ph/Default/Type = "{config.PHYSICS}"'],   # shipped reference list
    "modular": _modular("g4ion"),                             # controlled modular baseline
    "qmd":     _modular("g4ion-QMD"),                         # ion inelastic -> QMD
    "inclxx":  _modular("g4ion-inclxx"),                      # ion inelastic -> INCLXX
}

# The single param line templates.build_param_file emits for the physics list; the
# splice replaces exactly this with a config's block.
_PHYS_TARGET = f's:Ph/Default/Type = "{config.PHYSICS}"'


def _read_scalar_csv(path: Path):
    """Last field of the first non-comment row of a TOPAS whole-volume scorer
    summary (identical convention to kernel_gen.generate._read_scalar_csv)."""
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


def _run_dir(base: Path, cfg: str, species: str, node: int, seed: int) -> Path:
    return base / cfg / f"{species}_n{node}_s{seed}"


def _organ_files_present(rundir: Path, organ_names: list[str]) -> bool:
    for o in organ_names:
        fd, fi = templates.scorer_csv_names(o)
        if _read_scalar_csv(rundir / f"{fd}.csv") is None or \
           _read_scalar_csv(rundir / f"{fi}.csv") is None:
            return False
    return True


def _run_one(rundir: Path, material: str, wall: float, species: str, node: int,
             seed: int, cfg: str, threads: int, env: dict, organ_names: list[str],
             force: bool) -> bool:
    """Generate the spliced param file, run TOPAS, return True on success."""
    rundir.mkdir(parents=True, exist_ok=True)
    if not force and _organ_files_present(rundir, organ_names):
        return True
    text = templates.build_param_file(material, wall, species, node, seed,
                                      threads=threads, composition=COMPOSITION,
                                      wall_geometry=WALL_GEOMETRY)
    n = text.count(_PHYS_TARGET)
    if n != 1:
        raise RuntimeError(f"expected exactly one physics line {_PHYS_TARGET!r} in the "
                           f"generated param file, found {n}; templates.py physics "
                           f"emission changed -- update _PHYS_TARGET.")
    text = text.replace(_PHYS_TARGET, "\n".join(PHYSICS_CONFIGS[cfg]))
    (rundir / "run.txt").write_text(text)
    with (rundir / "topas.log").open("w") as log:
        proc = subprocess.run([str(TOPAS_BIN), "run.txt"], cwd=rundir,
                              env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not _organ_files_present(rundir, organ_names):
        print(f"      FAIL {cfg}/{species}/n{node}/s{seed} "
              f"(rc={proc.returncode}); see {rundir/'topas.log'}")
        return False
    return True


def _collect_point(base: Path, cfg: str, material: str, wall: float, seeds: int,
                   species_meta: dict, organs: list) -> dict:
    """One areal-density anchor as generate._collect_point builds it (R = D/phi_ff),
    but reading only the CONFIG's run tree. organs = [(name, wT), ...]."""
    organ_names = [o for o, _ in organs]
    out_species = {}
    for sname, meta in species_meta.items():
        nodes = meta["nodes_pernuc_mev"]
        R = {o: {"D": [0.0] * len(nodes), "I": [0.0] * len(nodes)} for o in organ_names}
        Rsem = {o: {"D": [0.0] * len(nodes), "I": [0.0] * len(nodes)} for o in organ_names}
        for j in range(len(nodes)):
            for o in organ_names:
                fd, fi = templates.scorer_csv_names(o)
                for qkey, fbase in (("D", fd), ("I", fi)):
                    vals = []
                    for seed in range(1, seeds + 1):
                        rn = _run_dir(base, cfg, sname, j, seed)
                        v = _read_scalar_csv(rn / f"{fbase}.csv")
                        if v is not None:
                            vals.append(v / config.PHI_FF_CM2)
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


def _fold_qbar(point: dict, phi_MV: float, organs: list) -> dict:
    """Fold the anchor with the PRODUCTION fold and reduce to Q-bar figures.

    Returns per-organ Q-bar (I/D), the wT-weighted effective Q-bar (E/D_eff), and the
    effective absorbed & dose-eq rates. The kernel was validated on the rigidity LIS
    form, so pin it before folding (fold_gcr_thinwall does the same)."""
    dosimetry.set_lis_form(dosimetry.KERNEL_LIS_FORM)
    f = dosimetry._thinwall_fold_point(point, phi_MV, organs)
    HI, HD = f["HI"], f["HD"]
    per_organ_q = {k: (HI[k] / HD[k] if HD[k] > 0 else float("nan")) for k, _ in organs}
    E = sum(wT * HI[k] for k, wT in organs)     # Sv/s
    Deff = sum(wT * HD[k] for k, wT in organs)  # Gy/s
    return {"per_organ_q": per_organ_q, "q_eff": (E / Deff if Deff > 0 else float("nan")),
            "E_sv_s": E, "Deff_gy_s": Deff}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir", help="run scratch dir (holds one subtree per config)")
    p.add_argument("--material", default="aluminium", help="wall material (default aluminium)")
    p.add_argument("--areal", type=float, default=10.0,
                   help="thin-wall areal density g/cm^2 (default 10, the Q-bar over-read anchor)")
    p.add_argument("--configs", default="ftfp,qmd",
                   help="comma list from %s (default ftfp,qmd)" % ",".join(PHYSICS_CONFIGS))
    p.add_argument("--species", default="all",
                   help="comma list of species or 'all' (default all -- REQUIRED for a "
                        "valid folded Q-bar; a subset only gives per-node ratios)")
    p.add_argument("--seeds", type=int, default=2,
                   help="seeds averaged per node (default 2; more = tighter Rsem, same Q-bar)")
    p.add_argument("--threads", type=int, default=0, help="TOPAS threads (0 = all cores)")
    p.add_argument("--phi-mv", type=float, default=400.0,
                   help="solar modulation phi for the fold (default 400)")
    p.add_argument("--report-organ", default="mid",
                   help="organ whose per-node Q-bar table is printed (default mid)")
    p.add_argument("--force", action="store_true", help="re-run nodes even if CSVs exist")
    args = p.parse_args(argv)

    if not TOPAS_BIN.exists():
        raise SystemExit(f"TOPAS binary not found at {TOPAS_BIN}; build it first (see CLAUDE.md).")
    env = dict(os.environ)
    if "TOPAS_G4_DATA_DIR" not in env:
        raise SystemExit("TOPAS_G4_DATA_DIR is unset. Run:\n"
                         "    export TOPAS_G4_DATA_DIR=~/G4Data\n"
                         "then re-run (the sampled spectrum + Geant4 data must be on the same env).")

    cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in cfgs:
        if c not in PHYSICS_CONFIGS:
            raise SystemExit(f"unknown config {c!r}; choose from {list(PHYSICS_CONFIGS)}")

    species_meta_all = config.species_for(COMPOSITION)
    if args.species == "all":
        species_meta = species_meta_all
        subset = False
    else:
        want = [s.strip() for s in args.species.split(",") if s.strip()]
        bad = [s for s in want if s not in species_meta_all]
        if bad:
            raise SystemExit(f"unknown species {bad}; choose from {list(species_meta_all)}")
        species_meta = {s: species_meta_all[s] for s in want}
        subset = True

    organs = config.organs_from_reference()          # [(name, wT), ...]
    organ_names = [o for o, _ in organs]
    if args.report_organ not in organ_names:
        raise SystemExit(f"--report-organ {args.report_organ!r} not in {organ_names}")
    base = Path(args.outdir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    n_runs = sum(len(m["nodes_pernuc_mev"]) for m in species_meta.values()) * args.seeds
    print("=" * 78)
    print("HZE FRAGMENTATION-MODEL SENSITIVITY  (TARGET-BLIND: ion model chosen on")
    print("accelerator-data agreement, never fitted to a dose; OLTARIS is post-hoc only)")
    print("=" * 78)
    print(f"  anchor      : {args.areal:g} g/cm^2 {args.material}, slab geometry, composition={COMPOSITION}")
    print(f"  species     : {', '.join(species_meta)}"
          + ("   [SUBSET -> folded Q-bar NOT comparable to the full kernel; "
             "per-node ratios only]" if subset else ""))
    print(f"  configs     : {', '.join(cfgs)}")
    print(f"  seeds/node  : {args.seeds}   |  fold phi = {args.phi_mv:g} MV, "
          f"LIS form = {dosimetry.KERNEL_LIS_FORM}")
    print(f"  ~TOPAS runs : {n_runs} per config  ({n_runs * len(cfgs)} total)")
    print(f"  OLTARIS Q-bar yardstick (post-hoc): ~{OLTARIS_QBAR_REF}\n")

    results = {}
    for cfg in cfgs:
        print(f"--- config {cfg}  [{' | '.join(PHYSICS_CONFIGS[cfg])}]")
        t0 = time.time()
        ok = fail = 0
        for sname, meta in species_meta.items():
            for j in range(len(meta["nodes_pernuc_mev"])):
                for seed in range(1, args.seeds + 1):
                    rd = _run_dir(base, cfg, sname, j, seed)
                    if _run_one(rd, args.material, args.areal, sname, j, seed, cfg,
                                args.threads, env, organ_names, args.force):
                        ok += 1
                    else:
                        fail += 1
            print(f"    {sname:>3}: done ({len(meta['nodes_pernuc_mev'])} nodes x "
                  f"{args.seeds} seeds)", flush=True)
        point = _collect_point(base, cfg, args.material, args.areal, args.seeds,
                               species_meta, organs)
        fold = _fold_qbar(point, args.phi_mv, organs)
        results[cfg] = {"point": point, "fold": fold, "ok": ok, "fail": fail,
                        "wall_min": (time.time() - t0) / 60.0}
        print(f"    -> effective Q-bar = {fold['q_eff']:.3f}   "
              f"(runs ok {ok}, fail {fail}, wall {results[cfg]['wall_min']:.1f} min)\n")

    # ---- comparison table -------------------------------------------------
    SEC_PER_YEAR = 3.15576e7
    print("=" * 78)
    print("  SUMMARY -- effective (wT-weighted) Q-bar, folded exactly as the kernel folds")
    print("=" * 78)
    print(f"  {'config':<9} {'Q-bar_eff':>10} {'vs ftfp':>9} {'vs OLTARIS':>11} "
          f"{'E(mSv/yr)*':>11} {'D_eff(mGy/yr)*':>14}")
    q_ftfp = results.get("ftfp", {}).get("fold", {}).get("q_eff")
    for cfg in cfgs:
        f = results[cfg]["fold"]
        q = f["q_eff"]
        vs_ftfp = f"{q / q_ftfp:.3f}" if q_ftfp else "  --"
        e_msv = f["E_sv_s"] * SEC_PER_YEAR * 1e3
        d_mgy = f["Deff_gy_s"] * SEC_PER_YEAR * 1e3
        print(f"  {cfg:<9} {q:>10.3f} {vs_ftfp:>9} {q / OLTARIS_QBAR_REF:>11.2f} "
              f"{e_msv:>11.1f} {d_mgy:>14.3f}")
    print("  * continuous free-field rates (no mission duty factor); Q-bar is the "
          "phi-robust headline.")
    if "qmd" in results and q_ftfp:
        dq = results["qmd"]["fold"]["q_eff"] - q_ftfp
        closed = (q_ftfp - results["qmd"]["fold"]["q_eff"]) / (q_ftfp - OLTARIS_QBAR_REF) \
            if q_ftfp > OLTARIS_QBAR_REF else float("nan")
        verdict = ("REDUCIBLE: QMD moves Q-bar toward OLTARIS -> part of the thin over-read "
                   "is an ion-model choice." if dq < 0 else
                   "NOT reduced by QMD: the spread survives the accelerator-preferred model "
                   "-> defensible cross-code conservatism.")
        print(f"\n  QMD vs ftfp: dQ-bar = {dq:+.3f}  "
              f"(closes {closed*100:4.0f}% of the ftfp->OLTARIS gap)")
        print(f"  >> {verdict}")

    # ---- per-node transparency table -------------------------------------
    org = args.report_organ
    print(f"\n  PER-NODE raw Q-bar = I/D in the '{org}' organ (phi_ff & abundance cancel):")
    header = "   " + f"{'species':>7} {'E(MeV/n)':>10}" + "".join(f"{c:>9}" for c in cfgs)
    print(header)
    for sname, meta in species_meta.items():
        nodes = meta["nodes_pernuc_mev"]
        for j, e in enumerate(nodes):
            cells = []
            for cfg in cfgs:
                R = results[cfg]["point"]["species"][sname]["R"][org]
                d, i = R["D"][j], R["I"][j]
                cells.append(f"{(i/d):>9.2f}" if d > 0 else f"{'--':>9}")
            print(f"   {sname:>7} {e:>10.0f}" + "".join(cells))

    # ---- CSV --------------------------------------------------------------
    fields = ["config", "areal_gcm2", "material", "seeds", "phi_mv", "q_eff",
              "q_eff_vs_ftfp", "q_eff_vs_oltaris", "E_mSv_yr", "Deff_mGy_yr",
              "runs_ok", "runs_fail", "wall_min"]
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cfg in cfgs:
            f = results[cfg]["fold"]
            q = f["q_eff"]
            w.writerow({
                "config": cfg, "areal_gcm2": f"{args.areal:g}", "material": args.material,
                "seeds": args.seeds, "phi_mv": f"{args.phi_mv:g}", "q_eff": f"{q:.4f}",
                "q_eff_vs_ftfp": f"{q/q_ftfp:.4f}" if q_ftfp else "",
                "q_eff_vs_oltaris": f"{q/OLTARIS_QBAR_REF:.4f}",
                "E_mSv_yr": f"{f['E_sv_s']*SEC_PER_YEAR*1e3:.2f}",
                "Deff_mGy_yr": f"{f['Deff_gy_s']*SEC_PER_YEAR*1e3:.4f}",
                "runs_ok": results[cfg]["ok"], "runs_fail": results[cfg]["fail"],
                "wall_min": f"{results[cfg]['wall_min']:.1f}",
            })
    print(f"\n  wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
