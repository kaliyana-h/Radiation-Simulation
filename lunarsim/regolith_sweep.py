"""lunarsim.regolith_sweep -- thick-regolith GCR dose-vs-depth sweep.

Purpose
-------
Produce the tool's own dose-vs-regolith-depth curve at phi = 475 MV so it can be
overlaid on Akisheva & Gourinat (OLTARIS / Badhwar-O'Neill-2014, 475 MV). The
committed thin-wall response kernel spans only ~2-19 g/cm^2; Akisheva's 0.3-3 m of
regolith is ~55-542 g/cm^2, which needs full MC transport -- this module builds it.

Design (both forks pinned with the user, 2026-08-25)
----------------------------------------------------
* 1-D SLAB, FLUX-CORRECT illumination -- reuses the VALIDATED thin-wall kernel
  geometry (kernel_gen.templates): a flat regolith slab of the true geometric
  depth sits directly above a concentric water organ-phantom, illuminated by the
  flux-correct upper-hemisphere ring pattern whose band-effective <1/cos theta> =
  2.0 (templates._ring_directions). This is the OLTARIS/HZETRN slant-path
  convention -- the apples-to-apples comparison to Akisheva.
* 100 GeV/n ENERGY CEILING -- behind 1-3 m the surviving dose is carried by the
  multi-GeV tail, so the source spectrum is integrated to 1e5 MeV/n, not the
  20 GeV/n default (see project memory: compute-scaling-priorities).
* CLEAN phi_ff NORMALIZATION -- dose is normalised as R = D_organ / phi_ff, the
  same first-principles incident-planar-fluence normalisation the kernel uses
  (phi_ff = primaries / (pi * beam_spot^2)); it does NOT use the habitat flood's
  empirical outer-gauge CAL, which is tied to a curved-habitat gauge geometry.
  Physical dose rate = R * (real per-species flux at 475 MV), summed over species
  exactly as dosimetry.assess_composition sums the habitat flood.

Each (depth, species, seed) is one standalone TOPAS flood run. The USER runs the
batch on the PC (assistant builds the harness, user drives the MC). Then:

    python -m lunarsim.regolith_sweep generate  <outdir>   # writes param files + run.sh
    #  ... user runs  bash <outdir>/run.sh  on the PC ...
    python -m lunarsim.regolith_sweep collect   <outdir>   # -> dose_vs_depth.csv

Regolith column
---------------
The REDMoon Apollo-17 3-layer profile (lunar_environment.txt): 0-22 cm @ 1.76,
22-49 cm @ 2.11, 49+ cm @ 1.78 g/cm^3, measured as DEPTH BELOW THE SURFACE. The
phantom sits at depth `d`; the column above it is the top `d` cm of that profile
(surface layer at the top of the slab, deep layer against the phantom).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from . import dosimetry
from .kernel_gen import config, templates

# --------------------------------------------------------------------------
# Sweep grid + source knobs.
# --------------------------------------------------------------------------
DEPTHS_M = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]   # Akisheva 0.3-3 m + bare reference
PHI_MV = 475.0                                    # Akisheva / BON-2014 modulation
EMAX_PER_NUC_MEV = 1.0e5                           # 100 GeV/n ceiling (tail-dominated)

# Illumination scale -- kept large enough that a grazing ring (theta_eff = 75.5 deg)
# crossing the thickest (3 m) slab still enters the slab TOP face and reaches the
# phantom without leaking out the side. beam_spot stays tight (22 cm) around the
# 15 cm phantom so phi_ff and per-primary efficiency match the committed kernel;
# only the stand-off and slab lateral half-extent are enlarged vs config.
BEAM_RADIUS_CM = 1400.0
BEAM_SPOT_CM = config.BEAM_SPOT_CM        # 22.0 -- identical to the kernel
SLAB_HL_CM = 1400.0                        # grazing sweep through 3 m ~ 1160 cm < 1400
WORLD_HALF_CM = 1550.0

# phi_ff -- incident planar number-fluence per seed, = primaries / (pi beam_spot^2).
# primaries = rings * azimuth * histories (per seed). Depends ONLY on beam_spot and
# the primary count, so R = D/phi_ff is normalisation-consistent with the kernel.
def phi_ff_cm2(histories: int) -> float:
    primaries = config.RINGS * config.AZIMUTH * histories
    return primaries / (math.pi * BEAM_SPOT_CM ** 2)


# --------------------------------------------------------------------------
# REDMoon Apollo-17 regolith column (depth below surface -> material, density).
# Material definition blocks copied verbatim from lunar_environment.txt so the
# transport matches the habitat runs.
# --------------------------------------------------------------------------
_REG_COMPONENTS = ('8 "Oxygen" "Sodium" "Magnesium" "Aluminum" "Silicon" '
                   '"Calcium" "Titanium" "Iron"')
_REG_FRACTIONS = "8 0.4184 0.0026 0.0585 0.0604 0.1885 0.0773 0.0569 0.1374"

# (material name, density g/cm^3, depth_top_cm, depth_bottom_cm below surface)
REGOLITH_LAYERS = [
    ("LunarReg176", 1.76, 0.0, 22.0),
    ("LunarReg211", 2.11, 22.0, 49.0),
    ("LunarReg178", 1.78, 49.0, float("inf")),
]


def _regolith_material_defs() -> list:
    out = []
    for name, rho, _, _ in REGOLITH_LAYERS:
        out += [
            f's:Ma/{name}/State        = "Solid"',
            f"d:Ma/{name}/Density      = {rho} g/cm3",
            f"sv:Ma/{name}/Components  = {_REG_COMPONENTS}",
            f"uv:Ma/{name}/Fractions   = {_REG_FRACTIONS}",
        ]
    return out


def column_for_depth(depth_cm: float):
    """Return (slabs, areal_gcm2) for a column of `depth_cm` above the phantom.

    slabs is a list of (material, density, hlz_cm, centre_z_cm): stacked TsBoxes,
    surface layer on top, deep layer against the phantom north pole (z=WALL_RMIN).
    z_surface = WALL_RMIN + depth; a layer at depth-below-surface [s0, s1] occupies
    z in [z_surface - s1, z_surface - s0]."""
    rmin = config.WALL_RMIN_CM
    z_surface = rmin + depth_cm
    slabs, areal = [], 0.0
    for name, rho, s0, s1 in REGOLITH_LAYERS:
        lo = min(s0, depth_cm)
        hi = min(s1, depth_cm)
        thick = hi - lo
        if thick <= 1e-9:
            continue
        z_top = z_surface - lo
        z_bot = z_surface - hi
        slabs.append((name, rho, thick / 2.0, 0.5 * (z_top + z_bot)))
        areal += rho * thick
    return slabs, areal


# --------------------------------------------------------------------------
# Run naming.
# --------------------------------------------------------------------------
def depth_tag(depth_m: float) -> str:
    return f"{int(round(depth_m * 100)):03d}cm"


def run_name(depth_m: float, species: str, seed: int) -> str:
    return f"d{depth_tag(depth_m)}_{species}_s{seed}"


# --------------------------------------------------------------------------
# Parameter-file emission.
# --------------------------------------------------------------------------
def build_param_file(depth_m: float, species_row, histories: int, seed: int,
                     threads: int = 0) -> str:
    name, particle, z, a, abundance, group = species_row
    depth_cm = depth_m * 100.0
    slabs, areal = column_for_depth(depth_cm)

    # per-nucleon continuous spectrum to 100 GeV/n; TOPAS BeamEnergy = TOTAL KE.
    import make_source as ms
    energies, weights = ms.gcr_spectrum(PHI_MV, z=z, a=a, emax=EMAX_PER_NUC_MEV)

    L = []
    L.append("# ============================================================")
    L.append("# Regolith dose-vs-depth sweep run (Akisheva/OLTARIS cross-check)")
    L.append(f"#   depth={depth_m:g} m ({depth_cm:g} cm)  areal={areal:.1f} g/cm^2")
    L.append(f"#   species={name} particle={particle} (Z={z} A={a}) abund={abundance:g}")
    L.append(f"#   phi={PHI_MV:g} MV  Emax={EMAX_PER_NUC_MEV:g} MeV/n  seed={seed}")
    L.append(f"#   illumination=rings{config.RINGS}x az{config.AZIMUTH}"
             f" x hist{histories}  beam_spot={BEAM_SPOT_CM:g} cm")
    L.append(f"#   phi_ff={phi_ff_cm2(histories):.6g} /cm^2  (generated by lunarsim.regolith_sweep)")
    L.append("# ============================================================")
    L.append("")
    L.append(f"i:Ts/Seed            = {seed}")
    L.append(f"i:Ts/NumberOfThreads = {threads}")
    L.append('b:Ts/PauseBeforeQuit = "False"')
    L.append("i:Ts/ShowHistoryCountAtInterval = 0")
    L.append(f's:Ph/Default/Type = "{config.PHYSICS}"')
    L.append("")
    L.append('s:Ge/World/Type      = "TsBox"')
    L.append('s:Ge/World/Material  = "Vacuum"')
    L.append(f"d:Ge/World/HLX       = {WORLD_HALF_CM:.1f} cm")
    L.append(f"d:Ge/World/HLY       = {WORLD_HALF_CM:.1f} cm")
    L.append(f"d:Ge/World/HLZ       = {WORLD_HALF_CM:.1f} cm")
    L.append('b:Ge/World/Invisible = "true"')
    L.append("")
    L.append("# --- regolith composition (from lunar_environment.txt) ---")
    L.extend(_regolith_material_defs())
    L.append("")
    # Stacked regolith slabs (skip at the 0 m bare-phantom reference point).
    for i, (mat, rho, hlz, cz) in enumerate(slabs):
        L.append(f's:Ge/Reg{i}/Type     = "TsBox"')
        L.append(f's:Ge/Reg{i}/Parent   = "World"')
        L.append(f's:Ge/Reg{i}/Material = "{mat}"')
        L.append(f"d:Ge/Reg{i}/HLX      = {SLAB_HL_CM:.1f} cm")
        L.append(f"d:Ge/Reg{i}/HLY      = {SLAB_HL_CM:.1f} cm")
        L.append(f"d:Ge/Reg{i}/HLZ      = {hlz:.4f} cm")
        L.append(f"d:Ge/Reg{i}/TransZ   = {cz:.4f} cm")
        L.append(f'b:Ge/Reg{i}/Invisible = "true"')
        L.append("")
    # Concentric water organ shells + absorbed & ICRP dose-equivalent scorers.
    for sname, r_in, r_out in config.SHELLS:
        comp = "Shell_" + sname
        L.append(f's:Ge/{comp}/Type     = "TsSphere"')
        L.append(f's:Ge/{comp}/Parent   = "World"')
        L.append(f's:Ge/{comp}/Material = "G4_WATER"')
        L.append(f"d:Ge/{comp}/RMin     = {r_in:.4f} cm")
        L.append(f"d:Ge/{comp}/RMax     = {r_out:.4f} cm")
        f_d, f_i = templates.scorer_csv_names(sname)
        for quant, fbase in (("DoseToMedium", f_d), ("DoseEquivalent_ICRP", f_i)):
            sc = f"{comp}_{'D' if quant == 'DoseToMedium' else 'I'}"
            L.append(f's:Sc/{sc}/Quantity   = "{quant}"')
            L.append(f's:Sc/{sc}/Component  = "{comp}"')
            L.append(f's:Sc/{sc}/OutputFile = "{fbase}"')
            L.append(f's:Sc/{sc}/OutputType = "csv"')
            L.append(f's:Sc/{sc}/IfOutputFileAlreadyExists = "Overwrite"')
        L.append("")
    # Flux-correct upper-hemisphere continuous source.
    L.append(f"d:Ge/BeamRadius = {BEAM_RADIUS_CM:.1f} cm")
    L.append(f"d:So/BeamSpot   = {BEAM_SPOT_CM:.1f} cm")
    L.append("")
    for idx, (phi_deg, theta_deg) in enumerate(templates._ring_directions()):
        grp, src = templates._ring_groups(idx, phi_deg, theta_deg)
        L.extend(grp)
        tag = f"R{idx}"
        L.append(f's:So/{tag}/Type      = "Beam"')
        L.append(f's:So/{tag}/Component = "{src}"')
        L.append(f's:So/{tag}/BeamParticle = "{particle}"')
        L.append(f's:So/{tag}/BeamEnergySpectrumType    = "Continuous"')
        L.append(f"dv:So/{tag}/BeamEnergySpectrumValues  = {len(energies)}"
                 f" {ms._fmt_vec(energies)} MeV")
        L.append(f"uv:So/{tag}/BeamEnergySpectrumWeights = {len(weights)}"
                 f" {ms._fmt_vec(weights)}")
        L.append(f's:So/{tag}/BeamPositionDistribution = "Flat"')
        L.append(f's:So/{tag}/BeamPositionCutoffShape  = "Ellipse"')
        L.append(f"d:So/{tag}/BeamPositionCutoffX      = So/BeamSpot cm")
        L.append(f"d:So/{tag}/BeamPositionCutoffY      = So/BeamSpot cm")
        L.append(f's:So/{tag}/BeamAngularDistribution  = "None"')
        L.append(f"i:So/{tag}/NumberOfHistoriesInRun   = {histories}")
        L.append("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# generate / collect.
# --------------------------------------------------------------------------
def generate(outdir: Path, histories: int, seeds: int, threads: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    species = dosimetry.GCR_COMPOSITION
    runs = []
    for depth_m in DEPTHS_M:
        for sp in species:
            for seed in range(1, seeds + 1):
                rn = run_name(depth_m, sp[0], seed)
                rdir = outdir / rn
                rdir.mkdir(exist_ok=True)
                (rdir / f"{rn}.txt").write_text(
                    build_param_file(depth_m, sp, histories, seed, threads))
                runs.append(rn)
    manifest = {
        "phi_MV": PHI_MV, "emax_per_nuc_MeV": EMAX_PER_NUC_MEV,
        "depths_m": DEPTHS_M, "histories": histories, "seeds": seeds,
        "phi_ff_cm2": phi_ff_cm2(histories),
        "species": [s[0] for s in species], "runs": runs,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = ["#!/usr/bin/env bash",
             "# Regolith dose-vs-depth sweep -- run on the PC (24 cores).",
             "# export TOPAS_G4_DATA_DIR=~/G4Data first.",
             "set -e", f'cd "$(dirname "$0")"', ""]
    for rn in runs:
        lines.append(f'( cd {rn} && ~/topas/bin/topas {rn}.txt )')
    (outdir / "run.sh").write_text("\n".join(lines) + "\n")
    print(f"generated {len(runs)} runs ({len(DEPTHS_M)} depths x {len(species)} "
          f"species x {seeds} seed(s)) in {outdir}")
    print(f"  histories/source={histories}  primaries/species/seed="
          f"{config.RINGS * config.AZIMUTH * histories}  phi_ff={manifest['phi_ff_cm2']:.4g} /cm^2")
    print(f"  run:  bash {outdir/'run.sh'}     then:  python -m lunarsim.regolith_sweep collect {outdir}")


def _read_scalar_csv(path: Path):
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


# Organs reported in the dose-vs-depth curve (skin = entrance, core = deep BFO-like).
_REPORT_ORGANS = ("skin", "core")


def collect(outdir: Path) -> dict:
    manifest = json.loads((outdir / "manifest.json").read_text())
    seeds = manifest["seeds"]
    phi_ff = manifest["phi_ff_cm2"]
    species = dosimetry.GCR_COMPOSITION
    SEC_PER_YEAR = 365.25 * 24 * 3600.0

    rows = []
    for depth_m in manifest["depths_m"]:
        _, areal = column_for_depth(depth_m * 100.0)
        organ = {o: {"D": 0.0, "I": 0.0} for o in _REPORT_ORGANS}
        usable = True
        for sp in species:
            sname, particle, z, a, abundance, group = sp
            flux_i = dosimetry.gcr_species_fluence_rate(z, a, abundance, PHI_MV)
            for o in _REPORT_ORGANS:
                f_d, f_i = templates.scorer_csv_names(o)
                for qk, fbase in (("D", f_d), ("I", f_i)):
                    vals = []
                    for seed in range(1, seeds + 1):
                        rn = run_name(depth_m, sname, seed)
                        v = _read_scalar_csv(outdir / rn / f"{fbase}.csv")
                        if v is not None:
                            vals.append(v)
                    if not vals:
                        usable = False
                        continue
                    # R = D/phi_ff [Gy or Sv cm^2];  rate = R * flux_i [/cm^2/s].
                    R = statistics.fmean(vals) / phi_ff
                    organ[o][qk] += R * flux_i          # Gy/s (D) or Sv/s (I)
        row = {"depth_m": depth_m, "areal_gcm2": round(areal, 1), "usable": usable}
        for o in _REPORT_ORGANS:
            row[f"{o}_absorbed_mGy_yr"] = round(organ[o]["D"] * SEC_PER_YEAR * 1e3, 3)
            row[f"{o}_doseeq_mSv_yr"] = round(organ[o]["I"] * SEC_PER_YEAR * 1e3, 2)
        rows.append(row)

    out_csv = outdir / "dose_vs_depth.csv"
    fields = (["depth_m", "areal_gcm2", "usable"]
              + [f"{o}_absorbed_mGy_yr" for o in _REPORT_ORGANS]
              + [f"{o}_doseeq_mSv_yr" for o in _REPORT_ORGANS])
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"phi={PHI_MV:g} MV  dose-vs-regolith-depth (GCR, {len(species)} species summed)")
    print(f"{'depth':>6} {'areal':>7}  {'skin_mSv/yr':>11} {'core_mSv/yr':>11}  {'':<4}")
    for r in rows:
        flag = "" if r["usable"] else "(incomplete)"
        print(f"{r['depth_m']:6.2f} {r['areal_gcm2']:7.1f}  "
              f"{r['skin_doseeq_mSv_yr']:11.1f} {r['core_doseeq_mSv_yr']:11.1f}  {flag}")
    print(f"\nwrote {out_csv}")
    return {"rows": rows, "csv": str(out_csv)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write per-(depth,species,seed) param files + run.sh")
    g.add_argument("outdir", type=Path)
    g.add_argument("--histories", type=int, default=2000, help="histories per source (32 sources/seed)")
    g.add_argument("--seeds", type=int, default=1, help="seeds per point (>=2 for error bars)")
    g.add_argument("--threads", type=int, default=0, help="TOPAS threads (0 = all cores)")
    c = sub.add_parser("collect", help="fold scored CSVs -> dose_vs_depth.csv")
    c.add_argument("outdir", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "generate":
        generate(args.outdir, args.histories, args.seeds, args.threads)
    else:
        collect(args.outdir)


if __name__ == "__main__":
    main()
