"""kernel_gen.let_harness -- EMIT/COLLECT a thin-anchor LET-in-water harness that
cross-checks the thin-wall Q-bar over-read against the measurement arbiter
(CRaTER / arXiv 2511.09040), target-blind.

WHY THIS EXISTS
---------------
The tool's thin-wall dose-averaged <Q> reads ~2.3-2.5x OLTARIS at thin shielding
([[thinwall-qbar-fragmentation]]). Part of that gap is a LEGITIMATE weighting
offset -- our headline <Q> is an organ-effective-dose average, whereas OLTARIS /
CRaTER / the arXiv 2511.09040 Geant4(INCLXX) reference all quote <Q> in a water
POINT. The fall-off SHAPE of <Q>(depth) is the weighting-robust discriminator
(see [[crater-qfactor-depth-reference]]). This harness removes the weighting
confound: it scores LET-in-water and dose-averaged <Q> in a thin water-shell
detector -- the SAME geometry class as the arbiter -- so the tool's water-point
<Q>(depth) curve is directly comparable to flight data + an independent Geant4,
WITHOUT tuning to any dose target.

THREE-PHASE WORKFLOW (identical pattern to kernel_gen.generate)
---------------------------------------------------------------
  EMIT    (this module, pure Python, laptop)  -> one TOPAS param file per
          (material, wall anchor, species, energy node, seed) + manifest.json
          + run_all.sh.  Reuses templates._ring_directions / _ring_groups so the
          illumination is byte-identical to the shipped kernel's flux-correct
          upper-hemisphere ring field.
  RUN     (TOPAS, the PC -- the USER drives this)  bash run_all.sh.  Requires the
          LETSpectrumWater extension to be built (cmake -DTOPAS_EXTENSIONS_DIR
          && make) for the ntuple path; the native DoseToMedium /
          DoseEquivalent_ICRP scorers run with the stock binary.
  COLLECT (this module, pure Python, laptop)  -> folds the per-node detector
          output over the GCR free-field spectrum (reusing dosimetry's flux
          integral) into water-point <Q>(depth) and -- when the ntuple CSVs are
          present -- fluence/dose LET spectra, then prints them beside the
          hardcoded CRaTER / arXiv 2511.09040 reference anchors.

DETECTOR GEOMETRY -- a water POINT, not an organ phantom
--------------------------------------------------------
A THIN water shell (1 mm, RMin=DET_RMIN_CM just inside the wall's RMin=15 cm),
concentric with the shell wall. Two reasons it is a shell and not a small central
sphere: (1) it mirrors the arbiter's own detector (arXiv 2511.09040 = a 1 mm
water shell at 175 mm), so the quantity is the same; (2) a small central sphere
would subtend only ~(r/BeamSpot)^2 ~ 0.8% of the 22 cm ring-disc beam and starve
the LET spectrum, whereas a ~15 cm-radius shell catches the same ~46% the organ
phantom does. Being 1 mm thick it barely degrades the transmitted beam, so the
LET it records is the shielded water-point LET.

TARGET-BLIND: no dose value or limit feeds the MC or the fold. The CRaTER /
arXiv 2511.09040 numbers in REFERENCE_QBAR_DEPTH are POST-HOC overlays only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from . import config, templates

# --------------------------------------------------------------------------
# LET-harness-specific constants (kept OUT of kernel_gen.config so the shipped
# dose kernel's constants are untouched).
# --------------------------------------------------------------------------
# Thin anchors spanning the arbiter's <Q>(depth) fall-off: deep-space 0, a single
# EVA-class wall 2.025, and the 10->~20 g/cm^2 band where the measured <Q> reaches
# its ~3.5 minimum. All are <= the 19 g/cm^2 kernel/flood crossover (the thin
# regime where the over-read lives); thicker walls are the flood regime and are
# not in question here.
LET_GRID_GCM2 = [0.0, 2.025, 10.0, 20.0]

# Default composition -- the shipped corrected H/He/C/Si/Fe table, so the HZE
# (Z>=3) split that drives the over-read is resolved per element.
LET_COMPOSITION = "corrected"

# Spectrum statistics need far more primaries than the dose kernel's 128/node:
# a water POINT catches a thin annulus of the disc beam and we want per-LET-bin
# counts, including the sparse Si/Fe HZE tail. 4x8 rings x 480 = 15360
# primaries/node/seed. (The HZE species get the SAME primary count as H -- their
# low ABUNDANCE enters only the downstream fold weight, never the MC -- so every
# species' per-node spectrum is equally well sampled.)
LET_HISTORIES = 480
LET_SEEDS = 1            # one seed suffices; histories + the multi-node fold average

# Water-POINT detector: a 1 mm shell just inside the wall inner radius.
DET_RMIN_CM = config.WALL_RMIN_CM - 0.2   # 14.8 cm (inside the 15 cm wall RMin)
DET_THICK_CM = 0.1                        # 1 mm, matching arXiv 2511.09040
DET_RMAX_CM = DET_RMIN_CM + DET_THICK_CM  # 14.9 cm

# Free-field planar fluence one node run delivers (per seed), the normalization
# the folded response divides by -- the LET analogue of config.PHI_FF_CM2 but at
# LET_HISTORIES, not HISTORIES.
PHI_FF_LET_CM2 = config.RINGS * config.AZIMUTH * LET_HISTORIES / (math.pi * config.BEAM_SPOT_CM ** 2)

PHI_DEFAULT_MV = 400.0   # solar-minimum modulation (matches the kernel fold, ~ arbiter's 398)

# Detector component + scorer names.
DET_COMP = "DetWater"
SC_LET = "Det_LET"      # LETSpectrum_Water ntuple
SC_D = "Det_D"          # DoseToMedium (absorbed)
SC_I = "Det_I"          # DoseEquivalent_ICRP (in-sim ICRP-60 dose-eq)

# --------------------------------------------------------------------------
# External arbiter -- POST-HOC overlay only (see crater-qfactor-depth-reference).
# Measured + modeled GCR mean quality factor <Q> vs Al shielding depth.
# --------------------------------------------------------------------------
REFERENCE_QBAR_DEPTH = [
    # (Al depth g/cm^2, <Q>, source label)
    (0.22, 5.8, "CRaTER/LRO D1-D2 (lunar orbit)"),
    (0.0, 5.6, "arXiv 2511.09040 model (deep space)"),
    (2.0, 5.8, "arXiv 2511.09040 model"),
    (20.0, 3.49, "TGO/Liulin-MO measured (+-0.26)"),
    (20.0, 3.5, "arXiv 2511.09040 model (minimum)"),
]
# Detector/tissue <Q> anchors behind spacecraft/LEO mass (not depth-resolved):
REFERENCE_QBAR_POINT = [
    (3.82, "MSL-RAD cruise (Zeitlin 2013, +-0.25)"),
    (3.19, "ISS DOSTEL solar-min (+-0.12)"),
]

# LET-spectrum binning for the fluence/dose dPhi/dlnL overlay (keV/um, log).
LET_BINS = [0.1 * (10 ** (i / 10.0)) for i in range(0, 51)]   # 0.1 .. 10^4 keV/um, 10/decade


# ==========================================================================
# ICRP-60 quality factor (identical to DoseEquivalentICRP.cc / the scorer)
# ==========================================================================
def quality_factor_icrp60(let_kev_um: float) -> float:
    L = let_kev_um
    if L < 10.0:
        return 1.0
    if L <= 100.0:
        return 0.32 * L - 2.2
    return 300.0 / math.sqrt(L)


# ==========================================================================
# EMIT
# ==========================================================================
def _run_name(material, wall_gcm2, species, node, seed) -> str:
    return f"let_{material}_w{wall_gcm2:g}_{species}_n{node}_s{seed}"


def enumerate_runs(material: str, grid, seeds: int, composition: str):
    species = config.species_for(composition)
    for wall in grid:
        for sname, s in species.items():
            for node in range(len(s["nodes_pernuc_mev"])):
                for seed in range(1, seeds + 1):
                    yield wall, sname, node, seed


def build_let_param_file(material: str, wall_gcm2: float, species_name: str,
                         node_index: int, seed: int, threads: int,
                         composition: str) -> str:
    """One runnable TOPAS file: flux-correct ring field -> shell wall -> a thin
    water-shell detector carrying LETSpectrum_Water + DoseToMedium +
    DoseEquivalent_ICRP."""
    mat = config.MATERIALS[material]
    species = config.species_for(composition)[species_name]
    nodes = species["nodes_pernuc_mev"]
    e_pernuc = nodes[node_index]
    a = species["a"]
    total_energy_mev = e_pernuc * a
    particle = species["particle"]
    wall_thick = (wall_gcm2 / mat["density"]) if wall_gcm2 > 0 else 0.0
    world_half = config.BEAM_RADIUS_CM + 50.0

    L = []
    L.append("# ============================================================")
    L.append("# Thin-anchor LET-in-water harness (thin-wall Q-bar cross-check)")
    L.append(f"#   material={material} ({mat['topas']})  wall={wall_gcm2:g} g/cm^2"
             f"  thick={wall_thick:.4f} cm")
    L.append(f"#   species={species_name} particle={particle}"
             f"  node={node_index} E={e_pernuc:g} MeV/n  total={total_energy_mev:g} MeV")
    L.append(f"#   seed={seed}  rings{config.RINGS}xaz{config.AZIMUTH}xhist{LET_HISTORIES}"
             f"  phi_ff_let={PHI_FF_LET_CM2:.6g} /cm^2")
    L.append(f"#   detector = 1mm G4_WATER shell r in [{DET_RMIN_CM:g},{DET_RMAX_CM:g}] cm"
             f"  (water POINT, arXiv 2511.09040 class)")
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
    L.append(f"d:Ge/World/HLX       = {world_half:.1f} cm")
    L.append(f"d:Ge/World/HLY       = {world_half:.1f} cm")
    L.append(f"d:Ge/World/HLZ       = {world_half:.1f} cm")
    L.append('b:Ge/World/Invisible = "true"')
    L.append("")
    if mat["defn"]:
        L.append(f"# --- {material} material definition ---")
        L.extend(mat["defn"])
        L.append("")
    # Concentric shell wall (near-radial, zero-obliquity), identical to the
    # shipped kernel geometry; skipped at the 0 g/cm^2 deep-space anchor.
    if wall_gcm2 > 0:
        L.append('s:Ge/Wall/Type     = "TsSphere"')
        L.append('s:Ge/Wall/Parent   = "World"')
        L.append(f's:Ge/Wall/Material = "{mat["topas"]}"')
        L.append(f"d:Ge/Wall/RMin     = {config.WALL_RMIN_CM:.4f} cm")
        L.append(f"d:Ge/Wall/RMax     = {config.WALL_RMIN_CM + wall_thick:.4f} cm")
        L.append('b:Ge/Wall/Invisible = "true"')
        L.append("")
    # Thin water-shell POINT detector + its three scorers.
    L.append(f's:Ge/{DET_COMP}/Type     = "TsSphere"')
    L.append(f's:Ge/{DET_COMP}/Parent   = "World"')
    L.append(f's:Ge/{DET_COMP}/Material = "G4_WATER"')
    L.append(f"d:Ge/{DET_COMP}/RMin     = {DET_RMIN_CM:.4f} cm")
    L.append(f"d:Ge/{DET_COMP}/RMax     = {DET_RMAX_CM:.4f} cm")
    L.append("")
    # LET ntuple (needs the LETSpectrumWater extension built).
    L.append(f's:Sc/{SC_LET}/Quantity   = "LETSpectrum_Water"')
    L.append(f's:Sc/{SC_LET}/Component  = "{DET_COMP}"')
    L.append(f's:Sc/{SC_LET}/OutputFile = "{SC_LET}"')
    L.append(f's:Sc/{SC_LET}/OutputType = "csv"')
    L.append(f's:Sc/{SC_LET}/IfOutputFileAlreadyExists = "Overwrite"')
    L.append("")
    # Native scalar twins -- run with the stock binary, no extension required.
    for sc, quant in ((SC_D, "DoseToMedium"), (SC_I, "DoseEquivalent_ICRP")):
        L.append(f's:Sc/{sc}/Quantity   = "{quant}"')
        L.append(f's:Sc/{sc}/Component  = "{DET_COMP}"')
        L.append(f's:Sc/{sc}/OutputFile = "{sc}"')
        L.append(f's:Sc/{sc}/OutputType = "csv"')
        L.append(f's:Sc/{sc}/IfOutputFileAlreadyExists = "Overwrite"')
    L.append("")
    # Mono-energetic flux-correct upper-hemisphere ring source.
    L.append(f"d:Ge/BeamRadius = {config.BEAM_RADIUS_CM:.1f} cm")
    L.append(f"d:So/BeamSpot   = {config.BEAM_SPOT_CM:.1f} cm")
    L.append("")
    for idx, (phi, theta) in enumerate(templates._ring_directions()):
        grp, src = templates._ring_groups(idx, phi, theta)
        L.extend(grp)
        tag = f"K{idx}"
        L.append(f's:So/{tag}/Type      = "Beam"')
        L.append(f's:So/{tag}/Component = "{src}"')
        L.append(f's:So/{tag}/BeamParticle = "{particle}"')
        L.append(f"d:So/{tag}/BeamEnergy = {total_energy_mev:.6g} MeV")
        L.append(f"u:So/{tag}/BeamEnergySpread = 0.0")
        L.append(f's:So/{tag}/BeamPositionDistribution = "Flat"')
        L.append(f's:So/{tag}/BeamPositionCutoffShape  = "Ellipse"')
        L.append(f"d:So/{tag}/BeamPositionCutoffX      = So/BeamSpot cm")
        L.append(f"d:So/{tag}/BeamPositionCutoffY      = So/BeamSpot cm")
        L.append(f's:So/{tag}/BeamAngularDistribution  = "None"')
        L.append(f"i:So/{tag}/NumberOfHistoriesInRun   = {LET_HISTORIES}")
        L.append("")
    return "\n".join(L) + "\n"


def _write_runner(out: Path) -> None:
    script = out / "run_all.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "# Parallel TOPAS batch for the thin-anchor LET harness. Each run is\n"
        "# single-thread; NPROC runs execute concurrently.\n"
        "# NOTE: the Det_LET ntuple needs the LETSpectrumWater extension built:\n"
        "#   cmake -DTOPAS_EXTENSIONS_DIR=~/topas_extensions && make\n"
        "# The Det_D / Det_I scalars run with the stock binary regardless.\n"
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


def emit(materials, out: Path, grid, seeds: int, threads: int, composition: str) -> None:
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "harness": "let", "materials": list(materials), "grid_gcm2": list(grid),
        "seeds": seeds, "threads": threads, "composition": composition,
        "histories": LET_HISTORIES, "phi_ff_let_cm2": PHI_FF_LET_CM2,
        "detector": {"rmin_cm": DET_RMIN_CM, "rmax_cm": DET_RMAX_CM, "material": "G4_WATER"},
        "scorers": {"let": SC_LET, "dose": SC_D, "doseeq": SC_I},
        "runs": [],
    }
    n = 0
    for material in materials:
        for wall, sname, node, seed in enumerate_runs(material, grid, seeds, composition):
            rn = _run_name(material, wall, sname, node, seed)
            rdir = runs_dir / rn
            rdir.mkdir(exist_ok=True)
            (rdir / "run.txt").write_text(
                build_let_param_file(material, wall, sname, node, seed, threads, composition))
            manifest["runs"].append({"name": rn, "material": material, "wall_gcm2": wall,
                                     "species": sname, "node": node, "seed": seed})
            n += 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_runner(out)
    print(f"emitted {n} LET-harness runs into {out}")
    print(f"  materials={list(materials)}  grid={list(grid)} g/cm^2  composition={composition}")
    print(f"  seeds={seeds}  histories/source={LET_HISTORIES}  (= {config.RINGS*config.AZIMUTH*LET_HISTORIES} primaries/node/seed)")
    print(f"  detector = 1mm G4_WATER shell r in [{DET_RMIN_CM:g},{DET_RMAX_CM:g}] cm  phi_ff_let={PHI_FF_LET_CM2:.6g} /cm^2")
    print(f"  next (PC): cmake -DTOPAS_EXTENSIONS_DIR=~/topas_extensions && make   # once, for the ntuple")
    print(f"             export TOPAS_G4_DATA_DIR=~/G4Data && bash {out}/run_all.sh")


# ==========================================================================
# COLLECT
# ==========================================================================
def _read_scalar_csv(path: Path):
    """Last field of the first data row of a TOPAS whole-volume scorer CSV."""
    if not path.exists():
        return None
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                return float(s.split(",")[-1])
            except ValueError:
                continue
    return None


def _iter_ntuple_rows(path: Path):
    """Yield (let_kev_um, edep_mev, steplen_um, charge) from a LETSpectrum_Water
    ntuple CSV, skipping the TOPAS column-header comment lines."""
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",")
            if len(parts) < 4:
                continue
            try:
                let = float(parts[0]); edep = float(parts[1])
                step = float(parts[2]); chg = int(round(float(parts[3])))
            except ValueError:
                continue
            yield let, edep, step, chg


def _band_edges(nodes):
    """Geometric-mean band edges around the per-nucleon nodes (as the deterministic
    dosimetry fold uses)."""
    return ([nodes[0]]
            + [math.sqrt(nodes[j] * nodes[j + 1]) for j in range(len(nodes) - 1)]
            + [nodes[-1]])


def _flux_weights(species_meta, phi_MV, dosimetry):
    """Per-node free-field fluence weight fw_j = abundance * calib * raw_integral
    over band j, for one species. Mirrors thin_qbar_decomp / _thinwall_fold_point."""
    nodes = species_meta["nodes_pernuc_mev"]
    edges = _band_edges(nodes)
    calib = dosimetry._calibration_factor()
    return [species_meta["abundance"] * calib *
            dosimetry._raw_gcr_integral(species_meta["z"], species_meta["a"], phi_MV,
                                        edges[j], edges[j + 1], n=400)
            for j in range(len(nodes))]


def _collect_material(runs_dir: Path, material: str, grid, seeds: int,
                      composition: str, phi_MV: float, dosimetry) -> dict:
    """Water-point <Q>(depth) for one material, both the native-scorer path
    (always available) and the ntuple path (present only once the extension is
    built + run). Also the dose/fluence LET spectra and the HZE dose fraction."""
    species_meta = config.species_for(composition)
    n_prim = config.RINGS * config.AZIMUTH * LET_HISTORIES   # per seed
    out_points = []
    for wall in grid:
        # Native scalar fold: <Q>_native = sum fw*R_I / sum fw*R_D.
        num_native = den_native = 0.0
        # Ntuple fold: dose-weighted <Q> + LET spectra, split HZE vs light.
        num_nt = den_nt = 0.0
        nbins = len(LET_BINS) - 1
        dose_spec = [0.0] * nbins          # dose-weighted dD/dlnL
        flu_spec = [0.0] * nbins           # fluence-weighted (track length) dPhi/dlnL
        dose_hze = 0.0; dose_light = 0.0
        have_ntuple = False
        for sname, meta in species_meta.items():
            fw = _flux_weights(meta, phi_MV, dosimetry)
            for j in range(len(meta["nodes_pernuc_mev"])):
                # --- native ---
                dvals, ivals = [], []
                for seed in range(1, seeds + 1):
                    rn = _run_name(material, wall, sname, j, seed)
                    dv = _read_scalar_csv(runs_dir / rn / f"{SC_D}.csv")
                    iv = _read_scalar_csv(runs_dir / rn / f"{SC_I}.csv")
                    if dv is not None:
                        dvals.append(dv)
                    if iv is not None:
                        ivals.append(iv)
                if dvals and ivals:
                    R_D = statistics.fmean(dvals) / PHI_FF_LET_CM2
                    R_I = statistics.fmean(ivals) / PHI_FF_LET_CM2
                    num_native += fw[j] * R_I
                    den_native += fw[j] * R_D
                # --- ntuple (pool all seeds) ---
                n_seen_seeds = 0
                for seed in range(1, seeds + 1):
                    rn = _run_name(material, wall, sname, j, seed)
                    p = runs_dir / rn / f"{SC_LET}.csv"
                    if not p.exists():
                        continue
                    n_seen_seeds += 1
                for seed in range(1, seeds + 1):
                    rn = _run_name(material, wall, sname, j, seed)
                    p = runs_dir / rn / f"{SC_LET}.csv"
                    if not p.exists():
                        continue
                    have_ntuple = True
                    # weight per row: node flux / primaries contributing (all pooled seeds)
                    w = fw[j] / (n_prim * max(n_seen_seeds, 1))
                    for let, edep, step, chg in _iter_ntuple_rows(p):
                        if let <= 0.0:
                            continue
                        q = quality_factor_icrp60(let)
                        dw = edep * w
                        num_nt += q * dw
                        den_nt += dw
                        if abs(chg) >= 3:
                            dose_hze += dw
                        else:
                            dose_light += dw
                        b = _bin_index(let)
                        if b is not None:
                            dose_spec[b] += dw
                            flu_spec[b] += step * w
        qbar_native = (num_native / den_native) if den_native else None
        qbar_nt = (num_nt / den_nt) if den_nt else None
        hze_frac = (dose_hze / den_nt) if den_nt else None
        out_points.append({
            "wall_gcm2": wall,
            "qbar_waterpoint_native": qbar_native,
            "qbar_waterpoint_ntuple": qbar_nt if have_ntuple else None,
            "hze_dose_fraction": hze_frac if have_ntuple else None,
            "dose_let_spectrum": dose_spec if have_ntuple else None,
            "fluence_let_spectrum": flu_spec if have_ntuple else None,
        })
    return {"material": material, "phi_MV": phi_MV, "let_bins_kev_um": LET_BINS,
            "points": out_points}


def _bin_index(let):
    if let < LET_BINS[0] or let >= LET_BINS[-1]:
        return None
    # log bins, 10/decade from 0.1
    i = int(10.0 * math.log10(let / LET_BINS[0]))
    if 0 <= i < len(LET_BINS) - 1:
        return i
    return None


def collect(out: Path) -> dict:
    # dosimetry import is deferred so EMIT works without the full package env.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import lunarsim.dosimetry as dosimetry
    dosimetry.set_lis_form(dosimetry.KERNEL_LIS_FORM)   # thin regime == kernel fold

    manifest = json.loads((out / "manifest.json").read_text())
    runs_dir = out / "runs"
    grid = manifest["grid_gcm2"]
    seeds = manifest["seeds"]
    composition = manifest.get("composition", LET_COMPOSITION)
    phi = PHI_DEFAULT_MV
    result = {"harness": "let", "phi_MV": phi, "composition": composition,
              "reference_qbar_depth": REFERENCE_QBAR_DEPTH,
              "reference_qbar_point": REFERENCE_QBAR_POINT, "materials": {}}
    for material in manifest["materials"]:
        result["materials"][material] = _collect_material(
            runs_dir, material, grid, seeds, composition, phi, dosimetry)
    (out / "let_qbar_depth.json").write_text(json.dumps(result, indent=2))
    _print_report(result)
    print(f"\nwrote {out/'let_qbar_depth.json'}")
    return result


def _print_report(result: dict) -> None:
    print("\n=== water-point <Q>(depth): tool vs arbiter (TARGET-BLIND overlay) ===")
    for material, md in result["materials"].items():
        print(f"\n-- {material} (phi={result['phi_MV']:g} MV, {result['composition']}) --")
        print(f"{'wall g/cm^2':>12} {'<Q>native':>10} {'<Q>ntuple':>10} {'HZE dose%':>10}")
        for p in md["points"]:
            qn = p["qbar_waterpoint_native"]
            qt = p["qbar_waterpoint_ntuple"]
            hz = p["hze_dose_fraction"]
            print(f"{p['wall_gcm2']:>12g} "
                  f"{(f'{qn:.2f}' if qn is not None else '   --'):>10} "
                  f"{(f'{qt:.2f}' if qt is not None else '   --'):>10} "
                  f"{(f'{100*hz:.1f}' if hz is not None else '  --'):>10}")
    print("\n  arbiter <Q>(depth) reference anchors (post-hoc, NOT fit targets):")
    for depth, q, label in sorted(result["reference_qbar_depth"]):
        print(f"    {depth:>6g} g/cm^2 Al : <Q>={q:.2f}  [{label}]")
    print("  detector/tissue <Q> anchors (not depth-resolved):")
    for q, label in result["reference_qbar_point"]:
        print(f"    <Q>={q:.2f}  [{label}]")
    print("\n  READ: the weighting-robust test is the SHAPE -- does tool <Q> fall from"
          "\n  ~5.6 (0 g/cm^2) to a ~3.5 minimum near 10-20 g/cm^2? A too-slow fall is"
          "\n  the HZE-over-retention signature. Absolute offset at 0 is partly the"
          "\n  legitimate water-point-vs-effective-dose weighting (control before quoting).")


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("emit", help="write TOPAS param files + manifest + run_all.sh")
    pe.add_argument("-o", "--out", type=Path, required=True, help="output batch dir")
    pe.add_argument("-m", "--materials", nargs="+", default=["aluminium"],
                    choices=list(config.MATERIALS), help="wall materials (default: aluminium)")
    pe.add_argument("--grid", type=float, nargs="+", default=LET_GRID_GCM2,
                    help=f"areal-density anchors g/cm^2 (default {LET_GRID_GCM2})")
    pe.add_argument("--seeds", type=int, default=LET_SEEDS)
    pe.add_argument("--threads", type=int, default=1)
    pe.add_argument("--composition", default=LET_COMPOSITION, choices=config.COMPOSITIONS)

    pc = sub.add_parser("collect", help="fold detector output -> water-point <Q>(depth) + LET spectra")
    pc.add_argument("-o", "--out", type=Path, required=True, help="batch dir from emit")

    args = ap.parse_args(argv)
    if args.cmd == "emit":
        emit(args.materials, args.out, args.grid, args.seeds, args.threads, args.composition)
    elif args.cmd == "collect":
        collect(args.out)


if __name__ == "__main__":
    main()
