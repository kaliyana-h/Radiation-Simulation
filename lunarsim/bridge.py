"""TOPAS bridge: HabitatSpec -> runnable parameter file -> executed run -> results.

This is the "connected directly to TOPAS" layer. Given a HabitatSpec and a run
tier (how many primaries), it:
  1. lays out an isolated run directory,
  2. generates the GCR source include at the requested statistics (make_source.py),
  3. copies the shared lunar_environment.txt (regolith + materials) in,
  4. writes a complete, self-contained habitat .txt (header + physics + World +
     generated geometry + scorers + the two includes),
  5. runs the topas binary headless under that directory,
  6. parses the fluence/dose scorer CSVs into a RunResult.

Everything CWD-relative (TOPAS resolves includeFile against the working dir), so
each run is hermetic and many can execute in parallel from their own dirs.
"""
from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .spec import HabitatSpec
from .geometry import build_geometry, build_scorers, _outer_gauge_radius

# ----------------------------------------------------------------------
# Environment / install locations (override via env vars for other hosts)
# ----------------------------------------------------------------------
TOPAS_ROOT = Path(os.environ.get("TOPAS_ROOT", Path.home() / "topas"))
TOPAS_BIN = Path(os.environ.get("TOPAS_BIN", TOPAS_ROOT / "bin" / "topas"))
G4_DATA_DIR = Path(os.environ.get("TOPAS_G4_DATA_DIR", Path.home() / "G4Data"))
MAKE_SOURCE = TOPAS_ROOT / "make_source.py"
ENV_INCLUDE = TOPAS_ROOT / "lunar_environment.txt"

PHYSICS = "FTFP_BERT_HP"          # REDMoon neutron transport (see lunar_environment.txt)

# ----------------------------------------------------------------------
# Source-dome sizing
# ----------------------------------------------------------------------
DEFAULT_BEAM_RADIUS_CM = 1400.0   # sky-dome source distance for the calibrated envelope
DEFAULT_BEAM_SPOT_CM = 900.0      # per-source disc radius (illuminated footprint)
# BeamRadius must clear the outer fluence gauge: each GCR source is a *parallel*
# beam launched from distance BeamRadius and fired inward, so a gauge point P is
# illuminated only when BeamRadius > P.n_hat.  Since P.n_hat <= |P| <= rg, keeping
# BeamRadius > rg guarantees every gauge point stays downstream of every source
# disc; otherwise fluence_outside re-collapses and the dose inflates (the tall/
# thick-cylinder corner found in the geometry audit).  150 cm keeps the whole
# currently-validated envelope (max gauge rg ~= 1200) at exactly 1400 and only
# grows designs whose gauge would otherwise approach or exceed the source plane.
_BEAM_RADIUS_MARGIN_CM = 150.0


def beam_radius_for(spec: HabitatSpec) -> float:
    """Source-dome distance for this design's run.

    Returns the default 1400 cm for every design inside the calibrated envelope
    (outer gauge rg <= 1200 cm) and only grows it for very large enclosures whose
    gauge would otherwise poke past the source plane.  Safe to vary per design:
    the GCR sources are parallel beams through a Vacuum World, so the field at the
    habitat -- and hence D / fluence_outside -- is invariant to BeamRadius (proven
    empirically at +1.4%), meaning no CAL re-anchor is needed.  gauge_corr in
    dosimetry restores fluence_outside to the fixed 800 cm reference independently."""
    return max(DEFAULT_BEAM_RADIUS_CM,
               _outer_gauge_radius(spec) + _BEAM_RADIUS_MARGIN_CM)


# ----------------------------------------------------------------------
# Run configuration
# ----------------------------------------------------------------------
@dataclass
class RunTier:
    """How much statistics to throw at a design. total primaries =
    rings * azimuth * histories_per_source."""
    name: str
    rings: int
    azimuth: int
    histories: int           # per source
    phi_mv: float = 400.0    # solar-minimum modulation (worst case)

    @property
    def total_primaries(self) -> int:
        return self.rings * self.azimuth * self.histories


QUICK_LOOK = RunTier("quick", rings=5, azimuth=8, histories=25)     # ~1e3, ~minutes
FULL_RUN = RunTier("full", rings=6, azimuth=12, histories=140)      # ~1e4
# Deliberately tiny: a viewer wants a legible cascade, not 1e3 overlapping tracks.
VIS_TIER = RunTier("vis", rings=5, azimuth=8, histories=2)          # ~80 primaries
TIERS = {"quick": QUICK_LOOK, "full": FULL_RUN}


# ----------------------------------------------------------------------
# Solar Particle Event scenario
# ----------------------------------------------------------------------
# A GCR run is a *chronic* field scored as a dose RATE over a mission. An SPE is
# a *single acute event*: a fixed total proton fluence (protons/cm^2, integrated
# over the whole event) arriving as a beam/cone from the Sun. So an SPE has no
# per-second flux -- it is normalised to a total event fluence and scored against
# the acute 30-day BFO limit, not an annual rate. This dataclass carries both the
# source-shape parameters (fed to make_source --mode spe) and the real event
# fluence used by dosimetry.assess_spe to set the absolute scale.
@dataclass(frozen=True)
class SPEScenario:
    name: str
    event: str               # named solar-max design event in make_source.SPE_EVENTS
    fluence_cm2: float       # total event proton fluence over the sampled 10-3000
                             #   MeV band, /cm^2 (the phi>30 anchor integrated to band)
    zenith_deg: float = 45.0     # Sun elevation: look-direction from zenith
    azimuth_deg: float = 0.0
    cone_deg: float = 25.0       # angular half-spread of the arriving beam
    r0_MV: Optional[float] = None   # override characteristic rigidity; None -> event default
    e0_MeV: Optional[float] = None  # LEGACY: exp-in-energy spectrum instead of rigidity


# The physically-worst SPE differs by shielding regime (a genuine physics fork).
# The make_source SPE spectra are now exp-in-RIGIDITY (King-1974 form), each event
# anchored to its measured phi(>30 MeV); fluence_cm2 is that anchor integrated over
# the source's sampled 10-3000 MeV band.
#
#   Behind the thick 147 g/cm^2 regolith wall the crew see, protons below ~420 MeV
#   are stopped, so residual deep-organ/BFO dose is driven by the HARD tail. Feb-1956
#   (GLE 5, R0=220 MV) is the hardest modern event: despite ~9x LESS total fluence
#   than Aug-1972, its penetrating fluence above the wall cutoff is ~11x higher, so it
#   is the most dangerous event *inside* the shielded habitat -- the design worst case.
WORST_CASE_SPE = SPEScenario(
    name="Worst-case SPE inside shield (Feb-1956 / GLE 5, hard)",
    event="feb1956", fluence_cm2=1.589e9, zenith_deg=45.0, cone_deg=25.0)

# Soft, extreme-fluence Aug-1972-class event: worst for SKIN / thin shielding / acute
# surface risk, but its soft protons are largely stopped by thick regolith. Kept for
# the crossover comparison (higher skin dose, lower deep dose than Feb-1956).
SOFT_SPE = SPEScenario(
    name="Soft high-fluence SPE (Aug-1972 class)",
    event="aug1972", fluence_cm2=1.386e10, zenith_deg=45.0, cone_deg=25.0)


# Viewer blocks. ParticleType colouring separates the penetrating secondary
# neutrons (green) from the red primary/knock-on protons. Appended only when a
# viewer is requested; headless batch runs never include any of this.
#
# Modes:
#   "qt"     live OpenGL window via Qt        (UseQt=true)  -- richest, but on
#            some WSL GL stacks Qt segfaults during context creation.
#   "oglx"   live OpenGL window via raw GLX   (UseQt=false) -- different code
#            path, often survives where Qt does not.
#   "vrml"   NO live window: writes g4_*.wrl  -- runs fully headless (the proven
#            path), open the file in a Windows 3D viewer. Guaranteed to render.

# Shared trajectory colouring + soft-hash cut (valid for every mode).
_COLOR_LINES = """iv:Gr/Color/cyan    = 4 0   230 230 255
iv:Gr/Color/magenta = 4 255 0   230 255
iv:Gr/Color/white   = 4 255 255 255 255
d:Gr/OnlyIncludeParticlesWithInitialKEAbove = 10. MeV
s:Gr/MyView/ColorBy                    = "ParticleType"
sv:Gr/MyView/ColorByParticleTypeNames  = 8 "proton" "neutron" "gamma" "e-" "e+" "pi+" "pi-" "alpha"
sv:Gr/MyView/ColorByParticleTypeColors = 8 "red" "green" "yellow" "cyan" "magenta" "white" "white" "blue\""""

_OPENGL_BLOCK = f"""
# ---- live OpenGL window (camera lifted from lunar_habitat.txt) ----
s:Gr/MyView/Type                = "OpenGL"
uv:Gr/MyView/UpVector           = 3 0.0 0.0 1.0
d:Gr/MyView/Theta               = 78.0 deg
d:Gr/MyView/Phi                 = 0.0 deg
s:Gr/MyView/Projection          = "Perspective"
d:Gr/MyView/PerspectiveAngle    = 30.0 deg
u:Gr/MyView/Zoom                = 0.40
b:Gr/MyView/IncludeAxes         = "false"
b:Gr/MyView/IncludeTrajectories = "true"
b:Gr/MyView/IncludeStepPoints   = "false"
i:Gr/MyView/TrajectoryLineWidth = 2
i:Gr/MyView/WindowSizeX         = 1600
i:Gr/MyView/WindowSizeY         = 1100

{_COLOR_LINES}
"""

# Headless file output: geometry + trajectories with colour, no GL context.
# View the .heprep in HepRApp (Java) or jas3. The proven-working code path.
_HEPREP_BLOCK = f"""
# ---- HepRep file output (headless; view the .heprep in HepRApp/jas3) ----
s:Gr/MyView/Type                = "HepRepFile"
b:Gr/MyView/IncludeAxes         = "false"
b:Gr/MyView/IncludeTrajectories = "true"
b:Gr/MyView/IncludeStepPoints   = "false"

{_COLOR_LINES}
"""

# Headless CPU ray-trace to a JPEG: geometry only (NO trajectories), but zero
# external viewer -- just open the .jpeg on Windows. Camera from lunar_habitat.txt.
_RAYTRACER_BLOCK = """
# ---- RayTracer JPEG output (headless; geometry only, no particle tracks) ----
s:Gr/MyView/Type             = "RayTracer"
uv:Gr/MyView/UpVector        = 3 0.0 0.0 1.0
d:Gr/MyView/Theta            = 78.0 deg
d:Gr/MyView/Phi              = 0.0 deg
u:Gr/MyView/Zoom             = 0.40
i:Gr/MyView/WindowSizeX      = 1200
i:Gr/MyView/WindowSizeY      = 900
b:Gr/MyView/IncludeAxes      = "false"
"""


def _viewer_block(mode: str) -> str:
    if mode in ("qt", "oglx"):
        return _OPENGL_BLOCK
    if mode == "heprep":
        return _HEPREP_BLOCK
    if mode == "raytracer":
        return _RAYTRACER_BLOCK
    return ""


@dataclass
class RunResult:
    spec: HabitatSpec
    tier: RunTier
    run_dir: Path
    returncode: int
    wall_seconds: float
    dose_gy: Optional[float] = None              # central crew phantom (point dose)
    phantom_doseeq_sv: Optional[float] = None    # central crew phantom, LET-weighted (ICRP-60 Q)
    phantom_doseeq_nasa_sv: Optional[float] = None  # same phantom, NASA/Cucinotta Q twin
    skin_dose_gy: Optional[float] = None         # inner-wall lining (habitat-wide dose)
    skin_doseeq_sv: Optional[float] = None       # inner-wall lining, LET-weighted (ICRP-60 Q)
    skin_doseeq_nasa_sv: Optional[float] = None  # same lining, NASA/Cucinotta Q twin
    neutron_doseeq_fraction: Optional[float] = None  # H_neutron / H_total on the skin lining
    fluence_inside: Optional[float] = None
    fluence_outside: Optional[float] = None
    log_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.dose_gy is not None

    @property
    def transmission(self) -> Optional[float]:
        """Fraction of fluence that penetrated the wall (lower is better)."""
        if self.fluence_inside is None or not self.fluence_outside:
            return None
        return self.fluence_inside / self.fluence_outside


# ----------------------------------------------------------------------
# Parameter-file assembly
# ----------------------------------------------------------------------
def _world_half_cm(spec: HabitatSpec, beam_radius_cm: float,
                   beam_spot_cm: float = 900.0) -> float:
    """World must contain the sky-dome source ring, the habitat, and the 3 m
    regolith column below. Margin to avoid clipping the outer source discs."""
    # a vertical cylinder reaches z = effective_height + wall at its roof; a
    # quonset reaches +/- (effective_height/2 + total_wall) along its axis once
    # the end-cap bulkheads are added -- still bounded by the vertical term below,
    # but both can exceed the (radial) outer radius, so size to the largest extent.
    vertical = spec.effective_height_cm + spec.total_wall_cm
    # a source disc of radius beam_spot on the sky-dome (radius beam_radius) has its
    # farthest emission point at hypot(beam_radius, beam_spot) off-origin along an
    # axis (max of R.sin@ + S.cos@ over zenith angle @); the world must contain it,
    # else TOPAS clips the widest discs and silently under-fills the far sky.
    source_reach = (beam_radius_cm ** 2 + beam_spot_cm ** 2) ** 0.5
    return max(source_reach, spec.outer_radius_cm, vertical, 300.0) + 300.0


def build_parameter_file(spec: HabitatSpec, tier: RunTier,
                         source_include: str, env_include: str,
                         beam_radius_cm: float = 1400.0,
                         beam_spot_cm: float = 900.0,
                         threads: int = 0, seed: int = 1,
                         viewer: str = "", ion_z: int = 0, ion_a: int = 0) -> str:
    """Return the full text of a runnable TOPAS habitat file.

    viewer="" (default) is headless for batch. "qt"/"oglx" open a live OpenGL
    window (Qt vs raw GLX) with PauseBeforeQuit so it stays up. "vrml" stays
    headless but writes g4_*.wrl geometry+trajectory files to open elsewhere."""
    spec.validate()
    half = _world_half_cm(spec, beam_radius_cm, beam_spot_cm)
    live_window = viewer in ("qt", "oglx")
    pause = "true" if live_window else "false"
    use_qt = "true" if viewer == "qt" else "false"
    vis_block = _viewer_block(viewer)
    return f"""# ============================================================
# Auto-generated lunar habitat run  --  design: {spec.safe_name}
#   shape={spec.shape}  inner_r={spec.inner_radius_cm:.1f} cm
#   walls={[ (w.material, w.thickness_cm) for w in spec.walls ]}
#   areal density={spec.areal_density_gcm2():.1f} g/cm2  tier={tier.name}
#   (generated by lunarsim.bridge -- do not hand-edit)
# ============================================================

i:Ts/Seed            = {seed}
i:Ts/NumberOfThreads = {threads}
b:Ts/PauseBeforeQuit = "{pause}"
b:Ts/UseQt           = "{use_qt}"

# Physics set absolutely so it overrides every include chain (see memory:
# lunar-environment-module -- duplicate-physics-chain gotcha).
s:Ph/Default/Type = "{PHYSICS}"

# World
s:Ge/World/Type      = "TsBox"
s:Ge/World/Material  = "Vacuum"
d:Ge/World/HLX       = {half:.1f} cm
d:Ge/World/HLY       = {half:.1f} cm
d:Ge/World/HLZ       = {half:.1f} cm
b:Ge/World/Invisible = "true"

{build_geometry(spec)}

{build_scorers(spec, ion_z, ion_a)}

# Lunar surface: regolith stack + realistic materials
includeFile = {env_include}

# GCR primary source ({tier.total_primaries} primaries, phi={tier.phi_mv:.0f} MV)
includeFile = {source_include}
{vis_block}"""


# ----------------------------------------------------------------------
# Source generation
# ----------------------------------------------------------------------
def generate_source(run_dir: Path, tier: RunTier,
                    beam_radius_cm: float = 1400.0,
                    beam_spot_cm: float = 900.0,
                    particle: str = "proton",
                    ion_z: int = 1, ion_a: int = 1,
                    spe: Optional[SPEScenario] = None) -> Path:
    """Invoke make_source.py to write a source include into run_dir.

    Default (spe=None) writes a GCR source: particle/ion_z/ion_a select the GCR
    species (protons by default); the heavy-ion species are transported one per
    run and summed in dosimetry. When `spe` is given, writes a directional SPE
    cone source instead (protons only), firing the whole tier budget from the one
    solar look-direction so a single-source event matches a GCR batch in cost."""
    if spe is not None:
        out = run_dir / "spe_source.txt"
        cmd = [
            sys.executable, str(MAKE_SOURCE),
            "--mode", "spe",
            "--histories", str(tier.total_primaries),   # one cone -> full budget
            "--spe-zenith", str(spe.zenith_deg),
            "--spe-azimuth", str(spe.azimuth_deg),
            "--spe-cone", str(spe.cone_deg),
            "--spe-event", spe.event,
            "--beam-radius", str(beam_radius_cm),
            "--beam-spot", str(beam_spot_cm),
            "--out", str(out),
        ]
        if spe.r0_MV is not None:            # override event's characteristic rigidity
            cmd += ["--spe-r0", str(spe.r0_MV)]
        if spe.e0_MeV is not None:           # LEGACY exp-in-energy spectrum
            cmd += ["--spe-e0", str(spe.e0_MeV)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out
    out = run_dir / "gcr_source.txt"
    cmd = [
        sys.executable, str(MAKE_SOURCE),
        "--mode", "gcr",
        "--rings", str(tier.rings),
        "--azimuth", str(tier.azimuth),
        "--histories", str(tier.histories),
        "--phi", str(tier.phi_mv),
        "--beam-radius", str(beam_radius_cm),
        "--beam-spot", str(beam_spot_cm),
        "--particle", particle,
        "--ion-z", str(ion_z),
        "--ion-a", str(ion_a),
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


# ----------------------------------------------------------------------
# Scorer CSV parsing
# ----------------------------------------------------------------------
def _read_scalar_csv(path: Path) -> Optional[float]:
    """TOPAS writes a one-line summary CSV for a whole-volume scorer; the
    numeric value is the last comma-separated field on the first data row."""
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


def parse_results(run_dir: Path) -> dict:
    return {
        "dose_gy": _read_scalar_csv(run_dir / "phantom_dose.csv"),
        "phantom_doseeq_sv": _read_scalar_csv(run_dir / "phantom_doseeq.csv"),
        "phantom_doseeq_nasa_sv": _read_scalar_csv(run_dir / "phantom_doseeq_nasa.csv"),
        "skin_dose_gy": _read_scalar_csv(run_dir / "skin_dose.csv"),
        "skin_doseeq_sv": _read_scalar_csv(run_dir / "skin_doseeq.csv"),
        "skin_doseeq_nasa_sv": _read_scalar_csv(run_dir / "skin_doseeq_nasa.csv"),
        # neutron-lineage twin of skin_doseeq (same CrewSkin lining); consumed into
        # neutron_doseeq_fraction below and popped before RunResult construction.
        "skin_doseeq_neutron_sv": _read_scalar_csv(run_dir / "skin_doseeq_neutron.csv"),
        "fluence_inside": _read_scalar_csv(run_dir / "fluence_inside.csv"),
        "fluence_outside": _read_scalar_csv(run_dir / "fluence_outside.csv"),
        # Shape-specific secondary linings, folded into skin_* below and popped
        # before RunResult construction. All None for the dome, whose single
        # hemisphere shell already covers the whole crew envelope.
        #   cylinder -> roof_* (flat roof underside)
        #   quonset  -> capa_*/capb_* (the two flat end bulkheads)
        "roof_dose_gy": _read_scalar_csv(run_dir / "roof_dose.csv"),
        "roof_doseeq_sv": _read_scalar_csv(run_dir / "roof_doseeq.csv"),
        "roof_doseeq_nasa_sv": _read_scalar_csv(run_dir / "roof_doseeq_nasa.csv"),
        "capa_dose_gy": _read_scalar_csv(run_dir / "capa_dose.csv"),
        "capa_doseeq_sv": _read_scalar_csv(run_dir / "capa_doseeq.csv"),
        "capa_doseeq_nasa_sv": _read_scalar_csv(run_dir / "capa_doseeq_nasa.csv"),
        "capb_dose_gy": _read_scalar_csv(run_dir / "capb_dose.csv"),
        "capb_doseeq_sv": _read_scalar_csv(run_dir / "capb_doseeq.csv"),
        "capb_doseeq_nasa_sv": _read_scalar_csv(run_dir / "capb_doseeq_nasa.csv"),
    }


def _crewskin_volumes_cm3(spec: HabitatSpec) -> tuple[float, float]:
    """(barrel-lining, roof-disc) volumes of the cylinder crew-skin scorer, in cm^3.

    Both linings are 2 cm-thick water. TOPAS's DoseToMedium 'Sum' is total dose =
    deposited energy / scoring-volume mass, so a whole-envelope skin dose is the
    mass- (hence, same material, volume-) weighted mean of the two faces. Geometry
    mirrors _cylinder: barrel r=[inner-2, inner] over full height H; roof disc
    r=[0, inner-6] over 2 cm (the inner-6 outer radius clears the InnerShell fluence
    gauge -- keep it in lockstep with the CrewRoof block in _cylinder)."""
    inner = spec.inner_radius_cm
    H = spec.effective_height_cm
    t = 2.0
    v_barrel = math.pi * (inner ** 2 - (inner - t) ** 2) * H
    v_roof = math.pi * (inner - 6.0) ** 2 * t
    return v_barrel, v_roof


def _quonset_skin_volumes_cm3(spec: HabitatSpec) -> tuple[float, float]:
    """(arch-lining, single-end-cap-lining) volumes of the quonset crew-skin scorer.

    Same mass-weighting rationale as _crewskin_volumes_cm3. Geometry mirrors
    _quonset: the arch lining is a half-annulus r=[inner-2, inner] over the full
    tunnel length (2*HL = effective_height); each end cap is a half-disc r=[0,
    inner-6] (the inner-6 radius clears the InnerShell gauge) 2 cm thick. The two
    ends are identical in volume but scored separately, so the caller folds this
    single-cap volume in twice."""
    inner = spec.inner_radius_cm
    length = spec.effective_height_cm          # 2 * HL
    t = 2.0
    v_arch = 0.5 * math.pi * (inner ** 2 - (inner - t) ** 2) * length
    v_cap = 0.5 * math.pi * (inner - 6.0) ** 2 * t
    return v_arch, v_cap


def _mass_weight_into_skin(results: dict, primary_vol: float,
                           secondaries: list[tuple]) -> None:
    """Fold secondary lining doses into skin_dose_gy / skin_doseeq_sv in place.

    `secondaries` is a list of (dose_gy, doseeq_sv, doseeq_nasa_sv, volume_cm3).
    Because TOPAS's 'Sum' dose is deposited energy / scoring-volume mass, the
    whole-envelope average is the mass- (same material, so volume-) weighted mean:
        D_env = sum(D_i * V_i) / sum(V_i)
    over the primary (arch/barrel) plus every present secondary. DoseEquivalent
    combines identically since DE*m = sum(E_k*Q_k) is additive (holds for either
    Q model). Dose and each dose-equivalent are accumulated independently so a
    secondary missing one CSV (None) drops out of that quantity's weighting
    without corrupting the others."""
    vol_idx = 3
    for key, idx in (("skin_dose_gy", 0), ("skin_doseeq_sv", 1),
                     ("skin_doseeq_nasa_sv", 2)):
        base = results.get(key)
        if base is None:
            continue
        num = base * primary_vol
        den = primary_vol
        for sec in secondaries:
            val = sec[idx]
            if val is not None:
                num += val * sec[vol_idx]
                den += sec[vol_idx]
        results[key] = num / den


def _fold_secondary_into_skin(spec: HabitatSpec, results: dict) -> None:
    """Mass-weight a shape's secondary wall linings into the reported skin dose.

    The base CrewSkin scorer covers only part of each non-dome envelope (the
    cylinder's side wall, the quonset's curved arch); the reported skin_* must be
    the crew's whole-inner-surface average. Secondary keys are always popped so
    RunResult(**results) sees only its own fields; folding happens only for shapes
    that define -- and actually produced CSVs for -- a secondary lining."""
    roof_d = results.pop("roof_dose_gy", None)
    roof_de = results.pop("roof_doseeq_sv", None)
    roof_den = results.pop("roof_doseeq_nasa_sv", None)
    capa_d = results.pop("capa_dose_gy", None)
    capa_de = results.pop("capa_doseeq_sv", None)
    capa_den = results.pop("capa_doseeq_nasa_sv", None)
    capb_d = results.pop("capb_dose_gy", None)
    capb_de = results.pop("capb_doseeq_sv", None)
    capb_den = results.pop("capb_doseeq_nasa_sv", None)
    if spec.shape == "cylinder":
        v_wall, v_roof = _crewskin_volumes_cm3(spec)
        _mass_weight_into_skin(results, v_wall,
                               [(roof_d, roof_de, roof_den, v_roof)])
    elif spec.shape == "quonset":
        v_arch, v_cap = _quonset_skin_volumes_cm3(spec)
        _mass_weight_into_skin(results, v_arch,
                               [(capa_d, capa_de, capa_den, v_cap),
                                (capb_d, capb_de, capb_den, v_cap)])


# ----------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------
def run_design(spec: HabitatSpec, tier: RunTier = QUICK_LOOK,
               run_dir: Optional[Path] = None, threads: int = 0,
               seed: int = 1, keep: bool = True,
               particle: str = "proton", ion_z: int = 1, ion_a: int = 1,
               spe: Optional[SPEScenario] = None) -> RunResult:
    """Generate, run, and parse a single design end-to-end (blocking).

    particle/ion_z/ion_a pick the GCR species (default protons). Passing `spe`
    switches the source to a directional solar-particle-event cone (protons); the
    resulting RunResult is scored with dosimetry.assess_spe, not assess()."""
    spec.validate()
    if run_dir is None:
        run_dir = Path(tempfile.mkdtemp(prefix=f"lunarsim_{spec.safe_name}_"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # env include must sit beside the run file (includeFile is CWD-relative)
    shutil.copy(ENV_INCLUDE, run_dir / "lunar_environment.txt")
    # size the source dome to this design so the outer gauge never pokes past it
    beam_radius = beam_radius_for(spec)
    generate_source(run_dir, tier, beam_radius_cm=beam_radius, particle=particle,
                    ion_z=ion_z, ion_a=ion_a, spe=spe)
    source_include = "spe_source.txt" if spe is not None else "gcr_source.txt"

    param_file = run_dir / "run.txt"
    param_file.write_text(build_parameter_file(
        spec, tier, source_include=source_include,
        env_include="lunar_environment.txt", beam_radius_cm=beam_radius,
        threads=threads, seed=seed, ion_z=ion_z, ion_a=ion_a))

    env = dict(os.environ, TOPAS_G4_DATA_DIR=str(G4_DATA_DIR))
    t0 = time.time()
    proc = subprocess.run([str(TOPAS_BIN), "run.txt"], cwd=run_dir,
                          env=env, capture_output=True, text=True)
    wall = time.time() - t0

    results = parse_results(run_dir)
    # Secondary-neutron dose fraction on the PRIMARY CrewSkin lining, taken before
    # secondaries fold in so numerator and denominator share the one basis. It is a
    # ratio of two dose-equivalents from the same run, so flux/gauge normalisation
    # cancels -- a pure diagnostic, no calibration. Exact for the dome (single
    # shell, no secondary lining); representative for cylinder/quonset, whose
    # secondary linings fold into the reported skin dose but not this ratio.
    _neu = results.pop("skin_doseeq_neutron_sv", None)
    _base_eq = results.get("skin_doseeq_sv")
    results["neutron_doseeq_fraction"] = (
        _neu / _base_eq if (_neu is not None and _base_eq) else None)
    _fold_secondary_into_skin(spec, results)
    log_tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    result = RunResult(spec=spec, tier=tier, run_dir=run_dir,
                       returncode=proc.returncode, wall_seconds=wall,
                       log_tail=log_tail, **results)
    if not keep and result.ok:
        shutil.rmtree(run_dir, ignore_errors=True)
    return result


def write_vis_run(spec: HabitatSpec, run_dir: Optional[Path] = None,
                  tier: RunTier = VIS_TIER, mode: str = "oglx",
                  beam_radius_cm: Optional[float] = None,
                  beam_spot_cm: float = DEFAULT_BEAM_SPOT_CM, seed: int = 1,
                  filename: str = "run.txt") -> Path:
    """Assemble a habitat run with a viewer, but do NOT execute it --
    visualisation is a foreground, user-launched activity.

    Builds the same design as run_design (multi-layer walls, regolith, GCR
    source) with a low primary count so the cascade is legible. `mode`:
      "oglx" (default) live OpenGL window via raw GLX  -- most robust on WSL,
      "qt"            live OpenGL window via Qt         -- richest if GL is happy,
      "vrml"          headless; writes g4_*.wrl to open in a Windows 3D viewer.
    Returns the path to the run file. Launch a live window yourself:

        export TOPAS_G4_DATA_DIR=~/G4Data DISPLAY=:0
        cd <run_dir> && ~/topas/bin/topas <filename>
    """
    spec.validate()
    if beam_radius_cm is None:
        beam_radius_cm = beam_radius_for(spec)
    if run_dir is None:
        run_dir = TOPAS_ROOT / f"vis_{spec.safe_name}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # includeFile is CWD-relative, so the includes must sit beside the run file
    shutil.copy(ENV_INCLUDE, run_dir / "lunar_environment.txt")
    generate_source(run_dir, tier, beam_radius_cm, beam_spot_cm)

    param_file = run_dir / filename
    param_file.write_text(build_parameter_file(
        spec, tier, source_include="gcr_source.txt",
        env_include="lunar_environment.txt", beam_radius_cm=beam_radius_cm,
        beam_spot_cm=beam_spot_cm,
        threads=1, seed=seed, viewer=mode))   # threads=1: viewer is single-threaded
    return param_file


if __name__ == "__main__":
    # Smoke test: run the default dome at quick-look statistics.
    from .spec import default_spec
    spec = default_spec()
    print(f"Running {spec.name}: {spec.shape} r={spec.inner_radius_cm}cm "
          f"wall={spec.total_wall_cm}cm ({spec.areal_density_gcm2():.0f} g/cm2)")
    res = run_design(spec, QUICK_LOOK)
    print(f"rc={res.returncode}  wall={res.wall_seconds:.1f}s  dir={res.run_dir}")
    print(f"dose={res.dose_gy}  trans={res.transmission}")
    if not res.ok:
        print("--- log tail ---\n" + res.log_tail)
