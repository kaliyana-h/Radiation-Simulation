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
from .geometry import build_geometry, build_scorers

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
    skin_dose_gy: Optional[float] = None         # inner-wall lining (habitat-wide dose)
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
def _world_half_cm(spec: HabitatSpec, beam_radius_cm: float) -> float:
    """World must contain the sky-dome source ring, the habitat, and the 3 m
    regolith column below. Margin to avoid clipping the outer source discs."""
    # a vertical cylinder reaches z = effective_height + wall at its roof; a
    # quonset reaches +/- effective_height/2 along its axis -- both can exceed
    # the (radial) outer radius, so size the box to the largest extent.
    vertical = spec.effective_height_cm + spec.total_wall_cm
    return max(beam_radius_cm, spec.outer_radius_cm, vertical, 300.0) + 300.0


def build_parameter_file(spec: HabitatSpec, tier: RunTier,
                         source_include: str, env_include: str,
                         beam_radius_cm: float = 900.0,
                         threads: int = 0, seed: int = 1,
                         viewer: str = "") -> str:
    """Return the full text of a runnable TOPAS habitat file.

    viewer="" (default) is headless for batch. "qt"/"oglx" open a live OpenGL
    window (Qt vs raw GLX) with PauseBeforeQuit so it stays up. "vrml" stays
    headless but writes g4_*.wrl geometry+trajectory files to open elsewhere."""
    spec.validate()
    half = _world_half_cm(spec, beam_radius_cm)
    live_window = viewer in ("qt", "oglx")
    pause = "true" if live_window else "false"
    use_qt = "true" if viewer == "qt" else "false"
    vis_block = _viewer_block(viewer)
    return f"""# ============================================================
# Auto-generated lunar habitat run  --  design: {spec.name}
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

{build_scorers(spec)}

# Lunar surface: regolith stack + realistic materials
includeFile = {env_include}

# GCR primary source ({tier.total_primaries} primaries, phi={tier.phi_mv:.0f} MV)
includeFile = {source_include}
{vis_block}"""


# ----------------------------------------------------------------------
# Source generation
# ----------------------------------------------------------------------
def generate_source(run_dir: Path, tier: RunTier,
                    beam_radius_cm: float = 900.0,
                    beam_spot_cm: float = 500.0) -> Path:
    """Invoke make_source.py to write a GCR source include into run_dir."""
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
        "skin_dose_gy": _read_scalar_csv(run_dir / "skin_dose.csv"),
        "fluence_inside": _read_scalar_csv(run_dir / "fluence_inside.csv"),
        "fluence_outside": _read_scalar_csv(run_dir / "fluence_outside.csv"),
    }


# ----------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------
def run_design(spec: HabitatSpec, tier: RunTier = QUICK_LOOK,
               run_dir: Optional[Path] = None, threads: int = 0,
               seed: int = 1, keep: bool = True) -> RunResult:
    """Generate, run, and parse a single design end-to-end (blocking)."""
    spec.validate()
    if run_dir is None:
        run_dir = Path(tempfile.mkdtemp(prefix=f"lunarsim_{spec.name}_"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # env include must sit beside the run file (includeFile is CWD-relative)
    shutil.copy(ENV_INCLUDE, run_dir / "lunar_environment.txt")
    generate_source(run_dir, tier)

    param_file = run_dir / "run.txt"
    param_file.write_text(build_parameter_file(
        spec, tier, source_include="gcr_source.txt",
        env_include="lunar_environment.txt", threads=threads, seed=seed))

    env = dict(os.environ, TOPAS_G4_DATA_DIR=str(G4_DATA_DIR))
    t0 = time.time()
    proc = subprocess.run([str(TOPAS_BIN), "run.txt"], cwd=run_dir,
                          env=env, capture_output=True, text=True)
    wall = time.time() - t0

    results = parse_results(run_dir)
    log_tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    result = RunResult(spec=spec, tier=tier, run_dir=run_dir,
                       returncode=proc.returncode, wall_seconds=wall,
                       log_tail=log_tail, **results)
    if not keep and result.ok:
        shutil.rmtree(run_dir, ignore_errors=True)
    return result


def write_vis_run(spec: HabitatSpec, run_dir: Optional[Path] = None,
                  tier: RunTier = VIS_TIER, mode: str = "oglx",
                  beam_radius_cm: float = 900.0,
                  beam_spot_cm: float = 500.0, seed: int = 1,
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
    if run_dir is None:
        run_dir = TOPAS_ROOT / f"vis_{spec.name}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # includeFile is CWD-relative, so the includes must sit beside the run file
    shutil.copy(ENV_INCLUDE, run_dir / "lunar_environment.txt")
    generate_source(run_dir, tier, beam_radius_cm, beam_spot_cm)

    param_file = run_dir / filename
    param_file.write_text(build_parameter_file(
        spec, tier, source_include="gcr_source.txt",
        env_include="lunar_environment.txt", beam_radius_cm=beam_radius_cm,
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
