"""kernel_gen.config -- constants for the thin-wall GCR response-kernel generator.

This package RECONSTRUCTS the offline Monte-Carlo harness that produced
``lunarsim/data/gcr_thinwall_kernel.json`` (the aluminium kernel). The original
generator script was never committed -- only its JSON output survived -- so the
few geometry constants it used (organ-shell radii, wall-shell shape) are NOT
recorded anywhere and are reconstructed here from the physics and pinned by an
ALUMINIUM validation run (see generate.py --validate).

What IS pinned exactly, straight from the committed kernel's ``meta`` block:

  * illumination = make_source.gcr_block(rings=4, azimuth=8, histories=4,
    beam_spot=22 cm): an equal-flux Lambert-weighted upper-hemisphere ring
    pattern of parallel disc beams firing inward. 4x8 = 32 sources x 4
    histories = 128 primaries per seed.
  * phi_ff_cm2 = 128 / (pi * 22^2) = 0.08418  (the planar free-field fluence
    the response R = D_organ / phi_ff is normalised to). VERIFIED: this is why
    meta.phi_ff_cm2 == 0.08418 to 4 sig figs.
  * species / per-nucleon energy nodes / abundances / organ wT -- copied
    verbatim from the committed kernel at runtime (see load_reference()) so the
    EVA kernel cannot drift from the Al one.

What is RECONSTRUCTED (defaults below; confirm with --validate before trusting
any EVA number): the water-phantom organ-shell radii and the wall-shell
geometry. These are the only free knobs; the Al validation run reproduces the
committed R values iff they are right.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# --- committed Al kernel: the single source of truth for species/nodes/organs --
_REPO_KERNEL = Path(__file__).resolve().parents[1] / "data" / "gcr_thinwall_kernel.json"


def load_reference() -> dict:
    """The committed aluminium kernel -- species table, energy nodes, organ
    weights and meta are copied FROM here so a regenerated kernel stays aligned."""
    with open(_REPO_KERNEL) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Illumination -- pinned exactly by the committed meta block.
# --------------------------------------------------------------------------
RINGS = 4            # meta.rings_azimuth[0]
AZIMUTH = 8          # meta.rings_azimuth[1]
HISTORIES = 4        # meta.histories  (per source, per seed)
BEAM_SPOT_CM = 22.0  # meta.beam_spot_cm  (per-source disc radius)
BEAM_RADIUS_CM = 200.0  # source stand-off distance; must exceed the wall outer radius

# planar free-field fluence R is normalised to (VERIFY: == meta.phi_ff_cm2).
PHI_FF_CM2 = RINGS * AZIMUTH * HISTORIES / (math.pi * BEAM_SPOT_CM ** 2)

# seeds per areal-density anchor. The committed Al kernel used 4/3/3; we default
# to 4 everywhere (Rsem only -- more seeds = tighter error bars, same R).
SEEDS = 4

PHYSICS = "FTFP_BERT_HP"  # identical to the habitat runs (lunar-environment module)

# --------------------------------------------------------------------------
# Water organ-shell phantom  -- RECONSTRUCTED, pinned by --validate.
# --------------------------------------------------------------------------
# Concentric water spheres. Depths (from the r=PHANTOM_R surface) were read off
# the committed proton R(E) penetration gating: skin takes the entrance plateau;
# shallow/mid/deep/core switch on in order as residual range grows, with the core
# requiring ~150 MeV protons (residual water range ~13 cm) -> core near centre.
PHANTOM_R_CM = 15.0
# (name, r_inner_cm, r_outer_cm) outermost last is fine; must tile [0, PHANTOM_R].
SHELLS = [
    ("core",    0.0,  4.0),   # depth 11-15 cm  (needs >=150 MeV/n protons)
    ("deep",    4.0,  9.0),   # depth  6-11 cm
    ("mid",     9.0, 12.0),   # depth  3-6  cm
    ("shallow", 12.0, 14.5),  # depth 0.5-3 cm
    ("skin",   14.5, 15.0),   # depth  0-0.5 cm (entrance plateau)
]

# --------------------------------------------------------------------------
# Wall shield -- RECONSTRUCTED, pinned by --validate.
# --------------------------------------------------------------------------
# A FLAT areal-density slab (the OLTARIS/HZETRN shielding convention this kernel
# is cross-checked against), NOT a concentric shell. A horizontal TsBox of
# thickness t = areal_density / density sits directly above the phantom
# (z in [WALL_RMIN, WALL_RMIN + t]). Every upper-hemisphere ring at zenith theta
# crosses it at slant path t/cos(theta), so the effective shielding is the
# flux-weighted <1/cos(theta)> ~ 1.70x the radial areal density.
#
# WHY A SLAB, NOT A SHELL (pinned by the Al --validate on 2026-08-23): a
# spherically-symmetric shell is theta-independent -- every ring crosses the same
# radial t -- so it CANNOT produce a 1/cos(theta) attenuation pattern. The Al
# regeneration with a shell wall came out biased HIGH by a factor that was flat
# across organ / quantity / species but grew with wall thickness (2.025->1.32,
# 10->1.57, 50->3.21 g/cm^2), and whose high-SNR geo-mean was 1.717 -- almost
# exactly mean(1/cos(theta)) over the four ring angles (1.70). That is the
# fingerprint of a planar slant-path shield the shell was missing.
WALL_RMIN_CM = PHANTOM_R_CM   # slab bottom face z (sits on the phantom north pole)
# Lateral half-extent of the shield slab. Must exceed the horizontal footprint a
# grazing ring ray sweeps while inside the slab so the ray enters the top face and
# exits the bottom face (path = t/cos(theta)). The grazing ring now fires at
# theta_eff=75.5 deg (flux-correct; see templates._ring_directions): a
# centre-aimed ray of that ring reaches x ~ 130 cm at the top face of the 50 g/cm^2
# Al slab (t=18.5 cm), so 200 cm covers it with margin and stays inside the world
# (half = BEAM_RADIUS + 50).
WALL_SLAB_HL_CM = 200.0

# --------------------------------------------------------------------------
# Calibration materials -- TOPAS name, density, and (for non-builtins) the
# composition block, copied verbatim from lunar_environment.txt so transport and
# areal density agree with the habitat runs.
# --------------------------------------------------------------------------
MATERIALS = {
    "aluminium": {"topas": "G4_Al", "density": 2.70, "defn": None},
    # EVA laminate, outboard layer of the two-layer suit. Using EVASuit alone as
    # the wall material for the EVA kernel (the thicker, dose-dominant layer);
    # LCVG is a thin skin-side liner folded in via areal density.
    "evasuit": {"topas": "EVASuit", "density": 0.252, "defn": [
        'sv:Ma/EVASuit/Components = 7 "Hydrogen" "Carbon" "Nitrogen" "Oxygen" "Fluorine" "Chlorine" "Aluminum"',
        'uv:Ma/EVASuit/Fractions = 7 0.04871 0.56290 0.05511 0.15718 0.14774 0.02669 0.00167',
        'b:Ma/EVASuit/NormalizeFractions = "True"',
        'd:Ma/EVASuit/Density = 0.252 g/cm3',
    ]},
    "lcvg": {"topas": "LCVG", "density": 0.5133, "defn": [
        'sv:Ma/LCVG/Components = 4 "Hydrogen" "Carbon" "Nitrogen" "Oxygen"',
        'uv:Ma/LCVG/Fractions = 4 0.1019 0.1320 0.0154 0.7507',
        'b:Ma/LCVG/NormalizeFractions = "True"',
        'd:Ma/LCVG/Density = 0.5133 g/cm3',
    ]},
}

# --------------------------------------------------------------------------
# Areal-density anchor grids.
# --------------------------------------------------------------------------
# Aluminium: the committed grid, used ONLY to validate the reconstruction.
# 0.0 is APPENDED (not prepended) so validate's positional zip against the 3
# committed points stays aligned (2.025/10/50); the bare-phantom point carries no
# committed reference and is used only for the first-principles normalization
# check (isolates the wall-independent baseline that transfers to the EVA kernel).
AL_GRID_GCM2 = [2.025, 10.0, 50.0, 0.0]
# EVA: the locked multi-anchor grid (0 = bare-crew reference; 0.5, 1.0 g/cm^2
# bracket a single suit ~0.28 and a doubled/patched suit). 0.0 means "no wall
# shell" -- the free-field-on-bare-phantom response.
EVA_GRID_GCM2 = [0.0, 0.5, 1.0]

# Output kernel file names (Al path stays byte-for-byte: we never overwrite it).
AL_VALIDATION_OUT = "gcr_thinwall_kernel_al_regen.json"   # compare-only artefact
EVA_OUT = "gcr_thinwall_kernel_eva.json"                  # the shipped EVA kernel
SI_OUT = "gcr_thinwall_kernel_si.json"                    # corrected-composition Al kernel


# --------------------------------------------------------------------------
# Composition selector.
# --------------------------------------------------------------------------
# Two composition schemes are available to the harness:
#
#   "reference" -- H/He/C/O/Fe, read VERBATIM from the committed Al kernel
#       (species_from_reference). This is the composition the shipped kernel was
#       built with. Used to VALIDATE that the reconstructed MC pipeline still
#       reproduces the committed R values before trusting a new run.
#
#   "corrected" -- H/He/C/Si/Fe, matching dosimetry.GCR_COMPOSITION and the
#       methodology-doc section 2.3 table (species_corrected). This resolves the
#       flood-vs-kernel composition mismatch: the flood engine transports Si
#       (Z=14, the Z=10-20 Ne-Ca group representative) with lumped-group
#       abundances, but the committed kernel transports O (a second CNO member,
#       no Z=10-20 representative) with per-element abundances. See the memory
#       note gcr-composition-engine-mismatch and the docx "Open items" section.
#
# IMPORTANT -- what needs new Monte Carlo under "corrected": a species' response
# R = D/phi_ff is a pure single-species transport quantity; the abundance never
# enters the MC, only the free-field flux fold downstream. H/He/C keep IDENTICAL
# node grids to the committed kernel, so they reproduce it exactly under --validate
# and positionally anchor the reconstruction. TWO species are physically new MC:
#   * Si (Z=14) replaces the committed O (Z=8) as the Ne-Ca group representative.
#   * Fe is REGRIDDED from the committed 3 nodes [400,600,1000] MeV/n to the 6-node
#     C/Si grid [180..5000]. The old grid folded only ~31% of the Fe GCR fluence
#     (it integrates strictly between its outer edges, 400-1000 MeV/n): it omitted
#     ~31% below 400 (the high-LET, high-Q stopping region) and ~38% above 1000
#     (the relativistic tail). That truncation -- NOT a Si over-weight -- is what
#     suppressed Fe to ~1/3.7 of the Ne-Ca band when Z^2 x abundance predicts
#     ~1/1.7; extending Fe to [180..5000] restores ~2.6x of its fluence coverage.
# Because Si and Fe carry new grids they have no positional committed baseline;
# --validate reports them separately (MC-only) and scores reproduction on H/He/C.

# Per-nucleon energy nodes shared with the committed kernel (do NOT change these
# for H/He/C/Fe or the corrected-composition --validate loses positional
# alignment with the committed R arrays).
_NODES_LIGHT = [80.0, 150.0, 300.0, 600.0, 1200.0, 2500.0, 6000.0]   # H, He
_NODES_C = [180.0, 350.0, 700.0, 1500.0, 3000.0, 5000.0]             # C
# Fe REGRIDDED to the C/Si 6-node grid (was the committed [400,600,1000], which
# folded only ~31% of the Fe GCR fluence -- see the composition note above). Fe is
# now new MC and no longer positionally validates against the committed 3-node Fe.
_NODES_FE = list(_NODES_C)                                           # Fe (regridded)
# Si is new. It reuses C's 6-node MeV/n grid rather than the committed O grid
# (5 nodes) -- denser and spanning the same range, which also lifts the
# single-low-node dominance flagged for the old O sampling (thin-shield Bragg
# limit). Si represents the Ne-Ca (Z=10-20) group; energy per nucleon, not total.
_NODES_SI = list(_NODES_C)

# Corrected composition == dosimetry.GCR_COMPOSITION (name, particle, z, a,
# abundance rel. H, group) plus each species' per-nucleon node grid.
CORRECTED_COMPOSITION = {
    "H":  {"z": 1,  "a": 1,  "abundance": 1.0,    "particle": "proton",
           "group": "Z=1, protons",              "nodes_pernuc_mev": list(_NODES_LIGHT)},
    "He": {"z": 2,  "a": 4,  "abundance": 0.105,  "particle": "alpha",
           "group": "Z=2, alpha particles",       "nodes_pernuc_mev": list(_NODES_LIGHT)},
    "C":  {"z": 6,  "a": 12, "abundance": 0.033,  "particle": "GenericIon(6,12)",
           "group": "Z=3-9 (CNO + light)",        "nodes_pernuc_mev": list(_NODES_C)},
    "Si": {"z": 14, "a": 28, "abundance": 0.012,  "particle": "GenericIon(14,28)",
           "group": "Z=10-20 (Ne-Ca)",            "nodes_pernuc_mev": list(_NODES_SI)},
    "Fe": {"z": 26, "a": 56, "abundance": 0.0020, "particle": "GenericIon(26,56)",
           "group": "Z=21-28 (Fe peak)",          "nodes_pernuc_mev": list(_NODES_FE)},
}


def organs_from_reference() -> list:
    return [tuple(o) for o in load_reference()["meta"]["organs"]]


def species_from_reference() -> dict:
    """{name: {z, a, abundance, particle, group, nodes_pernuc_mev}} from the
    committed kernel (thickest anchor -- identical species metadata at every
    anchor). Node energies and abundances are reused verbatim."""
    sp = load_reference()["points"][-1]["species"]
    out = {}
    for name, s in sp.items():
        out[name] = {
            "z": s["z"], "a": s["a"], "abundance": s["abundance"],
            "particle": s["particle"], "group": s["group"],
            "nodes_pernuc_mev": list(s["nodes_pernuc_mev"]),
        }
    return out


def species_corrected() -> dict:
    """The H/He/C/Si/Fe composition matching dosimetry.GCR_COMPOSITION (doc 2.3),
    same dict shape as species_from_reference(). Deep-copied so callers cannot
    mutate the module constant."""
    return {
        name: {"z": s["z"], "a": s["a"], "abundance": s["abundance"],
               "particle": s["particle"], "group": s["group"],
               "nodes_pernuc_mev": list(s["nodes_pernuc_mev"])}
        for name, s in CORRECTED_COMPOSITION.items()
    }


COMPOSITIONS = ("reference", "corrected")


def species_for(composition: str = "reference") -> dict:
    """Resolve a composition name to its {name: meta} species table."""
    if composition == "reference":
        return species_from_reference()
    if composition == "corrected":
        return species_corrected()
    raise ValueError(f"unknown composition {composition!r}; expected one of {COMPOSITIONS}")
