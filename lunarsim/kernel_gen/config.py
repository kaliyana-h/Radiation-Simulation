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
# Reproduction-gate reference. Once the deployed kernel is TAIL-EXTENDED (its fold
# runs to the free-field ceiling, so species carry extra high-energy nodes the base
# regen does not), the shipped file is no longer a clean base-grid baseline for
# --validate. If a pre-tail snapshot exists, validate against THAT so the gate keeps
# comparing like node grids; otherwise fall back to the deployed kernel.
_REPO_KERNEL_BASE = _REPO_KERNEL.with_suffix(".base.json")


def load_reference() -> dict:
    """The committed aluminium kernel -- species table, energy nodes, organ
    weights and meta are copied FROM here so a regenerated kernel stays aligned.
    Prefers the base-grid snapshot (gcr_thinwall_kernel.base.json) when present, so
    the reproduction gate stays meaningful after the deployed kernel is tail-extended."""
    ref = _REPO_KERNEL_BASE if _REPO_KERNEL_BASE.exists() else _REPO_KERNEL
    with open(ref) as fh:
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
# A concentric SHELL of thickness t = areal_density / density wrapping the phantom
# (a TsSphere, RMin=PHANTOM_R, RMax=RMin+t). Every upper-hemisphere ring crosses it
# near-radially, so the shielded areal density is ~ the nominal t (zero obliquity).
# The alternative "slab" mode (selectable, see WALL_GEOMETRIES) is a horizontal TsBox
# whose rings cross at slant path t/cos(theta), giving a flux-weighted <1/cos(theta)>
# ~ 1.70x slant -- that mode is retained only for regression/reproduction, NOT shipped.
#
# WHY A SHELL, NOT A SLAB (flipped 2026-10-13; supersedes the 2026-08-23 slab pin):
# the slab's constant ~1.70x slant OVER-attenuates absorbed dose at depth, and the
# over-attenuation GROWS with thickness (true obliquity is 1.32/1.57/3.21 at
# 2.025/10/50 g/cm^2, not one geo-mean), so the slab kernel's deep absorbed fell
# away from the same-physics flood (kern/flood 0.83->0.78 over 19->40). A shell
# removes the slant: the target-blind flood gate (paper/flood_absorbed_depth.py
# --refold) has the shell base reproducing flood/OLTARIS absorbed FLAT across
# 19->50 g/cm^2 (kern/flood 0.915/0.963/0.966/0.977; @50 = 0.988x OLTARIS 0.302).
# The 2026-08-23 "shell biased HIGH 1.32/1.57/3.21" finding was a CIRCULAR test --
# it measured a shell regen against the slab-TUNED kernel, i.e. that 1.70x slant IS
# the slab's own over-attenuation seen in reverse, not a shell bias. The correct
# yardstick is the same-physics flood, which the shell matches and the slab does not.
# NOTE: we ship the shell BASE (no ceiling tail). On the corrected geometry the tail
# over-read OLTARIS by ~21% at depth (it was a compensator for the slab defect, now
# removed), so tail.py's extension is NOT merged into the production kernel.
WALL_RMIN_CM = PHANTOM_R_CM   # slab bottom face z (sits on the phantom north pole)
# Lateral half-extent of the shield slab. Must exceed the horizontal footprint a
# grazing ring ray sweeps while inside the slab so the ray enters the top face and
# exits the bottom face (path = t/cos(theta)). The grazing ring now fires at
# theta_eff=75.5 deg (flux-correct; see templates._ring_directions): a
# centre-aimed ray of that ring reaches x ~ 130 cm at the top face of the 50 g/cm^2
# Al slab (t=18.5 cm), so 200 cm covers it with margin and stays inside the world
# (half = BEAM_RADIUS + 50).
WALL_SLAB_HL_CM = 200.0

# Wall placement modes build_param_file understands. "slab" is the flat
# t/cos(theta) slant shield documented above; "shell" is the concentric-shell
# (near-radial, zero-obliquity) wall. wall_geometry_probe.py (2026-10-13) showed
# the slab OVER-attenuates absorbed dose at depth (kernel/flood fell 0.71->0.60
# over 19->50 g/cm^2) while a shell @50 reproduces the same-physics flood/OLTARIS
# absorbed (0.302) -- so the slab, not cross-code physics, is the deep-slope
# defect. See WALL_GEOMETRY below for which one production uses.
WALL_GEOMETRIES = ("slab", "shell")

# PRODUCTION wall geometry -- the default emit/tail/collect use when a caller does
# not override it. SHIPPED "shell" on 2026-10-13: the shell base cleared the
# target-blind flood acceptance gate (paper/flood_absorbed_depth.py --refold),
# with kernel absorbed reproducing the same-physics flood FLAT across 19->50 g/cm^2
# (kern/flood 0.915/0.963/0.966/0.977; @50 = 0.988x OLTARIS 0.302), whereas the
# slab fell away at depth (0.83->0.78 over 19->40). "slab" stays selectable via
# WALL_GEOMETRIES for regression/reproduction only. See paper/KERNEL_SLAB_RUNBOOK.md.
WALL_GEOMETRY = "shell"

# --------------------------------------------------------------------------
# Calibration materials -- TOPAS name, density, and (for non-builtins) the
# composition block, copied verbatim from lunar_environment.txt so transport and
# areal density agree with the habitat runs.
# --------------------------------------------------------------------------
MATERIALS = {
    "aluminium": {"topas": "G4_Al", "density": 2.70, "defn": None},
    # Polyethylene (CH2) -- the default/most-common habitat wall material and the
    # most hydrogen-rich, so the least-defensible case for the aluminium fallback.
    # G4_POLYETHYLENE is a Geant4 builtin (defn=None), rho=0.94 g/cm^3.
    "polyethylene": {"topas": "G4_POLYETHYLENE", "density": 0.94, "defn": None},
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
# Polyethylene: MIRRORS the Al grid. Areal density (g/cm^2) is material-agnostic,
# so reusing Al's anchors keeps the poly-vs-Al comparison apples-to-apples at
# identical shielding mass. poly rho=0.94 -> these are 2.15/10.6/53 cm thick.
POLY_GRID_GCM2 = list(AL_GRID_GCM2)

# Output kernel file names (Al path stays byte-for-byte: we never overwrite it).
AL_VALIDATION_OUT = "gcr_thinwall_kernel_al_regen.json"   # compare-only artefact
EVA_OUT = "gcr_thinwall_kernel_eva.json"                  # the shipped EVA kernel
SI_OUT = "gcr_thinwall_kernel_si.json"                    # corrected-composition Al kernel
POLY_OUT = "gcr_thinwall_kernel_poly.json"                # the shipped polyethylene kernel
PERELEM_OUT = "gcr_thinwall_kernel_perelem.json"          # the 10-ion per-element Al kernel

# --------------------------------------------------------------------------
# Per-material anchor grid and shipped-kernel filename.
# --------------------------------------------------------------------------
# A material absent from GRID_FOR_MATERIAL falls back to the thin EVA grid (the
# original two-material default). Aluminium is deliberately absent from
# OUT_FOR_MATERIAL: its output is composition-keyed in collect() (corrected ->
# SI_OUT, reference -> AL_VALIDATION_OUT), never a single fixed name.
GRID_FOR_MATERIAL = {
    "aluminium": AL_GRID_GCM2,
    "polyethylene": POLY_GRID_GCM2,
    "evasuit": EVA_GRID_GCM2,
    "lcvg": EVA_GRID_GCM2,
}
OUT_FOR_MATERIAL = {
    "polyethylene": POLY_OUT,
    "evasuit": EVA_OUT,
    "lcvg": EVA_OUT,
}


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


# --------------------------------------------------------------------------
# High-energy TAIL extension (heavy-ion penetration tail / energy-ceiling run).
# --------------------------------------------------------------------------
# The committed kernel's top per-nucleon nodes are 6000 (H/He) and 5000 (C/Si/Fe)
# MeV/n, but the free-field GCR flux make_source emits runs up to
# make_source.GCR_EMAX_PER_NUC = 20000 MeV/n (20 GeV/n). dosimetry._thinwall_fold_point
# integrates the spectrum only between the lowest and HIGHEST node (edges = [nd[0]]
# ... [nd[-1]]): all flux ABOVE the top node is HARD-TRUNCATED, not extrapolated.
# That dropped 5-20 GeV/n band carries ~4-7% of each species' fluence (measured
# target-blind, laptop arithmetic) and a larger share of DOSE for the penetrating,
# high-Q heavy ions -- it is exactly the "energy-ceiling caveat" (memories
# [[compute-scaling-priorities]], [[effective-dose-option-c]] i7/i9).
#
# These TAIL nodes extend each species' grid UP TO the free-field ceiling so the
# fold's top edge coincides with GCR_EMAX_PER_NUC and nothing is dropped. They are
# APPENDED to -- never replace -- the committed grid, so the base R values stay
# byte-identical and the Al --validate gate is unaffected (the tail batch is run
# and MERGED in by kernel_gen.tail; it is NOT part of the validate path). The top
# node is pinned to 20000 MeV/n == GCR_EMAX_PER_NUC so zero residual flux is lost.
TAIL_CEILING_PERNUC_MEV = 20000.0   # == make_source.GCR_EMAX_PER_NUC (verify before a run)
TAIL_NODES_PERNUC_MEV = {
    "H":  [12000.0, TAIL_CEILING_PERNUC_MEV],   # total KE 12 / 20 GeV
    "He": [12000.0, TAIL_CEILING_PERNUC_MEV],   # total KE 48 / 80 GeV
    "C":  [10000.0, TAIL_CEILING_PERNUC_MEV],   # total KE 120 / 240 GeV
    "Si": [10000.0, TAIL_CEILING_PERNUC_MEV],   # total KE 280 / 560 GeV
    "Fe": [10000.0, TAIL_CEILING_PERNUC_MEV],   # total KE 560 / 1120 GeV (the long pole)
}


def species_corrected_tail() -> dict:
    """The corrected H/He/C/Si/Fe table carrying ONLY the high-energy TAIL nodes
    (TAIL_NODES_PERNUC_MEV). Same dict shape as species_corrected(); used by
    kernel_gen.tail to emit a focused tail batch whose responses are then merged
    onto the committed kernel's existing nodes. Node energies here are the APPEND
    set, not the full grid."""
    return {
        name: {"z": s["z"], "a": s["a"], "abundance": s["abundance"],
               "particle": s["particle"], "group": s["group"],
               "nodes_pernuc_mev": list(TAIL_NODES_PERNUC_MEV[name])}
        for name, s in CORRECTED_COMPOSITION.items()
    }


def species_matthia(phi_MV: float = 400.0) -> dict:
    """The corrected H/He/C/Si/Fe table with Matthiae-2013 group ABUNDANCES in
    place of the shipped representative values; identities, A, groupings and node
    grids are unchanged. Same dict shape as species_corrected().

    Abundances come from make_source.mattia_group_abundance (the ratio of each
    group's full-range Matthiae LIS integral to protons', self-calibrated via W<->phi),
    so a kernel generated from this table carries the physically-derived relative
    fluxes the per-element model (dosimetry PERELEMENT_MODEL='matthia2013') folds with.
    The abundance is baked into the kernel JSON at generation time and read back by
    dosimetry._thinwall_fold_point, so a Matthiae kernel MUST be folded with the
    matthia model active (its per-node SHAPE comes from _raw_gcr_integral, which only
    returns the Matthiae spectrum when the env model is set). phi_MV should match the
    fold's modulation (default 400 MV, the solar-minimum kernel fold)."""
    import make_source as ms  # repo-root module (same import tail.py uses)
    return {
        name: {"z": s["z"], "a": s["a"],
               "abundance": ms.mattia_group_abundance(s["z"], phi_MV),
               "particle": s["particle"], "group": s["group"],
               "nodes_pernuc_mev": list(s["nodes_pernuc_mev"])}
        for name, s in CORRECTED_COMPOSITION.items()
    }


# --------------------------------------------------------------------------
# PER-ELEMENT composition (Step 2 -- the real thin-wall <Q> fix).
# --------------------------------------------------------------------------
# The corrected/matthia tables collapse each HZE band to ONE representative Z
# (C for Z3-9, Si for Z10-20, Fe for Z21-28). With Matthiae abundances the
# resulting error in ABSORBED dose is small (<=12% per band -- rep Z^2 ~ the
# band's ab-weighted Sum Z^2), but the error in DOSE-EQUIVALENT is not: <Q> is
# nonlinear in LET, so assigning a whole band its rep's single Q over-reads the
# thin-wall <Q> (see [[let-scorer-thin-qbar-rootcause]], [[thinwall-qbar-fragmentation]]).
# This table resolves each band into its dose-dominant individual elements, each
# carried with its OWN Matthiae per-element abundance + its OWN Geant4
# fragmentation response (a one-time kernel regen; R is spectrum-independent so
# re-folds are free afterwards).
#
# The ion list is the minimal set covering ~92% of absorbed dose AND spanning
# each band's Q range (step2_ionset.py ranking, target-blind). A, particle and
# node grid follow the same rules as the 5-ion table (A = most-abundant isotope;
# proton/alpha for Z<=2 else GenericIon(z,a); light 7-node grid for Z<=2 else the
# 6-node HZE grid). Abundance is the per-element Matthiae value via
# mattia_element_abundance -- which, unlike mattia_group_abundance, returns the
# single element even for z=6/14/26 (those are band reps in the 5-ion model and
# would otherwise return the whole-band sum, double-counting N/O, Ne-Ca, etc.).
#
# Edit PERELEMENT_IONS to change the set; (name, z, a) only -- everything else is
# derived. Keep it ordered by Z.
PERELEMENT_IONS = [
    ("H",   1,  1),
    ("He",  2,  4),
    ("C",   6,  12),
    ("N",   7,  14),
    ("O",   8,  16),
    ("Ne",  10, 20),
    ("Mg",  12, 24),
    ("Si",  14, 28),
    ("Ca",  20, 40),
    ("Fe",  26, 56),
]


def _perelement_nodes(z: int) -> list:
    """Node grid for a per-element ion: the light 7-node grid for Z<=2, else the
    6-node HZE grid (same rule the 5-ion table uses for C/Si/Fe)."""
    return list(_NODES_LIGHT) if z <= 2 else list(_NODES_C)


def _perelement_particle(z: int, a: int) -> str:
    if z == 1:
        return "proton"
    if z == 2:
        return "alpha"
    return f"GenericIon({z},{a})"


def _perelement_assignment() -> dict:
    """Voronoi partition of every GCR element Z=1..28 onto the chosen ion set:
    each Z goes to the nearest ion by |dZ| (ties -> the heavier ion). Returns
    {ion_z: [assigned Zs]}. Guarantees the per-element kernel's TOTAL GCR flux
    equals the full 28-element sum (== the 5-ion model's total), so the ion list
    is a pure Z-RESOLUTION dial: more ions -> finer bands -> less <Q> distortion,
    never a change in total fluence. A dropped element's flux is carried by its
    nearest-Z neighbour (same-band, small dZ), not by a cross-band representative."""
    ion_zs = sorted(z for _, z, _ in PERELEMENT_IONS)
    assign = {z: [] for z in ion_zs}
    for z in range(1, 29):
        # nearest ion; tie -> heavier (max z). key sorts by (distance, -z) asc.
        best = min(ion_zs, key=lambda iz: (abs(iz - z), -iz))
        assign[best].append(z)
    return assign


def species_perelement(phi_MV: float = 400.0) -> dict:
    """Per-element GCR composition: each dose-dominant element carried
    individually with its OWN Matthiae-2013 spectrum + node grid, and an abundance
    that is the sum of its Voronoi-assigned neighbourhood (so total flux is
    conserved -- see _perelement_assignment). Same dict shape as species_corrected().
    A kernel generated from this table MUST be folded with the per-element
    abundances baked into its JSON (dosimetry reads meta.composition == 'perelement'
    and keys the abundance override on each element's own Z)."""
    import make_source as ms  # repo-root module (same import species_matthia uses)
    assign = _perelement_assignment()
    out = {}
    for name, z, a in PERELEMENT_IONS:
        ab = sum(ms.mattia_element_abundance(zz, phi_MV) for zz in assign[z])
        out[name] = {
            "z": z, "a": a,
            "abundance": ab,
            "particle": _perelement_particle(z, a),
            "group": f"Z={z} ({name}); carries Z={assign[z][0]}-{assign[z][-1]}"
                     if len(assign[z]) > 1 else f"Z={z} ({name})",
            "nodes_pernuc_mev": _perelement_nodes(z),
        }
    return out


# --------------------------------------------------------------------------
# PER-ELEMENT high-energy TAIL (the #3 energy-ceiling run for the 10-ion kernel).
# --------------------------------------------------------------------------
# The 5-ion TAIL above (species_corrected_tail) extends the OLD corrected kernel.
# The shipped THIN engine is now the 10-ion per-element kernel
# (gcr_thinwall_kernel_perelem.json), whose per-nucleon tops are 6000 (H/He) and
# 5000 (the 8 HZE ions) MeV/n -- the same truncation point. This per-element tail
# appends the high-energy nodes UP TO the free-field ceiling for EACH of the 10
# ions, so dosimetry._thinwall_fold_point's top edge reaches GCR_EMAX_PER_NUC and
# the penetrating 5-20 GeV/n HZE band (the population that keeps the deep water-
# point <Q> up -- memory crater-qfactor-depth-reference) is no longer dropped.
# Append nodes only; base R stays byte-identical, merged by kernel_gen.tail onto
# the per-element base kernel. Same append structure as TAIL_NODES_PERNUC_MEV.
def _perelement_tail_nodes(z: int) -> list:
    """Per-element TAIL append nodes: [12000, ceiling] for Z<=2 (base top 6000),
    else [10000, ceiling] (base top 5000). Top pinned to the free-field ceiling so
    the fold loses zero residual flux; strictly above each ion's base top so the
    merge guard (tail must extend, not overlap) is satisfied."""
    lead = 12000.0 if z <= 2 else 10000.0
    return [lead, TAIL_CEILING_PERNUC_MEV]


def species_perelement_tail(phi_MV: float = 400.0) -> dict:
    """The 10-ion per-element table carrying ONLY the high-energy TAIL append nodes
    (_perelement_tail_nodes). Same dict shape as species_perelement(); used by
    kernel_gen.tail to emit a per-element tail batch whose responses are merged onto
    the per-element base kernel (gcr_thinwall_kernel_perelem.json).

    Abundance is the per-element Voronoi sum (identical to species_perelement) and is
    carried for display/collect only -- the merge keeps the BASE kernel's abundance
    and never reads this one (R = D/phi_ff is abundance-independent transport), so a
    stale phi here cannot corrupt a merged kernel."""
    base = species_perelement(phi_MV)
    return {
        name: {"z": s["z"], "a": s["a"], "abundance": s["abundance"],
               "particle": s["particle"], "group": s["group"],
               "nodes_pernuc_mev": _perelement_tail_nodes(s["z"])}
        for name, s in base.items()
    }


COMPOSITIONS = ("reference", "corrected", "tail", "matthia", "perelement",
                "perelement_tail")

# Tail compositions kernel_gen.tail knows how to emit/merge: each appends high-
# energy nodes onto a base kernel without changing its base grid. Maps each to the
# base-kernel composition(s) it may be merged onto (merge guards on this).
TAIL_COMPOSITIONS = {
    "tail": ("reference", "corrected"),
    "perelement_tail": ("perelement",),
}


def species_for(composition: str = "reference") -> dict:
    """Resolve a composition name to its {name: meta} species table."""
    if composition == "reference":
        return species_from_reference()
    if composition == "corrected":
        return species_corrected()
    if composition == "tail":
        return species_corrected_tail()
    if composition == "matthia":
        return species_matthia()
    if composition == "perelement":
        return species_perelement()
    if composition == "perelement_tail":
        return species_perelement_tail()
    raise ValueError(f"unknown composition {composition!r}; expected one of {COMPOSITIONS}")
