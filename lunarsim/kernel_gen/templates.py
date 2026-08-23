"""kernel_gen.templates -- emit one TOPAS parameter file per
(material, areal-density anchor, species, energy node, seed).

The illumination is the equal-flux upper-hemisphere ring pattern make_source.py
writes for the habitat GCR source (rings x azimuth parallel disc beams firing
inward from BeamRadius), only mono-energetic: one BeamEnergy per node instead of
a continuous spectrum. The source NESTING (RotZ/RotY groups, RotX=180 to fire
inward) is copied verbatim from make_source._ring_groups. The zenith ANGLES,
however, use the FLUX-CORRECT band-effective secant (see _ring_directions), not
gcr_block's area-midpoint: the committed Al kernel's shielding response requires
the true continuous-hemisphere <1/cos(theta)> = 2.0, which four discrete midpoint
angles under-sample to 1.70 (pinned by the Al --validate; see _ring_directions).

Geometry: concentric water organ-shell spheres (config.SHELLS) shielded by a
FLAT areal-density slab of the wall material (horizontal TsBox, thickness =
areal_density / rho) sitting directly above the phantom -- the OLTARIS-style
slant-path shield, so each upper-hemisphere ring crosses t/cos(theta) (see
config.py "WHY A SLAB, NOT A SHELL"). Each shell carries two whole-volume
scorers -- DoseToMedium (absorbed, -> R["D"]) and DoseEquivalent_ICRP
(ICRP-60 dose-eq, -> R["I"]).
"""
from __future__ import annotations

import math

from . import config


def _ring_directions(rings: int = config.RINGS, azimuth: int = config.AZIMUTH):
    """(phi_deg, theta_deg) for every source: equal-area-in-sin^2 zenith rings x
    uniform azimuth over the upper hemisphere, but each ring fires at its
    FLUX-CORRECT band-effective zenith, not the band area-midpoint.

    Why not the midpoint (make_source.gcr_block's convention): the shield is
    crossed at slant path t/cos(theta), and what matters for the kernel is the
    flux-weighted <1/cos(theta)> of the whole hemisphere, which for the true
    continuous isotropic field is exactly 2.0. Four discrete AREA-MIDPOINT angles
    (20.7/37.8/52.2/69.3 deg) sum to only <1/cos> = 1.70 -- they badly
    under-sample the grazing band [60,90] deg where 1/cos(theta) blows up. The Al
    --validate pinned this exactly (2026-08-23): the committed kernel suppresses
    an 80 MeV proton (range ~4.9 g/cm^2 Al) to 0.22x its entrance dose through
    only 2.025 g/cm^2 -- impossible at normal incidence, but exactly what an
    effective <1/cos> ~ 2.0 does. So each ring instead fires at theta_eff with
    1/cos(theta_eff) = <1/cos(theta)>_band, the band-averaged secant:

        <1/cos>_band = 2*rings*( sqrt(1 - r/rings) - sqrt(1 - (r+1)/rings) )

    which restores the exact continuous-hemisphere <1/cos> = 2.0 while keeping
    rings x azimuth = [4,8], phi_ff and the 128-primary count unchanged. The four
    zenith angles become 21.1/38.1/52.9/75.5 deg -- only the grazing ring moves
    much (69.3 -> 75.5 deg, 1/cos 2.83 -> 4.0), which is the whole fix.
    """
    dirs = []
    for r in range(rings):
        s_lo, s_hi = r / rings, (r + 1) / rings
        mean_sec = 2.0 * rings * (math.sqrt(1.0 - s_lo) - math.sqrt(1.0 - s_hi))
        theta = math.degrees(math.acos(1.0 / mean_sec))
        for a in range(azimuth):
            phi = a * 360.0 / azimuth
            dirs.append((phi, theta))
    return dirs


def _ring_groups(idx: int, phi_deg: float, theta_deg: float):
    """Nested Azi(RotZ) -> Aim(RotY) -> Src(TransZ, RotX=180) firing inward.
    Byte-identical to make_source._ring_groups so the field geometry matches."""
    az, aim, src = f"KAzi{idx}", f"KAim{idx}", f"KSrc{idx}"
    return [
        f's:Ge/{az}/Type   = "Group"',
        f's:Ge/{az}/Parent = "World"',
        f'd:Ge/{az}/RotZ   = {phi_deg:.4f} deg',
        f's:Ge/{aim}/Type   = "Group"',
        f's:Ge/{aim}/Parent = "{az}"',
        f'd:Ge/{aim}/RotY   = {theta_deg:.4f} deg',
        f's:Ge/{src}/Type   = "Group"',
        f's:Ge/{src}/Parent = "{aim}"',
        f'd:Ge/{src}/TransZ = Ge/BeamRadius cm',
        f'd:Ge/{src}/RotX   = 180.0 deg',
    ], src


def _shell_component_name(shell_name: str) -> str:
    return "Shell_" + shell_name


def scorer_csv_names(shell_name: str) -> tuple[str, str]:
    """(absorbed_file, doseeq_file) basenames TOPAS writes (without .csv)."""
    return f"{shell_name}_D", f"{shell_name}_I"


def build_param_file(material: str, wall_gcm2: float, species_name: str,
                     node_index: int, seed: int, threads: int = 0) -> str:
    """Full text of one runnable TOPAS kernel-generation file."""
    mat = config.MATERIALS[material]
    species = config.species_from_reference()[species_name]
    nodes = species["nodes_pernuc_mev"]
    e_pernuc = nodes[node_index]
    a = species["a"]
    total_energy_mev = e_pernuc * a          # TOPAS BeamEnergy = TOTAL kinetic energy
    particle = species["particle"]           # 'proton' | 'alpha' | 'GenericIon(z,a)'

    wall_thick = (wall_gcm2 / mat["density"]) if wall_gcm2 > 0 else 0.0
    world_half = config.BEAM_RADIUS_CM + 50.0

    L = []
    L.append("# ============================================================")
    L.append("# Thin-wall GCR response-kernel generation run")
    L.append(f"#   material={material} ({mat['topas']})  wall={wall_gcm2:g} g/cm^2"
             f"  thick={wall_thick:.4f} cm")
    L.append(f"#   species={species_name} particle={particle}"
             f"  node={node_index} E={e_pernuc:g} MeV/n  total={total_energy_mev:g} MeV")
    L.append(f"#   seed={seed}  illumination=rings{config.RINGS}x az{config.AZIMUTH}"
             f" x hist{config.HISTORIES}  beam_spot={config.BEAM_SPOT_CM:g} cm")
    L.append(f"#   phi_ff={config.PHI_FF_CM2:.6g} /cm^2   (generated by lunarsim.kernel_gen)")
    L.append("# ============================================================")
    L.append("")
    L.append(f"i:Ts/Seed            = {seed}")
    L.append(f"i:Ts/NumberOfThreads = {threads}")
    L.append('b:Ts/PauseBeforeQuit = "False"')
    L.append("i:Ts/ShowHistoryCountAtInterval = 0")
    L.append(f's:Ph/Default/Type = "{config.PHYSICS}"')
    L.append("")
    # World (vacuum) -- no regolith: this is the clean direct-transmission kernel.
    L.append('s:Ge/World/Type      = "TsBox"')
    L.append('s:Ge/World/Material  = "Vacuum"')
    L.append(f"d:Ge/World/HLX       = {world_half:.1f} cm")
    L.append(f"d:Ge/World/HLY       = {world_half:.1f} cm")
    L.append(f"d:Ge/World/HLZ       = {world_half:.1f} cm")
    L.append('b:Ge/World/Invisible = "true"')
    L.append("")
    # Non-builtin material definition (EVASuit / LCVG).
    if mat["defn"]:
        L.append(f"# --- {material} material definition (from lunar_environment.txt) ---")
        L.extend(mat["defn"])
        L.append("")
    # Flat areal-density shield slab (skip at the 0 g/cm^2 bare-phantom anchor).
    # Horizontal TsBox of thickness t = areal/rho sitting on the phantom north
    # pole; every upper-hemisphere ring at zenith theta crosses it at t/cos(theta).
    if wall_gcm2 > 0:
        slab_hlz = wall_thick / 2.0
        slab_z = config.WALL_RMIN_CM + slab_hlz   # box centre -> spans [RMIN, RMIN+t]
        L.append('s:Ge/Wall/Type     = "TsBox"')
        L.append('s:Ge/Wall/Parent   = "World"')
        L.append(f's:Ge/Wall/Material = "{mat["topas"]}"')
        L.append(f"d:Ge/Wall/HLX      = {config.WALL_SLAB_HL_CM:.1f} cm")
        L.append(f"d:Ge/Wall/HLY      = {config.WALL_SLAB_HL_CM:.1f} cm")
        L.append(f"d:Ge/Wall/HLZ      = {slab_hlz:.4f} cm")
        L.append(f"d:Ge/Wall/TransZ   = {slab_z:.4f} cm")
        L.append('b:Ge/Wall/Invisible = "true"')
        L.append("")
    # Concentric water organ shells + their two scorers.
    for name, r_in, r_out in config.SHELLS:
        comp = _shell_component_name(name)
        L.append(f's:Ge/{comp}/Type     = "TsSphere"')
        L.append(f's:Ge/{comp}/Parent   = "World"')
        L.append(f's:Ge/{comp}/Material = "G4_WATER"')
        L.append(f"d:Ge/{comp}/RMin     = {r_in:.4f} cm")
        L.append(f"d:Ge/{comp}/RMax     = {r_out:.4f} cm")
        f_d, f_i = scorer_csv_names(name)
        for quant, fbase in (("DoseToMedium", f_d), ("DoseEquivalent_ICRP", f_i)):
            sc = f"{comp}_{'D' if quant == 'DoseToMedium' else 'I'}"
            L.append(f's:Sc/{sc}/Quantity   = "{quant}"')
            L.append(f's:Sc/{sc}/Component  = "{comp}"')
            L.append(f's:Sc/{sc}/OutputFile = "{fbase}"')
            L.append(f's:Sc/{sc}/OutputType = "csv"')
            L.append(f's:Sc/{sc}/IfOutputFileAlreadyExists = "Overwrite"')
        L.append("")
    # Mono-energetic upper-hemisphere ring source.
    L.append(f"d:Ge/BeamRadius = {config.BEAM_RADIUS_CM:.1f} cm")
    L.append(f"d:So/BeamSpot   = {config.BEAM_SPOT_CM:.1f} cm")
    L.append("")
    for idx, (phi, theta) in enumerate(_ring_directions()):
        grp, src = _ring_groups(idx, phi, theta)
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
        L.append(f"i:So/{tag}/NumberOfHistoriesInRun   = {config.HISTORIES}")
        L.append("")
    return "\n".join(L) + "\n"
