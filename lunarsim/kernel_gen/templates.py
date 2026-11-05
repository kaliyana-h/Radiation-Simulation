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
concentric SHELL of the wall material (TsSphere, RMin=PHANTOM_R, RMax=RMin+t,
t = areal_density / rho) -- the near-radial, zero-obliquity wall that reproduces
the same-physics flood's deep absorbed dose (shipped 2026-10-13; see config.py
"WHY A SHELL, NOT A SLAB"). The alternative FLAT slab (horizontal TsBox, rings
cross t/cos(theta)) stays selectable via config.WALL_GEOMETRIES for regression
only -- it over-attenuates absorbed dose at depth. Each shell carries two
whole-volume scorers -- DoseToMedium (absorbed, -> R["D"]) and
DoseEquivalent_ICRP (ICRP-60 dose-eq, -> R["I"]).
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


SKIN_MODES = ("phantom", "lining")
LINING_THICK_CM = 2.0     # flood CrewSkin lining thickness (geometry._dome)


def phi_ff(beam_spot_cm: float | None = None, histories: int | None = None) -> float:
    """Planar free-field fluence per run (/cm^2) for a given disc radius and
    histories per beam. Defaults reproduce config.PHI_FF_CM2 exactly."""
    if beam_spot_cm is None and histories is None:
        return config.PHI_FF_CM2
    spot = config.BEAM_SPOT_CM if beam_spot_cm is None else float(beam_spot_cm)
    hist = config.HISTORIES if histories is None else int(histories)
    return config.RINGS * config.AZIMUTH * hist / (math.pi * spot ** 2)


def build_param_file(material: str, wall_gcm2: float, species_name: str,
                     node_index: int, seed: int, threads: int = 0,
                     composition: str = "reference",
                     wall_geometry: str = "shell",
                     beam_spot_cm: float | None = None,
                     histories: int | None = None,
                     skin_mode: str = "phantom") -> str:
    """Full text of one runnable TOPAS kernel-generation file.

    wall_geometry selects how the areal-density shield is placed:
      "shell" (default, PRODUCTION) -- concentric TsSphere shell wrapping the
              phantom (r in [WALL_RMIN, WALL_RMIN+t]); rings cross it near-radially,
              so the shielded areal density is ~ the nominal t (zero obliquity).
              Shipped 2026-10-13: reproduces the same-physics flood's deep absorbed
              dose (kern/flood FLAT 0.915->0.977 over 19->50 g/cm^2; @50 = 0.988x
              OLTARIS 0.302). See config.py "WHY A SHELL, NOT A SLAB".
      "slab"  (regression/reproduction only) -- horizontal TsBox above the phantom;
              every ring crosses t/cos(theta), a ~1.70x constant slant that
              over-attenuates absorbed dose at depth (the deep-slope defect this
              ship fixed). Retained so the old committed kernel can be reproduced.

    DIAGNOSTIC knobs (paper/geometry_residual_probe.py; defaults leave the file
    byte-identical to the production kernel):
      beam_spot_cm / histories -- per-beam disc radius and histories. R is then
              normalised by phi_ff(beam_spot_cm, histories), not config.PHI_FF_CM2.
      skin_mode -- "phantom" (production): five concentric organ shells, skin =
              outer 0.5 cm of the self-shielded 15 cm phantom. "lining": the flood's
              CrewSkin definition instead -- ONE upper-hemisphere (DTheta 90) water
              lining, RMin = PHANTOM_R - 2 cm, RMax = PHANTOM_R, vacuum inside, scored
              as "skin". No other organs exist in that mode.
    """
    if wall_geometry not in config.WALL_GEOMETRIES:
        raise ValueError(f"wall_geometry must be one of {config.WALL_GEOMETRIES}, "
                         f"got {wall_geometry!r}")
    if skin_mode not in SKIN_MODES:
        raise ValueError(f"skin_mode must be one of {SKIN_MODES}, got {skin_mode!r}")
    spot = config.BEAM_SPOT_CM if beam_spot_cm is None else float(beam_spot_cm)
    hist = config.HISTORIES if histories is None else int(histories)
    if spot <= 0 or hist <= 0:
        raise ValueError("beam_spot_cm and histories must be positive")
    mat = config.MATERIALS[material]
    species = config.species_for(composition)[species_name]
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
             f" x hist{hist}  beam_spot={spot:g} cm")
    L.append(f"#   phi_ff={phi_ff(beam_spot_cm, histories):.6g} /cm^2"
             "   (generated by lunarsim.kernel_gen)")
    if skin_mode != "phantom":
        L.append(f"#   skin_mode={skin_mode}  (DIAGNOSTIC: flood-style CrewSkin lining)")
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
    # Areal-density shield (skip at the 0 g/cm^2 bare-phantom anchor).
    if wall_gcm2 > 0 and wall_geometry == "slab":
        # Flat areal-density shield slab: horizontal TsBox of thickness t = areal/rho
        # sitting on the phantom north pole; every upper-hemisphere ring at zenith
        # theta crosses it at t/cos(theta).
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
    elif wall_gcm2 > 0 and wall_geometry == "shell":
        # PRODUCTION wall: concentric TsSphere shell of thickness t = areal/rho
        # wrapping the phantom (r in [WALL_RMIN, WALL_RMIN+t]). A centre-aimed ring
        # ray crosses it near-radially (path ~ t, theta-independent), removing the
        # slab's t/cos(theta) slant. See config.py "WHY A SHELL, NOT A SLAB".
        L.append('s:Ge/Wall/Type     = "TsSphere"')
        L.append('s:Ge/Wall/Parent   = "World"')
        L.append(f's:Ge/Wall/Material = "{mat["topas"]}"')
        L.append(f"d:Ge/Wall/RMin     = {config.WALL_RMIN_CM:.4f} cm")
        L.append(f"d:Ge/Wall/RMax     = {config.WALL_RMIN_CM + wall_thick:.4f} cm")
        L.append('b:Ge/Wall/Invisible = "true"')
        L.append("")
    # Concentric water organ shells + their two scorers.
    if skin_mode == "lining":
        shells = [("skin", config.PHANTOM_R_CM - LINING_THICK_CM, config.PHANTOM_R_CM)]
    else:
        shells = config.SHELLS
    for name, r_in, r_out in shells:
        comp = _shell_component_name(name)
        L.append(f's:Ge/{comp}/Type     = "TsSphere"')
        L.append(f's:Ge/{comp}/Parent   = "World"')
        L.append(f's:Ge/{comp}/Material = "G4_WATER"')
        L.append(f"d:Ge/{comp}/RMin     = {r_in:.4f} cm")
        L.append(f"d:Ge/{comp}/RMax     = {r_out:.4f} cm")
        if skin_mode == "lining":
            # Upper hemisphere only, as the flood CrewSkin (geometry._sphere_shell).
            L.append(f"d:Ge/{comp}/SPhi     = 0.0 deg")
            L.append(f"d:Ge/{comp}/DPhi     = 360.0 deg")
            L.append(f"d:Ge/{comp}/STheta   = 0.0 deg")
            L.append(f"d:Ge/{comp}/DTheta   = 90.0 deg")
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
    L.append(f"d:So/BeamSpot   = {spot:.1f} cm")
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
        L.append(f"i:So/{tag}/NumberOfHistoriesInRun   = {hist}")
        L.append("")
    return "\n".join(L) + "\n"
