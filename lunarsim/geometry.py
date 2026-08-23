"""Parametric habitat geometry generator: HabitatSpec -> TOPAS parameter text.

Generalises the three hand-written habitat files into one generator so a student
can build *any* design. Emits the wall stack (one nested shell per WallLayer),
the inner/outer fluence scorer shells, and the tissue phantom + dose scorer.

Mirrors the proven structure of lunar_habitat.txt: the wall is a shell between
an inner and outer radius; thin air shells just inside/outside score fluence in
vs out; a 20 cm water sphere at crew height scores DoseToMedium.
"""
from __future__ import annotations

import math

from .spec import HabitatSpec, MATERIALS


# Standoff (cm) between an oversized habitat's enclosing radius and the outer gauge.
_GAUGE_STANDOFF_CM = 20.0

# Lateral regolith berm (cm) around a buried habitat's side wall. The buried shape
# sinks a vertical cylinder into the surface: only the flat ceiling takes the user's
# engineered wall stack, while the sides are native regolith. This fixed, deep berm
# (~356 g/cm^2 at rho 1.78) makes the lateral path effectively opaque, so the DEPTH
# of overburden above the ceiling is the single dominant, pedagogically-meaningful
# shielding knob -- not the side thickness. It is in-situ regolith, so it costs no
# launched mass (see spec.shell_mass_kg, which counts only the ceiling).
_BURIED_SIDE_REGOLITH_CM = 200.0

# Fixed outer-gauge radius (cm), IDENTICAL for every design. Chosen to clear the
# core workshop envelope (inner_radius up to ~3.5 m, all three shapes) while staying
# a safe margin inside the 900 cm sky-dome source. Designs larger than this grow the
# gauge to enclose them (see _outer_gauge_radius) and lose strict cross-shape
# comparability -- but those approach the source limit anyway.
#
# Re-anchored 800 -> 880 cm so the whole practical workshop envelope (enclosing
# corner up to ~860 cm) fits the FIXED gauge with gauge_corr == 1.0, instead of
# tripping the mild-extrapolation flag. This is dose-neutral: dosimetry scales
# OUTER_GAUGE_ANCHOR_CAL by (800/880)^2 in lockstep, using the SAME 1/R^2 gauge law
# the code already trusts in _gauge_corr (so every design reads the identical dose;
# only the flag envelope widens). Stays a margin inside the 900 cm beam footprint.
_OUTER_GAUGE_FIXED_CM = 880.0


def _outer_gauge_radius(spec: HabitatSpec) -> float:
    """Inner radius (cm) of the standardised outer-fluence gauge: a hemispherical air
    shell that measures the INCIDENT source field, IDENTICAL (same fixed radius) for
    every design.

    Why a *fixed* radius, not one sized per design. The outer gauge normalises
    absorbed dose to real units (dosimetry.assess divides by fluence_outside).
    fluence_outside is meant to be a pure SOURCE property: OuterShell sits outside all
    walls, so every primary crosses it heading in, BEFORE the habitat scatters
    anything -- its reading must therefore not depend on the habitat at all. Two
    earlier schemes got this wrong:
      * shape-MATCHED (hemisphere-for-dome, uncapped-barrel-for-cylinder): different
        shapes sample the anisotropic GCR field differently -> ~40% cross-shape bias.
      * per-design hemisphere sized to enclose each habitat: scalar fluence in a thin
        shell is track-length/volume ~ N_crossing / R^2, and N_crossing is fixed (all
        launched primaries cross once), so a bigger habitat -> bigger gauge -> LOWER
        fluence_outside -> INFLATED dose. Measured -47%/-66% across shapes: worse.
    A single fixed radius removes both: identical shape AND identical R for all
    designs, so fluence_outside is one honest constant that sees only the source
    (validated shape-invariant to ~+/-4%). The arbitrary absolute value it sets is
    re-anchored to the externally-validated dome by a global factor in dosimetry.

    Enclosing radius (farthest solid corner) is still computed so an OVERSIZED design
    -- one that would poke through the fixed gauge -- grows the gauge to enclose it
    instead of overlapping it (a fatal TOPAS abort). The cap term matters: on a
    cylinder/quonset the flat roof/end caps stack OUTWARD past the barrel by the full
    wall thickness (corner at H + wall_total, not the bare barrel rim).
    """
    # Fixed radius for every normal design; only grow if the habitat is too big to
    # fit inside it (graceful degradation, flagged by exceeding the fixed value).
    return max(_OUTER_GAUGE_FIXED_CM, _enclosing_radius_cm(spec) + _GAUGE_STANDOFF_CM)


def _enclosing_radius_cm(spec: HabitatSpec) -> float:
    """Radius (cm) of the farthest solid corner of the habitat from its centre --
    the smallest sphere that fully contains the walls. Used both to grow the outer
    gauge (above) and, in dosimetry, to test whether a design pokes outside the
    900 cm illuminated source footprint (beam_footprint_flag)."""
    outer = spec.outer_radius_cm
    wall_total = outer - spec.inner_radius_cm          # summed layer thickness
    if spec.shape == "dome":
        return outer
    if spec.shape == "cylinder":
        # farthest = outer top corner of the roof caps: (r=outer, z=H+wall_total)
        return math.hypot(outer, spec.effective_height_cm + wall_total)
    if spec.shape == "buried":
        # farthest = top rim of the regolith overburden cap: it spans radially out to
        # the side berm (r = inner + side) and up to z = H + ceiling + burial_depth.
        r = spec.inner_radius_cm + _BURIED_SIDE_REGOLITH_CM
        z = spec.effective_height_cm + wall_total + spec.burial_depth_cm
        return math.hypot(r, z)
    # quonset: arch of radius `outer`, half-length HL; end caps at Y=HL+wall
    return math.hypot(outer, spec.effective_height_cm / 2.0 + wall_total)


def _sphere_shell(name: str, parent: str, material: str,
                  rmin: float, rmax: float, colour: str,
                  style: str = "Solid") -> str:
    return f"""s:Ge/{name}/Type         = "TsSphere"
s:Ge/{name}/Parent       = "{parent}"
s:Ge/{name}/Material     = "{material}"
d:Ge/{name}/RMin         = {rmin:.3f} cm
d:Ge/{name}/RMax         = {rmax:.3f} cm
d:Ge/{name}/SPhi         = 0.0 deg
d:Ge/{name}/DPhi         = 360.0 deg
d:Ge/{name}/STheta       = 0.0 deg
d:Ge/{name}/DTheta       = 90.0 deg
s:Ge/{name}/DrawingStyle = "{style}"
"""


def _cyl_block(name: str, parent: str, material: str,
               rmin: float, rmax: float, hl: float, transz: float,
               colour: str, dphi: float = 360.0,
               rotx: float | None = None, style: str = "Solid",
               transy: float = 0.0) -> str:
    """Emit a TsCylinder block. dphi=180 + rotx=-90 gives a quonset half-arch
    (axis rotated from Z onto Y, flat diametral face on the z=0 ground plane).
    transy shifts along the world Y axis -- which, after rotx=-90, is the quonset's
    length axis -- so the same helper places the arch's flat end-cap bulkheads."""
    lines = [
        f's:Ge/{name}/Type         = "TsCylinder"',
        f's:Ge/{name}/Parent       = "{parent}"',
        f's:Ge/{name}/Material     = "{material}"',
        f'd:Ge/{name}/RMin         = {rmin:.3f} cm',
        f'd:Ge/{name}/RMax         = {rmax:.3f} cm',
        f'd:Ge/{name}/HL           = {hl:.3f} cm',
        f'd:Ge/{name}/SPhi         = 0.0 deg',
        f'd:Ge/{name}/DPhi         = {dphi:.1f} deg',
    ]
    if rotx is not None:
        lines.append(f'd:Ge/{name}/RotX         = {rotx:.1f} deg')
    if transy:
        lines.append(f'd:Ge/{name}/TransY       = {transy:.3f} cm')
    lines.append(f'd:Ge/{name}/TransZ       = {transz:.3f} cm')
    # NB: no Color line -- MATERIALS colours are hex ("#rrggbb") and TOPAS treats
    # '#' as a comment, truncating the value. The hex colours drive the GUI
    # previews; TOPAS uses its default component colours (as _sphere_shell does).
    lines.append(f's:Ge/{name}/DrawingStyle = "{style}"')
    return "\n".join(lines) + "\n"


def _phantom(spec: HabitatSpec, centre_z: float) -> str:
    """Solid tissue-equivalent crew phantom (point dose) at the given height."""
    return (f's:Ge/Phantom/Type     = "TsSphere"\n'
            f's:Ge/Phantom/Parent   = "World"\n'
            f's:Ge/Phantom/Material = "G4_WATER"\n'
            f'd:Ge/Phantom/RMin     = 0.0 cm\n'
            f'd:Ge/Phantom/RMax     = {spec.phantom_radius_cm:.3f} cm\n'
            f'd:Ge/Phantom/TransZ   = {centre_z:.3f} cm\n'
            f's:Ge/Phantom/Color    = "blue"\n'
            f's:Ge/Phantom/DrawingStyle = "Solid"\n')


def _dome(spec: HabitatSpec) -> tuple[str, list[str]]:
    """Return (geometry_text, wall_component_names) for a hemispherical dome."""
    lines: list[str] = ["# --- Habitat: hemispherical dome (parametric) ---"]
    wall_names: list[str] = []

    # one nested shell per wall layer, innermost first
    for i, ((ri, ro), layer) in enumerate(zip(spec.layer_radii_cm(), spec.walls)):
        name = f"Wall{i}"
        wall_names.append(name)
        topas_mat = MATERIALS[layer.material]["topas"]
        lines.append(f"# layer {i}: {layer.material} {layer.thickness_cm:.1f} cm "
                     f"({ri:.1f}->{ro:.1f} cm)")
        lines.append(_sphere_shell(name, "World", topas_mat, ri, ro,
                                   MATERIALS[layer.material]["colour"]))

    inner = spec.inner_radius_cm
    outer = spec.outer_radius_cm

    # fluence scorer shells (thin air shells just inside / just outside the wall).
    # InnerShell sits a little deeper than the crew-skin shell below so the two
    # don't overlap radially (both are children of the vacuum World).
    lines.append("# fluence scorer shells (air)")
    lines.append(_sphere_shell("InnerShell", "World", "G4_AIR",
                               inner - 5.0, inner - 3.0, "grey", style="Wireframe"))
    # standardised outer fluence gauge: hemispherical shell enclosing the habitat,
    # identical in shape across all designs so fluence_outside is shape-invariant
    # (see _outer_gauge_radius). For a dome the enclosing radius is just the wall.
    _rg = _outer_gauge_radius(spec)
    lines.append(_sphere_shell("OuterShell", "World", "G4_AIR",
                               _rg, _rg + 2.0, "grey", style="Wireframe"))

    # holistic crew-dose shell: a thin tissue-equivalent skin lining the whole
    # inner wall surface. Unlike the single central phantom it catches *every*
    # primary (and back-scattered secondary) that penetrates the dome anywhere,
    # giving a habitat-wide dose with far better statistics (much larger target).
    # Same material as the phantom (water) so the two dose numbers differ only by
    # geometry / self-shielding, not tissue model. Touches the wall at r=inner.
    lines.append("# holistic crew-dose shell (tissue-equivalent inner-wall lining)")
    lines.append(_sphere_shell("CrewSkin", "World", "G4_WATER",
                               inner - 2.0, inner, "blue", style="Wireframe"))

    # crew phantom at standing trunk height (fixed, not scaled with the dome)
    lines.append("# tissue-equivalent crew phantom")
    lines.append(_phantom(spec, spec.crew_height_cm))
    return "\n".join(lines), wall_names


def _cylinder(spec: HabitatSpec) -> tuple[str, list[str]]:
    """Vertical cylinder: barrel side walls (base on the ground at z=0) plus a
    flat, layered roof of stacked caps. Mirrors habitat_cylinder.txt, generalised
    to an arbitrary wall stack."""
    lines: list[str] = ["# --- Habitat: vertical cylinder (parametric) ---"]
    wall_names: list[str] = []
    H = spec.effective_height_cm
    inner = spec.inner_radius_cm
    outer = spec.outer_radius_cm

    # one barrel + one roof cap per wall layer, innermost layer first. The barrel
    # spans z in [0, H]; the caps stack flat on top (innermost material lowest),
    # giving a layered roof of the same total thickness as the side wall.
    for i, ((ri, ro), layer) in enumerate(zip(spec.layer_radii_cm(), spec.walls)):
        topas_mat = MATERIALS[layer.material]["topas"]
        colour = MATERIALS[layer.material]["colour"]
        t = layer.thickness_cm
        lines.append(f"# layer {i}: {layer.material} {t:.1f} cm "
                     f"({ri:.1f}->{ro:.1f} cm)")
        bname = f"Wall{i}"
        wall_names.append(bname)
        lines.append(_cyl_block(bname, "World", topas_mat, ri, ro,
                                H / 2.0, H / 2.0, colour))
        cname = f"Cap{i}"
        wall_names.append(cname)
        cap_z = H + (ri - inner) + t / 2.0       # bottom of this cap sits at z=H+(below)
        lines.append(_cyl_block(cname, "World", topas_mat, 0.0, outer,
                                t / 2.0, cap_z, colour))

    # fluence scorer shells (thin air cylinders just inside / outside the barrel)
    lines.append("# fluence scorer shells (air)")
    lines.append(_cyl_block("InnerShell", "World", "G4_AIR",
                            inner - 5.0, inner - 3.0, H / 2.0, H / 2.0,
                            "grey", style="Wireframe"))
    # standardised outer fluence gauge: a HEMISPHERE enclosing the cylinder (not a
    # barrel), so it samples the anisotropic GCR field identically to every other
    # shape and fluence_outside is shape-invariant (see _outer_gauge_radius).
    _rg = _outer_gauge_radius(spec)
    lines.append(_sphere_shell("OuterShell", "World", "G4_AIR",
                               _rg, _rg + 2.0, "grey", style="Wireframe"))

    # holistic crew-dose shell: tissue lining the barrel side wall (see _dome)
    lines.append("# holistic crew-dose shell (tissue-equivalent inner-wall lining)")
    lines.append(_cyl_block("CrewSkin", "World", "G4_WATER",
                            inner - 2.0, inner, H / 2.0, H / 2.0,
                            "blue", style="Wireframe"))

    # roof-underside disc: the barrel lining above covers only the vertical side
    # wall, leaving the flat roof's inner face unscored. This 2 cm water disc lines
    # that face (z=[H-2, H], flush under the innermost roof cap). Its outer radius
    # stops at inner-6 -- inside the InnerShell air fluence gauge (r=[inner-5,
    # inner-3]) which runs the full barrel height up to the roof -- so the disc
    # clears it instead of clipping it; the thin r=[inner-6, inner] annulus this
    # leaves unlined is a negligible fraction of the roof area. Its RoofDose is
    # mass-weighted into the reported skin dose in run_design (using the same
    # inner-6 radius, see _crewskin_volumes_cm3), giving a true whole-envelope crew
    # skin average rather than a side-wall-only one.
    lines.append(_cyl_block("CrewRoof", "World", "G4_WATER",
                            0.0, inner - 6.0, 1.0, H - 1.0,
                            "blue", style="Wireframe"))

    lines.append("# tissue-equivalent crew phantom (standing trunk height)")
    lines.append(_phantom(spec, spec.crew_height_cm))
    return "\n".join(lines), wall_names


def _buried(spec: HabitatSpec) -> tuple[str, list[str]]:
    """Vertical cylinder sunk into the lunar surface (regolith-covered). The crew
    volume is a cylinder standing on the ground plane; its SIDES and floor are
    native regolith (a deep fixed berm, see _BURIED_SIDE_REGOLITH_CM), and only the
    flat CEILING carries the user's engineered wall stack. A variable depth of loose
    regolith overburden is heaped on top of that ceiling -- the dominant, adjustable
    shield. Modelled above grade (z>=0) as a regolith-encased block rather than a
    below-grade carve: it reuses the validated _cylinder roof/lining machinery, never
    overlaps the z<0 environment slabs, and is shielding-identical (the sign of z is
    irrelevant to attenuation). The GUI preview depicts it as buried."""
    lines: list[str] = ["# --- Habitat: buried vertical cylinder (regolith-covered) ---"]
    wall_names: list[str] = []
    H = spec.effective_height_cm
    inner = spec.inner_radius_cm
    ceiling_total = spec.total_wall_cm
    side = _BURIED_SIDE_REGOLITH_CM
    reg = MATERIALS["regolith"]["topas"]
    reg_col = MATERIALS["regolith"]["colour"]

    # Engineered ceiling: one flat cap per wall layer, innermost lowest, spanning the
    # full interior radius (r in [0, inner]) and stacked z in [H, H+ceiling_total].
    # Mirrors the _cylinder roof caps, but the barrel side wall is regolith, not the
    # user's stack, so there are no `Wall{i}` barrels here.
    for i, ((ri, ro), layer) in enumerate(zip(spec.layer_radii_cm(), spec.walls)):
        topas_mat = MATERIALS[layer.material]["topas"]
        colour = MATERIALS[layer.material]["colour"]
        t = layer.thickness_cm
        lines.append(f"# ceiling layer {i}: {layer.material} {t:.1f} cm "
                     f"(z {H + (ri - inner):.1f}->{H + (ro - inner):.1f} cm)")
        cname = f"Cap{i}"
        wall_names.append(cname)
        cap_z = H + (ri - inner) + t / 2.0
        lines.append(_cyl_block(cname, "World", topas_mat, 0.0, inner,
                                t / 2.0, cap_z, colour))

    # Native regolith side berm: a thick annular wall around the crew cylinder,
    # r in [inner, inner+side], full height up to the top of the ceiling stack.
    lines.append(f"# native regolith side berm ({side:.0f} cm)")
    lines.append(_cyl_block("SideBerm", "World", reg, inner, inner + side,
                            (H + ceiling_total) / 2.0, (H + ceiling_total) / 2.0,
                            reg_col))

    # Regolith overburden heaped on the ceiling: a full disc (out to the berm rim)
    # of the user-set depth, sitting on top of the engineered ceiling. This is the
    # dominant, variable shield.
    lines.append(f"# regolith overburden ({spec.burial_depth_cm:.0f} cm)")
    over_z = H + ceiling_total + spec.burial_depth_cm / 2.0
    lines.append(_cyl_block("Overburden", "World", reg, 0.0, inner + side,
                            spec.burial_depth_cm / 2.0, over_z, reg_col))

    # fluence scorer shells (thin air cylinders just inside the barrel)
    lines.append("# fluence scorer shells (air)")
    lines.append(_cyl_block("InnerShell", "World", "G4_AIR",
                            inner - 5.0, inner - 3.0, H / 2.0, H / 2.0,
                            "grey", style="Wireframe"))
    # standardised outer fluence gauge: a HEMISPHERE enclosing the whole buried
    # block, identical in shape to every other design (see _outer_gauge_radius).
    _rg = _outer_gauge_radius(spec)
    lines.append(_sphere_shell("OuterShell", "World", "G4_AIR",
                               _rg, _rg + 2.0, "grey", style="Wireframe"))

    # holistic crew-dose shell: tissue lining the barrel side wall (see _dome)
    lines.append("# holistic crew-dose shell (tissue-equivalent inner-wall lining)")
    lines.append(_cyl_block("CrewSkin", "World", "G4_WATER",
                            inner - 2.0, inner, H / 2.0, H / 2.0,
                            "blue", style="Wireframe"))

    # roof-underside disc lining the ceiling's inner face (see _cylinder CrewRoof).
    lines.append(_cyl_block("CrewRoof", "World", "G4_WATER",
                            0.0, inner - 6.0, 1.0, H - 1.0,
                            "blue", style="Wireframe"))

    lines.append("# tissue-equivalent crew phantom (standing trunk height)")
    lines.append(_phantom(spec, spec.crew_height_cm))
    return "\n".join(lines), wall_names


def _quonset(spec: HabitatSpec) -> tuple[str, list[str]]:
    """Quonset hut: a half-cylinder arch lying on the ground, axis along Y
    (RotX=-90), flat diametral face on the z=0 plane. Mirrors habitat_quonset.txt,
    generalised to an arbitrary wall stack."""
    lines: list[str] = ["# --- Habitat: quonset half-cylinder arch (parametric) ---"]
    wall_names: list[str] = []
    HL = spec.effective_height_cm / 2.0          # half-length along the arch axis
    inner = spec.inner_radius_cm
    outer = spec.outer_radius_cm

    for i, ((ri, ro), layer) in enumerate(zip(spec.layer_radii_cm(), spec.walls)):
        topas_mat = MATERIALS[layer.material]["topas"]
        colour = MATERIALS[layer.material]["colour"]
        t = layer.thickness_cm
        lines.append(f"# layer {i}: {layer.material} {t:.1f} cm "
                     f"({ri:.1f}->{ro:.1f} cm)")
        name = f"Wall{i}"
        wall_names.append(name)
        lines.append(_cyl_block(name, "World", topas_mat, ri, ro, HL, 0.0,
                                colour, dphi=180.0, rotx=-90.0))

        # Flat end bulkheads sealing the two open ends of the tunnel. Without
        # these the arch shields only the curved wall and GCR streams in axially
        # through the bare semicircular openings, inflating the dose relative to a
        # closed dome. Each layer becomes a half-disc slab (radius 0->outer, upper
        # half, on the ground plane) of thickness t, stacked outward along the
        # length axis (world Y after rotx=-90) so the full stack -- and full areal
        # density -- caps each end. Offset mirrors the roof caps in _cylinder:
        # innermost flush at the arch end (Y=HL), each outer layer pushed out by
        # the cumulative thickness below it.
        cap_y = HL + (ri - inner) + t / 2.0
        for sign, tag in ((+1.0, "Pos"), (-1.0, "Neg")):
            cname = f"Cap{tag}{i}"
            wall_names.append(cname)
            lines.append(_cyl_block(cname, "World", topas_mat, 0.0, outer,
                                    t / 2.0, 0.0, colour, dphi=180.0,
                                    rotx=-90.0, transy=sign * cap_y))

    # fluence scorer shells (thin air half-cylinders inside / outside the arch)
    lines.append("# fluence scorer shells (air, half-cylinder)")
    lines.append(_cyl_block("InnerShell", "World", "G4_AIR",
                            inner - 5.0, inner - 3.0, HL, 0.0, "grey",
                            dphi=180.0, rotx=-90.0, style="Wireframe"))
    # standardised outer fluence gauge: a full HEMISPHERE enclosing the arch (not a
    # half-cylinder), so it samples the anisotropic GCR field identically to every
    # other shape and fluence_outside is shape-invariant (see _outer_gauge_radius).
    _rg = _outer_gauge_radius(spec)
    lines.append(_sphere_shell("OuterShell", "World", "G4_AIR",
                               _rg, _rg + 2.0, "grey", style="Wireframe"))

    # holistic crew-dose shell: tissue lining the inner arch surface (see _dome)
    lines.append("# holistic crew-dose shell (tissue-equivalent inner-wall lining)")
    lines.append(_cyl_block("CrewSkin", "World", "G4_WATER",
                            inner - 2.0, inner, HL, 0.0, "blue",
                            dphi=180.0, rotx=-90.0, style="Wireframe"))

    # end-cap linings: the arch CrewSkin above covers only the curved wall, leaving
    # the inner faces of the two flat bulkheads (CapPos/CapNeg) unscored. Each end
    # gets a 2 cm water half-disc flush against its bulkhead (Y=+/-HL inner face).
    # Outer radius stops at inner-6 to clear the InnerShell air gauge (r=[inner-5,
    # inner-3]) and the arch lining (r=[inner-2, inner]); the thin unlined rim is a
    # negligible fraction of the end area. Both ends are lined and scored separately
    # (not one end doubled) because a directional SPE can hit the two ends
    # unequally; run_design mass-weights CrewCapA/B into the reported skin dose.
    for tag, sign in (("A", +1.0), ("B", -1.0)):
        lines.append(_cyl_block(f"CrewCap{tag}", "World", "G4_WATER",
                                0.0, inner - 6.0, 1.0, 0.0, "blue",
                                dphi=180.0, rotx=-90.0,
                                transy=sign * (HL - 1.0), style="Wireframe"))

    lines.append("# tissue-equivalent crew phantom (standing trunk height)")
    lines.append(_phantom(spec, spec.crew_height_cm))
    return "\n".join(lines), wall_names


_BUILDERS = {
    "dome": _dome,
    "cylinder": _cylinder,
    "quonset": _quonset,
    "buried": _buried,
}


def build_geometry(spec: HabitatSpec) -> str:
    """Geometry + phantom parameter block for the given spec."""
    spec.validate()
    builder = _BUILDERS.get(spec.shape)
    if builder is None:
        raise NotImplementedError(
            f"shape {spec.shape!r} not generated yet "
            f"(available: {sorted(_BUILDERS)}); will mirror the existing "
            f"habitat_{spec.shape}.txt next.")
    geom, _wall_names = builder(spec)
    return geom


def build_scorers(spec: HabitatSpec, ion_z: int = 0, ion_a: int = 0) -> str:
    """Fluence-in / fluence-out / phantom-dose scorers.

    ion_z / ion_a (the primary's atomic number and mass) restrict the wall-fluence
    scorers to the incident primary species. The bare Fluence quantity counts the
    full scalar fluence including the wall's back-scattered delta-ray / fragment
    shower, which for heavy ions scales ~Z^2 and exactly cancels the Z^2 skin-dose
    enhancement in the per-species normalisation rate = D * Phi / F -- collapsing
    the HZE dose. Filtering to the primary's exact Z and A makes F the true incident
    species fluence, so the flux normalisation rescales primaries-to-primaries as
    intended. We filter by Z/A rather than particle name because Geant4 GenericIons
    are created dynamically and are NOT in the static particle table, so
    OnlyIncludeParticlesNamed rejects ion names ("Fe56" -> unknown particle); the
    atomic-number / atomic-mass filters work uniformly for protons, alphas and ions.
    For a heavy primary every wall fragment is lighter (Z<Z_primary), so the Z filter
    isolates the primary cleanly; the A filter additionally drops same-Z fragments of
    a different isotope. ion_z<=0 leaves the scorers unfiltered (legacy / non-GCR)."""
    if ion_z > 0:
        lines = [f"i:Sc/{{name}}/OnlyIncludeParticlesOfAtomicNumber = {ion_z}"]
        if ion_a > 0:
            lines.append(f"i:Sc/{{name}}/OnlyIncludeParticlesOfAtomicMass = {ion_a}")
        pfilter = "\n".join(lines) + "\n"
    else:
        pfilter = ""
    scorers = """# --- Scorers ---
s:Sc/OutsideWallFluence/Quantity   = "Fluence"
s:Sc/OutsideWallFluence/Component  = "OuterShell"
s:Sc/OutsideWallFluence/OutputFile = "fluence_outside"
s:Sc/OutsideWallFluence/IfOutputFileAlreadyExists = "Overwrite"
{outer_filter}
s:Sc/InsideWallFluence/Quantity    = "Fluence"
s:Sc/InsideWallFluence/Component   = "InnerShell"
s:Sc/InsideWallFluence/OutputFile  = "fluence_inside"
s:Sc/InsideWallFluence/IfOutputFileAlreadyExists = "Overwrite"
{inner_filter}
""".format(outer_filter=pfilter.format(name="OutsideWallFluence"),
           inner_filter=pfilter.format(name="InsideWallFluence")) + """

s:Sc/PhantomDose/Quantity   = "DoseToMedium"
s:Sc/PhantomDose/Component  = "Phantom"
s:Sc/PhantomDose/OutputFile = "phantom_dose"
s:Sc/PhantomDose/IfOutputFileAlreadyExists = "Overwrite"

# LET-weighted dose-equivalent (ICRP-60 Q(L)) on the central phantom, mirroring
# the skin lining. The phantom is a SOLID sphere where heavy ions stop (Bragg
# peak), so its field is far harder than the thin lining's; giving it its own
# emergent Q (instead of a flat default) is what stops the crew point dose being
# both over-weighted and physically wrong for HZE ions.
s:Sc/PhantomDoseEq/Quantity   = "DoseEquivalent_ICRP"
s:Sc/PhantomDoseEq/Component  = "Phantom"
s:Sc/PhantomDoseEq/OutputFile = "phantom_doseeq"
s:Sc/PhantomDoseEq/IfOutputFileAlreadyExists = "Overwrite"

# NASA/Cucinotta Q twin of the phantom dose-equivalent: same transport, same
# per-step LET, only the quality-factor mapping differs (harder GCR Q than
# ICRP-60). Scored alongside so its emergent Q is COMPUTED per design, not scaled.
s:Sc/PhantomDoseEqNasa/Quantity   = "DoseEquivalent_NASA"
s:Sc/PhantomDoseEqNasa/Component  = "Phantom"
s:Sc/PhantomDoseEqNasa/OutputFile = "phantom_doseeq_nasa"
s:Sc/PhantomDoseEqNasa/IfOutputFileAlreadyExists = "Overwrite"

# holistic habitat-wide dose: tissue-equivalent skin lining the whole inner wall
s:Sc/SkinDose/Quantity   = "DoseToMedium"
s:Sc/SkinDose/Component  = "CrewSkin"
s:Sc/SkinDose/OutputFile = "skin_dose"
s:Sc/SkinDose/IfOutputFileAlreadyExists = "Overwrite"

# LET-weighted dose-equivalent (ICRP-60 Q(L)) on the same skin lining: custom
# scorer that applies the per-particle quality factor at the source, so HZE ions
# are weighted correctly instead of by a single flat field Q. (DoseEquivalentICRP
# extension; value is Sv, scored in "Gy" units since Q is dimensionless.)
s:Sc/SkinDoseEq/Quantity   = "DoseEquivalent_ICRP"
s:Sc/SkinDoseEq/Component  = "CrewSkin"
s:Sc/SkinDoseEq/OutputFile = "skin_doseeq"
s:Sc/SkinDoseEq/IfOutputFileAlreadyExists = "Overwrite"

# NASA/Cucinotta Q twin of the skin-lining dose-equivalent (headline scorer).
s:Sc/SkinDoseEqNasa/Quantity   = "DoseEquivalent_NASA"
s:Sc/SkinDoseEqNasa/Component  = "CrewSkin"
s:Sc/SkinDoseEqNasa/OutputFile = "skin_doseeq_nasa"
s:Sc/SkinDoseEqNasa/IfOutputFileAlreadyExists = "Overwrite"

# Neutron-lineage twin of SkinDoseEq: the SAME ICRP-60 dose-equivalent, but
# summed only over energy deposited by charged particles descended from a neutron
# (DoseEquivalentICRPNeutron extension). Scored on the same CrewSkin lining as
# SkinDoseEq, so skin_doseeq_neutron / skin_doseeq is the wall-bred secondary
# (albedo) neutron dose fraction -- a normalisation-invariant diagnostic (both
# numerator and denominator share the run's flux/gauge scaling, which cancels).
s:Sc/SkinDoseEqNeutron/Quantity   = "DoseEquivalent_ICRP_Neutron"
s:Sc/SkinDoseEqNeutron/Component  = "CrewSkin"
s:Sc/SkinDoseEqNeutron/OutputFile = "skin_doseeq_neutron"
s:Sc/SkinDoseEqNeutron/IfOutputFileAlreadyExists = "Overwrite"
"""

    # Shape-specific secondary linings, folded into the reported skin dose by
    # run_design. The dome hemisphere already envelops its crew with one shell, so
    # it defines no extra component; emitting these scorers for it would reference
    # an undefined volume and abort the run.
    #   cylinder -> CrewRoof   (flat roof underside; the barrel lining is side-only)
    #   buried   -> CrewRoof   (same flat ceiling underside; sides are regolith)
    #   quonset  -> CrewCapA/B (the two flat end bulkheads; the arch lining is
    #               curved-wall-only)
    if spec.shape in ("cylinder", "buried"):
        scorers += """
# roof-underside disc lining (cylinder / buried)
s:Sc/RoofDose/Quantity   = "DoseToMedium"
s:Sc/RoofDose/Component  = "CrewRoof"
s:Sc/RoofDose/OutputFile = "roof_dose"
s:Sc/RoofDose/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/RoofDoseEq/Quantity   = "DoseEquivalent_ICRP"
s:Sc/RoofDoseEq/Component  = "CrewRoof"
s:Sc/RoofDoseEq/OutputFile = "roof_doseeq"
s:Sc/RoofDoseEq/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/RoofDoseEqNasa/Quantity   = "DoseEquivalent_NASA"
s:Sc/RoofDoseEqNasa/Component  = "CrewRoof"
s:Sc/RoofDoseEqNasa/OutputFile = "roof_doseeq_nasa"
s:Sc/RoofDoseEqNasa/IfOutputFileAlreadyExists = "Overwrite"
"""
    elif spec.shape == "quonset":
        scorers += """
# end-cap bulkhead linings (quonset only); A = +Y end, B = -Y end
s:Sc/CapADose/Quantity   = "DoseToMedium"
s:Sc/CapADose/Component  = "CrewCapA"
s:Sc/CapADose/OutputFile = "capa_dose"
s:Sc/CapADose/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/CapADoseEq/Quantity   = "DoseEquivalent_ICRP"
s:Sc/CapADoseEq/Component  = "CrewCapA"
s:Sc/CapADoseEq/OutputFile = "capa_doseeq"
s:Sc/CapADoseEq/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/CapADoseEqNasa/Quantity   = "DoseEquivalent_NASA"
s:Sc/CapADoseEqNasa/Component  = "CrewCapA"
s:Sc/CapADoseEqNasa/OutputFile = "capa_doseeq_nasa"
s:Sc/CapADoseEqNasa/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/CapBDose/Quantity   = "DoseToMedium"
s:Sc/CapBDose/Component  = "CrewCapB"
s:Sc/CapBDose/OutputFile = "capb_dose"
s:Sc/CapBDose/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/CapBDoseEq/Quantity   = "DoseEquivalent_ICRP"
s:Sc/CapBDoseEq/Component  = "CrewCapB"
s:Sc/CapBDoseEq/OutputFile = "capb_doseeq"
s:Sc/CapBDoseEq/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/CapBDoseEqNasa/Quantity   = "DoseEquivalent_NASA"
s:Sc/CapBDoseEqNasa/Component  = "CrewCapB"
s:Sc/CapBDoseEqNasa/OutputFile = "capb_doseeq_nasa"
s:Sc/CapBDoseEqNasa/IfOutputFileAlreadyExists = "Overwrite"
"""
    return scorers
