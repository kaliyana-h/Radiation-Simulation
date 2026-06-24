"""Parametric habitat geometry generator: HabitatSpec -> TOPAS parameter text.

Generalises the three hand-written habitat files into one generator so a student
can build *any* design. Emits the wall stack (one nested shell per WallLayer),
the inner/outer fluence scorer shells, and the tissue phantom + dose scorer.

Mirrors the proven structure of lunar_habitat.txt: the wall is a shell between
an inner and outer radius; thin air shells just inside/outside score fluence in
vs out; a 20 cm water sphere at crew height scores DoseToMedium.
"""
from __future__ import annotations

from .spec import HabitatSpec, MATERIALS


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
               rotx: float | None = None, style: str = "Solid") -> str:
    """Emit a TsCylinder block. dphi=180 + rotx=-90 gives a quonset half-arch
    (axis rotated from Z onto Y, flat diametral face on the z=0 ground plane)."""
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
    lines.append(_sphere_shell("OuterShell", "World", "G4_AIR",
                               outer + 1.0, outer + 3.0, "grey", style="Wireframe"))

    # holistic crew-dose shell: a thin tissue-equivalent skin lining the whole
    # inner wall surface. Unlike the single central phantom it catches *every*
    # primary (and back-scattered secondary) that penetrates the dome anywhere,
    # giving a habitat-wide dose with far better statistics (much larger target).
    # Same material as the phantom (water) so the two dose numbers differ only by
    # geometry / self-shielding, not tissue model. Touches the wall at r=inner.
    lines.append("# holistic crew-dose shell (tissue-equivalent inner-wall lining)")
    lines.append(_sphere_shell("CrewSkin", "World", "G4_WATER",
                               inner - 2.0, inner, "blue", style="Wireframe"))

    # crew phantom at mid-dome height
    lines.append("# tissue-equivalent crew phantom")
    lines.append(_phantom(spec, inner / 2.0))
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
    lines.append(_cyl_block("OuterShell", "World", "G4_AIR",
                            outer + 1.0, outer + 3.0, H / 2.0, H / 2.0,
                            "grey", style="Wireframe"))

    # holistic crew-dose shell: tissue lining the barrel side wall (see _dome)
    lines.append("# holistic crew-dose shell (tissue-equivalent inner-wall lining)")
    lines.append(_cyl_block("CrewSkin", "World", "G4_WATER",
                            inner - 2.0, inner, H / 2.0, H / 2.0,
                            "blue", style="Wireframe"))

    lines.append("# tissue-equivalent crew phantom (mid-height)")
    lines.append(_phantom(spec, H / 2.0))
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
        lines.append(f"# layer {i}: {layer.material} {layer.thickness_cm:.1f} cm "
                     f"({ri:.1f}->{ro:.1f} cm)")
        name = f"Wall{i}"
        wall_names.append(name)
        lines.append(_cyl_block(name, "World", topas_mat, ri, ro, HL, 0.0,
                                colour, dphi=180.0, rotx=-90.0))

    # fluence scorer shells (thin air half-cylinders inside / outside the arch)
    lines.append("# fluence scorer shells (air, half-cylinder)")
    lines.append(_cyl_block("InnerShell", "World", "G4_AIR",
                            inner - 5.0, inner - 3.0, HL, 0.0, "grey",
                            dphi=180.0, rotx=-90.0, style="Wireframe"))
    lines.append(_cyl_block("OuterShell", "World", "G4_AIR",
                            outer + 1.0, outer + 3.0, HL, 0.0, "grey",
                            dphi=180.0, rotx=-90.0, style="Wireframe"))

    # holistic crew-dose shell: tissue lining the inner arch surface (see _dome)
    lines.append("# holistic crew-dose shell (tissue-equivalent inner-wall lining)")
    lines.append(_cyl_block("CrewSkin", "World", "G4_WATER",
                            inner - 2.0, inner, HL, 0.0, "blue",
                            dphi=180.0, rotx=-90.0, style="Wireframe"))

    lines.append("# tissue-equivalent crew phantom (mid-arch height)")
    lines.append(_phantom(spec, inner / 2.0))
    return "\n".join(lines), wall_names


_BUILDERS = {
    "dome": _dome,
    "cylinder": _cylinder,
    "quonset": _quonset,
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


def build_scorers(spec: HabitatSpec) -> str:
    """Fluence-in / fluence-out / phantom-dose scorers."""
    return """# --- Scorers ---
s:Sc/OutsideWallFluence/Quantity   = "Fluence"
s:Sc/OutsideWallFluence/Component  = "OuterShell"
s:Sc/OutsideWallFluence/OutputFile = "fluence_outside"
s:Sc/OutsideWallFluence/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/InsideWallFluence/Quantity    = "Fluence"
s:Sc/InsideWallFluence/Component   = "InnerShell"
s:Sc/InsideWallFluence/OutputFile  = "fluence_inside"
s:Sc/InsideWallFluence/IfOutputFileAlreadyExists = "Overwrite"

s:Sc/PhantomDose/Quantity   = "DoseToMedium"
s:Sc/PhantomDose/Component  = "Phantom"
s:Sc/PhantomDose/OutputFile = "phantom_dose"
s:Sc/PhantomDose/IfOutputFileAlreadyExists = "Overwrite"

# holistic habitat-wide dose: tissue-equivalent skin lining the whole inner wall
s:Sc/SkinDose/Quantity   = "DoseToMedium"
s:Sc/SkinDose/Component  = "CrewSkin"
s:Sc/SkinDose/OutputFile = "skin_dose"
s:Sc/SkinDose/IfOutputFileAlreadyExists = "Overwrite"
"""
