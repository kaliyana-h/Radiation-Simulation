"""HabitatSpec — the single contract between the GUI and the TOPAS backend.

A student's design (shape, size, wall layers, materials) is captured here as a
plain dataclass. Every evaluation engine (analytical preview, full TOPAS run)
consumes a HabitatSpec; the GUI only ever produces one. Keeping this the sole
boundary is what lets the front end and the simulation back end evolve
independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

# ----------------------------------------------------------------------
# Material library
# ----------------------------------------------------------------------
# Maps a friendly GUI label -> (TOPAS material name, density g/cm3, hex colour).
# Density is carried here so we can compute habitat mass (the dose-vs-mass
# trade-off is the pedagogical core of the workshop) without a TOPAS run.
# "regolith" reuses the LunarReg178 material defined in lunar_environment.txt.
MATERIALS: dict[str, dict] = {
    "aluminium":    {"topas": "G4_Al",            "density": 2.70, "colour": "#9aa7b3"},
    "polyethylene": {"topas": "G4_POLYETHYLENE",  "density": 0.94, "colour": "#d9e3a0"},
    "water":        {"topas": "G4_WATER",         "density": 1.00, "colour": "#6fb3ff"},
    "concrete":     {"topas": "G4_CONCRETE",      "density": 2.30, "colour": "#b0b0a0"},
    "regolith":     {"topas": "LunarReg178",      "density": 1.78, "colour": "#8a6a4a"},
    "titanium":     {"topas": "G4_Ti",            "density": 4.51, "colour": "#c0c4c8"},
}

SHAPES = ("dome", "cylinder", "quonset")


@dataclass
class WallLayer:
    """One concentric shielding layer. Order in HabitatSpec.walls is
    innermost-first, so layer 0 is against the habitable volume."""
    material: str = "aluminium"
    thickness_cm: float = 5.0

    def validate(self) -> None:
        if self.material not in MATERIALS:
            raise ValueError(
                f"unknown material {self.material!r}; "
                f"choose from {sorted(MATERIALS)}")
        if self.thickness_cm <= 0:
            raise ValueError("wall thickness must be > 0 cm")


@dataclass
class HabitatSpec:
    """A complete, self-contained description of a student's habitat design."""
    name: str = "habitat"
    shape: str = "dome"                      # dome | cylinder | quonset
    inner_radius_cm: float = 400.0           # interior radius
    height_cm: Optional[float] = None        # axial length (cylinder/quonset); None = use radius
    walls: list[WallLayer] = field(default_factory=lambda: [WallLayer()])
    phantom_radius_cm: float = 20.0          # tissue-equivalent crew proxy

    # ---- validation -------------------------------------------------
    def validate(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {self.shape!r}")
        if self.inner_radius_cm <= 0:
            raise ValueError("inner_radius_cm must be > 0")
        if not self.walls:
            raise ValueError("a habitat needs at least one wall layer")
        for w in self.walls:
            w.validate()
        if self.phantom_radius_cm >= self.inner_radius_cm:
            raise ValueError("phantom must fit inside the habitat")

    # ---- derived geometry ------------------------------------------
    @property
    def total_wall_cm(self) -> float:
        return sum(w.thickness_cm for w in self.walls)

    @property
    def outer_radius_cm(self) -> float:
        return self.inner_radius_cm + self.total_wall_cm

    @property
    def effective_height_cm(self) -> float:
        if self.height_cm is not None:
            return self.height_cm
        if self.shape == "quonset":
            # a half-cylinder should read as a *lying* cylinder -- clearly longer
            # than its diameter (2*r), not a stubby hut. Cap the length so the
            # ends stay well inside the ~9 m GCR source dome (uniform flux).
            return min(self.inner_radius_cm * 3.0, 1200.0)
        return self.inner_radius_cm

    def layer_radii_cm(self) -> list[tuple[float, float]]:
        """(inner, outer) radius of each wall layer, innermost first."""
        r = self.inner_radius_cm
        out = []
        for w in self.walls:
            out.append((r, r + w.thickness_cm))
            r += w.thickness_cm
        return out

    # ---- areal density: the primary shielding figure of merit -------
    def areal_density_gcm2(self) -> float:
        """Total mass per unit frontal area through the wall stack (g/cm^2).
        This is what GCR dose-depth attenuation actually scales with, so it
        is the right quantity to label designs by and to key any cache on."""
        return sum(MATERIALS[w.material]["density"] * w.thickness_cm for w in self.walls)

    # ---- mass estimate (for the dose-vs-mass trade-off) -------------
    def shell_mass_kg(self) -> float:
        """Mass of the wall stack only (crude shell-volume estimate, kg).
        Dome = hemispherical shell; cylinder/quonset = barrel + caps approx."""
        import math
        mass = 0.0
        h = self.effective_height_cm
        for (ri, ro), w in zip(self.layer_radii_cm(), self.walls):
            rho = MATERIALS[w.material]["density"]          # g/cm^3
            if self.shape == "dome":
                vol = (2.0 / 3.0) * math.pi * (ro**3 - ri**3)          # half sphere shell
            elif self.shape == "cylinder":
                vol = math.pi * (ro**2 - ri**2) * h + (2.0/3.0)*math.pi*(ro**3 - ri**3)
            else:  # quonset: half-cylinder arch + two semicircular end walls
                vol = 0.5 * math.pi * (ro**2 - ri**2) * h + (2.0/3.0)*math.pi*(ro**3 - ri**3)
            mass += rho * vol            # g
        return mass / 1000.0             # -> kg

    def copy(self, **changes) -> "HabitatSpec":
        return replace(self, **changes)


def default_spec() -> HabitatSpec:
    """The dome design the GUI opens on (mirrors lunar_habitat.txt)."""
    return HabitatSpec(
        name="dome_default",
        shape="dome",
        inner_radius_cm=400.0,
        walls=[WallLayer("polyethylene", 5.0)],
    )
