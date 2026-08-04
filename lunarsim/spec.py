"""HabitatSpec — the single contract between the GUI and the TOPAS backend.

A student's design (shape, size, wall layers, materials) is captured here as a
plain dataclass. Every evaluation engine (analytical preview, full TOPAS run)
consumes a HabitatSpec; the GUI only ever produces one. Keeping this the sole
boundary is what lets the front end and the simulation back end evolve
independently.
"""
from __future__ import annotations

import json
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
    # Height of the phantom CENTRE above the floor (z=0). A standing adult's
    # trunk/BFO centre, which is what the dose limits are written against. Fixed,
    # NOT scaled with the habitat: crew do not get taller in a bigger dome.
    crew_height_cm: float = 100.0

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
        # The phantom sits OFF the origin, so "radius < inner_radius" is no longer
        # sufficient -- check the lifted sphere actually clears floor and shell.
        if self.crew_height_cm - self.phantom_radius_cm < 0:
            raise ValueError("crew phantom would sink below the floor: "
                             f"crew_height_cm ({self.crew_height_cm:g}) must be "
                             f">= phantom_radius_cm ({self.phantom_radius_cm:g})")
        if self.shape == "cylinder":
            # flat roof: clear it vertically; the barrel wall is the radial limit
            headroom = self.effective_height_cm
        else:
            # dome / quonset: circular section centred on the floor origin, so the
            # binding constraint is the slant distance to the shell
            headroom = self.inner_radius_cm
        if self.crew_height_cm + self.phantom_radius_cm > headroom:
            raise ValueError(
                f"crew phantom does not fit: centre {self.crew_height_cm:g} cm + "
                f"radius {self.phantom_radius_cm:g} cm exceeds the {headroom:g} cm "
                "interior. Increase the habitat size or lower crew_height_cm.")

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

    # ---- serialisation (design save / load) -------------------------
    # A design is one plain dict so it round-trips through JSON with no custom
    # decoder. The nested wall list is the only non-scalar, so it is spelled out
    # by hand rather than via dataclasses.asdict (which would also drag in any
    # future private fields). from_dict is tolerant of unknown keys so a file
    # saved by a newer build still loads.
    SCHEMA_VERSION = 1

    def to_dict(self) -> dict:
        return {
            "schema": self.SCHEMA_VERSION,
            "name": self.name,
            "shape": self.shape,
            "inner_radius_cm": self.inner_radius_cm,
            "height_cm": self.height_cm,
            "phantom_radius_cm": self.phantom_radius_cm,
            "crew_height_cm": self.crew_height_cm,
            "walls": [{"material": w.material, "thickness_cm": w.thickness_cm}
                      for w in self.walls],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "HabitatSpec":
        if not isinstance(data, dict):
            raise ValueError("design file must be a JSON object")
        walls = [WallLayer(material=w.get("material", "aluminium"),
                           thickness_cm=float(w.get("thickness_cm", 0.0)))
                 for w in data.get("walls", [])]
        if not walls:
            raise ValueError("design file has no wall layers")
        spec = cls(
            name=str(data.get("name", "habitat")),
            shape=str(data.get("shape", "dome")),
            inner_radius_cm=float(data.get("inner_radius_cm", 400.0)),
            height_cm=(None if data.get("height_cm") is None
                       else float(data["height_cm"])),
            walls=walls,
            phantom_radius_cm=float(data.get("phantom_radius_cm", 20.0)),
            crew_height_cm=float(data.get("crew_height_cm", 100.0)),
        )
        spec.validate()
        return spec

    @classmethod
    def from_json(cls, text: str) -> "HabitatSpec":
        return cls.from_dict(json.loads(text))


def default_spec() -> HabitatSpec:
    """The dome design the GUI opens on (mirrors lunar_habitat.txt)."""
    return HabitatSpec(
        name="dome_default",
        shape="dome",
        inner_radius_cm=400.0,
        walls=[WallLayer("polyethylene", 5.0)],
    )
