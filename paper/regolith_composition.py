#!/usr/bin/env python3
"""Regolith composition generator -- the single reproducible source for the
lunar regolith material definitions used in lunar_environment.txt.

Converts a published major-oxide soil assay (wt%) into the 8-element mass
fractions TOPAS needs, renormalised to 1.0, and can emit the TOPAS material
blocks verbatim.  Two sites are defined:

  mare      Apollo-17 high-Ti mare basalt (Taurus-Littrow).  The LunarReg*
            composition already shipped in lunar_environment.txt and the tool's
            DEFAULT -- every existing result/anchor is mare-surface.  NOTE: the
            shipped mare (O 0.4184 ...) was reduced from a fuller assay that
            carried trace oxides; recomputing from the 7 major oxides below
            lands at O 0.4171, ~0.3% off (negligible, within assay scatter).
            The shipped values stay CANONICAL -- this generator does not
            replace them; it documents the method and is the source for
            highland.

  highland  Apollo-16 feldspathic (anorthositic) highland soil.  The
            programmatically relevant south-pole / far-side terrain: much lower
            Fe/Ti (weak fast-neutron producers) and higher Al/Ca, so it
            generates a softer, weaker regolith albedo-neutron field.

TARGET-BLIND: these are measured soil assays, not fitted to any dose. Density
is held at the REDMoon profile (1.76/2.11/1.78 g/cm3) for BOTH sites so a
site swap is a composition-only change and the habitat mass / areal density
(spec.areal_density_gcm2) is identical between sites.

Run (laptop, no TOPAS):
    python3 -m paper.regolith_composition              # comparison table
    python3 -m paper.regolith_composition --emit mare  # TOPAS material blocks
    python3 -m paper.regolith_composition --emit highland
"""
from __future__ import annotations

import argparse
import sys

# Atomic masses (g/mol) for the 8 elements the regolith model carries.
_M = {"O": 16.00, "Na": 22.99, "Mg": 24.31, "Al": 26.98,
      "Si": 28.09, "Ca": 40.08, "Ti": 47.87, "Fe": 55.85}

# TOPAS component names (Geant4 element names) in the fixed vector order.
_ORDER = ["O", "Na", "Mg", "Al", "Si", "Ca", "Ti", "Fe"]
_TOPAS_NAME = {"O": "Oxygen", "Na": "Sodium", "Mg": "Magnesium", "Al": "Aluminum",
               "Si": "Silicon", "Ca": "Calcium", "Ti": "Titanium", "Fe": "Iron"}

# oxide -> (cation, n_cation, n_oxygen, molar mass of the oxide g/mol)
_OX = {"SiO2": ("Si", 1, 2, 60.08), "TiO2": ("Ti", 1, 2, 79.87),
       "Al2O3": ("Al", 2, 3, 101.96), "FeO": ("Fe", 1, 1, 71.85),
       "MgO": ("Mg", 1, 1, 40.30), "CaO": ("Ca", 1, 1, 56.08),
       "Na2O": ("Na", 2, 1, 61.98)}

# Published major-oxide assays (wt%).  Sources in the module docstring.
ASSAYS: dict[str, dict[str, float]] = {
    # Apollo-17 70181 high-Ti mare soil (the composition already shipped).
    "mare":     {"SiO2": 39.9, "TiO2": 9.4, "Al2O3": 11.3, "FeO": 17.5,
                 "MgO": 9.6, "CaO": 10.7, "Na2O": 0.35},
    # Apollo-16 mature feldspathic highland soil.
    "highland": {"SiO2": 45.0, "TiO2": 0.54, "Al2O3": 27.3, "FeO": 5.1,
                 "MgO": 5.7, "CaO": 15.7, "Na2O": 0.46},
}

# REDMoon density profile (Dobynde & Guo 2021), shared by both sites.
DENSITIES = (("176", 1.76), ("211", 2.11), ("178", 1.78))

# TOPAS material-name stem per site: LunarReg* (mare) / LunarHigh* (highland).
STEM = {"mare": "LunarReg", "highland": "LunarHigh"}


def element_fractions(assay: dict[str, float]) -> dict[str, float]:
    """Oxide wt% -> element mass fractions, renormalised to the 8 elements."""
    el = {k: 0.0 for k in _M}
    for ox, wt in assay.items():
        cat, n_cat, n_o, m_ox = _OX[ox]
        el[cat] += wt * (n_cat * _M[cat]) / m_ox
        el["O"] += wt * (n_o * _M["O"]) / m_ox
    tot = sum(el.values())
    return {k: v / tot for k, v in el.items()}


def topas_blocks(site: str) -> str:
    """The TOPAS material definitions for one site at the three REDMoon
    densities -- copy verbatim into lunar_environment.txt."""
    fr = element_fractions(ASSAYS[site])
    comps = " ".join(f'"{_TOPAS_NAME[e]}"' for e in _ORDER)
    fracs = " ".join(f"{fr[e]:.4f}" for e in _ORDER)
    stem = STEM[site]
    out: list[str] = []
    if site == "mare":
        out += ["# NOTE: recomputed mare (O 0.4171) differs ~0.3% from the",
                "# CANONICAL shipped mare (O 0.4184); do NOT replace the shipped",
                "# LunarReg* blocks with these -- it would move every anchor.",
                ""]
    for tag, rho in DENSITIES:
        name = f"{stem}{tag}"
        out += [
            f"s:Ma/{name}/State        = \"Solid\"",
            f"d:Ma/{name}/Density      = {rho:.2f} g/cm3",
            f"sv:Ma/{name}/Components  = 8 {comps}",
            f"uv:Ma/{name}/Fractions   = 8 {fracs}",
            "",
        ]
    return "\n".join(out).rstrip()


def _table() -> None:
    m = element_fractions(ASSAYS["mare"])
    h = element_fractions(ASSAYS["highland"])
    A = {"O": 16, "Na": 23, "Mg": 24, "Al": 27, "Si": 28, "Ca": 40, "Ti": 48, "Fe": 56}
    print(f"{'el':>3} {'mare':>9} {'highland':>9} {'high/mare':>10}")
    for e in _ORDER:
        r = h[e] / m[e] if m[e] else 0.0
        print(f"{e:>3} {m[e]:>9.4f} {h[e]:>9.4f} {r:>9.2f}x")
    print(f"\nmass-weighted <A>:   mare {sum(m[e]*A[e] for e in _ORDER):.2f}"
          f"   highland {sum(h[e]*A[e] for e in _ORDER):.2f}")
    print(f"heavy (Ti+Fe) frac:  mare {m['Ti']+m['Fe']:.4f}"
          f"   highland {h['Ti']+h['Fe']:.4f}")
    print("\nFractions renormalise to 1.0 by construction; density held at the "
          "REDMoon\nprofile for both sites (composition-only change).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", choices=sorted(STEM),
                    help="print the TOPAS material blocks for this site")
    args = ap.parse_args(argv)
    if args.emit:
        print(topas_blocks(args.emit))
    else:
        _table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
