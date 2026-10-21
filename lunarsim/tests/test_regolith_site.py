"""Regression guard for the regolith-site composition feature (Development #1).

HabitatSpec.regolith_site selects which lunar terrain surrounds the habitat --
"mare" (Apollo-17 high-Ti basalt, the DEFAULT and every shipped anchor) or
"highland" (Apollo-16 feldspathic). It is a COMPOSITION-ONLY choice: the REDMoon
density profile is identical between sites, so habitat mass and areal density do
not move, and leaving the field unset must reproduce the mare default exactly.

Four things must not silently regress:
  * the generator (paper/regolith_composition.py) produces a normalised 8-element
    vector for each site, with the highland/mare heavy-element contrast intact;
  * the spec round-trips the field and defaults to mare, and rejects bad sites;
  * bridge writes the correct Ge/RegolithMat/Layer* selectors per site, BEFORE
    the environment include, and the buried berm/overburden follow the site;
  * a site swap leaves areal density / shell mass invariant (mass-neutral), so no
    existing anchor can move.

TARGET-BLIND: no dose value, limit, or external reference appears here.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_regolith_site -v
"""
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim.spec import (HabitatSpec, WallLayer, REGOLITH_SITES,
                           regolith_layer_materials)
from lunarsim.bridge import build_parameter_file, QUICK_LOOK
from paper import regolith_composition as rc


# --------------------------------------------------------------------------
# Composition generator
# --------------------------------------------------------------------------
class TestCompositionGenerator(unittest.TestCase):

    def test_both_sites_normalise_to_one(self):
        for site in ("mare", "highland"):
            fr = rc.element_fractions(rc.ASSAYS[site])
            self.assertAlmostEqual(sum(fr.values()), 1.0, places=9)
            self.assertEqual(set(fr), set(rc._M))  # exactly the 8 elements

    def test_highland_is_a_weaker_heavy_producer(self):
        # The whole physics point: highland strips out the Ti+Fe that drive the
        # mare fast-neutron / HZE-fragment field.
        m = rc.element_fractions(rc.ASSAYS["mare"])
        h = rc.element_fractions(rc.ASSAYS["highland"])
        self.assertLess(h["Ti"] + h["Fe"], 0.25 * (m["Ti"] + m["Fe"]))
        self.assertGreater(h["Al"], 2.0 * m["Al"])

    def test_recomputed_mare_tracks_shipped_default(self):
        # The generator recomputes mare from the stated assay; it must stay within
        # the documented ~0.3% of the canonical shipped vector (it does NOT replace
        # it). Shipped mare, from lunar_environment.txt:
        shipped = {"O": 0.4184, "Na": 0.0026, "Mg": 0.0585, "Al": 0.0604,
                   "Si": 0.1885, "Ca": 0.0773, "Ti": 0.0569, "Fe": 0.1374}
        m = rc.element_fractions(rc.ASSAYS["mare"])
        for el, v in shipped.items():
            self.assertAlmostEqual(m[el], v, delta=0.0015)

    def test_emit_blocks_cover_three_densities(self):
        blocks = rc.topas_blocks("highland")
        for name in ("LunarHigh176", "LunarHigh211", "LunarHigh178"):
            self.assertIn(f"Ma/{name}/Fractions", blocks)


# --------------------------------------------------------------------------
# Spec field: round-trip, default, validation
# --------------------------------------------------------------------------
class TestSpecField(unittest.TestCase):

    def test_default_is_mare(self):
        self.assertEqual(HabitatSpec().regolith_site, "mare")

    def test_round_trip_preserves_site(self):
        for site in REGOLITH_SITES:
            s = HabitatSpec(regolith_site=site)
            self.assertEqual(HabitatSpec.from_json(s.to_json()).regolith_site, site)

    def test_legacy_design_without_field_loads_as_mare(self):
        # A design saved before this feature has no regolith_site key.
        legacy = {"name": "old", "shape": "dome", "inner_radius_cm": 400.0,
                  "walls": [{"material": "aluminium", "thickness_cm": 5.0}]}
        self.assertEqual(HabitatSpec.from_dict(legacy).regolith_site, "mare")

    def test_bad_site_rejected(self):
        with self.assertRaises(ValueError):
            HabitatSpec(regolith_site="mars").validate()


# --------------------------------------------------------------------------
# Bridge: per-site selectors, ordering, buried berm, mass-neutrality
# --------------------------------------------------------------------------
class TestBridgeWiring(unittest.TestCase):

    def _run_file(self, site, shape="buried"):
        spec = HabitatSpec(name="t", shape=shape, inner_radius_cm=300.0,
                           burial_depth_cm=100.0,
                           walls=[WallLayer("aluminium", 5.0)],
                           regolith_site=site)
        return spec, build_parameter_file(spec, QUICK_LOOK,
                                          "gcr_source.txt", "lunar_environment.txt")

    def test_selectors_match_site(self):
        for site in REGOLITH_SITES:
            _, txt = self._run_file(site)
            l1, l2, l3 = regolith_layer_materials(site)
            self.assertIn(f's:Ge/RegolithMat/Layer1 = "{l1}"', txt)
            self.assertIn(f's:Ge/RegolithMat/Layer2 = "{l2}"', txt)
            self.assertIn(f's:Ge/RegolithMat/Layer3 = "{l3}"', txt)

    def test_selectors_precede_environment_include(self):
        # TOPAS forbids redefining an already-set param, so the selectors MUST be
        # written before the include that references them.
        _, txt = self._run_file("highland")
        self.assertLess(txt.index("Ge/RegolithMat/Layer1"),
                        txt.index("includeFile = lunar_environment.txt"))

    def test_buried_berm_follows_site(self):
        _, txt = self._run_file("highland")
        self.assertIn('s:Ge/SideBerm/Material     = "LunarHigh178"', txt)
        self.assertIn('s:Ge/Overburden/Material     = "LunarHigh178"', txt)

    def test_site_swap_is_mass_neutral(self):
        # Composition-only: a mare<->highland swap must not move areal density or
        # shell mass for any shape, so no shielding anchor can shift.
        for shape in ("dome", "cylinder", "quonset", "buried"):
            mare = HabitatSpec(shape=shape, inner_radius_cm=300.0,
                               walls=[WallLayer("aluminium", 5.0)],
                               regolith_site="mare")
            high = mare.copy(regolith_site="highland")
            self.assertAlmostEqual(mare.areal_density_gcm2(),
                                   high.areal_density_gcm2(), places=9)
            self.assertAlmostEqual(mare.shell_mass_kg(),
                                   high.shell_mass_kg(), places=9)


if __name__ == "__main__":
    unittest.main()
