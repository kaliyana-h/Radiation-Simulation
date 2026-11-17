"""Guards for paper/delta_ray_cut_test.py (no TOPAS needed).

The A/B is valid only if modular and nodelta differ in nothing except the phantom's
electron cut, and the shipped ftfp config is the untouched production file.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_delta_ray_harness -v
"""
import importlib.util
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lunarsim.kernel_gen import config, templates  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "delta_ray_cut_test", _REPO_ROOT / "paper" / "delta_ray_cut_test.py")
drt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drt)


def _text(cfg):
    return drt.build_text(cfg, "aluminium", 54.0, "Fe", 0, 1, threads=1)


class DeltaRayHarness(unittest.TestCase):
    def test_ftfp_is_the_production_file(self):
        prod = templates.build_param_file("aluminium", 54.0, "Fe", 0, 1, threads=1,
                                          composition=drt.COMPOSITION,
                                          wall_geometry=drt.WALL_GEOMETRY)
        self.assertEqual(_text("ftfp"), prod)

    def test_nodelta_differs_from_modular_only_by_region_block(self):
        mod = _text("modular").splitlines()
        nod = _text("nodelta").splitlines()
        extra = [ln for ln in nod if ln not in mod]
        self.assertEqual([ln for ln in mod if ln not in nod], [])
        self.assertEqual(extra, drt.region_cut_lines([s[0] for s in config.SHELLS]))

    def test_every_scored_shell_is_in_the_region_and_wall_is_not(self):
        t = _text("nodelta")
        for name, _, _ in config.SHELLS:
            comp = templates._shell_component_name(name)
            self.assertIn(f's:Ge/{comp}/AssignToRegionNamed = "{drt.REGION}"', t)
            self.assertIn(f's:Ge/{comp}/Type', t)       # assigned component exists
        self.assertNotIn("Ge/Wall/AssignToRegionNamed", t)
        self.assertNotIn("CutForAllParticles", t)       # ignored by regions per guide
        self.assertEqual(t.count("CutForElectron"), 1)
        self.assertIn(f"ForRegion/{drt.REGION}/CutForElectron = {drt.NODELTA_E_CUT}", t)

    def test_modular_arms_drop_reference_list(self):
        for cfg in ("modular", "nodelta"):
            t = _text(cfg)
            self.assertNotIn('Ph/Default/Type = "FTFP_BERT_HP"', t)
            self.assertEqual(t.count('s:Ph/Default/Type = "Geant4_Modular"'), 1)
            self.assertIn('"g4ion"', t)

    def test_ratio_error(self):
        r, e = drt.ratio_and_err(2.0, 0.2, 1.0, 0.1)
        self.assertAlmostEqual(r, 2.0)
        self.assertAlmostEqual(e, 2.0 * (0.02) ** 0.5)


if __name__ == "__main__":
    unittest.main()
