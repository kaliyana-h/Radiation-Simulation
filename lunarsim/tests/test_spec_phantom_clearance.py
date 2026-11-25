"""The crew phantom must clear the in-wall scorer volumes (its World siblings)."""
import re
import unittest

from lunarsim import geometry
from lunarsim.spec import (HabitatSpec, WallLayer, INNER_SCORER_CLEARANCE_CM,
                           ROOF_LINING_CM, eva_suit_spec, default_spec)


def _spec(shape, R, H=None, crew=100.0, r=20.0):
    return HabitatSpec(name="t", shape=shape, inner_radius_cm=R, height_cm=H,
                       walls=[WallLayer("aluminium", 5.0)],
                       phantom_radius_cm=r, crew_height_cm=crew)


def _par(txt, comp, key):
    m = re.search(rf"d:Ge/{comp}/{key}\s*=\s*(-?[\d.]+) cm", txt)
    return float(m.group(1))


class ConstantsMatchGeometry(unittest.TestCase):
    def test_inner_gauge_and_linings(self):
        for shape in ("dome", "cylinder", "quonset", "buried"):
            s = _spec(shape, 400.0, 250.0)
            t = geometry.build_geometry(s)
            # innermost in-wall scorer face is the InnerShell gauge
            self.assertAlmostEqual(_par(t, "InnerShell", "RMin"),
                                   400.0 - INNER_SCORER_CLEARANCE_CM, msg=shape)
            self.assertGreaterEqual(_par(t, "CrewSkin", "RMin"),
                                    400.0 - INNER_SCORER_CLEARANCE_CM, msg=shape)
            if shape in ("cylinder", "buried"):
                bottom = _par(t, "CrewRoof", "TransZ") - _par(t, "CrewRoof", "HL")
                self.assertAlmostEqual(bottom, 250.0 - ROOF_LINING_CM)
            if shape == "quonset":
                face = _par(t, "CrewCapA", "TransY") - _par(t, "CrewCapA", "HL")
                self.assertAlmostEqual(face, 125.0 - ROOF_LINING_CM)


class Validate(unittest.TestCase):
    def test_shipped_presets_pass(self):
        default_spec().validate()
        eva_suit_spec().validate()
        _spec("dome", 150.0).validate()          # GUI slider minimum (1.5 m)

    def test_dome_band_inside_gauge_rejected(self):
        _spec("dome", 125.0).validate()          # 100 + 20 = 120 <= 120
        with self.assertRaises(ValueError):      # top 122 > 120: inside the gauge
            _spec("dome", 125.0, crew=102.0).validate()

    def test_barrel_radial_and_roof(self):
        _spec("cylinder", 25.0, 200.0, r=20.0).validate()
        with self.assertRaises(ValueError):
            _spec("cylinder", 24.0, 200.0, r=20.0).validate()
        _spec("buried", 300.0, 122.0).validate()
        with self.assertRaises(ValueError):       # top 120 > 121-2 roof disc
            _spec("buried", 300.0, 121.0).validate()

    def test_quonset_bulkhead(self):
        with self.assertRaises(ValueError):
            _spec("quonset", 300.0, H=40.0, crew=50.0).validate()


if __name__ == "__main__":
    unittest.main()
