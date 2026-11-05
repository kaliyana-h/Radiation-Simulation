"""Tests for paper/gate_ratio_decompose.py (step-3 kernel/flood gap harness).

Pure arithmetic on the committed kernels: no TOPAS, no MC.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_gate_decompose -v
"""
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim import dosimetry  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gate_ratio_decompose", _REPO_ROOT / "paper" / "gate_ratio_decompose.py")
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)

_CSV = ("areal_gcm2,al_cm,flood_eff_mSv_yr,flood_skin_mSv_yr,note,provenance\n"
        '19,7.04,200.0,220.0,"x | phantom rel_err  10.0%, skin 0.5%",\n'
        "25,9.26,,,,\n"
        '50,18.52,180.0,210.0,"phantom rel_err 12.0%, skin rel_err 1.0%",\n')


class TestGateDecompose(unittest.TestCase):

    def test_mass_fractions_sum_to_one(self):
        mf = gd.shell_mass_fractions([("a", 0.0, 4.0), ("b", 4.0, 15.0)])
        self.assertAlmostEqual(sum(mf.values()), 1.0)
        self.assertAlmostEqual(mf["a"], 64 / 3375)

    def test_load_flood_parses_both_note_formats_and_skips_blank(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.csv"
            p.write_text(_CSV)
            fl = gd.load_flood(p)
        self.assertEqual(sorted(fl), [19.0, 50.0])
        self.assertAlmostEqual(fl[19.0]["phantom_rel"], 0.10)
        self.assertAlmostEqual(fl[50.0]["skin_rel"], 0.01)
        self.assertEqual(fl[19.0]["stale"], ["unstamped"])

    def test_factors_multiply_to_gap(self):
        flood = {19.0: {"phantom": 200.0, "skin": 220.0, "phantom_rel": 0.1,
                        "skin_rel": 0.01, "stale": []}}
        (r,) = gd.decompose([19.0], flood)
        for q in ("phantom", "skin"):
            self.assertAlmostEqual(r[f"f_lis_{q}"] * r[f"f_comp_{q}"] * r[f"resid_{q}"],
                                   r[f"gap_{q}"], places=9)

    def test_shipped_config_is_the_gui_fold(self):
        """The 'shipped' column must equal what assess_gcr_thinwall shows."""
        from lunarsim.spec import HabitatSpec, WallLayer
        spec = HabitatSpec(name="t", shape="dome", inner_radius_cm=gd.DOME_INNER_R_CM,
                           walls=[WallLayer("aluminium", 19.0 / gd.AL_DENSITY)])
        a = dosimetry.assess_gcr_thinwall(spec)
        k = gd.kernel_point(19.0, dosimetry.KERNEL_LIS_FORM, "perelement")
        self.assertAlmostEqual(k["E"], a.annual_msv, places=6)

    def test_lis_patch_is_restored(self):
        before = dosimetry.KERNEL_LIS_FORM
        gd.kernel_point(10.0, "energy", "perelement")
        self.assertEqual(dosimetry.KERNEL_LIS_FORM, before)

    def test_log_slope_flat_and_sloped(self):
        flat = [{"areal_gcm2": g, "r": 1.3, "r_err": 0.01} for g in (10, 20, 40)]
        s, _e = gd.log_slope(flat, "r")
        self.assertAlmostEqual(s, 0.0, places=9)
        pw = [{"areal_gcm2": g, "r": g ** 0.25, "r_err": 0.01} for g in (10, 20, 40)]
        s, _e = gd.log_slope(pw, "r")
        self.assertAlmostEqual(s, 0.25, places=9)
        self.assertIsNone(gd.log_slope(flat[:1], "r"))


if __name__ == "__main__":
    unittest.main()
