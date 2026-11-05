"""Tests for paper/tail_vs_flood.py (step-4 tail vs flood-residual harness).

Pure arithmetic on the committed per-element kernel and synthetic copies of it:
no TOPAS, no MC, no tail JSON needed.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_tail_vs_flood -v
"""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim import dosimetry  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tail_vs_flood", _REPO_ROOT / "paper" / "tail_vs_flood.py")
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)

_FLOOD = {19.0: {"phantom": 200.0, "skin": 220.0, "phantom_rel": 0.1,
                 "skin_rel": 0.01, "stale": []}}


def _scaled_kernel(scale: float) -> dict:
    """The shipped per-element kernel with every response R multiplied by `scale`."""
    K = copy.deepcopy(dosimetry._load_gcr_thinwall_kernel_perelem())
    for p in K["points"]:
        for sp in p["species"].values():
            for organ in sp["R"].values():
                for q in organ:
                    organ[q] = [v * scale for v in organ[q]]
    return K


class TestTailVsFlood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.same = Path(cls.tmp.name) / "same.json"
        cls.same.write_text(json.dumps(dosimetry._load_gcr_thinwall_kernel_perelem()))
        cls.up10 = Path(cls.tmp.name) / "up10.json"
        cls.up10.write_text(json.dumps(_scaled_kernel(1.10)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_identical_tail_has_unit_lift(self):
        r = tv.lift_point(19.0, self.same, "energy")
        for q, _k in tv.QUANTITIES:
            self.assertAlmostEqual(r[f"lift_{q}"], 1.0, places=12)

    def test_scaled_tail_lift_and_residual_division(self):
        (r,) = tv.compare([19.0], self.up10, _FLOOD)
        for q, _k in tv.QUANTITIES:
            self.assertAlmostEqual(r[f"lift_{q}"], 1.10, places=9)
            self.assertAlmostEqual(r[f"lift_{q}_shipped_lis"], 1.10, places=9)
        for q in ("phantom", "skin"):
            self.assertAlmostEqual(r[f"resid_after_{q}"] * 1.10, r[f"resid_{q}"], places=9)

    def test_patch_is_restored(self):
        path0 = dosimetry._GCR_THINWALL_KERNEL_PERELEM
        load0 = dosimetry._load_gcr_thinwall_kernel_perelem
        tv.lift_point(10.0, self.up10, "energy")
        self.assertEqual(dosimetry._GCR_THINWALL_KERNEL_PERELEM, path0)
        self.assertIs(dosimetry._load_gcr_thinwall_kernel_perelem, load0)

    def test_rejects_non_perelement_kernel(self):
        K = dosimetry._load_gcr_thinwall_kernel("aluminium")
        p = Path(self.tmp.name) / "al.json"
        p.write_text(json.dumps(K))
        with self.assertRaises(ValueError):
            with tv.perelem_kernel(p):
                pass

    def test_default_base_is_pretail_and_tail_is_shipped(self):
        self.assertTrue(tv.DEFAULT_BASE.exists(), "pre-tail kernel backup missing")
        base = tv.top_nodes(json.loads(tv.DEFAULT_BASE.read_text()))
        ship = tv.top_nodes(dosimetry._load_gcr_thinwall_kernel_perelem())
        self.assertEqual(tv.DEFAULT_TAIL.resolve(),
                         dosimetry._GCR_THINWALL_KERNEL_PERELEM.resolve())
        for name in ship:
            self.assertLess(base[name], ship[name], name)

    def test_explicit_base_lift_exceeds_one(self):
        r = tv.lift_point(19.0, tv.DEFAULT_TAIL, "energy", tv.DEFAULT_BASE)
        self.assertGreater(r["lift_E"], 1.05)

    def test_shipped_kernel_has_no_missing_flux(self):
        f = tv.missing_flux_fraction(dosimetry._load_gcr_thinwall_kernel_perelem())
        self.assertAlmostEqual(f, 0.0, places=9)

    def test_pretail_missing_flux_fraction_is_small_positive(self):
        f = tv.missing_flux_fraction(json.loads(tv.DEFAULT_BASE.read_text()))
        self.assertGreater(f, 0.0)
        self.assertLess(f, 0.2)


if __name__ == "__main__":
    unittest.main()
