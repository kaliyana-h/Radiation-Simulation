"""paper/buried_depth_series.py: buried-habitat regolith depth curve (no TOPAS)."""
import csv
import dataclasses
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "paper"))
_spec = importlib.util.spec_from_file_location(
    "buried_depth_series", _REPO / "paper" / "buried_depth_series.py")
bds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bds)


def _fake_row(spec, tier, mode, rounds, margin, seed_offset, species=None):
    """Dose falls exp(-x/200) in overhead areal density; 3% errors."""
    x = spec.areal_density_gcm2()
    e = 300.0 * math.exp(-x / 200.0)
    return {"areal_gcm2": round(x, 3), "tier": tier.name, "rounds": rounds,
            "wall_s": 100.0, "total_primaries": 1, "phantom_msv": e,
            "phantom_rel": 0.02, "phantom_eq_rel": 0.03, "skin_msv": 1.2 * e,
            "skin_rel": 0.02, "skin_eq_rel": 0.03, "phantom_mgy_d": e / 1000,
            "skin_mgy_d": e / 900, "fluence_out": 1.0, "fluence_in": 1.0,
            "seed_offset": seed_offset, "provenance": "stamp", "utc": "t",
            "phantom_species": "{}"}


class Series(unittest.TestCase):
    def test_150_matches_shipped_buried150(self):
        s = bds.buried_spec(150.0)
        ref = bds.wsf.designs()["buried150"]
        self.assertEqual(dataclasses.replace(s, name=ref.name), ref)
        self.assertEqual(s.name, ref.name)

    def test_overhead_is_series_coordinate(self):
        for g in bds.DEPTHS_GCM2:
            self.assertAlmostEqual(bds.buried_spec(g).areal_density_gcm2(), g, places=6)
        with self.assertRaises(ValueError):
            bds.buried_spec(2.0)                 # thinner than the Al ceiling

    def test_side_berm_floor_then_tracks_depth(self):
        self.assertAlmostEqual(bds.describe(100.0)["side_gcm2"], 356.0, places=0)
        d = bds.describe(500.0)
        self.assertAlmostEqual(d["side_gcm2"], 500.0 - bds._ceiling_gcm2(), places=0)

    def test_default_depths_all_inside_footprint(self):
        self.assertEqual(bds.check_depths(bds.DEPTHS_GCM2), [])
        self.assertTrue(bds.check_depths([600.0]))     # 'marginal'

    def test_atten_length(self):
        lam, s = bds.atten_length(0.0, 100.0, 0.03, 200.0, 100.0 / math.e, 0.04)
        self.assertAlmostEqual(lam, 200.0)
        self.assertAlmostEqual(s, 200.0 * 0.05)
        self.assertEqual(bds.atten_length(0.0, 1.0, 0.1, 10.0, 1.1, 0.1), (None, None))


class RunAndReport(unittest.TestCase):
    def test_run_refuses_marginal(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                bds.main(["run", "--depths", "600", "--csv", str(Path(d) / "x.csv")])

    def test_run_two_samples_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "s.csv"
            with mock.patch.object(bds.wsf, "run_one", _fake_row):
                for off in ("0", "1000"):
                    bds.main(["run", "--depths", "55.6", "193.7", "--rounds", "2",
                              "--seed-offset", off, "--csv", str(out)])
            with out.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual({r["design"] for r in rows}, {"bur55.6", "bur193.7"})
            s = bds.report(out)
            self.assertEqual([p["overhead_gcm2"] for p in s], [55.6, 193.7])
            self.assertEqual([p["n"] for p in s], [2, 2])
            # pooling two equal samples halves the variance
            self.assertAlmostEqual(s[0]["e_ph"], 0.03 / math.sqrt(2))
            lam, _ = bds.atten_length(55.6, s[0]["phantom_msv"], s[0]["e_ph"],
                                      193.7, s[1]["phantom_msv"], s[1]["e_ph"])
            self.assertAlmostEqual(lam, 200.0, places=3)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "s.csv"
            with mock.patch.object(bds.wsf, "run_one", side_effect=AssertionError):
                bds.main(["run", "--dry-run", "--csv", str(out)])
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
