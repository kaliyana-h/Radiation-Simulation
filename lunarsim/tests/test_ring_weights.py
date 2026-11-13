"""Opt-in GCR ring weighting (make_source.ring_histories): default unchanged,
solid-angle shares exact and total-preserving."""
import importlib.util
import math
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _ms(env=None):
    old = os.environ.get("LUNARSIM_GCR_RING_WEIGHTS")
    try:
        if env is None:
            os.environ.pop("LUNARSIM_GCR_RING_WEIGHTS", None)
        else:
            os.environ["LUNARSIM_GCR_RING_WEIGHTS"] = env
        spec = importlib.util.spec_from_file_location("ms_ringw", ROOT / "make_source.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        if old is None:
            os.environ.pop("LUNARSIM_GCR_RING_WEIGHTS", None)
        else:
            os.environ["LUNARSIM_GCR_RING_WEIGHTS"] = old


class RingWeights(unittest.TestCase):
    def test_default_is_equal(self):
        ms = _ms()
        self.assertEqual(ms.GCR_RING_WEIGHTS, "equal")
        self.assertEqual(ms.ring_histories(6, 12, 140), [[140] * 12] * 6)
        self.assertEqual(ms.gcr_block(2, 4, 400.0, 10),
                         ms.gcr_block(2, 4, 400.0, 10, ring_weights="equal"))

    def test_shares_are_solid_angle(self):
        ms = _ms()
        s = ms.ring_band_shares(6)
        self.assertAlmostEqual(sum(s), 1.0, places=12)
        for r, x in enumerate(s):      # ΔΩ/2π of band between sin² = r/6 and (r+1)/6
            lo, hi = math.asin(math.sqrt(r / 6)), math.asin(math.sqrt((r + 1) / 6))
            self.assertAlmostEqual(x, math.cos(lo) - math.cos(hi), places=12)
        self.assertGreater(s[-1] / s[0], 4.0)     # grazing band dominates

    def test_solidangle_total_and_proportion(self):
        ms = _ms()
        for rings, az, h in ((6, 12, 140), (5, 8, 25), (6, 12, 8)):
            hist = ms.ring_histories(rings, az, h, "solidangle")
            per = [sum(x) for x in hist]
            self.assertEqual(sum(per), rings * az * h)
            for n, share in zip(per, ms.ring_band_shares(rings)):
                self.assertLessEqual(abs(n - share * rings * az * h), 1.0)
            for x in hist:
                self.assertLessEqual(max(x) - min(x), 1)

    def test_env_selects_and_block_emits(self):
        ms = _ms("solidangle")
        blk = ms.gcr_block(6, 12, 400.0, 140)
        self.assertIn("RING WEIGHTS: solidangle", blk)
        n = [int(l.split("=")[1]) for l in blk.splitlines() if "NumberOfHistoriesInRun" in l]
        self.assertEqual(len(n), 72)
        self.assertEqual(sum(n), 6 * 12 * 140)

    def test_bad_scheme(self):
        with self.assertRaises(ValueError):
            _ms().ring_histories(6, 12, 140, "lambert")


if __name__ == "__main__":
    unittest.main()
