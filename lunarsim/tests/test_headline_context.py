"""Regression guard for the GCR headline provenance / trust flags.

Every GCR headline must say which engine produced it, carry the areal-density
confidence band, warn when the design sits within +/-20% of the kernel<->flood
crossover, and label a flat-Q fallback. These are ADVISORY strings -- the test
checks only which flags fire and their severity, never a dose (target-blind).
Dash-free and TOPAS-free, so it is laptop-safe.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_headline_context -v
"""
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lunarsim import dosimetry
from lunarsim.headline_context import (headline_flags, is_q_fallback,
                                       NEAR_GATE_FRAC, DEFAULT_CROSSOVER_GCM2,
                                       COMBINED_ENVELOPE_NOTE)


def _keys(flags):
    return [f["key"] for f in flags]


class TestHeadlineContext(unittest.TestCase):
    def test_crossover_matches_shipped_kernel(self):
        K = dosimetry._load_gcr_thinwall_kernel("aluminium")
        self.assertEqual(K["meta"]["crossover_gcm2"], DEFAULT_CROSSOVER_GCM2)

    def test_engine_always_first_and_named(self):
        for eng, word in (("kernel", "kernel fold"), ("flood", "flood Monte Carlo")):
            f = headline_flags(54.0, eng, 2.5)
            self.assertEqual(f[0]["key"], "engine")
            self.assertIn(word, f[0]["text"])
        with self.assertRaises(ValueError):
            headline_flags(54.0, "magic", 2.5)

    def test_confidence_band_tracks_areal_density(self):
        lv = {ad: next(f for f in headline_flags(ad, "flood", 2.5)
                       if f["key"] == "confidence")["level"]
              for ad in (5.0, 25.0, 54.0)}
        self.assertEqual(lv, {5.0: "warn", 25.0: "alert", 54.0: "ok"})

    def test_near_gate_window(self):
        g = DEFAULT_CROSSOVER_GCM2
        inside = (g * (1 - NEAR_GATE_FRAC) + 0.01, g - 0.5, g, g + 0.5,
                  g * (1 + NEAR_GATE_FRAC) - 0.01)
        outside = (g * (1 - NEAR_GATE_FRAC) - 0.5, g * (1 + NEAR_GATE_FRAC) + 0.5,
                   5.0, 54.0)
        for ad in inside:
            self.assertIn("near_gate", _keys(headline_flags(ad, "flood", 2.5)), ad)
        for ad in outside:
            self.assertNotIn("near_gate", _keys(headline_flags(ad, "flood", 2.5)), ad)

    def test_near_gate_names_the_other_engine(self):
        below = next(f for f in headline_flags(17.0, "kernel", 2.0)
                     if f["key"] == "near_gate")["text"]
        above = next(f for f in headline_flags(21.0, "flood", 2.0)
                     if f["key"] == "near_gate")["text"]
        self.assertIn("flood MC", below)
        self.assertIn("thin-wall kernel", above)

    def test_q_fallback_only_on_exact_default(self):
        d = dosimetry.DEFAULT_QUALITY_FACTOR
        self.assertTrue(is_q_fallback(d))
        self.assertFalse(is_q_fallback(d + 1e-3))
        self.assertFalse(is_q_fallback(None))
        self.assertIn("q_fallback", _keys(headline_flags(54.0, "flood", d)))
        self.assertNotIn("q_fallback", _keys(headline_flags(54.0, "flood", 2.63)))

    def test_envelope_note_says_not_one_epoch(self):
        self.assertIn("not a single solar epoch", COMBINED_ENVELOPE_NOTE)


if __name__ == "__main__":
    unittest.main()
