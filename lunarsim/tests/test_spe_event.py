"""Regression guard for the selectable SPE design event (acute-exposure scenario).

The GUI exposes three canonical solar-particle events (Feb-1956 / Oct-1989 /
Aug-1972); the default is the HARD Feb-1956 because, behind a thick regolith wall,
the hard tail penetrates while the softer events are ranged out. Three things must
not silently regress:

  * the new OCT1989_SPE constant carries the fluence_cm2 that the fold's own
    normalisation implies (the phi(>30 MeV) anchor integrated over the sampled band);
  * the three events stay strictly ordered aug1972 < oct1989 < feb1956 in BOTH the
    penetrating fluence and the deep (BFO) dose-equivalent behind thick shielding --
    i.e. the hard event binds inside the habitat;
  * the per-event mean quality factor rises with spectral hardness.

TARGET-BLIND: no dose value, limit, or external reference appears here. Runs without
dash or TOPAS (the acute verdict is a response-kernel fold), so it is laptop-safe.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_spe_event -v
"""
import math
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import make_source as ms
from lunarsim import dosimetry
from lunarsim.bridge import WORST_CASE_SPE, OCT1989_SPE, SOFT_SPE
from lunarsim.spec import HabitatSpec, WallLayer


def _band_ratio(r0: float, n: int = 40000) -> float:
    """band-integral / >30 MeV integral of the King shape, as the fold normalises."""
    lo, hi = 10.0, 3000.0
    g = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    f = [dosimetry._spe_dNdE(e, r0, None) for e in g]
    band = sum(0.5 * (f[i] + f[i + 1]) * (g[i + 1] - g[i]) for i in range(n - 1))
    g2 = [30.0 * (hi / 30.0) ** (i / (n - 1)) for i in range(n)]
    f2 = [dosimetry._spe_dNdE(e, r0, None) for e in g2]
    a30 = sum(0.5 * (f2[i] + f2[i + 1]) * (g2[i + 1] - g2[i]) for i in range(n - 1))
    return band / a30


class _Result:
    """Minimal stand-in: assess_spe reads only result.spec (for the 1/R^2 gauge)."""
    def __init__(self, spec):
        self.spec = spec


class TestSpeEvent(unittest.TestCase):
    def setUp(self):
        self.spec = HabitatSpec(
            name="spe_test_ref", shape="dome", inner_radius_cm=300.0,
            walls=[WallLayer(material="aluminium", thickness_cm=20.0)],
            regolith_site="mare")
        self.result = _Result(self.spec)
        self.events = [SOFT_SPE, OCT1989_SPE, WORST_CASE_SPE]  # soft -> hard

    def test_oct1989_fluence_is_band_consistent(self):
        # fluence_cm2 must equal the >30 MeV anchor x the King band ratio for R0=150.
        r0 = ms.SPE_EVENTS["oct1989"][0]
        expect = ms.spe_integral_fluence("oct1989") * _band_ratio(r0)
        self.assertAlmostEqual(OCT1989_SPE.fluence_cm2 / expect, 1.0, places=2)

    def test_oct1989_lies_between_the_brackets(self):
        self.assertGreater(OCT1989_SPE.fluence_cm2, 0.0)
        # total band fluence ordering: hardest event has the LEAST fluence
        self.assertLess(WORST_CASE_SPE.fluence_cm2, OCT1989_SPE.fluence_cm2)
        self.assertLess(OCT1989_SPE.fluence_cm2, SOFT_SPE.fluence_cm2)
        # hardness ordering of the characteristic rigidity
        self.assertLess(ms.SPE_EVENTS["aug1972"][0], ms.SPE_EVENTS["oct1989"][0])
        self.assertLess(ms.SPE_EVENTS["oct1989"][0], ms.SPE_EVENTS["feb1956"][0])

    def test_penetrating_fluence_rises_with_hardness(self):
        pen = [dosimetry.assess_spe(self.result, s, skin=False).sim_fluence_cm2
               for s in self.events]
        self.assertTrue(pen[0] < pen[1] < pen[2],
                        f"penetrating fluence not monotone in hardness: {pen}")

    def test_deep_dose_rises_with_hardness(self):
        bfo = [dosimetry.assess_spe(self.result, s, skin=False).bfo_msv
               for s in self.events]
        self.assertTrue(bfo[0] < bfo[1] < bfo[2],
                        f"BFO dose not monotone in hardness behind thick wall: {bfo}")

    def test_quality_factor_rises_with_hardness(self):
        q = [dosimetry.assess_spe(self.result, s, skin=False).bfo_quality_factor
             for s in self.events]
        self.assertTrue(q[0] < q[1] < q[2],
                        f"BFO quality factor not monotone in hardness: {q}")

    def test_hard_event_is_the_shielded_worst_case(self):
        # The default (WORST_CASE_SPE) must be the worst at the deep shell.
        bfo = {s.event: dosimetry.assess_spe(self.result, s, skin=False).bfo_msv
               for s in self.events}
        self.assertEqual(max(bfo, key=bfo.get), WORST_CASE_SPE.event)


if __name__ == "__main__":
    unittest.main()
