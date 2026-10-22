"""Regression guard for the small-sample convergence gate (phantom estimator).

The batch loop stops when a dose relative error drops below target. The relative
error is sem/mean with sem = s/sqrt(n); for the small n a run reaches (2-12) the
sample stdev s is a skewed-LOW estimate of the true sigma, so the point-estimate
rel_err can dip under target on a lucky draw and freeze the headline at a biased
value (the measured 326->212 phantom collapse). jobs._gate_inflation fixes this
by stopping on the one-sided 84% UPPER confidence bound instead -- rel_err x a
chi^2 factor that is huge at n=2-3 and relaxes to ~1 as n grows.

Four things must not silently regress:
  * the factor is a decreasing sequence, >1 for small n and ~1 for the largest n;
  * it is a no-op (1.0) when n is unknown or below 2, and clamps above the cap;
  * a design whose point-estimate rel_err satisfies target at small n is NOT
    allowed to stop there (the artifact) but IS allowed once n is large enough;
  * the gate only ever DELAYS a stop -- it never stops a loop sooner than the
    raw point estimate would.

TARGET-BLIND: no dose value, limit, or external reference appears here. Runs
without dash, so it is laptop-safe.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_convergence_gate -v
"""
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim.jobs import _gate_inflation, _GATE_CONF_FACTOR


class TestGateInflationShape(unittest.TestCase):

    def test_noop_when_n_unusable(self):
        for n in (None, 0, 1):
            self.assertEqual(_gate_inflation(n), 1.0)

    def test_strictly_decreasing_in_n(self):
        vals = [_gate_inflation(n) for n in range(2, 13)]
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(a, b)

    def test_small_n_heavily_penalised_large_n_mild(self):
        # n=2 must all but forbid a stop; by the batch cap the penalty is mild.
        self.assertGreater(_gate_inflation(2), 4.0)
        self.assertGreater(_gate_inflation(3), 2.0)
        self.assertLess(_gate_inflation(12), 1.4)
        self.assertGreater(_gate_inflation(12), 1.0)

    def test_clamps_above_the_batch_cap(self):
        # n beyond the table (should not occur) must not KeyError; it clamps to
        # the gentlest tabulated inflation, never below 1.0.
        top = _GATE_CONF_FACTOR[max(_GATE_CONF_FACTOR)]
        self.assertEqual(_gate_inflation(99), top)
        self.assertGreaterEqual(_gate_inflation(99), 1.0)


class TestGateDecision(unittest.TestCase):
    """The gate arithmetic the loop runs: stop iff rel * factor(n) <= target."""

    def _stops(self, rel, n, target):
        return rel * _gate_inflation(n) <= target

    def test_artifact_draw_is_not_trusted_at_n4(self):
        # A phantom point estimate that LOOKS converged at n=4 (8% < 10% target)
        # must NOT stop the loop: inflated it is ~15%, above target.
        target, rel = 0.10, 0.08
        self.assertFalse(self._stops(rel, 4, target))

    def test_same_error_trusted_once_n_is_large(self):
        # The identical raw error, once built from enough batches, DOES stop:
        # 7% inflates past a 10% target at n=4 (~13%) but clears it at n=12 (~9%).
        target, rel = 0.10, 0.07
        self.assertFalse(self._stops(rel, 4, target))
        self.assertTrue(self._stops(rel, 12, target))

    def test_genuinely_tight_error_still_stops_early(self):
        # The gate must not punish a quantity that is truly well below target:
        # the skin lining (rel ~0.006) clears a 5% target even inflated at n=4.
        self.assertTrue(self._stops(0.006, 4, 0.05))

    def test_gate_only_delays_never_advances_a_stop(self):
        # For every n the inflated error is >= the raw error, so any stop the gate
        # allows the raw estimate would also have allowed -- it can only delay.
        for n in range(2, 13):
            self.assertGreaterEqual(_gate_inflation(n), 1.0)


if __name__ == "__main__":
    unittest.main()
