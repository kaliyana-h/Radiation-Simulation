"""Regression tests for the 2026-11-16 code-review fixes (no TOPAS needed).

    cd ~/topas && python3 -m unittest lunarsim.tests.test_review_fixes -v
"""
import math
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim import HabitatSpec, WallLayer, dosimetry  # noqa: E402
from lunarsim.dosimetry import GcrModel  # noqa: E402


def _ms(**kw):
    return dosimetry._make_source_for(GcrModel(**kw))


def _al_dome(areal):
    return HabitatSpec(name="rf", shape="dome", inner_radius_cm=400.0,
                       walls=[WallLayer("aluminium", areal / 2.699)])


class SolarPhaseMapping(unittest.TestCase):
    def test_W_mapping_is_independent_of_lis_form(self):
        rig, eng = _ms(lis_form="rigidity"), _ms(lis_form="energy")
        for phi in (400.0, 650.0, 1100.0):
            self.assertEqual(rig.matthia_W_for_phi(phi), eng.matthia_W_for_phi(phi))

    def test_W_rises_with_phi(self):
        ms = _ms()
        w = [ms.matthia_W_for_phi(p) for p in (400.0, 650.0, 1100.0)]
        self.assertLess(w[0], w[1])
        self.assertLess(w[1], w[2])
        self.assertGreater(w[1], 20.0)   # was 3 under the energy form (unmodulated)

    def test_W_map_matches_high_energy_swing(self):
        """2026-11-17: W is set by the >1 GeV proton-flux swing, not spectral shape
        (the shape fit gave W 90 at phi 1100 and under-modulated solar max)."""
        ms = _ms()
        self.assertEqual([ms.matthia_W_for_phi(p) for p in (400.0, 650.0, 1100.0)],
                         [4.0, 63.0, 122.0])
        for phi in (650.0, 1100.0):
            ff = (ms._proton_integral_above(lambda e: ms._forcefield_proton_rig(e, 400.0), 1000.0)
                  / ms._proton_integral_above(lambda e: ms._forcefield_proton_rig(e, phi), 1000.0))
            W0, W = ms.matthia_W_for_phi(400.0), ms.matthia_W_for_phi(phi)
            mat = (ms._proton_integral_above(lambda e: ms.mattia_flux(1, e, W0), 1000.0)
                   / ms._proton_integral_above(lambda e: ms.mattia_flux(1, e, W), 1000.0))
            self.assertLess(abs(mat / ff - 1.0), 0.03)

    def test_kernel_abundances_follow_phi(self):
        """The per-element kernel fold re-derives abundances at the requested phi, so
        solar max lowers the thin-wall dose more than the shape alone would."""
        s = _al_dome(10.0)
        lo = dosimetry.fold_gcr_thinwall(s, 400.0)["E_sv_s"]
        hi = dosimetry.fold_gcr_thinwall(s, 1100.0)["E_sv_s"]
        self.assertLess(hi / lo, 0.75)
        a400 = dosimetry._perelement_abundances(400.0)
        a1100 = dosimetry._perelement_abundances(1100.0)
        self.assertNotEqual(a400, a1100)


class GcrSpectrum(unittest.TestCase):
    def test_perelement_uses_single_element_matthiae_shape(self):
        ms = _ms(perelement="perelement")
        W = ms.matthia_W_for_phi(400.0)
        E, w = ms.gcr_spectrum(400.0, z=26, a=56)
        ref = [ms.mattia_flux(26, e / 56.0, W) for e in E]
        peak = max(ref)
        for got, want in zip(w, ref):
            self.assertAlmostEqual(got, want / peak, places=12)

    def test_matthia2013_uses_group_shape(self):
        ms = _ms(perelement="matthia2013")
        W = ms.matthia_W_for_phi(400.0)
        E, w = ms.gcr_spectrum(400.0, z=26, a=56)
        ref = [ms.mattia_group_flux(26, e / 56.0, W) for e in E]
        peak = max(ref)
        for got, want in zip(w, ref):
            self.assertAlmostEqual(got, want / peak, places=12)

    def test_unknown_model_raises(self):
        ms = _ms()
        saved = ms.GCR_PERELEMENT
        try:
            ms.GCR_PERELEMENT = "bogus"
            with self.assertRaises(ValueError):
                ms.gcr_spectrum(400.0)
        finally:
            ms.GCR_PERELEMENT = saved

    def test_default_grid_resolves_high_energy_tail(self):
        """TOPAS interpolates the density linearly between points; the default grid
        must keep the sampled >1 GeV/n fraction within 0.5% of a fine reference."""
        ms = _ms()

        def frac_above(n, cut):
            E, w = ms.gcr_spectrum(400.0, n=n)
            E, w = [0.0] + E, [0.0] + w
            tot = above = 0.0
            for i in range(len(E) - 1):
                a, b, fa, fb = E[i], E[i + 1], w[i], w[i + 1]
                seg = 0.5 * (fa + fb) * (b - a)
                tot += seg
                if cut <= a:
                    above += seg
                elif cut < b:
                    fc = fa + (fb - fa) * (cut - a) / (b - a)
                    above += 0.5 * (fc + fb) * (b - cut)
            return above / tot

        ref = frac_above(2000, 1000.0)
        self.assertLess(abs(frac_above(ms.SPECTRUM_POINTS, 1000.0) / ref - 1.0), 0.005)


class KernelErrorBars(unittest.TestCase):
    def _fold(self, point, organs):
        return dosimetry._thinwall_fold_point(point, 400.0, organs)

    def _point_and_organs(self):
        if not dosimetry._has_perelem_kernel():
            self.skipTest("per-element kernel not shipped")
        kern = dosimetry._load_gcr_thinwall_kernel_perelem()
        point = next(p for p in kern["points"] if p["wall_gcm2"] > 0)
        return point, kern["meta"]["organs"]

    def test_correlated_shells_at_least_independent_sum(self):
        point, organs = self._point_and_organs()
        f = self._fold(point, organs)
        indep = sum((w ** 2) * f["HIvar"][k] for k, w in organs)
        self.assertGreaterEqual(f["Evar"], indep * (1.0 - 1e-12))

    def test_bessel_applied_only_to_legacy_kernels(self):
        point, organs = self._point_and_organs()
        n = int(point["seeds"])
        legacy = self._fold(dict(point, sem_estimator=None), organs)["Evar"]
        sample = self._fold(dict(point, sem_estimator="sample"), organs)["Evar"]
        self.assertAlmostEqual(legacy / sample, n / (n - 1), places=9)


class ErrorLabel(unittest.TestCase):
    """GUI error labels: unknown (None) is said, not hidden; 0 and <1% are shown."""

    def test_pm(self):
        try:
            from lunarsim import gui
        except ImportError as e:                       # dash not installed
            self.skipTest(f"gui import: {e}")
        self.assertEqual(gui._pm(None), " (error n/a)")
        self.assertEqual(gui._pm(None, unknown=""), "")
        self.assertEqual(gui._pm(0.0), " ± 0.0%")
        self.assertEqual(gui._pm(0.005), " ± 0.5%")
        self.assertEqual(gui._pm(0.14), " ± 14%")


if __name__ == "__main__":
    unittest.main()
