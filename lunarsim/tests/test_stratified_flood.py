"""Tests for the energy-stratified flood (make_source bands + jobs estimator).

No TOPAS: band slicing is checked on the real make_source spectrum, and the
driver runs against a synthetic run_design whose per-primary tallies are known
per band, so the recombined estimate must reproduce the continuous expectation.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_stratified_flood -v
"""
import importlib.util
import io
import os
import random
import sys
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim import HabitatSpec, WallLayer, jobs, dosimetry, bridge  # noqa: E402
from lunarsim.bridge import RunTier, RunResult  # noqa: E402

TIER = RunTier("t", rings=2, azimuth=4, histories=40)


def _ms():
    return dosimetry._load_make_source()


def _spec():
    return HabitatSpec(name="strat", shape="dome", inner_radius_cm=300.0,
                       walls=[WallLayer("aluminium", 20.0)])


def _band_result(tier, k, d, f=2.0, noise=0.0, rng=None, **extra):
    """Synthetic band run: per-primary dose d[k] (+ relative noise), per-primary
    outer fluence f, skin twice the phantom."""
    n = tier.total_primaries
    eps = (1.0 + noise * rng.gauss(0.0, 1.0)) if noise else 1.0
    dose = n * d[k] * eps
    kw = dict(dose_gy=dose, phantom_doseeq_sv=3 * dose, phantom_doseeq_nasa_sv=4 * dose,
              skin_dose_gy=2 * n * d[k], skin_doseeq_sv=5 * n * d[k],
              skin_doseeq_nasa_sv=6 * n * d[k],
              fluence_inside=0.5 * n * f, fluence_outside=n * f)
    kw.update(extra)
    return RunResult(spec=_spec(), tier=tier, run_dir=Path("."), returncode=0,
                     wall_seconds=1.0 + 0.01 * n, log_tail=f"band{k}", **kw)


class TestBands(unittest.TestCase):

    def setUp(self):
        self.e, self.w = _ms().gcr_spectrum(400.0, z=1, a=1)

    def test_k1_block_byte_identical(self):
        ms = _ms()
        a = ms.gcr_block(2, 4, 400.0, 10)
        b = ms.gcr_block(2, 4, 400.0, 10, strata=1, stratum=0)
        self.assertEqual(a, b)
        self.assertNotIn("STRATUM", a)
        e, w = ms.stratum_spectrum(self.e, self.w, 1, 0)
        self.assertEqual((e, w), (list(self.e), list(self.w)))
        with self.assertRaises(ValueError):
            ms.stratum_spectrum(self.e, self.w, 1, 1)

    def test_fractions_sum_and_match_areas(self):
        ms = _ms()
        for K in (2, 3, 4, 6):
            p = ms.stratum_fractions(self.e, self.w, K)
            self.assertEqual(len(p), K)
            self.assertAlmostEqual(sum(p), 1.0, places=12)
            self.assertTrue(all(x > 0 for x in p))
            areas = [ms.topas_area(*ms.stratum_spectrum(self.e, self.w, K, k))
                     for k in range(K)]
            full = ms.topas_area(self.e, self.w)
            self.assertAlmostEqual(sum(areas) / full, 1.0, delta=1e-5)
            for pk, ak in zip(p, areas):
                self.assertAlmostEqual(pk, ak / sum(areas), places=12)

    def test_guard_and_shared_edges(self):
        ms = _ms()
        K = 4
        edges = ms.stratum_edges(len(self.e), K)
        self.assertEqual(edges[0], 0)
        self.assertEqual(edges[-1], len(self.e) - 1)
        prev_top = None
        for k in range(K):
            e, w = ms.stratum_spectrum(self.e, self.w, K, k)
            if k == 0:
                self.assertNotEqual(w[0], 0.0)           # natural 0 -> E_min ramp
                self.assertAlmostEqual(e[0], self.e[0], places=6)
            else:
                self.assertEqual(w[0], 0.0)              # guard point
                self.assertLess(e[0], e[1])
                self.assertAlmostEqual(e[0] / e[1], 1 - ms.STRATUM_GUARD_REL, places=9)
                self.assertAlmostEqual(e[1], prev_top, places=6)   # shared edge
            prev_top = e[-1]
        self.assertAlmostEqual(prev_top, self.e[-1], places=6)
        with self.assertRaises(ValueError):
            ms.stratum_edges(len(self.e), len(self.e))

    def test_block_header_and_points(self):
        blk = _ms().gcr_block(2, 4, 400.0, 10, strata=4, stratum=2)
        self.assertIn("ENERGY STRATUM 3/4", blk)
        line = next(l for l in blk.splitlines() if "BeamEnergySpectrumValues" in l)
        e, _ = _ms().stratum_spectrum(self.e, self.w, 4, 2)
        self.assertIn(f"= {len(e)} ", line)

    def test_band_fractions_follow_active_model(self):
        p = jobs.band_fractions(400.0, 1, 1, 4)
        self.assertEqual(p, _ms().stratum_fractions(self.e, self.w, 4))
        # upper bands carry the bulk of the flux at phi=400
        self.assertGreater(p[2], p[0])


class TestBridgeCmd(unittest.TestCase):

    def _cmd(self, **kw):
        with mock.patch.object(bridge.subprocess, "run") as run:
            with tempfile.TemporaryDirectory() as d:
                bridge.generate_source(Path(d), TIER, **kw)
            return run.call_args[0][0]

    def test_strata_flag_only_when_split(self):
        self.assertNotIn("--strata", self._cmd())
        self.assertNotIn("--strata", self._cmd(strata=1, stratum=0))
        cmd = self._cmd(strata=4, stratum=3)
        i = cmd.index("--strata")
        self.assertEqual(cmd[i:i + 4], ["--strata", "4", "--stratum", "3"])


class TestVirtualEstimator(unittest.TestCase):

    def test_recovers_continuous_expectation(self):
        p = [0.1, 0.2, 0.3, 0.4]
        d = [5.0, 1.0, 2.0, 0.5]
        bands = [_band_result(replace(TIER, histories=h), k, d)
                 for k, h in enumerate([1, 3, 20, 7])]
        v = jobs._stratified_virtual(TIER, bands, p)
        n_ref = TIER.total_primaries
        mean_d = sum(pk * dk for pk, dk in zip(p, d))
        self.assertAlmostEqual(v.dose_gy, n_ref * mean_d, places=9)
        self.assertAlmostEqual(v.skin_dose_gy, 2 * n_ref * mean_d, places=9)
        self.assertAlmostEqual(v.fluence_outside, n_ref * 2.0, places=9)
        self.assertAlmostEqual(v.transmission, 0.5, places=12)
        self.assertEqual(v.tier, TIER)
        self.assertAlmostEqual(v.wall_seconds, sum(b.wall_seconds for b in bands))

    def test_k1_is_identity(self):
        b = _band_result(TIER, 0, [3.0], neutron_doseeq_fraction=0.2,
                         si_charged_let_ratios=(0.1, 0.2))
        v = jobs._stratified_virtual(TIER, [b], [1.0])
        for f in jobs._EXTENSIVE_FIELDS + tuple(jobs._RATIO_BASE):
            self.assertEqual(getattr(v, f), getattr(b, f), f)
        for x, y in zip(v.si_charged_let_ratios, (0.1, 0.2)):
            self.assertAlmostEqual(x, y, places=12)

    def test_ratio_weighted_by_base_share(self):
        p = [0.5, 0.5]
        d = [1.0, 3.0]                       # band 1 carries 3/4 of the skin dose
        b0 = _band_result(replace(TIER, histories=10), 0, d, neutral_dose_fraction=0.4)
        b1 = _band_result(replace(TIER, histories=2), 1, d, neutral_dose_fraction=0.0)
        v = jobs._stratified_virtual(TIER, [b0, b1], p)
        self.assertAlmostEqual(v.neutral_dose_fraction, 0.1, places=12)

    def test_missing_tally_is_none(self):
        d = [1.0, 1.0]
        b0 = _band_result(TIER, 0, d, si_to_skin_ratio=0.3)
        b1 = _band_result(TIER, 1, d)
        v = jobs._stratified_virtual(TIER, [b0, b1], [0.5, 0.5])
        self.assertIsNone(v.si_to_skin_ratio)
        self.assertIsNone(v.si_charged_let_ratios)


class TestAllocation(unittest.TestCase):

    def test_proportional_floor(self):
        h = jobs._proportional_histories([0.01, 0.29, 0.7], TIER)
        self.assertEqual(h, [1, 12, 28])

    def test_neyman_budget_mix_floor(self):
        p, sig, cost = [0.2, 0.3, 0.5], [10.0, 1.0, 0.0], [0.01, 0.01, 0.01]
        sources = 8
        h = jobs._neyman_histories(p, sig, cost, budget_s=8.0, sources=sources)
        self.assertTrue(all(x >= 1 for x in h))
        # zero-variance band still keeps the defensive proportional share
        n_prop2 = 8.0 * 0.5 / (0.01 * 1.0)
        self.assertEqual(h[2], round(jobs.STRAT_DEFENSIVE * n_prop2 / sources))
        self.assertGreater(h[0], h[1])
        self.assertAlmostEqual(sum(h) * sources * 0.01, 8.0, delta=0.05 * 3)
        # alpha = 1 -> exactly proportional
        h1 = jobs._neyman_histories(p, sig, cost, 8.0, sources, alpha=1.0)
        self.assertEqual(h1, [round(8.0 * x / 0.01 / sources) for x in p])
        # tiny budget -> floor
        self.assertEqual(jobs._neyman_histories(p, sig, cost, 1e-6, sources), [1, 1, 1])
        # no variance anywhere -> proportional
        h0 = jobs._neyman_histories(p, [0, 0, 0], cost, 8.0, sources)
        self.assertEqual(h0, h1)
        with self.assertRaises(ValueError):
            jobs._neyman_histories(p, sig, [0.0, 1, 1], 8.0, sources)


class TestDriver(unittest.TestCase):
    """_run_composition_stratified against a synthetic run_design."""

    D = {"H": [4e-12, 2e-12, 1e-12, 3e-12]}

    def _fake(self, rng, noise, calls):
        def run_design(spec, tier, seed=1, keep=False, particle="proton", ion_z=1,
                       ion_a=1, cancel_cb=None, strata=1, stratum=0, **_):
            calls.append((seed, stratum, tier.histories))
            d = self.D["H"] if strata > 1 else [sum(
                pk * dk for pk, dk in zip(jobs.band_fractions(tier.phi_mv, 1, 1, 4),
                                          self.D["H"]))]
            return _band_result(tier, stratum, d, noise=noise, rng=rng)
        return run_design

    def _run(self, noise=0.0, **kw):
        comp_H = [("H", "proton", 1, 1, 1.0, "H")]
        calls = []
        rng = random.Random(7)
        with mock.patch.object(jobs, "run_design", self._fake(rng, noise, calls)):
            c = jobs.run_composition(_spec(), TIER, composition=comp_H,
                                     sampling="stratified", strata=4, **kw)
        return c, calls

    def test_exact_bands_match_continuous(self):
        c, calls = self._run(max_batches=3, min_batches=3)
        self.assertEqual(c.sampling, "stratified")
        self.assertEqual(c.strata, 4)
        self.assertEqual(c.provenance["sampling"], "stratified")
        self.assertEqual(c.provenance["strata"], 4)
        self.assertEqual(len(calls), 3 * 4)
        self.assertEqual(len({s for s, _, _ in calls}), 12)       # distinct seeds
        with dosimetry.use_model(c.model):          # the flood's model, not the default
            p = jobs.band_fractions(400.0, 1, 1, 4)
        st = jobs._species_tier(TIER, 1)
        want = st.total_primaries * sum(pk * dk for pk, dk in zip(p, self.D["H"]))
        r = c.species_results[0][1]
        self.assertAlmostEqual(r.dose_gy / want, 1.0, places=12)
        self.assertEqual(r.n_batches, 3)
        self.assertEqual(c.total_primaries,
                         sum(h * TIER.rings * TIER.azimuth for _, _, h in calls))
        # with the same synthetic physics, continuous gives the same number
        with mock.patch.object(jobs, "run_design", self._fake(None, 0.0, [])):
            cc = jobs.run_composition(_spec(), TIER,
                                      composition=[("H", "proton", 1, 1, 1.0, "H")],
                                      max_batches=3, min_batches=3)
        self.assertEqual(cc.sampling, "continuous")
        self.assertNotIn("sampling", cc.provenance)
        a = dosimetry.assess_composition(c.species_results, phi_MV=400.0, model=c.model)
        b = dosimetry.assess_composition(cc.species_results, phi_MV=400.0, model=cc.model)
        self.assertAlmostEqual(a.annual_msv / b.annual_msv, 1.0, places=9)

    def test_pilot_then_frozen_allocation(self):
        c, calls = self._run(noise=0.3, max_batches=5, min_batches=5)
        with dosimetry.use_model(c.model):
            p = jobs.band_fractions(400.0, 1, 1, 4)
        self.assertEqual(c.band_fractions["H"], p)
        st = jobs._species_tier(TIER, 1)
        pilot = [max(1, round(pk * st.histories)) for pk in p]
        by_round = [[h for s, _, h in calls[i * 4:(i + 1) * 4]] for i in range(5)]
        self.assertEqual(by_round[0], pilot)
        self.assertEqual(by_round[1], pilot)
        self.assertEqual(by_round[2], by_round[3])
        self.assertEqual(by_round[3], by_round[4])
        self.assertEqual(c.band_histories["H"], by_round[4])
        self.assertTrue(all(h >= 1 for h in by_round[4]))

    def test_cancel_trims_partial_round(self):
        state = {"n": 0}

        def cancel():
            state["n"] += 1
            return state["n"] > 6          # cancels mid round 2
        c, calls = self._run(max_batches=4, cancel_cb=cancel)
        self.assertEqual(c.species_results[0][1].n_batches, 1)

    def test_env_default_and_bad_mode(self):
        with mock.patch.dict(os.environ, {jobs.FLOOD_SAMPLING_ENV: ""}):
            self.assertEqual(jobs.default_flood_sampling(), "continuous")
        with mock.patch.dict(os.environ, {jobs.FLOOD_SAMPLING_ENV: "Stratified"}):
            self.assertEqual(jobs.default_flood_sampling(), "stratified")
        with mock.patch.dict(os.environ, {jobs.FLOOD_SAMPLING_ENV: "bogus"}):
            with self.assertRaises(ValueError):
                jobs.default_flood_sampling()
        with self.assertRaises(ValueError):
            jobs.run_composition(_spec(), TIER, sampling="bogus")


_hspec = importlib.util.spec_from_file_location(
    "stratified_flood_compare", _REPO_ROOT / "paper" / "stratified_flood_compare.py")
sfc = importlib.util.module_from_spec(_hspec)
_hspec.loader.exec_module(sfc)


class TestCompareHarness(unittest.TestCase):

    def test_designs_areal(self):
        d = sfc.designs()
        self.assertEqual(set(d), {"al19", "al30", "al54", "buried150"})
        for n, g in (("al19", 19), ("al30", 30), ("al54", 54), ("buried150", 150)):
            self.assertAlmostEqual(d[n].areal_density_gcm2(), g, places=9)

    def _row(self, d, mode, e, rel, wall, prov="git=x engine=flood"):
        return dict(design=d, areal_gcm2=19.0, sampling=mode, strata=1, tier="full",
                    rounds=12, wall_s=wall, total_primaries=1, phantom_msv=e,
                    phantom_rel=rel, skin_msv=e, skin_rel=rel / 2, phantom_mgy_d=0.3,
                    skin_mgy_d=0.3, band_histories="{}", provenance=prov, utc="t")

    def test_compare_fom_and_z(self):
        c = self._row("a", "continuous", 200.0, 0.12, 1000.0)
        s = self._row("a", "stratified", 205.0, 0.05, 1500.0)
        v = sfc.compare(c, s, 2.0)
        self.assertAlmostEqual(v["fom_gain_phantom"],
                               (0.12 ** 2 * 1000) / (0.05 ** 2 * 1500), places=12)
        self.assertAlmostEqual(v["phantom_rel_matched"], 0.05 * 1.5 ** 0.5, places=12)
        self.assertAlmostEqual(v["z_phantom"], 5 / math.hypot(24, 10.25), places=12)
        self.assertTrue(v["pass"])
        bad = sfc.compare(c, self._row("a", "stratified", 300.0, 0.05, 1500.0), 2.0)
        self.assertFalse(bad["agree"])
        weak = sfc.compare(c, self._row("a", "stratified", 200.0, 0.11, 1000.0), 2.0)
        self.assertTrue(weak["agree"])
        self.assertFalse(weak["better"])

    def test_report_pairs_latest_and_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.csv"
            for r in (self._row("a", "continuous", 200.0, 0.12, 1000.0),
                      self._row("a", "stratified", 400.0, 0.05, 1000.0),   # superseded
                      self._row("a", "stratified", 201.0, 0.05, 1000.0),
                      self._row("b", "continuous", 100.0, 0.1, 10.0)):
                sfc.append_row(p, r)
            with mock.patch("sys.stdout", new=io.StringIO()) as out:
                v = sfc.report(p, 2.0)
        self.assertEqual([x["design"] for x in v], ["a"])
        self.assertTrue(v[0]["pass"])
        self.assertIn("b          incomplete", out.getvalue())


if __name__ == "__main__":
    unittest.main()
