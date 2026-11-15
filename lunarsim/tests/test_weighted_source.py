"""Tests for the weighted phantom-biased GCR source (WeightedDiscBeam).

No TOPAS. The C++ generator's sampling + weight rule is mirrored in pure Python
and checked to be UNBIASED (weighted estimates reproduce the plain uniform disc),
the make_source emitter/validation and the bridge/jobs pass-through are checked
on their real code paths with the subprocess mocked.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_weighted_source -v
"""
import contextlib
import io
import math
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim import HabitatSpec, WallLayer, bridge, dosimetry, jobs  # noqa: E402
from lunarsim.bridge import RunTier, SourceBias, SPEScenario  # noqa: E402

TIER = RunTier("t", rings=2, azimuth=4, histories=40)
R = 900.0                     # disc radius (So/BeamSpot)


def _ms():
    return dosimetry._load_make_source()


def _spec():
    return HabitatSpec(name="wdb", shape="dome", inner_radius_cm=300.0,
                       walls=[WallLayer("aluminium", 20.0)])


def _sample(rng, cx, cy, rb, f):
    """Python mirror of WeightedDiscBeam::GeneratePrimaries -> (x, y, w)."""
    w_in, w_out = _ms().bias_weights(R, rb, f)
    if f > 0 and rng.random() < f:
        r, a = rb * math.sqrt(rng.random()), 2 * math.pi * rng.random()
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
    else:
        r, a = R * math.sqrt(rng.random()), 2 * math.pi * rng.random()
        x, y = r * math.cos(a), r * math.sin(a)
    inside = f > 0 and (x - cx) ** 2 + (y - cy) ** 2 < rb * rb
    return x, y, (w_in if inside else w_out)


def _shadow(theta_deg, phi_deg, target_z):
    """Target (0,0,z) projected onto a source disc along its ray, in the disc's
    own (x, y): the ray direction is -n (n = unit vector to the disc centre), so
    the perpendicular offset of the target from the disc axis is
    z * (z_hat - (z_hat.n) n); its length is z sin(theta)."""
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    n = (math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t))
    v = (-target_z * n[2] * n[0], -target_z * n[2] * n[1], target_z * (1 - n[2] ** 2))
    return math.sqrt(sum(c * c for c in v))


class WeightRule(unittest.TestCase):
    def test_weights_match_formula_and_mean_one(self):
        w_in, w_out = _ms().bias_weights(R, 60.0, 0.5)
        self.assertAlmostEqual(w_out, 2.0)
        self.assertAlmostEqual(w_in, 1.0 / (0.5 + 0.5 * (900 / 60) ** 2))
        # E_q[w] = 1 exactly: q-mass of circle * w_in + q-mass outside * w_out
        a = (60.0 / R) ** 2
        q_in = 0.5 * a + 0.5
        q_out = 0.5 * (1 - a)
        self.assertAlmostEqual(q_in * w_in + q_out * w_out, 1.0, places=12)

    def test_zero_fraction_is_plain_disc(self):
        self.assertEqual(_ms().bias_weights(R, 60.0, 0.0), (1.0, 1.0))

    def test_weighted_estimates_are_unbiased(self):
        """Weighted MC of (a) the phantom hit fraction, (b) an off-target annulus
        and (c) total weight reproduces the uniform-disc truth."""
        rng = random.Random(7)
        cx, cy, rb, f, rp = 70.0, -20.0, 60.0, 0.5, 20.0
        n = 200_000
        hit = ann = tot = 0.0
        hit2 = 0.0
        for _ in range(n):
            x, y, w = _sample(rng, cx, cy, rb, f)
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            tot += w
            if d2 < rp * rp:
                hit += w
                hit2 += w * w
            if 300.0 ** 2 < x * x + y * y < 500.0 ** 2:
                ann += w
        p_hit = (rp / R) ** 2
        p_ann = (500.0 ** 2 - 300.0 ** 2) / R ** 2
        se_hit = math.sqrt(max(hit2 / n - (hit / n) ** 2, 0) / n)
        self.assertLess(abs(hit / n - p_hit), 4 * se_hit)
        self.assertLess(abs(ann / n - p_ann), 4 * math.sqrt(p_ann * 4 / n))
        self.assertLess(abs(tot / n - 1.0), 0.01)

    def test_variance_gain_on_phantom(self):
        """The point of the exercise: per-history variance of the phantom-hit
        tally drops by ~1/(f*(R/rb)^2)-ish vs the uniform disc (>=50x here)."""
        rp, rb, f = 20.0, 60.0, 0.5
        p = (rp / R) ** 2
        var_uniform = p * (1 - p)
        w_in, _ = _ms().bias_weights(R, rb, f)
        q_hit = f * (rp / rb) ** 2 + (1 - f) * p     # q-prob of starting on target
        var_biased = q_hit * w_in ** 2 - p ** 2
        self.assertGreater(var_uniform / var_biased, 50.0)


class ShadowFits(unittest.TestCase):
    def test_offset_bounded_by_target_height(self):
        for th in (0.0, 18.4, 39.2, 50.8, 63.4, 71.6, 89.0):
            for ph in (0.0, 45.0, 200.0):
                off = _shadow(th, ph, 100.0)
                self.assertAlmostEqual(off, 100.0 * math.sin(math.radians(th)), places=9)
                self.assertLessEqual(off, 100.0 + 1e-9)


class MakeSourceEmit(unittest.TestCase):
    def setUp(self):
        self.ms = _ms()
        self.kw = dict(rings=2, azimuth=4, phi_MV=400.0, histories=7)

    def test_default_unchanged(self):
        a = self.ms.gcr_block(**self.kw)
        self.assertEqual(a, self.ms.gcr_block(**self.kw, bias=None))
        self.assertNotIn("WeightedDiscBeam", a)
        self.assertEqual(a.count('Type      = "Beam"'), 8)

    def test_biased_block(self):
        b = self.ms.gcr_block(**self.kw, bias=(100.0, 60.0, 0.5))
        self.assertEqual(b.count('Type      = "WeightedDiscBeam"'), 8)
        self.assertNotIn('"Beam"', b)
        self.assertNotIn("BeamPositionDistribution", b)
        for key in ("DiscRadius     = So/BeamSpot cm", "BiasTargetZ    = 100 cm",
                    "BiasRadius     = 60 cm", "BiasFraction   = 0.5"):
            self.assertEqual(b.count(key), 8, key)
        # histories are shared by ring solid angle (default since 2026-11-15):
        # per-source counts vary, the total (8 sources x 7) is preserved
        n = [int(ln.split("=")[1]) for ln in b.splitlines()
             if "NumberOfHistoriesInRun" in ln]
        self.assertEqual((len(n), sum(n)), (8, 56))

    def test_spectrum_components_identical(self):
        """Energy spectrum, particle and the group chain are byte-identical: only
        the start-point sampling differs."""
        a = self.ms.gcr_block(**self.kw)
        b = self.ms.gcr_block(**self.kw, bias=(100.0, 60.0, 0.5))
        keep = ("BeamEnergySpectrum", "BeamParticle", "/Component", "s:Ge/", "d:Ge/")
        pick = lambda t: [ln for ln in t.splitlines() if any(k in ln for k in keep)]
        self.assertEqual(pick(a), pick(b))

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.ms.gcr_block(**self.kw, bias=(100.0, 60.0, 1.0))
        with self.assertRaises(ValueError):
            self.ms.gcr_block(**self.kw, bias=(100.0, 60.0, -0.1))
        with self.assertRaises(ValueError):
            self.ms.gcr_block(**self.kw, bias=(100.0, 0.0, 0.5))
        with self.assertRaises(ValueError):          # |z| + r_b > disc
            self.ms.gcr_block(**self.kw, bias=(850.0, 60.0, 0.5))
        self.ms.gcr_block(**self.kw, bias=(840.0, 60.0, 0.5))   # exactly fits


class BridgePassThrough(unittest.TestCase):
    def _cmd(self, **kw):
        with mock.patch.object(bridge.subprocess, "run") as run:
            bridge.generate_source(Path("/nonexistent"), TIER, **kw)
        return run.call_args[0][0]

    def test_no_bias_command_unchanged(self):
        self.assertFalse(any(a.startswith("--bias") for a in self._cmd()))

    def test_bias_flags(self):
        cmd = self._cmd(bias=(100.0, 60.0, 0.5))
        i = cmd.index("--bias-fraction")
        self.assertEqual(float(cmd[i + 1]), 0.5)
        self.assertEqual(float(cmd[cmd.index("--bias-target-z") + 1]), 100.0)
        self.assertEqual(float(cmd[cmd.index("--bias-radius") + 1]), 60.0)

    def test_spe_rejects_bias(self):
        with self.assertRaises(ValueError):
            self._cmd(spe=SPEScenario("x", "aug1972", 1e9), bias=(100.0, 60.0, 0.5))

    def test_for_spec(self):
        s = _spec()
        self.assertEqual(SourceBias().for_spec(s),
                         (s.crew_height_cm, s.phantom_radius_cm + 40.0, 0.5))

    def test_run_design_forwards_bias(self):
        s = _spec()
        with mock.patch.object(bridge, "generate_source",
                               side_effect=RuntimeError("stop")) as gs:
            with self.assertRaises(RuntimeError):
                bridge.run_design(s, TIER, source_bias=SourceBias(0.3, 25.0))
        self.assertEqual(gs.call_args.kwargs["bias"],
                         (s.crew_height_cm, s.phantom_radius_cm + 25.0, 0.3))
        with mock.patch.object(bridge, "generate_source",
                               side_effect=RuntimeError("stop")) as gs:
            with self.assertRaises(RuntimeError):
                bridge.run_design(s, TIER)
        self.assertIsNone(gs.call_args.kwargs["bias"])

    def test_jobs_forwards_and_guards(self):
        with self.assertRaises(ValueError):
            jobs.run_composition(_spec(), TIER, sampling="stratified",
                                 source_bias=SourceBias())
        seen = []

        def fake(spec, tier, **kw):
            seen.append(kw.get("source_bias"))
            raise RuntimeError("stop")
        with mock.patch.object(jobs, "run_design", side_effect=fake):
            with self.assertRaises(RuntimeError):
                jobs.run_composition(_spec(), TIER, sampling="continuous",
                                     source_bias=SourceBias())
        self.assertEqual(seen, [SourceBias()])


class _Species:
    """Minimal ConvergedResult stand-in for assess_composition."""
    def __init__(self, dose, doseeq, eq_rel):
        self.spec = _spec()
        self.dose_gy = self.skin_dose_gy = dose
        self.fluence_outside = 1.0
        self.dose_rel_err = self.skin_dose_rel_err = 0.01
        self.phantom_doseeq_sv = self.skin_doseeq_sv = doseeq
        self.phantom_doseeq_nasa_sv = self.skin_doseeq_nasa_sv = doseeq
        self.phantom_doseeq_rel_err = self.skin_doseeq_rel_err = eq_rel
        self.phantom_doseeq_nasa_rel_err = self.skin_doseeq_nasa_rel_err = eq_rel
        self.neutron_doseeq_fraction = None


class DoseEquivalentError(unittest.TestCase):
    """E gets its own batch error: the phantom's Q-bar scatter (rare high-LET
    tracks) is invisible to the absorbed-dose error."""

    def _batch(self, d, h):
        return bridge.RunResult(spec=_spec(), tier=TIER, run_dir=Path("."), returncode=0,
                                wall_seconds=1.0, dose_gy=d, phantom_doseeq_sv=h,
                                phantom_doseeq_nasa_sv=1.2 * h, skin_dose_gy=d,
                                skin_doseeq_sv=0.5 * h)

    def test_combine_gives_doseeq_its_own_error(self):
        import statistics
        hs = (2.0e-13, 3.0e-13, 2.0e-13, 5.0e-13)       # Q scatters, D steady
        c = jobs._combine(_spec(), TIER, [self._batch(1.0e-13, h) for h in hs])
        self.assertEqual(c.dose_rel_err, 0.0)
        expect = statistics.stdev(hs) / 2.0 / statistics.fmean(hs)
        for name in ("phantom_doseeq_rel_err", "phantom_doseeq_nasa_rel_err",
                     "skin_doseeq_rel_err"):
            self.assertAlmostEqual(getattr(c, name), expect, places=12, msg=name)
        self.assertIsNone(c.skin_doseeq_nasa_rel_err)    # never scored
        one = jobs._combine(_spec(), TIER, [self._batch(1.0e-13, 2.0e-13)])
        self.assertIsNone(one.phantom_doseeq_rel_err)

    def test_assess_composition_eq_error_in_quadrature(self):
        H = dosimetry.GCR_COMPOSITION[0]
        for skin in (False, True):
            runs = [(H, _Species(1.0e-13, 2.0e-13, 0.10)),
                    (H, _Species(1.0e-13, 6.0e-13, 0.04))]
            a = dosimetry.assess_composition(runs, skin=skin)
            expect = ((2 * 0.10) ** 2 + (6 * 0.04) ** 2) ** 0.5 / 8
            self.assertAlmostEqual(a.eq_rel_err, expect, places=12)
            self.assertAlmostEqual(a.rel_err, 0.01 / 2 ** 0.5, places=12)
            runs[1][1].phantom_doseeq_rel_err = runs[1][1].skin_doseeq_rel_err = None
            self.assertIsNone(dosimetry.assess_composition(runs, skin=skin).eq_rel_err)

    def test_seed_offset_forwarded_and_guarded(self):
        with self.assertRaises(ValueError):
            jobs.run_composition(_spec(), TIER, sampling="stratified", seed_offset=5)
        seeds = []

        def fake(spec, tier, **kw):
            seeds.append(kw["seed"])
            raise RuntimeError("stop")
        for off, want in ((0, 1), (1000, 1001)):
            with mock.patch.object(jobs, "run_design", side_effect=fake):
                with self.assertRaises(RuntimeError):
                    jobs.run_composition(_spec(), TIER, sampling="continuous",
                                         seed_offset=off)
            self.assertEqual(seeds[-1], want)


class FomHarness(unittest.TestCase):
    """paper/weighted_source_fom.py: z_E uses the E errors, old rows fall back."""

    def setUp(self):
        sys.path.insert(0, str(_REPO_ROOT / "paper"))
        import weighted_source_fom as w
        self.w = w

    def test_margin_label_separates_sweep(self):
        lab = self.w._label
        self.assertEqual(lab({"mode": "beam", "bias_margin_cm": ""}), "beam")
        self.assertEqual(lab({"mode": "f0.2", "bias_margin_cm": "40.0"}), "f0.2")
        self.assertEqual(lab({"mode": "f0.2", "bias_margin_cm": "120"}), "f0.2m120")

    def test_weighted_consistency_flags_moved_answer(self):
        rows = [dict(self._row("f0.2", 190, 0.05, eq=0.04), bias_margin_cm="40"),
                dict(self._row("f0.2", 192, 0.05, eq=0.04), bias_margin_cm="120"),
                dict(self._row("f0.05", 260, 0.05, eq=0.04), bias_margin_cm="40")]
        rows[2]["phantom_mgy_d"] = 0.34
        with contextlib.redirect_stdout(io.StringIO()):
            out = {v["label"]: v for v in self.w.weighted_consistency(rows)}
        self.assertTrue(out["f0.2m120"]["agree"])
        self.assertFalse(out["f0.05"]["agree"])

    def test_species_breakdown_shares_and_tail(self):
        import json as _json
        from types import SimpleNamespace as NS
        a_ph = NS(annual_msv=200.0, contributions=[
            {"species": "H", "doseeq_rate_sv_s": 3.0},
            {"species": "Fe", "doseeq_rate_sv_s": 1.0}])
        res = [(("H",), NS(per_batch_phantom_doseeq=[1, 1, 1, 1], phantom_doseeq_rel_err=0.01)),
               (("Fe",), NS(per_batch_phantom_doseeq=[0, 0, 0, 4], phantom_doseeq_rel_err=0.5))]
        out = _json.loads(self.w.species_breakdown(a_ph, res))
        self.assertAlmostEqual(out["H"]["E"], 150.0)
        self.assertAlmostEqual(out["Fe"]["E"], 50.0)
        self.assertAlmostEqual(out["H"]["max_batch"], 0.25)
        self.assertAlmostEqual(out["Fe"]["max_batch"], 1.0)

    def test_species_split_groups_unit_and_biased(self):
        js = lambda fe: '{"H":{"E":100,"eE":0.02,"max_batch":0.1},"Fe":{"E":%g,"eE":0.3,"max_batch":0.5}}' % fe
        rows = [dict(mode="beam", bias_f="", phantom_species=js(120)),
                dict(mode="f0", bias_f="0.0", phantom_species=js(100)),
                dict(mode="f0.2", bias_f="0.2", phantom_species=js(40)),
                dict(mode="f0.5", bias_f="0.5", phantom_species="")]
        with contextlib.redirect_stdout(io.StringIO()):
            out = self.w.species_split(rows)
        self.assertAlmostEqual(out["Fe"]["unit"], 110.0)
        self.assertAlmostEqual(out["Fe"]["biased"], 40.0)
        self.assertAlmostEqual(out["H"]["z"], 0.0)

    def test_species_subset_filters_flood_composition(self):
        tier = RunTier("t", 1, 1, 1)
        self.assertIsNone(self.w.species_subset(tier, None))
        sub = self.w.species_subset(tier, ["H"])
        self.assertEqual([sp[0] for sp in sub], ["H"])
        with self.assertRaises(SystemExit):
            self.w.species_subset(tier, ["Xx"])

    def _row(self, mode, ph, rel, eq="", wall=1000.0, smp=""):
        return dict(design="al19", areal_gcm2="19", mode=mode, wall_s=wall,
                    phantom_msv=ph, phantom_rel=rel, skin_msv=200, skin_rel=0.02,
                    phantom_mgy_d=0.25, skin_mgy_d=0.27, fluence_out=1, fluence_in=1,
                    provenance="p", phantom_eq_rel=eq, skin_eq_rel=eq and 0.02,
                    seed_offset=smp)

    def test_eq_errors_used_and_fallback_flagged(self):
        old = self.w.compare(self._row("beam", 260, 0.05), self._row("f0.2", 200, 0.03), 2)
        self.assertFalse(old["eq_errors"])
        self.assertFalse(old["agree"])
        new = self.w.compare(self._row("beam", 260, 0.05, 0.12),
                             self._row("f0.2", 200, 0.03, 0.05), 2)
        self.assertTrue(new["eq_errors"])
        self.assertTrue(new["agree"])
        self.assertAlmostEqual(new["fom_gain_phantom"], (0.12 / 0.05) ** 2)

    def test_pool_inverse_variance(self):
        p = self.w.pool([self._row("beam", 100, 0.1, 0.1), self._row("beam", 200, 0.1, 0.05)])
        w1, w2 = 1 / 10.0 ** 2, 1 / 10.0 ** 2
        self.assertAlmostEqual(p["phantom_msv"], 150.0)
        self.assertAlmostEqual(p["e_ph"] * p["phantom_msv"], (1 / (w1 + w2)) ** 0.5)
        self.assertEqual(p["wall_s"], 2000.0)

    def test_header_upgrade_keeps_old_rows(self):
        import csv
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f.csv"
            old = [f for f in self.w.FIELDS if f not in
                   ("phantom_eq_rel", "skin_eq_rel", "seed_offset")]
            with path.open("w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=old)
                wr.writeheader()
                wr.writerow({k: "1" for k in old})
            self.w.append_row(path, {k: "2" for k in self.w.FIELDS})
            rows = list(csv.DictReader(path.open()))
            self.assertEqual([r["seed_offset"] for r in rows], ["", "2"])
            self.assertEqual(rows[0]["wall_s"], "1")
            self.assertTrue(path.with_suffix(".csv.pre_eqerr.bak").exists())


class GeneratorSource(unittest.TestCase):
    """The C++ extension ships in the tracked extensions/ dir with the CMake
    registration line TOPAS needs."""
    def test_first_line_and_params(self):
        cc = (_REPO_ROOT / "extensions" / "WeightedDiscBeam.cc").read_text()
        self.assertEqual(cc.splitlines()[0], "// Particle Generator for WeightedDiscBeam")
        for p in ("DiscRadius", "BiasTargetZ", "BiasRadius", "BiasFraction",
                  "TransformPrimaryForComponent", "p.weight"):
            self.assertIn(p, cc)


if __name__ == "__main__":
    unittest.main()
