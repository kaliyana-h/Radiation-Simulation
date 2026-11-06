"""Tests for the weighted phantom-biased GCR source (WeightedDiscBeam).

No TOPAS. The C++ generator's sampling + weight rule is mirrored in pure Python
and checked to be UNBIASED (weighted estimates reproduce the plain uniform disc),
the make_source emitter/validation and the bridge/jobs pass-through are checked
on their real code paths with the subprocess mocked.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_weighted_source -v
"""
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
                    "BiasRadius     = 60 cm", "BiasFraction   = 0.5",
                    "NumberOfHistoriesInRun   = 7"):
            self.assertEqual(b.count(key), 8, key)

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
