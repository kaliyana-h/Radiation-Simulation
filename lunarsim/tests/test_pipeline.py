"""Regression tests for the lunarsim pipeline -- the software invariants, not the
physics. These pin the parts that a silent edit could break without TOPAS ever
being run: the crew-skin dose fold, the per-shape scorer/geometry emission, the
lining volume formulas, and the CSV parsing.

Zero-dependency by design (stdlib unittest only), so it runs on any workshop
machine with no install step:

    cd ~/topas && python3 -m unittest discover -s lunarsim/tests -v
    # or, from anywhere:
    PYTHONPATH=~/topas python3 -m unittest lunarsim.tests.test_pipeline -v

No TOPAS, no G4 data, no network -- pure Python, sub-second. Physics validation
(cross-shape A/B, depth-dose sweeps, external anchoring) lives elsewhere; this
file only asserts the interface computes what it claims.
"""

import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

# Make `lunarsim` importable when run as a bare file from the tests dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim.spec import HabitatSpec, WallLayer, MATERIALS
from lunarsim import geometry, bridge


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _cyl_spec(inner=300.0):
    return HabitatSpec(name="c", shape="cylinder", inner_radius_cm=inner,
                       walls=[WallLayer("polyethylene", 10.0),
                              WallLayer("aluminium", 5.0),
                              WallLayer("regolith", 70.0)])


def _quon_spec(inner=300.0):
    return HabitatSpec(name="q", shape="quonset", inner_radius_cm=inner,
                       walls=[WallLayer("polyethylene", 10.0),
                              WallLayer("aluminium", 5.0),
                              WallLayer("regolith", 70.0)])


def _dome_spec(inner=300.0):
    return HabitatSpec(name="d", shape="dome", inner_radius_cm=inner,
                       walls=[WallLayer("polyethylene", 10.0),
                              WallLayer("regolith", 70.0)])


def _ge_param(text, name, param):
    """Pull a numeric d:Ge/<name>/<param> value out of emitted geometry text."""
    m = re.search(rf'Ge/{name}/{param}\s*=\s*(-?[\d.]+)', text)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------
# spec-level figures of merit
# --------------------------------------------------------------------------
class TestSpec(unittest.TestCase):
    def test_areal_density_is_sum_rho_t(self):
        spec = _cyl_spec()
        expect = (MATERIALS["polyethylene"]["density"] * 10.0
                  + MATERIALS["aluminium"]["density"] * 5.0
                  + MATERIALS["regolith"]["density"] * 70.0)
        self.assertAlmostEqual(spec.areal_density_gcm2(), expect, places=9)

    def test_outer_radius_is_inner_plus_wall(self):
        spec = _cyl_spec()
        self.assertAlmostEqual(spec.outer_radius_cm,
                               spec.inner_radius_cm + spec.total_wall_cm, places=9)

    def test_layer_radii_are_contiguous(self):
        spec = _cyl_spec()
        radii = spec.layer_radii_cm()
        self.assertAlmostEqual(radii[0][0], spec.inner_radius_cm, places=9)
        for (a_lo, a_hi), (b_lo, b_hi) in zip(radii, radii[1:]):
            self.assertAlmostEqual(a_hi, b_lo, places=9)  # no gap / overlap
        self.assertAlmostEqual(radii[-1][1], spec.outer_radius_cm, places=9)


# --------------------------------------------------------------------------
# scorer emission is shape-specific
# --------------------------------------------------------------------------
class TestScorerEmission(unittest.TestCase):
    def test_base_scorers_always_present(self):
        for spec in (_dome_spec(), _cyl_spec(), _quon_spec()):
            sc = geometry.build_scorers(spec)
            for tok in ("SkinDose", "SkinDoseEq", "PhantomDose", "PhantomDoseEq",
                        "InsideWallFluence", "OutsideWallFluence"):
                self.assertIn(tok, sc, f"{tok} missing for {spec.shape}")

    def test_cylinder_gets_roof_scorers_only(self):
        sc = geometry.build_scorers(_cyl_spec())
        self.assertIn('Sc/RoofDose/Component  = "CrewRoof"', sc)
        self.assertIn('Sc/RoofDoseEq/Component  = "CrewRoof"', sc)
        self.assertNotIn("CapADose", sc)
        self.assertNotIn("CapBDose", sc)

    def test_quonset_gets_cap_scorers_only(self):
        sc = geometry.build_scorers(_quon_spec())
        self.assertIn('Sc/CapADose/Component  = "CrewCapA"', sc)
        self.assertIn('Sc/CapBDose/Component  = "CrewCapB"', sc)
        self.assertIn('Sc/CapADoseEq/Component  = "CrewCapA"', sc)
        self.assertIn('Sc/CapBDoseEq/Component  = "CrewCapB"', sc)
        self.assertNotIn("RoofDose", sc)

    def test_dome_gets_no_secondary_scorers(self):
        sc = geometry.build_scorers(_dome_spec())
        for tok in ("RoofDose", "CapADose", "CapBDose"):
            self.assertNotIn(tok, sc)

    def test_ion_filter_applied_when_requested(self):
        base = geometry.build_scorers(_cyl_spec(), ion_z=0, ion_a=0)
        self.assertNotIn("OnlyIncludeParticlesOfAtomicNumber", base)
        z = geometry.build_scorers(_cyl_spec(), ion_z=26, ion_a=0)
        self.assertIn("OnlyIncludeParticlesOfAtomicNumber = 26", z)
        self.assertNotIn("OnlyIncludeParticlesOfAtomicMass", z)
        za = geometry.build_scorers(_cyl_spec(), ion_z=26, ion_a=56)
        self.assertIn("OnlyIncludeParticlesOfAtomicNumber = 26", za)
        self.assertIn("OnlyIncludeParticlesOfAtomicMass = 56", za)


# --------------------------------------------------------------------------
# geometry emission: components present + non-overlap invariants
# --------------------------------------------------------------------------
class TestGeometryEmission(unittest.TestCase):
    def test_cylinder_has_roof_lining_and_side_lining(self):
        geo = geometry.build_geometry(_cyl_spec())
        self.assertIn("CrewRoof", geo)
        self.assertIn("CrewSkin", geo)

    def test_cylinder_roof_disc_clears_inner_fluence_shell(self):
        # The regression that bit us: the roof disc must NOT overlap InnerShell.
        # CrewRoof RMax (inner-6) must sit strictly inside InnerShell RMin (inner-5).
        geo = geometry.build_geometry(_cyl_spec())
        roof_rmax = _ge_param(geo, "CrewRoof", "RMax")
        inner_shell_rmin = _ge_param(geo, "InnerShell", "RMin")
        self.assertIsNotNone(roof_rmax)
        self.assertLess(roof_rmax, inner_shell_rmin)

    def test_cylinder_side_lining_butts_against_wall(self):
        geo = geometry.build_geometry(_cyl_spec())
        skin_rmax = _ge_param(geo, "CrewSkin", "RMax")
        wall0_rmin = _ge_param(geo, "Wall0", "RMin")
        self.assertAlmostEqual(skin_rmax, wall0_rmin, places=6)  # lining meets wall

    def test_quonset_has_both_end_cap_linings_and_bulkheads(self):
        geo = geometry.build_geometry(_quon_spec())
        for name in ("CrewCapA", "CrewCapB", "CapPos0", "CapNeg0", "CrewSkin"):
            self.assertIn(name, geo, f"{name} missing from quonset geometry")

    def test_quonset_cap_linings_on_opposite_ends(self):
        geo = geometry.build_geometry(_quon_spec())
        ya = _ge_param(geo, "CrewCapA", "TransY")
        yb = _ge_param(geo, "CrewCapB", "TransY")
        self.assertIsNotNone(ya)
        self.assertIsNotNone(yb)
        self.assertAlmostEqual(ya, -yb, places=6)   # symmetric about mid-tunnel
        self.assertGreater(abs(ya), 0.0)

    def test_dome_has_no_roof_or_cap_components(self):
        geo = geometry.build_geometry(_dome_spec())
        for name in ("CrewRoof", "CrewCapA", "CrewCapB"):
            self.assertNotIn(name, geo)

    def test_outer_gauge_clears_every_solid(self):
        # The standardised hemispherical OuterShell gauge must ENCLOSE the whole
        # habitat: its RMin has to exceed the farthest corner of every emitted wall
        # / cap solid. A gauge that slices a solid is a fatal TOPAS geometry overlap
        # (this exact bug shipped once: the cap top stacks OUTWARD above the barrel
        # by the full wall thickness, and a barrel-rim-sized gauge cut through Cap2).
        # Corner of a sphere shell = RMax; of a cylinder/cap = hypot(RMax, |off|+HL)
        # where off is TransZ (cylinder) or TransY (quonset, length axis after RotX).
        for spec in (_dome_spec(), _cyl_spec(), _quon_spec()):
            geo = geometry.build_geometry(spec)
            rmin = _ge_param(geo, "OuterShell", "RMin")
            self.assertIsNotNone(rmin, f"{spec.shape}: no OuterShell gauge emitted")
            names = re.findall(r'Ge/(\w+)/Type', geo)
            for name in names:
                if name in ("OuterShell", "InnerShell", "World"):
                    continue
                rmax = _ge_param(geo, name, "RMax")
                if rmax is None:
                    continue
                hl = _ge_param(geo, name, "HL")
                if hl is None:                       # sphere shell (wall / phantom)
                    corner = rmax
                else:                                # cylinder barrel or flat cap
                    off = abs(_ge_param(geo, name, "TransZ") or 0.0) \
                        + abs(_ge_param(geo, name, "TransY") or 0.0)
                    corner = math.hypot(rmax, off + hl)
                self.assertGreater(
                    rmin, corner,
                    f"{spec.shape}: gauge RMin={rmin:.1f} does not clear "
                    f"{name} corner={corner:.1f} (overlap)")


# --------------------------------------------------------------------------
# source-dome sizing: BeamRadius must always clear the outer fluence gauge
# --------------------------------------------------------------------------
class TestBeamRadiusSizing(unittest.TestCase):
    """The GCR sources are parallel beams fired inward from distance BeamRadius;
    a gauge point is only illuminated when BeamRadius exceeds its along-axis
    projection. If BeamRadius <= gauge radius rg the top of the gauge falls above
    the source plane, fluence_outside collapses, and the dose inflates (the tall/
    thick-cylinder corner). beam_radius_for must keep BeamRadius > rg everywhere."""

    def test_validated_envelope_stays_at_default(self):
        # every design inside the calibrated envelope keeps the anchored 1400 cm
        # dome, so the validated dose numbers are unchanged.
        for spec in (_dome_spec(750.0), _cyl_spec(750.0), _quon_spec(750.0),
                     _dome_spec(300.0)):
            self.assertEqual(bridge.beam_radius_for(spec),
                             bridge.DEFAULT_BEAM_RADIUS_CM,
                             f"{spec.shape} r={spec.inner_radius_cm} moved off 1400")

    def test_beam_radius_clears_gauge_across_full_gui_envelope(self):
        # sweep the whole reachable GUI envelope (inner 1.5-8 m, walls to 3 m) for
        # all three shapes: BeamRadius must strictly exceed the outer gauge radius.
        for shape in ("dome", "cylinder", "quonset"):
            for inner in (150.0, 300.0, 500.0, 750.0, 800.0):
                for wt in (10.0, 85.0, 150.0, 250.0, 300.0):
                    spec = HabitatSpec(name="s", shape=shape, inner_radius_cm=inner,
                                       walls=[WallLayer("regolith", wt)])
                    rg = geometry._outer_gauge_radius(spec)
                    br = bridge.beam_radius_for(spec)
                    self.assertGreater(
                        br, rg,
                        f"{shape} inner={inner} wall={wt}: BeamRadius {br:.0f} "
                        f"does not clear gauge rg={rg:.0f} (fluence collapse)")

    def test_oversized_cylinder_grows_beam_radius(self):
        # the corner the audit flagged: a large thick cylinder whose gauge exceeds
        # the default 1400 must push BeamRadius out past it, not stay pinned at 1400.
        spec = HabitatSpec(name="big", shape="cylinder", inner_radius_cm=800.0,
                           walls=[WallLayer("regolith", 250.0)])
        rg = geometry._outer_gauge_radius(spec)
        br = bridge.beam_radius_for(spec)
        self.assertGreater(rg, bridge.DEFAULT_BEAM_RADIUS_CM)   # precondition: it breaks
        self.assertGreater(br, rg)                              # and the fix clears it

    def test_world_contains_the_grown_source_dome(self):
        # world must still enclose the enlarged source ring for oversized designs.
        spec = HabitatSpec(name="big", shape="cylinder", inner_radius_cm=800.0,
                           walls=[WallLayer("regolith", 300.0)])
        br = bridge.beam_radius_for(spec)
        half = bridge._world_half_cm(spec, br, bridge.DEFAULT_BEAM_SPOT_CM)
        source_reach = math.hypot(br, bridge.DEFAULT_BEAM_SPOT_CM)
        self.assertGreater(half, source_reach)


# --------------------------------------------------------------------------
# lining volume formulas (drive the mass weighting)
# --------------------------------------------------------------------------
class TestVolumes(unittest.TestCase):
    def test_cylinder_volumes_match_closed_form(self):
        spec = _cyl_spec()
        v_wall, v_roof = bridge._crewskin_volumes_cm3(spec)
        inner, H = spec.inner_radius_cm, spec.effective_height_cm
        self.assertAlmostEqual(
            v_wall, math.pi * (inner**2 - (inner - 2.0)**2) * H, places=3)
        self.assertAlmostEqual(
            v_roof, math.pi * (inner - 6.0)**2 * 2.0, places=3)

    def test_quonset_volumes_match_closed_form(self):
        spec = _quon_spec()
        v_arch, v_cap = bridge._quonset_skin_volumes_cm3(spec)
        inner, length = spec.inner_radius_cm, spec.effective_height_cm
        self.assertAlmostEqual(
            v_arch, 0.5 * math.pi * (inner**2 - (inner - 2.0)**2) * length, places=3)
        self.assertAlmostEqual(
            v_cap, 0.5 * math.pi * (inner - 6.0)**2 * 2.0, places=3)


# --------------------------------------------------------------------------
# the crew-skin dose fold -- the headline math
# --------------------------------------------------------------------------
class TestFold(unittest.TestCase):
    SECONDARY_KEYS = ("roof_dose_gy", "roof_doseeq_sv",
                      "capa_dose_gy", "capa_doseeq_sv",
                      "capb_dose_gy", "capb_doseeq_sv")

    def _base(self, skin_d, skin_de):
        d = {"skin_dose_gy": skin_d, "skin_doseeq_sv": skin_de,
             "dose_gy": 1.0, "phantom_doseeq_sv": 1.0,
             "fluence_inside": 1.0, "fluence_outside": 1.0}
        for k in self.SECONDARY_KEYS:
            d[k] = None
        return d

    def test_cylinder_fold_is_mass_weighted_mean(self):
        spec = _cyl_spec()
        v_wall, v_roof = bridge._crewskin_volumes_cm3(spec)
        res = self._base(5.0e-13, 1.0e-12)
        res["roof_dose_gy"] = 6.0e-13
        res["roof_doseeq_sv"] = 1.3e-12
        bridge._fold_secondary_into_skin(spec, res)
        self.assertAlmostEqual(
            res["skin_dose_gy"],
            (5.0e-13 * v_wall + 6.0e-13 * v_roof) / (v_wall + v_roof), places=25)
        self.assertAlmostEqual(
            res["skin_doseeq_sv"],
            (1.0e-12 * v_wall + 1.3e-12 * v_roof) / (v_wall + v_roof), places=24)

    def test_quonset_folds_both_caps(self):
        spec = _quon_spec()
        v_arch, v_cap = bridge._quonset_skin_volumes_cm3(spec)
        res = self._base(5.0e-13, 1.0e-12)
        res["capa_dose_gy"], res["capa_doseeq_sv"] = 6.0e-13, 1.30e-12
        res["capb_dose_gy"], res["capb_doseeq_sv"] = 6.2e-13, 1.35e-12
        bridge._fold_secondary_into_skin(spec, res)
        exp = (5.0e-13 * v_arch + 6.0e-13 * v_cap + 6.2e-13 * v_cap) / (v_arch + 2 * v_cap)
        self.assertAlmostEqual(res["skin_dose_gy"], exp, places=25)

    def test_dome_fold_is_noop(self):
        spec = _dome_spec()
        res = self._base(5.0e-13, 1.0e-12)
        bridge._fold_secondary_into_skin(spec, res)
        self.assertEqual(res["skin_dose_gy"], 5.0e-13)
        self.assertEqual(res["skin_doseeq_sv"], 1.0e-12)

    def test_secondary_keys_always_popped(self):
        # RunResult(**results) must never see stray roof_/cap_ kwargs, for any shape.
        for spec in (_dome_spec(), _cyl_spec(), _quon_spec()):
            res = self._base(5.0e-13, 1.0e-12)
            if spec.shape == "cylinder":
                res["roof_dose_gy"] = res["roof_doseeq_sv"] = 6.0e-13
            elif spec.shape == "quonset":
                res["capa_dose_gy"] = res["capb_dose_gy"] = 6.0e-13
            bridge._fold_secondary_into_skin(spec, res)
            for k in self.SECONDARY_KEYS:
                self.assertNotIn(k, res, f"{k} leaked for {spec.shape}")

    def test_missing_secondary_csv_drops_out_per_quantity(self):
        # If a secondary produced a dose CSV but not a doseeq CSV (or vice versa),
        # the present quantity still folds and the missing one is untouched -- one
        # missing file must not corrupt the other quantity.
        spec = _cyl_spec()
        v_wall, v_roof = bridge._crewskin_volumes_cm3(spec)
        res = self._base(5.0e-13, 1.0e-12)
        res["roof_dose_gy"] = 6.0e-13     # dose present
        res["roof_doseeq_sv"] = None      # doseeq missing
        bridge._fold_secondary_into_skin(spec, res)
        self.assertAlmostEqual(
            res["skin_dose_gy"],
            (5.0e-13 * v_wall + 6.0e-13 * v_roof) / (v_wall + v_roof), places=25)
        self.assertEqual(res["skin_doseeq_sv"], 1.0e-12)  # unchanged

    def test_fold_is_bounded_by_component_values(self):
        # A mass-weighted mean must lie between the min and max of its inputs.
        spec = _cyl_spec()
        res = self._base(4.0e-13, 8.0e-13)
        res["roof_dose_gy"] = 9.0e-13
        res["roof_doseeq_sv"] = 2.0e-12
        bridge._fold_secondary_into_skin(spec, res)
        self.assertGreaterEqual(res["skin_dose_gy"], 4.0e-13)
        self.assertLessEqual(res["skin_dose_gy"], 9.0e-13)


# --------------------------------------------------------------------------
# CSV scalar parsing
# --------------------------------------------------------------------------
class TestCsvParsing(unittest.TestCase):
    def test_reads_last_field_and_skips_comments(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("# TOPAS scorer header\n"
                         "# Sum : DoseToMedium ( Gy )\n"
                         "0, 0, 0, 5.25e-13\n")
            self.assertAlmostEqual(bridge._read_scalar_csv(p), 5.25e-13, places=25)

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(bridge._read_scalar_csv(Path(d) / "nope.csv"))

    def test_parse_results_maps_csvs_and_missing_to_none(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "skin_dose.csv").write_text("# h\n0, 3.0e-13\n")
            (run / "roof_dose.csv").write_text("# h\n0, 4.0e-13\n")
            out = bridge.parse_results(run)
            self.assertAlmostEqual(out["skin_dose_gy"], 3.0e-13, places=25)
            self.assertAlmostEqual(out["roof_dose_gy"], 4.0e-13, places=25)
            self.assertIsNone(out["skin_doseeq_sv"])   # file absent
            self.assertIsNone(out["capa_dose_gy"])     # file absent


class _FakeRun:
    """Minimal RunResult stand-in: the kernel fold reads only `.spec`."""
    def __init__(self, spec):
        self.spec = spec


class TestSpeKernelFold(unittest.TestCase):
    """The acute SPE dose now comes from folding the event spectrum against the
    shielded proton response kernel, not a rare-tail-starved direct MC. Pins the
    wiring and the validated behind-shield numbers so a silent edit can't regress
    them (physics validation of the kernel itself lives in the offline build)."""

    # the exact shielded 7.5 m dome the kernel R(E) was calibrated on
    def _cal_dome(self):
        return HabitatSpec(name="d", shape="dome", inner_radius_cm=750.0,
                           walls=[WallLayer("polyethylene", 10.0),
                                  WallLayer("aluminium", 5.0),
                                  WallLayer("regolith", 70.0)])

    def test_worst_case_reproduces_validated_behind_shield_dose(self):
        from lunarsim import dosimetry
        from lunarsim.bridge import WORST_CASE_SPE
        a = dosimetry.assess_spe(_FakeRun(self._cal_dome()), WORST_CASE_SPE,
                                 skin=False)
        self.assertEqual(a.method, "kernel-fold")
        self.assertTrue(a.calibrated)
        # feb1956 behind 147 g/cm^2: BFO ~4.5 mSv, skin ~8.2 mSv (offline fold)
        self.assertAlmostEqual(a.event_msv, 4.5, delta=1.0)   # headline = BFO
        self.assertAlmostEqual(a.skin_msv, 8.2, delta=1.5)
        self.assertGreater(a.skin_msv, a.event_msv)           # skin > BFO (depth)
        self.assertLess(a.fraction_of("nasa_30day"), 0.1)     # far under the limit

    def test_hard_event_dominates_behind_shield(self):
        # the shielding-dependent fork: the HARD event (feb1956) doses the crew far
        # more behind thick regolith than the SOFT high-fluence one (aug1972),
        # despite aug1972's ~5x larger total fluence -- soft protons are stopped.
        from lunarsim import dosimetry
        from lunarsim.bridge import WORST_CASE_SPE, SOFT_SPE
        r = _FakeRun(self._cal_dome())
        hard = dosimetry.assess_spe(r, WORST_CASE_SPE, skin=False)
        soft = dosimetry.assess_spe(r, SOFT_SPE, skin=False)
        self.assertGreater(hard.event_msv, 10 * soft.event_msv)

    def test_off_calibration_wall_is_flagged(self):
        from lunarsim import dosimetry
        from lunarsim.bridge import WORST_CASE_SPE
        thin = HabitatSpec(name="t", shape="dome", inner_radius_cm=750.0,
                           walls=[WallLayer("aluminium", 5.0)])   # ~13.5 g/cm^2
        a = dosimetry.assess_spe(_FakeRun(thin), WORST_CASE_SPE, skin=False)
        self.assertFalse(a.calibrated)   # kernel not valid off its shielded regime


if __name__ == "__main__":
    unittest.main(verbosity=2)
