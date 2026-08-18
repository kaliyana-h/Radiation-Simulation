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
import types
import unittest
from pathlib import Path

# Make `lunarsim` importable when run as a bare file from the tests dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim.spec import HabitatSpec, WallLayer, MATERIALS
from lunarsim import geometry, bridge, dosimetry


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

    def test_effective_height_defaults_by_shape(self):
        # the contract the GUI length control relies on: leave height_cm unset and
        # each shape resolves its own axial-length default.
        self.assertEqual(_cyl_spec(inner=300.0).effective_height_cm, 300.0)   # = radius
        self.assertEqual(_quon_spec(inner=300.0).effective_height_cm, 900.0)  # = 3*radius
        self.assertEqual(_dome_spec(inner=300.0).effective_height_cm, 300.0)  # = radius

    def test_quonset_length_default_is_capped(self):
        # a very wide tunnel must not run its ends outside the ~9 m GCR source dome
        self.assertEqual(_quon_spec(inner=600.0).effective_height_cm, 1200.0)  # min(3*600, 1200)

    def test_explicit_height_overrides_default(self):
        spec = HabitatSpec(name="c", shape="cylinder", inner_radius_cm=300.0,
                           height_cm=800.0,
                           walls=[WallLayer("aluminium", 5.0)])
        self.assertEqual(spec.effective_height_cm, 800.0)   # honoured verbatim


# --------------------------------------------------------------------------
# size-envelope guard: oversized designs self-flag their gauge extrapolation
# --------------------------------------------------------------------------
class TestGaugeSizeFlag(unittest.TestCase):
    def test_in_envelope_design_is_ok(self):
        # a 3 m habitat fits inside the fixed 880 cm gauge -> gauge_corr == 1.0
        for spec in (_dome_spec(300.0), _cyl_spec(300.0), _quon_spec(300.0)):
            self.assertAlmostEqual(dosimetry._gauge_corr(spec), 1.0, places=9)
            self.assertEqual(dosimetry.gauge_size_flag(spec), "ok")

    def test_oversized_cylinder_flags_strong(self):
        # an 8.5 m x 6 m drum: enclosing corner ~1160 cm forces the gauge to grow
        # well past the fixed 880 cm reference (gauge_corr ~ 0.56) -> strong.
        spec = HabitatSpec(name="big", shape="cylinder", inner_radius_cm=850.0,
                           height_cm=600.0,
                           walls=[WallLayer("polyethylene", 10.0),
                                  WallLayer("aluminium", 5.0),
                                  WallLayer("regolith", 70.0)])
        self.assertLess(dosimetry._gauge_corr(spec), dosimetry.GAUGE_CORR_STRONG)
        self.assertEqual(dosimetry.gauge_size_flag(spec), "strong")

    def test_footprint_flag_grades_illumination(self):
        # under-illumination guard: keys off the enclosing radius vs the 900 cm
        # illuminated source footprint, independent of the gauge normalisation.
        from lunarsim.bridge import DEFAULT_BEAM_SPOT_CM
        # a 3 m habitat sits well inside the lit disc -> fully sampled.
        self.assertEqual(dosimetry.beam_footprint_flag(_dome_spec(300.0)), "ok")
        # a dome whose outer radius is a hair inside 900 cm -> marginal.
        near = HabitatSpec(name="near", shape="dome", inner_radius_cm=880.0,
                           walls=[WallLayer("aluminium", 1.0)])
        self.assertLessEqual(geometry._enclosing_radius_cm(near), DEFAULT_BEAM_SPOT_CM)
        self.assertEqual(dosimetry.beam_footprint_flag(near), "marginal")
        # a 9.5 m dome pokes past the footprint -> under-sampled ('over').
        big = HabitatSpec(name="big", shape="dome", inner_radius_cm=950.0,
                          walls=[WallLayer("aluminium", 5.0)])
        self.assertGreater(geometry._enclosing_radius_cm(big), DEFAULT_BEAM_SPOT_CM)
        self.assertEqual(dosimetry.beam_footprint_flag(big), "over")

    def test_thresholds_are_monotone(self):
        # ordering the display relies on: ok >= MILD > mild band >= STRONG > strong
        self.assertGreater(dosimetry.GAUGE_CORR_MILD, dosimetry.GAUGE_CORR_STRONG)
        boundary = HabitatSpec(name="edge", shape="cylinder",
                               inner_radius_cm=300.0, height_cm=300.0,
                               walls=[WallLayer("regolith", 70.0)])
        # a mid-size design lands in exactly one of the three buckets
        self.assertIn(dosimetry.gauge_size_flag(boundary), ("ok", "mild", "strong"))


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

    def test_neutron_lineage_scorer_on_crewskin_every_shape(self):
        # The secondary-neutron dose fraction is a CrewSkin twin present for all
        # shapes (dome included -- it has no secondary lining but does have CrewSkin).
        for spec in (_dome_spec(), _cyl_spec(), _quon_spec()):
            sc = geometry.build_scorers(spec)
            self.assertIn('Sc/SkinDoseEqNeutron/Quantity   = "DoseEquivalent_ICRP_Neutron"',
                          sc, f"neutron scorer missing for {spec.shape}")
            self.assertIn('Sc/SkinDoseEqNeutron/Component  = "CrewSkin"', sc)
            self.assertIn('Sc/SkinDoseEqNeutron/OutputFile = "skin_doseeq_neutron"', sc)

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

    def test_parse_results_maps_neutron_doseeq_csv(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "skin_doseeq_neutron.csv").write_text("# h\n0, 3.7e-13\n")
            out = bridge.parse_results(run)
            self.assertAlmostEqual(out["skin_doseeq_neutron_sv"], 3.7e-13, places=25)


class _FakeSpeciesRun:
    """Minimal ConvergedResult stand-in for the dosimetry dose-weighting path."""
    def __init__(self, spec, doseeq, frac, dose=1.0e-13, fout=1.0, relerr=0.05):
        self.spec = spec
        self.skin_dose_gy = self.dose_gy = dose
        self.fluence_outside = fout
        self.skin_dose_rel_err = self.dose_rel_err = relerr
        self.skin_doseeq_sv = self.skin_doseeq_nasa_sv = doseeq
        self.phantom_doseeq_sv = self.phantom_doseeq_nasa_sv = doseeq
        self.neutron_doseeq_fraction = frac


class TestNeutronFraction(unittest.TestCase):
    """The secondary-neutron dose fraction: a ratio of two same-run dose-equivalents,
    combined across species dose-equivalent-weighted. Normalisation cancels, so these
    invariants hold without recomputing any flux/gauge factors."""

    def _two_species(self, fa, fb, ea=1.0e-12, eb=1.0e-12):
        spec = _dome_spec()
        H, Fe = dosimetry.GCR_COMPOSITION[0], dosimetry.GCR_COMPOSITION[4]
        return [(H, _FakeSpeciesRun(spec, ea, fa)),
                (Fe, _FakeSpeciesRun(spec, eb, fb))]

    def test_equal_fractions_combine_to_that_fraction(self):
        # If every species carries the same neutron fraction, the weighted mean is
        # it -- independent of the (very different) H/Fe dose-eq weights.
        a = dosimetry.assess_composition(self._two_species(0.4, 0.4), skin=True)
        self.assertAlmostEqual(a.neutron_fraction, 0.4, places=12)

    def test_single_species_passes_fraction_through(self):
        spec = _dome_spec()
        sr = [(dosimetry.GCR_COMPOSITION[0], _FakeSpeciesRun(spec, 1.0e-12, 0.31))]
        a = dosimetry.assess_composition(sr, skin=True)
        self.assertAlmostEqual(a.neutron_fraction, 0.31, places=12)

    def test_species_missing_fraction_drops_out(self):
        # A species with no neutron scorer (frac=None) leaves both numerator and
        # denominator, so the combined value is the remaining species' fraction.
        a = dosimetry.assess_composition(self._two_species(0.5, None), skin=True)
        self.assertAlmostEqual(a.neutron_fraction, 0.5, places=12)

    def test_none_everywhere_yields_none(self):
        a = dosimetry.assess_composition(self._two_species(None, None), skin=True)
        self.assertIsNone(a.neutron_fraction)

    def test_dose_eq_weighting_favours_the_heavier_contributor(self):
        # Identical species (same flux) so the weight is purely the dose-equivalent:
        # entry B carries 10x A's H, so the combined fraction sits near B's 0.2, far
        # below the 0.5 midpoint of a plain average.
        spec = _dome_spec()
        H = dosimetry.GCR_COMPOSITION[0]
        sr = [(H, _FakeSpeciesRun(spec, 1.0e-13, 0.8)),
              (H, _FakeSpeciesRun(spec, 1.0e-12, 0.2))]
        a = dosimetry.assess_composition(sr, skin=True)
        expected = (0.8 * 1.0e-13 + 0.2 * 1.0e-12) / (1.0e-13 + 1.0e-12)
        self.assertAlmostEqual(a.neutron_fraction, expected, places=12)
        self.assertLess(a.neutron_fraction, 0.5)

    def test_assess_passes_neutron_fraction_through(self):
        spec = _dome_spec()
        r = _FakeSpeciesRun(spec, 1.0e-12, 0.27)
        a = dosimetry.assess(r, phi_MV=400.0, skin=True)
        self.assertAlmostEqual(a.neutron_fraction, 0.27, places=12)


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


class TestGcrThinWallFold(unittest.TestCase):
    """Below the ~19 g/cm^2 crossover the flood normalisation over-counts wall-bred
    secondaries (a 7.5 mm Al dome reads a nonphysical ~3500 mSv/yr under flood), so
    the chronic-GCR headline is served instead by the phantom-matched kernel fold.
    Pins the wiring, the regime gate, and the validated folded numbers so a silent
    edit can't regress them (physics validation of the kernel lives offline)."""

    def _thin_dome(self):
        # 7.5 mm aluminium dome ~ 2.03 g/cm^2 -- the thinnest kernel anchor
        return HabitatSpec(name="d", shape="dome", inner_radius_cm=750.0,
                           walls=[WallLayer("aluminium", 0.75)])

    def test_thin_al_dome_reproduces_validated_fold(self):
        from lunarsim import dosimetry
        spec = self._thin_dome()
        self.assertTrue(dosimetry._gcr_thinwall_applies(spec))
        self.assertTrue(dosimetry._gcr_thinwall_calibrated(spec))
        a = dosimetry.assess_gcr_thinwall(spec, mission_days=365.0, phi_MV=400.0)
        self.assertEqual(a.regime, "thinwall")
        self.assertIsNone(a.neutron_fraction)      # no neutron twin in this kernel
        # offline fold: 342.6 mSv/yr ICRP effective at the thinnest anchor
        self.assertAlmostEqual(a.annual_msv, 342.6, delta=1.0)
        # squarely in the literature 300-400 mSv/yr bracket (post-hoc check only)
        self.assertGreater(a.annual_msv, 300.0)
        self.assertLess(a.annual_msv, 400.0)

    def test_per_species_effective_dose_sums_to_headline(self):
        from lunarsim import dosimetry
        a = dosimetry.assess_gcr_thinwall(self._thin_dome(), phi_MV=400.0)
        yr = dosimetry.SECONDS_PER_DAY * dosimetry.DAYS_PER_YEAR * 1e3
        esum = sum(c["doseeq_rate_sv_s"] for c in a.contributions) * yr
        self.assertAlmostEqual(esum, a.annual_msv, delta=0.5)

    def test_dose_decreases_with_wall_thickness(self):
        from lunarsim import dosimetry
        def E(tcm):
            s = HabitatSpec(name="t", shape="dome", inner_radius_cm=750.0,
                            walls=[WallLayer("aluminium", tcm)])
            return dosimetry.assess_gcr_thinwall(s, phi_MV=400.0).annual_msv
        # monotone falling across the thin band (0.75 -> ~3.7 -> ~7 cm Al)
        self.assertGreater(E(0.75), E(3.7))
        self.assertGreater(E(3.7), E(7.0))

    def test_near_gate_meets_flood_path_continuously(self):
        # log-log interpolation must land on the flood-validated ~130 mSv/yr as the
        # areal density approaches the ~19 g/cm^2 crossover, not leave a step there.
        from lunarsim import dosimetry
        s = HabitatSpec(name="g", shape="dome", inner_radius_cm=750.0,
                        walls=[WallLayer("aluminium", 7.0)])   # ~18.9 g/cm^2
        a = dosimetry.assess_gcr_thinwall(s, phi_MV=400.0)
        self.assertAlmostEqual(a.annual_msv, 130.0, delta=10.0)

    def test_thick_design_stays_on_flood_path(self):
        from lunarsim import dosimetry
        thick = HabitatSpec(name="std", shape="dome", inner_radius_cm=750.0,
                            walls=[WallLayer("polyethylene", 10.0),
                                   WallLayer("regolith", 70.0)])   # ~134 g/cm^2
        self.assertFalse(dosimetry._gcr_thinwall_applies(thick))

    def test_non_aluminium_thin_wall_is_flagged_indicative(self):
        from lunarsim import dosimetry
        poly = HabitatSpec(name="p", shape="dome", inner_radius_cm=750.0,
                           walls=[WallLayer("polyethylene", 1.0)])
        self.assertTrue(dosimetry._gcr_thinwall_applies(poly))
        self.assertFalse(dosimetry._gcr_thinwall_calibrated(poly))  # not Al-dominated


class _FakeComposition:
    """A run_composition stand-in result: carries just enough for the worker's
    post-run handling (`ok`, the rel-err fields) and the display fold (`spec`)."""
    def __init__(self, spec):
        self.spec = spec
        self.ok = True
        self.dose_rel_err = 0.05
        self.skin_dose_rel_err = 0.05


class TestCombinedThinWallRender(unittest.TestCase):
    """The _poll callback must serve the thin-wall Gate 1 fold for a COMBINED
    ('SPE+GCR') job below the crossover too -- the combined branch returns before
    the pure-GCR thin-wall gate, so a 7.5 mm Al dome was rendering the nonphysical
    ~3500 mSv/yr flood number under NASA Q. Drives the real callback."""

    class _FakeResult:
        """Spec-only fold stand-in that also carries the report statistics the
        combined branch stamps into the markdown."""
        def __init__(self, spec):
            self.spec = spec
            self.ok = True
            self.n_batches, self.total_primaries, self.wall_seconds = 12, 24000, 100.0

    def _job(self, spec):
        from lunarsim.jobs import JobStatus
        from lunarsim.bridge import WORST_CASE_SPE
        return types.SimpleNamespace(
            spec=spec, result=self._FakeResult(spec), spe=WORST_CASE_SPE,
            combined=True, status=JobStatus.DONE, progress=1.0, rel_err=0.1,
            progress_cap=12, max_batches=12, batches_done=12, elapsed=100.0)

    def _poll_with(self, spec):
        from lunarsim import gui
        job = self._job(spec)
        orig_get = gui.default_runner.get
        gui.default_runner.get = lambda jid: job
        try:
            return gui._poll(1, "fake")
        finally:
            gui.default_runner.get = orig_get

    def test_combined_thin_al_dome_serves_thinwall_gate1(self):
        spec = HabitatSpec(name="d", shape="dome", inner_radius_cm=750.0,
                           walls=[WallLayer("aluminium", 0.75)])   # ~2.03 g/cm^2
        _bar, _st, _metrics, _an, _dis, report, overlay = self._poll_with(spec)
        # Gate 1 overlay is the folded ICRP effective dose, NOT the flood artifact
        self.assertAlmostEqual(overlay["skin"], 342.6, delta=1.0)
        self.assertLess(overlay["skin"], 400.0)          # not the ~3500 flood number
        self.assertIn("thin-wall", report)               # regime is labelled
        self.assertIn("ICRP-60 Q(L)", report)            # folded on ICRP Q, not NASA
        self.assertIn("Gate 2", report)                  # SPE gate still present

    def test_report_carries_confidence_band(self):
        """The downloadable report labels the GCR headline with the advisory
        confidence band. A 2 g/cm^2 dome is the thin-wall (optimistic) regime.
        (The band is report-only; it is deliberately NOT shown on the GUI.)"""
        spec = HabitatSpec(name="d", shape="dome", inner_radius_cm=750.0,
                           walls=[WallLayer("aluminium", 0.75)])   # ~2.03 g/cm^2
        _bar, _st, _metrics, _an, _dis, report, _ov = self._poll_with(spec)
        self.assertIn("Confidence (thin-wall (optimistic))", report)


class TestCombinedJobOrchestration(unittest.TestCase):
    """A combined ('SPE+GCR') job must transport exactly ONE MC -- the GCR
    composition. The acute-SPE gate is a response-kernel fold with no MC of its
    own, so the SPE MC must never be launched (it was rare-tail-starved and could
    veto a GCR result it never fed). Guards the jobs.py orchestration, which the
    pure-fold tests above cannot reach."""

    def _wait_terminal(self, runner, jid, timeout=5.0):
        import time
        from lunarsim.jobs import JobStatus
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = runner.get(jid)
            if job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
                return job
            time.sleep(0.01)
        self.fail("combined job did not reach a terminal state in time")

    def test_combined_runs_gcr_mc_only_no_spe_mc(self):
        from lunarsim import jobs
        from lunarsim.jobs import LocalThreadRunner, JobStatus
        from lunarsim.bridge import WORST_CASE_SPE

        calls = {"composition": 0, "converged": 0}

        def fake_composition(spec, tier, **kw):
            calls["composition"] += 1
            return _FakeComposition(spec)

        def fake_converged(spec, tier, **kw):
            calls["converged"] += 1        # must never fire in a combined job
            return _FakeComposition(spec)

        orig_comp, orig_conv = jobs.run_composition, jobs.run_converged
        jobs.run_composition, jobs.run_converged = fake_composition, fake_converged
        try:
            runner = LocalThreadRunner(max_parallel=1)
            jid = runner.submit(self._cal_dome(), spe=WORST_CASE_SPE, combined=True)
            job = self._wait_terminal(runner, jid)
        finally:
            jobs.run_composition, jobs.run_converged = orig_comp, orig_conv

        self.assertEqual(job.status, JobStatus.DONE)
        self.assertEqual(calls["composition"], 1)   # the one GCR MC
        self.assertEqual(calls["converged"], 0)      # NO acute-SPE MC
        self.assertEqual(job.phase, "gcr")           # only the GCR phase runs
        self.assertIsNotNone(job.spe)                # SPE gate still folded at display
        # the field that carried the removed SPE run is gone
        self.assertNotIn("spe_result", job.__dataclass_fields__)

    # reuse the calibrated 7.5 m dome from the fold tests
    _cal_dome = TestSpeKernelFold._cal_dome


class TestSafeName(unittest.TestCase):
    """spec.safe_name must yield a path/param-file-safe token no matter what the
    (possibly imported) design name contains, so an odd name can never crash a
    run or corrupt run.txt. See bridge.py mkdtemp / vis-dir / header sites."""

    def _named(self, name):
        return HabitatSpec(name=name, shape="dome", inner_radius_cm=300.0,
                           walls=[WallLayer("aluminium", 5.0)])

    def test_plain_name_survives(self):
        self.assertEqual(self._named("dome_A").safe_name, "dome_A")

    def test_slash_and_space_and_punctuation_are_neutralised(self):
        sn = self._named("Bob's Dome/v2").safe_name
        self.assertRegex(sn, r"^[A-Za-z0-9._-]+$")
        self.assertNotIn("/", sn)
        self.assertNotIn(" ", sn)

    def test_newline_cannot_break_the_param_file(self):
        # a newline in the name would inject a live line into run.txt's header
        sn = self._named("evil\nSc/Foo/Quantity = Bad").safe_name
        self.assertNotIn("\n", sn)
        self.assertRegex(sn, r"^[A-Za-z0-9._-]+$")

    def test_empty_or_symbol_only_falls_back(self):
        self.assertEqual(self._named("").safe_name, "habitat")
        self.assertEqual(self._named("///").safe_name, "habitat")

    def test_length_is_capped(self):
        self.assertLessEqual(len(self._named("x" * 200).safe_name), 48)


class TestArealDensityConfidence(unittest.TestCase):
    """The advisory trust band that flags designs sitting on the crossover cliff.
    Advisory only -- it must never be mistaken for a computed dose."""

    def test_thick_wall_is_high_confidence(self):
        self.assertEqual(dosimetry.areal_density_confidence(54.0)["level"], "high")

    def test_thin_wall_is_medium_and_flagged_optimistic(self):
        band = dosimetry.areal_density_confidence(8.0)
        self.assertEqual(band["level"], "medium")
        self.assertIn("optimistic", band["message"].lower())

    def test_gate_band_is_low_confidence(self):
        for ad in (13.0, 19.0, 25.0, 39.0):
            self.assertEqual(dosimetry.areal_density_confidence(ad)["level"],
                             "low", f"ad={ad} should be low-confidence")

    def test_every_band_carries_a_message(self):
        for ad in (5.0, 19.0, 60.0):
            b = dosimetry.areal_density_confidence(ad)
            self.assertTrue(b["label"] and b["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
