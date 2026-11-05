"""Tests for the kernel_gen geometry knobs and paper/geometry_residual_probe.py.

No TOPAS: synthetic scorer CSVs are written from the pre-tail kernel itself, so a
faithful collect -> splice -> fold round-trip must give a lift of exactly 1.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_geometry_probe -v
"""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim.kernel_gen import config, generate, templates  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "geometry_residual_probe", _REPO_ROOT / "paper" / "geometry_residual_probe.py")
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


def _base():
    return json.loads(gp.BASE.read_text())


def _write_synthetic(vdir: Path, base: dict, phi: float, scale_skin: float = 1.0):
    """Scorer CSVs D = R_base * phi (skin x scale_skin) for every manifest run."""
    m = json.loads((vdir / "manifest.json").read_text())
    pts = {round(p["wall_gcm2"], 6): p for p in base["points"]}
    organs = ["skin"] if m["skin_mode"] == "lining" else \
        [o for o, _ in config.organs_from_reference()]
    for r in m["runs"]:
        sp = pts[round(r["wall_gcm2"], 6)]["species"][r["species"]]
        for o in organs:
            fd, fi = templates.scorer_csv_names(o)
            k = scale_skin if o == "skin" else 1.0
            for q, fb in (("D", fd), ("I", fi)):
                v = sp["R"][o][q][r["node"]] * phi * k
                (vdir / "runs" / r["name"] / f"{fb}.csv").write_text(
                    f"# synthetic\n{v!r}\n")


class TestTemplateKnobs(unittest.TestCase):

    def test_phi_ff_default_and_scaling(self):
        self.assertEqual(templates.phi_ff(), config.PHI_FF_CM2)
        self.assertAlmostEqual(templates.phi_ff(config.BEAM_SPOT_CM, config.HISTORIES),
                               config.PHI_FF_CM2, places=15)
        self.assertAlmostEqual(templates.phi_ff(44.0, 16), config.PHI_FF_CM2, places=12)
        self.assertAlmostEqual(gp.variant_phi_ff(gp.VARIANTS["spot"]) / config.PHI_FF_CM2,
                               17 / 4 * (22 / 45) ** 2, places=12)

    def test_explicit_defaults_are_byte_identical(self):
        a = templates.build_param_file("aluminium", 10.0, "H", 2, 1, 1, "perelement")
        b = templates.build_param_file("aluminium", 10.0, "H", 2, 1, 1, "perelement",
                                       "shell", beam_spot_cm=config.BEAM_SPOT_CM,
                                       histories=config.HISTORIES, skin_mode="phantom")
        self.assertEqual(a, b)

    def test_spot_knob_lines(self):
        t = templates.build_param_file("aluminium", 50.0, "Fe", 0, 1, 1, "perelement",
                                       beam_spot_cm=45.0, histories=17)
        self.assertIn("d:So/BeamSpot   = 45.0 cm", t)
        self.assertEqual(t.count("NumberOfHistoriesInRun   = 17"),
                         config.RINGS * config.AZIMUTH)
        self.assertNotIn("= 4\n", t.split("d:Ge/BeamRadius")[1])
        for name, _, _ in config.SHELLS:
            self.assertIn(f"Ge/Shell_{name}/Type", t)

    def test_spot_fits_world(self):
        # Disc edge of the grazing ring stays inside the world and above z=0.
        theta = max(th for _, th in templates._ring_directions())
        spot = gp.VARIANTS["spot"]["beam_spot_cm"]
        th = math.radians(theta)
        x = config.BEAM_RADIUS_CM * math.sin(th) + spot * math.cos(th)
        z = config.BEAM_RADIUS_CM * math.cos(th) - spot * math.sin(th)
        self.assertLess(x, config.BEAM_RADIUS_CM + 50.0)
        self.assertGreater(z, 0.0)
        wall_rmax = config.WALL_RMIN_CM + 50.0 / config.MATERIALS["aluminium"]["density"]
        self.assertGreater(spot, wall_rmax)          # covers the 50 g/cm^2 wall
        self.assertGreater(config.BEAM_SPOT_CM,
                           config.WALL_RMIN_CM + 10.0 / config.MATERIALS["aluminium"]["density"])

    def test_lining_mode(self):
        t = templates.build_param_file("aluminium", 10.0, "H", 0, 1, 1, "perelement",
                                       skin_mode="lining")
        self.assertIn("d:Ge/Shell_skin/RMin     = 13.0000 cm", t)
        self.assertIn("d:Ge/Shell_skin/RMax     = 15.0000 cm", t)
        self.assertIn("d:Ge/Shell_skin/DTheta   = 90.0 deg", t)
        self.assertIn('s:Sc/Shell_skin_D/OutputFile = "skin_D"', t)
        self.assertIn('s:Sc/Shell_skin_I/OutputFile = "skin_I"', t)
        for name in ("core", "deep", "mid", "shallow"):
            self.assertNotIn(f"Shell_{name}", t)
        self.assertIn("d:Ge/Wall/RMin     = 15.0000 cm", t)

    def test_bad_knobs_rejected(self):
        with self.assertRaises(ValueError):
            templates.build_param_file("aluminium", 10.0, "H", 0, 1, skin_mode="slab")
        with self.assertRaises(ValueError):
            templates.build_param_file("aluminium", 10.0, "H", 0, 1, histories=0)


class TestProbe(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.base = _base()
        gp.emit(cls.out, ["lining", "spot", "both"], seeds=2, threads=1, anchors=[10.0, 50.0])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_emit_counts_and_manifest(self):
        n_nodes = sum(len(s["nodes_pernuc_mev"])
                      for s in config.species_for("perelement").values())
        for v in ("lining", "spot", "both"):
            m = json.loads((self.out / v / "manifest.json").read_text())
            self.assertEqual(len(m["runs"]), 2 * 2 * n_nodes)
            self.assertAlmostEqual(m["phi_ff_cm2"], gp.variant_phi_ff(gp.VARIANTS[v]))
            self.assertTrue((self.out / v / "run_all.sh").exists())

    def test_splice_lining_keeps_base_organs(self):
        with tempfile.TemporaryDirectory() as empty:       # no CSVs -> all zero
            p = generate._collect_point(Path(empty), gp.MATERIAL, 50.0, 2,
                                        gp.COMPOSITION)
        K = gp.splice(self.base, [p], skin_only=True, tag="t")
        i = [round(q["wall_gcm2"], 6) for q in K["points"]].index(50.0)
        for s, sp in K["points"][i]["species"].items():
            bsp = self.base["points"][i]["species"][s]
            self.assertEqual(sp["R"]["core"], bsp["R"]["core"])
            self.assertTrue(all(v == 0.0 for v in sp["R"]["skin"]["I"]))
        self.assertEqual(K["meta"]["composition"], self.base["meta"]["composition"])

    def test_unrun_variant_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gp.emit(d, ["lining", "spot"], seeds=1, threads=1, anchors=[10.0])
            _write_synthetic(d / "lining", self.base,
                             gp.variant_phi_ff(gp.VARIANTS["lining"]), scale_skin=1.10)
            resid = d / "resid.csv"
            resid.write_text("areal_gcm2,resid_after_skin,resid_after_skin_err\n"
                             "10.0,1.10,0.005\n")
            rows = gp.report(d, resid, d / "probe.csv", base_draws=0)  # spot unrun
        self.assertEqual({r["variant"] for r in rows}, {"lining"})
        self.assertAlmostEqual(rows[0]["after_skin"], 1.0, places=9)

    def test_round_trip_unit_and_scaled_lift(self):
        resid = self.out / "resid.csv"
        resid.write_text("areal_gcm2,resid_after_skin,resid_after_skin_err\n"
                         "10.0,1.10,0.005\n50.0,1.265,0.008\n")
        _write_synthetic(self.out / "spot", self.base, gp.variant_phi_ff(gp.VARIANTS["spot"]))
        _write_synthetic(self.out / "lining", self.base,
                         gp.variant_phi_ff(gp.VARIANTS["lining"]), scale_skin=1.10)
        _write_synthetic(self.out / "both", self.base,
                         gp.variant_phi_ff(gp.VARIANTS["both"]), scale_skin=1.10)
        rows = gp.report(self.out, resid, self.out / "probe.csv", base_draws=0)
        by = {(r["variant"], r["areal_gcm2"]): r for r in rows}
        for g in (10.0, 50.0):
            s = by[("spot", g)]
            self.assertEqual(s["missing_runs"], 0)
            for q in ("E", "phantom", "skin"):
                self.assertAlmostEqual(s[f"lift_{q}"], 1.0, places=9)
            ln = by[("lining", g)]
            self.assertAlmostEqual(ln["lift_skin"], 1.10, places=9)
            # phantom mass-average includes the skin shell: small, in (1, 1.10)
            self.assertGreater(ln["lift_phantom"], 1.0)
            self.assertLess(ln["lift_phantom"], 1.05)
        self.assertAlmostEqual(by[("lining", 10.0)]["after_skin"], 1.0, places=9)
        self.assertTrue((self.out / "probe.csv").exists())
        msgs = gp.verdict(rows)
        self.assertTrue(any("null control @10 PASS" in m for m in msgs))
        self.assertAlmostEqual(by[("both", 10.0)]["after_skin"], 1.0, places=9)
        self.assertTrue(any("both (measured jointly): @10" in m for m in msgs))

    def test_base_noise_positive_and_in_errors(self):
        rel = gp.base_noise([10.0, 50.0], self.out, draws=4)
        for g in (10.0, 50.0):
            for q in ("E", "phantom", "skin"):
                self.assertGreater(rel[g][q], 0.0)
                self.assertLess(rel[g][q], 0.5)
        self.assertEqual(gp.base_noise([10.0], self.out, draws=0)[10.0]["skin"], 0.0)


if __name__ == "__main__":
    unittest.main()
