"""Beam-spot knob for regolith_sweep + paper/regolith_spot_probe.py (no TOPAS)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from lunarsim import regolith_sweep as rs
from lunarsim.kernel_gen import templates

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "regolith_spot_probe", _REPO / "paper" / "regolith_spot_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

_SP = ["H", "proton", 1, 1, 0.9, "H"]


class SpotKnob(unittest.TestCase):
    def test_default_param_file_unchanged(self):
        a = rs.build_param_file(1.0, _SP, 100, 1)
        b = rs.build_param_file(1.0, _SP, 100, 1, beam_spot_cm=rs.BEAM_SPOT_CM)
        self.assertEqual(a, b)
        self.assertIn("d:So/BeamSpot   = 22.0 cm", a)

    def test_spot_written(self):
        txt = rs.build_param_file(1.0, _SP, 100, 1, beam_spot_cm=90.0)
        self.assertIn("d:So/BeamSpot   = 90.0 cm", txt)
        self.assertIn("beam_spot=90 cm", txt)

    def test_phi_ff_scales_inverse_area(self):
        self.assertAlmostEqual(rs.phi_ff_cm2(100) / rs.phi_ff_cm2(100, 44.0), 4.0)
        self.assertEqual(rs.phi_ff_cm2(100), rs.phi_ff_cm2(100, rs.BEAM_SPOT_CM))

    def test_default_fits_all_depths(self):
        for d in rs.DEPTHS_M:
            self.assertEqual(rs.illumination_problems(d, rs.BEAM_SPOT_CM), [], d)

    def test_probe_spots_fit(self):
        self.assertEqual(probe.check_spots(probe.DEFAULT_SPOTS, probe.DEFAULT_DEPTHS), [])

    def test_guard_catches_disc_in_slab(self):
        p = rs.illumination_problems(3.0, 180.0)
        self.assertTrue(p and "below the slab top" in p[0], p)
        self.assertTrue(rs.illumination_problems(1.0, 240.0))

    def test_guard_matches_closed_form(self):
        # grazing ring: lowest disc point z = R cos(th) - spot sin(th)
        import math
        th = max(t for _, t in templates._ring_directions())
        z_low = rs.BEAM_RADIUS_CM * math.cos(math.radians(th))
        s_max = (z_low - (rs.config.WALL_RMIN_CM + 200.0)) / math.sin(math.radians(th))
        self.assertEqual(rs.illumination_problems(2.0, s_max - 1.0), [])
        self.assertTrue(rs.illumination_problems(2.0, s_max + 1.0))

    def test_generate_refuses_and_records(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                rs.generate(Path(d) / "x", 10, 1, 0, depths_m=[3.0], beam_spot_cm=180.0)
            self.assertFalse((Path(d) / "x").exists())
            rs.generate(Path(d) / "y", 10, 1, 0, depths_m=[1.0], beam_spot_cm=90.0)
            m = json.loads((Path(d) / "y" / "manifest.json").read_text())
            self.assertEqual(m["beam_spot_cm"], 90.0)
            self.assertEqual(m["depths_m"], [1.0])
            self.assertAlmostEqual(m["phi_ff_cm2"], rs.phi_ff_cm2(10, 90.0))
            pre = f"d{rs.depth_tag(1.0)}_"
            self.assertTrue(m["runs"] and all(r.startswith(pre) for r in m["runs"]))


def _fake_outputs(out, man, scale):
    for rn in man["runs"]:
        seed = int(rn.rsplit("_s", 1)[1])
        for o in rs._REPORT_ORGANS:
            for f in templates.scorer_csv_names(o):
                (out / rn / f"{f}.csv").write_text(f"# x\n{scale * (1 + 0.1 * seed)}\n")


class ProbeEndToEnd(unittest.TestCase):
    def test_histories_scale(self):
        self.assertEqual(probe.histories_for(44.0, 100, 2.0), 400)
        self.assertEqual(probe.histories_for(44.0, 100, 1.0), 200)
        self.assertEqual(probe.histories_for(22.0, 100, 2.0), 100)

    def test_generate_compare(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            probe.generate(root, [22.0, 44.0], [0.3, 1.0], base=10, exponent=2.0,
                           seeds=2, threads=0, phi=rs.PHI_MV)
            self.assertIn("spot44/run.sh", (root / "run_all.sh").read_text())
            # equal per-run scorer tallies at 4x the histories: spot 44 has the same
            # phi_ff, so the folded dose must be equal (ratio 1)
            for s in (22.0, 44.0):
                sd = probe.spot_dir(root, s)
                _fake_outputs(sd, json.loads((sd / "manifest.json").read_text()), 1.0)
            self.assertEqual(probe.compare(root), 0)
            rows = list(__import__("csv").DictReader((root / "spot_compare.csv").open()))
            self.assertEqual(len(rows), 4)
            for r in rows:
                self.assertAlmostEqual(float(r["skin_over_ref"]), 1.0)
                self.assertNotEqual(r["skin_relerr"], "")

    def test_compare_rows_ratio(self):
        mk = lambda v: [{"depth_m": 1.0, "areal_gcm2": 193.7, "usable": True,
                         "skin_doseeq_mSv_yr": v, "core_doseeq_mSv_yr": v / 2,
                         "skin_doseeq_relerr": 0.03, "core_doseeq_relerr": 0.04}]
        rows = probe.compare_rows({22.0: mk(25.0), 90.0: mk(50.0)})
        self.assertEqual(rows[1]["skin_over_ref"], 2.0)
        self.assertAlmostEqual(rows[1]["skin_ratio_relerr"], 0.0424, places=4)
        self.assertEqual(rows[0]["skin_ratio_relerr"], "")


if __name__ == "__main__":
    unittest.main()
