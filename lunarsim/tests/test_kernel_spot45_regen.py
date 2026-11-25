"""paper/kernel_spot45_regen.py: rebuild only the 50 g/cm2 anchor (no TOPAS)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from lunarsim.kernel_gen import config, generate, templates

_REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _REPO / "paper" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


regen = _load("kernel_spot45_regen")
probe = _load("geometry_residual_probe")
ORGANS = [o for o, _ in config.organs_from_reference()]


def _fake(d: Path, value: float):
    """Every run's scorers read `value` (seed-independent)."""
    m = json.loads((d / "manifest.json").read_text())
    for r in m["runs"]:
        rn = r["name"] if isinstance(r, dict) else r
        for o in ORGANS:
            for f in templates.scorer_csv_names(o):
                (d / "runs" / rn / f"{f}.csv").write_text(f"# x\n{value}\n")


class Spot45Regen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        probe.emit(root, ["spot"], seeds=2, threads=1, anchors=[10.0, 50.0])
        cls.probe_dir = root / "spot"
        cls.tail_dir = root / "tail"
        cls.n_tail = regen.emit_tail(cls.tail_dir, seeds=2)
        _fake(cls.probe_dir, 3.0)
        _fake(cls.tail_dir, 5.0)
        cls.K0 = json.loads(regen.SHIPPED.read_text())
        cls.K = regen.merge_into(cls.K0, regen.build_anchor(cls.probe_dir, cls.tail_dir))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tail_batch_is_50_only_45cm(self):
        n_nodes = sum(len(s["nodes_pernuc_mev"])
                      for s in config.species_for("perelement_tail").values())
        self.assertEqual(self.n_tail, n_nodes * 2)
        txt = next((self.tail_dir / "runs").iterdir()).joinpath("run.txt").read_text()
        self.assertIn("BeamSpot   = 45.0 cm", txt)
        self.assertTrue(all("_w50_" in r["name"] for r in
                            json.loads((self.tail_dir / "manifest.json").read_text())["runs"]))

    def test_other_anchors_untouched(self):
        for p0, p in zip(self.K0["points"], self.K["points"]):
            if p0["wall_gcm2"] != 50.0:
                self.assertEqual(p0, p)

    def test_anchor50_values_and_nodes(self):
        phi = regen.phi_ff()
        (p0,) = [p for p in self.K0["points"] if p["wall_gcm2"] == 50.0]
        (p,) = [p for p in self.K["points"] if p["wall_gcm2"] == 50.0]
        nb = {s: len(v["nodes_pernuc_mev"])
              for s, v in config.species_for("perelement").items()}
        for s, sp in p["species"].items():
            self.assertEqual(sp["nodes_pernuc_mev"], p0["species"][s]["nodes_pernuc_mev"])
            for k in ("z", "a", "abundance", "particle", "group"):
                self.assertEqual(sp[k], p0["species"][s][k])
            for o in ORGANS:
                R = sp["R"][o]["I"]
                for v in R[:nb[s]]:
                    self.assertAlmostEqual(v, 3.0 / phi)
                for v in R[nb[s]:]:
                    self.assertAlmostEqual(v, 5.0 / phi)
                self.assertEqual(set(sp["Rsem"][o]["D"]), {0.0})
        self.assertEqual(self.K["meta"]["anchor50_spot45"]["beam_spot_cm"], 45.0)

    def test_phi_matches_probe_variant(self):
        self.assertAlmostEqual(regen.phi_ff(), probe.variant_phi_ff(probe.VARIANTS["spot"]))

    def test_refuses_wrong_or_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            lin = Path(d)
            probe.emit(lin, ["lining"], seeds=1, threads=1, anchors=[50.0])
            with self.assertRaises(SystemExit):           # 22 cm lining variant
                regen.build_anchor(lin / "lining", self.tail_dir)
            t = Path(d) / "t"
            regen.emit_tail(t, seeds=1)                    # no outputs
            with self.assertRaises(SystemExit):
                regen.build_anchor(self.probe_dir, t)


if __name__ == "__main__":
    unittest.main()
