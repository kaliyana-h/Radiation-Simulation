"""Guards for paper/parent_q_test.py and the DoseEquivalent_ICRP_ParentQ twin (no TOPAS).

The A/B is valid only if the run file is the production file plus the _P scorer
lines and nothing else, and the C++ twin registers under the quantity name used.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_parent_q_harness -v
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lunarsim.kernel_gen import config, templates  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "parent_q_test", _REPO_ROOT / "paper" / "parent_q_test.py")
pq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pq)


def _prod():
    return templates.build_param_file("aluminium", 54.0, "Fe", 0, 1, threads=1,
                                      composition=pq.COMPOSITION,
                                      wall_geometry=pq.WALL_GEOMETRY)


class ParentQHarness(unittest.TestCase):
    def test_text_is_production_plus_parent_scorers_only(self):
        prod = _prod().splitlines()
        new = pq.build_text("aluminium", 54.0, "Fe", 0, 1, threads=1).splitlines()
        self.assertEqual([ln for ln in prod if ln not in new], [])
        expected = [ln for name, _, _ in config.SHELLS for ln in pq.scorer_lines(name)]
        self.assertEqual([ln for ln in new if ln not in prod], expected)
        # order of the production lines is preserved
        self.assertEqual([ln for ln in new if ln in prod], prod)

    def test_scorer_names_and_files(self):
        t = pq.build_text("aluminium", 54.0, "H", 0, 1, threads=1)
        for name, _, _ in config.SHELLS:
            comp = templates._shell_component_name(name)
            self.assertIn(f's:Sc/{comp}_P/Quantity   = "DoseEquivalent_ICRP_ParentQ"', t)
            self.assertIn(f's:Sc/{comp}_P/Component  = "{comp}"', t)
            self.assertIn(f's:Sc/{comp}_P/OutputFile = "{name}_P"', t)
            self.assertNotIn(pq.parent_csv_name(name), templates.scorer_csv_names(name))
        self.assertEqual(t.count('"DoseEquivalent_ICRP_ParentQ"'), len(config.SHELLS))
        self.assertIn('Ph/Default/Type = "FTFP_BERT_HP"', t)   # production physics
        self.assertNotIn("CutFor", t)                          # production cuts

    def test_cc_first_line_registers_the_quantity(self):
        first = pq.EXT_CC.read_text().splitlines()[0]
        self.assertEqual(first, f"// Scorer for {pq.QUANTITY}")
        self.assertTrue(pq.EXT_CC.with_suffix(".hh").exists())

    def test_paired_ratio(self):
        with tempfile.TemporaryDirectory() as d:
            rds = []
            for k, (i, p) in enumerate(((2.0, 2.2), (4.0, 4.8))):
                rd = Path(d) / f"s{k}"
                rd.mkdir()
                (rd / "mid_I.csv").write_text(f"# hdr\n{i}\n")
                (rd / "mid_P.csv").write_text(f"# hdr\n{p}\n")
                rds.append(rd)
            r, se, n = pq.paired_ratio(rds, "mid")
            self.assertEqual(n, 2)
            self.assertAlmostEqual(r, 1.15)
            self.assertAlmostEqual(se, 0.05)


if __name__ == "__main__":
    unittest.main()
