"""Guards for paper/nasa_q_breakdown.py (no TOPAS needed)."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "nasa_q_breakdown", _REPO_ROOT / "paper" / "nasa_q_breakdown.py")
nqb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nqb)


def _assess(annual, q, contribs):
    return SimpleNamespace(annual_msv=annual, quality_factor=q, rel_err=0.01,
                           contributions=contribs)


def _c(name, d, h, frac):
    return {"species": name, "dose_rate_gy_s": d, "doseeq_rate_sv_s": h,
            "quality_factor": h / d, "dose_fraction": frac, "rel_err": 0.02}


class Breakdown(unittest.TestCase):
    def setUp(self):
        # H: D 3, H_I 6, H_N 9 ; Fe: D 1, H_I 4, H_N 5  -> totals I 10, N 14
        self.ai = _assess(100.0, 2.5, [_c("H", 3, 6, 0.75), _c("Fe", 1, 4, 0.25)])
        self.an = _assess(140.0, 3.5, [_c("H", 3, 9, 0.75), _c("Fe", 1, 5, 0.25)])

    def test_species_h_sum_to_total(self):
        rows = nqb.breakdown(self.ai, self.an)
        sp = [r for r in rows if r["species"] != "TOTAL"]
        self.assertAlmostEqual(sum(r["h_icrp"] for r in sp), 100.0)
        self.assertAlmostEqual(sum(r["h_nasa"] for r in sp), 140.0)
        self.assertEqual(rows[-1]["species"], "TOTAL")
        self.assertAlmostEqual(rows[-1]["ratio"], 1.4)

    def test_excess_shares(self):
        rows = {r["species"]: r for r in nqb.breakdown(self.ai, self.an)}
        # H: 90-60 = 30 of 40 ; Fe: 50-40 = 10 of 40
        self.assertAlmostEqual(rows["H"]["excess_share"], 0.75)
        self.assertAlmostEqual(rows["Fe"]["excess_share"], 0.25)
        self.assertAlmostEqual(rows["H"]["ratio"], 1.5)
        self.assertAlmostEqual(rows["Fe"]["ratio"], 1.25)
        self.assertAlmostEqual(rows["Fe"]["q_nasa"], 5.0)

    def test_mismatched_species_raise(self):
        bad = _assess(140.0, 3.5, [_c("H", 3, 9, 1.0)])
        with self.assertRaises(ValueError):
            nqb.breakdown(self.ai, bad)

    def test_csv_rows_fields(self):
        rows = nqb.breakdown(self.ai, self.an)
        out = nqb.csv_rows(19.0, "skin", rows, {"H": 0.1, "TOTAL": 0.08}, "p", "d")
        self.assertEqual(set(out[0]), set(nqb.CSV_FIELDS))
        self.assertEqual(out[0]["f_neutron_skin"], "0.1000")
        self.assertEqual(out[1]["f_neutron_skin"], "--")      # Fe has no fraction
        ph = nqb.csv_rows(19.0, "phantom", rows, {"H": 0.1}, "p", "d")
        self.assertEqual(ph[0]["f_neutron_skin"], "")


if __name__ == "__main__":
    unittest.main()
