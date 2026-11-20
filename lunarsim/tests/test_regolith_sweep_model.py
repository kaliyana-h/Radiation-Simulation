"""regolith_sweep samples and normalises under ONE GCR model (2026-11-20 fix).

Before the fix the sweep paired hand-set GCR_COMPOSITION abundances (C/Si/Fe
4.4-7x the Matthiae group values) with single-element spectra and a flux
integral normalised only to 20 GeV/n. No TOPAS needed."""
import json
import tempfile
import unittest
from pathlib import Path

from lunarsim import dosimetry
from lunarsim import regolith_sweep as rs


class SweepModel(unittest.TestCase):
    def test_model_is_flood_with_sweep_ceiling(self):
        m = rs.sweep_model()
        self.assertEqual(m.perelement, dosimetry.current_model().flood().perelement)
        self.assertEqual(m.perelement, "matthia2013")
        self.assertEqual(m.emax_per_nuc, rs.EMAX_PER_NUC_MEV)

    def test_manifest_roundtrip(self):
        m = rs.sweep_model()
        import dataclasses
        self.assertEqual(rs.model_from_manifest({"gcr_model": dataclasses.asdict(m)}), m)
        self.assertIsNone(rs.model_from_manifest({}))


class Generate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        rs.generate(cls.out, histories=10, seeds=1, threads=0, phi_mv=rs.PHI_MV)
        cls.manifest = json.loads((cls.out / "manifest.json").read_text())
        cls.model = rs.sweep_model()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_manifest_stamped(self):
        self.assertEqual(rs.model_from_manifest(self.manifest), self.model)
        self.assertIn("engine=regolith_slab", self.manifest["provenance"])
        self.assertIn("perelement=matthia2013", self.manifest["provenance"])

    def test_composition_is_group_not_handset(self):
        with dosimetry.use_model(self.model):
            want = [list(s) for s in dosimetry.active_composition(rs.PHI_MV)]
        self.assertEqual(self.manifest["composition"], want)
        ab = {s[0]: s[4] for s in self.manifest["composition"]}
        hand = {s[0]: s[4] for s in dosimetry.GCR_COMPOSITION}
        for heavy in ("C", "Si", "Fe"):
            if heavy in ab and heavy in hand:
                self.assertLess(ab[heavy], hand[heavy] / 3.0, heavy)

    def test_param_file_records_model(self):
        rn = self.manifest["runs"][0]
        txt = (self.out / rn / f"{rn}.txt").read_text()
        self.assertIn(self.model.label(), txt)

    def test_fold_setup_uses_manifest(self):
        model, comp, prov = rs.fold_setup(self.manifest)
        self.assertEqual(model, self.model)
        self.assertEqual([list(s) for s in comp], self.manifest["composition"])
        self.assertTrue(prov)

    def test_flux_normalised_to_sweep_ceiling(self):
        # H flux under the sweep model includes the 20-100 GeV/n tail
        z, a = 1, 1
        with dosimetry.use_model(self.model):
            f100 = dosimetry.gcr_species_fluence_rate(z, a, 1.0, rs.PHI_MV)
        import dataclasses
        with dosimetry.use_model(dataclasses.replace(self.model, emax_per_nuc=2.0e4)):
            f20 = dosimetry.gcr_species_fluence_rate(z, a, 1.0, rs.PHI_MV)
        self.assertGreater(f100, f20)
        self.assertLess(f100 / f20, 1.05)


class CollectEndToEnd(unittest.TestCase):
    def test_collect_folds_manifest_abundances(self):
        from lunarsim.kernel_gen import templates
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rs.generate(out, histories=10, seeds=1, threads=0, phi_mv=rs.PHI_MV)
            man = json.loads((out / "manifest.json").read_text())
            for rn in man["runs"]:                       # unit D and I per run
                for o in rs._REPORT_ORGANS:
                    for f in templates.scorer_csv_names(o):
                        (out / rn / f"{f}.csv").write_text("# x\n1.0\n")
            res = rs.collect(out)
            model = rs.model_from_manifest(man)
            with dosimetry.use_model(model):
                flux = sum(dosimetry.gcr_species_fluence_rate(z, a, ab, rs.PHI_MV)
                           for _, _, z, a, ab, _ in man["composition"])
            want = flux / man["phi_ff_cm2"] * 365.25 * 86400 * 1e3
            r0 = res["rows"][0]
            self.assertTrue(r0["usable"])
            self.assertAlmostEqual(r0["skin_doseeq_mSv_yr"] / want, 1.0, places=3)
            self.assertIn("regolith_slab", r0["provenance"])


class Legacy(unittest.TestCase):
    def test_unstamped_refused(self):
        with self.assertRaises(SystemExit):
            rs.fold_setup({"phi_MV": 475.0})

    def test_unstamped_legacy_flag(self):
        model, comp, prov = rs.fold_setup({"phi_MV": 475.0}, legacy=True)
        self.assertEqual(list(comp), list(dosimetry.GCR_COMPOSITION))
        self.assertIn("legacy", prov)


if __name__ == "__main__":
    unittest.main()
