"""Regression guard for the crossover coherence of the SHIPPED per-element
thin-wall GCR kernel (memory crossover-discontinuity, let-scorer-thin-qbar-
rootcause, perelement-kernel-activation).

Two fixes shipped into the thin-wall kernel and must not silently regress:
  * WALL_GEOMETRY = "shell" (kernel_gen/config.py) -- removed the slab slant that
    over-attenuated deep ABSORBED dose, so the kernel reproduces the OLTARIS deep
    anchor (0.296 mGy/day, <Q> 2.16 at 50 g/cm^2 Al).
  * the 10-ion per-element kernel -- removed the 5-group Z^2-lumping that inflated
    <Q> to ~4 at the 19 g/cm^2 gate; it now hands off to the deep engine at the
    OLTARIS deep <Q>.

These assertions are pure-arithmetic folds of the committed kernel JSONs -- no
TOPAS, no MC, sub-second. The numbers are reproduced by paper/crossover_coherence.py.

TARGET-BLIND: OLTARIS values are a post-hoc reference, never a fit target.

    cd ~/topas && python3 -m unittest lunarsim.tests.test_crossover_coherence -v
"""
import json
import os
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lunarsim.spec import HabitatSpec, WallLayer
from lunarsim import dosimetry

AL_DENSITY = 2.7
PHI_MV = 400.0
# OLTARIS lunar-surface TOTAL at ~54 g/cm^2 Al (post-hoc reference only).
OLTARIS_D_MGY_DAY = 0.296
OLTARIS_QBAR = 2.16


def _al_spec(ad_gcm2):
    return HabitatSpec(name=f"al_{ad_gcm2:g}", shape="dome", inner_radius_cm=400.0,
                       walls=[WallLayer(material="aluminium",
                                        thickness_cm=ad_gcm2 / AL_DENSITY)])


def _fold(ad_gcm2, model):
    os.environ["LUNARSIM_GCR_PERELEMENT"] = model
    dosimetry.set_perelement_model(model)
    a = dosimetry.assess_gcr_thinwall(_al_spec(ad_gcm2), phi_MV=PHI_MV)
    return {"D_mgy_day": a.dose_rate_ugy_day / 1.0e3,
            "E_msv_yr": a.annual_msv, "qbar": a.quality_factor}


class TestPerElementKernelIntegrity(unittest.TestCase):
    """The committed per-element kernel must carry the two shipped fixes."""

    def setUp(self):
        path = _REPO_ROOT / "lunarsim" / "data" / "gcr_thinwall_kernel_perelem.json"
        self.assertTrue(path.exists(), "shipped per-element kernel JSON missing")
        with open(path) as fh:
            self.K = json.load(fh)

    def test_shell_geometry_and_composition(self):
        m = self.K["meta"]
        self.assertEqual(m["wall_geometry"], "shell",
                         "slab slant regressed -> deep absorbed over-attenuates")
        self.assertEqual(m["composition"], "perelement")
        self.assertEqual(m["crossover_gcm2"], 19.0)

    def test_ten_species_delumped(self):
        species = set(self.K["points"][0]["species"].keys())
        self.assertEqual(species,
                         {"H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Ca", "Fe"})


class TestCrossoverCoherence(unittest.TestCase):
    """Fold invariants across the ~19 g/cm^2 crossover. Numbers match
    paper/crossover_coherence.py; tolerances are loose enough to survive a
    behaviour-preserving kernel refold but tight enough to catch a physics change."""

    def setUp(self):
        self._saved = os.environ.get("LUNARSIM_GCR_PERELEMENT")
        self.assertTrue(dosimetry._has_perelem_kernel(),
                        "per-element kernel not present for coherence test")
        self.sweep = [2.025, 5.0, 10.0, 15.0, 19.0, 25.0, 40.0, 50.0]
        self.base = {ad: _fold(ad, "none") for ad in self.sweep}
        self.perel = {ad: _fold(ad, "perelement") for ad in self.sweep}

    def tearDown(self):
        dosimetry.set_perelement_model(self._saved or "none")

    def test_baseline_reproduces_oltaris_deep_anchor(self):
        # The shell fix: the 5-ion kernel lands on OLTARIS at 50 g/cm^2.
        b = self.base[50.0]
        self.assertAlmostEqual(b["D_mgy_day"] / OLTARIS_D_MGY_DAY, 1.0, delta=0.10)
        self.assertAlmostEqual(b["qbar"] / OLTARIS_QBAR, 1.0, delta=0.10)

    def test_qbar_monotone_with_shielding(self):
        for d in (self.base, self.perel):
            q = [d[ad]["qbar"] for ad in self.sweep]
            for i in range(len(q) - 1):
                self.assertGreaterEqual(q[i] + 1e-9, q[i + 1],
                                        f"<Q> not monotone at {self.sweep[i]} g/cm^2")

    def test_delumping_lowers_qbar_and_dose_everywhere(self):
        # The per-element fix only ever removes over-read: <Q> and E must drop.
        for ad in self.sweep:
            self.assertLessEqual(self.perel[ad]["qbar"], self.base[ad]["qbar"] + 1e-9)
            self.assertLessEqual(self.perel[ad]["E_msv_yr"],
                                 self.base[ad]["E_msv_yr"] + 1e-9)

    def test_qbar_continuous_at_gate(self):
        # The headline coherence win: per-element thin <Q> at the gate matches the
        # OLTARIS deep <Q> (was ~4.0 lumped). Pin the value AND the OLTARIS match.
        q_gate = self.perel[19.0]["qbar"]
        self.assertAlmostEqual(q_gate, 2.163, delta=0.08)
        self.assertAlmostEqual(q_gate / OLTARIS_QBAR, 1.0, delta=0.10)
        # the old lumped kernel over-reads here -- the defect we guard against
        self.assertGreater(self.base[19.0]["qbar"], 3.5)

    def test_gate_headline_matches_validated_shielded_dome(self):
        # per-element E at the gate (memory perelement-kernel-activation: lumped
        # over-read was ~539). Was ~130 mSv/yr under the legacy 2.0 /cm^2/s proton
        # pin; since 2026-10-27 the Matthiae models keep their own PAMELA-validated
        # absolute flux (dosimetry._calibration_factor), x1.215 on the kernel's
        # rigidity form -> ~158. Regression anchor, not a target.
        self.assertAlmostEqual(self.perel[19.0]["E_msv_yr"], 158.0, delta=15.0)
        self.assertGreater(self.base[19.0]["E_msv_yr"], 450.0)

    def test_organ_absorbed_in_self_shielding_band(self):
        # per-element organ-absorbed vs the OLTARIS bare point. Was ~0.56x under the
        # legacy 2.0 proton pin; x1.215 since the Matthiae absolute normalisation
        # (2026-10-27) -> ~0.68x. Still below the point: organ self-shielding.
        r = self.perel[50.0]["D_mgy_day"] / OLTARIS_D_MGY_DAY
        self.assertTrue(0.58 <= r <= 0.78, f"organ/point absorbed ratio {r:.3f} "
                        "outside the 0.58-0.78 self-shielding band")

    def test_perelement_requires_env_gate(self):
        # Double-gate footgun: with the env off, fold falls back to the lumped
        # kernel (the over-read), so the two models must differ at the gate.
        self.assertGreater(self.base[19.0]["qbar"] - self.perel[19.0]["qbar"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
