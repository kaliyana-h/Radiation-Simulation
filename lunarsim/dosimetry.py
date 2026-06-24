"""Dosimetry: turn a TOPAS run's per-primary absorbed dose into a physical
crew dose and a safety verdict -- the workshop's "how safe is my design?" answer.

The normalisation is self-consistent and reuses the source's own physics:

  * A TOPAS run fires N_sim simulated primaries and reports an absorbed dose
    D_sim (Gy) in the phantom and a *scalar* fluence F_sim (/cm^2) on the
    OuterShell (track-length / volume -- the same quantity for sim and reality).
  * The real lunar-surface GCR proton field has a scalar fluence rate
    Phi_real (/cm^2/s), obtained by integrating the *same* force-field-modulated
    LIS spectrum that make_source.py builds the beam from, over energy and over
    the unobstructed upper hemisphere (2*pi sr).
  * Because both fluences are measured identically and the source reproduces the
    real angular distribution, the dose scales linearly with fluence:

        dose_rate = D_sim * Phi_real / F_sim      [Gy/s]

So the absolute scale comes entirely from the GCR flux model -- the simulation
only has to get the *shielding response* (dose per unit incident fluence) right,
which is exactly what a Monte Carlo through the wall is good at.

Equivalent (effective) dose uses a single mean field quality factor Q to map
Gy -> Sv. That is an approximation pending the per-particle dose split (a
ParentID/particle TsVFilter -- project priority #2/#3); the secondary neutron
component carries a much higher w_R, so a single Q is documented as a placeholder
and exposed as a tunable.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Optional

from .bridge import RunResult, MAKE_SOURCE

E0_PROTON = 938.272            # MeV, proton rest energy
SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.0

# Mean field quality factor for a GCR-dominated field behind modest shielding.
# Literature mean Q for the deep-space/surface GCR field is ~2-6 (rises as
# shielding hardens the secondary neutron component). 3.5 is a defensible
# mid-range default; override per design once the particle split exists.
DEFAULT_QUALITY_FACTOR = 3.5

# Reference exposure limits (whole-body effective dose, mSv).
DOSE_LIMITS_MSV = {
    "nasa_30day":  250.0,     # NASA short-term BFO limit
    "nasa_annual": 500.0,     # NASA annual limit
    "career":      600.0,     # NASA career limit (3% REID, lower bound)
    "esa_career": 1000.0,     # ESA/Roscosmos career limit
}


# ----------------------------------------------------------------------
# GCR integral flux from the source's own spectrum
# ----------------------------------------------------------------------
def _load_make_source():
    spec = importlib.util.spec_from_file_location("make_source", MAKE_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gcr_scalar_fluence_rate(phi_MV: float = 400.0,
                            emin_mev: float = 10.0, emax_mev: float = 1.0e5,
                            n: int = 4000) -> float:
    """Upper-hemisphere scalar fluence rate of GCR protons at the lunar surface.

    Integrates the force-field-modulated LIS proton spectrum (Usoskin 2005, via
    make_source.lis_proton) over energy and the 2*pi-sr upper hemisphere.
    Returns protons /cm^2/s (scalar fluence, comparable to a TOPAS Fluence
    scorer)."""
    ms = _load_make_source()
    # differential flux dJ/dT [/(m^2 s sr GeV)] on a log-energy grid, trapezoid in T
    import math
    integral_m2_sr = 0.0       # /(m^2 s sr)
    T_prev = j_prev = None
    for k in range(n):
        T = emin_mev * (emax_mev / emin_mev) ** (k / (n - 1))   # MeV
        Tm = T + phi_MV                                         # z/a = 1 for p
        factor = (T * (T + 2 * E0_PROTON)) / (Tm * (Tm + 2 * E0_PROTON))
        j = ms.lis_proton(Tm / 1000.0) * factor                # /(m^2 s sr GeV)
        if T_prev is not None:
            dT_gev = (T - T_prev) / 1000.0
            integral_m2_sr += 0.5 * (j + j_prev) * dT_gev
        T_prev, j_prev = T, j
    integral_cm2_sr = integral_m2_sr / 1.0e4                    # -> /(cm^2 s sr)
    # scalar fluence from isotropic upper hemisphere = intensity * 2*pi sr
    return 2.0 * math.pi * integral_cm2_sr                      # /cm^2/s


# ----------------------------------------------------------------------
# Assessment
# ----------------------------------------------------------------------
@dataclass
class DoseAssessment:
    real_flux_cm2_s: float          # GCR scalar fluence rate used to normalise
    sim_fluence_cm2: float          # OuterShell scalar fluence (sim)
    dose_rate_gy_s: float           # absorbed dose rate at crew position
    quality_factor: float
    mission_days: float

    # ---- absorbed dose ----
    @property
    def dose_rate_ugy_day(self) -> float:
        return self.dose_rate_gy_s * SECONDS_PER_DAY * 1.0e6      # uGy/day

    @property
    def annual_mgy(self) -> float:
        return self.dose_rate_gy_s * SECONDS_PER_DAY * DAYS_PER_YEAR * 1.0e3

    # ---- equivalent (effective) dose ----
    @property
    def equiv_rate_msv_day(self) -> float:
        return self.dose_rate_gy_s * SECONDS_PER_DAY * self.quality_factor * 1.0e3

    @property
    def mission_msv(self) -> float:
        return self.equiv_rate_msv_day * self.mission_days

    @property
    def annual_msv(self) -> float:
        return self.equiv_rate_msv_day * DAYS_PER_YEAR

    # ---- safety verdict ----
    def fraction_of(self, limit_key: str = "career") -> float:
        return self.mission_msv / DOSE_LIMITS_MSV[limit_key]

    def verdict(self, limit_key: str = "career") -> str:
        f = self.fraction_of(limit_key)
        if f < 0.5:
            return "SAFE"
        if f < 1.0:
            return "MARGINAL"
        return "EXCEEDS LIMIT"

    def summary(self, limit_key: str = "career") -> dict:
        return {
            "dose_rate_uGy_per_day": round(self.dose_rate_ugy_day, 3),
            "annual_mGy": round(self.annual_mgy, 2),
            "equivalent_mSv_per_day": round(self.equiv_rate_msv_day, 4),
            "mission_mSv": round(self.mission_msv, 1),
            "mission_days": self.mission_days,
            "quality_factor": self.quality_factor,
            "limit_mSv": DOSE_LIMITS_MSV[limit_key],
            "fraction_of_limit": round(self.fraction_of(limit_key), 3),
            "verdict": self.verdict(limit_key),
        }


def assess(result: RunResult,
           mission_days: float = 365.0,
           quality_factor: float = DEFAULT_QUALITY_FACTOR,
           phi_MV: Optional[float] = None,
           skin: bool = False) -> Optional[DoseAssessment]:
    """Convert a completed RunResult into a physical DoseAssessment.

    skin=False (default) normalises the central crew phantom point dose; skin=True
    normalises the inner-wall lining shell -- a habitat-wide dose with far better
    statistics (it differs from the phantom mainly by having no self-shielding, so
    it tends to read a bit higher). Both go through the identical fluence
    normalisation.

    Returns None if the run lacks the dose / outer-fluence needed to normalise.
    phi_MV defaults to the run's own source modulation (worst-case solar min)."""
    dose_gy = result.skin_dose_gy if skin else result.dose_gy
    if dose_gy is None or not result.fluence_outside:
        return None
    if phi_MV is None:
        phi_MV = result.tier.phi_mv

    real_flux = gcr_scalar_fluence_rate(phi_MV)
    sim_fluence_cm2 = result.fluence_outside * 100.0          # /mm^2 -> /cm^2
    dose_rate_gy_s = dose_gy * real_flux / sim_fluence_cm2

    return DoseAssessment(
        real_flux_cm2_s=real_flux,
        sim_fluence_cm2=sim_fluence_cm2,
        dose_rate_gy_s=dose_rate_gy_s,
        quality_factor=quality_factor,
        mission_days=mission_days,
    )


if __name__ == "__main__":
    flux = gcr_scalar_fluence_rate(400.0)
    print(f"GCR proton scalar fluence rate (phi=400 MV): {flux:.3f} /cm2/s")
    # sanity: integral GCR proton flux at solar min is a few /cm2/s
