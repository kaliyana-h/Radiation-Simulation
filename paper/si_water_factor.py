#!/usr/bin/env python3
"""Silicon -> water absorbed-dose factor for the tool's OWN GCR charged field.

Chang'E-4 LND reports absorbed dose in silicon; Zhang et al. 2020 convert to water
with a constant 1.30 (CRaTER uses 1.33). That constant is not the tool's: the
ratio of mass stopping powers depends on the spectrum. This module folds the
tool's own input spectrum -- make_source.mattia_flux (Matthiae 2013, Z = 1-28,
10 MeV/n to the 20 GeV/n emission ceiling, W from the run's phi) -- with the
Bethe mass stopping power in water and in silicon over the 2 pi open sky, and
returns D_water / D_Si. Primary charged field only, no transport: it is the
factor for the LND CHARGED channel, the gate channel of lnd_absolute_recheck.

Result at phi = 400 MV (W = 4): ~1.23. Independent check: Matthiae & Berger 2024
(Space Weather 22 e2024SW004095; Geant4, Matthiae 2013, 300 um Si/tissue slabs)
model-compute 1.21 for the same charged field.

TARGET-BLIND: no dose value enters; the factor depends only on the input spectrum
and on stopping-power physics.

    python3 -m paper.si_water_factor            # print the factor and the doses
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_source as ms  # noqa: E402

_K = 0.307075             # MeV cm2/mol, 4 pi N_A r_e^2 m_e c^2
_ME = 0.510999            # MeV
_MU = 931.494             # MeV, atomic mass unit
_MP = 938.272             # MeV, proton
# (Z/A, mean excitation energy I in MeV), ICRU 37/49
MEDIA = {"water": (0.55509, 75.0e-6), "Si": (14.0 / 28.0855, 173.0e-6)}
_MEV_G_TO_UGY = 1.602176634e-10 * 1.0e6    # MeV/g -> uGy
ZMAX = 28


def bethe_mass_stopping(z: int, e_mevn: float, medium: str) -> float:
    """Bethe mass stopping power (MeV cm2/g) of a bare nucleus z at e_mevn MeV/n
    (no shell / density / Barkas terms -- they largely cancel in the Si/water
    ratio above 10 MeV/n)."""
    za, i_ex = MEDIA[medium]
    a = ms._MAT_AI[z - 1]
    m_n = _MP if z == 1 else _MU
    gamma = 1.0 + e_mevn / m_n
    b2 = 1.0 - 1.0 / gamma ** 2
    bg2 = b2 * gamma * gamma
    m_ion = m_n * a
    tmax = 2.0 * _ME * bg2 / (1.0 + 2.0 * gamma * _ME / m_ion + (_ME / m_ion) ** 2)
    log_term = 0.5 * math.log(2.0 * _ME * bg2 * tmax / i_ex ** 2) - b2
    return max(_K * z * z * za / b2 * log_term, 0.0)


def free_field_dose_ugy_h(medium: str, phi_mv: float = 400.0,
                          emin: float = 10.0, emax: float = None,
                          n: int = 3000) -> dict:
    """Primary charged absorbed dose rate (uGy/h) per element in `medium` under
    the 2 pi sr open sky, {z: rate}. Log-spaced trapezoid over [emin, emax] MeV/n."""
    emax = ms.GCR_EMAX_PER_NUC if emax is None else emax
    w = ms.matthia_W_for_phi(phi_mv)
    out = {}
    for z in range(1, ZMAX + 1):
        tot, e_prev, f_prev = 0.0, None, None
        for k in range(n):
            e = emin * (emax / emin) ** (k / (n - 1))
            f = ms.mattia_flux(z, e, w) * bethe_mass_stopping(z, e, medium)
            if e_prev is not None:
                tot += 0.5 * (f + f_prev) * (e - e_prev)
            e_prev, f_prev = e, f
        out[z] = tot * 2.0 * math.pi * _MEV_G_TO_UGY * 3600.0
    return out


def si_to_water_factor(phi_mv: float = 400.0) -> float:
    """D_water / D_Si for the tool's primary charged GCR field at phi_mv."""
    return (sum(free_field_dose_ugy_h("water", phi_mv).values())
            / sum(free_field_dose_ugy_h("Si", phi_mv).values()))


def main() -> int:
    phi = 400.0
    dw = sum(free_field_dose_ugy_h("water", phi).values())
    ds = sum(free_field_dose_ugy_h("Si", phi).values())
    print(f"phi = {phi:g} MV (W = {ms.matthia_W_for_phi(phi):g}), Z 1-{ZMAX}, "
          f"10 MeV/n - {ms.GCR_EMAX_PER_NUC:g} MeV/n, 2 pi sky, primary only")
    print(f"  water {dw:.2f} uGy/h   Si {ds:.2f} uGy/h   Si->water {dw / ds:.3f}")
    print("  (Zhang 2020 constant 1.30, CRaTER 1.33, Matthiae & Berger 2024 model 1.21)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
