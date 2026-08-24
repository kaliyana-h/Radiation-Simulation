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

import functools
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .bridge import RunResult, MAKE_SOURCE, SPEScenario, DEFAULT_BEAM_SPOT_CM
from .geometry import _outer_gauge_radius, _OUTER_GAUGE_FIXED_CM, _enclosing_radius_cm


def _gauge_corr(spec) -> float:
    """1/R^2 correction that restores fluence_outside to the fixed-gauge reference.

    OUTER_GAUGE_ANCHOR_CAL is pinned to the 3 m dome, whose OuterShell sits at
    exactly _OUTER_GAUGE_FIXED_CM. An oversized design grows the gauge
    (geometry._outer_gauge_radius) so it clears the walls instead of overlapping
    them; scalar fluence ~ N/area ~ 1/R^2, so fluence_outside then reads low by
    (rg/800)^2 and the normalised dose (dose ~ 1/fluence_outside) is inflated by
    the same factor. Multiplying by (800/rg)^2 undoes it, putting every design on
    the one source gauge the calibration assumes. It is exactly 1.0 whenever the
    gauge stays at the fixed radius (every design up to the workshop envelope,
    including the 3 m anchor) so CAL is untouched; it only bites the oversized
    designs whose gauge had to grow (e.g. tall cylinders). See memory
    [[outer-fluence-gauge-fix]]."""
    return (_OUTER_GAUGE_FIXED_CM / _outer_gauge_radius(spec)) ** 2


# Size-envelope guard (GCR outer gauge). The fixed 880 cm gauge -- and the +-4%
# cross-shape invariance the whole dose calibration rests on -- was validated only
# for designs that FIT inside it, i.e. gauge_corr == 1.0 (enclosing corner ~< 860 cm).
# An oversized design grows the gauge, dropping gauge_corr below 1.0. The 1/R^2
# correction (_gauge_corr) still restores the anchor, so the number is not wrong --
# but it now EXTRAPOLATES past the validated envelope, which deserves a caveat. These
# thresholds grade that extrapolation for the GUI, exactly as _spe_kernel_calibrated
# flags off-calibration walls on the shielding side. NB this flags the NORMALISATION
# extrapolation; a design that also pokes past the 900 cm illuminated footprint is
# physically under-sampled and is flagged separately by beam_footprint_flag.
GAUGE_CORR_MILD = 0.9      # <- gauge grew a little (enclosing corner ~> 908 cm)
GAUGE_CORR_STRONG = 0.7    # <- gauge grew a lot; treat the number as indicative


def gauge_size_flag(spec) -> str:
    """Envelope status of a design's GCR outer-gauge normalisation.

    'ok'     gauge at the fixed radius (gauge_corr ~ 1.0); inside the validated
             envelope, absolute dose is trustworthy.
    'mild'   gauge grew a little (gauge_corr in [0.7, 0.9)); the oversized design
             is a mild extrapolation -- treat the absolute number as slightly soft.
    'strong' gauge grew a lot (gauge_corr < 0.7); treat the number as indicative
             and anchor it against a <=6 m version of the same wall.
    """
    gc = _gauge_corr(spec)
    if gc >= GAUGE_CORR_MILD:
        return "ok"
    if gc >= GAUGE_CORR_STRONG:
        return "mild"
    return "strong"


# Illuminated-footprint guard (physics, not normalisation). Every GCR source disc is
# a parallel beam of finite radius DEFAULT_BEAM_SPOT_CM (900 cm) -- the lit footprint
# on the sky-dome. A habitat whose farthest solid corner exceeds that radius pokes
# outside the illuminated field: part of the wall receives NO primaries from the
# grazing directions, so the crew dose is physically UNDER-sampled. This is a
# different failure from gauge_size_flag: _gauge_corr rescales the normalisation but
# cannot conjure primaries that were never launched, so this cannot be corrected away
# -- only flagged. Threshold left of the hard 900 cm edge so a design brushing the
# rim is caught before it silently loses flux. Feeds a GUI banner (never a hard block:
# the run still produces a conservative-leaning lower bound the students can see).
BEAM_FOOTPRINT_MARGIN_CM = 30.0      # amber band just inside the 900 cm rim


def beam_footprint_flag(spec) -> str:
    """Whether a design fits inside the 900 cm illuminated GCR footprint.

    'ok'      enclosing corner comfortably inside the lit disc -- fully sampled.
    'marginal' within BEAM_FOOTPRINT_MARGIN_CM of the 900 cm rim; grazing directions
               start to clip, dose begins to read low. Trust it as a lower bound.
    'over'    enclosing corner beyond 900 cm; the wall extends past the illuminated
               field and the dose is under-sampled (biased low). Shrink the design or
               widen the source (bridge.DEFAULT_BEAM_SPOT_CM) and re-anchor.
    """
    enc = _enclosing_radius_cm(spec)
    if enc <= DEFAULT_BEAM_SPOT_CM - BEAM_FOOTPRINT_MARGIN_CM:
        return "ok"
    if enc <= DEFAULT_BEAM_SPOT_CM:
        return "marginal"
    return "over"

E0_PROTON = 938.272            # MeV, proton rest energy
SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.0

# --- GCR flux absolute calibration ---------------------------------------
# The bare Usoskin-2005 LIS integral (2*pi upper hemisphere, >=10 MeV/nuc, phi=400 MV)
# yields ~6.4 protons/cm^2/s, i.e. a ~12.8/cm^2/s 4*pi-equivalent -- about 3.2x the
# canonical solar-minimum omnidirectional GCR proton flux of ~4/cm^2/s. On the lunar
# surface the Moon blocks the lower hemisphere, so the incident scalar fluence the
# habitat actually sees is ~half the free-space 4*pi value: ~2 protons/cm^2/s. We
# calibrate the whole spectrum to that reference at phi=400 MV; the factor (~0.31) is
# energy- and species-independent, so it rescales the absolute dose without touching
# the spectral shape, the per-species ratios, or the simulated transport. This is the
# single lever that brings absorbed dose and dose-equivalent into the measured
# lunar-surface range (see git history / the dose-calibration investigation).
GCR_PROTON_FLUX_REF_2PI = 2.0   # protons/cm^2/s, solar-min lunar surface, phi=400 MV
GCR_CALIB_PHI_MV = 400.0        # reference modulation the calibration is pinned to

# --- Outer-gauge re-anchor -----------------------------------------------
# fluence_outside is scored on the OuterShell hemisphere gauge OUTSIDE all walls.
# Every launched primary crosses that shell once heading in (before it reaches the
# habitat), so the crossing count is fixed = primaries launched, and the scalar
# fluence tracklength/volume ~ N/area ~ 1/R^2. The gauge is therefore a pure
# property of the SOURCE, not the habitat -- but only if its radius is the same in
# every run. Sizing it per-design (to hug each shape) made bigger habitats read a
# lower fluence and thus an INFLATED normalised dose: a 1/R^2 artifact that broke
# cross-shape comparison (dome vs cyl vs quonset diverged -47%/-66%). geometry.py
# now pins the gauge to one fixed radius (_OUTER_GAUGE_FIXED_CM), which reads
# shape-invariant to +-3% (dome/cyl/quonset), i.e. a clean normalization constant.
#
# Changing the gauge radius rescales the ABSOLUTE fluence_outside, so the dose
# normalisation (dose ~ 1/fluence_outside) shifts by the same factor. This single
# constant re-anchors it so the 3 m dome reproduces its Chang'E-4-preserving
# absorbed dose (13.2 uGy/h at 22 g/cm^2). Multiplying the absorbed-dose RATE by it
# makes every shape normalise by the same reference incident fluence -> fair
# comparison. Because it scales absorbed dose and dose-eq identically, effective Q
# (a ratio) is untouched.
#
# Derivation (two stages, both 3 m dome, which is fully illuminated at every
# BeamSpot so its physical field is invariant -- only the source tiling changes):
#   Stage 1 (fixed-800 gauge, BeamSpot=500): CAL = 2.8554e-5 / 1.11454e-4 = 0.25620
#     = (fixed R=800 dome fo) / (old per-design-gauge dome fo).
#   Stage 2 (BeamSpot 500 -> 900, 2026-07-27): widening every design's beam disc to
#     a fixed 900 cm so large habitats are fully lit rescales the source density.
#     The 3 m dome's normalised dose (skin/fo) must stay identical, so CAL scales by
#     N_500/N_900 = 1.7933e-7 / 3.9997e-8 = 4.4836.  CAL = 0.25620 * 4.4836 = 1.14867.
#   Stage 3 (gauge 800 -> 880 cm, 2026-08-10): moving the FIXED gauge outward lowers
#     the reference incident fluence by the very 1/R^2 law _gauge_corr already trusts
#     for grown gauges, so CAL scales by (800/880)^2 to hold the 3 m dome's dose
#     EXACTLY constant. This is an internal re-anchor, not a new calibration: every
#     design reads the identical dose it did at 800 (fitting designs keep gauge_corr
#     == 1.0 with the reference simply moved; grown designs cancel algebraically),
#     and the only change is that the mild/strong extrapolation envelope now widens
#     to enclosing corners ~< 908 cm. Purely geometric -- no dose target enters.
# See memory [[outer-fluence-gauge-fix]] and [[gcr-source-module]].
OUTER_GAUGE_ANCHOR_CAL = 0.25620 * (1.7933e-7 / 3.9997e-8) * (800.0 / 880.0) ** 2   # = 0.94932

# Mean field quality factor for a GCR-dominated field behind modest shielding.
# Literature mean Q for the deep-space/surface GCR field is ~2-6 (rises as
# shielding hardens the secondary neutron component). 3.5 is a defensible
# mid-range default; override per design once the particle split exists.
DEFAULT_QUALITY_FACTOR = 3.5

# An SPE field is protons (plus their secondaries), so its mean quality factor is
# much lower than the HZE-laden GCR field -- close to 1 for the bare beam, a
# little above once the wall breeds secondary neutrons. Used only as a fallback:
# when the LET-weighted ICRP-60 Q(L) scorer is present the effective Q emerges
# from it (H/D), exactly as for GCR.
DEFAULT_SPE_QUALITY_FACTOR = 1.5

# ----------------------------------------------------------------------
# GCR elemental composition (the heavy-ion field, not just protons)
# ----------------------------------------------------------------------
# Protons are ~87% of GCR *by number* but carry only a fraction of the dose:
# alpha particles and the heavy (HZE) ions deposit energy ~ Z^2 per particle, so
# they dominate the dose-equivalent despite their low abundance. Simulating
# protons alone therefore *underestimates* the absolute dose by a large factor.
#
# We represent the full Z=1..28 field by five characteristic ions, each standing
# in for a group of neighbouring elements whose number fluxes are summed into one
# representative abundance (relative to H=1, per nucleon, near solar minimum).
# The spectral SHAPE is the force-field-modulated proton LIS scaled per nucleon
# (the standard first-order "all species share the per-nucleon spectrum" ansatz,
# with species-dependent modulation Phi=(Z/A)*phi); only the integral abundance
# and the ion identity (for transport/fragmentation in the wall) differ. Each
# species is transported in its OWN run and normalised by its OWN real flux, then
# the dose rates are summed -- so the wall's species-dependent shielding and
# fragmentation response is captured, not approximated by a flat scale factor.
#
# Abundances are representative group values (order-of-magnitude); refining them
# (and adding measured heavy-ion LIS spectra) is a documented future upgrade.
GCR_COMPOSITION = [
    # name, TOPAS particle,        Z,  A,   abundance (rel. H), group represented
    ("H",  "proton",             1,  1,  1.0,    "Z=1"),
    ("He", "alpha",              2,  4,  0.105,  "Z=2"),
    ("C",  "GenericIon(6,12)",   6,  12, 0.033,  "Z=3-9  (CNO + light)"),
    ("Si", "GenericIon(14,28)",  14, 28, 0.012,  "Z=10-20 (Ne-Ca)"),
    ("Fe", "GenericIon(26,56)",  26, 56, 0.0020, "Z=21-28 (Fe peak)"),
]

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


def _raw_gcr_integral(z: int = 1, a: int = 1, phi_MV: float = 400.0,
                      emin_mev: float = 10.0, emax_mev: Optional[float] = None,
                      n: int = 4000) -> float:
    """Uncalibrated upper-hemisphere scalar fluence rate of ONE GCR species at
    abundance 1.0 (raw LIS-shape integral, before absolute normalization).

    Integrates the force-field-modulated per-nucleon LIS spectrum (proton shape,
    Usoskin 2005, with species modulation Phi=(Z/A)*phi) over per-nucleon energy
    and the 2*pi-sr upper hemisphere. Returns particles /cm^2/s for a species of
    relative abundance 1 -- multiply by the species abundance for its real flux.
    Mirrors make_source.gcr_spectrum so the simulated and real spectra match.

    This is the bare model integral; callers should use _gcr_integral, which
    applies the absolute flux calibration (see _calibration_factor)."""
    ms = _load_make_source()
    if emax_mev is None:
        emax_mev = ms.GCR_EMAX_PER_NUC   # shared per-nucleon ceiling (matches source)
    integral_m2_sr = 0.0       # /(m^2 s sr)
    Phi = (z / a) * phi_MV     # effective modulation in MeV/nuc for this species
    T_prev = j_prev = None
    for k in range(n):
        T = emin_mev * (emax_mev / emin_mev) ** (k / (n - 1))   # per-nucleon MeV
        Tm = T + Phi
        factor = (T * (T + 2 * E0_PROTON)) / (Tm * (Tm + 2 * E0_PROTON))
        j = ms.lis_proton(Tm / 1000.0) * factor                # /(m^2 s sr GeV)
        if T_prev is not None:
            dT_gev = (T - T_prev) / 1000.0
            integral_m2_sr += 0.5 * (j + j_prev) * dT_gev
        T_prev, j_prev = T, j
    integral_cm2_sr = integral_m2_sr / 1.0e4                    # -> /(cm^2 s sr)
    # scalar fluence from isotropic upper hemisphere = intensity * 2*pi sr
    return 2.0 * math.pi * integral_cm2_sr                      # /cm^2/s


@functools.lru_cache(maxsize=1)
def _calibration_factor() -> float:
    """Energy- and species-independent scale that pins the raw LIS integral to the
    canonical solar-minimum GCR proton flux on the lunar surface.

    The bare model overstates the absolute proton flux by ~3.2x (it gives
    ~6.4/cm^2/s 2*pi at phi=400 MV vs the ~2/cm^2/s reference). Because the factor
    is a single multiplicative constant, it rescales every species' absolute flux
    -- and hence absorbed dose and dose-equivalent -- without altering the spectral
    shape, the per-species ratios, or anything in the simulated transport. Pinned at
    GCR_CALIB_PHI_MV so the spectrum still modulates correctly for other phi."""
    return GCR_PROTON_FLUX_REF_2PI / _raw_gcr_integral(1, 1, GCR_CALIB_PHI_MV)


def _gcr_integral(z: int = 1, a: int = 1, phi_MV: float = 400.0,
                  emin_mev: float = 10.0, emax_mev: Optional[float] = None,
                  n: int = 4000) -> float:
    """Calibrated upper-hemisphere scalar fluence rate of ONE GCR species at
    abundance 1.0 (/cm^2/s). Raw LIS integral times the absolute flux calibration."""
    return _calibration_factor() * _raw_gcr_integral(z, a, phi_MV, emin_mev, emax_mev, n)


def gcr_scalar_fluence_rate(phi_MV: float = 400.0,
                            emin_mev: float = 10.0, emax_mev: Optional[float] = None,
                            n: int = 4000) -> float:
    """Upper-hemisphere scalar fluence rate of GCR protons at the lunar surface
    (protons /cm^2/s). Thin wrapper over _gcr_integral for Z=A=1."""
    return _gcr_integral(1, 1, phi_MV, emin_mev, emax_mev, n)


def gcr_species_fluence_rate(z: int, a: int, abundance: float,
                             phi_MV: float = 400.0) -> float:
    """Real scalar fluence rate (/cm^2/s) of a GCR ion of charge Z, mass A and
    number abundance (relative to H), at the lunar surface."""
    return abundance * _gcr_integral(z, a, phi_MV)


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
    rel_err: Optional[float] = None         # statistical rel. error of the dose rate
    contributions: Optional[list] = None    # per-species dose-rate breakdown (composition)
    neutron_fraction: Optional[float] = None  # fraction of H carried by wall-bred secondary neutrons
    regime: Optional[str] = None            # None=flood MC path; "thinwall"=phantom-matched kernel fold

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


def _doseeq_attr(skin: bool, qf: str) -> str:
    """Name of the RunResult/ConvergedResult dose-equivalent attribute for the
    chosen scorer target (skin lining vs central phantom) and quality-factor model.
    qf="icrp" -> ICRP-60 Q(L) scorer; qf="nasa" -> NASA/Cucinotta Q twin. Both are
    scored on the SAME transport, so switching qf only re-reads a different Sv
    column -- the emergent Q is computed per design, never scaled."""
    if qf not in ("icrp", "nasa"):
        raise ValueError(f"qf must be 'icrp' or 'nasa', got {qf!r}")
    suffix = "_nasa_sv" if qf == "nasa" else "_sv"
    return ("skin_doseeq" if skin else "phantom_doseeq") + suffix


def assess(result: RunResult,
           mission_days: float = 365.0,
           quality_factor: float = DEFAULT_QUALITY_FACTOR,
           phi_MV: Optional[float] = None,
           skin: bool = False,
           qf: str = "icrp") -> Optional[DoseAssessment]:
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
    gc = _gauge_corr(result.spec)
    dose_rate_gy_s = dose_gy * real_flux / sim_fluence_cm2 * OUTER_GAUGE_ANCHOR_CAL * gc

    # Prefer the emergent ICRP-60 Q(L) from the matching LET-weighted scorer (H/D)
    # over the flat default. Both targets now carry their own dose-eq scorer, so
    # the central phantom's (harder, HZE-stopping) field gets its own Q instead of
    # the flat field value.
    eff_q = quality_factor
    doseeq_sv = getattr(result, _doseeq_attr(skin, qf), None)
    if doseeq_sv is not None and dose_gy:
        eff_q = doseeq_sv / dose_gy

    return DoseAssessment(
        real_flux_cm2_s=real_flux,
        sim_fluence_cm2=sim_fluence_cm2,
        dose_rate_gy_s=dose_rate_gy_s,
        quality_factor=eff_q,
        mission_days=mission_days,
        neutron_fraction=getattr(result, "neutron_doseeq_fraction", None),
    )


def assess_composition(species_results: list,
                       mission_days: float = 365.0,
                       quality_factor: float = DEFAULT_QUALITY_FACTOR,
                       phi_MV: float = 400.0,
                       skin: bool = False,
                       qf: str = "icrp") -> Optional[DoseAssessment]:
    """Sum the per-species crew dose over a GCR composition.

    species_results is a list of (species_tuple, result) pairs, where
    species_tuple == (name, particle, Z, A, abundance, group) (a GCR_COMPOSITION
    row) and result is that species' completed run (RunResult / ConvergedResult).

    Each species is normalised by its OWN real flux exactly as assess() does for
    protons -- dose_rate_i = D_i * Phi_i / F_i -- then the rates are summed. The
    quality factor maps the summed absorbed dose to dose-equivalent: when the
    matching LET-weighted dose-equivalent scorer is present (both the skin lining
    AND the central phantom now carry one), the per-species dose-equivalent is
    normalised exactly like the absorbed dose and the EFFECTIVE field Q emerges as
    (sum H / sum D) -- so HZE ions are weighted by their real high-LET quality, not
    a flat field value. Falls back to the supplied flat quality_factor only for
    older runs with no dose-eq CSV. Returns None if no species yielded a usable dose."""
    total_rate = 0.0
    total_eqrate = 0.0       # LET-weighted dose-equivalent rate (Sv/s)
    total_neu_eqrate = 0.0   # neutron-lineage part of that dose-equivalent rate
    neu_eqrate_denom = 0.0   # dose-eq rate summed only over species that reported a fraction
    total_flux = 0.0
    var = 0.0                # sum of (rate_i * relerr_i)^2  for combined error
    have_doseeq = True       # both targets carry a dose-eq scorer; cleared if any species lacks it
    have_relerr = True       # every species has >=2 batches; cleared if any lacks an error
    contributions = []
    sim_fluence_used = 0.0
    for sp, result in species_results:
        name, particle, z, a, abundance, group = sp
        dose_gy = result.skin_dose_gy if skin else result.dose_gy
        if dose_gy is None or not result.fluence_outside:
            continue
        flux_i = gcr_species_fluence_rate(z, a, abundance, phi_MV)
        sim_fluence_cm2 = result.fluence_outside * 100.0      # /mm^2 -> /cm^2
        gc = _gauge_corr(result.spec)
        rate_i = dose_gy * flux_i / sim_fluence_cm2 * OUTER_GAUGE_ANCHOR_CAL * gc
        relerr_raw = (result.skin_dose_rel_err if skin else result.dose_rel_err)
        # A species with <2 batches has no error estimate. That is UNKNOWN, not
        # zero: folding it in as 0.0 would let it pose as a perfectly-converged
        # zero-variance term and understate the combined error (see rel_err below).
        have_relerr = have_relerr and relerr_raw is not None
        relerr_i = relerr_raw or 0.0
        total_rate += rate_i
        total_flux += flux_i
        var += (rate_i * relerr_i) ** 2
        sim_fluence_used = sim_fluence_cm2                    # representative (protons)
        # LET-weighted dose-equivalent rate from the matching scorer (ICRP-60 Q(L)
        # or the NASA/Cucinotta Q twin, per `qf`).
        doseeq_sv = getattr(result, _doseeq_attr(skin, qf), None)
        eqrate_i = q_i = None
        if doseeq_sv is not None:
            eqrate_i = doseeq_sv * flux_i / sim_fluence_cm2 * OUTER_GAUGE_ANCHOR_CAL * gc
            total_eqrate += eqrate_i
            q_i = eqrate_i / rate_i if rate_i else None
            # dose-weight this species' neutron fraction by its dose-eq contribution
            # (a heavy ion at 20% neutron fraction weighs by how much H it carries)
            f_i = getattr(result, "neutron_doseeq_fraction", None)
            if f_i is not None:
                total_neu_eqrate += f_i * eqrate_i
                neu_eqrate_denom += eqrate_i
        else:
            have_doseeq = False                              # a species is missing it
        contributions.append({
            "species": name, "particle": particle, "group": group,
            "flux_cm2_s": flux_i, "dose_rate_gy_s": rate_i,
            "doseeq_rate_sv_s": eqrate_i, "quality_factor": q_i,
            "rel_err": relerr_i,
        })
    if not contributions:
        return None
    # fraction of the absorbed dose carried by each species (for display)
    for c in contributions:
        c["dose_fraction"] = c["dose_rate_gy_s"] / total_rate if total_rate else 0.0
    # None -- not 0.0 -- while any species still lacks an error estimate, so a
    # convergence loop watching this value reads "unknown" and keeps going rather
    # than stopping on an apparently perfect result built from no statistics.
    rel_err = (var ** 0.5 / total_rate) if (have_relerr and total_rate) else None
    # Effective field Q emerges from the LET-weighted scorer (sum H / sum D);
    # fall back to the flat default when the dose-eq data is unavailable.
    eff_q = (total_eqrate / total_rate
             if have_doseeq and total_eqrate > 0 and total_rate else quality_factor)
    # Combined secondary-neutron dose fraction: dose-equivalent-weighted mean of the
    # per-species fractions (numerator and denominator summed over the same species,
    # so a species missing the scorer simply drops out of both).
    neutron_fraction = (total_neu_eqrate / neu_eqrate_denom
                        if neu_eqrate_denom > 0 else None)
    return DoseAssessment(
        real_flux_cm2_s=total_flux,
        sim_fluence_cm2=sim_fluence_used,
        dose_rate_gy_s=total_rate,
        quality_factor=eff_q,
        mission_days=mission_days,
        rel_err=rel_err,
        contributions=contributions,
        neutron_fraction=neutron_fraction,
    )


# ----------------------------------------------------------------------
# Thin-wall phantom-matched GCR fold  (direct-transmission regime)
# ----------------------------------------------------------------------
# The flood-MC normalisation (OUTER_GAUGE_ANCHOR_CAL * _gauge_corr) bathes the crew
# in a BROAD wall-bred secondary shower -- correct behind thick shielding but a
# spatial "blackout" for a THIN wall, where the crew dose is instead the direct
# transmission of the primary column overhead (a 7.5 mm Al dome read an unphysical
# 3531 mSv/yr under the flood path). Task #17 mapped the crossover: BELOW ~19 g/cm^2
# areal density the phantom-matched narrow beam aimed at the crew is the valid regime;
# above it the broad shower dominates and the two paths meet at ~130 mSv/yr. This
# kernel serves the thin regime by folding a precomputed response R(E)=D_organ/Phi_ff
# (built by dense monoenergetic phantom-matched transport at three ALUMINIUM areal
# densities, per species per organ-shell) against the TRUE free-field GCR flux. It uses
# NO gauge cal and NO 1/R^2 correction -- the only scale is the legitimate free-field
# flux anchor (_calibration_factor), identical to the GCR flux the flood path already
# trusts. Folded per anchor then interpolated in areal density; no per-design MC. Only
# absorbed (D) and ICRP-60 (I) are carried, so the thin-wall headline is ICRP-60
# effective dose. See data/gcr_thinwall_kernel.json and memory
# [[variance-reduction-kernel]] / [[thin-shield-bragg-limit]].
# Per-calibration-material thin-wall kernels. The aluminium kernel is the committed
# reference; other materials (the EVA laminate) ship their own kernel built by the
# same offline harness (lunarsim.kernel_gen) and validated against the Al one in the
# thin-wall regime. A material whose kernel file is not present falls back to the
# aluminium kernel and is flagged indicative (see _gcr_thinwall_calibrated), so the
# tool runs correctly whether or not the EVA kernel has been shipped yet.
_GCR_THINWALL_KERNELS = {
    "aluminium": Path(__file__).with_name("data") / "gcr_thinwall_kernel.json",
    "evasuit":   Path(__file__).with_name("data") / "gcr_thinwall_kernel_eva.json",
}

# Wall materials that fold against each material-specific kernel. The EVA kernel is
# built through the EVASuit laminate and absorbs the thin LCVG liner via areal
# density, so both suit layers count as its calibration family.
_GCR_KERNEL_FAMILIES = {"evasuit": ("evasuit", "lcvg")}


def _has_gcr_kernel(material: str) -> bool:
    path = _GCR_THINWALL_KERNELS.get(material)
    return bool(path and path.exists())


@functools.lru_cache(maxsize=None)
def _load_gcr_thinwall_kernel(material: str = "aluminium") -> dict:
    """Thin-wall GCR response kernel for a calibration material. Falls back to the
    aluminium kernel when the requested material has no kernel file yet; callers gate
    whether that fallback counts as calibrated (see _gcr_thinwall_calibrated)."""
    path = _GCR_THINWALL_KERNELS.get(material) or _GCR_THINWALL_KERNELS["aluminium"]
    if not path.exists():
        path = _GCR_THINWALL_KERNELS["aluminium"]
    with open(path) as fh:
        return json.load(fh)


def _gcr_calibration_material(spec) -> str:
    """Which thin-wall kernel this design folds against. A suit-laminate wall uses the
    EVA kernel once it is shipped; every other wall (and any suit design where the EVA
    kernel is still absent) folds against the aluminium kernel."""
    from .spec import SUIT_MATERIALS
    if (any(w.material in SUIT_MATERIALS for w in spec.walls)
            and _has_gcr_kernel("evasuit")):
        return "evasuit"
    return "aluminium"


def _calibration_areal_fraction(spec, material: str) -> float:
    """Fraction of the wall's areal density belonging to the kernel's calibration
    family (aluminium; or the suit laminate for the EVA kernel, which folds LCVG in via
    areal density). 1.0 when the family fills the wall; lower for mixed walls, which
    still fold but are flagged indicative (the R(E) transport was measured through the
    calibration material)."""
    from .spec import MATERIALS
    family = _GCR_KERNEL_FAMILIES.get(material, (material,))
    tot = cal = 0.0
    for w in spec.walls:
        ad = MATERIALS[w.material]["density"] * w.thickness_cm
        tot += ad
        if w.material in family:
            cal += ad
    return cal / tot if tot else 0.0


def _thinwall_fold_point(point: dict, phi_MV: float, organs: list) -> dict:
    """Fold ONE areal-density anchor against the true free-field GCR flux.

    R (=D/Phi_ff) [Gy*cm^2] x calibrated free-field species flux [/cm^2/s] -> dose rate,
    summed over the per-nucleon spectrum (each node carries the LIS integral over its
    geometric-mean energy bin, computed by the same _raw_gcr_integral the GCR flux model
    uses). Returns per-organ absorbed (Gy/s) & ICRP dose-eq (Sv/s) rates, the dose-eq
    variance, and each species' wT-weighted effective-dose and absorbed contribution."""
    calib = _calibration_factor()
    HI = {k: 0.0 for k, _w in organs}          # ICRP dose-eq rate  (Sv/s)
    HD = {k: 0.0 for k, _w in organs}          # absorbed dose rate (Gy/s)
    HIvar = {k: 0.0 for k, _w in organs}
    per_species = {}
    for name, sp in point["species"].items():
        z, a, ab = sp["z"], sp["a"], sp["abundance"]
        nd = sp["nodes_pernuc_mev"]
        edges = ([nd[0]] + [math.sqrt(nd[j] * nd[j + 1]) for j in range(len(nd) - 1)]
                 + [nd[-1]])
        fw = [ab * calib * _raw_gcr_integral(z, a, phi_MV, edges[j], edges[j + 1], n=400)
              for j in range(len(nd))]
        spE = spD = 0.0
        for k, wT in organs:
            Ri, Si, Rd = sp["R"][k]["I"], sp["Rsem"][k]["I"], sp["R"][k]["D"]
            hi = sum(Ri[j] * fw[j] for j in range(len(nd)))
            hd = sum(Rd[j] * fw[j] for j in range(len(nd)))
            var = sum((Si[j] * fw[j]) ** 2 for j in range(len(nd)))
            HI[k] += hi; HD[k] += hd; HIvar[k] += var
            spE += wT * hi; spD += wT * hd
        per_species[name] = {"E_sv_s": spE, "D_gy_s": spD, "flux_cm2_s": sum(fw),
                             "particle": sp["particle"], "group": sp["group"]}
    return {"HI": HI, "HD": HD, "HIvar": HIvar, "per_species": per_species}


def fold_gcr_thinwall(spec, phi_MV: float = 400.0, material: Optional[str] = None) -> dict:
    """Thin-wall phantom-matched GCR fold, interpolated to the design's areal density.

    Folds each aluminium anchor (pure arithmetic, no MC) at phi_MV, then interpolates
    the per-organ dose rates and per-species contributions between the bracketing
    anchors. Interpolation is geometric in log(areal density) — dose falls off with
    shielding as a power law, so a log-log fit tracks the physics far better than a
    straight line and lands continuously on the flood-validated ~130 mSv/yr at the
    ~19 g/cm2 gate rather than leaving a step there. (Zero-valued endpoints, e.g. a
    penetration-gated node, fall back to linear.) Below the thinnest / above the
    thickest anchor it clamps to the end anchor (the regime gate keeps callers inside
    the measured band). Returns the whole-body effective absorbed rate (Gy/s),
    effective dose-eq rate (Sv/s) and its standard error, per-organ rows, and the
    per-species breakdown."""
    material = material or _gcr_calibration_material(spec)
    K = _load_gcr_thinwall_kernel(material)
    organs = K["meta"]["organs"]
    pts = K["points"]
    grid = [p["wall_gcm2"] for p in pts]
    ad = spec.areal_density_gcm2()
    folded = [_thinwall_fold_point(p, phi_MV, organs) for p in pts]

    if ad <= grid[0]:
        lo = hi = 0; t = 0.0
    elif ad >= grid[-1]:
        lo = hi = len(grid) - 1; t = 0.0
    else:
        hi = next(j for j in range(len(grid)) if grid[j] >= ad)
        lo = hi - 1
        if grid[lo] > 0.0:
            t = (math.log(ad) - math.log(grid[lo])) / (math.log(grid[hi]) - math.log(grid[lo]))
        else:
            # Zero-areal anchor (the bare-phantom reference in the EVA grid): the
            # log map is singular at 0, so bracket this cell linearly in areal density.
            t = (ad - grid[lo]) / (grid[hi] - grid[lo])

    def ip(fa, fb):
        if fa > 0.0 and fb > 0.0:
            return fa * (fb / fa) ** t
        return fa + t * (fb - fa)

    D_eff = E = Evar = 0.0
    rows = []
    for k, wT in organs:
        Hd = ip(folded[lo]["HD"][k], folded[hi]["HD"][k])
        Hi = ip(folded[lo]["HI"][k], folded[hi]["HI"][k])
        Hv = ip(folded[lo]["HIvar"][k], folded[hi]["HIvar"][k])
        D_eff += wT * Hd; E += wT * Hi; Evar += (wT ** 2) * Hv
        rows.append({"shell": k, "wT": wT, "absorbed_gy_s": Hd, "doseeq_sv_s": Hi})

    per_species = {}
    names = folded[lo]["per_species"].keys()
    for name in names:
        a0, a1 = folded[lo]["per_species"][name], folded[hi]["per_species"][name]
        per_species[name] = {
            "E_sv_s": ip(a0["E_sv_s"], a1["E_sv_s"]),
            "D_gy_s": ip(a0["D_gy_s"], a1["D_gy_s"]),
            "flux_cm2_s": ip(a0["flux_cm2_s"], a1["flux_cm2_s"]),
            "particle": a0["particle"], "group": a0["group"]}

    return {"D_eff_gy_s": D_eff, "E_sv_s": E, "sem_E_sv_s": math.sqrt(Evar),
            "rows": rows, "per_species": per_species,
            "flux_cm2_s": sum(s["flux_cm2_s"] for s in per_species.values()),
            "areal_gcm2": ad}


def _gcr_thinwall_applies(spec) -> bool:
    """True when the design is in the thin-wall direct-transmission regime (areal
    density below the measured crossover), so the phantom-matched fold -- not the broad
    flood MC -- is the valid normalisation. Above the crossover the flood path stands."""
    K = _load_gcr_thinwall_kernel(_gcr_calibration_material(spec))
    return spec.areal_density_gcm2() < K["meta"]["crossover_gcm2"]


def areal_density_confidence(ad_gcm2: float) -> dict:
    """How much to trust the displayed GCR effective-dose headline at this wall
    areal density, from the measured OLTARIS-Al cross-check map (memory
    crossover-discontinuity). Trust is governed by where the design sits relative
    to the ~19 g/cm^2 kernel<->flood crossover:

      * ad >= 40    -> HIGH: the flood MC path, OLTARIS-validated at depth
                       (skin-ICRP ~1.1x of OLTARIS at 50 g/cm^2).
      * ad <= 12    -> MEDIUM: thin-wall kernel regime, close but OPTIMISTIC
                       (~0.8x of OLTARIS at 10 g/cm^2) -- may under-warn.
      * 12 < ad < 40 -> LOW: straddles the crossover cliff and is *least* certain
                       right at the 19 g/cm^2 gate (kernel 0.52x below vs flood
                       2.6x above OLTARIS there); the headline can be ~2-3x off.

    Returns {level, label, message} for a GUI banner. Purely advisory -- it
    changes no computed dose. Boundaries are soft engineering cut points, not
    physics edges; treat a design near a boundary as the lower-confidence side."""
    if ad_gcm2 >= 40.0:
        return {"level": "high", "label": "validated regime",
                "message": ("Wall areal density is in the thick regime where the "
                            "flood MC is OLTARIS-validated (~1.1x). The headline "
                            "is trustworthy.")}
    if ad_gcm2 <= 12.0:
        return {"level": "medium", "label": "thin-wall (optimistic)",
                "message": ("Thin-wall kernel regime: reasonably close to OLTARIS "
                            "but OPTIMISTIC (~0.8x at 10 g/cm^2), so the true dose "
                            "may be somewhat higher than shown.")}
    return {"level": "low", "label": "crossover band",
            "message": ("Areal density straddles the ~19 g/cm^2 kernel/flood "
                        "crossover -- the least-certain band, where the headline "
                        "can be 2-3x off (optimistic below 19, conservative "
                        "above). Treat as indicative; a clearly thicker or "
                        "thinner wall gives a firmer number.")}


def _gcr_thinwall_calibrated(spec) -> bool:
    """True when the thin-wall fold sits squarely on its calibration: in-regime AND the
    wall is dominated by the chosen kernel's calibration material (aluminium, or the
    suit laminate when the EVA kernel is shipped -- the material the R(E) transport was
    measured through). A mixed / off-material thin wall still folds but is flagged
    indicative, mirroring _spe_kernel_calibrated on the shielding side."""
    material = _gcr_calibration_material(spec)
    return (_gcr_thinwall_applies(spec)
            and _calibration_areal_fraction(spec, material) >= 0.8)


def assess_gcr_thinwall(spec, mission_days: float = 365.0,
                        phi_MV: float = 400.0) -> Optional[DoseAssessment]:
    """Thin-wall phantom-matched effective dose as a DoseAssessment (drops into the
    same GCR display path as the flood assessment).

    The folded whole-body effective dose is carried by setting the assessment's
    absorbed-dose rate to the wT-weighted effective absorbed rate and its quality
    factor to the emergent E/D_eff, so annual_msv == the ICRP-60 effective dose and
    annual_mgy == the effective absorbed dose. regime='thinwall' marks it for the GUI;
    neutron_fraction is None (the thin-wall kernel carries no neutron-lineage twin)."""
    f = fold_gcr_thinwall(spec, phi_MV)
    D_eff, E = f["D_eff_gy_s"], f["E_sv_s"]
    if D_eff <= 0 or E <= 0:
        return None
    contributions = []
    for name, sp in f["per_species"].items():
        d, e = sp["D_gy_s"], sp["E_sv_s"]
        contributions.append({
            "species": name, "particle": sp["particle"], "group": sp["group"],
            "flux_cm2_s": sp["flux_cm2_s"], "dose_rate_gy_s": d,
            "doseeq_rate_sv_s": e, "quality_factor": (e / d) if d else None,
            "dose_fraction": d / D_eff if D_eff else 0.0, "rel_err": 0.0})
    return DoseAssessment(
        real_flux_cm2_s=f["flux_cm2_s"],
        sim_fluence_cm2=0.0,
        dose_rate_gy_s=D_eff,
        quality_factor=E / D_eff,
        mission_days=mission_days,
        rel_err=(f["sem_E_sv_s"] / E) if E else None,
        contributions=contributions,
        neutron_fraction=None,
        regime="thinwall",
    )


# ----------------------------------------------------------------------
# SPE proton response-kernel fold  (variance-reduced behind-shield dose)
# ----------------------------------------------------------------------
# Behind ~147 g/cm^2 of regolith only a few percent of even the hardest historical
# SPE penetrates (>~430 MeV), so a direct phantom MC is rare-tail-starved: the
# handful of protons that reach the crew land in one under-sampled energy bin and
# the deep-organ dose fluctuates wildly (a flukey node once read BFO=83 mSv). The
# fix is the OLTARIS/HZETRN response-function method: precompute, once, a per-organ
# proton fluence-to-dose response R(E) [Gy*cm^2] by dense monoenergetic transport
# through the shielded wall, then FOLD any event's proton spectrum against it. No
# per-event MC, no rare tail -- the response already integrates the transport. The
# kernel lives in data/spe_proton_kernel.json (17 dense just-penetrating nodes,
# 22 seeds); see memory [[thin-shield-bragg-limit]] and [[variance-reduction-kernel]].
_SPE_KERNEL_PATH = Path(__file__).with_name("data") / "spe_proton_kernel.json"


@functools.lru_cache(maxsize=1)
def _load_make_source():
    """Import make_source.py as a module (for SPE_EVENTS and spe_integral_fluence)."""
    spec = importlib.util.spec_from_file_location("make_source", str(MAKE_SOURCE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def _load_spe_kernel() -> dict:
    with open(_SPE_KERNEL_PATH) as fh:
        return json.load(fh)


def _spe_dNdE(E: float, r0: float, e0: Optional[float]) -> float:
    """Differential proton spectrum dN/dE on the energy grid.

    Default is King (1974) exp-in-RIGIDITY, dN/dR ~ exp(-R/R0); on the energy grid
    dN/dE = dN/dR * dR/dE = exp(-R/R0) * (E+mp)/R with R = sqrt(E(E+2mp)). The
    legacy exp-in-ENERGY form (dN/dE ~ exp(-E/E0)) is used only when e0 is given."""
    if e0 is not None:
        return math.exp(-E / e0)
    R = math.sqrt(E * (E + 2.0 * E0_PROTON))
    return math.exp(-R / r0) * (E + E0_PROTON) / R


def _spe_bin_fluence(event: str, r0: float, e0: Optional[float],
                     edges: list) -> list:
    """Event proton fluence [/cm^2] integrated into each kernel energy bin.

    The spectral SHAPE is normalised so its integral above 30 MeV equals the event's
    measured phi(>30 MeV) anchor (make_source.spe_integral_fluence), then integrated
    over the geometric-mean bin edges. This is the one-time event fluence per bin --
    the quantity R(E) is folded against."""
    ms = _load_make_source()
    lo, hi, n = 30.0, 25000.0, 40000
    grid = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    f = [_spe_dNdE(e, r0, e0) for e in grid]
    tot30 = sum(0.5 * (f[i] + f[i + 1]) * (grid[i + 1] - grid[i]) for i in range(n - 1))
    scale = ms.spe_integral_fluence(event) / tot30

    def integ(a: float, b: float) -> float:
        m = 600
        g = [a * (b / a) ** (i / (m - 1)) for i in range(m)]
        return scale * sum(
            0.5 * (_spe_dNdE(g[i], r0, e0) + _spe_dNdE(g[i + 1], r0, e0))
            * (g[i + 1] - g[i]) for i in range(m - 1))

    return [integ(edges[j], edges[j + 1]) for j in range(len(edges) - 1)]


def fold_spe(scenario: SPEScenario, spec) -> dict:
    """Fold an SPE proton spectrum against the shielded response kernel.

    Returns per-organ-shell {q: (dose_gy, sem_gy)} for q in D (absorbed),
    I (ICRP-60 dose-equivalent) and N (NASA/Cucinotta dose-equivalent), plus the
    penetrating fluence and the characteristic rigidity used. The kernel R(E) and
    its per-node MC error S(E) carry the CAL and the design's 1/R^2 gauge correction,
    exactly matching the GCR path's normalisation."""
    K = _load_spe_kernel()
    ms = _load_make_source()
    H = K["H"]
    N = H["nodes_mev"]
    edges = ([N[0]] + [math.sqrt(N[j] * N[j + 1]) for j in range(len(N) - 1)]
             + [N[-1]])
    e0 = getattr(scenario, "e0_MeV", None)
    r0 = getattr(scenario, "r0_MV", None) or ms.SPE_EVENTS[scenario.event][0]
    pb = _spe_bin_fluence(scenario.event, r0, e0, edges)

    cal = OUTER_GAUGE_ANCHOR_CAL * _gauge_corr(spec)
    shells = {}
    for shell, _wT in K["meta"]["organs"]:
        R, S = H["R"][shell], H["Rsem"][shell]
        res = {}
        for q in ("D", "I", "N"):
            val = sum(R[q][j] * pb[j] for j in range(len(N))) * cal
            sem = math.sqrt(sum((S[q][j] * pb[j]) ** 2 for j in range(len(N)))) * cal
            res[q] = (val, sem)
        shells[shell] = res
    return {"shells": shells, "penetrating_cm2": sum(pb), "r0_MV": r0,
            "anchor_cm2": ms.spe_integral_fluence(scenario.event)}


def _spe_kernel_calibrated(spec) -> bool:
    """True when the design sits in the shielded regime the kernel was measured on.

    R(E) bakes in the transport through the calibration wall (~147 g/cm^2 regolith),
    so folding is faithful only for designs of comparable areal density. Off-
    calibration walls (much thinner/thicker) still get a number, but flagged."""
    K = _load_spe_kernel()
    cal_ad = K["meta"]["cal_wall_areal_g_cm2"]
    return abs(spec.areal_density_gcm2() - cal_ad) / cal_ad <= 0.25


# ----------------------------------------------------------------------
# Solar Particle Event assessment (acute, event-total)
# ----------------------------------------------------------------------
@dataclass
class SPEAssessment:
    """Total dose delivered by a single solar particle event.

    Unlike DoseAssessment (a chronic dose RATE integrated over a mission), this is
    a one-off event total: the whole event's proton fluence deposits its dose over
    hours, so the verdict is against the acute 30-day BFO limit, not an annual
    rate. Same fluence normalisation as the GCR path -- dose scales linearly with
    incident fluence -- but the real scale is the event's integral fluence rather
    than a flux times mission time."""
    event_dose_gy: float            # absorbed dose for the whole event (headline shell)
    quality_factor: float
    event_fluence_cm2: float        # real total event proton fluence used to normalise
    sim_fluence_cm2: float          # penetrating (>~430 MeV) fluence reaching the crew
    scenario_name: str = ""
    rel_err: Optional[float] = None
    # --- kernel-fold extras (depth-resolved acute dose) ------------------
    method: str = "kernel-fold"     # "kernel-fold" (folded R(E)) | "direct-mc" (legacy)
    calibrated: bool = True         # design sits in the kernel's shielded regime
    bfo_dose_gy: Optional[float] = None       # deep (~5 cm) blood-forming-organ dose
    bfo_quality_factor: Optional[float] = None
    skin_dose_gy: Optional[float] = None      # surface (~0.1 cm) shell dose
    skin_quality_factor: Optional[float] = None

    @property
    def event_mgy(self) -> float:
        return self.event_dose_gy * 1.0e3

    @property
    def event_msv(self) -> float:
        return self.event_dose_gy * self.quality_factor * 1.0e3

    @property
    def bfo_msv(self) -> Optional[float]:
        """Blood-forming-organ dose-equivalent -- the binding acute constraint for a
        shielded habitat (NASA 30-day BFO limit 250 mSv). None if not folded."""
        if self.bfo_dose_gy is None or self.bfo_quality_factor is None:
            return None
        return self.bfo_dose_gy * self.bfo_quality_factor * 1.0e3

    @property
    def skin_msv(self) -> Optional[float]:
        """Skin/surface dose-equivalent (NASA 30-day skin limit 1500 mSv)."""
        if self.skin_dose_gy is None or self.skin_quality_factor is None:
            return None
        return self.skin_dose_gy * self.skin_quality_factor * 1.0e3

    def fraction_of(self, limit_key: str = "nasa_30day") -> float:
        return self.event_msv / DOSE_LIMITS_MSV[limit_key]

    def verdict(self, limit_key: str = "nasa_30day") -> str:
        f = self.fraction_of(limit_key)
        if f < 0.5:
            return "SAFE"
        if f < 1.0:
            return "MARGINAL"
        return "EXCEEDS LIMIT"

    def summary(self, limit_key: str = "nasa_30day") -> dict:
        return {
            "scenario": self.scenario_name,
            "event_mGy": round(self.event_mgy, 2),
            "event_mSv": round(self.event_msv, 1),
            "quality_factor": round(self.quality_factor, 2),
            "event_fluence_cm2": self.event_fluence_cm2,
            "limit_mSv": DOSE_LIMITS_MSV[limit_key],
            "fraction_of_limit": round(self.fraction_of(limit_key), 3),
            "verdict": self.verdict(limit_key),
        }


def assess_spe(result: RunResult, scenario: SPEScenario,
               quality_factor: float = DEFAULT_SPE_QUALITY_FACTOR,
               skin: bool = True) -> Optional[SPEAssessment]:
    """Convert an SPE scenario into a total-event dose assessment by kernel fold.

    Behind thick regolith a direct phantom MC of an SPE is rare-tail-starved (the
    penetrating flux piles into one under-sampled energy bin), so the dose is taken
    instead from folding the event's proton spectrum against the precomputed shielded
    response kernel R(E) (see fold_spe / [[thin-shield-bragg-limit]]). The transport,
    depth structure and per-organ NASA/ICRP quality factors are all baked into R(E),
    so this needs no per-event reruns and is immune to the sampling starvation.

    `result` supplies only the design geometry (result.spec) for the 1/R^2 gauge
    correction; its (noisy) SPE dose is no longer used. skin=True (default, as the
    GUI calls it) makes the headline the SKIN shell; the deep blood-forming-organ
    dose -- the binding acute constraint -- is always populated (bfo_msv). Returns
    None only if the geometry is missing."""
    spec = getattr(result, "spec", None)
    if spec is None:
        return None

    fold = fold_spe(scenario, spec)
    shells = fold["shells"]
    sd, sd_sem = shells["skin"]["D"]
    sn, _ = shells["skin"]["N"]         # NASA/Cucinotta dose-equivalent
    dd, dd_sem = shells["deep"]["D"]
    dn, _ = shells["deep"]["N"]

    # headline shell follows the `skin` flag (backward compat); both shells are
    # always populated so the caller can show skin and BFO side by side.
    hd, hd_sem, hn = (sd, sd_sem, sn) if skin else (dd, dd_sem, dn)
    eff_q = (hn / hd) if hd else quality_factor
    rel_err = (hd_sem / hd) if hd else None

    return SPEAssessment(
        event_dose_gy=hd,
        quality_factor=eff_q,
        event_fluence_cm2=fold["anchor_cm2"],
        sim_fluence_cm2=fold["penetrating_cm2"],
        scenario_name=scenario.name,
        rel_err=rel_err,
        method="kernel-fold",
        calibrated=_spe_kernel_calibrated(spec),
        bfo_dose_gy=dd,
        bfo_quality_factor=(dn / dd) if dd else None,
        skin_dose_gy=sd,
        skin_quality_factor=(sn / sd) if sd else None,
    )


if __name__ == "__main__":
    print(f"GCR composition fluxes (phi=400 MV), /cm2/s:")
    tot = 0.0
    for name, particle, z, a, ab, group in GCR_COMPOSITION:
        f = gcr_species_fluence_rate(z, a, ab, 400.0)
        tot += f
        print(f"  {name:3s} ({particle:16s}) abundance={ab:<7g} flux={f:8.3f}  [{group}]")
    print(f"  total scalar fluence rate: {tot:.3f} /cm2/s")
    # sanity: integral GCR proton flux at solar min is a few /cm2/s
