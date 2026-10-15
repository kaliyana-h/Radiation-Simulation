#!/usr/bin/env python3
"""
make_source.py  --  GCR / SPE primary-source generator for the lunar
                    radiation-simulation tool (TOPAS parameter files).

Two source modes, answering the "justify your particle directions" critique:

  GCR  : isotropic field over the unobstructed *upper hemisphere* of the
         lunar surface.  Discretised into zenith rings x azimuth sources,
         each a parallel ("None" angular distribution) disc firing inward,
         following Dobynde & Guo (2021, JGR Planets, 10.1029/2021JE006930).

         Directional weighting is exact, not ad-hoc: for an isotropic
         intensity crossing a flat surface, the flux from a zenith band
         [t1,t2] is proportional to (sin^2 t2 - sin^2 t1) (Lambert cosine
         law).  Choosing ring boundaries equally spaced in sin^2(theta)
         makes every ring carry EQUAL flux, so every source gets the same
         number of histories -- correct AND statistically uniform.

  SPE  : a single look-direction (the Sun) with a finite angular spread
         (a cone), i.e. NOT omnidirectional -- the correct shape for a
         solar particle event.

Energy spectrum (GCR): force-field-modulated local interstellar spectrum.
  LIS (protons), Burger et al. (2000) as parameterised by Usoskin et al. (2005).
  The parameterisation is a function of magnetic RIGIDITY P (GV), NOT kinetic
  energy; for a proton P = sqrt(T(T+2 m_p)) with T the per-nucleon kinetic energy:
      J_LIS = 1.9e4 * P^-2.78 / (1 + 0.4866 * P^-2.51)      [/(m^2 s sr GeV)]
  Force-field modulation at potential phi (MV), per-nucleon kinetic T (MeV):
      Phi  = (Z/A) * phi
      J(T) = J_LIS(T+Phi) * [T(T+2E0)] / [(T+Phi)(T+Phi+2E0)]
  Worst case == solar minimum == low phi (deep 2019/2020 minimum ~ 400 MV).

The source blocks reference the host file's d:Ge/BeamRadius and d:So/BeamSpot
(as the existing habitat files already do), so the generated gcr_source.txt
can be pulled into any habitat file with:  includeFile = gcr_source.txt

Usage:
  ./make_source.py                       # GCR, 5 rings x 8 az, phi=400 MV
  ./make_source.py --rings 6 --azimuth 12 --phi 450 --histories 40
  ./make_source.py --mode spe --spe-event aug1972 --spe-zenith 45 --spe-cone 20
  ./make_source.py --standalone --out gcr_demo.txt   # full runnable test file
"""

import argparse
import functools
import math
import os

E0_PROTON = 938.272  # MeV, proton rest energy per nucleon

# Per-nucleon kinetic-energy ceiling for the GCR spectrum (MeV/nucleon).
#
# TOPAS BeamEnergy is TOTAL kinetic energy = T_per_nuc * A, so an ion of mass A
# reaches A times this energy. That is the tension a single constant cannot
# resolve: dose scales per NUCLEON, but Geant4 cost scales with TOTAL energy. At
# 20 GeV/nuc, H is only 20 GeV total (cheap, and truncated well below where it
# stops mattering) while Fe (A=56) is already 1.12 TeV. A per-nucleon ceiling is
# thus the wrong SHAPE for the cap -- it is pinned here by the heaviest ion. At
# the old 1e5 cap, Si hit 2.8 TeV and Fe 5.6 TeV, whose FTFP_BERT_HP showers are
# intractable (a single primary failed to finish in 18 min and exhausted memory).
#
# This value COSTS DOSE: it is a compute compromise, not a free truncation.
# Measured on the 3 m dome (5 cm Al + 30 cm regolith), protons, 12 batches/arm,
# skin annual dose:
#     5 GeV/nuc -> 87.7 mSv | 20 GeV/nuc -> 107.2 mSv | 100 GeV/nuc -> 113.2 mSv
# So this ceiling under-reports proton dose by ~5.6% (4.4 sigma) against 100
# GeV/nuc, for 1.59x the wall time; shrinking increments put the asymptote ~7-8%
# above this value -- a known bias in the NON-conservative direction.
#
# The cause: dose is TAIL-DOMINATED. The 5-20 GeV/nuc slice is 4.27% of proton
# flux but ~22% of skin dose (~5x an average proton, each). Those primaries are
# minimum-ionizing, but they shower through 30 cm of regolith and the SHOWER is
# the dose -- flux fraction is NOT dose fraction, and reasoning from one to the
# other is off by ~5x here. So do NOT lower this to buy runtime (5 GeV/nuc costs
# 18% of dose to save 7% of wall), and do not add a wall-clock guard instead: it
# preferentially kills the slowest = highest-energy = most dose-dominant events,
# which is this same truncation with an undocumented cut point. Raise it PER
# SPECIES when compute allows -- H stays cheap at high ceilings, Fe does not.
#
# Used by BOTH the source generator and dosimetry._gcr_integral so the simulated
# and flux-normalisation spectra cannot drift.
#
# LUNARSIM_GCR_EMAX_PER_NUC overrides it for sensitivity studies (does lowering
# the ceiling to buy runtime change the dose?). Read from the environment rather
# than passed as a flag on purpose: make_source runs as a SUBPROCESS of the
# lunarsim bridge while dosimetry imports this module in the PARENT, and a
# subprocess inherits the parent's environment -- so one variable moves the
# sampled spectrum and the flux normalisation together and they cannot drift.
# A CLI flag would only reach the source and would silently desynchronise them.
GCR_EMAX_PER_NUC = float(os.environ.get("LUNARSIM_GCR_EMAX_PER_NUC", 2.0e4))  # 20 GeV/nucleon

# GCR spectral-SHAPE model for heavy ions. Two choices:
#
#   "pernuc"   (default, the validated baseline): every species borrows the
#              proton LIS shape, evaluated at its own per-nucleon modulated
#              energy Tm. This is the "universal in energy-per-nucleon" ansatz.
#
#   "rigidity" (per-element test): evaluate the proton LIS at the proton energy
#              that shares the ion's RIGIDITY, R = (A/Z)*sqrt(Tm(Tm+2E0)).
#              Rigidity -- not energy-per-nucleon -- is the variable in which the
#              GCR LIS is approximately universal and is the correct heliospheric
#              propagation variable (the basis of the ISO 15390 / Matthiae 2013
#              force-field models). Because A/Z ~ 2 for heavy ions, this shifts
#              their effective sampling energy up, and more at low energy than
#              high, HARDENING the normalised heavy-ion spectrum relative to the
#              proton-shape ansatz. Proton-preserving BY CONSTRUCTION: for Z=A=1,
#              R = sqrt(Tm(Tm+2E0)) and the rigidity-matched proton energy is
#              exactly Tm, so the proton baseline is byte-identical.
#
# CRITICAL -- this is a SOURCE-ONLY knob, unlike LUNARSIM_GCR_EMAX_PER_NUC which
# BOTH make_source and dosimetry read. It is read ONLY here, so dosimetry's flux
# normalisation stays on the "pernuc" integral and each species' TOTAL number
# flux is held frozen at its anchored (Chang'E-4-validated absorbed-dose) value.
# That is deliberate: the test redistributes a FIXED number of particles in
# energy according to the per-element shape and asks whether that alone moves the
# effective dose -- the iter-4 discipline (change the shape, hold the flux). It
# is a controlled probe, not the shipped model, so the source/normalisation
# "cannot drift" rule is intentionally relaxed here (flux is pinned by hand).
GCR_SHAPE = os.environ.get("LUNARSIM_GCR_SHAPE", "pernuc")

# LIS variable knob (reproducibility, NOT a physics choice). The Burger/Usoskin
# parameterisation is a function of magnetic RIGIDITY P (GV); "rigidity" (default,
# correct) converts kinetic energy T -> P before evaluating it. "energy" reproduces
# the legacy pre-fix behaviour that evaluated the form AT kinetic energy directly --
# kept ONLY so the LIS-fix before/after (paper/rerun harness) can regenerate the old
# spectrum for an apples-to-apples MC comparison. Leave it at "rigidity" for all real
# runs. Read here AND by dosimetry (via _load_make_source), so the sampled spectrum
# and the flux normalisation stay on the same form and cannot drift.
GCR_LIS_VAR = os.environ.get("LUNARSIM_GCR_LIS", "rigidity")

# Per-element GCR LIS model selector -- the thin-wall <Q> over-read fix.
#
#   "none"        (default, shipped baseline): every species borrows the proton
#                 LIS shape scaled per nucleon with species modulation
#                 Phi=(Z/A)*phi (the "proton-clone" ansatz). Byte-for-byte the
#                 validated behaviour. lis_proton + gcr_spectrum(shape=...) as before.
#
#   "matthia2013" (per-element fix): replace the proton-clone spectrum with the
#                 Matthiae et al. 2013 DLR "modified ISO" per-element GCR model
#                 (Adv. Space Res. 51(3) 329-338, doi:10.1016/j.asr.2012.09.022;
#                 coefficients verbatim from ssc-maire/CosRayModifiedISO). Each
#                 element Z=1-28 gets its OWN analytic spectrum; a single solar
#                 parameter W carries the modulation (force-field-NATIVE, unlike
#                 BON2020 which needs a Fokker-Planck solve -- see memory
#                 let-scorer-thin-qbar-rootcause). W is mapped from the tool's phi
#                 by matching the Matthiae proton SHAPE to the tool's own
#                 Burger/Usoskin+force-field proton spectrum (flux<->flux, NOT a
#                 dose target) so the switch is self-consistent at any phi.
#
# WHY this fixes the over-read (validated target-blind on the laptop,
# paper/mattia2013_lis_fold.py): the proton-clone ansatz over-weights every
# non-proton (modulation Phi=(Z/A)*phi under-modulates Z<A ions) AND carries
# heavy abundances ~3x too high, driving <Q>(0)=11.3 vs the arbiter 5.6. The
# Matthiae per-element LIS lands <Q>(0)=6.2, He/H=0.44 (measured 0.43), and the
# H/He dose split inside the Hong-2019 bands -- because the per-element spectra
# and relative abundances are physical, not proton-cloned.
#
# Read by BOTH make_source (beam shape + gcr_spectrum) AND dosimetry
# (_raw_gcr_integral flux normalisation) via dosimetry.set_perelement_model, so
# the sampled spectrum and the flux normalisation cannot drift (same discipline
# as GCR_LIS_VAR). The proton ABSOLUTE calibration is untouched: dosimetry pins
# H to the reference flux, and the per-element structure comes from the model's
# own coefficients.
GCR_PERELEMENT = os.environ.get("LUNARSIM_GCR_PERELEMENT", "none")


# --------------------------------------------------------------------------
# Matthiae et al. 2013 (DLR modified-ISO) per-element GCR LIS
# --------------------------------------------------------------------------
# Coefficients (Z-ordered, index Z-1) VERBATIM from the reference implementation
# ssc-maire/CosRayModifiedISO internalFunctions/pythonModifiedISO.py (the paper's
# Eq. set). Abundance is baked into C_I, so the per-element flux integrals ARE the
# physical relative abundances.
_MAT_AI = (1.0, 4.0, 6.9, 9.0, 10.8, 12.0, 14.0, 16.0, 19.0, 20.2,
           23.0, 24.3, 27.0, 28.1, 31.0, 32.1, 35.4, 39.9, 39.1, 40.1,
           44.9, 47.9, 50.9, 52.0, 54.9, 55.8, 58.9, 58.7)
_MAT_CI = (1.85e4, 3.69e3, 19.5, 17.7, 49.2, 103.0, 36.7, 87.4, 3.19, 16.4,
           4.43, 19.3, 4.17, 13.4, 1.15, 3.06, 1.30, 2.33, 1.87, 2.17,
           0.74, 2.63, 1.23, 2.12, 1.14, 9.32, 0.10, 0.48)
_MAT_GAMMAI = (2.74, 2.77, 2.82, 3.05, 2.96, 2.76, 2.89, 2.70, 2.82, 2.76,
               2.84, 2.70, 2.77, 2.66, 2.89, 2.71, 3.00, 2.93, 3.05, 2.77,
               2.97, 2.99, 2.94, 2.89, 2.74, 2.63, 2.63, 2.63)
_MAT_ALPHAI = (2.85, 3.12, 3.41, 4.30, 3.93, 3.18, 3.77, 3.11, 4.05, 3.11,
               3.14, 3.65, 3.46, 3.00, 4.04, 3.30, 4.40, 4.33, 4.49, 2.93,
               3.78, 3.79, 3.50, 3.28, 3.29, 3.01, 4.25, 3.52)
_MAT_EMIN = 10.0   # MeV/n -- model validity floor (ref impl raises below ~11)

# Representative-ion -> member-Z map. The MC transports 5 representative ions
# (H/He/C/Si/Fe); under Matthiae each representative carries the abundance-weighted
# SUM of its member elements' per-element spectra and the group-summed abundance,
# so the full 28-element model informs the beam shape and flux while the existing
# 5-species transport/kernel infrastructure (node grids, anchors, R values) is
# reused unchanged. A z not in this map is treated as the single element (z,).
MATTHIA_GROUP_ZS = {
    1:  (1,),
    2:  (2,),
    6:  tuple(range(3, 10)),    # Z=3-9  (CNO + light), rep C
    14: tuple(range(10, 21)),   # Z=10-20 (Ne-Ca),      rep Si
    26: tuple(range(21, 29)),   # Z=21-28 (Fe peak),    rep Fe
}


def mattia_flux(z, e_mevn, W):
    """Matthiae-2013 differential flux [particles cm^-2 s^-1 sr^-1 (MeV/n)^-1] for
    element z (1-28) at per-nucleon kinetic energy e_mevn (MeV/n) and solar
    modulation W (0-200; low W ~ solar min ~ high GCR). Verbatim model."""
    i = z - 1
    A = _MAT_AI[i]
    m = 0.938 if z == 1 else 0.939          # GeV/n per-nucleon rest mass
    c = _MAT_CI[i]
    alpha = _MAT_ALPHAI[i]
    gamm = _MAT_GAMMAI[i]
    x = 1e-3 * e_mevn                        # MeV/n -> GeV/n
    s = math.sqrt(x * (x + 2.0 * m))
    rigidity = (A / z) * s                   # GV
    beta = s / (x + m)
    R0 = 0.37 + 0.0003 * (W ** 1.45)
    delta = 0.02 * W + 4.7
    phi = c * (beta ** alpha) / (rigidity ** gamm) * (rigidity / (rigidity + R0)) ** delta
    return 1e-4 * phi * (A / z) * 1e-3 / beta


def _mattia_group_zs(rep_z):
    return MATTHIA_GROUP_ZS.get(rep_z, (rep_z,))


def mattia_group_flux(rep_z, e_mevn, W):
    """Sum of member-element Matthiae fluxes for a representative ion (per-nucleon
    energy e_mevn). For H/He this is the single element; for C/Si/Fe it is the
    abundance-weighted member sum (abundance is in each element's C_I)."""
    return sum(mattia_flux(z, e_mevn, W) for z in _mattia_group_zs(rep_z))


@functools.lru_cache(maxsize=256)
def _mattia_group_integral(rep_z, W, emin, emax, n):
    """Trapezoidal integral of the group Matthiae flux over per-nucleon energy
    [emin, emax] MeV/n (shape integral; relative units)."""
    lo = max(emin, _MAT_EMIN)
    if emax <= lo:
        return 0.0
    tot = 0.0
    tp = fp = None
    for k in range(n):
        e = lo * (emax / lo) ** (k / (n - 1))
        f = mattia_group_flux(rep_z, e, W)
        if tp is not None:
            tot += 0.5 * (f + fp) * (e - tp)
        tp, fp = e, f
    return tot


@functools.lru_cache(maxsize=64)
def matthia_W_for_phi(phi_MV):
    """Map the tool's force-field modulation phi (MV) to the Matthiae solar
    parameter W by matching the Matthiae proton spectral SHAPE to the tool's own
    Burger/Usoskin + force-field proton spectrum at phi (flux<->flux over
    100 MeV..10 GeV, NOT a dose target). Honours the active GCR_LIS_VAR form, so
    the match stays self-consistent with whichever engine is running. Returns the
    best-fit integer W in [0, 200]. (phi=400 MV -> W=4, consistent with phi=400
    ~ solar min and the Matthiae W<->Oulu relation giving W~1-5 at solar min.)"""
    es = [100.0 * (10000.0 / 100.0) ** (k / 20) for k in range(21)]
    def tool_p(e):                            # proton Phi=(1/1)*phi
        Tm = e + phi_MV
        factor = (e * (e + 2.0 * E0_PROTON)) / (Tm * (Tm + 2.0 * E0_PROTON))
        return lis_proton(Tm / 1000.0) * factor
    ltp = [math.log(tool_p(e)) for e in es]
    mtp = sum(ltp) / len(ltp)
    best_err, best_W = float("inf"), 0
    for Wi in range(201):
        lm = [math.log(mattia_flux(1, e, float(Wi))) for e in es]
        mm = sum(lm) / len(lm)
        err = sum(((lm[i] - mm) - (ltp[i] - mtp)) ** 2 for i in range(len(es)))
        if err < best_err:
            best_err, best_W = err, Wi
    return float(best_W)


def mattia_norm_integral(rep_z, phi_MV, emin_mev, emax_mev, n=400):
    """Matthiae group flux integral over [emin, emax] MeV/n, NORMALISED so that the
    FULL-range (10 MeV/n .. GCR_EMAX_PER_NUC) integral equals the PROTON full-range
    integral for every species. This preserves make_source/dosimetry's contract
    `real_flux = abundance * calib * raw_integral`: with abundance = the Matthiae
    group abundance rel. H, the per-species total lands at abundance*proton_flux and
    the per-energy-bin fractions follow the element's own Matthiae shape. The proton
    absolute calibration is therefore untouched; only the shape and relative
    abundances change. See GCR_PERELEMENT."""
    W = matthia_W_for_phi(phi_MV)
    full_z = _mattia_group_integral(rep_z, W, _MAT_EMIN, GCR_EMAX_PER_NUC, 4000)
    if full_z <= 0.0:
        return 0.0
    full_p = _mattia_group_integral(1, W, _MAT_EMIN, GCR_EMAX_PER_NUC, 4000)
    frac = _mattia_group_integral(rep_z, W, emin_mev, emax_mev, n) / full_z
    return frac * full_p


def mattia_group_abundance(rep_z, phi_MV):
    """Matthiae group abundance relative to H (ratio of full-range group flux
    integrals). This is the physical relative abundance that replaces the
    proton-clone GCR_COMPOSITION abundances when GCR_PERELEMENT='matthia2013'."""
    W = matthia_W_for_phi(phi_MV)
    full_z = _mattia_group_integral(rep_z, W, _MAT_EMIN, GCR_EMAX_PER_NUC, 4000)
    full_p = _mattia_group_integral(1, W, _MAT_EMIN, GCR_EMAX_PER_NUC, 4000)
    return full_z / full_p if full_p > 0 else 0.0


# --------------------------------------------------------------------------
# Energy spectra
# --------------------------------------------------------------------------
def lis_proton(T_GeV):
    """Local interstellar proton spectrum J_LIS at per-nucleon kinetic energy
    T (GeV), from the Burger et al. (2000) parameterisation given by Usoskin
    et al. (2005). That form is a function of magnetic RIGIDITY P (GV): for a
    proton P = sqrt(T(T + 2 m_p)), m_p = 0.938272 GeV. Evaluating it at kinetic
    energy directly (as earlier revisions did) over-weights the sub-GeV flux
    because P > T there; converting T -> P here makes the code match the cited
    source. Returns the differential intensity in /(m^2 s sr GeV). Both the
    source generator and dosimetry._raw_gcr_integral call this one function, so
    the sampled spectrum and the flux normalisation move together and cannot
    drift. The LUNARSIM_GCR_LIS knob selects "rigidity" (default, correct) or
    "energy" (legacy reproduction only). See paper/CODE_REFERENCES.md
    (LIS-variable verification)."""
    mp_GeV = E0_PROTON / 1000.0
    if GCR_LIS_VAR == "energy":                # legacy pre-fix form: evaluate AT T
        P_GV = T_GeV
    else:                                      # "rigidity" (default, correct): T -> P
        P_GV = math.sqrt(T_GeV * (T_GeV + 2.0 * mp_GeV))      # rigidity, GV
    return 1.9e4 * P_GV ** (-2.78) / (1.0 + 0.4866 * P_GV ** (-2.51))


def _rigidity_equiv_proton_MeV(Tm, z, a):
    """Proton kinetic energy (MeV) sharing the ion's rigidity at per-nucleon
    modulated energy Tm. R = (A/Z)*sqrt(Tm(Tm+2E0)); T_p = sqrt(R^2+E0^2)-E0.
    For a proton (z=a=1) this returns Tm exactly, preserving the baseline."""
    R = (a / z) * math.sqrt(Tm * (Tm + 2 * E0_PROTON))
    return math.sqrt(R * R + E0_PROTON * E0_PROTON) - E0_PROTON


def gcr_spectrum(phi_MV, z=1, a=1, emin=10.0, emax=GCR_EMAX_PER_NUC, n=28,
                 shape=None):
    """Force-field-modulated per-nucleon spectrum on a log energy grid.

    `shape` selects the heavy-ion spectral-shape model ("pernuc" default,
    "rigidity" for the per-element test); defaults to the GCR_SHAPE env knob.
    Returns (energies_MeV, weights) where weights are relative dN/dE
    (TOPAS normalises them internally)."""
    if shape is None:
        shape = GCR_SHAPE
    # Per-element Matthiae model (GCR_PERELEMENT='matthia2013') OVERRIDES the
    # proton-clone shape knob: each representative ion carries its group's summed
    # Matthiae spectrum, with W mapped from phi. The force-field factor/rigidity
    # path is bypassed -- Matthiae carries its own modulation (R0, delta via W).
    if GCR_PERELEMENT == "matthia2013":
        W = matthia_W_for_phi(phi_MV)
        energies, weights = [], []
        for k in range(n):
            T = emin * (emax / emin) ** (k / (n - 1))     # per-nucleon KE, MeV
            energies.append(T * a)        # TOPAS BeamEnergy = TOTAL kinetic energy
            weights.append(mattia_group_flux(z, T, W))
        wmax = max(weights)
        return energies, [w / wmax for w in weights]
    Phi = (z / a) * phi_MV  # effective modulation in MeV for this species
    energies, weights = [], []
    for k in range(n):
        T = emin * (emax / emin) ** (k / (n - 1))     # per-nucleon KE, MeV
        Tm = T + Phi
        factor = (T * (T + 2 * E0_PROTON)) / (Tm * (Tm + 2 * E0_PROTON))
        if shape == "rigidity":
            T_lis = _rigidity_equiv_proton_MeV(Tm, z, a)  # == Tm for protons
        else:
            T_lis = Tm
        j = lis_proton(T_lis / 1000.0) * factor
        energies.append(T * a)        # TOPAS BeamEnergy = TOTAL kinetic energy
        weights.append(j)
    wmax = max(weights)
    weights = [w / wmax for w in weights]             # scale to peak 1.0
    return energies, weights


def _proton_rigidity_MV(E_MeV):
    """Proton magnetic rigidity R = pc/|q| (MV) at kinetic energy E (MeV).
    R = sqrt(E (E + 2 mp)); for protons |q|=1 so R[MV] == pc[MeV]."""
    return math.sqrt(E_MeV * (E_MeV + 2.0 * E0_PROTON))


# Worst-case ("solar maximum") design SPE proton events.  A solar particle
# event peaks at SOLAR MAXIMUM (active Sun -> flares/CMEs) -- the opposite phase
# to the GCR background (worst at solar minimum).  Real SPE spectra are NOT
# exponential in ENERGY; the standard parameterisation (King 1974) is an
# exponential in magnetic RIGIDITY R:  dN/dR ~ exp(-R/R0), characterised by a
# single "spectral hardness" rigidity R0 (MV).  Converting to the energy grid
# TOPAS wants (dN/dE = dN/dR * dR/dE, dR/dE = (E+mp)/R):
#       dN/dE  ~  exp(-R/R0) * (E + mp) / R .
#
# Each event below is (R0_MV, phi_gt30_cm2, note) where phi_gt30 is the measured
# INTEGRAL proton fluence above 30 MeV (the standard shielding-design anchor).
# These are DESIGN-REFERENCE approximations of the canonical events -- soft,
# very-high-fluence (Aug 1972) through hard (Feb 1956) -- and should be checked
# against the primary fits (King 1974; Tylka & Dietrich 2009; Xapsos ESP model)
# before any DERIVED dose is quoted.  A single exponential-in-rigidity slightly
# UNDER-predicts the highest-energy tail of real events (which harden into a
# double power law / Band function); the hard events below bracket that.
#
# Which event is "worst" depends on shielding, and this is a genuine physics
# fork worth stating: behind THIN shielding / for SKIN + acute risk the soft,
# huge-fluence Aug 1972 dominates; behind THICK regolith (the tool's validated
# 147 g/cm2 case) the soft protons are stopped and a HARD event (Feb 1956)
# drives the residual deep-organ/BFO dose.  Default = aug1972 (the canonical
# crewed-mission worst case; the event that motivated SPE storm-shelter design).
SPE_EVENTS = {
    "aug1972": (100.0, 5.0e9,
                "Aug 1972 -- soft, extreme fluence; canonical crewed-mission worst case (King 1974)"),
    "oct1989": (150.0, 1.5e9,
                "Oct 1989 GLE series -- harder spectrum, large fluence"),
    "feb1956": (220.0, 1.0e9,
                "Feb 1956 (GLE 5) -- hardest modern event; worst behind thick shielding"),
}


def spe_spectrum(event="aug1972", emin=10.0, emax=3.0e3, n=28,
                 r0_MV=None, e0_MeV=None):
    """Worst-case (solar-maximum) SPE proton spectrum on a log energy grid.

    Default model: exponential in magnetic rigidity (King 1974 form) for a named
    design event in SPE_EVENTS -- dN/dE ~ exp(-R/R0)*(E+mp)/R.  Pass `r0_MV` to
    override the event's characteristic rigidity directly.  Pass `e0_MeV` to fall
    back to the LEGACY exponential-in-ENERGY placeholder (kept for reproducibility
    of earlier runs -- it is NOT physically faithful; prefer the rigidity model).

    Returns (energies_MeV, weights) with weights = relative dN/dE peak-scaled to
    1.0 (the SHAPE only, as TOPAS renormalises).  The event's absolute integral
    fluence (for turning scored dose into an event dose) is in SPE_EVENTS /
    spe_integral_fluence(); it is deliberately NOT folded into these weights."""
    energies, weights = [], []
    if e0_MeV is not None:                        # legacy exponential-in-energy
        for k in range(n):
            E = emin * (emax / emin) ** (k / (n - 1))
            energies.append(E)
            weights.append(math.exp(-E / e0_MeV))
    else:
        if r0_MV is None:
            if event not in SPE_EVENTS:
                raise ValueError("unknown SPE event %r; choose from %s"
                                 % (event, sorted(SPE_EVENTS)))
            r0_MV = SPE_EVENTS[event][0]
        for k in range(n):
            E = emin * (emax / emin) ** (k / (n - 1))
            R = _proton_rigidity_MV(E)
            energies.append(E)
            weights.append(math.exp(-R / r0_MV) * (E + E0_PROTON) / R)
    wmax = max(weights)
    return energies, [w / wmax for w in weights]


def spe_integral_fluence(event="aug1972"):
    """Measured integral proton fluence above 30 MeV (cm^-2) for a design event.
    This is the absolute normalisation the dose pipeline needs; the spectrum
    SHAPE from spe_spectrum() is peak-scaled and carries no absolute fluence."""
    return SPE_EVENTS[event][1]


# --------------------------------------------------------------------------
# Parameter-block emitters
# --------------------------------------------------------------------------
def _fmt_vec(values, fmt="{:.6g}"):
    return " ".join(fmt.format(v) for v in values)


def _spectrum_block(tag, energies, weights):
    n = len(energies)
    return [
        f'    s:So/{tag}/BeamEnergySpectrumType    = "Continuous"',
        f'    dv:So/{tag}/BeamEnergySpectrumValues  = {n} {_fmt_vec(energies)} MeV',
        f'    uv:So/{tag}/BeamEnergySpectrumWeights = {n} {_fmt_vec(weights)}',
    ]


def _ring_groups(idx, phi_deg, theta_deg, prefix):
    """Nested Azi(RotZ)->Aim(RotY)->Src(TransZ,RotX=180) firing inward."""
    az, aim, src = f"{prefix}Azi{idx}", f"{prefix}Aim{idx}", f"{prefix}Src{idx}"
    return [
        f'    s:Ge/{az}/Type   = "Group"',
        f'    s:Ge/{az}/Parent = "World"',
        f'    d:Ge/{az}/RotZ   = {phi_deg:.4f} deg',
        f'    s:Ge/{aim}/Type   = "Group"',
        f'    s:Ge/{aim}/Parent = "{az}"',
        f'    d:Ge/{aim}/RotY   = {theta_deg:.4f} deg',
        f'    s:Ge/{src}/Type   = "Group"',
        f'    s:Ge/{src}/Parent = "{aim}"',
        f'    d:Ge/{src}/TransZ = Ge/BeamRadius cm',
        f'    d:Ge/{src}/RotX   = 180.0 deg',
    ], src


def gcr_block(rings, azimuth, phi_MV, histories, particle="proton",
              beam_radius=1400.0, beam_spot=900.0, ion_z=1, ion_a=1):
    energies, weights = gcr_spectrum(phi_MV, z=ion_z, a=ion_a)
    lines = [
        "# ==========================================================",
        "# GCR primary source -- isotropic upper hemisphere",
        f"#   mode=GCR  rings={rings}  azimuth={azimuth}  phi={phi_MV} MV",
        f"#   particle={particle} (Z={ion_z} A={ion_a})  histories/source={histories}",
        "#   Rings are equal-area in sin^2(theta): equal flux per ring,",
        "#   so equal histories per source = exact Lambert weighting.",
        "# ==========================================================",
        "",
        "# Beam geometry (self-contained so this file is includeable as-is).",
        "# An included file cannot resolve chained references to parameters",
        "# defined in the host, so BeamRadius/BeamSpot are defined here.",
        f"d:Ge/BeamRadius = {beam_radius:.1f} cm   # sky-dome radius (source distance)",
        f"d:So/BeamSpot   = {beam_spot:.1f} cm   # per-source disc radius (illuminated footprint)",
        "",
    ]
    idx = 0
    for r in range(rings):
        # representative zenith of band [sqrt(r/R), sqrt((r+1)/R)] in sin^2
        s2 = (r + 0.5) / rings
        theta = math.degrees(math.asin(math.sqrt(s2)))
        for a in range(azimuth):
            phi = a * 360.0 / azimuth
            tag = f"GCR{idx}"
            grp, src = _ring_groups(idx, phi, theta, "G")
            lines += grp
            lines += [
                f'    s:So/{tag}/Type      = "Beam"',
                f'    s:So/{tag}/Component = "{src}"',
                f'    s:So/{tag}/BeamParticle = "{particle}"',
            ]
            lines += _spectrum_block(tag, energies, weights)
            lines += [
                f'    s:So/{tag}/BeamPositionDistribution = "Flat"',
                f'    s:So/{tag}/BeamPositionCutoffShape  = "Ellipse"',
                f'    d:So/{tag}/BeamPositionCutoffX      = So/BeamSpot cm',
                f'    d:So/{tag}/BeamPositionCutoffY      = So/BeamSpot cm',
                f'    s:So/{tag}/BeamAngularDistribution  = "None"',
                f'    i:So/{tag}/NumberOfHistoriesInRun   = {histories}',
                "",
            ]
            idx += 1
    lines.insert(0, f"# total sources: {rings * azimuth}")
    return "\n".join(lines)


def spe_block(zenith, azimuth_deg, cone_deg, histories, event="aug1972",
              particle="proton", beam_radius=1400.0, beam_spot=900.0,
              r0_MV=None, e0_MeV=None):
    energies, weights = spe_spectrum(event, r0_MV=r0_MV, e0_MeV=e0_MeV)
    if e0_MeV is not None:
        model = f"legacy exp-in-energy  E0={e0_MeV} MeV"
    else:
        r0 = r0_MV if r0_MV is not None else SPE_EVENTS[event][0]
        model = (f"exp-in-rigidity  event={event}  R0={r0:.0f} MV  "
                 f"phi(>30MeV)={spe_integral_fluence(event):.2g} /cm2")
    lines = [
        "# ==========================================================",
        "# SPE primary source -- directional cone from the Sun (SOLAR MAX)",
        f"#   look-direction zenith={zenith} deg  azimuth={azimuth_deg} deg",
        f"#   cone half-angle={cone_deg} deg  histories={histories}",
        f"#   spectrum: {model}",
        "#   NOT omnidirectional: a solar event arrives as a beam/cone.",
        "# ==========================================================",
        "",
        "# Beam geometry (self-contained so this file is includeable as-is).",
        f"d:Ge/BeamRadius = {beam_radius:.1f} cm   # sky-dome radius (source distance)",
        f"d:So/BeamSpot   = {beam_spot:.1f} cm   # per-source disc radius (illuminated footprint)",
        "",
    ]
    grp, src = _ring_groups(0, azimuth_deg, zenith, "S")
    lines += grp
    tag = "SPE0"
    lines += [
        f'    s:So/{tag}/Type      = "Beam"',
        f'    s:So/{tag}/Component = "{src}"',
        f'    s:So/{tag}/BeamParticle = "{particle}"',
    ]
    lines += _spectrum_block(tag, energies, weights)
    lines += [
        f'    s:So/{tag}/BeamPositionDistribution = "Flat"',
        f'    s:So/{tag}/BeamPositionCutoffShape  = "Ellipse"',
        f'    d:So/{tag}/BeamPositionCutoffX      = So/BeamSpot cm',
        f'    d:So/{tag}/BeamPositionCutoffY      = So/BeamSpot cm',
        # finite angular spread = the cone
        f'    s:So/{tag}/BeamAngularDistribution  = "Gaussian"',
        f'    d:So/{tag}/BeamAngularCutoffX       = {cone_deg:.3f} deg',
        f'    d:So/{tag}/BeamAngularCutoffY       = {cone_deg:.3f} deg',
        f'    d:So/{tag}/BeamAngularSpreadX       = {cone_deg / 2.0:.3f} deg',
        f'    d:So/{tag}/BeamAngularSpreadY       = {cone_deg / 2.0:.3f} deg',
        f'    i:So/{tag}/NumberOfHistoriesInRun   = {histories}',
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Standalone demo wrapper (minimal world + regolith + fluence scorer)
# --------------------------------------------------------------------------
def standalone_header():
    return """# Auto-generated standalone test for the GCR/SPE source module.
# Minimal geometry to validate the source: a regolith slab + a fluence scorer.
# (Full 4-layer Apollo-17 regolith and FTFP_BERT_HP tuning are step #2.)

i:Ts/Seed            = 1
i:Ts/ShowHistoryCountAtInterval = 200
b:Ts/PauseBeforeQuit = "false"
Ph/Default/Modules = 1 "g4em-standard_opt3"
s:Ph/ListName = "Default"
# Recommended production list (needs G4NDL high-precision neutron data):
# s:Ph/ListName = "FTFP_BERT_HP"

d:Ge/World/HLX = 12.0 m
d:Ge/World/HLY = 12.0 m
d:Ge/World/HLZ = 12.0 m
s:Ge/World/Material = "Vacuum"
b:Ge/World/Invisible = "true"

# (BeamRadius / BeamSpot are defined by the source block appended below.)

# target: simple silica regolith slab (top at z=0)
s:Ge/Regolith/Type     = "TsBox"
s:Ge/Regolith/Parent   = "World"
s:Ge/Regolith/Material  = "G4_SILICON_DIOXIDE"
d:Ge/Regolith/HLX = 400.0 cm
d:Ge/Regolith/HLY = 400.0 cm
d:Ge/Regolith/HLZ = 150.0 cm
d:Ge/Regolith/TransZ = -150.0 cm

# fluence scorer just under the surface
s:Sc/SurfaceFluence/Quantity   = "Fluence"
s:Sc/SurfaceFluence/Component  = "Regolith"
s:Sc/SurfaceFluence/OutputFile = "gcr_demo_fluence"
s:Sc/SurfaceFluence/OutputType = "csv"
s:Sc/SurfaceFluence/IfOutputFileAlreadyExists = "Overwrite"

"""


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["gcr", "spe"], default="gcr")
    p.add_argument("--rings", type=int, default=5)
    p.add_argument("--azimuth", type=int, default=8)
    p.add_argument("--phi", type=float, default=400.0, help="modulation potential (MV); low=solar min=worst case")
    p.add_argument("--histories", type=int, default=25, help="histories per source")
    p.add_argument("--particle", default="proton")
    p.add_argument("--ion-z", type=int, default=1, help="ion charge Z (per-nucleon spectrum modulation)")
    p.add_argument("--ion-a", type=int, default=1, help="ion mass number A (per-nucleon spectrum modulation)")
    p.add_argument("--spe-zenith", type=float, default=45.0)
    p.add_argument("--spe-azimuth", type=float, default=0.0)
    p.add_argument("--spe-cone", type=float, default=20.0, help="cone half-angle (deg)")
    p.add_argument("--spe-event", choices=sorted(SPE_EVENTS), default="aug1972",
                   help="worst-case (solar-max) design event; default aug1972 (canonical crewed worst case)")
    p.add_argument("--spe-r0", type=float, default=None,
                   help="override event characteristic rigidity R0 (MV); higher = harder")
    p.add_argument("--spe-e0", type=float, default=None,
                   help="LEGACY exp-in-ENERGY e-fold (MeV); not physical, overrides the rigidity model")
    p.add_argument("--beam-radius", type=float, default=1400.0, help="sky-dome source distance (cm)")
    p.add_argument("--beam-spot", type=float, default=900.0, help="per-source disc radius (cm); should cover the habitat footprint")
    p.add_argument("--standalone", action="store_true", help="emit a full runnable test file")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.mode == "gcr":
        body = gcr_block(args.rings, args.azimuth, args.phi, args.histories,
                         args.particle, args.beam_radius, args.beam_spot,
                         ion_z=args.ion_z, ion_a=args.ion_a)
        default_out = "gcr_source.txt"
    else:
        body = spe_block(args.spe_zenith, args.spe_azimuth, args.spe_cone,
                         args.histories, args.spe_event, args.particle,
                         args.beam_radius, args.beam_spot,
                         r0_MV=args.spe_r0, e0_MeV=args.spe_e0)
        default_out = "spe_source.txt"

    text = (standalone_header() + body + "\n") if args.standalone else (body + "\n")
    out = args.out or default_out
    with open(out, "w") as f:
        f.write(text)
    nsrc = args.rings * args.azimuth if args.mode == "gcr" else 1
    print(f"wrote {out}  (mode={args.mode}, sources={nsrc})")


if __name__ == "__main__":
    main()
