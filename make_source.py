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
  LIS (protons), Usoskin et al. (2005):
      J_LIS(T) = 1.9e4 * T^-2.78 / (1 + 0.4866 * T^-2.51)   [/(m^2 s sr GeV)]
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
  ./make_source.py --mode spe --spe-zenith 45 --spe-azimuth 0 --spe-cone 20
  ./make_source.py --standalone --out gcr_demo.txt   # full runnable test file
"""

import argparse
import math

E0_PROTON = 938.272  # MeV, proton rest energy per nucleon


# --------------------------------------------------------------------------
# Energy spectra
# --------------------------------------------------------------------------
def lis_proton(T_GeV):
    """Local interstellar proton spectrum, Usoskin et al. (2005)."""
    return 1.9e4 * T_GeV ** (-2.78) / (1.0 + 0.4866 * T_GeV ** (-2.51))


def gcr_spectrum(phi_MV, z=1, a=1, emin=10.0, emax=1.0e5, n=28):
    """Force-field-modulated per-nucleon spectrum on a log energy grid.

    Returns (energies_MeV, weights) where weights are relative dN/dE
    (TOPAS normalises them internally)."""
    Phi = (z / a) * phi_MV  # effective modulation in MeV for this species
    energies, weights = [], []
    for k in range(n):
        T = emin * (emax / emin) ** (k / (n - 1))     # per-nucleon KE, MeV
        Tm = T + Phi
        factor = (T * (T + 2 * E0_PROTON)) / (Tm * (Tm + 2 * E0_PROTON))
        j = lis_proton(Tm / 1000.0) * factor
        energies.append(T * a)        # TOPAS BeamEnergy = TOTAL kinetic energy
        weights.append(j)
    wmax = max(weights)
    weights = [w / wmax for w in weights]             # scale to peak 1.0
    return energies, weights


def spe_spectrum(e0_MeV=60.0, emin=10.0, emax=1.0e3, n=24):
    """First-pass SPE proton spectrum: exponential in energy, dN/dE ~ exp(-E/E0).
    (Placeholder for a measured event spectrum, e.g. 1972/1989; see TODO.)"""
    energies, weights = [], []
    for k in range(n):
        E = emin * (emax / emin) ** (k / (n - 1))
        energies.append(E)
        weights.append(math.exp(-E / e0_MeV))
    wmax = max(weights)
    return energies, [w / wmax for w in weights]


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
              beam_radius=900.0, beam_spot=500.0):
    energies, weights = gcr_spectrum(phi_MV)
    lines = [
        "# ==========================================================",
        "# GCR primary source -- isotropic upper hemisphere",
        f"#   mode=GCR  rings={rings}  azimuth={azimuth}  phi={phi_MV} MV",
        f"#   particle={particle}  histories/source={histories}",
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


def spe_block(zenith, azimuth_deg, cone_deg, histories, e0_MeV, particle="proton",
              beam_radius=900.0, beam_spot=500.0):
    energies, weights = spe_spectrum(e0_MeV)
    lines = [
        "# ==========================================================",
        "# SPE primary source -- directional cone from the Sun",
        f"#   look-direction zenith={zenith} deg  azimuth={azimuth_deg} deg",
        f"#   cone half-angle={cone_deg} deg  E0={e0_MeV} MeV  histories={histories}",
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
    p.add_argument("--spe-zenith", type=float, default=45.0)
    p.add_argument("--spe-azimuth", type=float, default=0.0)
    p.add_argument("--spe-cone", type=float, default=20.0, help="cone half-angle (deg)")
    p.add_argument("--spe-e0", type=float, default=60.0, help="SPE spectral e-fold energy (MeV)")
    p.add_argument("--beam-radius", type=float, default=900.0, help="sky-dome source distance (cm)")
    p.add_argument("--beam-spot", type=float, default=500.0, help="per-source disc radius (cm); should cover the habitat footprint")
    p.add_argument("--standalone", action="store_true", help="emit a full runnable test file")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.mode == "gcr":
        body = gcr_block(args.rings, args.azimuth, args.phi, args.histories,
                         args.particle, args.beam_radius, args.beam_spot)
        default_out = "gcr_source.txt"
    else:
        body = spe_block(args.spe_zenith, args.spe_azimuth, args.spe_cone,
                         args.histories, args.spe_e0, args.particle,
                         args.beam_radius, args.beam_spot)
        default_out = "spe_source.txt"

    text = (standalone_header() + body + "\n") if args.standalone else (body + "\n")
    out = args.out or default_out
    with open(out, "w") as f:
        f.write(text)
    nsrc = args.rings * args.azimuth if args.mode == "gcr" else 1
    print(f"wrote {out}  (mode={args.mode}, sources={nsrc})")


if __name__ == "__main__":
    main()
