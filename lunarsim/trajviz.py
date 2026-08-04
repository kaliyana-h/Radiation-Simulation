"""Headless particle-cascade visualisation for lunar-habitat designs.

The live OpenGL/Qt viewers are unusable on WSL2 (GLX context creation fails on
WSLg; Qt segfaults). Instead we capture particle crossings with TOPAS's built-in
PhaseSpace scorer on a set of concentric tracking shells, reconstruct each track
as a polyline (grouped by event+track, ordered by radius), and draw the habitat +
tracks with Plotly. The result is an interactive HTML file the student opens in a
browser -- no X server, no Qt, no rebuild -- and the same figure embeds in the
Dash GUI.

Pipeline:  spec -> bridge.write_traj_run -> TOPAS (headless) -> ps_*.phsp
           -> load_tracks -> build_figure -> HTML / Dash.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import importlib.util
from glob import glob
from pathlib import Path
from dataclasses import dataclass, field

# PDG -> (family label, colour).  Neutrinos are dropped (invisible, clutter).
_PDG = {
    2212: ("proton", "#e8443a"),
    2112: ("neutron", "#39b54a"),
    22:   ("gamma", "#f2c500"),
    11:   ("e-", "#2bb8d6"),
    -11:  ("e+", "#d23bd2"),
    211:  ("pi+", "#ff8c00"),
    -211: ("pi-", "#ff8c00"),
    111:  ("pi0", "#ff8c00"),
    13:   ("mu-", "#9b8cff"),
    -13:  ("mu+", "#9b8cff"),
}
_NEUTRINOS = {12, -12, 14, -14, 16, -16}


def _classify(pdg: int) -> tuple[str, str]:
    if pdg in _PDG:
        return _PDG[pdg]
    if pdg > 1_000_000_000:            # nuclear ion code 100ZZZAAAI
        return ("ion", "#3a6ee8")
    return ("other", "#888888")


@dataclass
class Track:
    particle: str
    colour: str
    pdg: int
    parent: int
    energy0: float                      # KE [MeV] at outermost crossing
    pts: list = field(default_factory=list)   # [(x, y, z, r)] sorted by radius


def parse_phsp(path: Path):
    """Yield (x, y, z, energy, pdg, event, track, parent) from a TOPAS ASCII phsp."""
    with open(path) as fh:
        for line in fh:
            c = line.split()
            if len(c) < 13:
                continue
            try:
                x, y, z = float(c[0]), float(c[1]), float(c[2])
                energy = float(c[5])
                pdg = int(c[7])
                event, track, parent = int(c[10]), int(c[11]), int(c[12])
            except ValueError:
                continue
            yield x, y, z, energy, pdg, event, track, parent


def load_tracks(run_dir, pattern="ps_*.phsp",
                min_energy_mev: float = 1.0,
                drop_neutrinos: bool = True) -> list[Track]:
    """Merge every phsp in run_dir into reconstructed Track polylines."""
    run_dir = Path(run_dir)
    raw: dict[tuple[int, int], dict] = {}
    for f in sorted(glob(str(run_dir / pattern))):
        for x, y, z, energy, pdg, event, track, parent in parse_phsp(f):
            if drop_neutrinos and pdg in _NEUTRINOS:
                continue
            key = (event, track)
            r = math.sqrt(x * x + y * y + z * z)
            d = raw.setdefault(key, {"pdg": pdg, "parent": parent, "pts": []})
            d["pts"].append((x, y, z, r, energy))

    tracks: list[Track] = []
    for (event, track), d in raw.items():
        pts = sorted(d["pts"], key=lambda p: p[3])      # by radius
        # outermost crossing energy = energy at the largest radius point
        e0 = max(d["pts"], key=lambda p: p[3])[4]
        if e0 < min_energy_mev:
            continue
        label, colour = _classify(d["pdg"])
        tracks.append(Track(
            particle=label, colour=colour, pdg=d["pdg"], parent=d["parent"],
            energy0=e0, pts=[(x, y, z, r) for (x, y, z, r, _e) in pts]))
    return tracks


def _rgba(hex_colour: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,alpha)' for translucent Plotly lines."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _polyline(tracks):
    """Concatenate tracks into one x/y/z polyline list with None breaks."""
    xs, ys, zs = [], [], []
    for t in tracks:
        if len(t.pts) < 2:
            continue
        for (x, y, z, _r) in t.pts:
            xs.append(x); ys.append(y); zs.append(z)
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


# primaries = incoming GCR rays (ParentID 0); secondaries = shower born in walls
_PRIMARY_HEX = "#ff5544"
_SECONDARY_HEX = "#5aa9e6"


def build_figure(run_dir, spec=None, max_tracks: int = 4000,
                 title: str | None = None, colour_by: str = "family",
                 show: str = "all"):
    """Plotly 3-D figure: habitat dome shells + reconstructed particle tracks.

    colour_by : "family" colours by particle type (proton/neutron/...), with
                primaries drawn bold/solid and secondaries thin/translucent.
                "origin" colours by where the track came from -- red = incoming
                GCR primary, blue = secondary created in the shielding.
    show      : "all" | "primary" | "secondary" -- which tracks to draw.
    """
    import plotly.graph_objects as go

    tracks = load_tracks(run_dir)
    # draw longest / most-energetic tracks first if we have to cap
    tracks.sort(key=lambda t: (len(t.pts), t.energy0), reverse=True)
    tracks = tracks[:max_tracks]
    if show == "primary":
        tracks = [t for t in tracks if t.parent == 0]
    elif show == "secondary":
        tracks = [t for t in tracks if t.parent != 0]

    fig = go.Figure()

    # --- habitat wall shells as translucent hemispheres ---
    if spec is not None:
        for (ri, ro), layer in zip(spec.layer_radii_cm(), spec.walls):
            _add_hemisphere(fig, go, ro, layer.material)
        # phantom marker at crew height (matches geometry.py)
        cz = spec.crew_height_cm
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[cz], mode="markers",
            marker=dict(size=6, color="#6ea9da", symbol="circle"),
            name="crew phantom", hoverinfo="name"))

    is_primary = lambda t: t.parent == 0

    if colour_by == "origin":
        # two legend entries: incoming GCR vs shielding-born secondaries
        groups = [("primary (incoming GCR)", _PRIMARY_HEX, 5, 1.0,
                   [t for t in tracks if is_primary(t)]),
                  ("secondary (born in shield)", _SECONDARY_HEX, 2, 0.5,
                   [t for t in tracks if not is_primary(t)])]
        for name, hexc, width, alpha, grp in groups:
            if not grp:
                continue
            x, y, z = _polyline(grp)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines",
                line=dict(color=_rgba(hexc, alpha), width=width),
                name=name, connectgaps=False))
    else:
        # colour by particle family; primaries bold/solid, secondaries thin/faint.
        # one legend entry per family (legendgroup keeps the two styles linked).
        order = ["proton", "neutron", "gamma", "e-", "e+", "pi+", "pi-",
                 "mu-", "mu+", "ion", "other"]
        fams = sorted({t.particle for t in tracks},
                      key=lambda f: order.index(f) if f in order else 99)
        for fam in fams:
            fam_tracks = [t for t in tracks if t.particle == fam]
            colour = fam_tracks[0].colour
            prim = [t for t in fam_tracks if is_primary(t)]
            sec = [t for t in fam_tracks if not is_primary(t)]
            shown_legend = False
            # secondaries first (thin/faint), then primaries on top (bold/solid)
            for grp, width, alpha in ((sec, 2, 0.45), (prim, 5, 1.0)):
                if not grp:
                    continue
                x, y, z = _polyline(grp)
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z, mode="lines",
                    line=dict(color=_rgba(colour, alpha), width=width),
                    name=fam, legendgroup=fam,
                    showlegend=not shown_legend, connectgaps=False))
                shown_legend = True

    rng = (spec.outer_radius_cm * 1.15) if spec is not None else 500
    fig.update_layout(
        title=title or "Habitat radiation cascade (GCR primaries + secondaries)",
        scene=dict(
            xaxis=dict(title="x [cm]", range=[-rng, rng]),
            yaxis=dict(title="y [cm]", range=[-rng, rng]),
            zaxis=dict(title="z [cm]", range=[0, rng]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),  # pulled back so it fits
            bgcolor="#0d0d0f"),
        paper_bgcolor="#0d0d0f", font_color="#dddddd",
        legend=dict(itemsizing="constant"))
    return fig


def _add_hemisphere(fig, go, radius: float, name: str, n: int = 24):
    """Translucent upper hemisphere mesh at the given outer radius."""
    us = [i * math.pi / 2 / n for i in range(n + 1)]      # polar 0..90
    vs = [j * 2 * math.pi / (2 * n) for j in range(2 * n + 1)]  # azimuth
    xs, ys, zs = [], [], []
    for u in us:
        for v in vs:
            xs.append(radius * math.sin(u) * math.cos(v))
            ys.append(radius * math.sin(u) * math.sin(v))
            zs.append(radius * math.cos(u))
    fig.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs, alphahull=0, opacity=0.10,
        color="#9aa0a6", name=name, showlegend=True, hoverinfo="name"))


def write_html(run_dir, out=None, spec=None, title=None,
               colour_by: str = "family", show: str = "all") -> Path:
    """Render the cascade to a standalone HTML file (open in any browser)."""
    run_dir = Path(run_dir)
    if out is None:
        out = run_dir / "cascade.html"
    fig = build_figure(run_dir, spec=spec, title=title,
                       colour_by=colour_by, show=show)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return Path(out)


# ----------------------------------------------------------------------
# Run generation:  spec -> headless PhaseSpace cascade run
# ----------------------------------------------------------------------
# The live OpenGL/Qt viewers are dead on WSL2, so we capture the cascade with
# PhaseSpace scorers on concentric tracking shells and a SINGLE isotropic source
# (one source => globally-unique event IDs => tracks group cleanly by event+track;
# the 40-disc dosimetry source resets event IDs per disc and cannot be grouped).


def _make_source_module():
    """Import make_source.py (sits at TOPAS_ROOT) for its GCR spectrum."""
    from .bridge import MAKE_SOURCE
    spec = importlib.util.spec_from_file_location("make_source", str(MAKE_SOURCE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shell(name: str, r: float, half_thick: float = 0.5) -> str:
    return (f's:Ge/{name}/Type = "TsSphere"\n'
            f's:Ge/{name}/Parent = "World"\n'
            f's:Ge/{name}/Material = "Vacuum"\n'
            f'd:Ge/{name}/RMin = {r - half_thick:.3f} cm\n'
            f'd:Ge/{name}/RMax = {r + half_thick:.3f} cm\n'
            f'd:Ge/{name}/DPhi = 360.0 deg\n'
            f'd:Ge/{name}/DTheta = 90.0 deg\n'
            f'b:Ge/{name}/Invisible = "true"\n')


def _phasespace(tag: str, surface: str) -> str:
    return (f's:Sc/{tag}/Quantity = "PhaseSpace"\n'
            f's:Sc/{tag}/Surface = "{surface}"\n'
            f's:Sc/{tag}/OutputType = "ASCII"\n'
            f's:Sc/{tag}/OutputFile = "ps_{tag}"\n'
            f'b:Sc/{tag}/IncludeEventID = "true"\n'
            f'b:Sc/{tag}/IncludeTrackID = "true"\n'
            f'b:Sc/{tag}/IncludeParentID = "true"\n'
            f's:Sc/{tag}/IfOutputFileAlreadyExists = "Overwrite"\n')


def _tracking_radii(spec) -> list[float]:
    """Concentric sampling radii: a few inside, a few outside, skipping the
    phantom band so a tracking shell never overlaps the crew sphere."""
    inner, outer = spec.inner_radius_cm, spec.outer_radius_cm
    pz, pr = spec.crew_height_cm, spec.phantom_radius_cm + 15.0
    radii: list[float] = []
    for frac in (0.30, 0.55, 0.80):
        r = frac * inner
        if abs(r - pz) > pr:                 # clear of the phantom
            radii.append(r)
    sky = max(900.0, outer * 2.0)
    for frac in (0.07, 0.45, 0.85):          # outer: just past wall -> near sky
        radii.append(outer + frac * (sky - outer))
    return radii


def build_cascade_run(spec, run_dir, n_histories: int = 120,
                      phi_mv: float = 400.0, seed: int = 1) -> Path:
    """Write a self-contained headless PhaseSpace cascade run for `spec`.

    Reuses geometry.build_geometry (walls + phantom), adds concentric vacuum
    tracking shells + a single isotropic GCR proton source on a sky shell, and a
    PhaseSpace scorer on every tracking shell and wall layer. Returns run.txt path.
    """
    from . import geometry
    from .bridge import ENV_INCLUDE
    spec.validate()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ENV_INCLUDE, run_dir / "lunar_environment.txt")

    ms = _make_source_module()
    energies, weights = ms.gcr_spectrum(phi_mv)
    spectrum = "\n".join(ms._spectrum_block("Sky", energies, weights)).replace("    ", "")

    sky = max(900.0, spec.outer_radius_cm * 2.0)
    world_hl = sky + 250.0
    radii = _tracking_radii(spec)

    parts: list[str] = [
        f"# Headless PhaseSpace cascade run -- design: {spec.name}",
        f"# walls={[(w.material, w.thickness_cm) for w in spec.walls]}  "
        f"phi={phi_mv} MV  histories={n_histories}",
        "",
        f"i:Ts/Seed = {seed}",
        "i:Ts/NumberOfThreads = 1",
        'b:Ts/PauseBeforeQuit = "false"',
        'b:Ts/UseQt = "false"',
        's:Ph/Default/Type = "FTFP_BERT_HP"',
        "",
        's:Ge/World/Type = "TsBox"',
        's:Ge/World/Material = "Vacuum"',
        f"d:Ge/World/HLX = {world_hl:.1f} cm",
        f"d:Ge/World/HLY = {world_hl:.1f} cm",
        f"d:Ge/World/HLZ = {world_hl:.1f} cm",
        'b:Ge/World/Invisible = "true"',
        "",
        geometry.build_geometry(spec),
        "",
        "# --- sky source surface (upper hemisphere, 2pi lunar sky) ---",
        _shell("SkySource", sky, half_thick=0.5),
        "",
        "# --- concentric vacuum tracking shells ---",
    ]
    for i, r in enumerate(radii):
        parts.append(_shell(f"T{i}", r))
    parts.append("\n# --- single isotropic GCR proton source ---")
    parts.append('s:So/Sky/Type = "Isotropic"')
    parts.append('s:So/Sky/Component = "SkySource"')
    parts.append('s:So/Sky/BeamParticle = "proton"')
    parts.append(spectrum.strip())
    parts.append(f"i:So/Sky/NumberOfHistoriesInRun = {n_histories}")
    parts.append("\n# --- PhaseSpace scorers (tracking shells + wall layers) ---")
    for i in range(len(radii)):
        parts.append(_phasespace(f"T{i}", f"T{i}/AnySurface"))
    for i in range(len(spec.walls)):
        parts.append(_phasespace(f"W{i}", f"Wall{i}/AnySurface"))
    parts.append("\nincludeFile = lunar_environment.txt\n")

    param_file = run_dir / "cascade.txt"
    param_file.write_text("\n".join(parts))
    return param_file


def run_cascade(spec, run_dir=None, n_histories: int = 120, phi_mv: float = 400.0,
                seed: int = 1, open_html: bool = False,
                colour_by: str = "family", show: str = "all") -> Path:
    """End-to-end: build + run (headless) + render the cascade HTML for `spec`.

    Returns the path to cascade.html (open it in a browser; also embeds in Dash
    via build_figure). No X server / Qt / rebuild required.
    """
    from .bridge import TOPAS_BIN, TOPAS_ROOT
    G4 = os.environ.get("TOPAS_G4_DATA_DIR", str(Path.home() / "G4Data"))
    if run_dir is None:
        run_dir = TOPAS_ROOT / f"cascade_{spec.name}"
    run_dir = Path(run_dir)
    param_file = build_cascade_run(spec, run_dir, n_histories, phi_mv, seed)

    env = dict(os.environ, TOPAS_G4_DATA_DIR=G4)
    proc = subprocess.run([str(TOPAS_BIN), param_file.name], cwd=run_dir,
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
        raise RuntimeError(f"TOPAS cascade run failed (rc={proc.returncode}):\n{tail}")

    html = write_html(run_dir, spec=spec, colour_by=colour_by, show=show,
                      title=f"{spec.name}: GCR radiation cascade "
                            f"({spec.areal_density_gcm2():.0f} g/cm2 shielding)")
    if open_html:
        import webbrowser
        webbrowser.open(html.as_uri())
    return html


if __name__ == "__main__":
    import argparse
    from collections import Counter
    from .spec import default_spec

    p = argparse.ArgumentParser(
        description="Run a headless GCR cascade for a habitat and render an "
                    "interactive Plotly HTML (no X server / Qt / rebuild).")
    p.add_argument("--name", default="dome", help="design name (output dir suffix)")
    p.add_argument("--radius", type=float, default=400.0,
                   help="inner radius [cm] (default dome design)")
    p.add_argument("--histories", type=int, default=120,
                   help="sky primaries to launch (more = denser cascade, slower)")
    p.add_argument("--phi", type=float, default=400.0, help="solar modulation [MV]")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--colour-by", choices=("family", "origin"), default="family",
                   help="family = colour by particle (primaries bold, secondaries "
                        "faint); origin = red incoming GCR vs blue secondaries")
    p.add_argument("--show", choices=("all", "primary", "secondary"), default="all",
                   help="draw all tracks, only incoming primaries, or only the shower")
    p.add_argument("--open", action="store_true",
                   help="open the HTML in a browser when done")
    p.add_argument("--no-run", action="store_true",
                   help="re-render an existing cascade_<name> dir (apply a new "
                        "--colour-by/--show) without re-running TOPAS")
    p.add_argument("--inspect", metavar="RUN_DIR",
                   help="don't run; just count tracks in an existing run dir")
    args = p.parse_args()

    if args.inspect:
        ts = load_tracks(args.inspect)
        print(f"{len(ts)} drawable tracks from {args.inspect}")
        print(Counter(t.particle for t in ts))
    else:
        from .bridge import TOPAS_ROOT
        spec = default_spec()
        spec.name = args.name
        spec.inner_radius_cm = args.radius
        if args.no_run:
            run_dir = TOPAS_ROOT / f"cascade_{spec.name}"
            print(f"Re-rendering {run_dir} "
                  f"(colour_by={args.colour_by}, show={args.show})...")
            html = write_html(run_dir, spec=spec, colour_by=args.colour_by,
                              show=args.show,
                              title=f"{spec.name}: GCR radiation cascade "
                                    f"({spec.areal_density_gcm2():.0f} g/cm2 shielding)")
        else:
            print(f"Running cascade for '{spec.name}' "
                  f"(r={spec.inner_radius_cm:.0f} cm, {args.histories} primaries)...")
            html = run_cascade(spec, n_histories=args.histories, phi_mv=args.phi,
                               seed=args.seed, open_html=args.open,
                               colour_by=args.colour_by, show=args.show)
        ts = load_tracks(html.parent)
        print(f"\nDONE -> {html}")
        print(f"  {len(ts)} tracks: {dict(Counter(t.particle for t in ts))}")
        print(f"  open it in your Windows browser, e.g.:  explorer.exe "
              f"$(wslpath -w {html})")
