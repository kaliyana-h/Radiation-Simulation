"""Radiation Simulation Tool -- Dash GUI for the lunar-habitat workshop.

Layout matches the agreed mockup:

  * Left rail  : branded "Radiation Sim" -> Habitat Geometry -> Primary Wall
                 (+ collapsible extra shielding layer) -> exposure -> Run.
  * Header     : "Radiation Simulation Tool" + a live config subtitle line.
  * Tabs       : Habitat Overview | GCR Environment | Dose Analysis.
  * Overview   : Dose cross-section / 3-D wireframe toggle over the habitat model.
  * Right rail : Dose Metrics card stack + Design Parameters.

The numbers are NOT analytical -- TOPAS runs in the background (Run button ->
job poll) and the Dose Metrics fill in when the run converges. Geometry, the
config line and Design Parameters update live as the design changes.

Run:  TOPAS_G4_DATA_DIR=~/G4Data ~/topas/.venv/bin/python -m lunarsim.gui
"""
from __future__ import annotations

import os
os.environ.setdefault("TOPAS_G4_DATA_DIR", os.path.expanduser("~/G4Data"))

import math
from dash import Dash, html, dcc, Input, Output, State, ALL, ctx, no_update
import plotly.graph_objects as go

import base64
import datetime as _dt
import json

from .spec import HabitatSpec, WallLayer, MATERIALS, SHAPES
from .bridge import FULL_RUN, WORST_CASE_SPE, DEFAULT_BEAM_SPOT_CM
from .dosimetry import (assess, assess_composition, assess_spe,
                        gcr_scalar_fluence_rate, DOSE_LIMITS_MSV,
                        gauge_size_flag, _gauge_corr,
                        beam_footprint_flag,
                        assess_gcr_thinwall, _gcr_thinwall_applies,
                        _gcr_thinwall_calibrated)
from .geometry import _enclosing_radius_cm
from .jobs import default_runner, JobStatus, ConvergedComposition
from .trajviz import run_cascade, build_figure

# ----------------------------------------------------------------------
# Theme  (dark, red-orange accent, light-blue metric values)
# ----------------------------------------------------------------------
BG       = "#0a0d12"      # main canvas, near-black
RAIL     = "#0e131a"      # side rails
CARD_BG  = "#141a22"      # panel / metric card
BORDER   = "#222a34"
INK      = "#e6edf3"      # primary text
MUTED    = "#8b97a7"      # secondary text
ACCENT   = "#f24f3d"      # red-orange: sliders, active tab, live values
METRIC   = "#6ea9da"      # light-blue metric numbers
GROUND   = "#6b7280"

SHAPE_LABELS = {"dome": "Dome (half-sphere)", "cylinder": "Cylinder",
                "quonset": "Half-cylinder (tunnel)"}
MATERIAL_OPTIONS = [{"label": m.capitalize(), "value": m} for m in MATERIALS]
VERDICT_COLOUR = {"SAFE": "#3fb950", "MARGINAL": "#d29922", "EXCEEDS LIMIT": ACCENT}

# --- Fixed scoring preset --------------------------------------------------
# Every evaluation uses THESE settings, hidden from the student, so that two
# teams' scores -- generated in separate sessions -- are on one comparable
# scale and cannot be gamed. The graded number is the habitat-wide (wall-lining)
# annual effective dose; converge it tightly since it is the official score.
SCORING_TIER = FULL_RUN
SCORING_MISSION_DAYS = 365
# Converge on BOTH the skin (wall-lining) dose -- the official score,
# _metric_cards reads a_skin.annual_msv -- and the central crew phantom, but to
# DIFFERENT targets: skin 5%, phantom 10% (SCORING_PHANTOM_SLACK = 2x).
#
# Why both: the phantom is the crew-representative geometry, not a curiosity. A
# human torso has a ~20 cm mean chord and the phantom ~27 cm, so both are thick
# enough to stop the degraded 50-200 MeV protons a thick shield produces; the
# 2 cm lining (~4 cm mean chord) is thick enough for none of them and is a
# SHIELDING figure of merit, not a person. It reads ~1.5x lower for that reason
# (see INTERFACE_SUMMARY.md 4.1), so leaving the phantom un-converged left the
# more crew-like number as the untrustworthy one. It now converges too.
#
# Why a looser target for it: the two converge at wildly different rates, and a
# shared target lets the phantom set the whole run length. Measured on a 3 m dome
# (5 cm Al + 30 cm regolith), by round:
#     round  2:  skin 0.0071   phantom 0.0765
#     round 12:  skin 0.0064   phantom 0.0490
# The skin is done at round 2 and never improves; the phantom is a 1/sqrt(N)
# grind that needed ten further rounds (~17 h) to crawl to 4.9% -- and BARELY
# inside 5%, so a slightly noisier design just hits SCORING_MAX_BATCHES and stops
# un-converged anyway. 10% is reached in ~4 rounds, which is ~1-2 extra hours,
# clears PHANTOM_MIN_BATCHES so a real error bar is shown instead of a batch
# count, and is honest about a quantity with 135x less scoring mass than the
# skin. Tighten the slack toward 1.0 if wall-time ever stops being the binding
# constraint.
# A floor on rounds, and the reason the above is not enough on its own. The
# phantom's rel_err at n=2 is a standard error off a TWO-sample stdev, which has
# ~76% relative uncertainty and is skewed LOW -- so it frequently reports a
# SMALLER error than the truth and satisfies the target spuriously. Measured 3 m
# dome: round 2 phantom reads 0.0765, which passes a 10% target outright, yet the
# same design still read 0.0490 at round 12; on the SHARC dome the n=2 estimate
# was 3% against a 5-round value of 7%. Converging on "both" without a floor
# therefore changes nothing -- the loop stops at round 2 exactly as before, on an
# error bar that cannot be believed. The floor makes the phantom take enough
# samples for its error to mean anything before any stop is allowed.
SCORING_TARGET_REL_ERR = 0.05
SCORING_MAX_BATCHES = 12
SCORING_MIN_BATCHES = 4         # must be >= 3 for a usable stdev; see above
SCORING_CONVERGE_ON = "both"
SCORING_PHANTOM_SLACK = 2.0     # phantom target = SLACK x SCORING_TARGET_REL_ERR

CARD = {"background": CARD_BG, "border": f"1px solid {BORDER}",
        "borderRadius": "10px", "padding": "16px", "marginBottom": "14px"}
SECTION = {"color": INK, "fontSize": "13px", "fontWeight": 700,
           "letterSpacing": "0.6px", "textTransform": "uppercase",
           "margin": "22px 0 12px"}
FIELD_LABEL = {"color": MUTED, "fontSize": "12px", "marginBottom": "6px"}
INPUT = {"width": "100%", "background": "#0b0f15", "color": INK,
         "border": f"1px solid {BORDER}", "borderRadius": "6px",
         "padding": "8px", "boxSizing": "border-box", "fontSize": "13px"}


# ----------------------------------------------------------------------
# Spec assembly from UI state  (UI is metres + millimetres; spec is cm)
# ----------------------------------------------------------------------
def _marks(values):
    """Slider tick labels with explicit light colour (inline, so it can't be
    overridden by Dash's own stylesheet load order)."""
    return {v: {"label": str(v), "style": {"color": "#b8c2cf", "fontSize": "10px"}}
            for v in values}


# Default wall stack the GUI opens on (innermost first; thickness in mm).
DEFAULT_LAYERS = [{"m": "aluminium", "t": 6.0}, {"m": "regolith", "t": 300.0}]

# The layer rows are a *fixed pool* baked into the initial layout (so the
# browser always sends their values); the `active-rows` store lists which pool
# rows are in use, innermost-first. No callback ever writes to a row's value —
# add/remove only edit this index list — so typing in a field is never fought
# by a "controlled value" re-render. Hidden (inactive) rows are ignored.
MAX_LAYERS = 8


def _layers_from_components(mats, ids, thks, active=None):
    """Collect the live layer-row widgets into the wall stack (innermost first)
    as a list of {m, t(mm)} dicts. Pattern-matching ALL returns each property
    in its own list; we key them by the row index in the component id, then
    read out in `active` order (the only rows that are part of the design).
    `active=None` falls back to every present row, index-ordered."""
    by_idx = {_id["index"]: (m, t) for _id, m, t in zip(ids, mats, thks)}
    order = active if active is not None else sorted(by_idx)
    return [{"m": (by_idx[i][0] or "aluminium"), "t": _num(by_idx[i][1])}
            for i in order if i in by_idx]


def _num(x):
    """Parse a thickness field (text input) to mm; blank/garbage -> 0.0."""
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return 0.0


def spec_from_inputs(name, shape, inner_r_m, layers, length_m=None) -> HabitatSpec:
    walls = [WallLayer(L.get("m") or "aluminium", float(L.get("t") or 0) / 10.0)
             for L in (layers or []) if float(L.get("t") or 0) > 0]
    if not walls:                                   # never build a wall-less habitat
        walls = [WallLayer("aluminium", 0.6)]
    shape = shape or "dome"
    # Axial length applies only to the two elongated shapes; a dome is a
    # hemisphere with no independent length, so it keeps height_cm=None and the
    # spec falls back to its radius-driven default.
    height_cm = (float(length_m) * 100.0
                 if (length_m and shape in ("cylinder", "quonset")) else None)
    return HabitatSpec(
        name=(name or "habitat").strip().replace(" ", "_") or "habitat",
        shape=shape,
        inner_radius_cm=float(inner_r_m) * 100.0,
        walls=walls,
        height_cm=height_cm,
    )


LAYER_BOX_STYLE = {"background": "#0b0f15", "border": f"1px solid {BORDER}",
                   "borderRadius": "8px", "padding": "10px 12px",
                   "marginBottom": "10px"}


def _layer_state(i, active):
    """The display state of pool row `i` given the ordered `active` index list:
    box style (hidden when i is not active), header label (numbered/located by
    its *position* in the stack, not its pool index), and the remove button's
    disabled flag and style. Shared by the initial render (`_layer_row`) and the
    callback that maintains it (`_render_layers`) so first paint and updates
    agree."""
    n = len(active)
    is_active = i in active
    box = LAYER_BOX_STYLE if is_active else {"display": "none"}
    if not is_active:
        pos, where = i, ""
    else:
        pos = active.index(i)
        if n == 1:
            where = "wall"
        elif pos == 0:
            where = "innermost"
        elif pos == n - 1:
            where = "outermost"
        else:
            where = f"layer {pos + 1}"
    rm_disabled = (n == 1)
    rm_style = {"background": "transparent",
                "color": MUTED if rm_disabled else ACCENT,
                "border": "none", "cursor": "pointer", "fontSize": "13px",
                "padding": "0 4px"}
    return box, f"Layer {pos + 1} · {where}", rm_disabled, rm_style


def _layer_row(i, material, thickness_mm, active_init):
    """One editable wall layer (material + thickness mm + remove button) in the
    fixed pool. index `i` is the row's fixed slot in the pool. The box, label
    and remove button carry stable ids so `_render_layers` can show/hide the row
    and relabel it as the active set changes; `active_init` sets the correct
    first-paint state so unused rows start hidden (no flash). The dropdown/input
    values are never outputs of any callback, so editing is never fought."""
    box, label, rm_disabled, rm_style = _layer_state(i, active_init)
    head = html.Div(style={"display": "flex", "justifyContent": "space-between",
                           "alignItems": "center", "marginBottom": "6px"}, children=[
        html.Span(label, id={"type": "layer-label", "index": i},
                  style={"color": INK, "fontSize": "12px", "fontWeight": 700}),
        html.Button("✕", id={"type": "layer-remove", "index": i},
                    disabled=rm_disabled, n_clicks=0, style=rm_style),
    ])
    return html.Div(id={"type": "layer-box", "index": i}, style=box, children=[
        head,
        dcc.Dropdown(id={"type": "layer-mat", "index": i}, value=material,
                     options=MATERIAL_OPTIONS, clearable=False,
                     style={"marginBottom": "8px"}),
        html.Div("Thickness (mm)", style=FIELD_LABEL),
        # type="text" (not "number"): the number widget doesn't reliably sync
        # typed digits to Dash -- only its spinners/clears fire -- so typed
        # thicknesses were lost. Text fires onChange every keystroke; _num()
        # parses it. inputMode shows a numeric keypad on touch devices.
        dcc.Input(id={"type": "layer-thk", "index": i}, type="text",
                  inputMode="decimal", value=f"{thickness_mm:g}",
                  debounce=False, style=INPUT),
    ])


# ----------------------------------------------------------------------
# 3-D wireframe habitat model
# ----------------------------------------------------------------------
def _hemisphere_lines(r, n_meridians=12, n_parallels=5, n_seg=26):
    """Lat/long wireframe of an upper hemisphere of radius r (one trace, None-gapped)."""
    xs, ys, zs = [], [], []
    for m in range(n_meridians):                       # meridians
        phi = 2 * math.pi * m / n_meridians
        for s in range(n_seg + 1):
            th = (math.pi / 2) * s / n_seg
            xs.append(r * math.sin(th) * math.cos(phi))
            ys.append(r * math.sin(th) * math.sin(phi))
            zs.append(r * math.cos(th))
        xs.append(None); ys.append(None); zs.append(None)
    for p in range(1, n_parallels + 1):                # parallels
        th = (math.pi / 2) * p / (n_parallels + 1)
        for s in range(n_seg + 1):
            phi = 2 * math.pi * s / n_seg
            xs.append(r * math.sin(th) * math.cos(phi))
            ys.append(r * math.sin(th) * math.sin(phi))
            zs.append(r * math.cos(th))
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def _cylinder_lines(r, H, n_meridians=12, n_seg=48):
    """Wireframe of a vertical open cylinder (base z=0, top z=H)."""
    xs, ys, zs = [], [], []
    for z in (0.0, H):                                  # base + top rings
        for s in range(n_seg + 1):
            phi = 2 * math.pi * s / n_seg
            xs.append(r * math.cos(phi)); ys.append(r * math.sin(phi)); zs.append(z)
        xs.append(None); ys.append(None); zs.append(None)
    for m in range(n_meridians):                        # vertical ribs
        phi = 2 * math.pi * m / n_meridians
        xs += [r * math.cos(phi), r * math.cos(phi), None]
        ys += [r * math.sin(phi), r * math.sin(phi), None]
        zs += [0.0, H, None]
    return xs, ys, zs


def _arch_lines(r, HL, n_long=9, n_seg=40):
    """Wireframe of a quonset half-cylinder: axis along Y in [-HL, HL], arch z>=0."""
    xs, ys, zs = [], [], []
    for y in (-HL, HL):                                 # end semicircle arches
        for s in range(n_seg + 1):
            t = math.pi * s / n_seg
            xs.append(r * math.cos(t)); ys.append(y); zs.append(r * math.sin(t))
        xs.append(None); ys.append(None); zs.append(None)
    for k in range(n_long):                             # longitudinal ribs (incl. ground edges)
        t = math.pi * k / (n_long - 1)
        xs += [r * math.cos(t), r * math.cos(t), None]
        ys += [-HL, HL, None]
        zs += [r * math.sin(t), r * math.sin(t), None]
    return xs, ys, zs


def wireframe_3d(spec: HabitatSpec) -> go.Figure:
    ri, ro = spec.inner_radius_cm / 100.0, spec.outer_radius_cm / 100.0
    fig = go.Figure()
    if spec.shape == "cylinder":
        H = spec.effective_height_cm / 100.0
        xi, yi, zi = _cylinder_lines(ri, H)
        xo, yo, zo = _cylinder_lines(ro, H + spec.total_wall_cm / 100.0)
        gx = [ro * math.cos(2 * math.pi * s / 72) for s in range(73)]
        gy = [ro * math.sin(2 * math.pi * s / 72) for s in range(73)]
        gz = [0] * 73
    elif spec.shape == "quonset":
        HL = spec.effective_height_cm / 200.0           # half-length, m
        xi, yi, zi = _arch_lines(ri, HL)
        xo, yo, zo = _arch_lines(ro, HL)
        gx = [-ro, ro, ro, -ro, -ro]                    # ground footprint rectangle
        gy = [-HL, -HL, HL, HL, -HL]
        gz = [0, 0, 0, 0, 0]
    else:                                               # dome
        xi, yi, zi = _hemisphere_lines(ri)
        xo, yo, zo = _hemisphere_lines(ro)
        gx = [ro * math.cos(2 * math.pi * s / 72) for s in range(73)]
        gy = [ro * math.sin(2 * math.pi * s / 72) for s in range(73)]
        gz = [0] * 73
    fig.add_trace(go.Scatter3d(x=xi, y=yi, z=zi, mode="lines",
                  line=dict(color="#9aa7b3", width=2), name="Inner surface"))
    fig.add_trace(go.Scatter3d(x=xo, y=yo, z=zo, mode="lines",
                  line=dict(color=METRIC, width=2), name="Outer surface"))
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines",
                  line=dict(color=GROUND, width=1), name="Ground level"))
    ax = dict(backgroundcolor=BG, gridcolor="#1d242d", zerolinecolor="#2a323d",
              color=MUTED, showspikes=False)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, height=470,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", x=0, y=1.04, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        scene=dict(aspectmode="data",
                   xaxis=dict(title="X (m)", **ax),
                   yaxis=dict(title="Y (m)", **ax),
                   zaxis=dict(title="Z (m)", **ax),
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))))
    return fig


# ----------------------------------------------------------------------
# 2-D dose cross-section (vertical slice through the dome)
# ----------------------------------------------------------------------
def _semi_annulus(ri, ro, colour, name):
    th = [i * math.pi / 60 for i in range(61)]
    xo = [ro * math.cos(t) for t in th]
    yo = [ro * math.sin(t) for t in th]
    xi = [ri * math.cos(t) for t in reversed(th)]
    yi = [ri * math.sin(t) for t in reversed(th)]
    return go.Scatter(x=xo + xi, y=yo + yi, fill="toself", mode="lines",
                      line=dict(color=BG, width=0.5), fillcolor=colour,
                      name=name, hoverinfo="name")


def _semi_disc(r, colour):
    """Filled HALF-disc sitting on the floor (y=0). A dome's cavity is a
    hemisphere -- drawing it as a full circle put half the interior underground
    and made the crew phantom look like it was floating."""
    th = [i * math.pi / 60 for i in range(61)]
    return go.Scatter(x=[r * math.cos(t) for t in th],
                      y=[r * math.sin(t) for t in th],
                      fill="toself", mode="lines",
                      line=dict(color=colour, width=0.5), fillcolor=colour,
                      hoverinfo="skip", showlegend=False)


def _rect_trace(x0, x1, y0, y1, colour, name, legend=True):
    return go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
                      fill="toself", mode="lines",
                      line=dict(color=BG, width=0.5), fillcolor=colour,
                      name=name, hoverinfo="name", showlegend=legend)


def _cross_arch(spec: HabitatSpec, dose=None) -> go.Figure:
    """Radial slice: a half-arch (dome and quonset share this cross-section)."""
    fig = go.Figure()
    outer = spec.outer_radius_cm
    span = outer * 1.25
    fig.add_shape(type="rect", x0=-span, x1=span, y0=-span * 0.35, y1=0,
                  fillcolor=MATERIALS["regolith"]["colour"], line_width=0, layer="below")
    fig.add_trace(_semi_disc(spec.inner_radius_cm, "#0b1d2e"))
    for i, ((ri, ro), w) in enumerate(zip(spec.layer_radii_cm(), spec.walls)):
        fig.add_trace(_semi_annulus(ri, ro, MATERIALS[w.material]["colour"],
                                    f"L{i}: {w.material} {w.thickness_cm:g} cm"))
    cz, pr = spec.crew_height_cm, spec.phantom_radius_cm   # matches geometry.py
    _add_scorers(fig, spec, dose, cz, pr, lining_r=spec.inner_radius_cm,
                 label_x=spec.inner_radius_cm * 0.62)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=10), height=470,
        showlegend=True, legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(visible=False, range=[-span, span], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-span * 0.35, span]))
    return fig


def _add_scorers(fig, spec, dose, cz, pr, lining_r, label_x, lining_y=None):
    """Draw the two things the simulation ACTUALLY scores -- the inner-wall lining
    and the central crew phantom -- and label them with real dose when a finished
    run matches this geometry.

    There is no spatially-resolved dose field to paint: TOPAS scores exactly these
    two locations (skin_doseeq_sv, phantom_doseeq_sv), so the figure shows where
    dose is measured and what was measured, rather than a fabricated gradient.
    """
    unit = (dose or {}).get("unit", "")
    skin_v = (dose or {}).get("skin")
    phan_v = (dose or {}).get("phantom")

    # --- crew phantom: label OUTSIDE the circle on a leader line so it stays
    # readable no matter how small the phantom is drawn ----------------------
    fig.add_shape(type="circle", x0=-pr, x1=pr, y0=cz - pr, y1=cz + pr,
                  fillcolor="#6fb3ff", line=dict(color="#cfe8ff", width=1))
    ptxt = "crew phantom"
    if phan_v is not None:
        ptxt += f"<br><b>{phan_v:,.0f}</b> {unit}"
    fig.add_annotation(x=pr * 0.7, y=cz + pr * 0.7, text=ptxt,
                       showarrow=True, arrowhead=0, arrowsize=1, arrowwidth=1,
                       arrowcolor="#cfe8ff", ax=48, ay=-34,
                       xanchor="left", align="left",
                       font=dict(color="#cfe8ff", size=10),
                       bgcolor="rgba(11,29,46,0.85)", borderpad=3)

    # --- inner-wall lining: the headline scorer, a thin ring on the whole
    # inner surface ---------------------------------------------------------
    ly = lining_r if lining_y is None else lining_y
    stxt = "inner-wall lining (skin)"
    if skin_v is not None:
        stxt += f"<br><b>{skin_v:,.0f}</b> {unit}"
    fig.add_annotation(x=label_x, y=ly * 0.78 if lining_y is None else ly,
                       text=stxt, showarrow=True, arrowhead=0, arrowwidth=1,
                       arrowcolor="#ffd479", ax=30, ay=-26,
                       xanchor="left", align="left",
                       font=dict(color="#ffd479", size=10),
                       bgcolor="rgba(11,29,46,0.85)", borderpad=3)

    if dose is None:
        fig.add_annotation(
            x=0, y=-0.06, xref="paper", yref="paper", showarrow=False,
            text="No dose overlaid — press Evaluate to score this design",
            font=dict(color=MUTED, size=10), xanchor="left")
    else:
        fig.add_annotation(
            x=0, y=-0.06, xref="paper", yref="paper", showarrow=False,
            text=f"Dose at the two scored locations · {dose.get('scenario','')}",
            font=dict(color=MUTED, size=10), xanchor="left")


def _cross_cylinder(spec: HabitatSpec, dose=None) -> go.Figure:
    """Axial slice through a vertical cylinder: rectangular interior, side walls,
    and a flat layered roof."""
    fig = go.Figure()
    inner, outer = spec.inner_radius_cm, spec.outer_radius_cm
    H = spec.effective_height_cm
    span = outer * 1.25
    top = (H + spec.total_wall_cm) * 1.12
    fig.add_shape(type="rect", x0=-span, x1=span, y0=-span * 0.35, y1=0,
                  fillcolor=MATERIALS["regolith"]["colour"], line_width=0, layer="below")
    fig.add_trace(_rect_trace(-inner, inner, 0, H, "#0b1d2e", "interior", legend=False))
    for i, ((ri, ro), w) in enumerate(zip(spec.layer_radii_cm(), spec.walls)):
        c = MATERIALS[w.material]["colour"]
        lab = f"L{i}: {w.material} {w.thickness_cm:g} cm"
        fig.add_trace(_rect_trace(ri, ro, 0, H, c, lab))                 # right wall (legend)
        fig.add_trace(_rect_trace(-ro, -ri, 0, H, c, lab, legend=False))  # left wall
        below = ri - inner
        fig.add_trace(_rect_trace(-outer, outer, H + below, H + below + w.thickness_cm,
                                  c, lab, legend=False))                  # roof cap
    cz, pr = spec.crew_height_cm, spec.phantom_radius_cm   # matches geometry.py
    _add_scorers(fig, spec, dose, cz, pr, lining_r=inner,
                 label_x=inner * 0.55, lining_y=H * 0.80)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=10), height=470,
        showlegend=True, legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(visible=False, range=[-span, span], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-span * 0.35, top]))
    return fig


def _spec_sig(spec: HabitatSpec) -> str:
    """Identity of a design. A dose overlay is only drawn when its signature still
    matches the design on screen, so editing the wall can never leave last run's
    numbers sitting on a geometry they were not scored against."""
    return (f"{spec.shape}|{spec.inner_radius_cm:.3f}|"
            + ";".join(f"{w.material}:{w.thickness_cm:.4f}" for w in spec.walls))


def cross_section(spec: HabitatSpec, dose=None) -> go.Figure:
    if spec.shape == "cylinder":
        return _cross_cylinder(spec, dose)
    return _cross_arch(spec, dose)  # dome + quonset share the half-arch slice


# ----------------------------------------------------------------------
# App  + CSS (dark dropdowns, red sliders, native disclosure)
# ----------------------------------------------------------------------
app = Dash(__name__, title="Radiation Simulation Tool",
           suppress_callback_exceptions=True)
server = app.server

# Theme CSS lives in assets/lunarsim.css (Dash auto-serves the assets folder;
# a custom index_string did not load reliably under Dash 3.x).


# ----------------------------------------------------------------------
# Reusable bits
# ----------------------------------------------------------------------
def slider_field(label, value_id, slider):
    return html.Div(style={"marginBottom": "18px"}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "alignItems": "baseline"}, children=[
            html.Span(label, style=FIELD_LABEL),
            html.Span(id=value_id, style={"color": ACCENT, "fontSize": "13px",
                                          "fontWeight": 700})]),
        slider,
    ])


def metric_card(label, value, sub, value_color=METRIC, accent=False):
    style = dict(CARD)
    if accent:
        style["borderLeft"] = f"3px solid {ACCENT}"
    return html.Div(style=style, children=[
        html.Div(label, style={"color": MUTED, "fontSize": "12px", "marginBottom": "8px"}),
        html.Div(value, style={"color": value_color, "fontSize": "28px",
                               "fontWeight": 800, "marginBottom": "6px"}),
        html.Div(sub, style={"color": MUTED, "fontSize": "11px"}),
    ])


def _kv(k, v):
    return html.Div(style={"display": "flex", "justifyContent": "space-between",
                           "fontSize": "13px", "padding": "5px 0",
                           "borderBottom": f"1px solid {BORDER}"}, children=[
        html.Span(k, style={"color": MUTED}), html.Span(v, style={"color": INK})])


PLACEHOLDER_METRICS = [
    html.Div(style={**CARD, "borderLeft": f"4px solid {BORDER}",
                    "background": "#161d27", "padding": "18px"}, children=[
        html.Div("RADIATION PROTECTION SCORE", style={
            "color": MUTED, "fontSize": "11px", "letterSpacing": "0.8px",
            "fontWeight": 700, "marginBottom": "8px"}),
        html.Div("—", style={"color": METRIC, "fontSize": "42px", "fontWeight": 900,
                             "lineHeight": "1"}),
        html.Div("Evaluate a design to compute its score (mSv/yr).",
                 style={"color": MUTED, "fontSize": "11px", "marginTop": "10px"})]),
    metric_card("Absorbed dose", "—", "evaluate a design to populate"),
    metric_card("Dose equivalent", "—", "ISS baseline = 0.70 mSv/day"),
    metric_card("Crew-phantom point dose", "—", "central diagnostic (not the score)"),
]


# ----------------------------------------------------------------------
# Left rail
# ----------------------------------------------------------------------
sidebar = html.Div(style={"width": "270px", "minWidth": "270px", "padding": "22px 20px",
                          "background": RAIL, "borderRight": f"1px solid {BORDER}",
                          "height": "100vh", "overflowY": "auto", "boxSizing": "border-box"},
                   children=[
    html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px"}, children=[
        html.Span("☢", style={"color": ACCENT, "fontSize": "20px"}),
        html.Span("Radiation Sim", style={"color": INK, "fontSize": "18px",
                                          "fontWeight": 800})]),
    html.P("Galactic Cosmic Ray exposure tool for conceptual habitat design",
           style={"color": MUTED, "fontSize": "12px", "lineHeight": "1.5",
                  "marginTop": "8px"}),
    html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),

    html.Div("Habitat Geometry", style=SECTION),
    html.Div("Habitat type", style=FIELD_LABEL),
    dcc.Dropdown(id="shape", value="dome", clearable=False,
                 options=[{"label": SHAPE_LABELS[s], "value": s} for s in SHAPES],
                 style={"marginBottom": "16px"}),
    slider_field("Inner radius (m)", "inner-r-val",
                 dcc.Slider(id="inner-r", min=1.5, max=8.0, step=0.05, value=2.5,
                            marks=_marks([2, 3, 4, 5, 6, 7, 8]), tooltip=None)),
    # Axial length — only meaningful for the two elongated shapes, so this field
    # is shown for cylinder/quonset and hidden for the dome (see _length_control).
    html.Div(id="length-field", style={"display": "none"}, children=[
        slider_field("Axial length (m)", "length-val",
                     dcc.Slider(id="length-slider", min=2.0, max=12.0, step=0.5,
                                value=6.0, marks=_marks([2, 4, 6, 8, 10, 12]),
                                tooltip=None))]),

    html.Div("Wall Layers", style=SECTION),
    html.Div("Innermost first. Add the structural shell, insulation and regolith "
             "overburden as your team designed them.",
             style={"color": MUTED, "fontSize": "11px", "lineHeight": "1.5",
                    "marginBottom": "12px"}),
    html.Div(id="layer-rows", children=[
        _layer_row(i,
                   DEFAULT_LAYERS[i]["m"] if i < len(DEFAULT_LAYERS) else "regolith",
                   DEFAULT_LAYERS[i]["t"] if i < len(DEFAULT_LAYERS) else 100.0,
                   list(range(len(DEFAULT_LAYERS))))
        for i in range(MAX_LAYERS)]),
    html.Button("+  Add layer", id="add-layer", n_clicks=0, style={
        "background": "transparent", "color": INK, "border": f"1px dashed {BORDER}",
        "borderRadius": "8px", "padding": "9px", "cursor": "pointer", "fontSize": "13px",
        "width": "100%", "marginTop": "2px"}),
    dcc.Store(id="active-rows", data=list(range(len(DEFAULT_LAYERS)))),

    html.Div("Exposure Scenario", style=SECTION),
    dcc.RadioItems(id="scenario", value="both",
                   options=[
                       {"label": " GCR + SPE", "value": "both"},
                       {"label": " GCR only", "value": "gcr"},
                       {"label": " Solar Particle Event only", "value": "spe"}],
                   style={"color": INK, "fontSize": "13px"},
                   labelStyle={"display": "flex", "alignItems": "center",
                               "marginBottom": "6px", "color": INK},
                   inputStyle={"marginRight": "8px", "accentColor": ACCENT}),
    html.Div(id="scenario-note",
             style={"color": MUTED, "fontSize": "11px", "lineHeight": "1.5",
                    "margin": "4px 0 0"}),

    html.Div("Scoring Conditions", style=SECTION),
    html.Div(id="scoring-conditions", style={**CARD, "padding": "12px"}),

    html.Button("▶  Evaluate protection", id="run", n_clicks=0, style={
        "background": ACCENT, "color": "#fff", "border": "none", "borderRadius": "8px",
        "padding": "12px", "cursor": "pointer", "fontWeight": 700, "fontSize": "14px",
        "width": "100%", "marginTop": "22px"}),
    html.Button("Cancel run", id="cancel", style={
        "background": "transparent", "color": MUTED, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "8px", "cursor": "pointer", "fontSize": "12px",
        "width": "100%", "marginTop": "8px"}),

    html.Div("Design File", style=SECTION),
    html.Div(style={"display": "flex", "gap": "8px"}, children=[
        html.Button("⭳ Save", id="save-design", n_clicks=0, style={
            "flex": "1", "background": "transparent", "color": INK,
            "border": f"1px solid {BORDER}", "borderRadius": "8px",
            "padding": "8px", "cursor": "pointer", "fontSize": "12px"}),
        dcc.Upload(id="load-design", style={"flex": "1"}, children=html.Div(
            "⭱ Load", style={"background": "transparent", "color": INK,
                             "border": f"1px dashed {BORDER}", "borderRadius": "8px",
                             "padding": "8px", "cursor": "pointer", "fontSize": "12px",
                             "textAlign": "center"})),
    ]),
    html.Button("📄 Download results report", id="save-report", n_clicks=0, style={
        "background": "transparent", "color": INK, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "8px", "cursor": "pointer", "fontSize": "12px",
        "width": "100%", "marginTop": "8px"}),
    html.Div(id="design-file-note",
             style={"color": MUTED, "fontSize": "11px", "marginTop": "6px",
                    "minHeight": "14px"}),
    dcc.Download(id="download-design"),
    dcc.Download(id="download-report"),
])


# ----------------------------------------------------------------------
# Centre column  (header, tabs, tab panels)
# ----------------------------------------------------------------------
TAB_STYLE = {"backgroundColor": "transparent", "border": "none", "color": MUTED,
             "padding": "10px 4px", "marginRight": "22px", "fontSize": "14px"}
TAB_SELECTED = {"backgroundColor": "transparent", "border": "none",
                "borderBottom": f"2px solid {ACCENT}", "color": ACCENT,
                "padding": "10px 4px", "marginRight": "22px", "fontSize": "14px",
                "fontWeight": 700}

overview_panel = html.Div(id="panel-overview", children=[
    dcc.RadioItems(id="view-mode", value="wire", inline=True,
                   options=[{"label": " Shielding cross-section", "value": "cross"},
                            {"label": " 3-D wireframe", "value": "wire"}],
                   style={"color": INK, "fontSize": "13px"},
                   labelStyle={"marginRight": "20px", "color": INK,
                               "display": "inline-flex", "alignItems": "center"},
                   inputStyle={"marginRight": "6px", "accentColor": ACCENT}),
    html.Div(id="view-title", style={"color": INK, "fontSize": "14px",
                                     "fontWeight": 700, "margin": "16px 0 4px"}),
    dcc.Graph(id="habitat-view", config={"displayModeBar": False}),

    html.Hr(style={"borderColor": BORDER, "margin": "22px 0 16px"}),
    html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px",
                    "flexWrap": "wrap"}, children=[
        html.Button("☄  Visualise particle cascade", id="run-cascade", n_clicks=0,
                    style={"background": "transparent", "color": ACCENT,
                           "border": f"1px solid {ACCENT}", "borderRadius": "8px",
                           "padding": "10px 16px", "cursor": "pointer",
                           "fontWeight": 700, "fontSize": "13px"}),
        dcc.RadioItems(id="cascade-colour", value="origin", inline=True,
                       options=[{"label": " In vs out (origin)", "value": "origin"},
                                {"label": " By particle", "value": "family"}],
                       style={"color": INK, "fontSize": "13px"},
                       labelStyle={"marginRight": "18px", "color": INK,
                                   "display": "inline-flex", "alignItems": "center"},
                       inputStyle={"marginRight": "6px", "accentColor": ACCENT}),
    ]),
    html.Div("Headless Monte-Carlo: launches GCR primaries onto this habitat and "
             "traces the secondary shower through the shielding (no live viewer "
             "needed). First run takes a few seconds.",
             style={"color": MUTED, "fontSize": "11px", "margin": "8px 0 4px"}),
    html.Div(id="cascade-status", style={"color": MUTED, "fontSize": "12px",
                                         "height": "18px", "margin": "6px 0"}),
    dcc.Loading(type="default", color=ACCENT, children=dcc.Graph(
        id="cascade-view", config={"displayModeBar": True},
        style={"height": "520px"}, figure=go.Figure(
            layout=dict(paper_bgcolor=BG, plot_bgcolor=BG, height=520,
                        margin=dict(l=0, r=0, t=0, b=0))))),
    dcc.Store(id="cascade-dir", data=None),
])

gcr_panel = html.Div(id="panel-gcr", style={"display": "none"}, children=[
    html.Div("GCR Environment", style={"color": INK, "fontSize": "15px",
                                       "fontWeight": 700, "marginBottom": "12px"}),
    html.Div(style=CARD, children=[
        html.P("Source: force-field-modulated Local Interstellar Spectrum "
               "(Usoskin 2005), sampled as an isotropic upper hemisphere at the "
               "lunar surface and transported with FTFP_BERT_HP. The proton "
               "spectrum below is the reference; the scored run transports the "
               "full weighted GCR ion composition — protons plus the heavier ions "
               "(He, C, O … up to Fe), whose per-ion dose shares appear in the "
               "Dose Analysis breakdown.", style={"color": MUTED,
                                                   "fontSize": "13px",
                                                   "lineHeight": "1.6"}),
        _kv("Solar modulation φ", "400 MV (solar minimum, worst case)"),
        _kv("Integral proton flux", f"{gcr_scalar_fluence_rate(400.0):.2f} /cm²/s"),
        _kv("Ion composition", "H + He … Fe (weighted GCR abundances)"),
        _kv("Angular distribution", "isotropic over 2π sr (sky dome)"),
    ]),
])

dose_panel = html.Div(id="panel-dose", style={"display": "none"}, children=[
    html.Div("Dose Analysis", style={"color": INK, "fontSize": "15px",
                                     "fontWeight": 700, "marginBottom": "12px"}),
    html.Div(id="dose-analysis-body", style=CARD, children=[
        html.Div("Run a simulation to see the per-quantity breakdown.",
                 style={"color": MUTED, "fontSize": "13px"})]),
])

centre = html.Div(style={"flex": "1", "padding": "26px 30px", "overflowY": "auto",
                         "height": "100vh", "boxSizing": "border-box"}, children=[
    html.H1("Radiation Simulation Tool", style={"color": INK, "fontSize": "30px",
                                                "fontWeight": 800, "margin": 0}),
    html.Div(id="subtitle", style={"color": MUTED, "fontSize": "13px",
                                   "margin": "8px 0 4px"}),
    # thin progress line
    html.Div(style={"background": "#11161d", "borderRadius": "4px", "height": "3px",
                    "overflow": "hidden", "margin": "10px 0 0"}, children=[
        html.Div(id="progress-bar", style={"background": ACCENT, "height": "100%",
                                           "width": "0%", "transition": "width .3s"})]),
    html.Div(id="status-line", style={"color": MUTED, "fontSize": "11px",
                                      "height": "16px", "marginBottom": "8px"}),
    dcc.Tabs(id="tabs", value="overview", style={"borderBottom": f"1px solid {BORDER}"},
             children=[
        dcc.Tab(label="\U0001F6F0 Habitat Overview", value="overview",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="\U0001F4CA GCR Environment", value="gcr",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="\U0001F4CB Dose Analysis", value="dose",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
    ]),
    html.Div(style={"marginTop": "18px"}, children=[
        overview_panel, gcr_panel, dose_panel]),
])


# ----------------------------------------------------------------------
# Right column  (Dose Metrics + Design Parameters)
# ----------------------------------------------------------------------
right = html.Div(style={"width": "320px", "minWidth": "320px", "padding": "26px 22px",
                        "background": RAIL, "borderLeft": f"1px solid {BORDER}",
                        "height": "100vh", "overflowY": "auto", "boxSizing": "border-box"},
                 children=[
    html.Div("Dose Metrics", style={"color": INK, "fontSize": "18px",
                                    "fontWeight": 800, "marginBottom": "16px"}),
    html.Div(id="dose-metrics-body", children=PLACEHOLDER_METRICS),
    html.Div("Design Parameters", style={"color": INK, "fontSize": "16px",
                                         "fontWeight": 700, "margin": "22px 0 12px"}),
    html.Div(id="design-params", style=CARD),
])

app.layout = html.Div(style={"display": "flex", "background": BG}, children=[
    sidebar, centre, right,
    dcc.Store(id="job-store", data=None),
    dcc.Store(id="report-store", data=None),
    dcc.Store(id="dose-overlay", data=None),
    dcc.Interval(id="poll", interval=1500, disabled=True),
])


# ----------------------------------------------------------------------
# Callbacks: tab switching
# ----------------------------------------------------------------------
@app.callback(
    Output("panel-overview", "style"), Output("panel-gcr", "style"),
    Output("panel-dose", "style"),
    Input("tabs", "value"))
def _switch_tab(tab):
    show, hide = {"display": "block"}, {"display": "none"}
    return (show if tab == "overview" else hide,
            show if tab == "gcr" else hide,
            show if tab == "dose" else hide)


# ----------------------------------------------------------------------
# Callbacks: exposure-scenario copy (GCR chronic vs acute SPE)
# ----------------------------------------------------------------------
# The scenario radio only changes WHICH job runs and how the result is judged;
# both use the same fixed statistics preset so the two verdicts stay comparable.
@app.callback(
    Output("scoring-conditions", "children"), Output("scenario-note", "children"),
    Input("scenario", "value"))
def _scenario_conditions(scenario):
    if scenario == "spe":
        note = ("A single worst-case solar proton event — the acute test. Scored as "
                "the TOTAL event dose against the 30-day blood-forming-organ limit, "
                "not a per-year rate.")
        rows = [
            _kv("🔒 Event", WORST_CASE_SPE.name),
            _kv("🔒 Fluence", f"{WORST_CASE_SPE.fluence_cm2:.0e} p/cm²"),
            _kv("🔒 Limit", f"{DOSE_LIMITS_MSV['nasa_30day']:.0f} mSv (30-day BFO)"),
        ]
    elif scenario == "both":
        note = ("One run, both hazards: the chronic GCR field AND a worst-case solar "
                "event, judged as two SEPARATE pass/fail gates (annual/career limit vs "
                "the 30-day limit). The two doses are never summed — they reward "
                "different shielding, so a design must clear both.")
        rows = [
            _kv("🔒 GCR field", "φ=400 MV (solar min)"),
            _kv("🔒 SPE event", WORST_CASE_SPE.name),
            _kv("🔒 Gates", "career limit  +  30-day BFO limit"),
            _kv("🔒 Statistics", "converged full run"),
        ]
    else:
        note = ("The chronic galactic-cosmic-ray field at solar minimum, integrated "
                "over the whole mission and judged against the annual / career limits.")
        rows = [
            _kv("🔒 GCR field", "φ=400 MV (solar min)"),
            _kv("🔒 Mission", f"{SCORING_MISSION_DAYS} days"),
            _kv("🔒 Statistics", "converged full run"),
        ]
    return rows, note


# ----------------------------------------------------------------------
# Callbacks: wall-layer stack (add / remove / render)
# ----------------------------------------------------------------------
# The layer rows are a fixed pool baked into the layout, so their values are
# always sent to the back end. `active-rows` lists which pool slots are in the
# stack, innermost-first. Add appends the lowest free slot; remove drops the
# clicked slot from the list. Neither touches any row's value, so typing in a
# field is never overwritten by a "controlled value" re-render or steals focus.
@app.callback(
    Output("active-rows", "data"),
    Input("add-layer", "n_clicks"),
    Input({"type": "layer-remove", "index": ALL}, "n_clicks"),
    State("active-rows", "data"),
    prevent_initial_call=True)
def _manage_layers(add_n, remove_n, active):
    active = list(active or [0])
    trig = ctx.triggered_id

    if trig == "add-layer" and add_n:
        free = next((i for i in range(MAX_LAYERS) if i not in active), None)
        if free is None:
            return no_update                       # pool full
        return active + [free]

    if (isinstance(trig, dict) and trig.get("type") == "layer-remove"
            and any(remove_n) and len(active) > 1):
        return [i for i in active if i != trig["index"]]

    return no_update                               # mount/no-op


@app.callback(
    Output({"type": "layer-box", "index": ALL}, "style"),
    Output({"type": "layer-label", "index": ALL}, "children"),
    Output({"type": "layer-remove", "index": ALL}, "disabled"),
    Output({"type": "layer-remove", "index": ALL}, "style"),
    Input("active-rows", "data"))
def _render_layers(active):
    active = list(active or [0])
    boxes, labels, disabled, rm_styles = [], [], [], []
    for i in range(MAX_LAYERS):
        box, label, rm_disabled, rm_style = _layer_state(i, active)
        boxes.append(box)
        labels.append(label)
        disabled.append(rm_disabled)
        rm_styles.append(rm_style)
    return boxes, labels, disabled, rm_styles


# ----------------------------------------------------------------------
# Callbacks: live geometry / labels / subtitle / design parameters
# ----------------------------------------------------------------------
@app.callback(
    Output("length-field", "style"),
    Output("length-val", "children"),
    Input("shape", "value"), Input("length-slider", "value"))
def _length_control(shape, length):
    """Length is an axial dimension only the two elongated shapes have. Show the
    slider for cylinder/quonset; hide it for the dome (which uses its radius)."""
    show = shape in ("cylinder", "quonset")
    return ({} if show else {"display": "none"}), f"{float(length or 6.0):.1f}"


@app.callback(
    Output("habitat-view", "figure"), Output("view-title", "children"),
    Output("subtitle", "children"), Output("design-params", "children"),
    Output("inner-r-val", "children"),
    Input("shape", "value"), Input("inner-r", "value"),
    Input({"type": "layer-mat", "index": ALL}, "value"),
    Input({"type": "layer-thk", "index": ALL}, "value"),
    Input("view-mode", "value"),
    Input("dose-overlay", "data"),
    Input("length-slider", "value"),
    State({"type": "layer-mat", "index": ALL}, "id"),
    State("active-rows", "data"))
def _live(shape, inner_r, mats, thks, view_mode, overlay, length, ids, active):
    r_label = f"{float(inner_r or 2.5):.2f}"
    layers = _layers_from_components(mats, ids, thks, active)
    if not any(L["t"] > 0 for L in layers):
        # Every active layer's thickness is blank/zero (e.g. mid-edit). Don't
        # fabricate a fake wall and show a misleading score -- say what's wrong.
        return (no_update, no_update,
                "⚠ Enter a wall thickness (mm) for at least one layer",
                no_update, r_label)
    try:
        spec = spec_from_inputs("design", shape, inner_r, layers, length)
        spec.validate()
    except Exception:
        return (no_update, no_update, no_update, no_update, r_label)

    # Only overlay dose that was actually scored against THIS geometry.
    dose = overlay if (overlay and overlay.get("sig") == _spec_sig(spec)) else None
    fig = wireframe_3d(spec) if view_mode == "wire" else cross_section(spec, dose)
    if view_mode == "wire":
        title = f"3-D Habitat Model — {SHAPE_LABELS[spec.shape]}"
    elif dose:
        title = (f"Dose Cross-section — {SHAPE_LABELS[spec.shape]} "
                 f"· {dose.get('scenario', '')}")
    else:
        title = f"Shielding Cross-section — {SHAPE_LABELS[spec.shape]}"

    stack = " + ".join(f"{w.thickness_cm * 10:g} mm {w.material}" for w in spec.walls)
    subtitle = (f"Radiation protection score · {SHAPE_LABELS[spec.shape]} "
                f"| {stack} · φ=400 MV · {SCORING_MISSION_DAYS}-day lunar mission")

    params = [
        _kv("Inner radius", f"{spec.inner_radius_cm / 100:.2f} m"),
    ]
    if spec.shape in ("cylinder", "quonset"):
        axis = "Axial length" if spec.shape == "cylinder" else "Tunnel length"
        params.append(_kv(axis, f"{spec.effective_height_cm / 100:.2f} m"))
    params += [
        _kv("Wall stack", stack),
        _kv("Areal density", f"{spec.areal_density_gcm2():.1f} g/cm²"),
        _kv("Outer radius", f"{spec.outer_radius_cm / 100:.2f} m"),
        _kv("Shell mass", f"{spec.shell_mass_kg() / 1000:.1f} t"),
        _kv("Mission", f"{SCORING_MISSION_DAYS} days"),
    ]
    return fig, title, subtitle, params, r_label


# ----------------------------------------------------------------------
# Callbacks: particle-cascade visualisation (headless TOPAS -> Plotly)
# ----------------------------------------------------------------------
@app.callback(
    Output("cascade-view", "figure"), Output("cascade-dir", "data"),
    Output("cascade-status", "children"),
    Input("run-cascade", "n_clicks"), Input("cascade-colour", "value"),
    State("shape", "value"), State("inner-r", "value"),
    State({"type": "layer-mat", "index": ALL}, "value"),
    State({"type": "layer-mat", "index": ALL}, "id"),
    State({"type": "layer-thk", "index": ALL}, "value"),
    State("active-rows", "data"),
    State("length-slider", "value"),
    State("cascade-dir", "data"),
    prevent_initial_call=True)
def _cascade(n, colour_by, shape, inner_r, mats, ids, thks, active, length, cascade_dir):
    spec = spec_from_inputs("habitat", shape, inner_r,
                            _layers_from_components(mats, ids, thks, active), length)
    trigger = ctx.triggered_id

    if trigger == "cascade-colour":
        # cheap re-render of the existing run in the new colour scheme
        if not cascade_dir:
            return no_update, no_update, no_update
        fig = build_figure(cascade_dir, spec=spec, colour_by=colour_by,
                           title=CASCADE_TITLE)
        return _style_cascade(fig), cascade_dir, no_update

    # button: run a fresh headless cascade for this design
    try:
        html_path = run_cascade(spec, n_histories=100, colour_by=colour_by)
    except Exception as exc:
        return no_update, no_update, html.Span(
            f"cascade failed: {str(exc)[-200:]}", style={"color": ACCENT})
    run_dir = str(html_path.parent)
    fig = build_figure(run_dir, spec=spec, colour_by=colour_by,
                       title=CASCADE_TITLE)
    n_tracks = len(fig.data) and sum(
        1 for tr in fig.data if getattr(tr, "mode", "") == "lines")
    status = (f"cascade for '{spec.name}' ({spec.areal_density_gcm2():.0f} g/cm² "
              f"shielding) — drag to orbit, scroll to zoom")
    return _style_cascade(fig), run_dir, status


# Short on purpose: a left-aligned title runs rightward into the modebar icons,
# and the status line under the plot already names the design and its g/cm2.
# trajviz keeps its own longer default for the standalone HTML export.
CASCADE_TITLE = "Habitat radiation cascade"


def _style_cascade(fig):
    """Match the cascade figure to the app theme and frame, and pull the camera
    back so the whole hemisphere fits (default 3-D camera clips the bounding box)."""
    # The legend gets its own RIGHT-HAND COLUMN, never the top margin. A
    # horizontal legend up top wraps to as many rows as there are entries and
    # grows UPWARD through the title -- and the entry count is data-dependent
    # (colour_by="family" emits one per particle family, several more than
    # "origin"), so no fixed top margin can be deep enough. Stacked vertically on
    # the right it grows downward into 560px of free height instead, and the
    # title owns the top band alone.
    fig.update_layout(
        paper_bgcolor=BG, height=560,
        margin=dict(l=0, r=190, t=44, b=0),
        title=dict(x=0, xanchor="left", y=1, yanchor="top",
                   font=dict(size=14), pad=dict(t=12, l=6)),
        scene=dict(bgcolor=BG, aspectmode="data",
                   camera=dict(eye=dict(x=1.7, y=1.7, z=1.1))),
        legend=dict(orientation="v", x=1.0, xanchor="left", y=1, yanchor="top",
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)",
                    itemsizing="constant"))
    return fig


# ----------------------------------------------------------------------
# Callbacks: run / cancel / poll
# ----------------------------------------------------------------------
@app.callback(
    Output("job-store", "data"), Output("poll", "disabled"),
    Output("status-line", "children", allow_duplicate=True),
    Input("run", "n_clicks"),
    State("shape", "value"), State("inner-r", "value"),
    State({"type": "layer-mat", "index": ALL}, "value"),
    State({"type": "layer-mat", "index": ALL}, "id"),
    State({"type": "layer-thk", "index": ALL}, "value"),
    State("active-rows", "data"), State("scenario", "value"),
    State("length-slider", "value"),
    prevent_initial_call=True)
def _evaluate(n, shape, inner_r, mats, ids, thks, active, scenario, length):
    layers = _layers_from_components(mats, ids, thks, active)
    if not any(L["t"] > 0 for L in layers):
        # No layer has a thickness -- refuse rather than score a fabricated wall.
        return no_update, True, "⚠ Enter a wall thickness (mm) for at least one layer before evaluating."
    spec = spec_from_inputs("habitat", shape, inner_r, layers, length)
    if scenario == "spe":
        # One acute event: a single proton cone at the fixed event fluence.
        # Converge on the wall lining (the skin/BFO surface); no ion composition.
        jid = default_runner.submit(spec, tier=SCORING_TIER,
                                    target_rel_err=SCORING_TARGET_REL_ERR,
                                    max_batches=SCORING_MAX_BATCHES,
                                    converge_on="skin", composition=False,
                                    spe=WORST_CASE_SPE)
    elif scenario == "gcr":
        # Chronic GCR: fixed scoring preset -- identical conditions for every team.
        jid = default_runner.submit(spec, tier=SCORING_TIER,
                                    target_rel_err=SCORING_TARGET_REL_ERR,
                                    max_batches=SCORING_MAX_BATCHES,
                                    converge_on=SCORING_CONVERGE_ON,
                                    phantom_slack=SCORING_PHANTOM_SLACK,
                                    min_batches=SCORING_MIN_BATCHES,
                                    composition=True)   # protons + heavy ions, summed
    else:
        # Workshop default: BOTH gates from one GCR run -- the chronic GCR field
        # (MC) plus the acute SPE gate (folded from job.result.spec, no MC of its
        # own), reported as two separate verdicts (never summed).
        jid = default_runner.submit(spec, tier=SCORING_TIER,
                                    target_rel_err=SCORING_TARGET_REL_ERR,
                                    max_batches=SCORING_MAX_BATCHES,
                                    converge_on=SCORING_CONVERGE_ON,
                                    phantom_slack=SCORING_PHANTOM_SLACK,
                                    min_batches=SCORING_MIN_BATCHES,
                                    composition=True, spe=WORST_CASE_SPE,
                                    combined=True)
    return jid, False, no_update


@app.callback(Output("cancel", "n_clicks"), Input("cancel", "n_clicks"),
              State("job-store", "data"), prevent_initial_call=True)
def _cancel(n, jid):
    if jid:
        default_runner.cancel(jid)
    return no_update


@app.callback(
    Output("progress-bar", "style"), Output("status-line", "children"),
    Output("dose-metrics-body", "children"), Output("dose-analysis-body", "children"),
    Output("poll", "disabled", allow_duplicate=True),
    Output("report-store", "data"),
    Output("dose-overlay", "data"),
    Input("poll", "n_intervals"),
    State("job-store", "data"),
    prevent_initial_call=True)
def _poll(_n, jid):
    job = default_runner.get(jid) if jid else None
    if job is None:
        return no_update, no_update, no_update, no_update, True, no_update, no_update

    bar = {"background": ACCENT, "height": "100%", "transition": "width .3s",
           "width": f"{job.progress * 100:.0f}%"}
    err = f" ±{job.rel_err:.0%}" if job.rel_err else ""
    cap = job.progress_cap or job.max_batches
    if job.combined:
        kind = "SPE+GCR"          # one MC (GCR); the SPE gate is a fold, no phase
    else:
        kind = "SPE" if job.spe is not None else "GCR"
    status = (f"[{job.status.value}] {kind} · {job.spec.name} — batch "
              f"{job.batches_done}/{cap}{err} — {job.elapsed:.0f}s")

    terminal = job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)
    if not terminal:
        return bar, status, no_update, no_update, False, no_update, no_update

    if job.status != JobStatus.DONE or not (job.result and job.result.ok):
        msg = html.Div(f"Run {job.status.value}: {(job.error or '')[-300:]}",
                       style={"color": ACCENT, "fontSize": "13px"})
        return bar, status, no_update, [msg], True, no_update, no_update

    # --- combined branch: BOTH gates from one run, stacked, never summed ----
    if job.combined:
        # The SPE gate folds the event spectrum against the shielded response
        # kernel; it reads only .spec, so the GCR composition result feeds it
        # (there is no separate SPE MC in a combined job any more). It is the
        # binding Gate 2 dose and is needed whichever way Gate 1 resolves.
        a_spe = assess_spe(job.result, job.spe, skin=False)  # headline = BFO (binding)
        if a_spe is None:
            msg = html.Div("Combined run finished but the SPE gate produced no "
                           "usable dose. Try more batches.",
                           style={"color": ACCENT, "fontSize": "13px"})
            return bar, status, no_update, [msg], True, no_update, no_update

        # Gate 1: below the ~19 g/cm2 crossover the flood normalisation over-counts
        # wall-bred secondaries, so serve the phantom-matched thin-wall kernel fold
        # (ICRP-60 Q(L)) instead. It is spec-only, so it does not depend on the flood
        # lining recording signal -- check it before touching the flood assessments.
        # Gate 2 (SPE) is unaffected; the flood MC still ran and supplies statistics.
        a_tw = (assess_gcr_thinwall(job.spec, mission_days=SCORING_MISSION_DAYS,
                                    phi_MV=SCORING_TIER.phi_mv)
                if _gcr_thinwall_applies(job.spec) else None)
        if a_tw is not None:
            calibrated = _gcr_thinwall_calibrated(job.spec)
            g1_metrics = _thinwall_metric_cards(a_tw, job, calibrated)
            g1_analysis = _thinwall_analysis_body(a_tw, job, calibrated)
            g1_gcr = (None, a_tw, None)
            g1_overlay_skin, g1_phantom = a_tw.annual_msv, None
        else:
            a = _assess(job.result, skin=False)       # central phantom (noisy)
            a_skin = _assess(job.result, skin=True)   # habitat-wide lining, ICRP-60 Q
            a_skin_nasa = _assess(job.result, skin=True, qf="nasa")  # NASA Q (headline)
            primary = a_skin_nasa or a_skin or a
            if primary is None:
                msg = html.Div("Combined run finished but the GCR gate produced no "
                               "usable dose — a wall-lining scorer recorded no "
                               "signal. Try more batches.",
                               style={"color": ACCENT, "fontSize": "13px"})
                return bar, status, no_update, [msg], True, no_update, no_update
            calibrated = None
            s = primary.summary("career")
            g1_metrics = _metric_cards(a, a_skin, s, job, a_skin_nasa=a_skin_nasa)
            g1_analysis = _analysis_body(a, a_skin, s, job, a_skin_nasa=a_skin_nasa)
            g1_gcr = (a, a_skin, a_skin_nasa)
            g1_overlay_skin, g1_phantom = primary.annual_msv, a

        metrics = [_gate_header("Gate 1 · Chronic GCR field (annual)"),
                   *g1_metrics,
                   _gate_header("Gate 2 · Solar particle event (acute)"),
                   *_spe_metric_cards(a_spe, job)]
        analysis = [_gate_header("Gate 1 · Chronic GCR field (annual)"),
                    *g1_analysis,
                    _gate_header("Gate 2 · Solar particle event (acute)"),
                    *_spe_analysis_body(a_spe, job)]
        report = _report_text(job, gcr=g1_gcr, spe=a_spe, thinwall=calibrated)
        # Overlay the chronic GCR field: it is the annual number the geometry is
        # mainly judged on, and the two gates are never summed.
        overlay = {"sig": _spec_sig(job.spec), "skin": g1_overlay_skin,
                   "phantom": g1_phantom.annual_msv if g1_phantom else None,
                   "unit": "mSv/yr", "scenario": "GCR annual"}
        return (bar, status, metrics, analysis, True, report, overlay)

    # --- acute SPE branch: a single event dose vs the 30-day limit ----------
    if job.spe is not None:
        a_spe = assess_spe(job.result, job.spe, skin=False)  # headline = BFO (binding)
        if a_spe is None:
            msg = html.Div("Event run finished but produced no usable dose — the "
                           "wall-lining scorer recorded no signal. Try more batches.",
                           style={"color": ACCENT, "fontSize": "13px"})
            return bar, status, no_update, [msg], True, no_update, no_update
        report = _report_text(job, spe=a_spe)
        # SPE scores the lining only; there is no phantom event dose to show.
        # the lining position shows the SKIN dose; the BFO headline is on the card
        spe_overlay = {"sig": _spec_sig(job.spec), "skin": a_spe.skin_msv,
                       "phantom": None, "unit": "mSv/event",
                       "scenario": "worst-case SPE"}
        return (bar, status, _spe_metric_cards(a_spe, job),
                _spe_analysis_body(a_spe, job), True, report, spe_overlay)

    # --- chronic GCR branch -------------------------------------------------
    # Thin-wall (direct-transmission) regime: below the ~19 g/cm2 crossover the
    # flood normalisation over-counts wall-bred secondaries, so serve the
    # phantom-matched kernel fold instead. The flood MC still ran (its statistics
    # are shown), but the headline is the folded ICRP effective dose.
    if _gcr_thinwall_applies(job.spec):
        a_tw = assess_gcr_thinwall(job.spec, mission_days=SCORING_MISSION_DAYS,
                                   phi_MV=SCORING_TIER.phi_mv)
        if a_tw is not None:
            calibrated = _gcr_thinwall_calibrated(job.spec)
            report = _report_text(job, gcr=(None, a_tw, None), thinwall=calibrated)
            overlay = {"sig": _spec_sig(job.spec), "skin": a_tw.annual_msv,
                       "phantom": None, "unit": "mSv/yr",
                       "scenario": "GCR annual (thin-wall)"}
            return (bar, status, _thinwall_metric_cards(a_tw, job, calibrated),
                    _thinwall_analysis_body(a_tw, job, calibrated), True, report, overlay)

    a = _assess(job.result, skin=False)       # central phantom (noisy point dose)
    a_skin = _assess(job.result, skin=True)   # habitat-wide lining, ICRP-60 Q (cross-check)
    a_skin_nasa = _assess(job.result, skin=True, qf="nasa")  # NASA Q lining (headline)
    primary = a_skin_nasa or a_skin or a      # whichever assessment we actually have
    if primary is None:
        msg = html.Div("Run finished but produced no usable dose — no scorer "
                       "recorded any signal. Try more batches or a thinner design.",
                       style={"color": ACCENT, "fontSize": "13px"})
        return bar, status, no_update, [msg], True, no_update, no_update
    s = primary.summary("career")
    report = _report_text(job, gcr=(a, a_skin, a_skin_nasa))
    return (bar, status, _metric_cards(a, a_skin, s, job, a_skin_nasa=a_skin_nasa),
            _analysis_body(a, a_skin, s, job, a_skin_nasa=a_skin_nasa), True, report,
            _overlay(job.spec, primary, a, "mSv/yr", "GCR annual"))


def _overlay(spec, a_skin, a_phantom, unit, scenario):
    """Package the two scored doses for the cross-section, tagged with the design
    they were scored against."""
    return {"sig": _spec_sig(spec),
            "skin": a_skin.annual_msv if a_skin else None,
            "phantom": a_phantom.annual_msv if a_phantom else None,
            "unit": unit, "scenario": scenario}


def _gate_header(text):
    """A stacked-section banner separating the two gates in a combined run."""
    return html.Div(text, style={
        "color": ACCENT, "fontSize": "12px", "letterSpacing": "0.6px",
        "fontWeight": 800, "textTransform": "uppercase",
        "margin": "18px 0 6px", "paddingBottom": "4px",
        "borderBottom": f"1px solid {MUTED}"})


def _assess(result, skin, qf="icrp"):
    """Dispatch to the per-species composition assessment (protons + heavy ions
    summed) or the single-species path, depending on the result type. qf selects
    the quality-factor model behind the dose-equivalent: "icrp" (ICRP-60 Q(L)) or
    "nasa" (NASA/Cucinotta Q) -- both scored on the same transport, so the switch
    only re-reads a different Sv column."""
    if isinstance(result, ConvergedComposition):
        return assess_composition(result.species_results,
                                  mission_days=SCORING_MISSION_DAYS,
                                  phi_MV=SCORING_TIER.phi_mv, skin=skin, qf=qf)
    return assess(result, mission_days=SCORING_MISSION_DAYS, skin=skin, qf=qf)


def _score_card(score, rel_txt, verdict, frac, qf_label=None, cmp_line=None,
                size_note=None, limit_label="NASA career limit"):
    """The headline the student records: habitat-wide annual effective dose.

    qf_label names the quality-factor model behind `score` (both the flood headline
    and the thin-wall fold run on ICRP-60 Q(L), so the quantity does not flip across
    the ~19 g/cm2 crossover); cmp_line carries the NASA/Cucinotta Q conservative
    cross-check as a band. limit_label names the limit `frac` is a fraction of --
    plain "career limit" (the 600 mSv career limit is quality-factor-independent, so
    both the flood and the thin-wall fold share it).
    size_note, when present, is a (colour, text) advisory that the design is
    oversized and its dose extrapolates past the validated size envelope."""
    colour = VERDICT_COLOUR.get(verdict, METRIC)
    style = dict(CARD)
    style.update({"borderLeft": f"4px solid {colour}", "background": "#161d27",
                  "padding": "18px"})
    children = [
        html.Div("RADIATION PROTECTION SCORE", style={
            "color": MUTED, "fontSize": "11px", "letterSpacing": "0.8px",
            "fontWeight": 700, "marginBottom": "8px"}),
        html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "8px"},
                 children=[
            html.Span(f"{score:.1f}", style={"color": colour, "fontSize": "42px",
                                             "fontWeight": 900, "lineHeight": "1"}),
            html.Span(f"mSv/yr{rel_txt}", style={"color": INK, "fontSize": "14px"})]),
        html.Div(f"{verdict} · {frac:.0f}% of {limit_label}"
                 + (f" · {qf_label}" if qf_label else ""), style={
            "color": colour, "fontSize": "12px", "fontWeight": 700, "marginTop": "10px"}),
    ]
    if cmp_line:
        children.append(html.Div(cmp_line, style={
            "color": MUTED, "fontSize": "11px", "marginTop": "6px"}))
    if size_note:
        note_colour, note_text = size_note
        children.append(html.Div("⚠ " + note_text, style={
            "color": note_colour, "fontSize": "11px", "fontWeight": 600,
            "marginTop": "8px", "lineHeight": "1.35"}))
    return html.Div(style=style, children=children)


def _size_note(spec):
    """(colour, text) advisory when a design is too big for the validated gauge
    envelope, else None. Keys off geometry only (gauge_corr) -- target-blind."""
    flag = gauge_size_flag(spec)
    if flag == "ok":
        return None
    gc = _gauge_corr(spec)
    pct = f"(gauge {gc:.2f}× — normalisation grew to enclose the design)"
    if flag == "mild":
        return ("#d29922",
                "Design exceeds the validated size envelope; absolute dose is a "
                f"mild extrapolation {pct}.")
    return (ACCENT,
            "Design is well beyond the validated size envelope; treat the absolute "
            f"dose as indicative and anchor it against a ≤6 m version {pct}.")


def _footprint_banner(spec):
    """Prominent full-width banner (NOT a hard block) shown when a design pokes past
    the 900 cm illuminated GCR footprint, so part of the wall receives no incident
    primaries and the crew dose is physically under-sampled. Returns an html.Div or
    None. Keys off geometry only (enclosing radius vs beam spot) -- target-blind.

    Distinct from _size_note: that flags NORMALISATION extrapolation (the 1/R^2 gauge
    correction still restores the anchor); this flags MISSING FLUX, which no
    correction can recover -- so it is the more serious, louder advisory."""
    flag = beam_footprint_flag(spec)
    if flag == "ok":
        return None
    enc = _enclosing_radius_cm(spec)
    spot_m = DEFAULT_BEAM_SPOT_CM / 100.0
    if flag == "marginal":
        colour, bg = "#d29922", "rgba(210,153,34,0.12)"
        head = "Design brushes the edge of the simulated field"
        body = (f"The habitat’s farthest corner ({enc/100:.1f} m) sits within "
                f"{DEFAULT_BEAM_SPOT_CM - enc:.0f} cm of the {spot_m:.0f} m illuminated "
                "GCR footprint. Grazing directions start to clip, so the dose reads "
                "slightly low — trust it as a lower bound, not an exact figure.")
    else:  # over
        colour, bg = ACCENT, "rgba(242,79,61,0.14)"
        head = "Design exceeds the simulated radiation field — dose under-sampled"
        body = (f"The habitat’s farthest corner ({enc/100:.1f} m) extends past the "
                f"{spot_m:.0f} m illuminated GCR footprint, so part of the outer wall "
                "receives no incident particles from low angles. The dose below is "
                "biased LOW and should be read only as a lower bound. For a trustworthy "
                "number, shrink the habitat below the footprint (or widen the source "
                "field and re-anchor). The run still completed — it was not blocked.")
    return html.Div(style={
        "background": bg, "border": f"1px solid {colour}",
        "borderLeft": f"5px solid {colour}", "borderRadius": "10px",
        "padding": "14px 16px", "marginBottom": "2px"}, children=[
        html.Div("⚠ " + head, style={
            "color": colour, "fontSize": "13px", "fontWeight": 800,
            "letterSpacing": "0.3px", "marginBottom": "5px"}),
        html.Div(body, style={
            "color": INK, "fontSize": "12px", "lineHeight": "1.45"}),
    ])


# Below this many batches the phantom's error bar is NOT REPORTED at all.
# SCORING_CONVERGE_ON = "both" should now keep the phantom above this floor on
# its own (10% takes ~4 rounds), so this is a backstop rather than the usual
# path -- it still fires when a run is CANCELLED early or caps out at
# SCORING_MAX_BATCHES on a design too noisy to reach target. It matters because
# an n=2 dose_rel_err is a standard error built from a TWO-SAMPLE standard
# deviation. Measured on the SHARC 6 m dome (147.5 g/cm2), same design,
# 2 rounds vs 5 -- i.e. what the old skin-only convergence used to report:
#     2 rounds:  phantom 511.4 mSv/yr, quoted +-3%     skin 292.1 +-1%
#     5 rounds:  phantom 418.2 mSv/yr, quoted +-7%     skin 285.8 +-0.7%
# The phantom moved 18% -- six times its own quoted bar -- while the skin moved
# 2%. An n=2 stdev has ~76% relative uncertainty and is skewed LOW, so two
# batches landing near each other mint a confident-looking bar around a wrong
# number. Widening via Student-t does not rescue it (n=2 gives only ~1.8x, 3% ->
# 5.5%, still far short of 18%): two samples cannot error-bar a quantity this
# heavy-tailed, so we state the batch count instead of inventing a precision.
# The point dose itself is still shown -- a useful diagnostic, just not one
# measured to 3%.
#
# Bound to SCORING_MIN_BATCHES on purpose: the display gate and the convergence
# floor answer the same question ("are there enough samples to believe this?"),
# so they must not drift apart. Raising the floor must never leave the card
# hiding a bar the run actually earned, and lowering it must never let the card
# show one it did not.
PHANTOM_MIN_BATCHES = SCORING_MIN_BATCHES


def _phantom_err_text(result) -> str:
    """' ±N%' only when enough batches back it; otherwise a batch count."""
    rel = getattr(result, "dose_rel_err", None)
    n = getattr(result, "phantom_batches", 0)
    if rel and n >= PHANTOM_MIN_BATCHES:
        return f" ±{rel:.0%}"
    if n:
        return f" ({n} batch{'es' if n != 1 else ''} — not converged, no error bar)"
    return " (not converged)"


def _metric_cards(a, a_skin, s, job, a_skin_nasa=None):
    # Absorbed-dose source: use the habitat-wide skin lining so this card and the
    # Dose-equivalent card below describe the SAME location -- their ratio is then
    # the real emergent skin Q. (Sourcing absorbed from the high-variance central
    # phantom while dose-eq comes from the skin invites a nonsense apparent-Q.)
    # Absorbed dose is quality-factor-independent, so both Q models share it.
    # Fall back to the phantom only if the lining scorer is unavailable.
    ab = a_skin if a_skin is not None else a
    # Headline dose-equivalent runs on the ICRP-60 Q(L) habitat-wide skin scorer,
    # the SAME quantity+convention as the below-gate thin-wall kernel, so the
    # displayed number no longer flips quality-factor model across the ~19 g/cm2
    # crossover. The ICRP skin-lining headline is OLTARIS-validated in the thick
    # regime (~1.1x at 50 g/cm2, where final designs sit) and stays the
    # conservative, safe-erring choice through the messy middle band. The
    # NASA/Cucinotta Q twin is kept as a labelled conservative cross-check (it
    # over-reads ~1.5x at thick); older runs with no NASA CSV simply omit it.
    head = a_skin if a_skin is not None else a
    cmp_nasa = a_skin_nasa if (a_skin is not None and a_skin_nasa is not None) else None
    aeq = head if head is not None else a
    ratio = aeq.equiv_rate_msv_day / 0.70 if aeq.equiv_rate_msv_day else 0.0
    # The graded score uses the habitat-wide (wall-lining) scorer: it samples the
    # whole inner surface, so it is far lower-variance than the central crew
    # phantom and gives a number two teams can be compared on. Fall back to the
    # phantom only if the lining scorer is unavailable.
    if head is not None:
        score = head.annual_msv
        s_head = head.summary("career")
        verdict = s_head["verdict"]
        frac = s_head["fraction_of_limit"] * 100
        skin_rel = getattr(job.result, "skin_dose_rel_err", None)
        rel_txt = f" ± {skin_rel:.0%}" if skin_rel else ""
        qf_label = "ICRP-60 Q(L)"
        cmp_line = (f"NASA/Cucinotta Q conservative cross-check: "
                    f"{cmp_nasa.annual_msv:.1f} mSv/yr"
                    if cmp_nasa is not None else None)
    else:
        score, verdict = ab.annual_msv, s["verdict"]
        frac, rel_txt, qf_label, cmp_line = s["fraction_of_limit"] * 100, "", None, None
    eq_sub = (f"ISS baseline = 0.70 mSv/day | ratio: {ratio:.2f}×"
              + (f" | NASA Q: {cmp_nasa.equiv_rate_msv_day:.3f}" if cmp_nasa is not None else ""))
    phantom_pt = (f"{a.annual_msv:.1f} mSv/year" if a is not None else "n/a")
    phantom_rel = _phantom_err_text(job.result)
    phantom_note = ("central self-shielded point — noisier diagnostic, not the score"
                    if not phantom_rel else
                    f"central self-shielded point{phantom_rel} — noisier diagnostic, "
                    "not the score")
    size_note = _size_note(job.result.spec) if job.result is not None else None
    banner = _footprint_banner(job.result.spec) if job.result is not None else None
    return ([banner] if banner else []) + [
        _score_card(score, rel_txt, verdict, frac, qf_label=qf_label,
                    cmp_line=cmp_line, size_note=size_note,
                    limit_label="career limit"),
        metric_card("Absorbed dose",
                    f"{ab.dose_rate_ugy_day / 1000:.3f} mGy/day",
                    f"= {ab.annual_mgy:.1f} mGy/year"),
        metric_card("Dose equivalent",
                    f"{aeq.equiv_rate_msv_day:.3f} mSv/day", eq_sub),
        metric_card("Crew-phantom point dose", phantom_pt, phantom_note),
    ]


def _analysis_body(a, a_skin, s, job, a_skin_nasa=None):
    skin_err = getattr(job.result, "skin_dose_rel_err", None)
    # Headline dose-equivalent runs on the ICRP-60 Q(L) skin scorer -- the same
    # quantity+convention as the below-gate kernel, so the displayed number no
    # longer flips Q-model at the ~19 g/cm2 crossover. NASA/Cucinotta Q is the
    # labelled conservative cross-check. Absorbed-dose rows are Q-independent.
    head = a_skin if a_skin is not None else a
    cmp_nasa = a_skin_nasa if (a_skin is not None and a_skin_nasa is not None) else None
    aeq = head if head is not None else a
    ab = a if a is not None else a_skin       # absorbed-dose / flux fallback source
    seq = aeq.summary("career")
    head_qf = "ICRP-60 Q(L)"
    head_txt = (f"{head.annual_msv:.1f} mSv/year"
                f"{f' ± {skin_err:.0%}' if skin_err else ''}"
                if head is not None else "n/a")
    phantom_pt = f"{a.annual_msv:.1f} mSv/year" if a is not None else "n/a"
    # mean quality factor: emergent H/D for each model present
    if cmp_nasa is not None:
        qf_txt = (f"{seq['quality_factor']:.2f} (ICRP-60) · "
                  f"{cmp_nasa.summary('career')['quality_factor']:.2f} (NASA)")
    else:
        qf_txt = f"{seq['quality_factor']:.2f}"
    rows = [
        _kv(f"◆ PROTECTION SCORE (habitat-wide, {head_qf})", head_txt)]
    if cmp_nasa is not None:
        rows.append(_kv("   NASA/Cucinotta Q cross-check (conservative)",
                        f"{cmp_nasa.annual_msv:.1f} mSv/year"))
    rows += [
        _kv("Crew-phantom dose (point, diagnostic)", phantom_pt),
        _kv("Absorbed dose rate", f"{s['dose_rate_uGy_per_day']:.3f} µGy/day"),
        _kv("Equivalent dose rate", f"{aeq.equiv_rate_msv_day * 1000:.2f} µSv/day"),
        _kv("Effective dose (mission)", f"{seq['mission_mSv']:.1f} mSv "
            f"over {int(seq['mission_days'])} days"),
        _kv("Fraction of career limit", f"{seq['fraction_of_limit'] * 100:.1f} %"),
        _kv("Quality factor (mean, LET-weighted)", qf_txt),
        _neutron_fraction_row(a_skin if a_skin is not None else head),
        _kv("Wall transmission", f"{job.result.transmission:.2f}"
            if job.result.transmission else "n/a"),
        _kv("GCR flux used", f"{ab.real_flux_cm2_s:.2f} /cm²/s"),
        _kv("Statistics", f"{job.result.n_batches} batches, "
            f"{job.result.total_primaries:,} primaries, {job.result.wall_seconds:.0f}s"),
        *_species_breakdown(a_skin),
        html.Div("Above ~19 g/cm² the headline is the geometry-aware flood "
                 "wall-lining dose; below it, the thin-wall kernel fold. The two "
                 "engines hand off at the ~19 g/cm² gate, the least-certain point of "
                 "the sweep: the flood errs conservative (safe) through the ~19–30 "
                 "g/cm² band and lands within ~10% of cross-code (OLTARIS) effective "
                 "dose at thick shielding, where final designs sit; the kernel just "
                 "below the gate is the more optimistic estimate. The true value "
                 "sits between them near the gate.",
                 style={"color": MUTED, "fontSize": "10px",
                        "marginTop": "12px", "fontStyle": "italic"}),
        html.Div("Dose is summed over the GCR ion composition (H, He, C, Si, Fe "
                 "groups), each transported separately and normalised to its real "
                 "flux. Equivalent dose is LET-weighted per step by a custom scorer "
                 "(the headline uses ICRP-60 Q(L); NASA/Cucinotta Q is shown "
                 "alongside as a conservative cross-check), so high-LET ions and "
                 "secondaries carry their own quality factor and the mean Q shown "
                 "above is the emergent H/D ratio.",
                 style={"color": MUTED, "fontSize": "10px",
                        "marginTop": "12px", "fontStyle": "italic"}),
    ]
    return rows


def _neutron_fraction_row(a):
    """Row: share of the crew dose-equivalent carried by wall-bred secondary
    (albedo) neutrons. Emitted only when the neutron-lineage scorer reported a
    fraction; renders nothing (empty Div) otherwise so older runs are unaffected."""
    f = getattr(a, "neutron_fraction", None) if a is not None else None
    if f is None:
        return html.Div()
    return _kv("↳ from secondary neutrons (albedo)", f"{f * 100:.0f}% of dose-equivalent")


def _species_breakdown(a_skin):
    """A small per-ion table of each GCR species' share of the absorbed dose --
    the visible proof that heavy ions (not just protons) drive the score."""
    contrib = getattr(a_skin, "contributions", None) if a_skin else None
    if not contrib:
        return []
    rows = []
    for c in contrib:
        q = c.get("quality_factor")
        qtxt = f", Q={q:.1f}" if q else ""
        rows.append(_kv(f"  {c['species']} ({c['group']})",
                        f"{c['dose_fraction'] * 100:.0f}% of dose{qtxt}"))
    return [html.Div("Dose share by GCR ion", style={
        "color": MUTED, "fontSize": "11px", "fontWeight": 700,
        "marginTop": "12px", "marginBottom": "4px"}), *rows]


# ----------------------------------------------------------------------
# Thin-wall (direct-transmission) GCR rendering
# ----------------------------------------------------------------------
# When the wall is too thin to build a full secondary shower, crew dose is
# direct primary transmission through the column above them, NOT the broad
# wall-bred field the flood normalisation assumes -- so the flood path grossly
# over-counts (a 7.5 mm Al dome reads ~3500 mSv/yr under flood, a spatial
# blackout artefact). Below the ~19 g/cm2 crossover we serve instead the
# phantom-matched kernel fold (dosimetry.fold_gcr_thinwall): a per-species,
# per-organ R(E) transported through aluminium and folded against the true
# free-field GCR flux -- no gauge, no 1/R^2, target-blind. The headline is the
# ICRP-60 whole-body effective dose; there is no NASA twin or neutron-lineage
# scorer in this kernel, and the phantom IS the score (no separate point dose).
def _thinwall_note(calibrated):
    """(colour, text) banner explaining the thin-wall regime; amber + 'indicative'
    when the wall is not aluminium-dominated (folded on the Al transport kernel)."""
    if calibrated:
        return ("#3fb950",
                "Thin-wall regime: dose is the phantom-matched direct-transmission "
                "fold (aluminium kernel) folded against the free-field GCR flux — not "
                "the flood normalisation, which over-counts wall-bred secondaries when "
                "the wall is too thin to build a full shower.")
    return ("#d29922",
            "Thin-wall regime, non-aluminium wall: folded on the aluminium transport "
            "kernel, so the absolute dose is indicative — the regime is right, the "
            "material scaling approximate.")


def _thinwall_metric_cards(a, job, calibrated):
    s = a.summary("career")
    rel_txt = f" ± {a.rel_err:.0%}" if a.rel_err else ""
    banner = _footprint_banner(job.spec)
    return ([banner] if banner else []) + [
        _score_card(a.annual_msv, rel_txt, s["verdict"], s["fraction_of_limit"] * 100,
                    qf_label="ICRP-60 Q(L) · thin-wall phantom-matched",
                    cmp_line=None,
                    # The regime is explained in the Analysis footnote, so the card
                    # stays clean; only the non-Al "indicative" caveat rides here.
                    size_note=(None if calibrated else _thinwall_note(calibrated)),
                    limit_label="career limit"),
        metric_card("Absorbed dose", f"{a.dose_rate_ugy_day / 1000:.3f} mGy/day",
                    f"= {a.annual_mgy:.1f} mGy/year"),
        metric_card("Dose equivalent", f"{a.equiv_rate_msv_day:.3f} mSv/day",
                    f"ISS baseline = 0.70 mSv/day | ratio: "
                    f"{a.equiv_rate_msv_day / 0.70:.2f}×"),
        metric_card("Method", "kernel fold (phantom-matched)",
                    "direct-transmission thin-wall regime — no flood normalisation"),
    ]


def _thinwall_analysis_body(a, job, calibrated):
    s = a.summary("career")
    calib_note = ("" if calibrated else
                  "  ⚠ non-aluminium wall — folded on the Al kernel, indicative only")
    return [
        _kv("◆ PROTECTION SCORE (whole-body effective, ICRP-60 Q(L))",
            f"{a.annual_msv:.1f} mSv/year{f' ± {a.rel_err:.0%}' if a.rel_err else ''}"),
        _kv("Verdict (career limit)", s["verdict"]),
        _kv("Absorbed dose rate", f"{a.dose_rate_ugy_day:.3f} µGy/day"),
        _kv("Equivalent dose rate", f"{a.equiv_rate_msv_day * 1000:.2f} µSv/day"),
        _kv("Effective dose (mission)", f"{s['mission_mSv']:.1f} mSv "
            f"over {int(s['mission_days'])} days"),
        _kv("Fraction of career limit", f"{s['fraction_of_limit'] * 100:.1f} %"),
        _kv("Quality factor (mean, LET-weighted)", f"{s['quality_factor']:.2f}"),
        _kv("Wall areal density", f"{job.spec.areal_density_gcm2():.1f} g/cm²"),
        _kv("GCR flux used (free-field)", f"{a.real_flux_cm2_s:.2f} /cm²/s"),
        _kv("Method", "phantom-matched kernel fold" + calib_note),
        *_species_breakdown(a),
        html.Div("Below ~19 g/cm² the wall is too thin to build a full secondary "
                 "shower, so crew dose is direct primary transmission — the flood "
                 "normalisation (right behind thick shielding) over-counts here. Dose "
                 "is instead folded from each GCR ion's spectrum against a precomputed "
                 "aluminium response kernel R(E) (per-organ shells, ICRP-60 Q baked "
                 "in) and normalised to the true free-field flux — no flood gauge, no "
                 "distance correction. Whole-body effective dose is the wₜ-weighted "
                 "organ sum.",
                 style={"color": MUTED, "fontSize": "10px",
                        "marginTop": "12px", "fontStyle": "italic"}),
    ]


# ----------------------------------------------------------------------
# SPE result rendering (acute single-event dose vs the 30-day limit)
# ----------------------------------------------------------------------
def _spe_score_card(a):
    """The acute headline: total event dose-equivalent vs the 30-day BFO limit."""
    verdict = a.verdict("nasa_30day")
    colour = VERDICT_COLOUR.get(verdict, METRIC)
    frac = a.fraction_of("nasa_30day") * 100
    rel = f" ± {a.rel_err:.0%}" if a.rel_err else ""
    style = dict(CARD)
    style.update({"borderLeft": f"4px solid {colour}", "background": "#161d27",
                  "padding": "18px"})
    return html.Div(style=style, children=[
        html.Div("SOLAR EVENT DOSE (ACUTE)", style={
            "color": MUTED, "fontSize": "11px", "letterSpacing": "0.8px",
            "fontWeight": 700, "marginBottom": "8px"}),
        html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "8px"},
                 children=[
            html.Span(f"{a.event_msv:.0f}", style={"color": colour, "fontSize": "42px",
                                                   "fontWeight": 900, "lineHeight": "1"}),
            html.Span(f"mSv / event{rel}", style={"color": INK, "fontSize": "14px"})]),
        html.Div(f"{verdict} · {frac:.0f}% of the {DOSE_LIMITS_MSV['nasa_30day']:.0f} "
                 "mSv 30-day BFO limit", style={
            "color": colour, "fontSize": "12px", "fontWeight": 700, "marginTop": "10px"}),
        html.Div(f"skin {a.skin_msv:.0f} mSv · {a.skin_msv / 1500 * 100:.0f}% of the "
                 "1500 mSv 30-day skin limit"
                 if a.skin_msv is not None else "",
                 style={"color": MUTED, "fontSize": "11px", "marginTop": "4px"}),
    ])


def _spe_metric_cards(a, job):
    return [
        _spe_score_card(a),
        metric_card("Absorbed dose, BFO (event)", f"{a.event_mgy:.2f} mGy",
                    "deep (~5 cm) organ total, folded behind the shield"),
        metric_card("Quality factor, BFO", f"{a.quality_factor:.2f}",
                    "NASA/Cucinotta Q for the penetrating proton field"),
        metric_card("30-day BFO limit",
                    f"{DOSE_LIMITS_MSV['nasa_30day']:.0f} mSv",
                    f"event delivers {a.fraction_of('nasa_30day') * 100:.1f}% of it"),
    ]


def _spe_analysis_body(a, job):
    # The acute dose comes entirely from the response-kernel fold (dosimetry.
    # fold_spe); there is no SPE MC run to read statistics from.
    s = a.summary("nasa_30day")
    verdict = s["verdict"]
    skin_line = (f"{a.skin_msv:.0f} mSv · {a.skin_msv / 1500 * 100:.1f}% of 1500 limit"
                 if a.skin_msv is not None else "n/a")
    calib_note = ("" if a.calibrated else
                  "  ⚠ design is off the kernel's shielded calibration; treat as indicative")
    return [
        _kv("◆ BFO EVENT DOSE-EQUIVALENT", f"{a.event_msv:.0f} mSv"
            f"{f' ± {a.rel_err:.0%}' if a.rel_err else ''}"),
        _kv("Verdict (30-day BFO limit)", verdict),
        _kv("Skin dose-equivalent (event)", skin_line),
        _kv("Absorbed dose, BFO (event)", f"{a.event_mgy:.1f} mGy"),
        _kv("Quality factor, BFO (NASA/Cucinotta)", f"{a.quality_factor:.2f}"),
        _kv("Fraction of 30-day BFO limit", f"{s['fraction_of_limit'] * 100:.1f} %"),
        _kv("Event proton fluence (>30 MeV)", f"{a.event_fluence_cm2:.1e} /cm²"),
        _kv("Penetrating fluence (reaches crew)", f"{a.sim_fluence_cm2:.1e} /cm²"),
        _kv("Scenario", a.scenario_name + calib_note),
        _kv("Method", "response-kernel fold (variance-reduced)"),
        html.Div("A single acute solar particle event delivers its whole proton "
                 "fluence over hours, so the dose is a one-off total judged against the "
                 "30-day blood-forming-organ limit, not an annual rate. Behind thick "
                 "regolith only a few percent of even the hardest event penetrates, so a "
                 "direct phantom MC is rare-tail-starved; the dose is instead folded from "
                 "the event's proton spectrum against a precomputed shielded response "
                 "kernel R(E) (per-depth, per-organ NASA/ICRP quality factors baked in). "
                 "BFO is the deep (~5 cm) shell; skin is the surface shell.",
                 style={"color": MUTED, "fontSize": "10px",
                        "marginTop": "12px", "fontStyle": "italic"}),
    ]


# ----------------------------------------------------------------------
# Design save / load  +  results report  (#2)
# ----------------------------------------------------------------------
def _report_text(job, gcr=None, spe=None, thinwall=None) -> str:
    """One-page markdown report of the completed run: design, verdict, breakdown.
    Built here (in the poll) while the assessments are in hand and stashed in the
    report-store, so the download button just serves the string.

    A combined run passes BOTH gcr and spe; the two hazards are reported as
    separate gates (never summed). The SPE gate is a fold (no MC of its own);
    job.result holds the GCR composition and supplies all reported statistics."""
    spec = job.spec
    stack = " + ".join(f"{w.thickness_cm * 10:g} mm {w.material}" for w in spec.walls)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# Lunar habitat radiation report — {spec.name}",
         f"_Generated {stamp}_", "",
         "## Design",
         f"- Shape: {SHAPE_LABELS.get(spec.shape, spec.shape)}",
         f"- Inner radius: {spec.inner_radius_cm / 100:.2f} m",
         f"- Wall stack (innermost first): {stack}",
         f"- Areal density: {spec.areal_density_gcm2():.1f} g/cm²",
         f"- Shell mass: {spec.shell_mass_kg() / 1000:.1f} t", ""]

    if gcr is not None and spe is not None:
        L += ["> Two independent gates — the acute solar event and the chronic GCR "
              "field have different dose limits and clocks, so they are judged "
              "separately and never summed.", ""]

    combined = gcr is not None and spe is not None

    if gcr is not None:
        a, a_skin, *rest = gcr
        a_skin_nasa = rest[0] if rest else None
        # Headline runs on the ICRP-60 Q(L) skin scorer (same convention as the
        # below-gate kernel); NASA/Cucinotta Q is the conservative cross-check.
        head = a_skin if a_skin is not None else a
        cmp_nasa = a_skin_nasa if (a_skin is not None and a_skin_nasa is not None) else None
        head_qf = "ICRP-60 Q(L)"
        s = head.summary("career")
        L += ["## Gate 1 — chronic GCR field (annual)" if combined
              else "## Scenario — chronic GCR field (annual)",
              f"- Solar modulation: φ=400 MV (solar minimum)",
              f"- Mission: {SCORING_MISSION_DAYS} days",
              f"- **Protection score (habitat-wide, {head_qf}): {head.annual_msv:.1f} mSv/yr**"]
        if cmp_nasa is not None:
            L.append(f"- NASA/Cucinotta Q conservative cross-check: "
                     f"{cmp_nasa.annual_msv:.1f} mSv/yr")
        if a is not None:
            L.append(f"- Crew-phantom point dose: {a.annual_msv:.1f} mSv/yr")
        qf_line = f"- Mean quality factor: {s['quality_factor']:.2f} ({head_qf})"
        if cmp_nasa is not None:
            qf_line += f" · {cmp_nasa.summary('career')['quality_factor']:.2f} (NASA)"
        L += [f"- Absorbed dose: {head.annual_mgy:.1f} mGy/yr",
              qf_line,
              f"- Career limit: {s['limit_mSv']:.0f} mSv "
              f"({s['fraction_of_limit'] * 100:.0f}% used)",
              f"- **Verdict: {s['verdict']}**"]
        if thinwall is not None:
            L += ["- Regime: thin-wall phantom-matched fold (wall < ~19 g/cm², below "
                  "the flood-normalisation crossover — direct primary transmission, "
                  "not the wall-bred secondary field)"
                  + ("" if thinwall else
                     "  ⚠ non-aluminium wall — folded on the Al kernel, indicative only")]
        elif a_skin is not None:
            L += ["- Regime: geometry-aware flood wall-lining (wall ≥ ~19 g/cm², above "
                  "the thin-wall crossover). Conservative/safe-erring through the "
                  "~19–30 g/cm² band; validated within ~10% of cross-code (OLTARIS) "
                  "effective dose at thick shielding. The kernel→flood hand-off at "
                  "the gate is the least-certain point of the sweep."]
        contrib = getattr(a_skin, "contributions", None) if a_skin else None
        if contrib:
            L += ["", "### Dose share by GCR ion"]
            for c in contrib:
                q = c.get("quality_factor")
                qtxt = f", Q={q:.1f}" if q else ""
                L.append(f"- {c['species']} ({c['group']}): "
                         f"{c['dose_fraction'] * 100:.0f}% of dose{qtxt}")
        L += [f"- Statistics: {job.result.n_batches} batches, "
              f"{job.result.total_primaries:,} primaries, {job.result.wall_seconds:.0f}s", ""]

    if spe is not None:
        s = spe.summary("nasa_30day")
        skin_txt = (f"{spe.skin_msv:.0f} mSv ({spe.skin_msv / 1500 * 100:.1f}% of 1500)"
                    if spe.skin_msv is not None else "n/a")
        L += ["## Gate 2 — worst-case solar particle event (acute)" if combined
              else "## Scenario — worst-case solar particle event (acute)",
              f"- Event: {spe.scenario_name}",
              f"- Event proton fluence (>30 MeV): {spe.event_fluence_cm2:.1e} /cm²",
              f"- Penetrating fluence (reaches crew): {spe.sim_fluence_cm2:.1e} /cm²",
              f"- **BFO event dose-equivalent: {spe.event_msv:.0f} mSv**",
              f"- Skin event dose-equivalent: {skin_txt}",
              f"- Absorbed dose (BFO): {spe.event_mgy:.2f} mGy",
              f"- Mean quality factor (BFO, NASA/Cucinotta): {s['quality_factor']:.2f}",
              f"- 30-day BFO limit: {s['limit_mSv']:.0f} mSv "
              f"({s['fraction_of_limit'] * 100:.1f}% used)",
              f"- **Verdict: {s['verdict']}**",
              f"- Method: response-kernel fold (variance-reduced)"
              + ("" if spe.calibrated else
                 "  ⚠ off shielded calibration — indicative only")]

    return "\n".join(L)


@app.callback(
    Output("download-design", "data"),
    Input("save-design", "n_clicks"),
    State("shape", "value"), State("inner-r", "value"),
    State({"type": "layer-mat", "index": ALL}, "value"),
    State({"type": "layer-mat", "index": ALL}, "id"),
    State({"type": "layer-thk", "index": ALL}, "value"),
    State("active-rows", "data"),
    State("length-slider", "value"),
    prevent_initial_call=True)
def _save_design(n, shape, inner_r, mats, ids, thks, active, length):
    spec = spec_from_inputs("habitat", shape, inner_r,
                            _layers_from_components(mats, ids, thks, active), length)
    return dict(content=spec.to_json(), filename=f"{spec.name}.json")


@app.callback(
    Output("download-report", "data"),
    Output("design-file-note", "children", allow_duplicate=True),
    Input("save-report", "n_clicks"),
    State("report-store", "data"),
    prevent_initial_call=True)
def _save_report(n, report):
    if not report:
        return no_update, "Evaluate a design first — no results to report yet."
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    return dict(content=report, filename=f"radiation-report-{stamp}.md"), ""


# Import is a ONE-SHOT bulk set triggered by a file upload (never by a keystroke),
# so writing the layer widgets' values here is safe under the controlled-input
# rule that forbids fighting live typing. We fill the fixed pool innermost-first
# and set active-rows to match; unused slots keep a placeholder value but stay
# hidden. shape/inner-r are plain single outputs.
@app.callback(
    Output("shape", "value"), Output("inner-r", "value"),
    Output("length-slider", "value"),
    Output("active-rows", "data", allow_duplicate=True),
    Output({"type": "layer-mat", "index": ALL}, "value"),
    Output({"type": "layer-thk", "index": ALL}, "value"),
    Output("design-file-note", "children"),
    Input("load-design", "contents"),
    prevent_initial_call=True)
def _load_design(contents):
    if not contents:
        return (no_update,) * 7
    try:
        _, b64 = contents.split(",", 1)
        text = base64.b64decode(b64).decode("utf-8")
        spec = HabitatSpec.from_json(text)
    except Exception as exc:
        return (no_update, no_update, no_update, no_update, no_update, no_update,
                html.Span(f"Load failed: {str(exc)[-120:]}", style={"color": ACCENT}))

    walls = spec.walls[:MAX_LAYERS]
    mats = [(walls[i].material if i < len(walls) else "aluminium")
            for i in range(MAX_LAYERS)]
    thks = [(f"{walls[i].thickness_cm * 10:g}" if i < len(walls) else "0")
            for i in range(MAX_LAYERS)]
    active = list(range(len(walls)))
    # Reflect the loaded axial length on the slider (clamped to its range); the
    # field itself stays hidden for a dome. effective_height_cm resolves the
    # radius-driven default when the saved design left height_cm unset.
    length = max(2.0, min(12.0, spec.effective_height_cm / 100.0))
    note = html.Span(f"Loaded '{spec.name}'.", style={"color": "#3fb950"})
    return (spec.shape, spec.inner_radius_cm / 100.0, length,
            active, mats, thks, note)


def main_entry():
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main_entry()
