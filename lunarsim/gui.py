"""Radiation Simulation Tool -- Dash GUI for the lunar-habitat workshop.

Layout matches the agreed mockup:

  * Left rail  : branded "Radiation Sim" -> Habitat Geometry -> Primary Wall
                 (+ collapsible extra shielding layer) -> exposure -> Run.
  * Header     : "Radiation Simulation Tool" + a live config subtitle line.
  * Tabs       : Spacecraft Overview | GCR Environment | Dose Analysis |
                 Design Comparison.
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

from .spec import HabitatSpec, WallLayer, MATERIALS, SHAPES
from .bridge import FULL_RUN
from .dosimetry import assess, gcr_scalar_fluence_rate, DOSE_LIMITS_MSV
from .jobs import default_runner, JobStatus
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
# Converge on the SKIN (wall-lining) dose -- the official score -- not the noisy
# central phantom, and to a tight 5% so genuinely different designs separate
# beyond the statistical band instead of overlapping inside it.
SCORING_TARGET_REL_ERR = 0.05
SCORING_MAX_BATCHES = 12
SCORING_CONVERGE_ON = "skin"

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


def _layers_from_components(mats, ids, thks):
    """Collect the live layer-row widgets into an innermost-first list of
    {m, t(mm)} dicts. Pattern-matching ALL returns each property in its own
    list; we sort by the row index carried in the component id so the order
    matches the visual stack regardless of how Dash batches them."""
    rows = sorted(zip(ids, mats, thks), key=lambda r: r[0]["index"])
    return [{"m": m or "aluminium", "t": float(t or 0)} for _id, m, t in rows]


def spec_from_inputs(name, shape, inner_r_m, layers) -> HabitatSpec:
    walls = [WallLayer(L.get("m") or "aluminium", float(L.get("t") or 0) / 10.0)
             for L in (layers or []) if float(L.get("t") or 0) > 0]
    if not walls:                                   # never build a wall-less habitat
        walls = [WallLayer("aluminium", 0.6)]
    return HabitatSpec(
        name=(name or "habitat").strip().replace(" ", "_") or "habitat",
        shape=shape or "dome",
        inner_radius_cm=float(inner_r_m) * 100.0,
        walls=walls,
    )


def _layer_row(i, material, thickness_mm, n_total):
    """One editable wall layer (material + thickness mm + remove button).
    index `i` is the position in the stack, innermost = 0."""
    if n_total == 1:
        where = "wall"
    elif i == 0:
        where = "innermost"
    elif i == n_total - 1:
        where = "outermost"
    else:
        where = f"layer {i + 1}"
    head = html.Div(style={"display": "flex", "justifyContent": "space-between",
                           "alignItems": "center", "marginBottom": "6px"}, children=[
        html.Span(f"Layer {i + 1} · {where}",
                  style={"color": INK, "fontSize": "12px", "fontWeight": 700}),
        html.Button("✕", id={"type": "layer-remove", "index": i},
                    disabled=(n_total == 1), n_clicks=0,
                    style={"background": "transparent",
                           "color": MUTED if n_total == 1 else ACCENT,
                           "border": "none", "cursor": "pointer", "fontSize": "13px",
                           "padding": "0 4px"}),
    ])
    return html.Div(style={"background": "#0b0f15", "border": f"1px solid {BORDER}",
                           "borderRadius": "8px", "padding": "10px 12px",
                           "marginBottom": "10px"}, children=[
        head,
        dcc.Dropdown(id={"type": "layer-mat", "index": i}, value=material,
                     options=MATERIAL_OPTIONS, clearable=False,
                     style={"marginBottom": "8px"}),
        html.Div("Thickness (mm)", style=FIELD_LABEL),
        dcc.Input(id={"type": "layer-thk", "index": i}, type="number",
                  value=thickness_mm, min=0.1, step=1, style=INPUT),
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


def _rect_trace(x0, x1, y0, y1, colour, name, legend=True):
    return go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
                      fill="toself", mode="lines",
                      line=dict(color=BG, width=0.5), fillcolor=colour,
                      name=name, hoverinfo="name", showlegend=legend)


def _cross_arch(spec: HabitatSpec) -> go.Figure:
    """Radial slice: a half-arch (dome and quonset share this cross-section)."""
    fig = go.Figure()
    outer = spec.outer_radius_cm
    span = outer * 1.25
    fig.add_shape(type="rect", x0=-span, x1=span, y0=-span * 0.35, y1=0,
                  fillcolor=MATERIALS["regolith"]["colour"], line_width=0, layer="below")
    fig.add_shape(type="circle", x0=-spec.inner_radius_cm, x1=spec.inner_radius_cm,
                  y0=-spec.inner_radius_cm, y1=spec.inner_radius_cm,
                  fillcolor="#0b1d2e", line_width=0)
    for i, ((ri, ro), w) in enumerate(zip(spec.layer_radii_cm(), spec.walls)):
        fig.add_trace(_semi_annulus(ri, ro, MATERIALS[w.material]["colour"],
                                    f"L{i}: {w.material} {w.thickness_cm:g} cm"))
    cz, pr = spec.inner_radius_cm / 2.0, spec.phantom_radius_cm
    fig.add_shape(type="circle", x0=-pr, x1=pr, y0=cz - pr, y1=cz + pr,
                  fillcolor="#6fb3ff", line=dict(color="#cfe8ff", width=1))
    fig.add_annotation(x=0, y=cz, text="crew", showarrow=False,
                       font=dict(color=BG, size=10))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=10), height=470,
        showlegend=True, legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(visible=False, range=[-span, span], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-span * 0.35, span]))
    return fig


def _cross_cylinder(spec: HabitatSpec) -> go.Figure:
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
    cz, pr = H / 2.0, spec.phantom_radius_cm
    fig.add_shape(type="circle", x0=-pr, x1=pr, y0=cz - pr, y1=cz + pr,
                  fillcolor="#6fb3ff", line=dict(color="#cfe8ff", width=1))
    fig.add_annotation(x=0, y=cz, text="crew", showarrow=False,
                       font=dict(color=BG, size=10))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=10), height=470,
        showlegend=True, legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(visible=False, range=[-span, span], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-span * 0.35, top]))
    return fig


def cross_section(spec: HabitatSpec) -> go.Figure:
    if spec.shape == "cylinder":
        return _cross_cylinder(spec)
    return _cross_arch(spec)        # dome + quonset share the half-arch slice


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
                 dcc.Slider(id="inner-r", min=1.5, max=6.0, step=0.05, value=2.5,
                            marks=_marks([2, 3, 4, 5, 6]), tooltip=None)),

    html.Div("Wall Layers", style=SECTION),
    html.Div("Innermost first. Add the structural shell, insulation and regolith "
             "overburden as your team designed them.",
             style={"color": MUTED, "fontSize": "11px", "lineHeight": "1.5",
                    "marginBottom": "12px"}),
    html.Div(id="layer-rows"),
    html.Button("+  Add layer", id="add-layer", n_clicks=0, style={
        "background": "transparent", "color": INK, "border": f"1px dashed {BORDER}",
        "borderRadius": "8px", "padding": "9px", "cursor": "pointer", "fontSize": "13px",
        "width": "100%", "marginTop": "2px"}),
    dcc.Store(id="layers", data=DEFAULT_LAYERS),

    html.Div("Scoring Conditions", style=SECTION),
    html.Div(style={**CARD, "padding": "12px"}, children=[
        html.Div("Fixed for every team so the scores are directly comparable.",
                 style={"color": MUTED, "fontSize": "11px", "lineHeight": "1.5",
                        "marginBottom": "6px"}),
        _kv("🔒 GCR field", "φ=400 MV (solar min)"),
        _kv("🔒 Mission", f"{SCORING_MISSION_DAYS} days"),
        _kv("🔒 Statistics", "converged full run"),
    ]),

    html.Button("▶  Evaluate protection", id="run", n_clicks=0, style={
        "background": ACCENT, "color": "#fff", "border": "none", "borderRadius": "8px",
        "padding": "12px", "cursor": "pointer", "fontWeight": 700, "fontSize": "14px",
        "width": "100%", "marginTop": "22px"}),
    html.Button("Cancel run", id="cancel", style={
        "background": "transparent", "color": MUTED, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "8px", "cursor": "pointer", "fontSize": "12px",
        "width": "100%", "marginTop": "8px"}),
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
                   options=[{"label": " Dose cross-section", "value": "cross"},
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
               "(Usoskin 2005) for GCR protons, sampled as an isotropic upper "
               "hemisphere at the lunar surface and transported with "
               "FTFP_BERT_HP.", style={"color": MUTED, "fontSize": "13px",
                                       "lineHeight": "1.6"}),
        _kv("Solar modulation φ", "400 MV (solar minimum, worst case)"),
        _kv("Integral proton flux", f"{gcr_scalar_fluence_rate(400.0):.2f} /cm²/s"),
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

compare_panel = html.Div(id="panel-compare", style={"display": "none"}, children=[
    html.Div("Design Comparison", style={"color": INK, "fontSize": "15px",
                                         "fontWeight": 700, "marginBottom": "12px"}),
    html.Div(style=CARD, children=[
        html.Div("Save and compare multiple designs side-by-side — coming soon.",
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
        dcc.Tab(label="\U0001F6F0 Spacecraft Overview", value="overview",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="\U0001F4CA GCR Environment", value="gcr",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="\U0001F4CB Dose Analysis", value="dose",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="\U0001F5BC Design Comparison", value="compare",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
    ]),
    html.Div(style={"marginTop": "18px"}, children=[
        overview_panel, gcr_panel, dose_panel, compare_panel]),
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
    dcc.Interval(id="poll", interval=1500, disabled=True),
])


# ----------------------------------------------------------------------
# Callbacks: tab switching
# ----------------------------------------------------------------------
@app.callback(
    Output("panel-overview", "style"), Output("panel-gcr", "style"),
    Output("panel-dose", "style"), Output("panel-compare", "style"),
    Input("tabs", "value"))
def _switch_tab(tab):
    show, hide = {"display": "block"}, {"display": "none"}
    return (show if tab == "overview" else hide,
            show if tab == "gcr" else hide,
            show if tab == "dose" else hide,
            show if tab == "compare" else hide)


# ----------------------------------------------------------------------
# Callbacks: wall-layer stack (add / remove / render)
# ----------------------------------------------------------------------
# The store is the source of truth for the *structure* of the stack; the
# row widgets hold the live edits. Add/remove reads the current widget values
# back into the store (so edits survive a re-render) and applies the change.
# The store only updates on add/remove -- never on a keystroke -- so typing in
# a thickness box does not re-render the rows and steal focus.
@app.callback(
    Output("layers", "data"),
    Input("add-layer", "n_clicks"),
    Input({"type": "layer-remove", "index": ALL}, "n_clicks"),
    State({"type": "layer-mat", "index": ALL}, "value"),
    State({"type": "layer-mat", "index": ALL}, "id"),
    State({"type": "layer-thk", "index": ALL}, "value"),
    prevent_initial_call=True)
def _manage_layers(add_n, remove_n, mats, ids, thks):
    trig = ctx.triggered_id
    layers = _layers_from_components(mats, ids, thks)
    if trig == "add-layer" and add_n:
        layers.append({"m": "regolith", "t": 100.0})
    elif (isinstance(trig, dict) and trig.get("type") == "layer-remove"
          and any(remove_n)):
        layers = [L for L, rid in zip(layers, sorted(ids, key=lambda d: d["index"]))
                  if rid["index"] != trig["index"]]
        if not layers:
            layers = [{"m": "aluminium", "t": 6.0}]
    else:
        return no_update                      # mount/no-op: leave the store alone
    return layers


@app.callback(Output("layer-rows", "children"), Input("layers", "data"))
def _render_layers(layers):
    layers = layers or DEFAULT_LAYERS
    n = len(layers)
    return [_layer_row(i, L["m"], L["t"], n) for i, L in enumerate(layers)]


# ----------------------------------------------------------------------
# Callbacks: live geometry / labels / subtitle / design parameters
# ----------------------------------------------------------------------
@app.callback(
    Output("habitat-view", "figure"), Output("view-title", "children"),
    Output("subtitle", "children"), Output("design-params", "children"),
    Output("inner-r-val", "children"),
    Input("shape", "value"), Input("inner-r", "value"),
    Input({"type": "layer-mat", "index": ALL}, "value"),
    Input({"type": "layer-thk", "index": ALL}, "value"),
    Input("view-mode", "value"),
    State({"type": "layer-mat", "index": ALL}, "id"))
def _live(shape, inner_r, mats, thks, view_mode, ids):
    r_label = f"{float(inner_r or 2.5):.2f}"
    try:
        layers = _layers_from_components(mats, ids, thks)
        spec = spec_from_inputs("design", shape, inner_r, layers)
        spec.validate()
    except Exception:
        return (no_update, no_update, no_update, no_update, r_label)

    fig = wireframe_3d(spec) if view_mode == "wire" else cross_section(spec)
    title = (f"3-D Habitat Model — {SHAPE_LABELS[spec.shape]}" if view_mode == "wire"
             else f"Dose Cross-section — {SHAPE_LABELS[spec.shape]}")

    stack = " + ".join(f"{w.thickness_cm * 10:g} mm {w.material}" for w in spec.walls)
    subtitle = (f"Radiation protection score · {SHAPE_LABELS[spec.shape]} "
                f"| {stack} · φ=400 MV · {SCORING_MISSION_DAYS}-day lunar mission")

    params = [
        _kv("Inner radius", f"{spec.inner_radius_cm / 100:.2f} m"),
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
    State("cascade-dir", "data"),
    prevent_initial_call=True)
def _cascade(n, colour_by, shape, inner_r, mats, ids, thks, cascade_dir):
    spec = spec_from_inputs("habitat", shape, inner_r,
                            _layers_from_components(mats, ids, thks))
    trigger = ctx.triggered_id

    if trigger == "cascade-colour":
        # cheap re-render of the existing run in the new colour scheme
        if not cascade_dir:
            return no_update, no_update, no_update
        fig = build_figure(cascade_dir, spec=spec, colour_by=colour_by)
        return _style_cascade(fig), cascade_dir, no_update

    # button: run a fresh headless cascade for this design
    try:
        html_path = run_cascade(spec, n_histories=100, colour_by=colour_by)
    except Exception as exc:
        return no_update, no_update, html.Span(
            f"cascade failed: {str(exc)[-200:]}", style={"color": ACCENT})
    run_dir = str(html_path.parent)
    fig = build_figure(run_dir, spec=spec, colour_by=colour_by)
    n_tracks = len(fig.data) and sum(
        1 for tr in fig.data if getattr(tr, "mode", "") == "lines")
    status = (f"cascade for '{spec.name}' ({spec.areal_density_gcm2():.0f} g/cm² "
              f"shielding) — drag to orbit, scroll to zoom")
    return _style_cascade(fig), run_dir, status


def _style_cascade(fig):
    """Match the cascade figure to the app theme and frame, and pull the camera
    back so the whole hemisphere fits (default 3-D camera clips the bounding box)."""
    fig.update_layout(
        paper_bgcolor=BG, height=560,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(bgcolor=BG, aspectmode="data",
                   camera=dict(eye=dict(x=1.7, y=1.7, z=1.1))),
        legend=dict(orientation="h", x=0, y=1.06, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"))
    return fig


# ----------------------------------------------------------------------
# Callbacks: run / cancel / poll
# ----------------------------------------------------------------------
@app.callback(
    Output("job-store", "data"), Output("poll", "disabled"),
    Input("run", "n_clicks"),
    State("shape", "value"), State("inner-r", "value"),
    State({"type": "layer-mat", "index": ALL}, "value"),
    State({"type": "layer-mat", "index": ALL}, "id"),
    State({"type": "layer-thk", "index": ALL}, "value"),
    prevent_initial_call=True)
def _evaluate(n, shape, inner_r, mats, ids, thks):
    spec = spec_from_inputs("habitat", shape, inner_r,
                            _layers_from_components(mats, ids, thks))
    # Always the fixed scoring preset -- identical conditions for every team.
    jid = default_runner.submit(spec, tier=SCORING_TIER,
                                target_rel_err=SCORING_TARGET_REL_ERR,
                                max_batches=SCORING_MAX_BATCHES,
                                converge_on=SCORING_CONVERGE_ON)
    return jid, False


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
    Input("poll", "n_intervals"),
    State("job-store", "data"),
    prevent_initial_call=True)
def _poll(_n, jid):
    job = default_runner.get(jid) if jid else None
    if job is None:
        return no_update, no_update, no_update, no_update, True

    bar = {"background": ACCENT, "height": "100%", "transition": "width .3s",
           "width": f"{job.progress * 100:.0f}%"}
    err = f" ±{job.rel_err:.0%}" if job.rel_err else ""
    status = (f"[{job.status.value}] {job.spec.name} — batch "
              f"{job.batches_done}/{job.max_batches}{err} — {job.elapsed:.0f}s")

    terminal = job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)
    if not terminal:
        return bar, status, no_update, no_update, False

    if job.status != JobStatus.DONE or not (job.result and job.result.ok):
        msg = html.Div(f"Run {job.status.value}: {(job.error or '')[-300:]}",
                       style={"color": ACCENT, "fontSize": "13px"})
        return bar, status, no_update, [msg], True

    a = assess(job.result, mission_days=SCORING_MISSION_DAYS)
    a_skin = assess(job.result, mission_days=SCORING_MISSION_DAYS, skin=True)
    s = a.summary("career")
    return bar, status, _metric_cards(a, a_skin, s, job), _analysis_body(a, a_skin, s, job), True


def _score_card(score, rel_txt, verdict, frac):
    """The headline the student records: habitat-wide annual effective dose."""
    colour = VERDICT_COLOUR.get(verdict, METRIC)
    style = dict(CARD)
    style.update({"borderLeft": f"4px solid {colour}", "background": "#161d27",
                  "padding": "18px"})
    return html.Div(style=style, children=[
        html.Div("RADIATION PROTECTION SCORE", style={
            "color": MUTED, "fontSize": "11px", "letterSpacing": "0.8px",
            "fontWeight": 700, "marginBottom": "8px"}),
        html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "8px"},
                 children=[
            html.Span(f"{score:.1f}", style={"color": colour, "fontSize": "42px",
                                             "fontWeight": 900, "lineHeight": "1"}),
            html.Span(f"mSv/yr{rel_txt}", style={"color": INK, "fontSize": "14px"})]),
        html.Div(f"{verdict} · {frac:.0f}% of NASA career limit", style={
            "color": colour, "fontSize": "12px", "fontWeight": 700, "marginTop": "10px"}),
    ])


def _metric_cards(a, a_skin, s, job):
    ratio = a.equiv_rate_msv_day / 0.70 if a.equiv_rate_msv_day else 0.0
    # The graded score uses the habitat-wide (wall-lining) scorer: it samples the
    # whole inner surface, so it is far lower-variance than the central crew
    # phantom and gives a number two teams can be compared on. Fall back to the
    # phantom only if the lining scorer is unavailable.
    if a_skin is not None:
        score = a_skin.annual_msv
        s_skin = a_skin.summary("career")
        verdict = s_skin["verdict"]
        frac = s_skin["fraction_of_limit"] * 100
        skin_rel = getattr(job.result, "skin_dose_rel_err", None)
        rel_txt = f" ± {skin_rel:.0%}" if skin_rel else ""
    else:
        score, verdict = a.annual_msv, s["verdict"]
        frac, rel_txt = s["fraction_of_limit"] * 100, ""
    phantom_rel = f" ±{job.result.dose_rel_err:.0%}" if job.result.dose_rel_err else ""
    return [
        _score_card(score, rel_txt, verdict, frac),
        metric_card("Absorbed dose",
                    f"{a.dose_rate_ugy_day / 1000:.3f} mGy/day",
                    f"= {a.annual_mgy:.1f} mGy/year"),
        metric_card("Dose equivalent",
                    f"{a.equiv_rate_msv_day:.3f} mSv/day",
                    f"ISS baseline = 0.70 mSv/day | ratio: {ratio:.2f}×"),
        metric_card("Crew-phantom point dose",
                    f"{a.annual_msv:.1f} mSv/year",
                    f"central self-shielded point{phantom_rel} — noisier diagnostic, "
                    "not the score"),
    ]


def _analysis_body(a, a_skin, s, job):
    skin_err = getattr(job.result, "skin_dose_rel_err", None)
    skin_txt = (f"{a_skin.annual_msv:.1f} mSv/year"
                f"{f' ± {skin_err:.0%}' if skin_err else ''}"
                if a_skin is not None else "n/a")
    return [
        _kv("◆ PROTECTION SCORE (habitat-wide)", skin_txt),
        _kv("Crew-phantom dose (point, diagnostic)", f"{a.annual_msv:.1f} mSv/year"),
        _kv("Absorbed dose rate", f"{s['dose_rate_uGy_per_day']:.3f} µGy/day"),
        _kv("Equivalent dose rate", f"{a.equiv_rate_msv_day * 1000:.2f} µSv/day"),
        _kv("Effective dose (mission)", f"{s['mission_mSv']:.1f} mSv "
            f"over {int(s['mission_days'])} days"),
        _kv("Fraction of career limit", f"{s['fraction_of_limit'] * 100:.1f} %"),
        _kv("Quality factor (mean)", f"{s['quality_factor']}"),
        _kv("Wall transmission", f"{job.result.transmission:.2f}"
            if job.result.transmission else "n/a"),
        _kv("GCR flux used", f"{a.real_flux_cm2_s:.2f} /cm²/s"),
        _kv("Statistics", f"{job.result.n_batches} batches, "
            f"{job.result.total_primaries:,} primaries, {job.result.wall_seconds:.0f}s"),
        html.Div("Equivalent dose uses a single mean field quality factor; secondary "
                 "neutrons (high w_R) are not yet weighted per-particle, so treat mSv "
                 "as indicative.", style={"color": MUTED, "fontSize": "10px",
                                          "marginTop": "12px", "fontStyle": "italic"}),
    ]


def main_entry():
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main_entry()
