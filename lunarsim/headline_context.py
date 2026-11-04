"""Provenance / trust flags that ride on every GCR headline in the GUI and report.

Dash-free on purpose (so it is laptop-testable): gui.py renders these, this module
only decides WHAT to say. Every flag is ADVISORY -- nothing here changes a computed
dose, and nothing reads a dose target or a limit (target-blind). Inputs are the
design's areal density, which engine served the number, and the assessment's mean Q.

Flags, in display order:
  engine      which engine produced the headline (thin-wall kernel fold vs flood MC)
  confidence  dosimetry.areal_density_confidence HIGH / MEDIUM / LOW band
  near_gate   areal density within +/-NEAR_GATE_FRAC of the kernel<->flood crossover,
              where a small wall change can switch engines and step the headline
  q_fallback  the mean Q is the flat DEFAULT_QUALITY_FACTOR, i.e. no LET-weighted
              dose-equivalent scorer reported and the dose-equivalent is NOT emergent
"""
from __future__ import annotations

from .dosimetry import DEFAULT_QUALITY_FACTOR, areal_density_confidence

NEAR_GATE_FRAC = 0.20          # +/-20% of the crossover areal density
DEFAULT_CROSSOVER_GCM2 = 19.0  # kernel meta "crossover_gcm2" (all shipped kernels)

ENGINE_TEXT = {
    "kernel": ("Engine: thin-wall kernel fold — precomputed phantom-matched R(E) "
               "folded against the free-field GCR flux (no flood normalisation)."),
    "flood": ("Engine: flood Monte Carlo — this design transported in Geant4, "
              "scored on the habitat-wide wall lining."),
}

CONFIDENCE_LEVEL = {"high": "ok", "medium": "warn", "low": "alert"}

COMBINED_ENVELOPE_NOTE = (
    "Conservative envelope, not a single solar epoch: the GCR gate uses the "
    "selected solar-modulation phase and the SPE gate the selected design event "
    "independently. Large SPEs cluster near solar maximum, when GCR is suppressed, "
    "so the default pairing (solar-minimum GCR + hardest event) is a worst + worst "
    "bound. Pick a high solar-modulation phase to pair the event with the GCR "
    "field it would actually co-occur with.")


# ---------------------------------------------------------------------------
# Systematic (model) uncertainty band -- shown NEXT TO the MC ± (statistics only).
# Each term is a documented model spread, not a fit; none reads a dose target.
# Symmetric terms combine in quadrature (log-symmetric: lo = 1/(1+s)); one-sided
# terms are known-direction biases and multiply onto one edge only.
GCR_MODEL_REL = 0.15       # GCR input spectrum/normalisation (Matthiae vs PAMELA /
                           # BON2020 spread; memory flood-heavy-abundance-overweight)
TRANSPORT_REL = 0.25       # cross-code transport spread (Geant4 vs HZETRN/OLTARIS,
                           # 20-30%); SUBSUMES the in-Geant4 physics-list swap
PHYSICS_LIST_REL = 0.074   # QMD vs modular fragmentation (fragmentation_sensitivity);
                           # listed for transparency, inside TRANSPORT_REL
CEILING_SURFACE = 0.076    # 20 GeV/n energy-ceiling tail: +7.6% at 0 g/cm2 ...
CEILING_DEEP = 0.21        # ... growing to +21% at 50 g/cm2 (energy-ceiling-tail-
CEILING_DEEP_GCM2 = 50.0   # measured); shipped kernels/flood stop at the ceiling
GATE_STEP = 1.35           # kernel->flood headline step at the crossover (~1.3-1.4x,
                           # dosimetry.fold_gcr_thinwall); RE-MEASURE after step-3 regen
NASA_Q_RATIO = 1.20        # ICRP-60 -> NASA/Cucinotta Q at the organ level (Option C
                           # i3); a CONVENTION choice, reported, not added to the band


def _ceiling_bias(ad_gcm2: float) -> float:
    x = min(max(ad_gcm2, 0.0), CEILING_DEEP_GCM2) / CEILING_DEEP_GCM2
    return CEILING_SURFACE + (CEILING_DEEP - CEILING_SURFACE) * x


def _gate_weight(ad_gcm2: float, crossover_gcm2: float) -> float:
    """1 at the crossover, fading linearly to 0 at the near-gate window edge."""
    half = NEAR_GATE_FRAC * crossover_gcm2
    return max(0.0, 1.0 - abs(ad_gcm2 - crossover_gcm2) / half)


def systematic_band(ad_gcm2: float, engine: str,
                    crossover_gcm2: float = DEFAULT_CROSSOVER_GCM2) -> dict:
    """Multiplicative systematic band for a GCR headline: {lo, hi, terms}.

    terms is a list of {key, label, lo, hi, kind, in_band} (multipliers; kind
    'sym' = two-sided spread, 'bias' = known-direction one-sided). The headline
    times lo/hi brackets the model spread; MC statistics are NOT included (the GUI
    shows them as the separate ±). Purely advisory -- never feeds the physics."""
    if engine not in ENGINE_TEXT:
        raise ValueError(f"unknown engine {engine!r}")
    terms = [
        {"key": "gcr_model", "label": "GCR input spectrum",
         "lo": 1 / (1 + GCR_MODEL_REL), "hi": 1 + GCR_MODEL_REL, "kind": "sym",
         "in_band": True},
        {"key": "transport", "label": "transport code (cross-code spread)",
         "lo": 1 / (1 + TRANSPORT_REL), "hi": 1 + TRANSPORT_REL, "kind": "sym",
         "in_band": True},
        {"key": "physics_list", "label": "Geant4 physics list (within transport)",
         "lo": 1 / (1 + PHYSICS_LIST_REL), "hi": 1 + PHYSICS_LIST_REL,
         "kind": "sym", "in_band": False},
        {"key": "ceiling", "label": "energy ceiling 20 GeV/n (missing tail)",
         "lo": 1.0, "hi": 1 + _ceiling_bias(ad_gcm2),
         "kind": "bias", "in_band": True},
    ]
    w = _gate_weight(ad_gcm2, crossover_gcm2)
    if w > 0:
        step = 1 + (GATE_STEP - 1) * w
        # the kernel reads low of the flood at the hand-off, the flood high
        lo, hi = (1.0, step) if engine == "kernel" else (1 / step, 1.0)
        terms.append({"key": "gate", "label": "engine hand-off",
                      "lo": lo, "hi": hi, "kind": "bias", "in_band": True})
    terms.append({"key": "q_model", "label": "quality-factor convention (NASA Q)",
                  "lo": 1.0, "hi": NASA_Q_RATIO, "kind": "bias", "in_band": False})

    used = [t for t in terms if t["in_band"]]
    sym = sum((t["hi"] - 1) ** 2 for t in used if t["kind"] == "sym") ** 0.5
    lo, hi = 1 / (1 + sym), 1 + sym
    for t in used:
        if t["kind"] == "bias":
            lo *= t["lo"]
            hi *= t["hi"]
    return {"lo": lo, "hi": hi, "terms": terms}


def band_text(value: float, band: dict, unit: str = "mSv/yr") -> str:
    """'systematic 142-246 mSv/yr' -- the bracket the model spread allows."""
    return (f"systematic {value * band['lo']:.0f}–{value * band['hi']:.0f} {unit}")


def is_q_fallback(quality_factor) -> bool:
    """True when the mean Q is exactly the flat default, i.e. no LET-weighted
    dose-equivalent scorer was present. An emergent H/D landing on 3.5 to 1e-9
    is not a realistic coincidence."""
    if quality_factor is None:
        return False
    return abs(float(quality_factor) - DEFAULT_QUALITY_FACTOR) < 1e-9


def headline_flags(ad_gcm2: float, engine: str, quality_factor=None,
                   crossover_gcm2: float = DEFAULT_CROSSOVER_GCM2) -> list:
    """Ordered list of {key, level, text} dicts for one GCR headline.
    level is one of 'info' | 'ok' | 'warn' | 'alert'."""
    if engine not in ENGINE_TEXT:
        raise ValueError(f"unknown engine {engine!r}")
    flags = [{"key": "engine", "level": "info", "text": ENGINE_TEXT[engine]}]

    band = areal_density_confidence(ad_gcm2)
    flags.append({"key": "confidence", "level": CONFIDENCE_LEVEL[band["level"]],
                  "label": band["label"],
                  "text": f"Confidence: {band['level'].upper()} ({band['label']}) — "
                          f"{band['message']}"})

    if abs(ad_gcm2 - crossover_gcm2) <= NEAR_GATE_FRAC * crossover_gcm2:
        side = "below" if ad_gcm2 < crossover_gcm2 else "at/above"
        other = "flood MC" if ad_gcm2 < crossover_gcm2 else "thin-wall kernel"
        flags.append({"key": "near_gate", "level": "alert",
                      "text": f"Near the engine hand-off: {ad_gcm2:.1f} g/cm² is "
                              f"{side} the {crossover_gcm2:.0f} g/cm² crossover "
                              f"(within ±{NEAR_GATE_FRAC:.0%}). A small wall change "
                              f"switches to the {other} and can step the headline; "
                              "compare against a design clearly on either side."})

    if is_q_fallback(quality_factor):
        flags.append({"key": "q_fallback", "level": "warn",
                      "text": f"Mean Q is the flat {DEFAULT_QUALITY_FACTOR:g} "
                              "default — no LET-weighted dose-equivalent scorer "
                              "reported, so the dose-equivalent is assumed, not "
                              "emergent. Treat it as indicative."})
    return flags
