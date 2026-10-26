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
