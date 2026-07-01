# Lunar Habitat Radiation Tool — Work Summary

*A browser-based design-and-evaluate tool for conceptual lunar habitat shielding,
built on TOPAS/Geant4 Monte-Carlo radiation transport.*

---

## 1. What it is

A browser-based **design-and-evaluate tool** for conceptual lunar habitat
shielding. A user designs a habitat (shape, size, multi-layer walls) through a
graphical interface; the tool runs a **full Geant4/TOPAS Monte-Carlo radiation
transport simulation** in the background and reports the crew's radiation dose
against spaceflight exposure limits. It is built as a workshop instrument: every
design is scored under identical, locked conditions so that separately-designed
habitats are directly comparable.

---

## 2. How it was built — architecture

The system is a clean two-layer design with a single data contract between them:

- **Front end:** a **Plotly Dash** web application (`gui.py`), served locally
  (Flask) at `127.0.0.1:8050`. Pure Python — no separate JavaScript codebase.
  Custom dark theme via a CSS asset file.
- **The contract — `HabitatSpec` (`spec.py`):** every design is captured as one
  plain Python dataclass (shape, inner radius, ordered wall layers with material
  + thickness). The GUI only ever *produces* a `HabitatSpec`; every engine only
  ever *consumes* one. This boundary lets the interface and the physics back end
  evolve independently.
- **Back end (the physics chain):**
  - `geometry.py` — turns a `HabitatSpec` into TOPAS geometry + scorers (with the
    per-species particle filters).
  - `make_source.py` — generates the GCR particle source (force-field-modulated
    Usoskin-2005 interstellar spectrum, isotropic upper hemisphere). *See §6.*
  - `bridge.py` — assembles the TOPAS parameter file and runs the simulation.
  - `jobs.py` — a background job runner with **statistical convergence** (keeps
    adding Monte-Carlo batches until the dose reaches a target precision), plus
    progress/cancel.
  - `dosimetry.py` — converts simulated per-primary dose into a real, calibrated
    crew dose rate and a safety verdict.
- **Material & physics library** is centralised: six materials (aluminium,
  polyethylene, water, concrete, lunar regolith, titanium) each carry their TOPAS
  name, density, and display colour in one table.

---

## 3. Interface functionality (what the user sees)

**Left rail — design controls:**
- Habitat type: dome, cylinder, or quonset (tunnel)
- Inner radius slider (1.5–6 m)
- **Dynamic wall-layer stack** — add/remove up to 8 concentric layers, each with
  a material dropdown and thickness; labelled innermost→outermost. (Built on a
  fixed-pool pattern so typing in a field is never interrupted by re-renders.)
- A **locked "Scoring Conditions" card** (φ=400 MV solar-minimum field, 365-day
  mission, converged full run) — fixed and hidden from tampering so all designs
  score on one scale.
- "Evaluate protection" / "Cancel run" buttons.

**Centre — live visualisation, three tabs:**
- **Spacecraft Overview:** live **3-D wireframe** of the habitat *and* a **2-D
  dose cross-section** (showing the layered wall, regolith ground, and the crew
  phantom). Both update instantly as the design changes. Plus a **particle-cascade
  visualiser** — a headless Monte-Carlo that traces the GCR shower through the
  shielding, colourable by origin (in/out) or particle type.
- **GCR Environment:** documents the radiation source (spectrum, solar modulation,
  integral proton flux, angular distribution).
- **Dose Analysis:** the full per-quantity breakdown, including the **per-ion dose
  share** (H, He, C, Si, Fe groups) — visible proof that heavy ions, not just
  protons, drive the dose.

**Right rail — results:** a metric-card stack — the headline **Radiation
Protection Score** (annual effective dose, colour-coded SAFE/MARGINAL/EXCEEDS vs
the NASA career limit), absorbed dose, dose-equivalent (vs ISS baseline),
crew-phantom point dose — plus a live **Design Parameters** panel (areal density,
outer radius, estimated shell mass in tonnes).

---

## 4. Scientific properties

- **Real Monte-Carlo, not analytics:** numbers come from TOPAS/Geant4 transport
  (FTFP_BERT_HP physics), not formulas.
- **Full GCR composition:** five representative ion groups spanning Z=1–28, each
  transported separately and normalised to its own real flux, then summed — so the
  wall's species-dependent shielding and fragmentation is captured.
- **Absolute calibration:** the GCR flux is anchored to the canonical
  solar-minimum value (~4 protons/cm²/s, 4π). The resulting **absorbed dose
  (~0.41 mGy/day) matches the first direct lunar-surface measurement** (Chang'E-4
  LND, Zhang et al. 2020).
- **LET-weighted quality factor:** dose-equivalent uses a per-step ICRP-60 Q(L)
  scorer, so high-LET ions carry their own quality weighting; the reported mean Q
  emerges as the dose-equivalent/dose ratio.
- **Statistically honest:** runs converge to a target precision (5%) and every
  reported figure carries its statistical error band.

---

## 5. Current status & known limitations

- End-to-end working and validated against flight data for absorbed dose.
- **Dose-equivalent and quality factor read slightly low** vs the bare-surface
  Chang'E-4 value — consistent with the shielding present, plus the documented
  first-order approximation of representing the heavy-ion spectrum with a shared
  per-nucleon proton shape across 5 ion groups. Refinement (measured per-element
  heavy-ion spectra) is a flagged future upgrade.
- SPE (solar particle event) source exists in the generator but isn't yet wired
  into the GUI scoring path.

---

## 6. The radiation source generator — `make_source.py` (detail)

This module is the bridge between **published space-radiation physics** and the
**TOPAS Monte-Carlo geometry**. It does not transport anything itself; it *writes
a TOPAS parameter file* describing the incoming particle field — what particles
arrive, with what energies, and from what directions — which the rest of the
pipeline then includes and simulates. It supports two physically distinct source
modes.

### 6.1 Energy spectrum — where the particles' energies come from

The Galactic Cosmic Ray (GCR) spectrum is **not invented or hand-tuned** — it's
the standard force-field model used throughout the heliophysics literature:

- **Interstellar baseline (LIS):** the proton spectrum outside the heliosphere
  uses the **Usoskin et al. (2005)** analytic fit
  `J_LIS(T) = 1.9×10⁴ · T⁻²·⁷⁸ / (1 + 0.4866·T⁻²·⁵¹)`
  with the *exact published coefficients*.
- **Solar modulation (force-field approximation, Gleeson & Axford 1968):** the
  Sun's activity suppresses low-energy cosmic rays. This is applied through a
  single modulation potential φ (in MV): each particle's spectrum is shifted by
  `Φ = (Z/A)·φ` and rescaled by the relativistic factor
  `T(T+2E₀)/[(T+Φ)(T+Φ+2E₀)]`. The tool runs at **φ = 400 MV**, the deep
  2019–2020 solar minimum — deliberately the *worst-case* GCR environment (least
  solar shielding, highest dose).
- **Per-nucleon scaling for ions:** the spectrum is computed per nucleon and
  shared across ion species, then converted to TOPAS's expected **total kinetic
  energy** (`BeamEnergy = T_per_nucleon × A`). This is the documented first-order
  approximation behind the slightly-low quality factor — a known, flagged
  simplification, not an error.
- **Energy ceiling (a real engineering decision, documented in code):** the
  per-nucleon spectrum is capped at **20 GeV/nucleon** (`GCR_EMAX_PER_NUC`). This
  is a deliberate physics-informed cutoff: above it, heavy ions like Fe (A=56)
  would reach multi-TeV total energies whose hadronic showers are computationally
  intractable (a single such primary exhausted memory after 18 minutes), yet that
  tail carries only ~0.45% of the flux and those particles are near-minimum-
  ionizing — so truncating costs <~1% of dose. Crucially, **the same constant is
  shared by the source generator and the dosimetry integral**, guaranteeing the
  simulated spectrum and the flux-normalisation spectrum are identical.

### 6.2 Angular distribution — where the particles come *from* (the rigorous part)

A common weakness in conceptual radiation models is firing particles from a single
direction. This module answers that directly with a **physically exact isotropic
field**:

- The GCR field is modelled as **isotropic over the unobstructed upper hemisphere**
  of the lunar surface (the ground blocks the lower half), following **Dobynde &
  Guo (2021)**.
- The sky-dome is discretised into **zenith rings × azimuth sectors** (default
  5 × 8 = 40 sources), each a parallel disc beam firing inward toward the habitat.
- The directional weighting is **analytically exact, not ad-hoc**: for an isotropic
  field crossing a flat surface (Lambert's cosine law), the flux from a zenith band
  is proportional to `sin²θ₂ − sin²θ₁`. By choosing ring boundaries **equally
  spaced in sin²θ**, every ring carries *equal* flux — so every source can be given
  the *same* number of histories. This makes the angular sampling both physically
  correct *and* statistically uniform (no ring is under-sampled). Each ring's
  representative zenith is taken at the band midpoint in sin²θ.
- Geometrically, each source is built as a nested TOPAS group hierarchy — azimuth
  rotation (RotZ) → zenith aim (RotY) → translate out to the sky-dome radius and
  flip to fire inward (RotX = 180°).

### 6.3 SPE mode — solar particle events

A second mode generates a **directional cone source** representing a solar particle
event arriving from the Sun's location: a single look-direction (configurable
zenith/azimuth) with a finite Gaussian angular spread (a cone half-angle), and an
exponential proton spectrum `dN/dE ∝ exp(−E/E₀)`. This is explicitly *not*
omnidirectional — the correct shape for a beamed solar event. (Currently a
first-pass placeholder spectrum; wiring it into the GUI scoring path is a flagged
next step.)

### 6.4 Design characteristics worth highlighting

- **Self-contained, includeable output:** the generated source file defines its own
  beam geometry (sky-dome radius, per-source footprint) so it can be dropped into
  any habitat parameter file with a single `includeFile` line — clean separation
  between *environment* and *habitat*.
- **Fully parameterised CLI:** rings, azimuth sectors, modulation potential,
  particle/ion species (Z, A), histories, beam geometry, and a `--standalone` mode
  that emits a complete runnable validation file (regolith slab + fluence scorer)
  for sanity-checking the source in isolation.
- **Single source of truth:** every physical constant and the energy ceiling are
  shared with the dosimetry module, so the *simulated* field and the *real-flux
  normalisation* can never drift apart.

**In short:** the spectrum is taken straight from the cosmic-ray literature
(Usoskin LIS + Gleeson–Axford modulation), the angular field is an exact
Lambert-weighted isotropic hemisphere (Dobynde & Guo), and the whole thing is
emitted as a reusable, self-documenting TOPAS source file — with every
approximation (per-nucleon ion scaling, 20 GeV/nuc cutoff) explicitly justified
in-code rather than hidden.
