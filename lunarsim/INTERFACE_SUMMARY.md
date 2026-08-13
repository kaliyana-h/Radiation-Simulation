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
- **Axial length slider (2–12 m)** — shown only for the two elongated shapes
  (cylinder axial length, quonset tunnel length) and hidden for the dome. Left
  untouched, each shape keeps its radius-driven default (cylinder = radius,
  quonset = 3×radius capped at 12 m); the slider makes that length a first-class
  design parameter, which matters because a longer barrel shields the crew
  slightly better against the axial GCR streaming in through the ends. The 12 m
  ceiling matches the quonset's internal cap that keeps the tunnel ends inside
  the ~9 m GCR source dome (uniform flux).
- **Dynamic wall-layer stack** — add/remove up to 8 concentric layers, each with
  a material dropdown and thickness; labelled innermost→outermost. (Built on a
  fixed-pool pattern so typing in a field is never interrupted by re-renders.)
- An **Exposure Scenario toggle** — *GCR (chronic annual field)* vs *Solar Particle
  Event (acute)* — so a team sees both the chronic annual/career verdict and the
  acute 30-day-limit verdict for the same design.
- A **locked "Scoring Conditions" card** whose contents track the scenario (GCR:
  φ=400 MV solar-minimum field, 365-day mission, converged full run; SPE:
  worst-case-inside-shield event — Feb-1956 / GLE 5, the hardest modern event —
  scored against the 250 mSv 30-day BFO limit) — fixed and hidden from tampering so
  all designs score on one scale.
- "Evaluate protection" / "Cancel run" buttons.
- A **Design File** section — **Save** the current design to a `.json` file,
  **Load** one back (one-shot import that repopulates every control), and
  **Download results report** (a one-page Markdown report: design, verdict, and
  per-ion / event breakdown).

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
- **Secondary-neutron dose fraction:** a lineage-filtered twin of the ICRP-60 Q(L)
  scorer (`DoseEquivalent_ICRP_Neutron`), scored on the *same* CrewSkin lining,
  sums only the dose-equivalent deposited by charged particles that descend from a
  neutron (full track ancestry via `TsTrackInformation`, enabled with
  `SetNeedsTrackingAction`; creator-process fallback if ancestry is unavailable).
  The reported diagnostic `H_neutron / H_total` is therefore the share of the
  behind-shield crew dose-equivalent carried by **wall-bred (albedo) secondary
  neutrons** — the quantity that *rises* as shielding hardens and the reason a
  thick regolith stack has diminishing returns. Being a ratio of two dose-
  equivalents from one run, its flux/gauge normalisation cancels: it needs no
  calibration. Combined across the GCR composition it is dose-equivalent-weighted;
  it is exact for the dome (single shell) and representative for cylinder/quonset
  (whose secondary linings fold into the reported skin dose but not this ratio).
  This closes the workshop critique that albedo neutrons were transported but not
  reported.
- **Statistically honest:** runs converge to a target precision (5%) and every
  reported figure carries its statistical error band.

### 4.1 Why the two dose numbers differ — cavity theory

The tool scores dose in **two** places, and they disagree by roughly a factor of
1.5: the inner-wall lining converges near **286 mSv/yr** while the central crew
phantom converges near **418 mSv/yr** (SHARC 6 m dome, 147.5 g/cm²). This is not
a bug and not a disagreement about how hazardous the environment is. Both
scorers are the same material (`G4_WATER`) reporting the same quantity (J/kg).
The only difference is geometry — and geometry is sufficient to explain the
entire gap.

**The intuition to discard first:** the phantom does *not* read high because it
is small, nor is the lining diluted by its mass. For a particle that passes
*through* a body, dose is independent of both size and shape. Total track length
inside any convex volume in a uniform field is Φ·V, so energy imparted is
Φ·V·(dE/dx) while mass is ρ·V, and the volume cancels exactly:

> D = (Φ / ρ) · dE/dx

A 33.5 kg ball and a 4 509 kg shell in the same field receive **identical** dose
from everything that transits them, despite the 135× mass ratio. The observed
difference therefore cannot come from transit and cannot come from mass.

**What it does come from: particles that stop.** Stopping is the one case where
the track-length identity breaks, because a stopping particle deposits its whole
residual energy inside whatever contains it, at a steeply rising stopping power.
Containment is a size question, and the two scorers differ sharply:

| scorer | mass | mean chord (4V/S) | contains Bragg peak up to |
|---|---|---|---|
| crew phantom (solid 40 cm sphere) | 33.5 kg | 26.7 cm | ~203 MeV proton (255 MeV on a full diameter) |
| inner-wall lining (2 cm shell) | 4 508.8 kg | 4.0 cm | ~69 MeV proton |

The lining is only 2 cm thick radially but has a 4 cm mean chord, because most
tracks cross it at a grazing angle. Energies via the Bragg–Kleeman relation
R = αEᵖ (α = 0.0022 cm, p = 1.77 for water), checked against NIST PSTAR to
better than 2% over 50–250 MeV.

That gap matters enormously here, because the spectrum emerging from the inside
of 147.5 g/cm² of PE + Al + regolith is *precisely* a spectrum of degraded
50–200 MeV secondaries. A thick shield does not remove the GCR field so much as
grind it into the band the phantom can stop and the lining cannot.

**This is a named, classical effect.** In dosimetry terms the lining is a
**Bragg–Gray cavity** — small compared to the ranges crossing it, so it perturbs
nothing and simply reports the external fluence — while the phantom approaches a
**large cavity**, where particles start and stop inside and it builds its own
internal equilibrium spectrum. Burlin's intermediate-cavity theory interpolates
between the two regimes as a function of cavity size relative to range, which is
exactly the axis these two scorers sit on.

**The numbers back it out quantitatively.** Stripping Q off both readings:

| | absorbed | Q | equivalent |
|---|---|---|---|
| skin (lining) | 147.3 mGy/yr | 1.94 | 285.8 mSv/yr |
| crew phantom | 229.8 mGy/yr | 1.82 | 418.2 mSv/yr |
| ratio | **1.56×** | 0.94× | 1.46× |

If transit contributes identically to both, the phantom's excess is a pure
stopping term accounting for **~36% of its total dose** — energy the lining is
structurally incapable of collecting, no matter how long the run.

**An independent confirmation from a different quantity.** The phantom's mean Q
(1.82) is *lower* than the lining's (1.94), which looks backwards until you
recall the shape of ICRP-60 Q(L): it rises with LET only to ~100 keV/µm (peak
Q ≈ 30) and then falls as 300/√L — the *overkill* branch, encoding the fact that
dumping more energy into an already-dead cell buys no extra damage. Bragg track
ends sit far out on that falling branch, so the extra dose the phantom collects
arrives with a *suppressed* quality factor. Adding a slug of stopping-power dose
should therefore raise absorbed dose while pulling mean Q down — observed:
absorbed **+56%**, mean Q **−6%**. Two quantities that could have moved
independently instead move in the pattern the stopping hypothesis predicts.

**So the two scorers answer different questions.** The lining answers *"what dose
does a thin layer of tissue anywhere on the hull receive"* — the habitat-wide
shielding figure of merit, and the number the design score is built on. The
phantom answers *"what dose does a 40 cm ball of water at the crew position
receive"*, and reads higher because it is simply a better calorimeter. It is a
useful diagnostic, **not** a warning that the crew position is hotter than the
wall.

**Decisive test, not yet run:** sweep `phantom_radius_cm` over 5 / 10 / 20 / 40 cm.
A stopping-dominated dose falls roughly as 1/R; a transit-dominated one stays
flat. That would settle the mechanism by measurement rather than inference.

**Convergence — both quantities, different targets.** Because the phantom is the
crew-representative geometry rather than a curiosity, the run now converges on
**both**: skin to 5%, phantom to 10% (`SCORING_PHANTOM_SLACK = 2.0`), with a
**4-round floor** (`SCORING_MIN_BATCHES`) before any stop is allowed.

The looser phantom target is a wall-time decision, not a standards one. The two
quantities converge at wildly different rates — the skin is done in ~2 rounds and
never improves, while the phantom is a 1/√N grind with 135× less scoring mass.
Measured on a 3 m dome, holding the phantom to the same 5% took **12 rounds
(~17 h)** to reach 4.9%, i.e. it only just arrived and a slightly noisier design
would cap out un-converged anyway. 10% is reached in ~4 rounds.

The **floor is the part that actually does the work.** A phantom `rel_err` from
n=2 is a standard error off a two-sample stdev: ~76% relative uncertainty, skewed
*low*, so it routinely under-reports and satisfies a target spuriously. On the
3 m dome round 2 reads 0.0765 — which passes a 10% target outright — while the
same design still read 0.0490 at round 12; on the SHARC dome the n=2 estimate was
3% against a 5-round value of 7%, and the point dose itself moved 511 → 418
mSv/yr, six times its own quoted bar. Converging on "both" *without* a floor
therefore changes nothing: the loop stops at round 2 exactly as before, on an
error bar that cannot be believed.

`PHANTOM_MIN_BATCHES` (the display gate that shows a batch count instead of a
±%) is bound to `SCORING_MIN_BATCHES` so the two cannot drift apart. It is now a
backstop rather than the usual path — it still fires on a cancelled run or one
that caps out at `SCORING_MAX_BATCHES`.

The acute-SPE gate no longer runs an MC of its own. In a combined
("SPE+GCR") job only the GCR composition is transported; the SPE verdict is a
response-kernel fold (`dosimetry.assess_spe` → `fold_spe`) that reads nothing
but the design's `spec`. A direct SPE MC behind thick regolith is rare-tail-
starved — only a few percent of even the hardest event penetrates the wall — so
its deep-organ dose was unreliable and is discarded in favour of the fold. The
practical effect: the SPE gate is now instant, and it can no longer veto a GCR
result it never fed.

**Cost:** ~2 rounds → ~4 rounds per scoring run, roughly +1–2 hours.

#### References

*Cavity theory and the fluence–track-length identity*
- F.H. Attix, *Introduction to Radiological Physics and Radiation Dosimetry*,
  Wiley (1986) — Ch. 1–2 (fluence, energy imparted), Ch. 8 (charged-particle
  equilibrium), Ch. 10 (Bragg–Gray), Ch. 13 (Burlin intermediate cavity).
- E.B. Podgorsak (ed.), *Radiation Oncology Physics: A Handbook for Teachers and
  Students*, IAEA (2005) — Ch. 2, 9. Freely downloadable.

*Mean chord length*
- ICRU Report 36, *Microdosimetry* (1983) — ⟨ℓ⟩ = 4V/S for a convex body in an
  isotropic field (Cauchy's theorem).
- A.M. Kellerer, "Chord-length distributions and related quantities for
  spheroids," *Radiation Research* **98** (1984) — full distributions, not just
  the mean.

*Range–energy relation*
- W.H. Bragg & R. Kleeman, *Philosophical Magazine* **10** (1905) — original
  R = αEᵖ form.
- T. Bortfeld, "An analytical approximation of the Bragg curve for therapeutic
  proton beams," *Medical Physics* **24**(12) (1997) 2024–2033 — source of
  α = 0.0022 cm, p = 1.77 for water.
- ICRU Report 49, *Stopping Powers and Ranges for Protons and Alpha Particles*
  (1993); NIST PSTAR (Berger et al., NIST Std. Ref. Database 124) — CSDA ranges
  used to check the approximation.

*Quality factor and the overkill branch*
- ICRP Publication 60, *1990 Recommendations of the ICRP*, Ann. ICRP **21**(1–3)
  (1991) — the three-branch Q(L) implemented by the `DoseEquivalent_ICRP` scorer.
- ICRU Report 40, *The Quality Factor in Radiation Protection* (1986) — the
  document that formalised the peak-and-decline into Q(L).
- G.W. Barendsen, "Responses of cultured cells, tumours and normal tissues to
  radiations of different linear energy transfer," *Curr. Top. Radiat. Res. Q.*
  **4** (1968) — measured RBE peak near 100 keV/µm and its decline.
- ICRP Publication 103 (2007) replaced Q(L) with radiation weighting factors for
  most terrestrial purposes but **retained Q(L) for high-LET mixed fields** —
  which is why the space-radiation community, and this tool, still use it.

*Thick shields producing degraded secondary spectra*
- J.W. Wilson et al., *Shielding Strategies for Human Space Exploration*,
  NASA CP-3360 (1997).
- F.A. Cucinotta, M.Y. Kim & L. Ren, "Evaluating shielding effectiveness for
  reducing space radiation cancer risks," *Radiation Measurements* **41** (2006)
  1173–1185 — diminishing returns of thick aluminium.
- T.C. Slaba, S.R. Blattnig & F.F. Badavi, "Faster and more accurate transport
  procedures for HZETRN," *J. Comput. Phys.* **229** (2010).
- M. Durante & F.A. Cucinotta, "Physical basis of radiation protection in space
  travel," *Rev. Mod. Phys.* **83** (2011) 1245 — best single review covering
  both the shielding physics and the quality-factor side.

*Detector-size dependence measured in flight — closest direct empirical support*
- G. Reitz, T. Berger, P. Bilski et al., "Astronaut's organ doses inferred from
  measurements in a human phantom outside the International Space Station,"
  *Radiation Research* **171**(2) (2009) 225–235 — the MATROSHKA torso phantom
  flew precisely because thin detectors and tissue-equivalent volumes were known
  to disagree. The same effect as above, observed with real hardware.

*Validation anchors and source term used elsewhere in this document*
- S. Zhang, R.F. Wimmer-Schweingruber, J. Yu et al., "First measurements of the
  radiation dose on the lunar surface," *Science Advances* **6**(39) (2020)
  eaaz1334 — the Chang'E-4 LND absorbed-dose comparison.
- R.C. Singleterry Jr. et al., "OLTARIS: On-line tool for the assessment of
  radiation in space," *Acta Astronautica* **68**(7–8) (2011) 1086–1097 — source
  of the ~286 mSv/yr skin anchor.
- I.G. Usoskin, K. Alanko-Huotari, G.A. Kovaltsov & K. Mursula, "Heliospheric
  modulation of cosmic rays: Monthly reconstruction for 1951–2004," *JGR* **110**
  (2005) A12108, resting on L.J. Gleeson & W.I. Axford, "Solar modulation of
  galactic cosmic rays," *ApJ* **154** (1968) 1011. *See §6.1.*

> Citations were compiled from working knowledge; author/title/year are reliable,
> but **verify volume and page numbers** before quoting them in a submitted report.

---

## 5. Current status & known limitations

- End-to-end working and validated against flight data for absorbed dose.
- **Dose-equivalent and quality factor read slightly low** vs the bare-surface
  Chang'E-4 value — consistent with the shielding present, plus the documented
  first-order approximation of representing the heavy-ion spectrum with a shared
  per-nucleon proton shape across 5 ion groups. Refinement (measured per-element
  heavy-ion spectra) is a flagged future upgrade.
- **SPE (solar particle event)** is wired into the GUI as a second scenario: a
  single worst-case proton event (fixed integral fluence) scored as a *total event
  dose* against the acute 30-day BFO limit, rather than a chronic annual rate. The
  event spectrum is a physically-faithful **exponential-in-rigidity** form (King
  1974), with named solar-maximum design events (`SPE_EVENTS`) each anchored to
  their measured integral fluence above 30 MeV. The default worst case is the one
  most dangerous *behind the shield*: because 147 g/cm² of regolith stops protons
  below ~430 MeV, a soft huge-fluence event (Aug-1972) is largely absorbed while a
  hard event (Feb-1956 / GLE 5) drives the residual deep-organ dose — a genuine
  shielding-dependent fork the tool now models explicitly.
- **Size-envelope advisory.** The dose normalisation and its ±4% cross-shape
  invariance were validated for designs that fit inside the fixed 800 cm outer
  gauge (the workshop envelope, inner radius up to ~3.5 m). A larger design grows
  the gauge; the 1/R² correction (`_gauge_corr`) still restores the anchor so the
  number is not wrong, but it extrapolates past the validated envelope. The score
  card now **self-flags** this: `dosimetry.gauge_size_flag` grades the design as
  `ok` / `mild` / `strong` from `gauge_corr` alone (target-blind, geometry-only),
  and the GUI appends an amber (mild, `gauge_corr` 0.7–0.9) or red (strong, < 0.7)
  advisory under the Radiation Protection Score telling the user to anchor an
  oversized result against a ≤6 m version. This mirrors the SPE `calibrated` flag
  on the shielding side.
- **Secondary-neutron dose fraction is now reported** (previously an open
  workshop critique: wall-bred albedo neutrons were transported but not
  surfaced). A lineage-filtered twin scorer (`DoseEquivalent_ICRP_Neutron`)
  scores dose-equivalent only from tracks with a neutron anywhere in their
  ancestry, and the pipeline reports `f = H_neutron / H_total`. The fraction is
  normalisation-invariant (both numerator and denominator are scored in the same
  run, so flux/gauge calibration cancels), exact for the dome and representative
  for the cylinder/quonset (computed on the primary crew-skin lining, pre-fold).

### 5.1 Physics validation runs (2026-08)

Four independent cross-checks were run against external references, plus one
follow-up probing the unshielded limit. Each is target-blind: the reference
value never enters the calculation, only the post-hoc comparison.

| # | Check | Reference | Tool result | Verdict |
|---|---|---|---|---|
| 1 | **Proton Bragg range** — 100 MeV pencil beam into water, 260 Z-bins | NIST PSTAR CSDA range 7.718 cm | peak at **7.725 cm** (258/260 bins non-zero) | **0.09% — transport range exact** |
| 2 | **Physics-list spread** — proton-only validated dome, 3 lists | model-form uncertainty | FTFP_BERT_HP **117.6**, Shielding **112.5**, QGSP_BIC_HP **102.7** mSv/yr | **13.5% band**; default sits on the conservative high edge; neutron fraction (~0.60) list-independent |
| 3 | **Unshielded surface anchor** — 1 cm Al dome (~2.7 g/cm²), full GCR, skin | literature ~300–400 mSv/yr | skin **2750 mSv/yr**, eff_Q 8.97 | trend valid but absolute ~2× inflated (thin-shield HZE Bragg over-catch) — **out of clean regime** |
| 4 | **Lunar neutron albedo** — bare Apollo-17 regolith under GCR protons, leakage phase-space plane | LRO LEND leakage spectrum shape | evaporation peak 1–3 MeV + fast tail to ~300 MeV + epithermal 1/E + thermal; 0 downward neutrons; 30.6% in 0.5–8 MeV band | **matches textbook lunar leakage shape**; corroborates the ~60% albedo-neutron dose fraction |

**Follow-up — can the unshielded number be made more accurate?** Re-running case
3 to read the **central crew phantom** (effective-dose proxy) rather than the thin
skin lining, converged on the phantom (16 rounds, 3h25m, rel-err stalled
0.11–0.13), gave **phantom E = 2117 mSv/yr** — only 21% below the skin reading
(2686) and still ~5.6× the ~380 mSv/yr published anchor. Si + Fe carry 49% of the
phantom dose at rel-err ~0.3. **Conclusion:** switching scoring *depth* cannot
rescue an unshielded number, because at ~2.7 g/cm² there is no material to strip
the low-energy heavy ions — the flux reaching *any* tissue depth is already
Bragg-over-caught. This is a transport-**regime** limit, not a scorer-depth
artifact. The trustworthy deliverables remain the **thick-shielded dome
(~130 mSv/yr)** and the **thin-slab light-ion partials**; the unshielded case is
a literature bracket (~300–400), not a tool output, absent variance reduction on
the heavy-ion Bragg tail.

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
- **Energy ceiling (a compute compromise with a measured cost):** the per-nucleon
  spectrum is capped at **20 GeV/nucleon** (`GCR_EMAX_PER_NUC`). The ceiling is
  set by the heaviest ion, because TOPAS beam energy is *total* kinetic energy
  (`T_per_nucleon × A`): Fe (A=56) already reaches **1.12 TeV** at this cap, and
  at the previous 1e5 setting Si and Fe reached 2.8 and 5.6 TeV, whose hadronic
  showers are computationally intractable (a single such primary exhausted memory
  after 18 minutes).

  This truncation is **not free**. Measured sensitivity (3 m dome, 5 cm Al + 30 cm
  regolith, protons, 12 batches per arm, skin annual dose):

  | ceiling | skin annual dose | vs. current | wall |
  |---|---|---|---|
  | 5 GeV/nucleon | 87.7 mSv | −18.2% (12σ) | 0.93× |
  | **20 GeV/nucleon (current)** | **107.2 mSv** | — | 1.00× |
  | 100 GeV/nucleon | 113.2 mSv | +5.6% (4.4σ) | 1.59× |

  So the current ceiling **under-reports proton dose by ~5.6%** relative to a
  100 GeV/nucleon ceiling — a known bias in the *non-conservative* direction.
  Shrinking increments (+22.2%, then +5.6%) suggest the true asymptote lies
  ~7–8% above the current value.

  The cause is that **dose is tail-dominated**: the 5–20 GeV/nucleon slice is only
  **4.27% of the proton flux but carries ~22% of the skin dose** — roughly 5× the
  dose of an average proton, each. Those primaries are near-minimum-ionizing, but
  they shower through 30 cm of regolith and *the shower is where the dose appears*.
  **Flux fraction is not dose fraction** — reasoning from one to the other is off
  by ~5× here, which is why the ceiling cannot be justified on flux grounds alone.
  Raising it is a priority once more compute is available, and it should become
  **per-species**: a per-nucleon cap is the wrong shape, since dose scales per
  nucleon while cost scales with total energy (H is only 20 GeV total at this
  ceiling and stays cheap when raised; Fe does not).

  Crucially, **the same constant is shared by the source generator and the
  dosimetry integral** (overridable together via `LUNARSIM_GCR_EMAX_PER_NUC` for
  sensitivity studies), guaranteeing the simulated spectrum and the
  flux-normalisation spectrum are identical.

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
zenith/azimuth) with a finite angular spread (a cone half-angle). This is explicitly
*not* omnidirectional — the correct shape for a beamed solar event.

The proton spectrum is the standard **exponential-in-rigidity** (King 1974) form —
`dN/dR ∝ exp(−R/R₀)`, where `R = √(E(E+2mₚ))` is the magnetic rigidity and `R₀` the
characteristic rigidity (MV) that sets the hardness — converted onto TOPAS's kinetic
-energy grid. Because SPEs peak at **solar maximum** (the opposite phase to the
worst-case GCR field), `make_source.SPE_EVENTS` carries named solar-max design
events, each with its `R₀` and its measured integral fluence Φ(>30 MeV):

| event | R₀ (MV) | Φ(>30 MeV) /cm² | character |
|---|---|---|---|
| `aug1972` | 100 | 5×10⁹ | soft, extreme fluence — canonical crewed worst case |
| `oct1989` | 150 | 1.5×10⁹ | harder, large fluence (GLE series) |
| `feb1956` | 220 | 1×10⁹ | hardest modern event (GLE 5) |

This mode **is wired into the GUI scoring path**: the interface offers an *exposure
scenario* selector (chronic GCR vs acute SPE), and choosing SPE runs the design
against `WORST_CASE_SPE`. That default is set to the event most dangerous *inside*
the shielded habitat — **`feb1956`** — not the largest event in free space. The
reason is a genuine physics fork: 147 g/cm² of regolith stops protons below
~430 MeV, so only ~3 % of Feb-1956's protons (and ~0.05 % of soft Aug-1972's)
penetrate to the crew. Behind thick shielding the residual dose is set by the *hard
tail*, so Feb-1956 delivers ~40× the behind-shield dose of Aug-1972 despite carrying
5× less total fluence. (The ordering flips for skin / thin-shield exposure, where
Aug-1972's soft proton flood dominates — `SOFT_SPE` is kept for that comparison.)
The result is scored as a **single-event dose against the 250 mSv 30-day BFO limit**,
rather than as an annual rate.

Because only a rare high-energy tail penetrates thick regolith, a direct phantom MC
of an SPE is sampling-starved (the penetrating flux piles into one under-sampled
energy bin and the deep-organ dose fluctuates wildly). The tool therefore scores the
acute SPE dose by **folding** the event spectrum against a pre-built, variance-reduced
proton response kernel R(E) — the same OLTARIS/HZETRN response-function method used
for GCR — rather than by a per-event MC. This lives in `dosimetry.assess_spe`
(`fold_spe` + `data/spe_proton_kernel.json`, 17 dense just-penetrating nodes × 22
seeds through the shielded wall); the per-depth, per-organ NASA/ICRP quality factors
are baked into R(E), so no reruns are needed and the rare-tail starvation is gone.

The headline is the **BFO (deep, ~5 cm) dose-equivalent vs the 250 mSv 30-day limit**
— the binding acute constraint — with the skin (surface) dose reported alongside
against its own 1500 mSv limit. For the 147 g/cm² dome the fold puts even the hardest
historical SPE (Feb-1956) at **~4.5 mSv BFO / ~8 mSv skin** — roughly 1–2 % of the
acute NASA limits. R(E) is measured on the shielded regime it was calibrated on, so a
design whose areal density departs from ~147 g/cm² is flagged *off-calibration /
indicative* rather than silently trusted.

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
