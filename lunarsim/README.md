# lunarsim — lunar-surface radiation dose tool

`lunarsim` estimates the radiation dose to a crew inside a lunar-surface habitat
from the design's shape, size and wall stack. It is for theoretical and planned
missions and for early concept-design comparisons. Galactic cosmic rays (GCR) and
a solar particle event (SPE) are transported with Geant4 (through TOPAS), or
folded through precomputed Geant4 response kernels. The tool reports:

- annual GCR effective dose in mSv/yr
- the acute dose from an SPE
- an honest uncertainty statement

Lower dose is more protective.

The dose is computed **target-blind**: no dose value or dose limit is fed back
into the physics. Limits are used only to colour a verdict after the dose has
been computed.

---

## Part 1 — Using the tool

### Launching the GUI

```bash
export TOPAS_G4_DATA_DIR=~/G4Data        # required every session
cd ~/topas
python3 -m lunarsim.gui                  # serves http://127.0.0.1:8050
```

The GUI runs with `debug=False`, so it does not reload when the code changes;
restart it after editing. On the shared PC it runs as the `lunarsim-gui` systemd
service. After a pull or a data change, run `sudo systemctl restart lunarsim-gui`
(see `STAFF_RUNBOOK.md`).

### Inputs

| Input | Meaning |
|---|---|
| Shape | Dome (half-sphere), vertical cylinder, half-cylinder (quonset), or buried. |
| Inner radius / length | Size of the habitable volume. |
| Wall layers | Concentric layers, innermost first, each a material and a thickness. The available materials are aluminium, polyethylene, water, concrete, regolith, titanium, stainless steel and the EVA-suit laminates. |
| Burial / regolith site | Regolith cover, and the composition of the terrain beneath the habitat. Mare is the default; highland is the alternative. |
| Scenario | Chronic GCR, the acute SPE, or both. Both is reported as two separate gates that are never summed. |
| Solar-cycle phase | GCR modulation φ. 400 MV is solar minimum and the default worst case; 650 and 1100 MV are the other phases. |
| SPE design event | Feb-1956 (hard; the default because it is the worst case behind shielding), Oct-1989, or Aug-1972. |

### Reading the result

The headline card shows these numbers:

```
RADIATION PROTECTION SCORE
   197.7 mSv/yr          (10 g/cm² Al, kernel fold)
   systematic 153–260 mSv/yr · ± above is MC statistics only
```

- **The number** is the annual effective dose (ICRP-60 Q(L)) over the habitat.
- **The ±** is Monte Carlo statistics only. It shrinks with more histories. The
  thin-wall kernel fold has no ± of its own.
- **The systematic band** is the spread allowed by the models. More histories do
  not shrink it. Hover over the band to see each term. `headline_context.systematic_band`
  defines the terms:

  | Term | Size | In band |
  |---|---|---|
  | GCR input spectrum (Matthiä vs PAMELA/BON2020) | ×/÷ 1.15 | yes (quadrature) |
  | Transport code (Geant4 vs HZETRN/OLTARIS) | ×/÷ 1.25 | yes (quadrature) |
  | Geant4 physics list (QMD vs modular) | ±7% | no; contained in the transport term |
  | Energy above the 20 GeV/n source ceiling (neither engine; estimated) | +1.5% at the surface → +4% at 50 g/cm² | yes (one-sided) |
  | Engine hand-off near 19 g/cm² | up to ×1.20 (measured), fading to zero at ±20% | yes (one-sided) |
  | Q convention (NASA/Cucinotta vs ICRP) | +20% | no; reported as a cross-check |

  When comparing two designs, a difference smaller than the band is not a
  meaningful difference.

- **Verdicts** compare the mission dose with several agencies' career limits.
  The verdict is policy layered on the physics; it is not a physics output.
- **Context flags** name the engine that produced the number and give a
  confidence level for the areal density. They also warn when a design sits near
  the engine hand-off.
- The downloadable report adds a **provenance stamp**, for example:

  ```
  git=a5ae72f+dirty engine=kernel lis_form=rigidity perelement=perelement cal=0.949335 kernel_sha=…
  ```

  It records the code version, GCR model and calibration behind the number.

### Two engines

GCR dose is computed by one of two engines, chosen by the wall's areal density.
The hand-off is at about **19 g/cm²**.

| Wall | Engine | Why |
|---|---|---|
| < 19 g/cm² | **Thin-wall kernel fold** (`dosimetry.fold_gcr_thinwall`). A precomputed per-element (10-ion), per-organ response R(E) is folded against the free-field GCR flux. This uses the rigidity LIS. | In this regime a thin wall does not build up a secondary shower, so the flood normalisation over-counts. The fold is fast and needs no Monte Carlo. |
| ≥ 19 g/cm² | **Flood Monte Carlo** (`jobs.run_composition`). The design itself is transported in Geant4: 5 ions, energy LIS, the matthia2013 composition, converged on the wall lining and on the phantom. | At these thicknesses the field is dominated by wall-bred secondaries, so the design has to be transported. |

The step between the two engines at the gate is the least certain part of the
sweep, and it is carried in the systematic band. Materials with their own kernel
are aluminium, polyethylene and the EVA laminate. Any other wall is folded on
the aluminium kernel and flagged *indicative*.

### Validation status (summary)

- **Chang'E-4 LND (charged component).** A thin-Si detector analogue gives
  0.87–0.90× the measured LND charged dose (z ≈ −1), ⟨Q⟩ 3.95 against
  4.3 ± 0.7. The neutral component is reported but not used as a gate, because
  the measurement is contaminated by the lander platform.
- **Skin dose** agrees with SHARC/OLTARIS lunar-surface values.
- **Thick-shield absorbed dose** against OLTARIS at 54 g/cm² Al is being
  re-verified after the Matthiä absolute-normalisation change.
- **Mean Q against depth** is checked against the CRaTER/TGO reference. ⟨Q⟩ is
  0.9× the reference at the surface and reads mildly low at depth.

---

## Part 2 — Maintainer guide

### Pipeline

```
HabitatSpec ─► geometry ─► bridge ─► jobs ─► dosimetry ─► headline_context ─► gui
```

| Module | Responsibility |
|---|---|
| `spec.py` | `HabitatSpec` / `WallLayer`: the single contract between the GUI and the engines. Also holds the materials library, the radii and the areal-density/mass figures of merit. |
| `geometry.py` | Emits TOPAS geometry and scorers: nested wall shells, the fluence gauges, the central organ phantom and the habitat-wide CrewSkin lining. |
| `bridge.py` | Builds the runnable parameter file, generates the source via `make_source.py` (in a subprocess that is passed the active model's env explicitly), runs `topas` and parses the CSVs into a `RunResult`. |
| `jobs.py` | Converge-by-error batch loops (`run_converged`, `run_composition`, with a χ² upper-bound gate) and the background `LocalThreadRunner`. |
| `dosimetry.py` | Flux normalisation, assessments (`assess_composition`, `assess_gcr_thinwall`, `fold_spe`), the kernel folds, `GcrModel` and provenance. |
| `headline_context.py` | Dash-free advisory flags plus `systematic_band`. These are labels only and never change a dose. |
| `gui.py` | Dash app: design editor, previews, evaluate → poll → cards, and the report. |
| `kernel_gen/` | PC-side kernel regeneration (`generate.py`, `tail.py`). Output JSON is provenance-stamped. |
| `data/` | Shipped kernels: thin-wall Al/poly/EVA/per-element, plus the SPE kernel. The per-element kernel runs to the 20 GeV/n flood source ceiling (tail promoted 2026-11-05; the pre-tail file is kept as `*_pretail.json`). |

### The GCR model is an explicit value

`dosimetry.GcrModel` is a frozen dataclass with four fields: `lis_form`,
`perelement`, `emax_per_nuc` and `shape`. The `LUNARSIM_GCR_*` env vars only
seed the default (`GcrModel.from_env()`). Code that needs a different model
scopes it rather than mutating the env:

```python
from dataclasses import replace
from lunarsim import dosimetry
with dosimetry.use_model(replace(dosimetry.current_model(), lis_form="energy")):
    ...   # make_source, calibration, flux integrals all see this model
```

- `use_model` is thread-local, and `make_source` is cached per model.
- The thin-wall fold scopes `KERNEL_LIS_FORM`, and the flood scopes
  `current_model().flood()`. Neither one can leak into the other.
- `assess_composition(..., model=comp.model)` reuses the model the Monte Carlo
  actually ran.
- Every assessment carries a `provenance` dict; `provenance_str` gives its
  one-line form. Harnesses write it into their CSVs, and
  `provenance_mismatch` flags rows that are stale or unstamped.

### Invariants

- **Target-blind.** No dose, reference value or limit feeds back into the
  physics.
- **Calibration is not refit.** `OUTER_GAUGE_ANCHOR_CAL` is fixed; when a number
  is wrong, fix the root cause rather than re-tuning the calibration.
- Converge on the quantity you display. `SCORING_CONVERGE_ON = "both"` converges
  both the lining and the phantom.

### Tests

These tests run without TOPAS. The GUI tests also need `dash`.

```bash
python3 -m unittest lunarsim.tests.test_pipeline lunarsim.tests.test_crossover_coherence \
  lunarsim.tests.test_convergence_gate lunarsim.tests.test_headline_context \
  lunarsim.tests.test_regolith_site lunarsim.tests.test_spe_event lunarsim.tests.test_tail_harness \
  lunarsim.tests.test_gate_decompose lunarsim.tests.test_tail_vs_flood \
  lunarsim.tests.test_geometry_probe
```

### Environment

| Variable | Purpose |
|---|---|
| `TOPAS_G4_DATA_DIR` | Geant4 data (required). |
| `TOPAS_ROOT`, `TOPAS_BIN` | Engine location (default `~/topas`). |
| `LUNARSIM_THREADS` | TOPAS threads per run. |
| `LUNARSIM_GCR_LIS` / `_PERELEMENT` / `_EMAX_PER_NUC` / `_SHAPE` | Seed the default `GcrModel`. The per-element kernel needs `perelement`, which is the default. |
| `LUNARSIM_SI_DETECTOR_UM` | Adds a thin-Si detector, used for the LND validation. |

### Programmatic use

```python
from lunarsim import HabitatSpec, WallLayer, run_composition, assess_composition
from lunarsim import dosimetry
from lunarsim.bridge import FULL_RUN

spec = HabitatSpec(name="d", shape="dome", inner_radius_cm=300.0,
                   walls=[WallLayer("aluminium", 0.6), WallLayer("regolith", 50.0)])
if spec.areal_density_gcm2() < 19.0:
    a = dosimetry.assess_gcr_thinwall(spec, phi_MV=400.0)          # kernel fold
else:
    comp = run_composition(spec, tier=FULL_RUN, converge_on="both")
    a = assess_composition(comp.species_results, phi_MV=400.0, skin=True,
                           model=comp.model)                        # flood MC
print(f"{a.annual_msv:.1f} mSv/yr", dosimetry.provenance_str(a.provenance))
```
