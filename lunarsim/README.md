# lunarsim — Lunar-Habitat Radiation Evaluator

`lunarsim` turns a student's lunar-habitat design (shape, size, wall layers and
materials) into a single, comparable **radiation protection score**: the annual
effective dose to the crew, in **mSv/yr — lower is more protective**. It runs a
real Monte-Carlo radiation transport simulation (Geant4, via TOPAS) of galactic
cosmic rays striking the habitat on the lunar surface, behind the design's actual
wall stack.

It was built for a **space-station-design workshop**: after a team designs its
habitat, the team's radiation lead opens this tool, enters the design, and reads
off the score. Organisers compare the two teams' scores afterwards to decide
whose design is more protective. It is **one student, one design, one evaluation
per session** — not a side-by-side two-team panel.

---

## Part 1 — Operator / student guide

### What the tool answers

> *"Given my habitat's shape, size, and wall layers, how much radiation dose
> would the crew receive per year on the lunar surface — and is that within
> astronaut safety limits?"*

Every evaluation uses the **same hidden, locked simulation settings** (see
[The fixed scoring preset](#the-fixed-scoring-preset)), so two designs evaluated
in separate sessions land on one comparable scale and the score can't be gamed by
fiddling with run settings.

### Launching the GUI

```bash
export TOPAS_G4_DATA_DIR=~/G4Data        # required every session
~/topas/.venv/bin/python -m lunarsim.gui  # serves http://127.0.0.1:8050
```

Open **http://127.0.0.1:8050** in a browser. The server runs with
`debug=False`, so **there is no auto-reload** — if you edit the code, stop the
process (Ctrl-C) and relaunch.

If you run from outside the `~/topas` directory, prepend `PYTHONPATH=$HOME/topas`
so the package is importable.

### Entering a design

The left sidebar holds the design inputs:

| Field | Meaning |
|---|---|
| **Shape** | `Dome` (half-sphere), `Cylinder` (vertical), or `Half-cylinder` (quonset tunnel). |
| **Inner radius** | Interior radius of the habitable volume, in metres. |
| **Wall Layers** | One or more concentric shielding layers, **innermost first**. Each layer has a material and a thickness in mm. Use **+ Add layer** / **✕** to build the stack. |

Available **materials**: aluminium, polyethylene, water, concrete, regolith
(lunar simulant, the in-situ option), titanium.

The centre panel shows a live cross-section of the design and a **Design
Parameters** readout (wall stack, total thickness, areal density in g/cm², shell
mass). These update as you type — no simulation runs yet.

### Reading the score

Press **▶ Evaluate protection**. The tool runs the simulation in the background
(progress bar + batch counter) and then shows the headline card:

```
RADIATION PROTECTION SCORE
   48.1 mSv/yr   ± 1.7%
   SAFE   (8 % of NASA career limit)
```

- **The number** is the annual effective dose to the crew. **Lower is more
  protective.** This is the figure to report to the organisers.
- **The ± band** is the statistical uncertainty of the Monte-Carlo estimate
  (converged to ≤ 5 %). If two designs' bands overlap, they are a **statistical
  tie**, not a clear win.
- **The verdict** (`SAFE` / `MARGINAL` / `EXCEEDS LIMIT`) compares the mission
  dose to the **NASA career limit (600 mSv)**: `SAFE` < 50 %, `MARGINAL`
  50–100 %, `EXCEEDS LIMIT` ≥ 100 %.

Supporting cards show absorbed dose, dose equivalent, and a **central crew-phantom
point dose** — the last is labelled a *noisier diagnostic, not the score*. It has
much larger statistical scatter and is there only for cross-checking; **always
quote the protection score, not the phantom number.**

### Why small wall changes barely move the score

Galactic cosmic rays are extremely penetrating. Adding tens of millimetres of
wall changes the dose only a few percent — often **within the ± band**, so the
score looks unchanged. That is physically correct, not a malfunction: meaningful
protection on the Moon needs *large* shielding contrasts (tens of centimetres of
regolith, not millimetres of metal). To see your design choices clearly separate
the score, vary thickness in the **tens-of-cm** range.

---

## Part 2 — Maintainer / architecture guide

### The pipeline

```
HabitatSpec ──► geometry ──► bridge ──► jobs ──► dosimetry ──► gui
  (spec.py)   (geometry.py) (bridge.py) (jobs.py) (dosimetry.py) (gui.py)
```

| Module | Responsibility |
|---|---|
| `spec.py` | `HabitatSpec` / `WallLayer` dataclasses — the single contract between front end and back end. Holds the `MATERIALS` library, geometry derivations (`layer_radii_cm`, `outer_radius_cm`), and figures of merit (`areal_density_gcm2`, `shell_mass_kg`). The GUI only ever *produces* a `HabitatSpec`; every engine *consumes* one. |
| `geometry.py` | `build_geometry(spec)` emits the TOPAS parameter text: one nested shell per wall layer (innermost first), thin air shells to score fluence in/out, the central tissue **Phantom**, and the habitat-wide **CrewSkin** wall lining. One builder per shape (`_dome`, `_cylinder`, `_quonset`). |
| `bridge.py` | Assembles a complete runnable TOPAS file (`build_parameter_file`), generates the GCR source via `make_source.py`, runs the `topas` binary in a temp dir, and parses the scorer CSVs into a `RunResult`. Also `RunTier` (statistics presets) and the visualisation runners. |
| `jobs.py` | Non-blocking, **converge-by-error** evaluation. `run_converged` runs independent-seed batches until the chosen dose's relative error ≤ target (or a batch cap). `LocalThreadRunner` runs each job in a background thread so the GUI stays responsive; `default_runner` is the shared instance. |
| `dosimetry.py` | Converts a `RunResult`'s per-primary absorbed dose into a physical crew dose and a safety verdict, normalising by the real lunar-surface GCR flux (`assess`, `DoseAssessment`, `DOSE_LIMITS_MSV`). |
| `gui.py` | The Dash web app. Pattern-matching callbacks for the dynamic wall-layer editor, a live cross-section preview, the trajectory-cascade view, and the evaluate → poll → score-card flow. |
| `trajviz.py` | `run_cascade` — a short, low-statistics run that records particle tracks for the 3-D "see the radiation" cascade visualisation. |

### How a dose number is produced

1. **Geometry** — each `WallLayer` becomes a real concentric shell in TOPAS, so
   thickness and material genuinely change the simulated transport.
2. **Source** — `make_source.py` builds a force-field-modulated GCR proton field,
   isotropic over the *unobstructed upper hemisphere* (the Moon body blocks the
   lower half), as ring sources over zenith angle.
3. **Scoring** — two dose scorers: the central **Phantom** (a 20 cm water sphere
   — a point dose, statistically noisy) and **CrewSkin** (a thin tissue shell
   lining the whole inner wall — habitat-wide, far better statistics). Fluence is
   scored just inside and just outside the wall.
4. **Normalisation** (`dosimetry.assess`) — the simulation only fires ~10⁴
   primaries, so the absolute scale comes from the real GCR flux:

   ```
   dose_rate [Gy/s] = D_sim · Φ_real / F_sim
   ```

   where `D_sim` is the scored absorbed dose, `F_sim` is the outer-wall fluence
   (≈ the incident field), and `Φ_real` is the integrated real GCR scalar
   fluence rate at the lunar surface. So the Monte-Carlo only has to get the
   *shielding response* (dose per incident fluence) right.
5. **Effective dose** — Gy → Sv via a single mean field quality factor
   (`DEFAULT_QUALITY_FACTOR = 3.5`, a documented placeholder; the per-particle
   primary/secondary split is a planned upgrade). The headline is annual
   effective dose in mSv/yr from the **CrewSkin** scorer.

### The fixed scoring preset

Defined at the top of `gui.py` and applied to every evaluation so all scores are
comparable:

| Constant | Value | Why |
|---|---|---|
| `SCORING_TIER` | `FULL_RUN` (≈ 10⁴ primaries/batch) | Enough statistics per batch. |
| `SCORING_MISSION_DAYS` | `365` | One-year reference mission. |
| `SCORING_CONVERGE_ON` | `"skin"` | Converge the quantity actually displayed (the CrewSkin headline), not the noisy phantom. |
| `SCORING_TARGET_REL_ERR` | `0.05` | Tighten the headline to ±5 % so genuinely different designs separate beyond the noise band. |
| `SCORING_MAX_BATCHES` | `12` | Batch cap if convergence is slow. |
| GCR modulation | `φ = 400 MV` (solar minimum) | Worst-case GCR intensity. |
| Physics list | `FTFP_BERT_HP` | High-precision neutron transport (REDMoon methodology). |

> **Maintainer note:** converge on the metric you display. Converging on the
> central phantom (a ±~30 % diagnostic) while showing the skin headline leaves
> the score's precision incidental, and two designs that really differ can overlap
> inside their error bars and read as identical. `run_converged(..., converge_on=)`
> selects which relative error the loop watches.

### Environment & dependencies

- `TOPAS_G4_DATA_DIR` must point at the Geant4 data (`~/G4Data`) — set every
  session.
- `bridge.py` locates the engine via `TOPAS_ROOT` (default `~/topas`),
  `TOPAS_BIN` (`~/topas/bin/topas`), `make_source.py`, and
  `lunar_environment.txt` (the 3-layer Apollo-17 regolith + materials include).
  Override with the matching environment variables if the tree moves.
- Python: `~/topas/.venv/bin/python`. Dash provides the web UI; Plotly the
  previews.

### Extending the tool

- **New shape** — add a `_yourshape(spec) -> (geometry_text, wall_names)` builder
  in `geometry.py`, register it in `_BUILDERS`, and add the name to
  `spec.SHAPES`.
- **New material** — add an entry to `spec.MATERIALS` (GUI label → TOPAS material
  name, density g/cm³, hex preview colour). It appears in the dropdown
  automatically.
- **Sharper dose-equivalent** — replace the single `DEFAULT_QUALITY_FACTOR` with
  a per-particle split: a `TsVFilter` on `ParentID` to separate primaries from
  secondaries (albedo neutrons carry a much higher radiation weighting), then
  weight each component. This is the main pending physics upgrade.

### Quick programmatic use

```python
import os; os.environ.setdefault("TOPAS_G4_DATA_DIR", os.path.expanduser("~/G4Data"))
from lunarsim import HabitatSpec, WallLayer, FULL_RUN, run_converged, assess

spec = HabitatSpec(shape="dome", inner_radius_cm=250.0,
                   walls=[WallLayer("aluminium", 0.6), WallLayer("regolith", 50.0)])
result = run_converged(spec, FULL_RUN, target_rel_err=0.05,
                       max_batches=12, converge_on="skin")
score = assess(result, mission_days=365, skin=True)
print(f"{score.annual_msv:.1f} mSv/yr  ± {result.skin_dose_rel_err:.1%}")
```
