# Radiation-Simulation

Monte-Carlo radiation transport for lunar-habitat design, built on
[TOPAS](https://www.topasmc.org/) (a parameter-file wrapper around Geant4). The
repository contains a full TOPAS build plus **`lunarsim`** — a parametric
lunar-habitat radiation evaluator with a web GUI, written for a
space-station-design workshop.

## What's here

| Path | Description |
|---|---|
| **`lunarsim/`** | The habitat radiation evaluator: takes a design (shape, size, wall layers, materials) and returns a comparable **radiation protection score** (annual effective dose, mSv/yr). Web GUI + Python API. **See [`lunarsim/README.md`](lunarsim/README.md) for the full manual.** |
| `make_source.py` | Generates the galactic-cosmic-ray (GCR) and solar-particle-event TOPAS source includes (isotropic upper-hemisphere GCR; directional SPE). |
| `lunar_environment.txt` | Shared lunar surface include: 3-layer Apollo-17 regolith stack + `FTFP_BERT_HP` physics, used by every habitat. |
| `bin/`, `lib/`, `include/` | The compiled TOPAS engine and headers. |
| `al_slab_scorer.txt`, `ProtonAluminium.txt`, `lunar_habitat.txt`, … | Standalone hand-written TOPAS parameter files (earlier proton-slab work and the original habitat designs). |
| `CLAUDE.md` | Build/run instructions for the base TOPAS engine and the extension system. |

## Quick start (the habitat evaluator)

```bash
export TOPAS_G4_DATA_DIR=~/G4Data         # Geant4 data, required every session
~/topas/.venv/bin/python -m lunarsim.gui   # serve the GUI at http://127.0.0.1:8050
```

Open **http://127.0.0.1:8050**, enter a habitat design, and press
**▶ Evaluate protection**. Full usage — inputs, how to read the score, the fixed
scoring preset, the architecture, and how to extend it — is in
**[`lunarsim/README.md`](lunarsim/README.md)**.

## Building base TOPAS

The compiled engine is included, but to rebuild (or build the C++ extensions),
see **[`CLAUDE.md`](CLAUDE.md)** for prerequisites and the `cmake` / `make`
workflow. Geant4 data files (`~/G4Data`) are downloaded separately and are not in
this repository.

## How the score works (one paragraph)

`lunarsim` simulates GCR protons striking the habitat on the lunar surface,
through the design's actual wall stack, and scores the dose to a tissue layer
lining the inner wall. Because only ~10⁴ primaries are simulated, the absolute
scale is fixed by normalising to the real measured GCR flux; the Monte-Carlo only
has to get the *shielding response* right. The result is an annual effective dose
in **mSv/yr — lower is more protective** — with a statistical ± band and a verdict
against NASA astronaut dose limits. Galactic cosmic rays are highly penetrating,
so meaningful protection needs large shielding contrasts (tens of cm of regolith),
not millimetres of metal — a core lesson the tool is designed to teach.
