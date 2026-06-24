# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Prerequisites (Debian/Ubuntu):**
```bash
apt install -y libexpat1-dev libgl1-mesa-dev libglu1-mesa-dev libxt-dev xorg-dev build-essential libharfbuzz-dev
```

**Build without extensions (base TOPAS):**
```bash
cd ~/topas
unzip Geant4Headers.zip   # only needed once
cmake .
make
```

**Build with C++ extensions:**
```bash
cmake -DTOPAS_EXTENSIONS_DIR=~/topas_extensions
make
```
After adding new extension files, re-run both `cmake` and `make`. For changes to existing extension files only, `make` alone is sufficient.

**Run a simulation:**
```bash
export TOPAS_G4_DATA_DIR=~/G4Data   # required every session
cd /path/to/parameter/file/directory
~/topas/bin/topas MySimulation.txt
```

**Run the project's proton-aluminium simulation:**
```bash
export TOPAS_G4_DATA_DIR=~/G4Data
cd ~/topas
./bin/topas al_slab_scorer.txt
```

## Architecture

TOPAS wraps Geant4 behind a parameter-file–driven interface. Simulations are defined entirely in `.txt` parameter files — no C++ is required unless adding custom components. The binary (`bin/topas`) reads parameter files and drives Geant4 internally using prebuilt libraries in `lib/`.

**Parameter file namespaces:**
- `Ge/` — geometry components (volumes, materials, positions)
- `So/` — particle sources (beam type, energy, angular distribution)
- `Sc/` — scorers (quantities to measure, output files)
- `Ph/` — physics lists
- `Gr/` — visualisation views
- `Ts/` — top-level session settings (threads, seed, verbosity)

**Extension system:** Custom C++ classes are registered at build time. `CMakeHandleExtensions.cmake` parses the first line of each `.cc` file in the extensions directory to determine its role. That first line must follow the exact pattern:
```cpp
// Component for MyComponentName
// Scorer for MyQuantityName
// Filter for MyFilterName
// Particle Source for MySourceName
// Physics List for MyListName
// etc.
```
`TsExtensionManager` (generated from `extensions/*.in` templates) routes parameter-file type names to the correct C++ constructor at runtime.

**Extension base classes (in `include/`):**
| Base class | Purpose |
|---|---|
| `TsVGeometryComponent` | Custom geometry shapes |
| `TsVScorer` | Custom scoring quantities — override `ProcessHits()` |
| `TsVFilter` | Particle filters for scorers |
| `TsVGenerator` | Custom particle generators |
| `TsVMagneticField` / `TsVElectroMagneticField` | Custom fields |
| `TsVOutcomeModel` | Radiobiological outcome models |

**Scorer development:** Subclass `TsVScorer`, implement `ProcessHits(G4Step*, G4TouchableHistory*)` as the per-hit callback, and call `ResolveSolid(step)` before accessing volume or material. Cache parameters in `UpdateForSpecificParameterChange()` rather than fetching them inside `ProcessHits`.

**Lifecycle hooks available in extensions:** `BeginSession`, `BeginRun`, `BeginHistory`, `EndHistory`, `EndRun`, `EndSession` — only one class per hook per build.

## Project-Specific Simulations

| File | Description |
|---|---|
| `al_slab_scorer.txt` | 200 MeV proton beam through a 5 cm Al slab; fluence and dose scored in a downstream water volume. Uses `QGSP_BIC` physics. |
| `ProtonAluminium.txt` | 48 MeV proton beam through a thin Al sheet; dose scored in 10 Z-bins. Uses `g4em-standard_opt0`. |

Output CSV files (`fluence.csv`, `dose.csv`, `ProtonAlDose.csv`) are written to the working directory.

## Key Notes

- Geant4 data files (`~/G4Data`) must be downloaded separately — they are not included in this repo.
- CMake caches `TOPAS_EXTENSIONS_DIR`; re-run `cmake` if the path changes.
- Extension `.cc` files placed outside the `topas/` directory tree (e.g. `~/topas_extensions/`) are the recommended pattern.
- The `Geant4Headers.zip` file contains headers needed only when building with extensions; unzip into `~/topas/` once.
