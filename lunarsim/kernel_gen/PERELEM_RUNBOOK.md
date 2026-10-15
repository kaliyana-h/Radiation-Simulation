# Per-element (10-ion) GCR kernel — PC handoff runbook

**Purpose (target-blind).** The shipped kernel carries 5 Z²-lumped representatives
(H, He, C=rep Z3-9, Si=rep Z10-20, Fe=rep Z21-28). Collapsing a whole Z-band onto
one representative's Z² is fine for *absorbed* dose (rep/true within ±12 % once
Matthiä abundances are used) but wrong for *dose-equivalent*, because ⟨Q⟩ is
non-linear in LET: one rep-Z assigns the entire band a single Q. The LET-scorer
water-point test (memory `let-scorer-thin-qbar-rootcause`) pins the thin-wall ⟨Q⟩
over-read to exactly this input lumping — at **zero shielding** (transport-free)
Si standing for the Z=10-20 band carries 38 % of water-point dose at ⟨Q⟩≈15.7.

This run replaces the 5 lumped reps with **10 individually carried elements**
(H, He, C, N, O, Ne, Mg, Si, Ca, Fe), each with its **own single-element
Matthiä-2013 spectrum** AND its **own Geant4 fragmentation**. Flux is conserved
exactly: every Z=1-28 is assigned to its nearest carried ion (Voronoi), so the
total GCR flux equals the 5-ion total to 8 decimals — the ion set is a pure
Z-resolution dial, not a flux change.

This does **not** chase any external number (OLTARIS/Chang'E/SHARC/CRaTER). The
acceptance test is the tool-vs-itself ⟨Q⟩ and dose-eq delta only.

## DO THE CHEAP TEST FIRST (decides whether the full regen is worth the cores)

The LET harness is transport-free water-point ⟨Q⟩ — it needs **no** dose kernel,
only the 10-ion composition, and sizes the per-band ⟨Q⟩ error directly. Run this
before committing the PC to the full kernel regen.

```bash
# on the PC, after `git pull`
cd ~/topas
export TOPAS_G4_DATA_DIR=~/G4Data

# emit + run the per-element LET cross-check (minutes, not the full regen)
python3 -m lunarsim.kernel_gen.let_harness emit -o runs/let_perelem \
        --materials aluminium --composition perelement
# the LET ntuple path needs the LETSpectrumWater extension built once (already
# built on the PC from the 2026-10-14 run): cmake -DTOPAS_EXTENSIONS_DIR=~/topas_extensions && make
export TOPAS_G4_DATA_DIR=~/G4Data
bash runs/let_perelem/run_all.sh
python3 -m lunarsim.kernel_gen.let_harness collect -o runs/let_perelem
```

**Read it against the shipped `corrected` run** (same harness, 5-ion): the
decisive number is the per-species ⟨Q⟩ split at the **0 g/cm²** anchor. If the
10-ion ⟨Q⟩(0) drops materially toward the arXiv-2511.09040 arbiter (~5.6) and the
Si/Fe band's dose-eq share falls, the lumping was the culprit and the full regen
is justified. If ⟨Q⟩(0) barely moves, the over-read is cross-code physics (null)
and we stop here — no full regen.

## What the full regen runs (only after the LET test says go)

- **composition:** `perelement` (`config.species_perelement`) — 10 ions, Voronoi
  per-element abundances baked into the kernel JSON.
- **nodes (per-nucleon, MeV/n):** H/He 7 nodes `[80…6000]`; the 8 heavy ions
  6 nodes `[180…5000]` (same grids as the committed C/Si/Fe).
- **anchors:** the shipped Al grid `[2.025, 10.0, 50.0, 0.0]` g/cm², shell
  geometry (the deep-slope fix), cal_material = aluminium.
- **illumination:** unchanged — 4 rings × 8 azimuth × 4 histories = 128 primaries
  per run, 4 seeds.
- **run count:** **992 runs** (vs ~512 for the 5-ion kernel ≈ 1.94×). New HZE
  species added over the committed set: **N, O, Ne, Mg, Ca** (C/Si/Fe already ran).

**Walltime — estimate, not a measurement.** ~1.94× the 5-ion regen plus heavier
per-run cost for the 5 new HZE species (Ca, Fe are the pole). Only 128 primaries
per run, so per-run cost is minutes. On the 24-core PC (992 runs ≈ 42 waves of 24)
plausibly ~1-3 h. **`run_all.sh` logs per-run wall time — read the first wave to
calibrate before assuming the rest.** Each species is independent; bump seeds on a
noisy one later without rerunning the rest.

## PC commands — full regen (USER runs these; assistant cannot run TOPAS)

```bash
# on the PC, after `git pull`
cd ~/topas

# 0) sanity — counts + species (laptop or PC; no TOPAS)
python3 -m lunarsim.kernel_gen.generate plan --material aluminium \
        --composition perelement --seeds 4

# 1) emit the param files + runner  (pure text)
python3 -m lunarsim.kernel_gen.generate emit --material aluminium \
        --out runs/perelem --composition perelement --seeds 4

# 2) RUN (the only slow step)
export TOPAS_G4_DATA_DIR=~/G4Data
time bash runs/perelem/run_all.sh        # 24 single-thread jobs at a time

# 3) collect -> lunarsim/data/gcr_thinwall_kernel_perelem.json
python3 -m lunarsim.kernel_gen.generate collect --out runs/perelem
```

The collect writes `gcr_thinwall_kernel_perelem.json` into `runs/perelem/` (named
from `config.PERELEM_OUT`). Copy it to `lunarsim/data/` to activate:

```bash
cp runs/perelem/gcr_thinwall_kernel_perelem.json lunarsim/data/
```

## How it activates in the tool

`dosimetry.fold_gcr_thinwall` auto-detects the kernel: when
`LUNARSIM_GCR_PERELEMENT=perelement` **and** the file is present, it loads the
10-ion kernel, sets the element-aware LIS model, and folds with the baked Voronoi
abundances (no 5-group override). If the file is absent it resets the model and
falls back to the material kernel — so a half-finished regen never silently folds
a per-element spectrum on a 5-ion kernel. Restart the Dash server after the data
file lands.

## Acceptance (target-blind) — how to read it

Fold a fixed design at every anchor with the env unset (5-ion) and with
`LUNARSIM_GCR_PERELEMENT=perelement` (10-ion), and compare:

- **Expected:** absorbed dose ~unchanged (±12 %, the Z²-lumping was already small
  there); dose-equivalent E **lower** at the thin anchors, driven by the Z=10-20
  band's ⟨Q⟩ dropping once Ne/Mg/Si/Ca are resolved instead of all-Si. The thin
  anchors move most; the 50 g/cm² anchor barely moves (why flood/OLTARIS are fine).
- **Cross-check:** the per-element ⟨Q⟩(depth) should now fall toward the measured
  CRaTER/arXiv-2511.09040 ⟨Q⟩(Al-depth) arbiter it currently overshoots.
- **Reject (do NOT ship):** absorbed dose moving >~15 %, a NaN, or ⟨Q⟩ rising —
  points to a bad batch or a flux-conservation break. Check `runs/perelem/<run>/`
  logs and re-run the laptop smoke (flux ratio must be 1.0000).

The deliverable is the measured ⟨Q⟩ reduction vs the CRaTER arbiter, recorded
where the thin-wall over-read caveat lives in the paper — not a shift in the
headline E.
