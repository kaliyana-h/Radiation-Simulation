# Heavy-ion energy-ceiling TAIL run — PC handoff runbook

**Purpose (target-blind).** The committed kernel tops out at 6000 MeV/n (H/He) and
5000 MeV/n (C/Si/Fe), but the free-field GCR spectrum runs to
`make_source.GCR_EMAX_PER_NUC = 20000` MeV/n. `dosimetry._thinwall_fold_point`
integrates flux only up to the **highest node** and hard-truncates everything
above it — so the 5–20 GeV/n heavy-ion tail is currently *dropped*, not
extrapolated. Measured truncated share: ≈4–7 % of each species' fluence, more in
dose for the penetrating high-Q heavies. This run **measures** that tail so the
paper can retire the "energy-ceiling untested" caveat (converts a *stated*
uncertainty into a *measured* delta).

This does **not** chase any external number (OLTARIS/Chang'E/SHARC). The
acceptance test is the tool-vs-itself fold delta only. Base kernel R values are
left byte-identical; the tail is appended, so `--validate` is unaffected.

## What gets run

- **composition:** `tail` (new; `config.species_corrected_tail`) — carries *only*
  the append nodes, never the base grid.
- **nodes (per-nucleon, MeV/n):** H/He `[12000, 20000]`, C/Si/Fe `[10000, 20000]`.
  Top node == 20000 == `GCR_EMAX_PER_NUC`, so after the merge the fold's top edge
  sits on the ceiling and nothing is dropped.
- **anchors:** the shipped Al kernel's grid `[2.025, 10.0, 50.0, 0.0]` g/cm²
  (cal_material = aluminium). The 0.0 and 2.025 anchors are where the tail lift is
  largest (least attenuation of the penetrating tail).
- **illumination:** unchanged — 4 rings × 8 azimuth × 4 histories = 128 primaries
  per run, flux-correct band secants (`<1/cos>`=2.0), PHI_FF = 0.08418.
- **run count:** 5 species × 2 tail nodes × 4 anchors × 4 seeds = **160 runs**,
  single-thread each, parallelised over cores by `run_all.sh` (`xargs -P nproc`).

**Walltime — estimate, not a measurement.** The 10 heavy-ion runs (C/Si/Fe at
10–20 GeV/n; Fe@20 GeV/n = 1.12 TeV total KE) are the pole — TeV hadronic showers,
but only 128 primaries each, so per-run cost is minutes, not hours. H/He are cheap.
On the 24-core PC (160 runs ≈ 7 waves of 24) the whole batch is plausibly
~20–60 min. **`run_all.sh` logs per-run wall time; read the first wave's log to
calibrate before assuming the rest.** If Fe/Si dominate unexpectedly, bump only
their seeds later — each species is independent.

## PC commands (USER runs these; assistant cannot run TOPAS)

```bash
# on the PC, after `git pull`
cd ~/topas

# 0) sanity — counts + ceiling match (laptop or PC; no TOPAS)
python3 -m lunarsim.kernel_gen.tail plan --seeds 4

# 1) emit the tail param files + runner  (pure text)
python3 -m lunarsim.kernel_gen.tail emit --out runs/tail --seeds 4

# 2) RUN (the only slow step)
export TOPAS_G4_DATA_DIR=~/G4Data
time bash runs/tail/run_all.sh      # 24 single-thread jobs at a time

# 3) merge tail responses onto the committed kernel -> new file
python3 -m lunarsim.kernel_gen.tail merge --out runs/tail \
        --base lunarsim/data/gcr_thinwall_kernel.json \
        --dest lunarsim/data/gcr_thinwall_kernel_tail.json

# 4) acceptance — target-blind fold delta (tool vs itself)
python3 -m lunarsim.kernel_gen.tail delta \
        --base lunarsim/data/gcr_thinwall_kernel.json \
        --tail lunarsim/data/gcr_thinwall_kernel_tail.json
```

## Acceptance (target-blind) — how to read step 4

`delta` folds both kernels at every anchor and prints the % change in effective
dose-eq **E** and absorbed **D_eff**, plus the per-species E lift at the thinnest
shielded anchor.

- **Expected:** a small **positive** lift, **larger at the thin anchors** and for
  the **penetrating heavy ions** (Fe/Si/C), shrinking with shielding. That is the
  previously-truncated tail now being counted — the caveat is measured.
- **Reject (do NOT ship):** a large jump (>~15 %), a negative delta, or a NaN —
  points to a bad/incomplete batch. Check `runs/tail/<run>/topas.log` and the merge
  guard output (tail nodes must sit strictly above the base top node).

## To ship (only after step 4 looks sane)

```bash
cp lunarsim/data/gcr_thinwall_kernel.json lunarsim/data/gcr_thinwall_kernel.json.pre-tail
cp lunarsim/data/gcr_thinwall_kernel_tail.json lunarsim/data/gcr_thinwall_kernel.json
# re-run the Al validate gate to confirm the base R values are untouched:
python3 -m lunarsim.kernel_gen.generate validate --out runs/<base-regen-dir>
# restart the Dash server after the data file changes.
```

Record the measured delta (the step-4 table) in the paper where the energy-ceiling
caveat currently lives — that is the deliverable, not the shift in the headline E.
