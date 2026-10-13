# Kernel deep-slope re-diagnosis — PC handoff runbook

**The open accuracy defect.** The deep-shield KERNEL under-reads ABSORBED dose at
depth: kernel/flood absorbed fell **0.71 → 0.60 across 19 → 50 g/cm² Al** (Sep-21,
`flood_absorbed_depth.py`). The same-physics FLOOD engine (identical
TOPAS/FTFP_BERT_HP) gets the deep absorbed right, so this is **not** cross-code
physics — it is `kernel_gen`'s transport geometry: the wall is modelled as a FLAT
slab whose oblique ring paths (`t/cos θ`) over-attenuate, and that slant penalty
compounds *in the exponent* with thickness. A real regolith/dome wall is a
concentric SHELL (near-radial crossing), so the slab is geometrically wrong at
depth. `wall_geometry_probe.py` already showed a dome-faithful shell @50 g/cm²
folds to **0.302 mGy/day = OLTARIS 0.302** vs the slab's 0.184 (0.61×).

**Why re-diagnose now.** Two things changed since Sep-21:
1. The kernel is now **tail-extended** — its fold includes the 5–20 GeV/n band it
   used to drop (+27% absorbed at 50 g/cm²). That penetrating band is exactly the
   deep component, so the tail extension should have **already closed part of the
   plunge**. We need the *residual* plunge on today's kernel, not the stale number.
2. We never ran the shell bracket at the **thin** anchors (2.025, 10 g/cm²), where
   the kernel is currently ~right — so we don't yet know if a pure-shell regen
   would *over-read* there (which would mean the fix is a thickness-dependent
   obliquity, not a flat shell swap).

**TARGET-BLIND.** Nothing here feeds a dose value into the physics. FLOOD is a
same-physics MC yardstick; OLTARIS is a post-hoc reference. We only measure what
the tool already returns, and we validate the kernel against the *flood*, not
against any external number.

---

## Step 1 — refold (instant, NO MC; run anywhere, but the CSV lives on the PC)

Recomputes only the kernel column against **today's** kernel, reusing the cached
flood rows. Tells us how much of the plunge the tail extension already closed.

```bash
cd ~/topas
python3 paper/flood_absorbed_depth.py runs/floodabs --refold
```

Read the `kern/flood` column of the trend table:
- **Still FALLS with depth** (e.g. ~0.76 → ~0.63) → the deep slab over-attenuation
  survives the tail; the wall-geometry fix is still the open item → go to Step 2.
- **FLAT and ≈1.0 across 19–50** → the tail extension alone closed it; the defect
  is retired. Record the residual and stop — no regen needed.

(If the CSV is missing — fresh PC checkout — run Step 1b once to rebuild the flood
ground truth; it is pure transport and unaffected by the tail, so Sep-21 values
still hold if you have them.)

### Step 1b — (only if no cached flood CSV) rebuild the flood ground truth
```bash
export TOPAS_G4_DATA_DIR=~/G4Data
python3 paper/flood_absorbed_depth.py runs/floodabs --depths 19 30 40 50
# ~50 min/depth, 5-species skin-converged flood MC; resumable (fills the CSV).
```

---

## Step 2 — shell bracket at ALL anchors (decides pure-shell vs thickness-dependent)

Only if Step 1 shows a residual deep plunge. Emit the zero-obliquity SHELL kernel
at the thin anchors too, so the probe can test whether a shell over-reads where the
slab is currently right.

```bash
# laptop or PC: emit (pure text, no MC)
python3 paper/wall_geometry_probe.py emit runs/wallprobe --seeds 4 --anchors 2.025,10,50

# PC: run (after git pull)
export TOPAS_G4_DATA_DIR=~/G4Data
bash runs/wallprobe/run_all.sh

# either machine: fold + verdict
python3 paper/wall_geometry_probe.py report runs/wallprobe
```

Read the probe's own VERDICT block:
- **`THIN REGIME SAFE`** (largest thin shell/slab ≤ 1.20×) → a pure concentric-shell
  `kernel_gen` regen fixes depth without breaking the thin anchors → **GO** on a
  shell regen (change `wall_geometry` default slab→shell; re-validate Al).
- **`THIN REGIME AT RISK`** → obliquity is thickness-dependent (dome curvature); the
  fix is a **per-anchor obliquity** derived from the flood/kernel ratio curve, NOT a
  flat shell swap. Derive the per-anchor `<1/cosθ>` that lands each anchor on the
  flood absorbed, and bake that thickness profile into `config.py`.

---

### Reading Step 2 — the tail and geometry levers are NOT additive

Two cautions, both learned from the 2026-10-13 refold:

1. **Anchor at 50 g/cm² only.** In the 19–40 band the FLOOD yardstick itself
   over-reads (flood@19 = 0.40 vs OLTARIS@19 = 0.267); flood lands on OLTARIS only
   at 50 (0.3065 ≈ 0.302). Read 19–40 as *trend*, decide on the 50 anchor. Add a
   fresh 50 row (`--depths 50`) if the CSV lacks one.
2. **Do not stack shell-regen on top of the tail.** `wall_geometry_probe` folds the
   6-node `corrected` grid, and pre-tail a shell @50 already folded to 0.302 =
   OLTARIS *with the band truncated*. The tail band adds ~+27% absorbed at 50 to
   EITHER geometry, so shell + tail would overshoot OLTARIS by ~27%. Measure the two
   levers jointly: compare the shell bracket against the **6-node base** slab
   (`gcr_thinwall_kernel.base.json`) to isolate the *pure geometry* lift, then model
   geometry × tail together before committing a regen. The post-tail residual the
   fix must close is ~24% at 50 (kern/flood ≈ 0.76), **not** the old ~40%.

## Acceptance (target-blind)

The fix is accepted when, after regen, the **kernel absorbed reproduces the flood
absorbed** across 19–50 g/cm² (kern/flood flat ≈ 1.0 within flood SEM) AND the
thin anchors (2.025, 10) stay within their current ~1.0× of flood — i.e. the depth
SLOPE is corrected without disturbing where the kernel already agrees. The Al
`--validate` gate must still pass (base species/node/organ grid unchanged).

**Do NOT** touch CAL to close this — CAL is a global scale; this is a depth-shape
defect. A CAL nudge would mis-fit every other depth.

## What this does NOT change
- The **flood-regime habitat headlines** (≥19 g/cm², E≈135–200 mSv/yr, 7.5 m SHARC)
  run on the flood engine and are unaffected — this is a kernel-regime (thin-to-mid
  shell) accuracy fix.
- The quality factor is already right at depth (kernel Q̄@50 = 2.18 ≈ OLTARIS 2.16);
  the defect is purely in absorbed dose. Do not re-open the Q model.
