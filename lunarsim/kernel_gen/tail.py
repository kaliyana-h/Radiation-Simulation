#!/usr/bin/env python3
"""kernel_gen.tail -- high-energy TAIL extension for the thin-wall GCR kernel.

WHY THIS EXISTS
---------------
The committed kernel's top per-nucleon nodes are 6000 (H/He) and 5000 (C/Si/Fe)
MeV/n, but make_source emits free-field GCR flux up to GCR_EMAX_PER_NUC =
20000 MeV/n. dosimetry._thinwall_fold_point integrates the spectrum only between
the lowest and HIGHEST node, so all flux above the top node is HARD-TRUNCATED --
not extrapolated. That dropped 5-20 GeV/n band is ~4-7% of each species' fluence
and a larger share of dose for the penetrating, high-Q heavy ions. It is the
"energy-ceiling caveat" the paper currently STATES as an uncertainty instead of
MEASURING (memories compute-scaling-priorities, effective-dose-option-c i7/i9).

This module runs a focused batch at the APPEND nodes config.TAIL_NODES_PERNUC_MEV
(top node pinned to the free-field ceiling, 20000 MeV/n) and MERGES the responses
onto the committed kernel's existing nodes. The base R values are preserved
byte-for-byte, so the Al --validate gate is unaffected: the tail is additive.
After the merge the fold's top edge == GCR_EMAX_PER_NUC and nothing is dropped,
which converts the stated ceiling caveat into a measured delta (see `delta`).

TARGET-BLIND: no dose value, limit, or external code (OLTARIS/Chang'E/SHARC) is
used anywhere in the emit/merge/delta path. `delta` reports only the fractional
change the tail makes to the tool's own number; judge convergence first.

HANDOFF WORKFLOW (assistant builds; USER runs the MC on the 24-core PC)
-----------------------------------------------------------------------
  # 0) sanity: confirm the ceiling still matches make_source, and see run counts
  python -m lunarsim.kernel_gen.tail plan --seeds 4

  # 1) EMIT the tail param files (laptop or PC; pure text, no TOPAS)
  python -m lunarsim.kernel_gen.tail emit --out runs/tail --seeds 4

  # 2) RUN on the PC (the only slow, TOPAS step):
  export TOPAS_G4_DATA_DIR=~/G4Data
  cd ~/topas && time bash runs/tail/run_all.sh   # 24 single-thread jobs at a time

  # 3) MERGE the tail responses onto the committed kernel:
  python -m lunarsim.kernel_gen.tail merge --out runs/tail \
         --base lunarsim/data/gcr_thinwall_kernel.json \
         --dest lunarsim/data/gcr_thinwall_kernel_tail.json

  # 4) target-blind acceptance: how much did the tail move the tool's own E?
  python -m lunarsim.kernel_gen.tail delta \
         --base lunarsim/data/gcr_thinwall_kernel.json \
         --tail lunarsim/data/gcr_thinwall_kernel_tail.json

  # 5) to ship: copy _tail.json over gcr_thinwall_kernel.json (keep the old as a
  #    backup) ONLY after `delta` reads as a small, physically-sensible lift.

The tail batch is EMITTED with composition="tail" (config.species_corrected_tail),
so it reuses generate.emit / run_all.sh / _collect_point unchanged; only the merge
and the delta report live here.

PER-ELEMENT variant (the #3 run for the SHIPPED 10-ion kernel)
--------------------------------------------------------------
The shipped THIN engine is the 10-ion per-element kernel, not the old 5-ion
corrected one. Pass `--composition perelement_tail` (config.species_perelement_tail)
to every step to extend THAT kernel's 6000/5000 MeV/n tops to the ceiling instead.
The base/dest are then the per-element files:

  python3 -m lunarsim.kernel_gen.tail plan  --seeds 4 --composition perelement_tail
  python3 -m lunarsim.kernel_gen.tail emit  --out runs/tail_perelem --seeds 4 \
         --composition perelement_tail
  # ... run runs/tail_perelem/run_all.sh on the PC ...
  python3 -m lunarsim.kernel_gen.tail merge --out runs/tail_perelem \
         --base lunarsim/data/gcr_thinwall_kernel_perelem.json \
         --dest lunarsim/data/gcr_thinwall_kernel_perelem_tail.json
  python3 -m lunarsim.kernel_gen.tail delta \
         --base lunarsim/data/gcr_thinwall_kernel_perelem.json \
         --tail lunarsim/data/gcr_thinwall_kernel_perelem_tail.json

merge auto-detects the composition from the batch manifest and guards that --base
is the matching base kernel (config.TAIL_COMPOSITIONS). delta auto-sets the
per-element fold model from the base kernel's meta.composition, so the folded
numbers are element-aware without an env export.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import config, generate


# --------------------------------------------------------------------------
# EMIT (thin wrapper over generate.emit with composition="tail")
# --------------------------------------------------------------------------
def emit(out: Path, seeds: int, threads: int, material: str = "aluminium",
         wall_geometry: str = config.WALL_GEOMETRY,
         composition: str = "tail") -> None:
    _check_ceiling(composition)
    generate.emit(material, out, seeds, threads, composition=composition,
                  wall_geometry=wall_geometry)


def _check_ceiling(composition: str = "tail") -> None:
    """Fail loudly if the tail ceiling has drifted from make_source's free-field
    ceiling, or if any species in the tail table does not top out exactly ON that
    ceiling -- the whole point is for every tail top node to land ON GCR_EMAX_PER_NUC
    so the fold loses no residual flux."""
    if composition not in config.TAIL_COMPOSITIONS:
        raise SystemExit(
            f"composition {composition!r} is not a tail composition; expected one of "
            f"{tuple(config.TAIL_COMPOSITIONS)}")
    try:
        import make_source as ms  # repo-root module
        emax = float(ms.GCR_EMAX_PER_NUC)
    except Exception as exc:        # pragma: no cover - import guard
        print(f"  (note: could not import make_source to verify ceiling: {exc})")
        emax = None
    if emax is not None and abs(emax - config.TAIL_CEILING_PERNUC_MEV) > 1.0:
        raise SystemExit(
            f"TAIL ceiling {config.TAIL_CEILING_PERNUC_MEV:g} MeV/n != make_source."
            f"GCR_EMAX_PER_NUC {emax:g}. The tail's top node must equal the free-field "
            f"ceiling or the fold still truncates. Update config.TAIL_CEILING_PERNUC_MEV "
            f"to {emax:g}.")
    for name, s in config.species_for(composition).items():
        top = s["nodes_pernuc_mev"][-1]
        if abs(top - config.TAIL_CEILING_PERNUC_MEV) > 1.0:
            raise SystemExit(
                f"tail species {name} top node {top:g} != ceiling "
                f"{config.TAIL_CEILING_PERNUC_MEV:g} MeV/n: the fold would still "
                f"truncate above it")


# --------------------------------------------------------------------------
# MERGE (append tail nodes onto the committed kernel)
# --------------------------------------------------------------------------
def merge(out: Path, base: Path, dest: Path) -> dict:
    """Append the tail-batch responses in RUNDIR to the committed kernel `base`,
    writing the extended kernel to `dest`. Per point (matched by wall_gcm2) and
    species, the tail nodes_pernuc_mev and R/Rsem arrays are concatenated after the
    existing ones. Base values are copied through untouched."""
    manifest = json.loads((out / "manifest.json").read_text())
    composition = manifest.get("composition")
    if composition not in config.TAIL_COMPOSITIONS:
        raise SystemExit(f"manifest composition is {composition!r}, expected a tail "
                         f"composition {tuple(config.TAIL_COMPOSITIONS)} -- did you "
                         f"`emit` with this module?")
    material = manifest["material"]
    seeds = manifest["seeds"]
    runs_dir = out / "runs"
    K = json.loads(base.read_text())
    organs = [name for name, _ in config.organs_from_reference()]

    # Guard: the base kernel must be the one this tail composition extends -- merging
    # a 10-ion perelement_tail onto the 5-ion base (or vice versa) would silently
    # merge only the overlapping species and ship a corrupted kernel.
    base_comp = K["meta"].get("composition", "reference")
    allowed_base = config.TAIL_COMPOSITIONS[composition]
    if base_comp not in allowed_base:
        raise SystemExit(
            f"base kernel composition {base_comp!r} does not match tail composition "
            f"{composition!r} (expected base in {allowed_base}); wrong --base?")

    tail_by_wall = {
        p["wall_gcm2"]: p for p in
        (generate._collect_point(runs_dir, material, w, seeds, composition)
         for w in manifest["grid_gcm2"])
    }

    # Species-set guard: base and tail must carry the SAME species, or the per-base
    # loop below would drop tail-only species and mis-pair the rest.
    base_species = set(K["points"][0]["species"])
    tail_species = set(next(iter(tail_by_wall.values()))["species"])
    if base_species != tail_species:
        raise SystemExit(
            f"species mismatch: base has {sorted(base_species)}, tail batch has "
            f"{sorted(tail_species)} -- base kernel and tail composition disagree")

    appended = 0
    for p in K["points"]:
        tp = tail_by_wall.get(p["wall_gcm2"])
        if tp is None:
            raise SystemExit(f"no tail point for base anchor {p['wall_gcm2']} g/cm^2 "
                             f"(tail grid={list(tail_by_wall)})")
        for sname, bs in p["species"].items():
            ts = tp["species"].get(sname)
            if ts is None:
                raise SystemExit(f"tail batch is missing species {sname} at wall "
                                 f"{p['wall_gcm2']} -- composition mismatch")
            tnodes = ts["nodes_pernuc_mev"]
            # guard: tail nodes must sit strictly ABOVE the existing top node
            if tnodes and tnodes[0] <= bs["nodes_pernuc_mev"][-1]:
                raise SystemExit(
                    f"tail node {tnodes[0]} <= base top {bs['nodes_pernuc_mev'][-1]} "
                    f"for {sname}: tail must extend, not overlap, the base grid")
            bs["nodes_pernuc_mev"] = bs["nodes_pernuc_mev"] + list(tnodes)
            for o in organs:
                for q in ("D", "I"):
                    bs["R"][o][q] = bs["R"][o][q] + ts["R"][o][q]
                    bs["Rsem"][o][q] = bs["Rsem"][o][q] + ts["Rsem"][o][q]
            appended += len(tnodes)

    tail_nodes_meta = {name: list(s["nodes_pernuc_mev"])
                       for name, s in config.species_for(composition).items()}
    K["meta"]["tail_extension"] = {
        "composition": composition,
        "nodes_pernuc_mev": tail_nodes_meta,
        "ceiling_pernuc_mev": config.TAIL_CEILING_PERNUC_MEV,
        "seeds": seeds,
        "note": ("High-energy tail appended by kernel_gen.tail so the fold's top "
                 "edge reaches the free-field ceiling (GCR_EMAX_PER_NUC); base "
                 "nodes unchanged. Retires the energy-ceiling truncation caveat."),
    }
    base_note = K["meta"].get("note", "")
    K["meta"]["note"] = ("TAIL-EXTENDED: per-species nodes run up to the "
                         f"{config.TAIL_CEILING_PERNUC_MEV:g} MeV/n free-field ceiling; "
                         "base grid preserved. " + base_note).strip()
    dest.write_text(json.dumps(K, indent=2))
    print(f"merged {appended} tail (species x node x point) responses onto {base.name}")
    print(f"  -> {dest}")
    print(f"  to ship (after `delta` looks sane): cp {dest} "
          f"lunarsim/data/gcr_thinwall_kernel.json  (back up the old one first)")
    return K


# --------------------------------------------------------------------------
# DELTA (target-blind: how much does the tail move the tool's own number?)
# --------------------------------------------------------------------------
def delta(base: Path, tail: Path, phi_MV: float = 400.0) -> None:
    """Fold the base and tail-extended kernels at every areal-density anchor and
    print the fractional change in effective dose-eq E and absorbed D_eff. Pure
    arithmetic on the two kernels -- no geometry, no external reference. This is the
    acceptance read: the tail should ADD a small, positive, monotone-ish lift
    (bigger for thin walls, where the penetrating tail is least attenuated)."""
    import os
    from .. import dosimetry as dz

    Kb = json.loads(base.read_text())
    Kt = json.loads(tail.read_text())
    organs = Kb["meta"]["organs"]
    wmap = {name: w for name, w in organs}

    # The per-element kernel must be folded with the per-element model active
    # (double-gate footgun, memory perelement-kernel-activation): otherwise the
    # fold reads the element-blind spectrum and the delta is meaningless. Drive the
    # model straight off the base kernel's own composition so this is automatic.
    model = "perelement" if Kb["meta"].get("composition") == "perelement" else "none"
    os.environ["LUNARSIM_GCR_PERELEMENT"] = model
    dz.set_perelement_model(model)
    print(f"  (fold model = {model}, from base meta.composition="
          f"{Kb['meta'].get('composition')!r})")

    def fold_point(point):
        f = dz._thinwall_fold_point(point, phi_MV, organs)
        E = sum(wmap[k] * f["HI"][k] for k, _ in organs)
        D = sum(wmap[k] * f["HD"][k] for k, _ in organs)
        return E, D, f["per_species"]

    bpts = {p["wall_gcm2"]: p for p in Kb["points"]}
    tpts = {p["wall_gcm2"]: p for p in Kt["points"]}
    SV_PER_S_TO_MSV_YR = 1.0e3 * 3.15576e7

    print(f"=== tail delta (phi={phi_MV:g} MV) -- target-blind: tool vs itself ===")
    print(f"{'wall g/cm2':>10} | {'E base':>12} {'E tail':>12} {'dE %':>7} | "
          f"{'D_eff base':>12} {'D_eff tail':>12} {'dD %':>7}   (mSv|mGy /yr)")
    for w in sorted(bpts):
        Eb, Db, psb = fold_point(bpts[w])
        Et, Dt, pst = fold_point(tpts[w])
        Eb_y, Et_y = Eb * SV_PER_S_TO_MSV_YR, Et * SV_PER_S_TO_MSV_YR
        Db_y, Dt_y = Db * SV_PER_S_TO_MSV_YR, Dt * SV_PER_S_TO_MSV_YR
        dE = (Et / Eb - 1) * 100 if Eb else float("nan")
        dD = (Dt / Db - 1) * 100 if Db else float("nan")
        print(f"{w:>10g} | {Eb_y:12.3f} {Et_y:12.3f} {dE:+7.2f} | "
              f"{Db_y:12.3f} {Dt_y:12.3f} {dD:+7.2f}")
    # per-species lift at the thinnest shielded anchor (where the tail matters most)
    w_thin = min(w for w in bpts if w > 0)
    _, _, psb = fold_point(bpts[w_thin])
    _, _, pst = fold_point(tpts[w_thin])
    print(f"\n  per-species E lift at {w_thin:g} g/cm^2 (thinnest shielded anchor):")
    for name in psb:
        eb, et = psb[name]["E_sv_s"], pst[name]["E_sv_s"]
        d = (et / eb - 1) * 100 if eb else float("nan")
        print(f"    {name:3} {psb[name]['group']:<24} dE {d:+6.2f}%")
    print("\n  READ (target-blind):")
    print("  * A small positive lift means the ceiling truncation is now MEASURED, not")
    print("    stated. The lift GROWS with depth (the penetrating >ceiling band is")
    print("    enriched as the soft spectrum attenuates) -- that direction is correct")
    print("    physics, NOT a bad batch. A NEGATIVE jump is the bad-batch signal.")
    print("  * The per-species column says WHICH ions the tail lifts. A proton-led lift")
    print("    LOWERS <Q> (adds low-Q dose), so a ceiling extension cannot cure a deep")
    print("    <Q> UNDERshoot -- confirm that against the per-species read before")
    print("    attributing any deep-<Q> defect to the ceiling.")
    print("  * SHIP CAVEAT: if the deep lift pushes the kernel >~15% above its depth")
    print("    anchor it would break the validated deep absorbed gate (kernel/flood~1.0")
    print("    at 50 g/cm^2, two-gate shell fix) and test_crossover_coherence. Production")
    print("    folds the thin-wall kernel only BELOW the 19 g/cm^2 gate (flood above), so")
    print("    a large deep lift means KEEP THIS AS A DIAGNOSTIC, do not promote to default.")


# --------------------------------------------------------------------------
def _plan(seeds: int, material: str = "aluminium",
          composition: str = "tail") -> None:
    _check_ceiling(composition)
    runs = list(generate.enumerate_runs(material, seeds, composition))
    species = config.species_for(composition)
    grid = generate._grid_for(material)
    nodes = sum(len(s["nodes_pernuc_mev"]) for s in species.values())
    prim = config.RINGS * config.AZIMUTH * config.HISTORIES
    print(f"TAIL batch plan  (composition={composition}, material={material})")
    print(f"  ceiling          = {config.TAIL_CEILING_PERNUC_MEV:g} MeV/n "
          f"(== make_source.GCR_EMAX_PER_NUC)")
    print(f"  anchors g/cm^2   = {grid}")
    print(f"  tail nodes/species:")
    for name, s in species.items():
        tot = [e * s["a"] for e in s["nodes_pernuc_mev"]]
        print(f"    {name:3} {s['nodes_pernuc_mev']} MeV/n  -> total KE {tot} MeV")
    print(f"  species x tail-nodes = {nodes}")
    print(f"  seeds            = {seeds}")
    print(f"  TOTAL RUNS       = {len(runs)}   ({prim} primaries each, single-thread)")
    print(f"  heaviest run     = Fe @ {config.TAIL_CEILING_PERNUC_MEV:g} MeV/n = "
          f"{config.TAIL_CEILING_PERNUC_MEV*56/1e3:g} GeV total KE (the walltime pole)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="print tail run counts + per-species total KE")
    pp.add_argument("--seeds", type=int, default=config.SEEDS)
    pp.add_argument("--material", default="aluminium", choices=sorted(config.MATERIALS))
    pp.add_argument("--composition", default="tail",
                    choices=sorted(config.TAIL_COMPOSITIONS),
                    help="tail=5-ion (old corrected base); perelement_tail=10-ion "
                         "(shipped per-element base). default tail")

    pe = sub.add_parser("emit", help="write the tail TOPAS param files + runner")
    pe.add_argument("--out", required=True, type=Path)
    pe.add_argument("--seeds", type=int, default=config.SEEDS)
    pe.add_argument("--threads", type=int, default=1)
    pe.add_argument("--material", default="aluminium", choices=sorted(config.MATERIALS))
    pe.add_argument("--wall-geometry", choices=config.WALL_GEOMETRIES,
                    default=config.WALL_GEOMETRY,
                    help="MUST match the base kernel's geometry being merged onto "
                         f"(default {config.WALL_GEOMETRY})")
    pe.add_argument("--composition", default="tail",
                    choices=sorted(config.TAIL_COMPOSITIONS),
                    help="tail=5-ion (old corrected base); perelement_tail=10-ion "
                         "(shipped per-element base). default tail")

    pm = sub.add_parser("merge", help="append tail responses onto the committed kernel")
    pm.add_argument("--out", required=True, type=Path)
    pm.add_argument("--base", required=True, type=Path)
    pm.add_argument("--dest", required=True, type=Path)

    pd = sub.add_parser("delta", help="target-blind fold comparison base vs tail")
    pd.add_argument("--base", required=True, type=Path)
    pd.add_argument("--tail", required=True, type=Path)
    pd.add_argument("--phi", type=float, default=400.0)

    args = p.parse_args(argv)
    if args.cmd == "plan":
        _plan(args.seeds, args.material, args.composition)
    elif args.cmd == "emit":
        emit(args.out, args.seeds, args.threads, args.material, args.wall_geometry,
             args.composition)
    elif args.cmd == "merge":
        merge(args.out, args.base, args.dest)
    elif args.cmd == "delta":
        delta(args.base, args.tail, args.phi)


if __name__ == "__main__":
    main()
