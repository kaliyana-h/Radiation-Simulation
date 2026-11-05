#!/usr/bin/env python3
"""Geometry residual probe: what is left of the kernel-vs-flood SKIN gap once the
20 GeV/n tail is promoted?

After the tail (paper/tail_vs_flood.csv) the flood/kernel skin ratio is

    g/cm^2       5     10     19     25     30     40     50
    resid_skin 1.155  1.103  1.093  1.105  1.125  1.204  1.265

i.e. a FLAT ~1.10 floor plus a RISE above ~30 g/cm^2. Two kernel-geometry
hypotheses can be tested without touching the flood, CAL or the shipped kernel:

  spot    -- BEAM-SPOT TRUNCATION (candidate for the RISE). Each kernel beam is a
             22 cm disc aimed at the phantom centre. The Al shell wall reaches
             r = 15 + t: 18.7 cm at 10 g/cm^2 (inside the disc) but 33.5 cm at 50.
             Primaries with impact parameter 22-33.5 cm hit the wall, never the
             disc, so their wall-bred secondaries are missing from the kernel. The
             variant widens the disc to 45 cm (histories 4 -> 17 so phi_ff stays
             ~0.085). Prediction if truncation drives the rise: lift ~1.00 at 10
             (null control: the 22 cm disc already covers the 18.7 cm wall) and
             > 1 at 50.
  lining  -- SKIN DEFINITION (candidate for the FLAT part). The flood scores skin
             as a 2 cm upper-hemisphere water LINING with vacuum behind it
             (geometry._dome CrewSkin); the kernel scores it as the outer 0.5 cm of
             a self-shielded 15 cm phantom. The variant replaces the phantom with a
             flood-style lining (RMin 13, RMax 15, DTheta 90, vacuum inside). Only
             the skin organ changes; the other organs are copied from the base.
             Prediction if the definition explains the floor: lift ~1.10 at both
             anchors.

A third candidate, the regolith FLOOR (albedo neutrons), is not probed: the flood's
measured neutron fraction at a 3 m dome is ~0.02-0.04, too small to carry 10%.

Both variants run at the 10 and 50 g/cm^2 anchors only, on the per-element
composition, and are compared against the PRE-TAIL kernel (identical node grid), so
no base re-run is needed. The lift is K_variant / K_pretail at an exact anchor, and
"after" = resid_after_skin / lift.

TARGET-BLIND: no dose value feeds the physics. Both variants are transport
diagnostics; the flood residual is only a post-hoc yardstick. CAL is untouched.

Usage
-----
  # 1) laptop or PC: emit (both variants, or --variants lining)
  python3 paper/geometry_residual_probe.py emit runs/geoprobe

  # 2) PC: run each variant batch
  export TOPAS_G4_DATA_DIR=~/G4Data
  bash runs/geoprobe/lining/run_all.sh
  bash runs/geoprobe/spot/run_all.sh

  # 3) fold + verdict (any variant that has finished)
  python3 paper/geometry_residual_probe.py report runs/geoprobe
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lunarsim.kernel_gen import config, generate, templates  # noqa: E402

_spec = importlib.util.spec_from_file_location("tail_vs_flood", HERE / "tail_vs_flood.py")
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)

MATERIAL = "aluminium"
COMPOSITION = "perelement"
ANCHORS_GCM2 = [10.0, 50.0]
LIS_FORM = "energy"          # same LIS form tail_vs_flood's resid_after_* used
RESID_CSV = HERE / "tail_vs_flood.csv"
OUT_CSV = HERE / "geometry_residual_probe.csv"
BASE = tv.DEFAULT_BASE       # pre-tail kernel: same nodes as a fresh perelement emit

# name -> template knobs. spot: 45 cm disc > 33.5 cm wall at 50 g/cm^2;
# histories 17 ~ 4*(45/22)^2 keeps the per-run fluence (and MC noise) ~unchanged.
VARIANTS = {
    "spot":   {"beam_spot_cm": 45.0, "histories": 17, "skin_mode": "phantom"},
    "lining": {"beam_spot_cm": None, "histories": None, "skin_mode": "lining"},
}


def variant_phi_ff(knobs: dict) -> float:
    return templates.phi_ff(knobs["beam_spot_cm"], knobs["histories"])


# ---------------------------------------------------------------------------- emit
def emit(outdir: Path, variants: list, seeds: int, threads: int, anchors: list) -> None:
    species = config.species_for(COMPOSITION)
    for v in variants:
        knobs = VARIANTS[v]
        vdir = outdir / v
        runs = vdir / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        manifest = {"material": MATERIAL, "composition": COMPOSITION,
                    "wall_geometry": "shell", "variant": v, **knobs,
                    "phi_ff_cm2": variant_phi_ff(knobs), "seeds": seeds,
                    "threads": threads, "grid_gcm2": anchors, "runs": []}
        for w in anchors:
            for sname, s in species.items():
                for j in range(len(s["nodes_pernuc_mev"])):
                    for seed in range(1, seeds + 1):
                        rn = generate._run_name(MATERIAL, w, sname, j, seed)
                        (runs / rn).mkdir(exist_ok=True)
                        (runs / rn / "run.txt").write_text(templates.build_param_file(
                            MATERIAL, w, sname, j, seed, threads, COMPOSITION, "shell",
                            **knobs))
                        manifest["runs"].append({"name": rn, "wall_gcm2": w,
                                                 "species": sname, "node": j,
                                                 "seed": seed})
        (vdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        generate._write_runner(vdir)
        print(f"[{v}] {len(manifest['runs'])} runs -> {vdir}  "
              f"(spot {knobs['beam_spot_cm'] or config.BEAM_SPOT_CM:g} cm, "
              f"hist {knobs['histories'] or config.HISTORIES}, "
              f"skin_mode {knobs['skin_mode']}, phi_ff {manifest['phi_ff_cm2']:.5f})")
        print(f"     run: bash {vdir}/run_all.sh")


# -------------------------------------------------------------------------- report
def splice(base: dict, points: list, skin_only: bool, tag: str) -> dict:
    """The base kernel with its anchors at points' wall_gcm2 replaced. skin_only
    keeps every non-skin organ's R/Rsem from the base (the lining run has no
    other organs)."""
    K = copy.deepcopy(base)
    by_wall = {round(p["wall_gcm2"], 6): i for i, p in enumerate(K["points"])}
    for p in points:
        i = by_wall.get(round(p["wall_gcm2"], 6))
        if i is None:
            raise ValueError(f"base kernel has no anchor at {p['wall_gcm2']} g/cm^2")
        bp = K["points"][i]
        if set(bp["species"]) != set(p["species"]):
            raise ValueError(f"species mismatch at {p['wall_gcm2']}: "
                             f"{sorted(bp['species'])} vs {sorted(p['species'])}")
        new = copy.deepcopy(p)
        for sname, sp in new["species"].items():
            bsp = bp["species"][sname]
            if list(bsp["nodes_pernuc_mev"]) != list(sp["nodes_pernuc_mev"]):
                raise ValueError(f"node grid mismatch for {sname} at {p['wall_gcm2']}")
            if skin_only:
                for organ in bsp["R"]:
                    if organ != "skin":
                        sp["R"][organ] = copy.deepcopy(bsp["R"][organ])
                        sp["Rsem"][organ] = copy.deepcopy(bsp["Rsem"][organ])
        K["points"][i] = new
    K["meta"] = {**K["meta"], "diagnostic_variant": tag,
                 "note": f"DIAGNOSTIC geometry_residual_probe variant '{tag}' -- "
                         "never ship."}
    return K


def _collect_seed(runs: Path, wall: float, seed: int, phi: float, organs: list) -> dict:
    """One-seed point (R only), the same arithmetic as generate._collect_point."""
    out = {}
    for sname, meta in config.species_for(COMPOSITION).items():
        n = len(meta["nodes_pernuc_mev"])
        R = {o: {"D": [0.0] * n, "I": [0.0] * n} for o in organs}
        for j in range(n):
            rn = generate._run_name(MATERIAL, wall, sname, j, seed)
            for o in organs:
                fd, fi = templates.scorer_csv_names(o)
                for q, fb in (("D", fd), ("I", fi)):
                    v = generate._read_scalar_csv(runs / rn / f"{fb}.csv")
                    R[o][q][j] = (v / phi) if v is not None else 0.0
        out[sname] = {**{k: meta[k] for k in ("z", "a", "abundance", "particle",
                                               "group", "nodes_pernuc_mev")},
                      "R": R,
                      "Rsem": {o: {q: [0.0] * n for q in R[o]} for o in R}}
    return {"wall_gcm2": wall, "seeds": 1, "species": out}


def missing_runs(vdir: Path, manifest: dict) -> int:
    organ = "skin"
    fd, _ = templates.scorer_csv_names(organ)
    return sum(not (vdir / "runs" / r["name"] / f"{fd}.csv").exists()
               for r in manifest["runs"])


def load_resid(path: Path = RESID_CSV) -> dict:
    out = {}
    if not path.exists():
        return out
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for q in ("skin", "phantom"):
                v = (r.get(f"resid_after_{q}") or "").strip()
                e = (r.get(f"resid_after_{q}_err") or "").strip()
                if v:
                    row[q] = (float(v), float(e) if e else 0.0)
            if row:
                out[round(float(r["areal_gcm2"]), 6)] = row
    return out


def fold_lift(K: dict, path: Path, gcm2: float) -> dict:
    path.write_text(json.dumps(K))
    return tv.lift_point(gcm2, path, LIS_FORM, base_path=BASE)


def report_variant(vdir: Path, base: dict, resid: dict) -> list:
    manifest = json.loads((vdir / "manifest.json").read_text())
    v = manifest["variant"]
    phi = manifest["phi_ff_cm2"]
    seeds = manifest["seeds"]
    skin_only = manifest["skin_mode"] == "lining"
    miss = missing_runs(vdir, manifest)
    if miss == len(manifest["runs"]):
        print(f"[{v}] not run yet (0/{miss} runs have output) -- skipped")
        return []
    if miss:
        print(f"[{v}] WARNING: {miss}/{len(manifest['runs'])} runs have no output "
              "-- missing runs read as zero response; finish the batch first")
    organs = ["skin"] if skin_only else [o for o, _ in config.organs_from_reference()]
    anchors = manifest["grid_gcm2"]
    runs = vdir / "runs"
    pts = [generate._collect_point(runs, MATERIAL, w, seeds, COMPOSITION, phi)
           for w in anchors]
    rows = []
    for w in anchors:
        full = fold_lift(splice(base, [p for p in pts if p["wall_gcm2"] == w],
                                skin_only, v), vdir / f"kernel_{v}.json", w)
        per_seed = []
        for s in range(1, seeds + 1):
            ps = _collect_seed(runs, w, s, phi, organs)
            per_seed.append(fold_lift(splice(base, [ps], skin_only, f"{v}_s{s}"),
                                      vdir / f"kernel_{v}_seed.json", w))
        row = {"variant": v, "areal_gcm2": w, "missing_runs": miss}
        for q in ("E", "phantom", "skin"):
            row[f"lift_{q}"] = full[f"lift_{q}"]
            vals = [r[f"lift_{q}"] for r in per_seed]
            row[f"lift_{q}_err"] = (statistics.stdev(vals) / math.sqrt(len(vals))
                                    if len(vals) > 1 else 0.0)
        rr = resid.get(round(w, 6), {})
        for q in ("skin", "phantom"):
            if q in rr:
                val, err = rr[q]
                lift = row[f"lift_{q}"]
                row[f"resid_{q}"] = val
                if lift <= 0.0:
                    continue
                row[f"after_{q}"] = val / lift
                row[f"after_{q}_err"] = (val / lift) * math.hypot(
                    err / val, row[f"lift_{q}_err"] / lift)
        rows.append(row)
    return rows


def verdict(rows: list) -> list:
    """Plain-language read-out per variant. Thresholds are 2-sigma on the variant's
    own seed spread (the pre-tail base adds comparable noise that is not included)."""
    msgs = []
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["areal_gcm2"]] = r
    sp = by.get("spot", {})
    if 10.0 in sp and 50.0 in sp:
        a, b = sp[10.0], sp[50.0]
        null_ok = abs(a["lift_skin"] - 1.0) <= max(2 * a["lift_skin_err"], 0.02)
        rise = b["lift_skin"] - 1.0
        sig = rise > 2 * max(b["lift_skin_err"], 1e-9)
        msgs.append(f"spot: null control @10 {'PASS' if null_ok else 'FAIL'} "
                    f"(lift {a['lift_skin']:.3f}); @50 lift {b['lift_skin']:.3f}"
                    f"{' (significant)' if sig else ' (not significant)'}")
        if "resid_skin" in a and "resid_skin" in b:
            want = b["resid_skin"] / a["resid_skin"] - 1.0
            got = b["lift_skin"] / a["lift_skin"] - 1.0
            frac = got / want if want else float("nan")
            msgs.append(f"spot: explains {100 * frac:.0f}% of the 10->50 skin rise "
                        f"({100 * want:.1f}% wanted, {100 * got:.1f}% gained)")
    ln = by.get("lining", {})
    if ln:
        lifts = ", ".join(f"@{g:g} {r['lift_skin']:.3f}+/-{r['lift_skin_err']:.3f}"
                          for g, r in sorted(ln.items()))
        afters = ", ".join(f"@{g:g} {r['after_skin']:.3f}" for g, r in sorted(ln.items())
                           if "after_skin" in r)
        msgs.append(f"lining: skin lift {lifts}; residual after lining {afters} "
                    "(~1.00 at 10 => the definition IS the flat floor)")
    if sp and ln:
        for g in sorted(set(sp) & set(ln)):
            if "resid_skin" in sp[g]:
                both = sp[g]["resid_skin"] / (sp[g]["lift_skin"] * ln[g]["lift_skin"])
                msgs.append(f"both (product, assumes independence): @{g:g} residual "
                            f"{sp[g]['resid_skin']:.3f} -> {both:.3f}")
    return msgs


def report(outdir: Path, resid_csv: Path = RESID_CSV, out_csv: Path = OUT_CSV) -> list:
    base = json.loads(BASE.read_text())
    resid = load_resid(resid_csv)
    if not resid:
        print(f"note: no resid_after_* rows in {resid_csv}; printing lifts only")
    rows = []
    for v in VARIANTS:
        vdir = outdir / v
        if (vdir / "manifest.json").exists():
            rows += report_variant(vdir, base, resid)
    if not rows:
        print(f"no variant manifests under {outdir}")
        return rows
    hdr = (f"{'variant':7s} {'g/cm2':>6s} {'lift_skin':>15s} {'lift_E':>7s} "
           f"{'resid_skin':>10s} {'after_skin':>15s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ls = f"{r['lift_skin']:.3f}+/-{r['lift_skin_err']:.3f}"
        af = (f"{r['after_skin']:.3f}+/-{r['after_skin_err']:.3f}"
              if "after_skin" in r else "-")
        rs = f"{r['resid_skin']:.3f}" if "resid_skin" in r else "-"
        print(f"{r['variant']:7s} {r['areal_gcm2']:6g} {ls:>15s} {r['lift_E']:7.3f} "
              f"{rs:>10s} {af:>15s}")
    print()
    for m in verdict(rows):
        print("  " + m)
    keys = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in ("variant", "areal_gcm2"), k))
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("emit")
    pe.add_argument("outdir", type=Path)
    pe.add_argument("--variants", default="lining,spot",
                    help="comma list from " + ",".join(VARIANTS))
    pe.add_argument("--seeds", type=int, default=config.SEEDS)
    pe.add_argument("--threads", type=int, default=1)
    pe.add_argument("--anchors", default=",".join(f"{a:g}" for a in ANCHORS_GCM2))
    pr = sub.add_parser("report")
    pr.add_argument("outdir", type=Path)
    pr.add_argument("--resid-csv", type=Path, default=RESID_CSV)
    pr.add_argument("--out-csv", type=Path, default=OUT_CSV)
    a = p.parse_args(argv)
    if a.cmd == "emit":
        vs = [v.strip() for v in a.variants.split(",") if v.strip()]
        bad = [v for v in vs if v not in VARIANTS]
        if bad:
            p.error(f"unknown variant(s) {bad}")
        anchors = [float(x) for x in a.anchors.split(",")]
        base_walls = {round(pt["wall_gcm2"], 6)
                      for pt in json.loads(BASE.read_text())["points"]}
        off = [x for x in anchors if round(x, 6) not in base_walls]
        if off:
            p.error(f"anchors {off} are not pre-tail kernel anchors {sorted(base_walls)}")
        emit(a.outdir, vs, a.seeds, a.threads, anchors)
    else:
        report(a.outdir, a.resid_csv, a.out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
