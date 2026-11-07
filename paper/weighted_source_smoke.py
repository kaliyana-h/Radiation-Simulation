#!/usr/bin/env python3
"""WeightedDiscBeam smoke test (step 2): does the position-biased source give the
SAME answer as the plain Beam source, and do the scorers honour the weights?

Three modes per design, same tier and rounds (no early stop):
  beam    -- production make_source block (TOPAS "Beam", flat disc)
  wdb0    -- WeightedDiscBeam with BiasFraction 0  (same sampling, new code path)
  biased  -- WeightedDiscBeam, SourceBias(fraction=0.5, margin_cm=40)

Checks (pure Python, no numpy -- runs on the PC venv):
  (a) wdb0 vs beam   : |z| < 2 on phantom AND skin E  -> the generator itself
                       (transform, energy, particle) matches the Beam source.
  (b) biased vs beam : |z| < 2 on phantom AND skin E  -> the estimator is unbiased.
  (c) weight sanity  : biased/beam outer+inner gauge fluence within [0.75, 1.33],
                       phantom+skin absorbed dose within [1/3, 3]. A scorer that IGNORES the
                       track weight shows up as a gross factor here (inner fluence
                       ~x5, phantom ~x50), not as a few-sigma shift.
  info: FOM = 1/(rel^2 * wall) ratio biased/beam for phantom and skin (step 3
        decides on FOM with paired FULL-tier runs; quick tier is indicative only).

TARGET-BLIND: the bias uses only the phantom's position and size. CAL untouched.
LETSpectrumWater ignores track weight: do not use it with biased runs.

Usage
-----
  python3 paper/weighted_source_smoke.py run --dry-run      # laptop: plan only
  # PC (after rebuilding TOPAS with the extension), in tmux:
  .venv/bin/python paper/weighted_source_smoke.py run | tee runs/wdb_smoke.log
  .venv/bin/python paper/weighted_source_smoke.py report
Rows are appended after every (design, mode); rerun with --modes to fill gaps.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CSV_DEFAULT = _REPO / "paper" / "weighted_source_smoke.csv"
MODES = ("beam", "wdb0", "biased")
FIELDS = ["design", "areal_gcm2", "mode", "bias_f", "bias_margin_cm", "tier",
          "rounds", "wall_s", "total_primaries",
          "phantom_msv", "phantom_rel", "skin_msv", "skin_rel",
          "phantom_mgy_d", "skin_mgy_d", "fluence_out", "fluence_in",
          "provenance", "utc"]
SANITY = (0.75, 1.33)       # gauges: high-count, tight band
GROSS = (1 / 3, 3.0)         # doses: phantom is noisy at quick tier; catch x5/x50 only


def designs() -> dict:
    from lunarsim import HabitatSpec, WallLayer
    from lunarsim.spec import MATERIALS
    rho_al = MATERIALS["aluminium"]["density"]
    return {"al19": HabitatSpec(name="wdb_al19", shape="dome", inner_radius_cm=300.0,
                                walls=[WallLayer("aluminium", 19.0 / rho_al)])}


def bias_for(mode: str):
    from lunarsim.bridge import SourceBias
    return {"beam": None, "wdb0": SourceBias(fraction=0.0),
            "biased": SourceBias(fraction=0.5, margin_cm=40.0)}[mode]


# ----------------------------------------------------------------------------
# PC run
# ----------------------------------------------------------------------------
def run_one(spec, tier, mode: str, rounds: int) -> dict:
    from lunarsim import dosimetry, jobs
    sb = bias_for(mode)

    def progress(done, total, rel):
        r = f"{rel:.3f}" if rel is not None else "-"
        print(f"    [{mode}] {done}/{total} batches  rel {r}", flush=True)

    t0 = time.time()
    comp = jobs.run_composition(spec, tier, target_rel_err=0.0, min_batches=rounds,
                                max_batches=rounds, converge_on="both",
                                sampling="continuous", source_bias=sb,
                                progress_cb=progress)
    if not comp.ok:
        raise RuntimeError(f"{spec.name} {mode}: run failed rc={comp.returncode}\n"
                           f"{comp.log_tail[-2000:]}")
    a_ph = dosimetry.assess_composition(comp.species_results, phi_MV=tier.phi_mv,
                                        skin=False, model=comp.model)
    a_sk = dosimetry.assess_composition(comp.species_results, phi_MV=tier.phi_mv,
                                        skin=True, model=comp.model)
    # raw gauges summed over species (same primaries per mode -> directly comparable)
    fout = sum(r.fluence_outside or 0.0 for _, r in comp.species_results)
    fin = sum(r.fluence_inside or 0.0 for _, r in comp.species_results)
    return {
        "design": "", "areal_gcm2": round(spec.areal_density_gcm2(), 3), "mode": mode,
        "bias_f": "" if sb is None else sb.fraction,
        "bias_margin_cm": "" if sb is None else sb.margin_cm, "tier": tier.name,
        "rounds": min(r.n_batches for _, r in comp.species_results),
        "wall_s": round(comp.wall_seconds or (time.time() - t0), 1),
        "total_primaries": comp.total_primaries,
        "phantom_msv": a_ph.annual_msv, "phantom_rel": a_ph.rel_err,
        "skin_msv": a_sk.annual_msv, "skin_rel": a_sk.rel_err,
        "phantom_mgy_d": a_ph.dose_rate_gy_s * 86400e3,
        "skin_mgy_d": a_sk.dose_rate_gy_s * 86400e3,
        "fluence_out": fout, "fluence_in": fin,
        "provenance": dosimetry.provenance_str(a_sk.provenance),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_run(args) -> None:
    from lunarsim.bridge import QUICK_LOOK, FULL_RUN
    tier = {"quick": QUICK_LOOK, "full": FULL_RUN}[args.tier]
    ds = designs()
    print(f"tier {tier.name} ({tier.total_primaries} primaries/run/species), "
          f"{args.rounds} rounds, modes {args.modes}, csv {args.csv}")
    for n, s in ds.items():
        print(f"  {n:8s} {s.shape:6s} {s.areal_density_gcm2():7.2f} g/cm2  "
              f"crew z {s.crew_height_cm:g} cm  phantom r {s.phantom_radius_cm:g} cm")
        for m in args.modes:
            b = bias_for(m)
            print(f"    {m:7s} bias={None if b is None else b.for_spec(s)}")
    if args.dry_run:
        return
    for n, s in ds.items():
        for m in args.modes:
            print(f"== {n} {m}", flush=True)
            row = run_one(s, tier, m, args.rounds)
            row["design"] = n
            append_row(args.csv, row)
            print(f"   phantom {row['phantom_msv']:.1f} mSv/yr +/- "
                  f"{100 * row['phantom_rel']:.1f}%  skin {row['skin_msv']:.1f} +/- "
                  f"{100 * row['skin_rel']:.1f}%  wall {row['wall_s']:.0f} s", flush=True)
    report(args.csv)


# ----------------------------------------------------------------------------
# report (laptop-safe)
# ----------------------------------------------------------------------------
def _z(m1, r1, m2, r2):
    s = math.hypot(m1 * r1, m2 * r2)
    return (m2 - m1) / s if s > 0 else float("nan")


def _fom(rel, wall):
    return 1.0 / (rel * rel * wall) if rel and wall else float("nan")


def _ratio(a, b):
    return a / b if b else float("nan")


def compare(ref: dict, alt: dict) -> dict:
    f = {k: float(ref[k]) for k in FIELDS[9:17]} | {"wall_s": float(ref["wall_s"])}
    g = {k: float(alt[k]) for k in f}
    z_ph = _z(f["phantom_msv"], f["phantom_rel"], g["phantom_msv"], g["phantom_rel"])
    z_sk = _z(f["skin_msv"], f["skin_rel"], g["skin_msv"], g["skin_rel"])
    ratios = {k: _ratio(g[k], f[k]) for k in
              ("fluence_out", "fluence_in", "phantom_mgy_d", "skin_mgy_d")}
    sane = (all(SANITY[0] <= ratios[k] <= SANITY[1] for k in ("fluence_out", "fluence_in"))
            and all(GROSS[0] <= ratios[k] <= GROSS[1] for k in ("phantom_mgy_d", "skin_mgy_d")))
    return {"z_phantom": z_ph, "z_skin": z_sk, "ratios": ratios,
            "agree": abs(z_ph) < 2.0 and abs(z_sk) < 2.0, "sane": sane,
            "fom_phantom": _fom(g["phantom_rel"], g["wall_s"])
            / _fom(f["phantom_rel"], f["wall_s"]),
            "fom_skin": _fom(g["skin_rel"], g["wall_s"]) / _fom(f["skin_rel"], f["wall_s"])}


def report(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    rows = list(csv.DictReader(path.open()))
    latest = {(r["design"], r["mode"]): r for r in rows}     # last row wins
    out = []
    for d in dict.fromkeys(r["design"] for r in rows):
        ref = latest.get((d, "beam"))
        print(f"\n== {d}")
        for m in MODES:
            r = latest.get((d, m))
            if r:
                print(f"  {m:7s} phantom {float(r['phantom_msv']):7.1f} +/- "
                      f"{100 * float(r['phantom_rel']):4.1f}%   skin "
                      f"{float(r['skin_msv']):7.1f} +/- {100 * float(r['skin_rel']):4.1f}%"
                      f"   wall {float(r['wall_s']):6.0f} s")
        if not ref:
            print("  no 'beam' reference row yet")
            continue
        for m, label in (("wdb0", "(a) wdb0 vs beam"), ("biased", "(b,c) biased vs beam")):
            alt = latest.get((d, m))
            if not alt:
                print(f"  {label}: missing")
                continue
            v = compare(ref, alt)
            v.update(design=d, mode=m)
            out.append(v)
            rs = "  ".join(f"{k} {x:.3f}" for k, x in v["ratios"].items())
            print(f"  {label}: z_phantom {v['z_phantom']:+.2f}  z_skin {v['z_skin']:+.2f}"
                  f"  -> {'AGREE' if v['agree'] else 'DISAGREE'}")
            print(f"      ratios {rs}  -> {'sane' if v['sane'] else 'WEIGHT-HANDLING SUSPECT'}")
            print(f"      FOM x  phantom {v['fom_phantom']:.2f}  skin {v['fom_skin']:.2f}"
                  f"  (quick tier: indicative only)")
            if ref["provenance"].split(" engine")[0] != alt["provenance"].split(" engine")[0]:
                print("      WARNING: provenance differs between modes")
    if out:
        have = {v["mode"] for v in out}
        ok = all(v["agree"] and v["sane"] for v in out)
        verdict = ("INCOMPLETE (missing " + ", ".join(sorted({"wdb0", "biased"} - have)) + ")"
                   if have != {"wdb0", "biased"} else ("PASS" if ok else "FAIL"))
        print(f"\nSMOKE {verdict}: (a) generator matches Beam, (b) biased "
              f"estimator agrees, (c) no gross weight factor. Step 3 (paired FOM) next "
              f"only on PASS.")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="PC: three-mode smoke runs (TOPAS)")
    r.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES))
    r.add_argument("--rounds", type=int, default=6)
    r.add_argument("--tier", choices=("quick", "full"), default="quick")
    r.add_argument("--dry-run", action="store_true", help="print the plan only")
    for q in (r, sub.add_parser("report", help="compare modes and print verdicts")):
        q.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    args = p.parse_args(argv)
    {"run": cmd_run, "report": lambda a: report(a.csv)}[args.cmd](args)


if __name__ == "__main__":
    main()
