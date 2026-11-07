#!/usr/bin/env python3
"""Position-biased GCR source, step 3: paired figure-of-merit comparison.

Question: at matched walltime, does the WeightedDiscBeam source
(jobs.run_composition(source_bias=SourceBias(f, margin))) cut the phantom's
statistical error without moving the answer, and what does it cost the skin?

Each design runs the plain Beam source and one or more bias fractions, all with
the SAME tier and rounds (target_rel_err=0, min=max=rounds, so none stops early).
  FOM      = 1 / (rel^2 * wall)        per scored quantity (phantom, skin)
  speedup  = T_beam / T_mode, with T = wall * max(rel_phantom, rel_skin)^2
             = the wall-time ratio to reach the SAME converge_on="both" target
             (phantom_slack 1), i.e. what the production gate would actually save.

Pass, per (design, f):
  agree   : |z| < 2 on phantom AND skin annual E vs beam
  better  : phantom FOM gain >= --min-gain (default 2)
The speedup is reported alongside: a phantom gain that the skin pays back in full
(speedup <= 1) is not worth flipping a default for.

Seeds repeat per round index, so every mode replays the same seeds. The modes
still sample differently, so they are NOT correlated pairs: the z-scores treat
them as independent.

TARGET-BLIND (bias uses only the phantom's position + size); CAL untouched;
production default stays the plain Beam source -- flipping it is the user's call.
LETSpectrumWater ignores track weight: not used here.

Usage
-----
  python3 paper/weighted_source_fom.py run --dry-run
  # PC, in tmux (al19 first; ~3 h per mode at full tier):
  .venv/bin/python paper/weighted_source_fom.py run --designs al19 \\
      | tee runs/wdb_fom_al19.log
  .venv/bin/python paper/weighted_source_fom.py report
Rows are appended after every (design, mode); rerun with --designs/--fractions
to fill gaps. Last row per (design, mode) wins.
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

CSV_DEFAULT = _REPO / "paper" / "weighted_source_fom.csv"
FIELDS = ["design", "areal_gcm2", "mode", "bias_f", "bias_margin_cm", "tier",
          "rounds", "wall_s", "total_primaries",
          "phantom_msv", "phantom_rel", "skin_msv", "skin_rel",
          "phantom_mgy_d", "skin_mgy_d", "fluence_out", "fluence_in",
          "provenance", "utc"]


def designs() -> dict:
    """Same design set as the stratified comparison (paper/stratified_flood_compare.py)."""
    from lunarsim import HabitatSpec, WallLayer
    from lunarsim.spec import MATERIALS
    rho_al = MATERIALS["aluminium"]["density"]
    rho_reg = MATERIALS["regolith"]["density"]
    out = {}
    for g in (19.0, 54.0):
        out[f"al{g:g}"] = HabitatSpec(name=f"wdb_al{g:g}", shape="dome",
                                      inner_radius_cm=300.0,
                                      walls=[WallLayer("aluminium", g / rho_al)])
    ceiling = 1.0                                    # cm Al = 2.7 g/cm2
    depth = (150.0 - rho_al * ceiling) / rho_reg
    out["buried150"] = HabitatSpec(name="wdb_buried150", shape="buried",
                                   inner_radius_cm=300.0,
                                   walls=[WallLayer("aluminium", ceiling)],
                                   burial_depth_cm=depth)
    return out


def mode_name(f: float) -> str:
    return f"f{f:g}"


def bias_for(mode: str, margin_cm: float):
    from lunarsim.bridge import SourceBias
    if mode == "beam":
        return None
    return SourceBias(fraction=float(mode[1:]), margin_cm=margin_cm)


# ----------------------------------------------------------------------------
# PC run
# ----------------------------------------------------------------------------
def run_one(spec, tier, mode: str, rounds: int, margin_cm: float) -> dict:
    from lunarsim import dosimetry, jobs
    sb = bias_for(mode, margin_cm)

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
        "fluence_out": sum(r.fluence_outside or 0.0 for _, r in comp.species_results),
        "fluence_in": sum(r.fluence_inside or 0.0 for _, r in comp.species_results),
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
    names = args.designs or ["al19"]
    for n in names:
        if n not in ds:
            raise SystemExit(f"unknown design {n!r}; choose from {list(ds)}")
    for f in args.fractions:
        if not 0.0 < f < 1.0:
            raise SystemExit(f"bias fraction must be in (0, 1), got {f}")
    modes = ([] if args.skip_beam else ["beam"]) + [mode_name(f) for f in args.fractions]
    print(f"tier {tier.name} ({tier.total_primaries} primaries/run/species), "
          f"{args.rounds} rounds, modes {modes}, margin {args.margin:g} cm, csv {args.csv}")
    for n in names:
        s = ds[n]
        print(f"  {n:10s} {s.shape:6s} {s.areal_density_gcm2():7.2f} g/cm2")
        for m in modes:
            b = bias_for(m, args.margin)
            print(f"    {m:6s} bias={None if b is None else b.for_spec(s)}")
    if args.dry_run:
        return
    for n in names:
        for m in modes:
            print(f"== {n} {m}", flush=True)
            row = run_one(ds[n], tier, m, args.rounds, args.margin)
            row["design"] = n
            append_row(args.csv, row)
            print(f"   phantom {row['phantom_msv']:.1f} mSv/yr +/- "
                  f"{100 * row['phantom_rel']:.1f}%  skin {row['skin_msv']:.1f} +/- "
                  f"{100 * row['skin_rel']:.1f}%  wall {row['wall_s']:.0f} s", flush=True)
    report(args.csv, args.min_gain)


# ----------------------------------------------------------------------------
# report (laptop-safe)
# ----------------------------------------------------------------------------
def _z(m1, r1, m2, r2):
    s = math.hypot(m1 * r1, m2 * r2)
    return (m2 - m1) / s if s > 0 else float("nan")


def _fom(rel, wall):
    return 1.0 / (rel * rel * wall) if rel and wall else float("nan")


def compare(ref: dict, alt: dict, min_gain: float) -> dict:
    keys = ("phantom_msv", "phantom_rel", "skin_msv", "skin_rel", "wall_s",
            "phantom_mgy_d", "skin_mgy_d", "fluence_out", "fluence_in")
    f = {k: float(ref[k]) for k in keys}
    g = {k: float(alt[k]) for k in keys}
    z_ph = _z(f["phantom_msv"], f["phantom_rel"], g["phantom_msv"], g["phantom_rel"])
    z_sk = _z(f["skin_msv"], f["skin_rel"], g["skin_msv"], g["skin_rel"])
    z_ph_abs = _z(f["phantom_mgy_d"], f["phantom_rel"], g["phantom_mgy_d"], g["phantom_rel"])
    gain_ph = _fom(g["phantom_rel"], g["wall_s"]) / _fom(f["phantom_rel"], f["wall_s"])
    gain_sk = _fom(g["skin_rel"], g["wall_s"]) / _fom(f["skin_rel"], f["wall_s"])
    t_ref = f["wall_s"] * max(f["phantom_rel"], f["skin_rel"]) ** 2
    t_alt = g["wall_s"] * max(g["phantom_rel"], g["skin_rel"]) ** 2
    agree = abs(z_ph) < 2.0 and abs(z_sk) < 2.0
    better = gain_ph >= min_gain
    return {"z_phantom": z_ph, "z_skin": z_sk, "z_phantom_abs": z_ph_abs,
            "fom_gain_phantom": gain_ph, "fom_gain_skin": gain_sk,
            "speedup_both": t_ref / t_alt if t_alt else float("nan"),
            "gauge_out": g["fluence_out"] / f["fluence_out"] if f["fluence_out"] else float("nan"),
            "gauge_in": g["fluence_in"] / f["fluence_in"] if f["fluence_in"] else float("nan"),
            "agree": agree, "better": better, "pass": agree and better}


def report(path: Path, min_gain: float = 2.0) -> list:
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    rows = list(csv.DictReader(path.open()))
    latest = {(r["design"], r["mode"]): r for r in rows}
    out = []
    for d in dict.fromkeys(r["design"] for r in rows):
        modes = [m for (dd, m) in latest if dd == d]
        ref = latest.get((d, "beam"))
        print(f"\n== {d}  ({float(rows[0]['areal_gcm2']) if not ref else float(ref['areal_gcm2']):.1f} g/cm2)")
        for m in ["beam"] + sorted(x for x in modes if x != "beam"):
            r = latest.get((d, m))
            if r:
                print(f"  {m:6s} phantom {float(r['phantom_msv']):7.1f} +/- "
                      f"{100 * float(r['phantom_rel']):4.1f}%   skin "
                      f"{float(r['skin_msv']):7.1f} +/- {100 * float(r['skin_rel']):4.1f}%"
                      f"   wall {float(r['wall_s']):6.0f} s")
        if not ref:
            print("  no 'beam' reference row yet")
            continue
        for m in sorted(x for x in modes if x != "beam"):
            v = compare(ref, latest[(d, m)], min_gain)
            v.update(design=d, mode=m)
            out.append(v)
            verdict = "PASS" if v["pass"] else ("FAIL-disagree" if not v["agree"]
                                                else "no gain")
            print(f"  {m} vs beam: z_E phantom {v['z_phantom']:+.2f} skin {v['z_skin']:+.2f}"
                  f" (z_D phantom {v['z_phantom_abs']:+.2f})  gauges out "
                  f"{v['gauge_out']:.3f} in {v['gauge_in']:.3f}")
            print(f"      FOM x phantom {v['fom_gain_phantom']:.2f}  skin "
                  f"{v['fom_gain_skin']:.2f}  | both-gate speedup "
                  f"{v['speedup_both']:.2f}x  -> {verdict}")
            if ref["provenance"].split(" engine")[0] != latest[(d, m)]["provenance"].split(" engine")[0]:
                print("      WARNING: provenance differs between modes")
    if out:
        n = sum(v["pass"] for v in out)
        print(f"\n{n}/{len(out)} (design, f) PASS (|z|<2 phantom+skin E, phantom FOM "
              f">= {min_gain:g}). Check the both-gate speedup before any default flip; "
              f"that is the user's decision.")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="PC: paired runs (TOPAS)")
    r.add_argument("--designs", nargs="*", help="subset of al19 al54 buried150 (default al19)")
    r.add_argument("--fractions", nargs="*", type=float, default=[0.5, 0.2])
    r.add_argument("--margin", type=float, default=40.0, help="shadow margin, cm")
    r.add_argument("--rounds", type=int, default=12)
    r.add_argument("--tier", choices=("quick", "full"), default="full")
    r.add_argument("--skip-beam", action="store_true",
                   help="reuse an existing beam row for this design")
    r.add_argument("--dry-run", action="store_true", help="print the plan only")
    for q in (r, sub.add_parser("report", help="pair rows and print verdicts")):
        q.add_argument("--csv", type=Path, default=CSV_DEFAULT)
        q.add_argument("--min-gain", type=float, default=2.0)
    args = p.parse_args(argv)
    {"run": cmd_run, "report": lambda a: report(a.csv, a.min_gain)}[args.cmd](args)


if __name__ == "__main__":
    main()
