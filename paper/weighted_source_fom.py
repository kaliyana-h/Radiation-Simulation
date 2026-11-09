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

Errors: the annual-E z-scores, FOM and speedup use the DOSE-EQUIVALENT batch
error (DoseAssessment.eq_rel_err; per-species SEM of the per-round H, combined in
quadrature). The phantom's Q-bar scatters with rare high-LET tracks, so the
absorbed-dose error (rel_err) understates E's noise; z_D (absorbed, rel_err) is
reported alongside as the transport-only check. Rows written before eq_rel_err
existed fall back to rel_err and are flagged "absorbed-err only".

Seeds: round k uses seed k+1+seed_offset. With the default offset 0 every mode
(and every earlier full-tier continuous run of the same design) replays the same
seeds; --seed-offset N (e.g. 1000) draws an independent sample. The modes sample
differently even on equal seeds, so pairs are treated as independent. Rows are
grouped by seed_offset ("sample"); when a design has >1 sample, the samples are
also pooled per mode (inverse-variance mean, summed wall) and compared.

TARGET-BLIND (bias uses only the phantom's position + size); CAL untouched;
production default stays the plain Beam source -- flipping it is the user's call.
LETSpectrumWater ignores track weight: not used here.

Usage
-----
  python3 paper/weighted_source_fom.py run --dry-run
  # PC, in tmux (al19 first; ~3 h per mode at full tier):
  .venv/bin/python paper/weighted_source_fom.py run --designs al19 \\
      --seed-offset 1000 | tee runs/wdb_fom_al19_s1000.log
  .venv/bin/python paper/weighted_source_fom.py report
Rows are appended after every (design, mode); rerun with --designs/--fractions
to fill gaps. Last row per (design, seed_offset, mode) wins.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
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
          "provenance", "utc", "phantom_eq_rel", "skin_eq_rel", "seed_offset"]


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
def run_one(spec, tier, mode: str, rounds: int, margin_cm: float,
            seed_offset: int = 0) -> dict:
    from lunarsim import dosimetry, jobs
    sb = bias_for(mode, margin_cm)

    def progress(done, total, rel):
        r = f"{rel:.3f}" if rel is not None else "-"
        print(f"    [{mode}] {done}/{total} batches  rel {r}", flush=True)

    t0 = time.time()
    comp = jobs.run_composition(spec, tier, target_rel_err=0.0, min_batches=rounds,
                                max_batches=rounds, converge_on="both",
                                sampling="continuous", source_bias=sb,
                                seed_offset=seed_offset, progress_cb=progress)
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
        "phantom_eq_rel": "" if a_ph.eq_rel_err is None else a_ph.eq_rel_err,
        "skin_eq_rel": "" if a_sk.eq_rel_err is None else a_sk.eq_rel_err,
        "seed_offset": seed_offset,
    }


def _upgrade_header(path: Path) -> None:
    """A CSV written before the eq-error columns existed gets the new header
    (old rows keep blank new fields). The original is copied aside first."""
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames == FIELDS:
            return
        old = list(rd)
    shutil.copy2(path, path.with_suffix(path.suffix + ".pre_eqerr.bak"))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in old:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    if not new:
        _upgrade_header(path)
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
        # 0 = WeightedDiscBeam with no bias (unit weights): the beam's sampling
        # through the extension's code path, to separate generator from tail.
        if not 0.0 <= f < 1.0:
            raise SystemExit(f"bias fraction must be in [0, 1), got {f}")
    modes = ([] if args.skip_beam else ["beam"]) + [mode_name(f) for f in args.fractions]
    print(f"tier {tier.name} ({tier.total_primaries} primaries/run/species), "
          f"{args.rounds} rounds, modes {modes}, margin {args.margin:g} cm, "
          f"seed offset {args.seed_offset}, csv {args.csv}")
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
            row = run_one(ds[n], tier, m, args.rounds, args.margin, args.seed_offset)
            row["design"] = n
            append_row(args.csv, row)
            print(f"   phantom {row['phantom_msv']:.1f} mSv/yr +/- "
                  f"{_pct(row['phantom_eq_rel'])} (E) {100 * row['phantom_rel']:.1f}% (D)  "
                  f"skin {row['skin_msv']:.1f} +/- {_pct(row['skin_eq_rel'])} (E) "
                  f"{100 * row['skin_rel']:.1f}% (D)  wall {row['wall_s']:.0f} s", flush=True)
    report(args.csv, args.min_gain)


# ----------------------------------------------------------------------------
# report (laptop-safe)
# ----------------------------------------------------------------------------
def _z(m1, r1, m2, r2):
    s = math.hypot(m1 * r1, m2 * r2)
    return (m2 - m1) / s if s > 0 else float("nan")


def _fom(rel, wall):
    return 1.0 / (rel * rel * wall) if rel and wall else float("nan")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pct(v):
    x = _num(v)
    return "  n/a" if x is None else f"{100 * x:4.1f}%"


def _norm(r: dict) -> dict:
    """Floats for one CSV row. e_ph/e_sk = the E errors (eq_rel if recorded, else
    the absorbed rel_err with eq_ok False)."""
    keys = ("phantom_msv", "phantom_rel", "skin_msv", "skin_rel", "wall_s",
            "phantom_mgy_d", "skin_mgy_d", "fluence_out", "fluence_in")
    out = {k: float(r[k]) for k in keys}
    eph, esk = _num(r.get("phantom_eq_rel")), _num(r.get("skin_eq_rel"))
    out["eq_ok"] = eph is not None and esk is not None
    out["e_ph"] = eph if eph is not None else out["phantom_rel"]
    out["e_sk"] = esk if esk is not None else out["skin_rel"]
    return out


def pool(rows: list) -> dict:
    """Inverse-variance pool of independent samples of one (design, mode); wall
    and fluence gauges summed / averaged so FOM stays per unit wall."""
    rs = [r if "e_ph" in r else _norm(r) for r in rows]
    out = {"wall_s": sum(r["wall_s"] for r in rs),
           "fluence_out": sum(r["fluence_out"] for r in rs) / len(rs),
           "fluence_in": sum(r["fluence_in"] for r in rs) / len(rs),
           "eq_ok": all(r["eq_ok"] for r in rs)}
    for val, rel, dst in (("phantom_msv", "e_ph", "e_ph"), ("skin_msv", "e_sk", "e_sk"),
                          ("phantom_mgy_d", "phantom_rel", "phantom_rel"),
                          ("skin_mgy_d", "skin_rel", "skin_rel")):
        w = [1.0 / (r[val] * r[rel]) ** 2 for r in rs]
        m = sum(wi * r[val] for wi, r in zip(w, rs)) / sum(w)
        out[val] = m
        out[dst] = (1.0 / sum(w)) ** 0.5 / m
    return out


def compare(ref: dict, alt: dict, min_gain: float) -> dict:
    f = ref if "e_ph" in ref else _norm(ref)
    g = alt if "e_ph" in alt else _norm(alt)
    z_ph = _z(f["phantom_msv"], f["e_ph"], g["phantom_msv"], g["e_ph"])
    z_sk = _z(f["skin_msv"], f["e_sk"], g["skin_msv"], g["e_sk"])
    z_ph_abs = _z(f["phantom_mgy_d"], f["phantom_rel"], g["phantom_mgy_d"], g["phantom_rel"])
    gain_ph = _fom(g["e_ph"], g["wall_s"]) / _fom(f["e_ph"], f["wall_s"])
    gain_sk = _fom(g["e_sk"], g["wall_s"]) / _fom(f["e_sk"], f["wall_s"])
    t_ref = f["wall_s"] * max(f["e_ph"], f["e_sk"]) ** 2
    t_alt = g["wall_s"] * max(g["e_ph"], g["e_sk"]) ** 2
    agree = abs(z_ph) < 2.0 and abs(z_sk) < 2.0
    better = gain_ph >= min_gain
    return {"z_phantom": z_ph, "z_skin": z_sk, "z_phantom_abs": z_ph_abs,
            "fom_gain_phantom": gain_ph, "fom_gain_skin": gain_sk,
            "speedup_both": t_ref / t_alt if t_alt else float("nan"),
            "gauge_out": g["fluence_out"] / f["fluence_out"] if f["fluence_out"] else float("nan"),
            "gauge_in": g["fluence_in"] / f["fluence_in"] if f["fluence_in"] else float("nan"),
            "eq_errors": f["eq_ok"] and g["eq_ok"],
            "agree": agree, "better": better, "pass": agree and better}


def _print_cmp(label: str, v: dict, min_gain: float) -> None:
    verdict = "PASS" if v["pass"] else ("FAIL-disagree" if not v["agree"] else "no gain")
    flag = "" if v["eq_errors"] else "  [absorbed-err only: z_E/FOM understate E noise]"
    print(f"  {label}: z_E phantom {v['z_phantom']:+.2f} skin {v['z_skin']:+.2f}"
          f" (z_D phantom {v['z_phantom_abs']:+.2f})  gauges out "
          f"{v['gauge_out']:.3f} in {v['gauge_in']:.3f}{flag}")
    print(f"      FOM(E) x phantom {v['fom_gain_phantom']:.2f}  skin "
          f"{v['fom_gain_skin']:.2f}  | both-gate speedup "
          f"{v['speedup_both']:.2f}x  -> {verdict}")


DEFAULT_MARGIN_CM = 40.0


def _label(r: dict) -> str:
    """Mode name for grouping; a non-default shadow margin gets its own label
    (f0.2m120) so a margin sweep never pools with the default-margin rows."""
    m, mg = r["mode"], _num(r.get("bias_margin_cm"))
    if m == "beam" or mg is None or abs(mg - DEFAULT_MARGIN_CM) < 1e-9:
        return m
    return f"{m}m{mg:g}"


def weighted_consistency(rows: list, ref: str = "f0.2") -> list:
    """Beam-free unbiasedness check: every biased (fraction, margin) estimates the
    same expectation if the weights are right, so pool each label over all samples
    and z-test it against the reference label. Absorbed dose (D) is the clean
    test (its error is always recorded); E uses eq errors where present."""
    by = {}
    for r in rows:
        lab = _label(r)
        if lab != "beam":
            by.setdefault(lab, []).append(_norm(r))
    if ref not in by or len(by) < 2:
        return []
    pr = pool(by[ref])
    out = []
    print(f"  -- weighted-only consistency (each label pooled, z vs {ref})")
    for lab in sorted(by):
        pm = pool(by[lab])
        flag = "" if pm["eq_ok"] else " [E err absorbed-only]"
        line = (f"  {lab:9s} phantom E {pm['phantom_msv']:7.1f} +/- {100 * pm['e_ph']:4.1f}%"
                f"  D {pm['phantom_mgy_d']:.4g} +/- {100 * pm['phantom_rel']:4.1f}%"
                f"  (n={len(by[lab])}){flag}")
        if lab != ref:
            zE = _z(pr["phantom_msv"], pr["e_ph"], pm["phantom_msv"], pm["e_ph"])
            zD = _z(pr["phantom_mgy_d"], pr["phantom_rel"],
                    pm["phantom_mgy_d"], pm["phantom_rel"])
            line += f"   z_E {zE:+.2f}  z_D {zD:+.2f}"
            out.append({"label": lab, "z_E": zE, "z_D": zD,
                        "agree": abs(zE) < 2 and abs(zD) < 2})
        print(line)
    return out


def _sample(r: dict) -> int:
    x = _num(r.get("seed_offset"))
    return 0 if x is None else int(x)


def report(path: Path, min_gain: float = 2.0) -> list:
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    rows = list(csv.DictReader(path.open()))
    latest = {(r["design"], _sample(r), _label(r)): r for r in rows}
    out = []
    for d in dict.fromkeys(r["design"] for r in rows):
        samples = sorted({s for (dd, s, _) in latest if dd == d})
        a0 = next(r for (dd, _, _), r in latest.items() if dd == d)
        print(f"\n== {d}  ({float(a0['areal_gcm2']):.1f} g/cm2)   "
              f"E err = dose-equivalent batch SEM where recorded, (D) = absorbed")
        for smp in samples:
            modes = [m for (dd, s, m) in latest if dd == d and s == smp]
            print(f"  -- sample seed_offset={smp}")
            for m in ["beam"] + sorted(x for x in modes if x != "beam"):
                r = latest.get((d, smp, m))
                if r:
                    print(f"  {m:6s} phantom {float(r['phantom_msv']):7.1f} +/- "
                          f"{_pct(r.get('phantom_eq_rel'))} (D {_pct(r['phantom_rel'])})"
                          f"   skin {float(r['skin_msv']):7.1f} +/- "
                          f"{_pct(r.get('skin_eq_rel'))} (D {_pct(r['skin_rel'])})"
                          f"   wall {float(r['wall_s']):6.0f} s")
            ref = latest.get((d, smp, "beam"))
            if not ref:
                print("  no 'beam' reference row in this sample")
                continue
            for m in sorted(x for x in modes if x != "beam"):
                alt = latest[(d, smp, m)]
                v = compare(ref, alt, min_gain)
                v.update(design=d, mode=m, sample=smp)
                out.append(v)
                _print_cmp(f"{m} vs beam", v, min_gain)
                if ref["provenance"].split(" engine")[0] != alt["provenance"].split(" engine")[0]:
                    print("      WARNING: provenance differs between modes")
        # Pool only samples with real E errors: an absorbed-err-only row would get
        # an inflated inverse-variance weight and dominate the pool.
        by_mode = {}
        for (dd, s, m), r in latest.items():
            if dd == d and _norm(r)["eq_ok"]:
                by_mode.setdefault(m, []).append(r)
        if len(samples) > 1 and len(by_mode.get("beam", [])) < 2:
            print("  (no pooling: fewer than 2 samples with dose-equivalent errors)")
        if len(by_mode.get("beam", [])) >= 2:
            pref = pool(by_mode["beam"])
            print("  -- pooled over samples with E errors")
            for m in ["beam"] + sorted(x for x in by_mode if x != "beam"):
                pm = pool(by_mode[m])
                print(f"  {m:6s} phantom {pm['phantom_msv']:7.1f} +/- "
                      f"{100 * pm['e_ph']:4.1f}%   skin {pm['skin_msv']:7.1f} +/- "
                      f"{100 * pm['e_sk']:4.1f}%   wall {pm['wall_s']:6.0f} s "
                      f"(n={len(by_mode[m])})")
            for m in sorted(x for x in by_mode if x != "beam"):
                v = compare(pref, pool(by_mode[m]), min_gain)
                v.update(design=d, mode=m, sample="pooled")
                out.append(v)
                _print_cmp(f"{m} vs beam (pooled)", v, min_gain)
        weighted_consistency([r for (dd, _, _), r in latest.items() if dd == d])
    if out:
        n = sum(v["pass"] for v in out)
        print(f"\n{n}/{len(out)} comparisons PASS (|z|<2 phantom+skin E, phantom FOM "
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
    r.add_argument("--seed-offset", type=int, default=0,
                   help="round k uses seed k+1+offset; non-zero = independent sample")
    r.add_argument("--dry-run", action="store_true", help="print the plan only")
    for q in (r, sub.add_parser("report", help="pair rows and print verdicts")):
        q.add_argument("--csv", type=Path, default=CSV_DEFAULT)
        q.add_argument("--min-gain", type=float, default=2.0)
    args = p.parse_args(argv)
    {"run": cmd_run, "report": lambda a: report(a.csv, a.min_gain)}[args.cmd](args)


if __name__ == "__main__":
    main()
