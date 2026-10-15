#!/usr/bin/env python3
"""Per-element (10-ion) thin-wall kernel acceptance gate.

Runs the PERELEM_RUNBOOK acceptance test directly: folds the shipped 5-ion
`corrected` aluminium kernel and the new 10-ion `perelement` kernel at EVERY
anchor (no interpolation, no MC) and reports, per anchor:

  - organ-weighted absorbed dose rate  D_eff  (-> uGy/day)
  - organ-weighted ICRP effective dose rate  E  (-> uSv/day)
  - mean quality factor  <Q> = E / D_eff
  - the baseline->perelement ratio for each

Gate (target-blind, from the runbook):
  PASS  absorbed ~unchanged (|ratio-1| <= 0.15 at every anchor)
  PASS  dose-eq E LOWER at the thin anchors (ratio < 1)
  PASS  <Q> FALLS at thin (never rises)
  REJECT  absorbed moves >15%, any NaN, or <Q> rises

Reads the new kernel from an explicit path so you can gate BEFORE copying it
into lunarsim/data/ (so a bad batch never ships):

  python3 -m paper.perelem_gate runs/perelem/gcr_thinwall_kernel_perelem.json

With no argument it uses lunarsim/data/gcr_thinwall_kernel_perelem.json.
Run from the repo root (~/topas). No TOPAS / G4Data needed -- pure arithmetic.
"""
import json
import math
import sys
from pathlib import Path

from lunarsim import dosimetry as D

PHI_MV = 400.0
SPD = D.SECONDS_PER_DAY


def fold_anchors(kernel: dict, model: str) -> list:
    """Fold every anchor of `kernel` under per-element `model`, returning a list
    of (wall_gcm2, D_eff_gy_s, E_sv_s, Qbar) sorted by areal density."""
    D.set_lis_form(D.KERNEL_LIS_FORM)
    D.set_perelement_model(model)
    organs = kernel["meta"]["organs"]
    out = []
    for p in sorted(kernel["points"], key=lambda q: q["wall_gcm2"]):
        f = D._thinwall_fold_point(p, PHI_MV, organs)
        D_eff = sum(wT * f["HD"][k] for k, wT in organs)
        E = sum(wT * f["HI"][k] for k, wT in organs)
        Qbar = E / D_eff if D_eff else float("nan")
        out.append((p["wall_gcm2"], D_eff, E, Qbar))
    return out


def main() -> int:
    pe_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path("lunarsim/data/gcr_thinwall_kernel_perelem.json"))
    if not pe_path.exists():
        print(f"per-element kernel not found: {pe_path}", file=sys.stderr)
        return 2

    base = D._load_gcr_thinwall_kernel("aluminium")          # shipped 5-ion corrected
    with open(pe_path) as fh:
        pe = json.load(fh)

    b = fold_anchors(base, "none")
    p = fold_anchors(pe, "perelement")
    D.set_perelement_model("none")                           # leave process clean

    # anchors should match; align on wall_gcm2
    bmap = {round(w, 4): (Dg, E, Q) for w, Dg, E, Q in b}
    print(f"baseline  : shipped 5-ion corrected  ({len(b)} anchors)")
    print(f"perelement: {pe_path}  ({len(p)} anchors, "
          f"{len(pe['points'][0]['species'])} species)\n")

    hdr = ("wall", "D_abs base", "D_abs PE", "Dr", "E base", "E PE", "Er",
           "Q base", "Q PE")
    print(f"{hdr[0]:>6} | {hdr[1]:>11} {hdr[2]:>11} {hdr[3]:>6} | "
          f"{hdr[4]:>11} {hdr[5]:>11} {hdr[6]:>6} | {hdr[7]:>7} {hdr[8]:>7}")
    print(f"{'g/cm2':>6} | {'uGy/day':>11} {'uGy/day':>11} {'':>6} | "
          f"{'uSv/day':>11} {'uSv/day':>11} {'':>6} | {'':>7} {'':>7}")
    print("-" * 94)

    worst_Dr = 0.0
    q_rose = False
    e_rose_thin = False
    for w, Dg, E, Q in p:
        key = round(w, 4)
        if key not in bmap:
            print(f"{w:6.2f} | (no matching baseline anchor)")
            continue
        Db, Eb, Qb = bmap[key]
        Dr = Dg / Db if Db else float("nan")
        Er = E / Eb if Eb else float("nan")
        worst_Dr = max(worst_Dr, abs(Dr - 1.0))
        if Q > Qb + 1e-9:
            q_rose = True
        if w < 19.0 and Er > 1.0 + 1e-9:
            e_rose_thin = True
        print(f"{w:6.2f} | {Db*SPD*1e6:11.4f} {Dg*SPD*1e6:11.4f} {Dr:6.3f} | "
              f"{Eb*SPD*1e6:11.4f} {E*SPD*1e6:11.4f} {Er:6.3f} | "
              f"{Qb:7.3f} {Q:7.3f}")

    print("-" * 94)
    print("\nGATE:")
    ok_abs = worst_Dr <= 0.15
    print(f"  absorbed within +-15% at every anchor : "
          f"{'PASS' if ok_abs else 'REJECT'}  (worst |Dr-1| = {worst_Dr:.3f})")
    print(f"  <Q> never rises vs baseline           : "
          f"{'PASS' if not q_rose else 'REJECT'}")
    print(f"  dose-eq E lower at thin anchors        : "
          f"{'PASS' if not e_rose_thin else 'CHECK (E rose at a thin anchor)'}")
    verdict = "GO -- ship (cp to lunarsim/data/)" if (ok_abs and not q_rose) \
        else "HOLD -- inspect before shipping"
    print(f"\n  => {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
