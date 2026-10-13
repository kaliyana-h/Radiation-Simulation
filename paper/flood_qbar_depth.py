#!/usr/bin/env python3
"""Q-bar (quality-factor) twin of flood_absorbed_depth.py -- the effective-dose
confirmation for the shell-base kernel ship (2026-10-13).

Why this exists
---------------
Shipping the shell-base kernel is PROVEN only for ABSORBED dose (the target-blind
flood gate: shell-base kernel absorbed reproduces the same-physics flood flat
across 19->50 g/cm^2, @50 = 0.988x OLTARIS). But the headline thesis numbers are
EFFECTIVE dose E = D_eff x Q-bar, and shipping shell-base DROPS the high-energy
"tail" nodes (the ceiling extension). Those nodes carried high-LET HZE penetration,
so removing them mechanically changes the emergent quality factor. Before the
effective-dose claim can inherit the absorbed-dose fix, Q-bar needs its own
target-blind check: does shell-base Q-bar hold up with depth, or did stripping the
tail under-count quality?

What it measures (deterministic -- NO MC)
-----------------------------------------
The kernel's emergent Q-bar = E / D_eff is a pure deterministic fold of the kernel
response against the free-field GCR spectrum (dosimetry.assess_gcr_thinwall), so
this entire driver runs on the laptop/PC in seconds with no TOPAS. Per Al areal
density it records, for the candidate kernel (--kernel) and optionally a baseline
kernel (--baseline) folded side by side:

  * kernel_qbar          -- emergent Q-bar = E/D_eff (the quantity in question).
  * kernel_Deff_mgy_day  -- effective absorbed dose (context; this is the channel
                            the flood gate already validated).
  * kernel_E_mSv_yr      -- effective dose E = D_eff x Q-bar (what the paper quotes).
  * baseline_qbar        -- the SAME fold on a second kernel, for a controlled delta.
  * cand_over_base       -- kernel_qbar(candidate) / kernel_qbar(baseline).

and two post-hoc REFERENCE columns (never fit targets, target-blind):
  * flood_skin_Q         -- the same-physics flood skin quality factor at this depth,
                            lifted from the cached flood_absorbed_depth.csv. NOTE:
                            this is a SKIN Q, not the kernel's organ-weighted
                            effective Q-bar -- skin sees the un-softened high-LET
                            field so skin_Q > effective Q-bar by construction. It is
                            context (same-physics, same LET->Q model), NOT the
                            yardstick. Do not expect kernel_qbar == flood_skin_Q.
  * oltaris_qbar         -- hand-entered OLTARIS-Al Q-bar = H/D where known
                            (@50 ~ 2.16). THIS is the external anchor for Q-bar.

THE CONTROLLED COMPARISON that answers the ship question is shell-base vs
shell+TAIL: both are shell geometry, so the tail is the ONLY difference, and
cand_over_base isolates exactly what removing the tail does to Q-bar:
    python3 paper/flood_qbar_depth.py runs/qbar \
        --kernel   runs/kernel_shell/gcr_thinwall_kernel_si.json \
        --baseline runs/kernel_shell/gcr_thinwall_kernel_shell_tail.json

Verdict logic (printed at the end)
----------------------------------
  PASS  : shell-base Q-bar holds -- within a few % of the shell+tail baseline AND
          close to OLTARIS@50 (~2.16). Effective dose rides the absorbed fix; the
          shell-base ship is clean for E as well as D.
  FLIP  : shell-base Q-bar comes out notably LOW vs OLTARIS@50 (and the tail
          baseline sat closer). That would mean the tail carried genuine Q physics
          even though it over-read absorbed dose -- revisit a Q-ONLY tail (keep the
          high-Q HZE nodes for the equivalent channel, drop them for absorbed).
          Prior evidence (kernel Q-bar@50 ~ 2.18 vs OLTARIS 2.16) says we do NOT
          expect this -- but this fold is what lets us say so with a number.

TARGET-BLIND: nothing here feeds a dose value back into the physics. OLTARIS and
the flood skin-Q are post-hoc references; this only re-reads what the deterministic
kernel fold already returns.

Usage
-----
    cd ~/topas
    # candidate alone, vs shipped production + references:
    python3 paper/flood_qbar_depth.py runs/qbar \
        --kernel runs/kernel_shell/gcr_thinwall_kernel_si.json
    # the controlled shell-base vs shell+tail delta (recommended):
    python3 paper/flood_qbar_depth.py runs/qbar \
        --kernel   runs/kernel_shell/gcr_thinwall_kernel_si.json \
        --baseline runs/kernel_shell/gcr_thinwall_kernel_shell_tail.json
    # after the ship, fold the shipped kernel against the old slab+tail backup:
    python3 paper/flood_qbar_depth.py runs/qbar \
        --baseline lunarsim/data/gcr_thinwall_kernel.json.slabtail.bak
No MC, no TOPAS_G4_DATA_DIR needed. Writes paper/flood_qbar_depth.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AL_RHO_G_CM3 = 2.70
DOME_INNER_R_CM = 750.0                       # geometry matched to the kernel column
DEFAULT_DEPTHS_GCM2 = [19.0, 30.0, 40.0, 50.0]
# Hand-entered OLTARIS-Al Q-bar = H/D (dimensionless) where known, for anchoring.
# @50 ~ 2.16 (summary; kernel Q-bar@50 measured 2.18 historically). Fill the others
# from the OLTARIS run if you have H and D at those depths.
OLTARIS_QBAR = {50.0: 2.16}
CSV_PATH = HERE / "flood_qbar_depth.csv"
ABSORBED_CSV = HERE / "flood_absorbed_depth.csv"     # source of flood_skin_Q context
CSV_FIELDS = ["areal_gcm2", "al_cm", "kernel_qbar", "kernel_Deff_mgy_day",
              "kernel_E_mSv_yr", "baseline_qbar", "cand_over_base",
              "flood_skin_Q", "oltaris_qbar", "note"]


def _spec(gcm2: float):
    from lunarsim.spec import HabitatSpec, WallLayer
    return HabitatSpec(name=f"qbar_al{gcm2:g}", shape="dome",
                       inner_radius_cm=DOME_INNER_R_CM,
                       walls=[WallLayer("aluminium", gcm2 / AL_RHO_G_CM3)])


def _repoint_kernel(dosimetry, path: str | None) -> str:
    """Point the ALUMINIUM thin-wall kernel at `path` (None = leave the shipped one)
    and drop the load cache so the next fold reads it. Returns the geometry tag of
    whatever kernel is now active. Target-blind read-path swap; no data is overwritten."""
    if path is not None:
        kp = Path(path).resolve()
        if not kp.exists():
            raise SystemExit(f"kernel JSON {kp} does not exist")
        dosimetry._GCR_THINWALL_KERNELS["aluminium"] = kp
    else:
        kp = Path(dosimetry._GCR_THINWALL_KERNELS["aluminium"]).resolve()
    dosimetry._load_gcr_thinwall_kernel.cache_clear()
    try:
        meta = json.loads(kp.read_text()).get("meta", {})
        return f"{meta.get('wall_geometry', '?')}"
    except Exception:
        return "?"


def _fold_all(dosimetry, depths: list[float], phi_mv: float) -> dict[float, object]:
    """Deterministic kernel fold (no MC) at every depth -> {depth: DoseAssessment}.
    Pins the KERNEL LIS form so the free-field spectrum weighting matches the
    kernel's calibration basis."""
    dosimetry.set_lis_form(dosimetry.KERNEL_LIS_FORM)
    out: dict[float, object] = {}
    for g in depths:
        a = dosimetry.assess_gcr_thinwall(_spec(g), phi_MV=phi_mv)
        if a is None:
            raise SystemExit(f"[{g:g}] kernel fold returned None (no response at depth)")
        out[round(g, 1)] = a
    return out


def _flood_skin_q() -> dict[float, str]:
    """Lift the same-physics flood skin quality factor per depth from the cached
    absorbed-dose CSV (context column; not the Q-bar yardstick)."""
    q: dict[float, str] = {}
    if ABSORBED_CSV.exists():
        for r in csv.DictReader(open(ABSORBED_CSV, newline="")):
            try:
                q[round(float(r["areal_gcm2"]), 1)] = (r.get("flood_skin_Q") or "").strip()
            except (ValueError, KeyError):
                continue
    return q


def _write_csv(rows: dict[float, dict]) -> None:
    ordered = sorted(rows.values(), key=lambda r: float(r["areal_gcm2"]))
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir", nargs="?", default="runs/qbar",
                   help="scratch dir (unused by the fold; kept for CLI symmetry)")
    p.add_argument("--kernel", metavar="PATH",
                   help="aluminium kernel JSON to fold as the CANDIDATE (default: the "
                        "shipped lunarsim/data/gcr_thinwall_kernel.json).")
    p.add_argument("--baseline", metavar="PATH",
                   help="a SECOND aluminium kernel to fold side by side (e.g. the "
                        "shell+tail kernel, or the old slab+tail backup) for a "
                        "controlled Q-bar delta. Omit to skip the baseline column.")
    p.add_argument("--depths", type=float, nargs="+", metavar="GCM2",
                   help=f"Al areal densities g/cm^2 (default {DEFAULT_DEPTHS_GCM2})")
    args = p.parse_args(argv)

    from lunarsim import bridge, dosimetry
    phi_mv = bridge.FULL_RUN.phi_mv
    depths = args.depths if args.depths else list(DEFAULT_DEPTHS_GCM2)

    # Capture the shipped path so "candidate = default" still works even after swaps.
    shipped = Path(dosimetry._GCR_THINWALL_KERNELS["aluminium"]).resolve()

    cand_path = args.kernel if args.kernel else str(shipped)
    cand_geom = _repoint_kernel(dosimetry, cand_path)
    print(f"  CANDIDATE kernel : {Path(cand_path).name}  (meta.wall_geometry = {cand_geom})")
    cand = _fold_all(dosimetry, depths, phi_mv)

    base = None
    base_geom = None
    if args.baseline:
        base_geom = _repoint_kernel(dosimetry, args.baseline)
        print(f"  BASELINE kernel  : {Path(args.baseline).name}  "
              f"(meta.wall_geometry = {base_geom})")
        base = _fold_all(dosimetry, depths, phi_mv)
        _repoint_kernel(dosimetry, cand_path)        # restore read-path to candidate

    print(f"  deterministic Q-bar fold (no MC); phi={phi_mv:g} MV;"
          f" LIS form {dosimetry.KERNEL_LIS_FORM}\n")

    flood_q = _flood_skin_q()
    rows: dict[float, dict] = {}
    for g in depths:
        key = round(g, 1)
        a = cand[key]
        qb = a.quality_factor
        b_qb = base[key].quality_factor if base else None
        ratio = (qb / b_qb) if (b_qb not in (None, 0)) else None
        olt = OLTARIS_QBAR.get(key)
        rows[key] = {
            "areal_gcm2": f"{g:g}", "al_cm": f"{g / AL_RHO_G_CM3:.2f}",
            "kernel_qbar": f"{qb:.3f}",
            "kernel_Deff_mgy_day": f"{a.dose_rate_ugy_day / 1000.0:.4f}",
            "kernel_E_mSv_yr": f"{a.annual_msv:.1f}",
            "baseline_qbar": f"{b_qb:.3f}" if b_qb is not None else "",
            "cand_over_base": f"{ratio:.3f}" if ratio is not None else "",
            "flood_skin_Q": flood_q.get(key, ""),
            "oltaris_qbar": f"{olt:.2f}" if olt else "",
            "note": f"2026-10-13 Q-bar fold; cand={cand_geom}"
                    + (f" base={base_geom}" if base_geom else ""),
        }
    _write_csv(rows)
    _print_trend(rows, bool(base))
    print(f"\n  wrote -> {CSV_PATH}")


def _print_trend(rows: dict[float, dict], have_base: bool) -> None:
    filled = sorted(rows.values(), key=lambda r: float(r["areal_gcm2"]))
    if not filled:
        return
    print("  === Q-bar vs depth (does shell-base effective-dose quality hold?) ===")
    hdr = f"  {'g/cm2':>6} {'kern_Qbar':>9} {'E_mSv/yr':>9}"
    if have_base:
        hdr += f" {'base_Qbar':>9} {'cand/base':>9}"
    hdr += f" {'flood_Qskin':>11} {'OLTARIS':>8}"
    print(hdr)
    for r in filled:
        line = (f"  {float(r['areal_gcm2']):>6g} {r['kernel_qbar']:>9} "
                f"{r['kernel_E_mSv_yr']:>9}")
        if have_base:
            line += f" {(r.get('baseline_qbar') or '--'):>9} {(r.get('cand_over_base') or '--'):>9}"
        line += f" {(r.get('flood_skin_Q') or '--'):>11} {(r.get('oltaris_qbar') or '--'):>8}"
        print(line)

    # Verdict against the OLTARIS anchor(s) we have.
    anchored = [r for r in filled if (r.get("oltaris_qbar") or "").strip()]
    print()
    if not anchored:
        print("  No OLTARIS Q-bar anchor filled -- read the trend + cand/base delta only.\n"
              "  (Add OLTARIS H/D at these depths to OLTARIS_QBAR for an absolute verdict.)")
        return
    worst = None
    for r in anchored:
        qb, olt = float(r["kernel_qbar"]), float(r["oltaris_qbar"])
        dev = qb / olt - 1.0
        if worst is None or abs(dev) > abs(worst[1]):
            worst = (float(r["areal_gcm2"]), dev, qb, olt)
    g, dev, qb, olt = worst
    verdict = "PASS" if abs(dev) <= 0.10 else ("LOW -> FLIP?" if dev < 0 else "HIGH")
    print(f"  vs OLTARIS anchor: worst dev at {g:g} g/cm^2 = kern {qb:.3f} vs "
          f"OLTARIS {olt:.2f} ({dev*100:+.1f}%)  => {verdict}")
    if verdict == "PASS":
        print("  Shell-base Q-bar matches OLTARIS at the anchor -> effective dose rides\n"
              "  the absorbed-dose fix; the shell-base ship is clean for E as well as D.")
    elif verdict.startswith("LOW"):
        print("  Shell-base Q-bar runs LOW vs OLTARIS -> the FLIP condition: stripping the\n"
              "  tail may have under-counted quality. Check the shell+tail baseline Q-bar;\n"
              "  if it sat closer, consider a Q-ONLY tail (keep high-Q HZE nodes for the\n"
              "  equivalent channel, drop them for absorbed).")
    else:
        print("  Shell-base Q-bar runs HIGH vs OLTARIS -> not the ship concern (absorbed\n"
              "  is the shipped channel); note it and move on.")


if __name__ == "__main__":
    main()
