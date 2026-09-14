"""Re-run harness for the GCR proton-LIS rigidity fix.

Background
----------
`make_source.lis_proton` implements the Burger/Usoskin local interstellar
spectrum

    j_LIS(P) = 1.9e4 * P^-2.78 / (1 + 0.4866 * P^-2.51)   [/(m^2 s sr GeV)]

which is a function of magnetic RIGIDITY P (GV), not kinetic energy T (GeV).
The legacy code evaluated the form AT kinetic energy (P := T); the fix converts
T -> P = sqrt(T*(T + 2*m_p)) before evaluating it. `dosimetry` samples the same
`lis_proton`, so the sampled spectrum and the flux normalisation move together
and cannot drift. The `LUNARSIM_GCR_LIS` env knob selects the form:
"rigidity" (default, correct) vs "energy" (legacy, kept only for this harness).

Because `_calibration_factor()` pins the TOTAL proton number flux to 2.0/cm^2/s
for BOTH forms, the fix is a pure spectral REDISTRIBUTION of a fixed particle
count, not a flux change. The harder rigidity spectrum strips the low-energy
protons that dominate thin-wall dose, so behind thin/moderate shielding the
effective dose DROPS; it only crosses over to an increase behind thick
shielding (~>=50 g/cm^2 absorbed).

What this harness does
----------------------
  compare            offline before/after fold table (no MC). Uses the env knob
                     in isolated subprocesses so the two LIS forms can't share
                     the module cache. Prints the new test targets.
  anchor  <outdir>   MC: re-runs the Chang'E-4 3 m-dome flood (5 cm Al + 5 cm
                     regolith, ~22 g/cm^2) under BOTH LIS forms and reports the
                     absorbed dose rate (uGy/h) each, the ratio, the current
                     OUTER_GAUGE_ANCHOR_CAL, and the CAL_new that restores the
                     13.2 uGy/h Chang'E-4 anchor. HEAVY -- run on the PC.
  regolith <outdir>  MC: sets up the regolith depth sweep (the CAL-independent,
                     direct deep-dose test of the fix) and prints the run/collect
                     commands. HEAVY -- run on the PC.
  runbook            print the ordered step list.

The MC subcommands only PREPARE / DRIVE runs; you execute them on the 24-core
PC. Nothing here feeds a dose target into the physics (target-blind): the
re-anchor is a post-hoc geometric constant tied to a measured surface dose.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Chang'E-4 LND surface anchor (Zhang et al. 2020, Sci. Adv. 6, eaaz1334):
# 5 cm Al + 5 cm regolith ~= 22 g/cm^2, absorbed 13.2 uGy/h.
ANCHOR_TARGET_UGY_H = 13.2
ANCHOR_SHAPE = "dome"
ANCHOR_INNER_R_CM = 300.0
ANCHOR_WALLS = [("aluminium", 5.0), ("regolith", 5.0)]

# The offline before/after check is done on the thinnest validated kernel anchor:
# a 7.5 mm Al dome (~2.03 g/cm^2), the design behind the 342.6 -> 295.4 mSv/yr
# test targets in tests/test_pipeline.py.
THIN_SHAPE = "dome"
THIN_INNER_R_CM = 750.0
THIN_WALLS = [("aluminium", 0.75)]

LIS_FORMS = ("energy", "rigidity")   # legacy, fixed


# ----------------------------------------------------------------------
# compare  (offline, no MC)
# ----------------------------------------------------------------------
_FOLD_SNIPPET = """
import json
from lunarsim.spec import HabitatSpec, WallLayer
from lunarsim import dosimetry
# fresh exec of make_source under this process's LUNARSIM_GCR_LIS
dosimetry._load_make_source.cache_clear()
spec = HabitatSpec(name="cmp", shape={shape!r}, inner_radius_cm={r},
                   walls=[WallLayer(m, t) for m, t in {walls!r}])
a = dosimetry.assess_gcr_thinwall(spec)
ms = dosimetry._load_make_source()
out = {{
    "areal_gcm2": spec.areal_density_gcm2(),
    "annual_msv": a.annual_msv,          # ICRP effective dose
    "annual_mgy": a.annual_mgy,          # effective absorbed dose
    "eff_q": a.quality_factor,           # emergent E / D_eff
    "raw_2pi": dosimetry._raw_gcr_integral(1, 1, 400.0),
    "calib": dosimetry._calibration_factor(),
}}
print("RESULT " + json.dumps(out))
"""


def _fold_in_subprocess(lis_var: str, shape: str, r: float, walls) -> dict:
    """Run one offline thin-wall fold with LUNARSIM_GCR_LIS=lis_var in a fresh
    interpreter, so the two forms never share make_source's module cache."""
    env = dict(os.environ, LUNARSIM_GCR_LIS=lis_var)
    snippet = _FOLD_SNIPPET.format(shape=shape, r=r, walls=walls)
    proc = subprocess.run([sys.executable, "-c", snippet], cwd=str(REPO_ROOT),
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"fold subprocess ({lis_var}) failed")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def cmd_compare(_args) -> None:
    old = _fold_in_subprocess("energy", THIN_SHAPE, THIN_INNER_R_CM, THIN_WALLS)
    new = _fold_in_subprocess("rigidity", THIN_SHAPE, THIN_INNER_R_CM, THIN_WALLS)

    print("LIS rigidity fix -- offline before/after (no MC)")
    print(f"  design: {THIN_SHAPE}, inner_r={THIN_INNER_R_CM:.0f} cm, "
          f"walls={THIN_WALLS}  ({old['areal_gcm2']:.2f} g/cm^2)\n")

    print("  quantity                 OLD (energy)   NEW (rigidity)     ratio")
    print("  " + "-" * 64)

    def row(label, ov, nv, fmt="{:>12.3f}"):
        ratio = (nv / ov) if ov else float("nan")
        print(f"  {label:<22}" + fmt.format(ov) + "   " + fmt.format(nv)
              + f"   {ratio:>7.3f}")

    row("effective E (mSv/yr)", old["annual_msv"], new["annual_msv"])
    row("effective D (mGy/yr)", old["annual_mgy"], new["annual_mgy"])
    row("emergent Q (E/D)", old["eff_q"], new["eff_q"])
    row("raw 2pi integral", old["raw_2pi"], new["raw_2pi"])
    row("calibration factor", old["calib"], new["calib"])
    print()
    print("  Flux is pinned to 2.0 /cm^2/s for BOTH forms (calib self-corrects),")
    print("  so this is a spectral redistribution, not a flux change.\n")

    print("  New test targets (tests/test_pipeline.py, 2.03 g/cm^2 dome):")
    print(f"    test_thin_al_dome_reproduces_validated_fold: "
          f"annual_msv {old['annual_msv']:.1f} -> {new['annual_msv']:.1f}")
    print(f"    test_combined_thin_al_dome_serves_thinwall_gate1: "
          f"overlay['skin'] {old['annual_msv']:.1f} -> {new['annual_msv']:.1f}")
    print("  (Both asserts already updated to 295.4 in the repo; this confirms the number.)")


# ----------------------------------------------------------------------
# anchor  (MC -- run on the PC)
# ----------------------------------------------------------------------
def _run_anchor_mc(lis_var: str, run_dir: Path, tier):
    """One Chang'E-4-dome flood MC under the chosen LIS form. Returns the
    parsed DoseAssessment-derived absorbed rates. HEAVY (invokes TOPAS)."""
    from lunarsim.spec import HabitatSpec, WallLayer
    from lunarsim import bridge, dosimetry

    os.environ["LUNARSIM_GCR_LIS"] = lis_var          # source subprocess inherits it
    dosimetry._load_make_source.cache_clear()          # dosimetry re-reads it too
    spec = HabitatSpec(name=f"change4_{lis_var}", shape=ANCHOR_SHAPE,
                       inner_radius_cm=ANCHOR_INNER_R_CM,
                       walls=[WallLayer(m, t) for m, t in ANCHOR_WALLS])
    print(f"  [{lis_var}] running flood MC ({spec.areal_density_gcm2():.1f} g/cm^2, "
          f"tier={tier.name}, {tier.total_primaries} primaries) ... this takes a while",
          flush=True)
    res = bridge.run_design(spec, tier=tier, run_dir=run_dir, keep=True)
    if not res.ok:
        raise SystemExit(f"[{lis_var}] MC run failed (rc={res.returncode}); "
                         f"see {run_dir}/topas_stdout.log")
    a_skin = dosimetry.assess(res, skin=True)
    a_phan = dosimetry.assess(res, skin=False)
    return {
        "skin_ugy_h": a_skin.dose_rate_ugy_day / 24.0,
        "phantom_ugy_h": a_phan.dose_rate_ugy_day / 24.0,
        "wall_seconds": res.wall_seconds,
    }


def cmd_anchor(args) -> None:
    from lunarsim import bridge, dosimetry
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cal_old = dosimetry.OUTER_GAUGE_ANCHOR_CAL

    # The absorbed skin dose behind ~22 g/cm^2 is dominated by rare Fe/Si
    # surface hits; FULL_RUN (10,080 primaries) gives ~120 Si / ~20 Fe and the
    # skin rate is noise-limited. --mult scales histories/source so the heavy-ion
    # tail converges before any CAL is derived. mult=20 ~= 200k primaries/form.
    mult = max(1, int(args.mult))
    base = bridge.FULL_RUN
    tier = dataclasses.replace(base, name=f"anchor{mult}x",
                               histories=base.histories * mult)

    print("Chang'E-4 re-anchor MC (Zhang et al. 2020, target "
          f"{ANCHOR_TARGET_UGY_H} uGy/h absorbed)")
    print(f"  anchor design: {ANCHOR_SHAPE}, inner_r={ANCHOR_INNER_R_CM:.0f} cm, "
          f"walls={ANCHOR_WALLS}")
    print(f"  statistics: {mult}x FULL_RUN = {tier.total_primaries} primaries/form")
    print(f"  current OUTER_GAUGE_ANCHOR_CAL = {cal_old:.5f}\n")

    results = {v: _run_anchor_mc(v, outdir / f"lis_{v}", tier) for v in LIS_FORMS}

    print("\n  absorbed dose rate (uGy/h), CAL_old applied:")
    print("  form        skin      phantom")
    for v in LIS_FORMS:
        r = results[v]
        print(f"  {v:<9} {r['skin_ugy_h']:>8.3f}  {r['phantom_ugy_h']:>8.3f}")

    # Pick the basis (skin vs phantom) whose OLD/energy value reproduces the
    # 13.2 uGy/h anchor -- that is the basis the CAL was pinned on.
    old = results["energy"]
    basis = min(("skin_ugy_h", "phantom_ugy_h"),
                key=lambda k: abs(old[k] - ANCHOR_TARGET_UGY_H))
    new_rate = results["rigidity"][basis]
    old_rate = old[basis]
    cal_new = cal_old * ANCHOR_TARGET_UGY_H / new_rate

    print(f"\n  anchor basis: {basis.split('_')[0]} "
          f"(OLD reads {old_rate:.3f} ~= {ANCHOR_TARGET_UGY_H} target)")
    print(f"  NEW (rigidity) reads {new_rate:.3f} uGy/h with CAL_old -> "
          f"ratio {new_rate / old_rate:.4f}")
    print(f"  CAL_new = CAL_old * {ANCHOR_TARGET_UGY_H} / {new_rate:.3f} "
          f"= {cal_new:.5f}")
    print("\n  To re-anchor: set OUTER_GAUGE_ANCHOR_CAL to the value above in")
    print("  lunarsim/dosimetry.py (leave the geometric-derivation comment, add a")
    print("  Stage-4 line: LIS rigidity fix re-anchor). Then re-run the test suite.")

    summary = {"cal_old": cal_old, "basis": basis, "results": results,
               "cal_new": cal_new, "target_ugy_h": ANCHOR_TARGET_UGY_H,
               "mult": mult, "primaries_per_form": tier.total_primaries}
    (outdir / "anchor_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {outdir / 'anchor_summary.json'}")


# ----------------------------------------------------------------------
# regolith  (MC -- run on the PC; CAL-independent)
# ----------------------------------------------------------------------
def cmd_regolith(args) -> None:
    outdir = Path(args.outdir).resolve()
    # The sweep uses a CLEAN first-principles phi_ff normalisation (CAL-independent),
    # so it is the DIRECT test of the fix at depth -- no re-anchor needed first.
    # Shell out to the regolith_sweep CLI (drift-proof) with the LIS knob set.
    env = dict(os.environ, LUNARSIM_GCR_LIS="rigidity")
    print("Regolith depth sweep (rigidity LIS, CAL-independent deep-dose test)")
    subprocess.run([sys.executable, "-m", "lunarsim.regolith_sweep",
                    "generate", str(outdir)], cwd=str(REPO_ROOT), env=env, check=True)
    print(f"\n  generated under {outdir}. Now run the MC and collect:")
    print(f"    bash {outdir}/run.sh")
    print(f"    python -m lunarsim.regolith_sweep collect {outdir}")
    print(f"  -> {outdir}/dose_vs_depth.csv (compare against the committed curve).")
    print("\n  For an explicit before/after at depth, repeat with")
    print("  LUNARSIM_GCR_LIS=energy into a second outdir and diff the CSVs.")


# ----------------------------------------------------------------------
# runbook
# ----------------------------------------------------------------------
_RUNBOOK = """\
LIS rigidity-fix re-run runbook  (PC: TopasSimulationPC, over Tailscale)
========================================================================
The code change (make_source.lis_proton T->P + LUNARSIM_GCR_LIS knob) and the
updated test targets are already committed on the laptop side. On the PC:

  0. Sync + restart:
       cd ~/topas && git pull
       # restart the Dash server if it was running (it caches the old code)

  1. Offline sanity (fast, no MC) -- confirm the before/after numbers:
       python -m lunarsim.rerun_lis_fix compare
     Expect effective E ~342.6 (energy) -> ~295.4 (rigidity) at 2.03 g/cm^2.

  2. Re-anchor MC (heavy) -- restore the Chang'E-4 13.2 uGy/h anchor:
       python -m lunarsim.rerun_lis_fix anchor runs/lis_anchor
     Read CAL_new from the output, edit OUTER_GAUGE_ANCHOR_CAL in
     lunarsim/dosimetry.py, then: python -m pytest lunarsim/tests -q

  3. Regolith depth sweep (heavy, CAL-independent) -- the direct deep-dose test:
       python -m lunarsim.rerun_lis_fix regolith runs/lis_regolith
       bash runs/lis_regolith/run.sh
       python -m lunarsim.regolith_sweep collect runs/lis_regolith
     Compare runs/lis_regolith/dose_vs_depth.csv against the committed curve;
     expect thin depths slightly lower, deep depths slightly higher.

  4. Commit the re-anchored CAL + refreshed sweep CSV/plot (you own commits).
"""


def cmd_runbook(_args) -> None:
    print(_RUNBOOK)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compare", help="offline before/after fold table (no MC)")
    a = sub.add_parser("anchor", help="MC: Chang'E-4 re-anchor (heavy)")
    a.add_argument("outdir")
    a.add_argument("--mult", type=int, default=1,
                   help="statistics multiplier on FULL_RUN histories "
                        "(20 ~= 200k primaries/form; needed for the Fe/Si skin tail)")
    g = sub.add_parser("regolith", help="MC: set up the depth sweep (heavy)")
    g.add_argument("outdir")
    sub.add_parser("runbook", help="print the ordered step list")
    args = p.parse_args(argv)

    {"compare": cmd_compare, "anchor": cmd_anchor,
     "regolith": cmd_regolith, "runbook": cmd_runbook}[args.cmd](args)


if __name__ == "__main__":
    main()
