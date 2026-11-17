#!/usr/bin/env python3
"""Energy-balance check on a finished paper/delta_ray_cut_test.py run tree (no TOPAS).

The 2026-11-17 diagnostic showed that raising the phantom's electron production cut
(0.05 mm -> 1 mm -> 100 cm) RAISES absorbed dose in the 3 cm 'mid' shell 2-6x, while the
region alone (default cut) is bit-identical to modular. Delta rays carry ~17% of an
ion's loss, and the shells are cm thick, so a physical effect cannot do that. This
script tests whether energy is conserved:

  E_dep  = sum over ALL water shells of D[Gy] x mass[kg]       (J -> MeV)
  E_inc  = sum over ring sources of BeamEnergy x NumberOfHistoriesInRun  (from run.txt)

E_dep / E_inc is an upper-bound check: not every history reaches the phantom, so the
ratio must be < 1 in every config. A ratio > 1, or a config that deposits several times
more TOTAL energy than modular, is energy non-conservation (a transport/cut artifact),
not delta-ray redistribution. The per-shell ratios show WHERE the excess sits.

    .venv/bin/python paper/delta_ray_energy_balance.py runs/delta54diag
"""
import argparse
import math
import re
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lunarsim.kernel_gen import config, templates  # noqa: E402
import fragmentation_sensitivity as fs  # noqa: E402

MEV_PER_J = 1.0 / 1.602176634e-13
WATER_KG_PER_CM3 = 1.0e-3

_RUN_RE = re.compile(r"^(?P<sp>[A-Za-z]+)_n(?P<node>\d+)_s(?P<seed>\d+)$")


def shell_masses_kg() -> dict:
    return {name: WATER_KG_PER_CM3 * 4.0 / 3.0 * math.pi * (r_out ** 3 - r_in ** 3)
            for name, r_in, r_out in config.SHELLS}


def incident_mev(run_txt: str) -> float:
    """Sum of BeamEnergy x NumberOfHistoriesInRun over every ring source K<i>."""
    energy, hist = {}, {}
    for ln in run_txt.splitlines():
        m = re.match(r"d:So/(K\d+)/BeamEnergy\s*=\s*([0-9.eE+-]+)\s*MeV", ln)
        if m:
            energy[m.group(1)] = float(m.group(2))
        m = re.match(r"i:So/(K\d+)/NumberOfHistoriesInRun\s*=\s*(\d+)", ln)
        if m:
            hist[m.group(1)] = int(m.group(2))
    if not energy or set(energy) != set(hist):
        raise ValueError("could not pair BeamEnergy / NumberOfHistoriesInRun in run.txt")
    return sum(energy[k] * hist[k] for k in energy)


def shell_edep_mev(rundir: Path, masses: dict) -> dict | None:
    out = {}
    for name in masses:
        d = fs._read_scalar_csv(rundir / f"{templates.scorer_csv_names(name)[0]}.csv")
        if d is None:
            return None
        out[name] = d * masses[name] * MEV_PER_J
    return out


def collect(base: Path) -> dict:
    """{(species, node): {cfg: [(E_inc, {shell: E_dep}), ...per seed]}}"""
    masses = shell_masses_kg()
    data = {}
    for cfgdir in sorted(p for p in base.iterdir() if p.is_dir()):
        for rundir in sorted(p for p in cfgdir.iterdir() if p.is_dir()):
            m = _RUN_RE.match(rundir.name)
            if not m or not (rundir / "run.txt").exists():
                continue
            edep = shell_edep_mev(rundir, masses)
            if edep is None:
                continue
            einc = incident_mev((rundir / "run.txt").read_text())
            key = (m.group("sp"), int(m.group("node")))
            data.setdefault(key, {}).setdefault(cfgdir.name, []).append((einc, edep))
    return data


def mean_sem(xs):
    if len(xs) < 2:
        return xs[0], float("nan")
    return statistics.fmean(xs), statistics.stdev(xs) / math.sqrt(len(xs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=Path)
    args = ap.parse_args()
    data = collect(args.outdir)
    if not data:
        print(f"no finished runs under {args.outdir}")
        return 1
    shells = [s[0] for s in config.SHELLS]
    for (sp, node), bycfg in sorted(data.items()):
        ref = "modular" if "modular" in bycfg else sorted(bycfg)[0]
        ref_tot = statistics.fmean(sum(e.values()) for _, e in bycfg[ref])
        ref_sh = {s: statistics.fmean(e[s] for _, e in bycfg[ref]) for s in shells}
        print(f"\n{sp} node {node}   (E_dep/E_inc must be < 1; 'x ref' = vs {ref})")
        print(f"  {'config':8} {'n':>2} {'E_dep/E_inc':>14} {'total x ref':>12}  "
              + "  ".join(f"{s:>8}" for s in shells))
        for cfg in sorted(bycfg, key=lambda c: (c != "ftfp", c != ref, c)):
            runs = bycfg[cfg]
            frac, frac_e = mean_sem([sum(e.values()) / ei for ei, e in runs])
            tot = statistics.fmean(sum(e.values()) for _, e in runs)
            per = "  ".join(
                f"{statistics.fmean(e[s] for _, e in runs) / ref_sh[s]:>8.2f}"
                if ref_sh[s] > 0 else f"{'--':>8}" for s in shells)
            flag = "  <-- E_dep > E_inc: NOT CONSERVED" if frac > 1.0 else ""
            print(f"  {cfg:8} {len(runs):>2} {frac:>8.4f}±{frac_e:<5.4f} "
                  f"{tot / ref_tot:>12.2f}  {per}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
