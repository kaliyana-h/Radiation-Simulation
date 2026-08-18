#!/usr/bin/env python3
"""Threads A/B: prove multithreaded TOPAS scoring is statistically identical to
single-threaded -- i.e. LUNARSIM_THREADS changes only SPEED, not the physics.

TOPAS MT gives each worker its own RNG substream, so multi- and single-threaded
runs are NOT bit-identical: they are independent estimates of the same true dose.
This runs K independent-seed runs in each mode and checks the two MEANS agree
within their combined standard error. A scoring bug in the MT merge would show up
as the multi-thread mean sitting off the single-thread mean by many sigma (large
|z|). As a bonus it reports the single/multi wall-time speedup on THIS machine.

Run on a box with the topas binary + G4Data (the PC, or any built checkout):

    .venv/bin/python scripts/threads_ab.py

Tunables (env): AB_SEEDS (default 5), AB_TIER (quick|full, default full),
    AB_SINGLE (the single-thread baseline, default 1 = one genuine core; do NOT
    use 0 here -- TOPAS reads 0 as ALL cores, so it is not a single-thread
    baseline), AB_MULTI (the multi-thread value to test, default -2).
"""
import os
import statistics
import sys
import time

# allow running as a plain script (scripts/ is not the import root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunarsim.bridge import run_design, QUICK_LOOK, FULL_RUN
from lunarsim.spec import default_spec

SEEDS = int(os.environ.get("AB_SEEDS", "5"))
TIER = FULL_RUN if os.environ.get("AB_TIER", "full") == "full" else QUICK_LOOK
SINGLE = int(os.environ.get("AB_SINGLE", "1"))   # 1 = one genuine core (NOT 0)
MULTI = int(os.environ.get("AB_MULTI", "-2"))

# metric attr -> label; first fully-populated one gates the verdict
METRICS = [("skin_doseeq_sv", "skin H*"),
           ("skin_dose_gy", "skin dose"),
           ("dose_gy", "phantom D")]


def run_mode(threads, label):
    """K seeded runs at a fixed thread count; returns {attr: [values]} + walls."""
    cols = {attr: [] for attr, _ in METRICS}
    walls = []
    for i in range(SEEDS):
        t0 = time.time()
        r = run_design(default_spec(), TIER, threads=threads, seed=1000 + i, keep=False)
        walls.append(time.time() - t0)
        if not r.ok:
            print(f"  [{label}] seed {1000 + i} FAILED rc={r.returncode}")
            sys.exit(2)
        for attr, _ in METRICS:
            cols[attr].append(getattr(r, attr))
        print(f"  [{label}] seed {1000 + i}: "
              + "  ".join(f"{lbl}={getattr(r, attr)}" for attr, lbl in METRICS)
              + f"  ({walls[-1]:.1f}s)")
    return cols, walls


def _stats(xs):
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")
    return m, se


def compare(label, a, b):
    """Return True if the two means agree within 3 combined SE, None if a metric
    is not populated (custom scorer absent from this binary)."""
    if any(v is None for v in a + b):
        print(f"  {label:10s} n/a (metric not populated in this binary)")
        return None
    ma, sea = _stats(a)
    mb, seb = _stats(b)
    comb = (sea ** 2 + seb ** 2) ** 0.5
    z = (ma - mb) / comb if comb else float("nan")
    pct = 100 * (mb - ma) / ma if ma else float("nan")
    verdict = "CONSISTENT" if abs(z) < 3 else "SUSPECT"
    print(f"  {label:10s} single={ma:.4g}+-{sea:.2g}  multi={mb:.4g}+-{seb:.2g}"
          f"  diff={pct:+.1f}%  z={z:+.2f}  -> {verdict}")
    return abs(z) < 3


def main():
    tier_name = os.environ.get("AB_TIER", "full")
    print(f"THREADS A/B  design=default dome  tier={tier_name}  seeds={SEEDS}  "
          f"single={SINGLE} vs multi={MULTI}\n")
    print(f"single-thread (threads={SINGLE}):")
    c0, w0 = run_mode(SINGLE, "single")
    print(f"multi-thread (threads={MULTI}):")
    c1, w1 = run_mode(MULTI, "multi")

    print("\nverdict (means must agree within combined SE; |z|<3 = unbiased):")
    results = [compare(lbl, c0[attr], c1[attr]) for attr, lbl in METRICS]
    gated = [r for r in results if r is not None]

    m0, m1 = statistics.mean(w0), statistics.mean(w1)
    spd = m0 / m1 if m1 else float("nan")
    print(f"  {'speedup':10s} single {m0:.1f}s/run  ->  multi {m1:.1f}s/run"
          f"  = {spd:.1f}x faster on this box")

    ok = bool(gated) and all(gated)
    print("\nRESULT:", "PASS -- MT is statistically identical to single-thread"
          if ok else "FAIL -- means diverge or nothing measurable; investigate")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
