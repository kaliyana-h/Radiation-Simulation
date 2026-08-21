#!/usr/bin/env python3
"""Queue stress-test harness -- proves the JOB QUEUE holds up under workshop load.

Unlike test_pipeline.py (pure Python, no TOPAS, sub-second), this script REALLY
runs TOPAS through the real LocalThreadRunner, because the thing it validates is
operational, not physical: does the serial queue serialise, stay live, isolate a
cancelled job, and turn designs around fast enough for two teams? It asserts
NOTHING about dose values (target-blind) -- only about queue behaviour, liveness
and throughput.

Run it ON THE PC (that's the box under test), with the GUI service STOPPED so the
two don't fight over cores:

    sudo systemctl stop lunarsim-gui          # free the box
    export TOPAS_G4_DATA_DIR=~/G4Data
    cd ~/topas

    # fast smoke (~minutes): 6 cheap jobs, serial, quick tier
    python3 -m lunarsim.tests.stress_queue

    # realistic soak: 12 mixed jobs at the workshop's real tier + cancel test
    python3 -m lunarsim.tests.stress_queue --jobs 12 --tier full \
            --max-batches 8 --cancel-test --json ~/stress_report.json

    sudo systemctl start lunarsim-gui         # put it back

What it checks (each becomes a PASS/FAIL invariant + a nonzero exit on failure):
  * serialisation   -- observed concurrent RUNNING never exceeds max_parallel
  * liveness        -- no wedge: every submitted job reaches a terminal state
                       before the overall timeout (a stuck semaphore would hang
                       QUEUED jobs forever; this catches it)
  * cancellation    -- (--cancel-test) a job cancelled mid-run reaches CANCELLED
                       AND releases its slot so the following job still completes
  * throughput      -- reports jobs/hour and a projected jobs/day so you can size
                       the queue against the expected workshop volume

Mirrors production defaults: max_parallel=1 (each design gets the whole box) and
the same run_composition/run_converged paths the GUI submits. Raise --parallel
only to explore concurrent designs (and remember to divide LUNARSIM_THREADS).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make `lunarsim` importable when run as a bare file, matching test_pipeline.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TOPAS_G4_DATA_DIR", os.path.expanduser("~/G4Data"))

from lunarsim.spec import HabitatSpec, WallLayer
from lunarsim.bridge import QUICK_LOOK, FULL_RUN, WORST_CASE_SPE
from lunarsim.jobs import LocalThreadRunner, JobStatus


# --------------------------------------------------------------------------
# The workload: a spread of designs two teams would plausibly submit. Kinds map
# to the three distinct runner code paths (composition / single-species / SPE) so
# the soak exercises all of them, not just the cheap one.
# --------------------------------------------------------------------------
_WALLS = {
    "thin_al":     [WallLayer("aluminium", 5.0), WallLayer("regolith", 30.0)],
    "thick_reg":   [WallLayer("aluminium", 2.0), WallLayer("regolith", 60.0)],
    "poly_lined":  [WallLayer("polyethylene", 10.0), WallLayer("regolith", 40.0)],
    "water_lined": [WallLayer("aluminium", 3.0), WallLayer("water", 15.0),
                    WallLayer("regolith", 35.0)],   # the slow, H-rich case
}
_SHAPES = ["dome", "cylinder", "quonset"]
# (kind, submit-kwargs) -- the runner path each job takes.
_KINDS = {
    "gcr":    dict(composition=True),                 # workshop headline (all species)
    "proton": dict(composition=False),                # single-species, cheapest
    "spe":    dict(spe=WORST_CASE_SPE),               # acute-event cone source
}


def _build_workload(n: int) -> list[tuple[str, HabitatSpec, dict]]:
    """A deterministic, varied job list of length n: cycles shapes x wall stacks
    and weights kinds toward the realistic mix (mostly full GCR composition runs,
    some single-species, an occasional SPE)."""
    wall_keys = list(_WALLS)
    kind_cycle = (["gcr"] * 3 + ["proton"] * 2 + ["spe"])   # ~50% gcr, 33% proton, 17% spe
    jobs = []
    for i in range(n):
        shape = _SHAPES[i % len(_SHAPES)]
        wk = wall_keys[i % len(wall_keys)]
        kind = kind_cycle[i % len(kind_cycle)]
        inner = 250.0 + 50.0 * (i % 4)                # 250..400 cm
        spec = HabitatSpec(name=f"stress_{i:02d}_{shape}_{wk}", shape=shape,
                           inner_radius_cm=inner, walls=list(_WALLS[wk]))
        jobs.append((kind, spec, dict(_KINDS[kind])))
    return jobs


# --------------------------------------------------------------------------
# Instrumentation: sample the runner's live state so we can prove serialisation
# and catch a stall. Runs in its own thread; reads Job timestamps the runner
# already records (submitted/started/finished).
# --------------------------------------------------------------------------
@dataclass
class Monitor:
    runner: LocalThreadRunner
    interval: float = 0.5
    max_concurrent: int = 0
    peak_queue_depth: int = 0
    loadavg_samples: list[float] = field(default_factory=list)
    mem_avail_min_mb: float = float("inf")
    _stop: threading.Event = field(default_factory=threading.Event)

    def _sample(self) -> None:
        jobs = self.runner.list()
        running = sum(1 for j in jobs if j.status == JobStatus.RUNNING)
        queued = sum(1 for j in jobs if j.status == JobStatus.QUEUED)
        self.max_concurrent = max(self.max_concurrent, running)
        self.peak_queue_depth = max(self.peak_queue_depth, queued)
        try:
            self.loadavg_samples.append(os.getloadavg()[0])
        except (OSError, AttributeError):
            pass
        self.mem_avail_min_mb = min(self.mem_avail_min_mb, _mem_available_mb())

    def run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)
        self._sample()   # one final reading

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()


def _mem_available_mb() -> float:
    """MemAvailable from /proc (Linux); +inf if unreadable, so it never fails the
    min()."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("inf")


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def _wait_all_terminal(runner: LocalThreadRunner, jids: list[str],
                       timeout_s: float, poll: float, verbose: bool) -> bool:
    """Block until every job id is terminal or the timeout trips. Returns True if
    all reached a terminal state (liveness), False on timeout (a wedge)."""
    deadline = time.time() + timeout_s
    terminal = {JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED}
    last_line = ""
    while time.time() < deadline:
        jobs = [runner.get(j) for j in jids]
        done = sum(1 for j in jobs if j.status in terminal)
        running = [j for j in jobs if j.status == JobStatus.RUNNING]
        if verbose:
            cur = running[0] if running else None
            line = (f"  {done}/{len(jids)} terminal | "
                    f"running={cur.spec.name if cur else '-'} "
                    f"b={cur.batches_done if cur else 0} "
                    f"queued={sum(1 for j in jobs if j.status == JobStatus.QUEUED)}")
            if line != last_line:
                print(line, flush=True)
                last_line = line
        if done == len(jids):
            return True
        time.sleep(poll)
    return False


def _cancel_test(runner: LocalThreadRunner, poll: float, verbose: bool) -> dict:
    """Submit two jobs; cancel the first once it is RUNNING; confirm it reaches
    CANCELLED and that the SECOND still completes (slot released cleanly). This is
    the 'one abnormal job must not wedge the queue' contract."""
    spec_a = HabitatSpec(name="cancel_victim", shape="dome", inner_radius_cm=300.0,
                         walls=[WallLayer("aluminium", 5.0), WallLayer("regolith", 40.0)])
    spec_b = HabitatSpec(name="cancel_follower", shape="dome", inner_radius_cm=300.0,
                         walls=[WallLayer("aluminium", 5.0), WallLayer("regolith", 40.0)])
    ja = runner.submit(spec_a, tier=QUICK_LOOK, target_rel_err=0.5, max_batches=8,
                       composition=False)
    jb = runner.submit(spec_b, tier=QUICK_LOOK, target_rel_err=0.5, max_batches=2,
                       composition=False)
    # wait for the victim to actually start, then cancel it
    started = False
    for _ in range(120):
        if runner.get(ja).status == JobStatus.RUNNING:
            started = True
            break
        time.sleep(poll)
    cancelled_accepted = runner.cancel(ja) if started else False
    if verbose:
        print(f"  [cancel-test] victim started={started} cancel_accepted="
              f"{cancelled_accepted}", flush=True)
    # both must reach terminal; victim CANCELLED, follower DONE
    ok_all = _wait_all_terminal(runner, [ja, jb], timeout_s=1800, poll=poll,
                                verbose=verbose)
    va, vb = runner.get(ja), runner.get(jb)
    result = {
        "victim_started": started,
        "cancel_accepted": cancelled_accepted,
        "victim_status": va.status.value,
        "follower_status": vb.status.value,
        "all_terminal": ok_all,
        # PASS: victim cancelled cleanly AND the follower still completed -> the
        # semaphore slot was released by the abnormal exit.
        "pass": (va.status == JobStatus.CANCELLED and vb.status == JobStatus.DONE
                 and ok_all),
    }
    return result


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=6, help="number of designs to submit")
    p.add_argument("--parallel", type=int, default=1,
                   help="runner max_parallel (production=1; >1 explores concurrency)")
    p.add_argument("--tier", choices=["quick", "full"], default="quick",
                   help="statistics tier per batch (quick=~minutes, full=workshop)")
    p.add_argument("--max-batches", type=int, default=2,
                   help="convergence batch cap (2 = fast smoke; 8 = real soak)")
    p.add_argument("--target-rel-err", type=float, default=0.5,
                   help="convergence target (loose default so smoke stops at cap)")
    p.add_argument("--cancel-test", action="store_true",
                   help="also run the cancel-under-load isolation test")
    p.add_argument("--poll", type=float, default=2.0, help="status poll interval (s)")
    p.add_argument("--per-job-budget", type=float, default=1800.0,
                   help="seconds per job before the liveness timeout trips")
    p.add_argument("--json", default=None, help="write the full report as JSON here")
    p.add_argument("-q", "--quiet", action="store_true", help="less live output")
    args = p.parse_args()

    tier = QUICK_LOOK if args.tier == "quick" else FULL_RUN
    verbose = not args.quiet
    runner = LocalThreadRunner(max_parallel=args.parallel)
    workload = _build_workload(args.jobs)

    print(f"[stress] host cores={os.cpu_count()}  parallel={args.parallel}  "
          f"tier={args.tier}({tier.total_primaries} prim/batch)  "
          f"jobs={args.jobs}  max_batches={args.max_batches}", flush=True)
    print(f"[stress] LUNARSIM_THREADS={os.environ.get('LUNARSIM_THREADS', 'unset')}  "
          f"G4DATA={os.environ.get('TOPAS_G4_DATA_DIR')}", flush=True)

    mon = Monitor(runner)
    mon.start()
    t0 = time.time()

    jids = []
    for kind, spec, kw in workload:
        jid = runner.submit(spec, tier=tier, target_rel_err=args.target_rel_err,
                            max_batches=args.max_batches, **kw)
        jids.append(jid)
    if verbose:
        print(f"[stress] submitted {len(jids)} jobs, draining queue...", flush=True)

    timeout = args.per_job_budget * args.jobs / max(1, args.parallel) + 120.0
    all_terminal = _wait_all_terminal(runner, jids, timeout_s=timeout,
                                      poll=args.poll, verbose=verbose)

    cancel_result = None
    if args.cancel_test:
        if verbose:
            print("[stress] running cancel-under-load isolation test...", flush=True)
        cancel_result = _cancel_test(runner, poll=args.poll, verbose=verbose)

    mon.stop()
    total_wall = time.time() - t0

    # -- gather per-job outcomes --------------------------------------------
    rows = []
    for (kind, spec, _kw), jid in zip(workload, jids):
        j = runner.get(jid)
        queue_wait = ((j.started_at or j.finished_at or time.time()) - j.submitted_at)
        run_s = ((j.finished_at or time.time()) - (j.started_at or j.submitted_at))
        rows.append({
            "id": jid, "kind": kind, "shape": spec.shape, "name": spec.name,
            "status": j.status.value, "batches_done": j.batches_done,
            "queue_wait_s": round(queue_wait, 1), "run_s": round(run_s, 1),
            "error": (j.error or "")[:200] if j.status == JobStatus.ERROR else "",
        })

    run_times = [r["run_s"] for r in rows if r["status"] == "done"]
    n_ok = sum(1 for r in rows if r["status"] == "done")
    n_err = sum(1 for r in rows if r["status"] == "error")
    n_cancel = sum(1 for r in rows if r["status"] == "cancelled")
    throughput_hr = (n_ok / total_wall * 3600.0) if total_wall > 0 else 0.0

    # -- invariants ----------------------------------------------------------
    inv = {
        # serialisation: never more concurrent runs than the semaphore allows
        "serialisation_ok": mon.max_concurrent <= args.parallel,
        # liveness: nothing stuck non-terminal at timeout
        "liveness_no_wedge_ok": all_terminal,
        # no job crashed the runner (errors are allowed to EXIST but must be
        # captured, not hang -- liveness already covers the hang case)
        "all_captured_ok": all(r["status"] in ("done", "error", "cancelled")
                               for r in rows),
    }
    if cancel_result is not None:
        inv["cancellation_ok"] = cancel_result["pass"]
    passed = all(inv.values())

    report = {
        "config": {
            "jobs": args.jobs, "parallel": args.parallel, "tier": args.tier,
            "primaries_per_batch": tier.total_primaries,
            "max_batches": args.max_batches, "target_rel_err": args.target_rel_err,
            "cancel_test": args.cancel_test,
        },
        "host": {
            "cpu_count": os.cpu_count(),
            "lunarsim_threads": os.environ.get("LUNARSIM_THREADS", "unset"),
        },
        "jobs": rows,
        "aggregate": {
            "total_wall_s": round(total_wall, 1),
            "n_ok": n_ok, "n_error": n_err, "n_cancelled": n_cancel,
            "max_concurrent_observed": mon.max_concurrent,
            "peak_queue_depth": mon.peak_queue_depth,
            "throughput_jobs_per_hr": round(throughput_hr, 2),
            "projected_jobs_per_8h_day": round(throughput_hr * 8, 1),
            "run_s_mean": round(statistics.fmean(run_times), 1) if run_times else None,
            "run_s_median": round(statistics.median(run_times), 1) if run_times else None,
            "run_s_max": round(max(run_times), 1) if run_times else None,
        },
        "resources": {
            "loadavg_1min_max": round(max(mon.loadavg_samples), 2) if mon.loadavg_samples else None,
            "mem_available_min_mb": (round(mon.mem_avail_min_mb, 0)
                                     if mon.mem_avail_min_mb != float("inf") else None),
        },
        "cancellation": cancel_result,
        "invariants": inv,
        "pass": passed,
    }

    _print_report(report)
    if args.json:
        Path(args.json).expanduser().write_text(json.dumps(report, indent=2))
        print(f"\n[stress] wrote JSON report -> {args.json}", flush=True)

    return 0 if passed else 1


def _print_report(rep: dict) -> None:
    agg, inv = rep["aggregate"], rep["invariants"]
    print("\n" + "=" * 72)
    print("QUEUE STRESS REPORT")
    print("=" * 72)
    print(f"{'job':<26}{'kind':<8}{'status':<11}{'queue_s':>9}{'run_s':>9}{'batches':>9}")
    print("-" * 72)
    for r in rep["jobs"]:
        print(f"{r['name'][:25]:<26}{r['kind']:<8}{r['status']:<11}"
              f"{r['queue_wait_s']:>9}{r['run_s']:>9}{r['batches_done']:>9}")
        if r["error"]:
            print(f"    ERROR: {r['error']}")
    print("-" * 72)
    print(f"total wall            : {agg['total_wall_s']} s")
    print(f"ok / error / cancelled: {agg['n_ok']} / {agg['n_error']} / {agg['n_cancelled']}")
    print(f"max concurrent RUNNING: {agg['max_concurrent_observed']} "
          f"(cap {rep['config']['parallel']})")
    print(f"peak queue depth      : {agg['peak_queue_depth']}")
    print(f"run time mean/med/max : {agg['run_s_mean']} / {agg['run_s_median']} "
          f"/ {agg['run_s_max']} s")
    print(f"throughput            : {agg['throughput_jobs_per_hr']} jobs/hr "
          f"(~{agg['projected_jobs_per_8h_day']} per 8h day)")
    res = rep["resources"]
    print(f"loadavg max / mem min : {res['loadavg_1min_max']} / "
          f"{res['mem_available_min_mb']} MB avail")
    if rep["cancellation"] is not None:
        c = rep["cancellation"]
        print(f"cancel test           : victim={c['victim_status']} "
              f"follower={c['follower_status']} -> {'PASS' if c['pass'] else 'FAIL'}")
    print("-" * 72)
    for k, v in inv.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 72)
    print(f"OVERALL: {'PASS' if rep['pass'] else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
