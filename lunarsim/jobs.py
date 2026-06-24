"""Job runner: non-blocking, converge-by-error evaluation of habitat designs.

The GUI must stay responsive while TOPAS grinds, and a single short run is too
noisy to put a safety verdict on. This module solves both:

  * Each submitted design runs in a background thread (the GUI polls a Job).
  * Statistics are built by *batching*: independent-seed runs are accumulated and
    the dose mean +/- standard error tracked after each batch. Batches stop when
    the relative error drops below a target (converged) or a batch cap is hit.
    This is the cheap, embarrassingly-parallel route to a trustworthy number and
    gives a natural progress signal (batches done / cap).

`JobRunner` is an abstract interface; `LocalThreadRunner` is the in-process
implementation (thread pool + lock). A future shared-server deployment can
implement the same interface over a real queue (Celery/RQ) without touching the
GUI or the dosimetry.
"""
from __future__ import annotations

import statistics
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .spec import HabitatSpec
from .bridge import RunTier, RunResult, QUICK_LOOK, run_design


# ----------------------------------------------------------------------
# Converged result (duck-types as a RunResult for dosimetry.assess)
# ----------------------------------------------------------------------
@dataclass
class ConvergedResult:
    """Batch-combined estimate. Exposes .dose_gy / .fluence_outside / .tier so it
    drops straight into dosimetry.assess() like a single RunResult."""
    spec: HabitatSpec
    tier: RunTier
    n_batches: int
    total_primaries: int
    wall_seconds: float
    dose_gy: Optional[float] = None
    skin_dose_gy: Optional[float] = None        # habitat-wide inner-wall lining
    dose_rel_err: Optional[float] = None        # standard error / mean
    skin_dose_rel_err: Optional[float] = None   # standard error / mean (skin)
    fluence_inside: Optional[float] = None
    fluence_outside: Optional[float] = None
    per_batch_dose: list[float] = field(default_factory=list)
    returncode: int = 0
    log_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.dose_gy is not None

    @property
    def transmission(self) -> Optional[float]:
        if self.fluence_inside is None or not self.fluence_outside:
            return None
        return self.fluence_inside / self.fluence_outside


def _combine(spec: HabitatSpec, tier: RunTier,
             batches: list[RunResult]) -> ConvergedResult:
    doses = [b.dose_gy for b in batches if b.dose_gy is not None]
    skin = [b.skin_dose_gy for b in batches if b.skin_dose_gy is not None]
    fin = [b.fluence_inside for b in batches if b.fluence_inside is not None]
    fout = [b.fluence_outside for b in batches if b.fluence_outside is not None]
    n = len(doses)
    mean = statistics.fmean(doses) if doses else None
    rel = None
    if n >= 2 and mean:
        sem = statistics.stdev(doses) / (n ** 0.5)     # standard error of mean
        rel = sem / mean
    skin_mean = statistics.fmean(skin) if skin else None
    skin_rel = None
    if len(skin) >= 2 and skin_mean:
        skin_rel = statistics.stdev(skin) / (len(skin) ** 0.5) / skin_mean
    return ConvergedResult(
        spec=spec, tier=tier, n_batches=len(batches),
        total_primaries=tier.total_primaries * len(batches),
        wall_seconds=sum(b.wall_seconds for b in batches),
        dose_gy=mean, dose_rel_err=rel,
        skin_dose_gy=skin_mean, skin_dose_rel_err=skin_rel,
        fluence_inside=statistics.fmean(fin) if fin else None,
        fluence_outside=statistics.fmean(fout) if fout else None,
        per_batch_dose=doses,
        returncode=0 if batches and all(b.returncode == 0 for b in batches) else 1,
        log_tail=batches[-1].log_tail if batches else "",
    )


# ----------------------------------------------------------------------
# Converge-by-error driver (blocking; the runner calls this in a thread)
# ----------------------------------------------------------------------
def _rel_err_of(combined: ConvergedResult, converge_on: str) -> Optional[float]:
    """Pick the relative error the convergence loop watches. The workshop score
    is the habitat-wide skin dose, so converging on 'skin' tightens the quantity
    we actually display -- the central phantom is a much noisier diagnostic and
    converging on it wastes batches while leaving the headline imprecise."""
    return (combined.skin_dose_rel_err if converge_on == "skin"
            else combined.dose_rel_err)


def run_converged(spec: HabitatSpec, tier: RunTier = QUICK_LOOK,
                  target_rel_err: float = 0.10,
                  min_batches: int = 2, max_batches: int = 8,
                  converge_on: str = "phantom",
                  progress_cb: Optional[Callable[[int, int, Optional[float]], None]] = None,
                  cancel_cb: Optional[Callable[[], bool]] = None) -> ConvergedResult:
    """Run independent-seed batches until the chosen dose relative error <= target.

    converge_on selects which quantity must converge: "phantom" (central crew
    point dose, default) or "skin" (habitat-wide inner-wall lining -- the
    workshop score). progress_cb(done, cap, rel_err) reports that same quantity.
    cancel_cb() -> True stops cleanly between batches."""
    batches: list[RunResult] = []
    for i in range(max_batches):
        if cancel_cb and cancel_cb():
            break
        res = run_design(spec, tier, seed=i + 1, keep=False)
        if not res.ok:
            # surface the failure immediately rather than averaging garbage
            cr = _combine(spec, tier, batches)
            cr.returncode = res.returncode or 1
            cr.log_tail = res.log_tail
            return cr
        batches.append(res)
        combined = _combine(spec, tier, batches)
        rel = _rel_err_of(combined, converge_on)
        if progress_cb:
            progress_cb(len(batches), max_batches, rel)
        if len(batches) >= min_batches and rel is not None \
                and rel <= target_rel_err:
            break
    return _combine(spec, tier, batches)


# ----------------------------------------------------------------------
# Job state
# ----------------------------------------------------------------------
class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    spec: HabitatSpec
    tier: RunTier
    target_rel_err: float
    max_batches: int
    converge_on: str = "phantom"
    status: JobStatus = JobStatus.QUEUED
    batches_done: int = 0
    rel_err: Optional[float] = None
    result: Optional[ConvergedResult] = None
    error: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    _cancel: bool = False

    @property
    def progress(self) -> float:
        """0..1. Reaches 1 when finished; otherwise batches/cap."""
        if self.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            return 1.0
        if self.max_batches <= 0:
            return 0.0
        return min(self.batches_done / self.max_batches, 0.99)

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - (self.started_at or self.submitted_at)


# ----------------------------------------------------------------------
# Runner interface + local implementation
# ----------------------------------------------------------------------
class JobRunner(ABC):
    @abstractmethod
    def submit(self, spec: HabitatSpec, tier: RunTier = QUICK_LOOK,
               target_rel_err: float = 0.10, max_batches: int = 8,
               converge_on: str = "phantom") -> str: ...

    @abstractmethod
    def get(self, job_id: str) -> Optional[Job]: ...

    @abstractmethod
    def list(self) -> list[Job]: ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool: ...


class LocalThreadRunner(JobRunner):
    """In-process runner. A semaphore caps concurrent TOPAS runs because each
    run_design uses all CPU cores (threads=0); default 1 avoids oversubscription
    on a single workshop laptop. Raise max_parallel only on a fat box where you
    pin threads-per-run instead."""

    def __init__(self, max_parallel: int = 1):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max_parallel)

    # -- interface ----------------------------------------------------
    def submit(self, spec: HabitatSpec, tier: RunTier = QUICK_LOOK,
               target_rel_err: float = 0.10, max_batches: int = 8,
               converge_on: str = "phantom") -> str:
        spec.validate()
        job = Job(id=uuid.uuid4().hex[:8], spec=spec, tier=tier,
                  target_rel_err=target_rel_err, max_batches=max_batches,
                  converge_on=converge_on)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._worker, args=(job,), daemon=True).start()
        return job.id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.submitted_at)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                job._cancel = True
                return True
        return False

    # -- worker -------------------------------------------------------
    def _worker(self, job: Job) -> None:
        with self._slots:                       # blocks here while queued
            if job._cancel:
                self._finish(job, JobStatus.CANCELLED)
                return
            with self._lock:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()

            def progress_cb(done: int, cap: int, rel: Optional[float]) -> None:
                with self._lock:
                    job.batches_done = done
                    job.rel_err = rel

            try:
                result = run_converged(
                    job.spec, job.tier,
                    target_rel_err=job.target_rel_err,
                    max_batches=job.max_batches,
                    converge_on=job.converge_on,
                    progress_cb=progress_cb,
                    cancel_cb=lambda: job._cancel,
                )
            except Exception as exc:            # pragma: no cover - defensive
                with self._lock:
                    job.error = f"{type(exc).__name__}: {exc}"
                self._finish(job, JobStatus.ERROR)
                return

            if job._cancel and not result.ok:
                self._finish(job, JobStatus.CANCELLED)
                return
            if not result.ok:
                with self._lock:
                    job.error = result.log_tail or "TOPAS run failed"
                    job.result = result
                self._finish(job, JobStatus.ERROR)
                return
            with self._lock:
                job.result = result
                job.rel_err = _rel_err_of(result, job.converge_on)
            self._finish(job, JobStatus.DONE)

    def _finish(self, job: Job, status: JobStatus) -> None:
        with self._lock:
            job.status = status
            job.finished_at = time.time()


# module-level default runner the GUI can import directly
default_runner = LocalThreadRunner(max_parallel=1)


if __name__ == "__main__":
    import os
    os.environ.setdefault("TOPAS_G4_DATA_DIR", os.path.expanduser("~/G4Data"))
    from .spec import HabitatSpec, WallLayer
    from .dosimetry import assess

    runner = LocalThreadRunner()
    spec = HabitatSpec(name="job_test", shape="dome", inner_radius_cm=300.0,
                       walls=[WallLayer("aluminium", 2.0), WallLayer("regolith", 40.0)])
    jid = runner.submit(spec, target_rel_err=0.15, max_batches=4)
    print(f"submitted job {jid}")
    while True:
        job = runner.get(jid)
        print(f"  [{job.status.value}] batches={job.batches_done}/{job.max_batches} "
              f"rel_err={job.rel_err} progress={job.progress:.0%} elapsed={job.elapsed:.0f}s")
        if job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            break
        time.sleep(3)
    if job.result and job.result.ok:
        a = assess(job.result, mission_days=365)
        print(f"converged: dose={job.result.dose_gy:.3e} Gy "
              f"+/-{job.result.dose_rel_err:.0%} over {job.result.n_batches} batches "
              f"({job.result.total_primaries} primaries)")
        print(f"crew dose: {a.mission_msv:.0f} mSv/yr -> {a.verdict()}")
    else:
        print("job failed:", job.error)
