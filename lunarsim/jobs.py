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
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Optional

from .spec import HabitatSpec
from .bridge import RunTier, RunResult, QUICK_LOOK, run_design
from .dosimetry import assess_composition, GCR_COMPOSITION


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
    phantom_doseeq_sv: Optional[float] = None   # central phantom, LET-weighted (ICRP-60 Q)
    skin_dose_gy: Optional[float] = None        # habitat-wide inner-wall lining
    skin_doseeq_sv: Optional[float] = None      # same lining, LET-weighted (ICRP-60 Q)
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
    phaneq = [b.phantom_doseeq_sv for b in batches if b.phantom_doseeq_sv is not None]
    skin = [b.skin_dose_gy for b in batches if b.skin_dose_gy is not None]
    skineq = [b.skin_doseeq_sv for b in batches if b.skin_doseeq_sv is not None]
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
        phantom_doseeq_sv=statistics.fmean(phaneq) if phaneq else None,
        skin_dose_gy=skin_mean, skin_dose_rel_err=skin_rel,
        skin_doseeq_sv=statistics.fmean(skineq) if skineq else None,
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
    """Pick the relative error the convergence loop watches.

    "skin"    the habitat-wide inner-wall lining -- the workshop headline score.
    "phantom" the central crew point dose -- a much noisier diagnostic.
    "both"    require BOTH below target; the binding (larger) error drives the
              loop, so neither the headline nor the phantom is reported converged
              while it is still imprecise. Returns None until both have an error
              estimate (>=2 batches), so the loop never stops on a half-measured
              pair."""
    if converge_on == "skin":
        return combined.skin_dose_rel_err
    if converge_on == "phantom":
        return combined.dose_rel_err
    s, p = combined.skin_dose_rel_err, combined.dose_rel_err
    if s is None or p is None:
        return None
    return max(s, p)


def run_converged(spec: HabitatSpec, tier: RunTier = QUICK_LOOK,
                  target_rel_err: float = 0.10,
                  min_batches: int = 2, max_batches: int = 8,
                  converge_on: str = "phantom",
                  particle: str = "proton", ion_z: int = 1, ion_a: int = 1,
                  progress_cb: Optional[Callable[[int, int, Optional[float]], None]] = None,
                  cancel_cb: Optional[Callable[[], bool]] = None) -> ConvergedResult:
    """Run independent-seed batches until the chosen dose relative error <= target.

    converge_on selects which quantity must converge: "phantom" (central crew
    point dose, default) or "skin" (habitat-wide inner-wall lining -- the
    workshop score). particle/ion_z/ion_a select the GCR species (default
    protons). progress_cb(done, cap, rel_err) reports that same quantity.
    cancel_cb() -> True stops cleanly between batches."""
    batches: list[RunResult] = []
    for i in range(max_batches):
        if cancel_cb and cancel_cb():
            break
        res = run_design(spec, tier, seed=i + 1, keep=False,
                         particle=particle, ion_z=ion_z, ion_a=ion_a)
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
# Multi-species GCR composition (protons + heavy ions, summed)
# ----------------------------------------------------------------------
# Heavy ions (alpha + HZE) carry a large share of the GCR dose despite being a
# small fraction by number, so a proton-only run underestimates the absolute
# scale. We transport each species in its OWN converged run (so the wall's
# species-dependent shielding/fragmentation is real) and sum the per-species
# normalised dose rates in dosimetry.assess_composition.
#
# HISTORICAL: minor HZE species (C/Si/Fe) used to be capped at 2 batches
# (_MINOR_MAX_BATCHES below) to save wall-time, on the grounds that they add
# little to the SKIN dose-equivalent. That cap was lifted (run_composition now
# gives every species max_batches): the SOLID central phantom catches rare heavy-
# ion Bragg-peak stops, so its per-species dose has enormous variance and only
# converges with a full batch budget. Capping it at 2 batches left the crew point
# dose dominated by one or two stochastic stops and reported a misleadingly small
# combined error. The constants are retained for reference / a future cheap tier.
_DOMINANT_ABUNDANCE = 0.05      # (retired) abundance threshold for the full budget
_MINOR_MAX_BATCHES = 2          # (retired) former cap for the rare HZE species

# Per-primary TOPAS cost scales ~ with total beam energy = T_per_nuc * A, so a
# heavy ion is ~A times costlier than a proton of the same per-nucleon energy.
# Heavy species are also MINOR dose contributors (C ~6%, Si ~2%, Fe ~0.4%), so
# their statistical noise barely moves the summed score (a 2% contributor at 20%
# rel-err adds 0.4% to the combined error). We therefore scale each species'
# histories by ~1/A (referenced to He, A=4, so the dominant H/He keep full
# statistics), which equalises per-batch wall-time across species instead of
# letting Si/Fe batches run ~A times longer. Floored so the rarest ions still
# get a usable sample.
_COST_REF_A = 4                 # reference mass (He) for equal-cost history scaling
_MIN_HISTORIES = 8              # floor so heavy ions keep a usable sample


def _species_tier(tier: RunTier, a: int) -> RunTier:
    """Clone `tier` with histories scaled ~1/A so each species' batch costs about
    the same wall-time. Ions up to He (A<=4) are unscaled."""
    scale = min(1.0, _COST_REF_A / a)
    hist = max(_MIN_HISTORIES, round(tier.histories * scale))
    return replace(tier, histories=hist)


@dataclass
class ConvergedComposition:
    """Batch-combined estimate over a GCR composition. Holds one ConvergedResult
    per species and the combined statistics. Duck-types the attributes the GUI
    reads off a ConvergedResult; the dose itself is produced by
    dosimetry.assess_composition(self.species_results, ...)."""
    spec: HabitatSpec
    tier: RunTier
    species_results: list           # list[(species_tuple, ConvergedResult)]
    n_batches: int = 0
    total_primaries: int = 0
    wall_seconds: float = 0.0
    dose_rel_err: Optional[float] = None        # combined (central phantom)
    skin_dose_rel_err: Optional[float] = None   # combined (habitat-wide skin)
    returncode: int = 0
    log_tail: str = ""

    @property
    def ok(self) -> bool:
        return (self.returncode == 0 and bool(self.species_results)
                and all(r.ok for _, r in self.species_results))

    @property
    def transmission(self) -> Optional[float]:
        # report the proton (first / dominant) species as representative
        return self.species_results[0][1].transmission if self.species_results else None


def run_composition(spec: HabitatSpec, tier: RunTier = QUICK_LOOK,
                    target_rel_err: float = 0.05,
                    min_batches: int = 2, max_batches: int = 12,
                    converge_on: str = "both",
                    composition: list = GCR_COMPOSITION,
                    progress_cb: Optional[Callable[[int, int, Optional[float]], None]] = None,
                    cancel_cb: Optional[Callable[[], bool]] = None) -> ConvergedComposition:
    """Run one converged simulation per GCR species and combine them.

    Each species is transported with its own ion/spectrum and converged on
    `converge_on` (default "both": the habitat-wide skin lining AND the central
    crew phantom). Every species gets the full max_batches: the old HZE cap was
    lifted because the solid central phantom only converges for heavy ions with a
    full batch budget -- with the cap the phantom point dose was dominated by a
    couple of under-sampled Bragg-peak stops and reported a misleadingly small
    error (see _MINOR_MAX_BATCHES note). progress_cb reports cumulative batches
    across all species against the total batch budget."""
    caps = [max_batches for _sp in composition]
    total_cap = sum(caps)
    species_results: list = []
    done_base = 0
    for sp, cap in zip(composition, caps):
        name, particle, z, a, abundance, group = sp
        if cancel_cb and cancel_cb():
            break

        def species_cb(done: int, _cap: int, rel: Optional[float],
                       _base=done_base) -> None:
            if progress_cb:
                progress_cb(_base + done, total_cap, rel)

        sp_tier = _species_tier(tier, a)   # scale histories ~1/A to bound batch cost
        cr = run_converged(spec, sp_tier, target_rel_err=target_rel_err,
                           min_batches=min_batches, max_batches=cap,
                           converge_on=converge_on,
                           particle=particle, ion_z=z, ion_a=a,
                           progress_cb=species_cb, cancel_cb=cancel_cb)
        species_results.append((sp, cr))
        done_base += cap
        if not cr.ok:
            # a species failed -- surface immediately rather than a partial sum
            comp = ConvergedComposition(spec=spec, tier=tier,
                                        species_results=species_results,
                                        returncode=cr.returncode or 1,
                                        log_tail=cr.log_tail)
            return comp

    comp = ConvergedComposition(
        spec=spec, tier=tier, species_results=species_results,
        n_batches=sum(r.n_batches for _, r in species_results),
        total_primaries=sum(r.total_primaries for _, r in species_results),
        wall_seconds=sum(r.wall_seconds for _, r in species_results),
        returncode=0 if species_results and all(r.returncode == 0 for _, r in species_results) else 1,
        log_tail=species_results[-1][1].log_tail if species_results else "",
    )
    # combined statistical error of the summed dose (flux-weighted, in quadrature)
    a_skin = assess_composition(species_results, phi_MV=tier.phi_mv, skin=True)
    a_phan = assess_composition(species_results, phi_MV=tier.phi_mv, skin=False)
    comp.skin_dose_rel_err = a_skin.rel_err if a_skin else None
    comp.dose_rel_err = a_phan.rel_err if a_phan else None
    return comp


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
    composition: bool = False
    status: JobStatus = JobStatus.QUEUED
    batches_done: int = 0
    progress_cap: Optional[int] = None      # progress denominator (composition: summed)
    rel_err: Optional[float] = None
    result: Optional[object] = None         # ConvergedResult or ConvergedComposition
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
        denom = self.progress_cap or self.max_batches
        if denom <= 0:
            return 0.0
        return min(self.batches_done / denom, 0.99)

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
               converge_on: str = "phantom", composition: bool = False) -> str: ...

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
               converge_on: str = "phantom", composition: bool = False) -> str:
        spec.validate()
        job = Job(id=uuid.uuid4().hex[:8], spec=spec, tier=tier,
                  target_rel_err=target_rel_err, max_batches=max_batches,
                  converge_on=converge_on, composition=composition)
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
                    job.progress_cap = cap
                    job.rel_err = rel

            try:
                if job.composition:
                    result = run_composition(
                        job.spec, job.tier,
                        target_rel_err=job.target_rel_err,
                        max_batches=job.max_batches,
                        converge_on=job.converge_on,
                        progress_cb=progress_cb,
                        cancel_cb=lambda: job._cancel,
                    )
                else:
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
