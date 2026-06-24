"""lunarsim -- parametric lunar-habitat radiation evaluation, TOPAS in the loop.

Pipeline:  HabitatSpec (spec) -> geometry -> bridge (assemble/run/parse) -> GUI.
"""
from .spec import HabitatSpec, WallLayer, MATERIALS, SHAPES, default_spec
from .geometry import build_geometry, build_scorers
from .bridge import (
    RunTier, RunResult, QUICK_LOOK, FULL_RUN, VIS_TIER, TIERS,
    build_parameter_file, run_design, write_vis_run,
)
from .dosimetry import (
    DoseAssessment, assess, gcr_scalar_fluence_rate,
    DOSE_LIMITS_MSV, DEFAULT_QUALITY_FACTOR,
)
from .jobs import (
    Job, JobStatus, JobRunner, LocalThreadRunner, ConvergedResult,
    run_converged, default_runner,
)
from .trajviz import (
    run_cascade, build_cascade_run, load_tracks, build_figure, write_html,
)

__all__ = [
    "HabitatSpec", "WallLayer", "MATERIALS", "SHAPES", "default_spec",
    "build_geometry", "build_scorers",
    "RunTier", "RunResult", "QUICK_LOOK", "FULL_RUN", "VIS_TIER", "TIERS",
    "build_parameter_file", "run_design", "write_vis_run",
    "DoseAssessment", "assess", "gcr_scalar_fluence_rate",
    "DOSE_LIMITS_MSV", "DEFAULT_QUALITY_FACTOR",
    "Job", "JobStatus", "JobRunner", "LocalThreadRunner", "ConvergedResult",
    "run_converged", "default_runner",
]
