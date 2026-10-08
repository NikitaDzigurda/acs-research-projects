"""Repository paths and a guard keeping patient-level outputs out of Git-visible trees."""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class StudyPaths:
    root: Path
    study: Path
    inputs: Path
    reference: Path
    run: Path
    statistics: Path


def study_paths(root: Path) -> StudyPaths:
    root = Path(root).resolve()
    study = root / "studies/acs"
    private = study / "artifacts/private"
    raw_run = os.environ.get("ACS_RUN_DIR", "runs/current")
    run = Path(raw_run)
    if not run.is_absolute():
        run = private / run
    run = run.resolve()
    if not run.is_relative_to(private.resolve()):
        raise ValueError("ACS_RUN_DIR must stay inside studies/acs/artifacts/private")
    return StudyPaths(root, study, study / "data/private/inputs",
                      private / "reference", run, private / "statistics")
