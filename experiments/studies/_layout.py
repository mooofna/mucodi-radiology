"""Per-cohort output layout <RUNS_OUT>/<slug>/<cohort>/{cache,<variant>}; key on per-model slug not wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class CohortPaths(NamedTuple):
    cache: Path
    cohort_dir: Path

    def variant(self, slug: str) -> Path:
        """Output dir for one probe variant (the unique cell key)."""
        return self.cohort_dir / slug


def cohort_paths(runs_out: Path, slug: str, cohort: str) -> CohortPaths:
    """Shared cache + cohort dir for one model x cohort (``slug`` = per-model id, not wrapper)."""
    cohort_dir = runs_out / slug / cohort
    return CohortPaths(cache=cohort_dir / "cache", cohort_dir=cohort_dir)
