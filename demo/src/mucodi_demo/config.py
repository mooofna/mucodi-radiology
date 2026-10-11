"""Machine-specific paths, read from ``configs/local.toml`` (not tracked).

Everything the pipeline writes goes below ``work_dir`` (default ``demo/.local``), so nothing heavy
ever lands in git.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = DEMO_ROOT / "configs" / "local.toml"


@dataclass(frozen=True)
class Paths:
    dropbox_runs: Path  # the Dropbox share's runs/ folder
    thesis_code: Path  # the thesis workspace's code/ folder (result CSV and JSON files)
    thesis_tables: Path  # tidy per-table CSVs exported from the thesis
    work_dir: Path

    @property
    def checkpoints(self) -> Path:
        return self.work_dir / "checkpoints"

    @property
    def assets(self) -> Path:
        return self.work_dir / "assets"

    @property
    def downloads(self) -> Path:
        return self.work_dir / "downloads"


def load_paths(config: Path = LOCAL_CONFIG) -> Paths:
    if not config.exists():
        raise FileNotFoundError(
            f"{config} is missing; copy configs/local.example.toml and fill in the paths"
        )
    raw = tomllib.loads(config.read_text())["paths"]

    def p(key: str, default: str | None = None) -> Path:
        value = raw.get(key, default)
        if value is None:
            raise KeyError(f"[paths].{key} is required in {config}")
        path = Path(value).expanduser()
        return path if path.is_absolute() else (DEMO_ROOT / path)

    return Paths(
        dropbox_runs=p("dropbox_runs"),
        thesis_code=p("thesis_code"),
        thesis_tables=p("thesis_tables"),
        work_dir=p("work_dir", ".local"),
    )
