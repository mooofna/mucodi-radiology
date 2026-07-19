"""Per-volume CT-RATE metadata lookup (slope/intercept/spacing from the metadata CSV); module-cached."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class CTRateMeta:
    slope: float          # RescaleSlope
    intercept: float      # RescaleIntercept
    xy_spacing: float     # mm; assumes isotropic XY
    z_spacing: float      # mm


def _parse_xy(raw: str) -> float:
    """Parse the '[0.341, 0.341]' CSV cell to a single XY spacing in mm (isotropic XY)."""
    return float(raw[1:].split(",")[0])


@lru_cache(maxsize=8)
def load_metadata_lookup(csv_path: str) -> Dict[str, CTRateMeta]:
    """Load + parse a metadata CSV, keyed by VolumeName; cached by exact path string."""
    df = pd.read_csv(csv_path)
    out: Dict[str, CTRateMeta] = {}
    for _, row in df.iterrows():
        out[str(row["VolumeName"])] = CTRateMeta(
            slope=float(row["RescaleSlope"]),
            intercept=float(row["RescaleIntercept"]),
            xy_spacing=_parse_xy(str(row["XYSpacing"])),
            z_spacing=float(row["ZSpacing"]),
        )
    return out


def lookup_meta(csv_path: Path | str, volume_name: str) -> CTRateMeta:
    """Look up one volume's metadata; KeyError if missing."""
    table = load_metadata_lookup(str(csv_path))
    if volume_name not in table:
        raise KeyError(
            f"VolumeName {volume_name!r} not in {csv_path} ({len(table)} rows). "
            f"Check that the CT-RATE metadata CSV was staged (see the parent repo's dataprep stager).",
        )
    return table[volume_name]
