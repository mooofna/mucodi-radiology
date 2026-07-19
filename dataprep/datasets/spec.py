"""``DatasetSpec`` -- metadata to re-obtain a dataset plus ``stage`` / ``build_labels`` hooks (either may be None)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class Access(str, Enum):
    HF = "hf"            # HuggingFace gated/public (HF_TOKEN)
    ZENODO = "zenodo"    # Zenodo record (often a share-link / access token)
    KAGGLE = "kaggle"    # Kaggle competition (kaggle credentials)
    REDIVIS = "redivis"  # Redivis API (REDIVIS_API_TOKEN)
    IDC = "idc"          # NCI Imaging Data Commons (idc-index)
    S3 = "s3"            # AWS S3 (signed or --no-sign-request open data)
    MANUAL = "manual"    # documented manual download; no programmatic stager
    DERIVED = "derived"  # no own imaging (annotation layer over another dataset)


# Role gates whether the dataset enters the leak-free scored mean.
Role = str  # "scored" | "kd-corpus" | "descriptive" | "anchor" | "candidate"


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata + curation hooks for one dataset."""

    name: str
    access: Access
    modality: str                       # e.g. "chest_ct", "head_ct"
    role: Role
    source: str                         # source id / repo / record / URL stem
    token_env: Optional[str] = None
    committed_outputs: tuple[str, ...] = ()  # repo-relative label/manifest paths, checked by --verify
    notes: str = ""
    stage: Optional[Callable[..., None]] = None
    build_labels: Optional[Callable[..., None]] = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        cred = f" [{self.token_env}]" if self.token_env else ""
        return f"{self.name} ({self.access.value}, {self.role}){cred}"
