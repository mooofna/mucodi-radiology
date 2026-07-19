"""Central registry of dataset specs (each module defines a module-level ``SPEC``)."""
from __future__ import annotations

from dataprep.datasets import (
    coca,
    coca_gated,
    ctrate,
    dlcs24,
    lidc,
    lndb,
    midrc_ricord,
    mosmed,
    osic,
    radchestct,
    rspect,
    stoic2021,
)
from dataprep.datasets.spec import DatasetSpec

_MODULES = [
    coca, coca_gated, ctrate, dlcs24, lidc, lndb,
    midrc_ricord, mosmed, osic, radchestct, rspect, stoic2021,
]

REGISTRY: dict[str, DatasetSpec] = {m.SPEC.name: m.SPEC for m in _MODULES}


def get(name: str) -> DatasetSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}") from None
