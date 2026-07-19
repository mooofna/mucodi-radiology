"""Disk I/O, provenance, and feature-cache loaders for rate_eval."""

from .cache import SimpleCheckpointManager, SimpleResumableDataset
from .cache_meta import (
    CACHE_META_FILENAME,
    CacheMeta,
    check_dataset_config_drift,
    read_cache_meta,
    update_cache_meta_finish,
    write_cache_meta_start,
)
from .download_lock import ModelDownloadLock
from .feature_loaders import (
    load_features_from_cache,
    load_features_from_cache_split,
)
from .result_schema import HeadSpec, MetricBlock, ResultSummary

__all__ = [
    "CACHE_META_FILENAME",
    "CacheMeta",
    "HeadSpec",
    "MetricBlock",
    "ModelDownloadLock",
    "ResultSummary",
    "SimpleCheckpointManager",
    "SimpleResumableDataset",
    "check_dataset_config_drift",
    "load_features_from_cache",
    "load_features_from_cache_split",
    "read_cache_meta",
    "update_cache_meta_finish",
    "write_cache_meta_start",
]
