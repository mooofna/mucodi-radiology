"""Unit tests for rate_eval.io.cache_meta."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rate_eval.io.cache_meta import (
    CACHE_META_FILENAME,
    CacheMeta,
    DatasetProvenance,
    PreprocessProvenance,
    SplitInfo,
    WrapperProvenance,
    check_dataset_config_drift,
    read_cache_meta,
    update_cache_meta_finish,
    write_cache_meta_start,
)


def test_minimal_cache_meta_roundtrip(tmp_path):
    meta = CacheMeta(
        wrapper=WrapperProvenance(name="ctclip_zero_shot"),
        dataset=DatasetProvenance(name="ctrate"),
    )
    meta.write(tmp_path)
    assert (tmp_path / CACHE_META_FILENAME).exists()
    reloaded = read_cache_meta(tmp_path)
    assert reloaded.wrapper.name == "ctclip_zero_shot"
    assert reloaded.dataset.name == "ctrate"
    assert reloaded.schema_version == "1.0"


def test_full_cache_meta_roundtrip(tmp_path):
    meta = CacheMeta(
        wrapper=WrapperProvenance(
            name="tangerine_vit",
            config_yaml_path="configs/model/tangerine_vit.yaml",
            config_yaml_sha256="0123abcd",
        ),
        dataset=DatasetProvenance(
            name="coca",
            config_yaml_path="configs/dataset/coca_for_tangerine.yaml",
            config_yaml_sha256="abcd0123",
            splits_used={
                "train": SplitInfo(path="/path/train.jsonl", sha256="aa", n_rows=120),
                "valid": SplitInfo(path="/path/dev.jsonl", sha256="bb", n_rows=20),
            },
        ),
        preprocess=PreprocessProvenance(
            pipeline="interp_clip_norm",
            target_shape=[256, 256, 256],
            hu_clip=[-1200.0, 800.0],
            output_norm="minmax",
        ),
        notes="written by Phase 5 hook",
    )
    meta.write(tmp_path)
    reloaded = read_cache_meta(tmp_path)
    assert reloaded.preprocess.target_shape == [256, 256, 256]
    assert reloaded.dataset.splits_used["valid"].n_rows == 20


def test_read_cache_meta_returns_none_when_absent(tmp_path):
    assert read_cache_meta(tmp_path) is None


def test_write_then_update_with_sample_counts(tmp_path):
    """The two-step hook: write at start, update with counts at finish."""
    cache = tmp_path / "wrapper_dataset"
    cache.mkdir()
    write_cache_meta_start(
        cache,
        wrapper_name="curia2",
        dataset_name="rspect",
        rate_eval_version="2.2.0",
        num_gpus=4,
        batch_size_per_gpu=16,
        preprocess={"pipeline": "passthrough"},
    )
    meta = read_cache_meta(cache)
    assert meta.extraction.started_utc is not None
    assert meta.extraction.finished_utc is None
    assert meta.extraction.num_samples == {}
    assert meta.preprocess.pipeline == "passthrough"

    update_cache_meta_finish(cache, n_samples_per_split={"train": 100, "test": 30})
    meta = read_cache_meta(cache)
    assert meta.extraction.finished_utc is not None
    assert meta.extraction.num_samples == {"train": 100, "test": 30}


def test_update_no_op_when_sidecar_absent(tmp_path):
    cache = tmp_path / "wrapper_dataset"
    cache.mkdir()
    out = update_cache_meta_finish(cache, n_samples_per_split={"train": 1})
    assert out is None


def test_drift_returns_none_when_sha_matches(tmp_path):
    cfg = tmp_path / "dataset.yaml"
    cfg.write_text("loader:\n  class: NiftiCTDataset\n")
    cache = tmp_path / "cache_cell"
    cache.mkdir()
    write_cache_meta_start(
        cache, wrapper_name="W", dataset_name="D", dataset_config_path=cfg
    )
    assert check_dataset_config_drift(cache, cfg) is None


def test_drift_returns_message_when_sha_differs(tmp_path):
    cfg = tmp_path / "dataset.yaml"
    cfg.write_text("loader:\n  class: NiftiCTDataset\n")
    cache = tmp_path / "cache_cell"
    cache.mkdir()
    write_cache_meta_start(
        cache, wrapper_name="W", dataset_name="D", dataset_config_path=cfg
    )
    cfg.write_text("loader:\n  class: NiftiCTDataset\n  source: nifti\n")
    msg = check_dataset_config_drift(cache, cfg)
    assert msg is not None
    assert "drift" in msg


def test_drift_returns_none_when_sidecar_absent(tmp_path):
    cfg = tmp_path / "dataset.yaml"
    cfg.write_text("loader:\n  class: NiftiCTDataset\n")
    cache = tmp_path / "legacy_cache"
    cache.mkdir()
    assert check_dataset_config_drift(cache, cfg) is None


def test_cache_meta_rejects_missing_required_wrapper_name():
    with pytest.raises(Exception):
        CacheMeta.model_validate(
            {"wrapper": {}, "dataset": {"name": "d"}}
        )


def test_extra_fields_allowed_at_top_level():
    """`extra='allow'` lets future fields land in legacy files without breaking parse."""
    meta = CacheMeta.model_validate(
        {
            "wrapper": {"name": "w"},
            "dataset": {"name": "d"},
            "future_field": {"experimental": True},
        }
    )
    payload = meta.model_dump()
    assert payload.get("future_field") == {"experimental": True}
