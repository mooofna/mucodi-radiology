"""Unit tests for rate_eval.config.dataset_schema.DatasetConfig."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from rate_eval.config.dataset_schema import DatasetConfig, LoaderSpec, PreprocessSpec


def test_legacy_yaml_parses_under_extra_allow():
    """A legacy YAML must parse -- extra fields preserved via extra='allow'."""
    raw = textwrap.dedent(
        """
        loader:
          class: CTRateForCTClip
        data:
          train_json: /path/train.jsonl
          valid_json: /path/dev.jsonl
          test_json: /path/test.jsonl
        img_paths_key: nii_path
        modality: chest_ct
        ctclip_upstream_dir: /opt/CT-CLIP   # extra-allow
        """
    )
    cfg = DatasetConfig.model_validate(yaml.safe_load(raw))
    assert cfg.loader.class_ == "CTRateForCTClip"
    assert cfg.loader.preprocess is None
    assert cfg.data.train_json == "/path/train.jsonl"
    assert cfg.img_paths_key == "nii_path"


def test_new_shape_with_preprocess_block_parses():
    """The new NiftiCTDataset shape parses fully -- preprocess block validated."""
    raw = textwrap.dedent(
        """
        loader:
          class: NiftiCTDataset
          source: nifti
          preprocess:
            pipeline: clip_interp_norm
            target_shape: [240, 480, 480]
            hu_clip: [-1000.0, 1000.0]
            output_norm: halfrange_centered
        data:
          train_json: /path/train.jsonl
          valid_json: /path/dev.jsonl
          test_json: /path/test.jsonl
        modality: chest_ct
        """
    )
    cfg = DatasetConfig.model_validate(yaml.safe_load(raw))
    assert cfg.loader.class_ == "NiftiCTDataset"
    assert cfg.loader.source == "nifti"
    assert cfg.loader.preprocess.pipeline == "clip_interp_norm"
    assert cfg.loader.preprocess.target_shape == [240, 480, 480]
    assert cfg.loader.preprocess.hu_clip == [-1000.0, 1000.0]
    assert cfg.loader.preprocess.output_norm == "halfrange_centered"


def test_npz_source_with_custom_key():
    """NPZ source loader accepts a custom `npz_key` (RAD-ChestCT uses 'ct')."""
    raw = textwrap.dedent(
        """
        loader:
          class: NiftiCTDataset
          source: npz
          npz_key: ct
          preprocess:
            pipeline: interp
            target_shape: [256, 256, 256]
        modality: chest_ct
        """
    )
    cfg = DatasetConfig.model_validate(yaml.safe_load(raw))
    assert cfg.loader.source == "npz"
    assert cfg.loader.npz_key == "ct"


def test_preprocess_spec_rejects_unknown_pipeline():
    with pytest.raises(Exception):
        PreprocessSpec.model_validate({"pipeline": "bogus_recipe"})


def test_preprocess_spec_allows_extras_post_phase_4x():
    """PreprocessSpec allows extras; recipe conformance is enforced later in __post_init__."""
    spec = PreprocessSpec.model_validate(
        {"pipeline": "interp", "target_shape": [256, 256, 256], "target_shapes": [128, 128, 128]}
    )
    payload = spec.model_dump()
    assert payload["target_shape"] == [256, 256, 256]
    assert payload["target_shapes"] == [128, 128, 128]


def test_loader_spec_accepts_legacy_aliases():
    """Legacy YAMLs use `loader.config` to point at a nested file; alias preserved."""
    cfg = LoaderSpec.model_validate({"class": "Legacy", "config": "/path/sub.yaml"})
    assert cfg.class_ == "Legacy"
    assert cfg.config == "/path/sub.yaml"


def test_all_existing_configs_parse():
    """All YAMLs under configs/dataset/ must parse via DatasetConfig (no drift breaks the canonical shape)."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    configs_dir = project_root / "configs" / "dataset"
    if not configs_dir.is_dir():
        pytest.skip(f"configs/dataset/ not present at {configs_dir}")

    yamls = sorted(configs_dir.glob("*.yaml"))
    if not yamls:
        pytest.skip("no YAMLs under configs/dataset/")

    failed = []
    for yml in yamls:
        try:
            raw = yaml.safe_load(yml.read_text())
            if raw is None or "loader" not in raw:
                # skip non-dataset YAMLs
                continue
            DatasetConfig.model_validate(raw)
        except Exception as exc:
            failed.append((yml.name, str(exc)))

    if failed:
        msg = "\n".join(f"  - {name}: {err}" for name, err in failed)
        pytest.fail(f"{len(failed)} YAML(s) failed schema validation:\n{msg}")
