"""Integration test: each NiftiCTDataset pipeline recipe byte-matches the legacy _preprocess_volume."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rate_eval.datasets.nifti import NiftiCTDataset


def _make_synth_nifti(out_path: Path, shape=(16, 32, 32)) -> None:
    """Write a synthetic NIfTI to `out_path` with deterministic HU-ish values."""
    data = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 5000.0)
    # nib.Nifti1Image expects (X, Y, Z); preprocess transpose(2,0,1) lands at (D=Z, H=X, W=Y)
    img = nib.Nifti1Image(data.transpose(1, 2, 0), affine=np.eye(4))
    nib.save(img, str(out_path))


def _make_jsonl(out_path: Path, nifti_path: Path) -> None:
    out_path.write_text(
        json.dumps({"sample_name": "synth_sample_0", "nii_path": str(nifti_path)}) + "\n"
    )


def _config(tmp_path: Path, *, pipeline: str, **preprocess_extra):
    nifti = tmp_path / "scan.nii.gz"
    _make_synth_nifti(nifti)
    jsonl = tmp_path / "split.jsonl"
    _make_jsonl(jsonl, nifti)
    preprocess = {"pipeline": pipeline, **preprocess_extra}
    return OmegaConf.create(
        {
            "loader": {
                "class": "NiftiCTDataset",
                "source": "nifti",
                "preprocess": preprocess,
            },
            "data": {
                "train_json": str(jsonl),
                "valid_json": str(jsonl),
                "test_json": str(jsonl),
            },
            "img_paths_key": "nii_path",
            "modality": "chest_ct",
        }
    )


def test_passthrough_pipeline_returns_native_shape(tmp_path):
    """Curia-2 recipe: load NIfTI -> (1, D, H, W) raw HU, no preprocessing."""
    cfg = _config(tmp_path, pipeline="passthrough")
    ds = NiftiCTDataset(cfg, split="valid")
    vol = ds._preprocess_volume(ds.samples[0]["image_path"])
    # synth NIfTI (32,32,16) -> read_nifti transposes to (16,32,32)
    assert vol.shape == (1, 16, 32, 32)
    assert vol.dtype == torch.float32


def test_interp_pipeline_matches_legacy_pillar0_op_order(tmp_path):
    """Pipeline 'interp' must byte-match the legacy COCA x Pillar-0 `_preprocess_volume` op order."""
    cfg = _config(tmp_path, pipeline="interp", target_shape=[8, 16, 16])
    ds = NiftiCTDataset(cfg, split="valid")
    out = ds._preprocess_volume(ds.samples[0]["image_path"])

    nii = nib.as_closest_canonical(nib.load(ds.samples[0]["image_path"]))
    data = nii.get_fdata().astype(np.float32).transpose(2, 0, 1)
    expected = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    expected = F.interpolate(expected, size=(8, 16, 16), mode="trilinear", align_corners=False)
    expected = expected.squeeze(0)
    assert torch.equal(out, expected)


def test_clip_interp_norm_matches_legacy_ctclip_op_order(tmp_path):
    """Pipeline 'clip_interp_norm' must byte-match the legacy COCA x CT-CLIP `_preprocess_volume` op order."""
    cfg = _config(
        tmp_path,
        pipeline="clip_interp_norm",
        target_shape=[8, 16, 16],
        hu_clip=[-1000.0, 1000.0],
        output_norm="halfrange_centered",
    )
    ds = NiftiCTDataset(cfg, split="valid")
    out = ds._preprocess_volume(ds.samples[0]["image_path"])

    nii = nib.as_closest_canonical(nib.load(ds.samples[0]["image_path"]))
    data = nii.get_fdata().astype(np.float32).transpose(2, 0, 1)
    expected = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    expected = expected.clamp(min=-1000.0, max=1000.0)
    expected = F.interpolate(expected, size=(8, 16, 16), mode="trilinear", align_corners=False)
    expected = expected.squeeze(0) / 1000.0
    assert torch.equal(out, expected)


def test_interp_clip_norm_matches_legacy_tangerine_op_order(tmp_path):
    """Pipeline 'interp_clip_norm' must byte-match the legacy COCA x TANGERINE `_preprocess_volume` op order."""
    cfg = _config(
        tmp_path,
        pipeline="interp_clip_norm",
        target_shape=[8, 16, 16],
        hu_clip=[-1200.0, 800.0],
        output_norm="minmax",
    )
    ds = NiftiCTDataset(cfg, split="valid")
    out = ds._preprocess_volume(ds.samples[0]["image_path"])

    nii = nib.as_closest_canonical(nib.load(ds.samples[0]["image_path"]))
    data = nii.get_fdata().astype(np.float32).transpose(2, 0, 1)
    expected = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    expected = F.interpolate(expected, size=(8, 16, 16), mode="trilinear", align_corners=False)
    expected = expected.squeeze(0)
    expected = expected.clamp(min=-1200.0, max=800.0)
    expected = (expected - (-1200.0)) / (800.0 - (-1200.0))
    assert torch.equal(out, expected)


def test_npz_source_loader(tmp_path):
    """NPZ source path matches the RAD-ChestCT loading pattern."""
    npz_path = tmp_path / "scan.npz"
    data = np.arange(16 * 32 * 32, dtype=np.float32).reshape((16, 32, 32)) - 5000.0
    np.savez_compressed(npz_path, ct=data)
    jsonl = tmp_path / "split.jsonl"
    jsonl.write_text(json.dumps({"sample_name": "s0", "nii_path": str(npz_path)}) + "\n")

    cfg = OmegaConf.create(
        {
            "loader": {
                "class": "NiftiCTDataset",
                "source": "npz",
                "npz_key": "ct",
                "preprocess": {"pipeline": "passthrough"},
            },
            "data": {
                "train_json": str(jsonl),
                "valid_json": str(jsonl),
                "test_json": str(jsonl),
            },
            "img_paths_key": "nii_path",
            "modality": "chest_ct",
        }
    )
    ds = NiftiCTDataset(cfg, split="valid")
    out = ds._preprocess_volume(ds.samples[0]["image_path"])
    assert out.shape == (1, 16, 32, 32)
    # data[0,0,0] == -5000.0 -> out[0,0,0,0] == -5000.0 (no preprocessing)
    assert out[0, 0, 0, 0].item() == pytest.approx(-5000.0)


def test_get_accession_returns_sample_name(tmp_path):
    cfg = _config(tmp_path, pipeline="passthrough")
    ds = NiftiCTDataset(cfg, split="valid")
    assert ds.get_accession(0) == "synth_sample_0"
    assert ds.get_all_accessions() == ["synth_sample_0"]
