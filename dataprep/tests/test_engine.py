"""Unit tests for the shared dataset-curation primitives (dataprep.datasets.engine)."""
from __future__ import annotations

import hashlib
import json

from dataprep.datasets import engine


def _legacy_split(key: str) -> str:
    """The canonical split every legacy builder used, recomputed independently."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("dev" if h < 85 else "test")


def test_stable_split_matches_canonical_for_all_buckets():
    for i in range(2000):
        key = f"pid_{i}"
        assert engine.stable_split(key) == _legacy_split(key), key


def test_stable_split_is_deterministic():
    assert engine.stable_split("abc") == engine.stable_split("abc")


def test_stable_split_boundaries():
    buckets = {engine.stable_split(f"k{i}") for i in range(500)}
    assert buckets == {"train", "dev", "test"}


def test_qa_key_format():
    assert engine.qa_key("Lung nodule") == "Is lung nodule present? (0=No, 1=Yes)"
    assert engine.qa_key("Calcification") == "Is calcification present? (0=No, 1=Yes)"


def test_slug():
    assert engine.slug("Medical material") == "medical_material"
    assert engine.slug("Interlobular septal thickening") == "interlobular_septal_thickening"
    assert engine.slug("  Arterial/Wall  ") == "arterial_wall"


def test_write_json_byte_format(tmp_path):
    # committed byte format: indent=2, no trailing newline, no sort_keys
    obj = {"b": 1, "a": {"x": [1, 2]}}
    p = tmp_path / "o.json"
    engine.write_json(obj, p)
    raw = p.read_bytes()
    assert raw == json.dumps(obj, indent=2).encode()
    assert not raw.endswith(b"\n")
    assert engine.read_json(p) == obj


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1}, {"b": "two"}]
    p = tmp_path / "o.jsonl"
    engine.write_jsonl(rows, p)
    assert p.read_text() == '{"a": 1}\n{"b": "two"}\n'
    assert engine.read_jsonl(p) == rows


def test_data_root_relative(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    inside = tmp_path / "radiology" / "x" / "v.npz"
    assert engine.data_root_relative(inside) == "${DATA_ROOT}/radiology/x/v.npz"
    outside = tmp_path.parent / "elsewhere" / "v.npz"
    assert engine.data_root_relative(outside) == str(outside)


def test_require_env(monkeypatch):
    monkeypatch.setenv("SOME_TOKEN", "secret")
    assert engine.require_env("SOME_TOKEN") == "secret"
    monkeypatch.delenv("SOME_TOKEN", raising=False)
    try:
        engine.require_env("SOME_TOKEN", hint="set it")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "SOME_TOKEN" in str(e)
