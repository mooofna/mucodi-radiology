"""Byte-identity regeneration test for the RAD-ChestCT label pipeline (the faithfulness gate)."""
from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

import pytest

from dataprep.datasets import engine, radchestct

_EVAL = Path("data/evaluation")
_LABELS = _EVAL / "radchestct_labels.json"
_ANCHOR = Path(radchestct._ANCHOR_LABELS)

pytestmark = pytest.mark.skipif(
    not (_LABELS.is_file() and _ANCHOR.is_file()),
    reason="committed radchestct_labels.json / anchor labels not present",
)


def _diff(regen_dir, pattern):
    regen = sorted(p.name for p in regen_dir.glob(pattern))
    mism = [n for n in regen if not (_EVAL / n).is_file()
            or not filecmp.cmp(regen_dir / n, _EVAL / n, shallow=False)]
    committed_only = [p.name for p in _EVAL.glob(pattern) if p.name not in regen]
    return regen, mism, committed_only


def test_full_chain_byte_identical_to_committed(tmp_path):
    shutil.copy(_LABELS, tmp_path / "radchestct_labels.json")
    radchestct.build_labels(tmp_path)  # qa -> per_class -> merged16 -> paperfaithful16

    for pattern in ("radchestct_labels_qa.json",
                    "radchestct_labels__*.json",
                    "radchestct16_labels__*.json"):
        regen, mism, committed_only = _diff(tmp_path, pattern)
        assert regen, f"nothing regenerated for {pattern}"
        assert not mism, f"{pattern}: byte mismatch vs committed: {mism}"
        assert not committed_only, f"{pattern}: committed files not regenerated: {committed_only}"


def test_calcification_is_paper_faithful_not_max(tmp_path):
    """Calcification uses the Draelos-native token (2562 pos), not max(arterial, coronary) (2626)."""
    shutil.copy(_LABELS, tmp_path / "radchestct_labels.json")
    radchestct.build_labels(tmp_path)
    calc = engine.read_json(tmp_path / "radchestct16_labels__calcification.json")
    n_pos = sum(int(next(iter(r["qa_results"]["default_qa"][0].values()))) for r in calc.values())
    assert n_pos == 2562, f"expected paper-faithful 2562 positives, got {n_pos}"
    # mosaic dropped, interlobular added
    assert not (tmp_path / "radchestct16_labels__mosaic_attenuation_pattern.json").exists()
    assert (tmp_path / "radchestct16_labels__interlobular_septal_thickening.json").exists()
