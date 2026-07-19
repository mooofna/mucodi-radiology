"""Test the RAD-ChestCT 16-class merge builder (Calcification = max(arterial, coronary))."""
from __future__ import annotations

import json
from pathlib import Path

from dataprep.datasets.radchestct import build_merged16


def _per_class(values: dict[str, int], qa_key: str) -> dict:
    return {
        acc: {
            "split": "test",
            "patient_id": acc,
            "qa_results": {"default_qa": [{qa_key: int(v)}]},
        }
        for acc, v in values.items()
    }


def _val(entry: dict) -> int:
    return int(next(iter(entry["qa_results"]["default_qa"][0].values())))


def test_merge_is_elementwise_max_and_carries_other_classes(tmp_path: Path):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    accs = ["trn1", "trn2", "trn3", "trn4"]
    arterial = {"trn1": 1, "trn2": 0, "trn3": 1, "trn4": 0}
    coronary = {"trn1": 0, "trn2": 0, "trn3": 1, "trn4": 1}
    other = {"trn1": 1, "trn2": 1, "trn3": 0, "trn4": 0}
    (eval_dir / "radchestct_labels__arterial_wall_calcification.json").write_text(
        json.dumps(_per_class(arterial, "Is arterial wall calcification present? (0=No, 1=Yes)"))
    )
    (eval_dir / "radchestct_labels__coronary_artery_wall_calcification.json").write_text(
        json.dumps(_per_class(coronary, "Is coronary artery wall calcification present? (0=No, 1=Yes)"))
    )
    (eval_dir / "radchestct_labels__cardiomegaly.json").write_text(
        json.dumps(_per_class(other, "Is cardiomegaly present? (0=No, 1=Yes)"))
    )

    build_merged16(eval_dir, eval_dir)

    out_files = sorted(eval_dir.glob("radchestct16_labels__*.json"))
    slugs = {f.name.split("__")[1].replace(".json", "") for f in out_files}
    # arterial + coronary merge into calcification; cardiomegaly carries through.
    assert slugs == {"calcification", "cardiomegaly"}
    assert not any("arterial" in s or "coronary" in s for s in slugs)

    merged = json.loads((eval_dir / "radchestct16_labels__calcification.json").read_text())
    for acc in accs:
        assert _val(merged[acc]) == max(arterial[acc], coronary[acc]), acc
    carried = json.loads((eval_dir / "radchestct16_labels__cardiomegaly.json").read_text())
    for acc in accs:
        assert _val(carried[acc]) == other[acc]
