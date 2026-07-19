"""Integration test for the `rate-evaluate compare` aligner (row/cluster alignment, clean errors)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rate_eval.cli.evaluate import _build_parser


def _write_oof(path: Path, scores, labels, pids):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"fold": i % 5, "patient_id": p, "label": int(y), "score": s}
        for i, (p, y, s) in enumerate(zip(pids, labels, scores))
    ]
    path.write_text(json.dumps(rows))


def _synth_oof(n_patients=12, seed=0):
    rng = np.random.default_rng(seed)
    pids, labels = [], []
    for i in range(n_patients):
        y = i % 2
        for _ in range(2):  # two sibling volumes per patient
            pids.append(f"valid_{1000 + i}")
            labels.append(y)
    return pids, labels


def _run_compare(p_list, out_dir):
    argv = ["compare", "--predictions-oof", *[str(p) for p in p_list],
            "--output-dir", str(out_dir), "--n-boot", "200", "--seed", "42"]
    args = _build_parser().parse_args(argv)
    args.func(args)


def test_compare_keeps_all_rows_and_clusters(tmp_path):
    pids, labels = _synth_oof(12)
    rng = np.random.default_rng(1)
    # model A separable, model B noisier
    sa = [float(rng.standard_normal() * 0.4 + (1 if y else -1)) for y in labels]
    sb = [float(rng.standard_normal() * 1.2 + (0.3 if y else -0.3)) for y in labels]
    pa = tmp_path / "modelA" / "predictions_oof.json"
    pb = tmp_path / "modelB" / "predictions_oof.json"
    _write_oof(pa, sa, labels, pids)
    _write_oof(pb, sb, labels, pids)
    out = tmp_path / "cmp"
    _run_compare([pa, pb], out)

    res = json.loads((out / "pairwise.json").read_text())
    assert res["n_rows"] == 24
    assert res["n_patients"] == 12
    assert res["ci_resample_unit"] == "patient_cluster"
    assert "modelA__vs__modelB" in res["pairs"]
    pair = res["pairs"]["modelA__vs__modelB"]
    assert set(pair) >= {"diff_auroc", "ci", "p_value", "p_holm", "reject_at_alpha"}


def test_compare_multiclass_scores_raise_clean_message(tmp_path):
    pids, labels = _synth_oof(12)
    # softmax (list) scores -> must raise SystemExit
    sa = [[0.2, 0.3, 0.5] for _ in labels]
    sb = [[0.5, 0.3, 0.2] for _ in labels]
    pa = tmp_path / "mcA" / "predictions_oof.json"
    pb = tmp_path / "mcB" / "predictions_oof.json"
    _write_oof(pa, sa, labels, pids)
    _write_oof(pb, sb, labels, pids)
    with pytest.raises(SystemExit, match="multiclass compare unsupported"):
        _run_compare([pa, pb], tmp_path / "cmp_mc")


def test_compare_mismatched_oof_raises(tmp_path):
    pids, labels = _synth_oof(12)
    rng = np.random.default_rng(2)
    sa = [float(rng.standard_normal()) for _ in labels]
    pa = tmp_path / "mA" / "predictions_oof.json"
    _write_oof(pa, sa, labels, pids)
    # model B shuffled -> (patient_id, label) sequence differs
    perm = list(reversed(range(len(pids))))
    pb = tmp_path / "mB" / "predictions_oof.json"
    _write_oof(pb, [sa[i] for i in perm], [labels[i] for i in perm], [pids[i] for i in perm])
    with pytest.raises(SystemExit, match="alignment mismatch"):
        _run_compare([pa, pb], tmp_path / "cmp_mm")
