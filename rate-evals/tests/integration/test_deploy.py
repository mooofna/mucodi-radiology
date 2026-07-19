"""Integration test for the crossval -> deploy cell (deploy_one_class) and macro_aggregate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rate_eval.evaluation.deploy import deploy_one_class

DIM = 8
CLASSES = ("c0", "c1")  # class ck's label = feature dim k
SRC_Q = {"c0": "Is the abnormality 'C0' present?", "c1": "Is the abnormality 'C1' present?"}
TGT_Q = {"c0": "Is c0 present? (0=No, 1=Yes)", "c1": "Is c1 present? (0=No, 1=Yes)"}


def _base_vec(rng: np.random.Generator) -> np.ndarray:
    """Random vector with a STRONG +/- signal in the two class dims (robust to L2-norm)."""
    v = (rng.standard_normal(DIM).astype(np.float32)) * 0.2
    for k in range(len(CLASSES)):
        v[k] = 2.0 if rng.random() < 0.5 else -2.0
    return v


def _write_source(cache_dir: Path, n_patients: int, seed: int) -> dict[str, Path]:
    """CT-RATE-shaped source cache + one label JSON per class. Returns {class: labels_path}."""
    rng = np.random.default_rng(seed)
    emb = cache_dir / "embeddings" / "test"
    emb.mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {ck: {} for ck in CLASSES}
    for i in range(n_patients):
        base = _base_vec(rng)
        for study in ("a", "b"):
            acc = f"valid_{1000 + i}_{study}_1"
            vec = base + rng.standard_normal(DIM).astype(np.float32) * 0.05
            np.savez(emb / f"{acc}.npz", embedding=vec)
            for k, ck in enumerate(CLASSES):
                labels[ck][acc] = {
                    "split": "test", "patient_id": 1000 + i,
                    "qa_results": {"default_qa": [{SRC_Q[ck]: int(base[k] > 0)}]},
                }
    out: dict[str, Path] = {}
    for ck in CLASSES:
        p = cache_dir / f"source_labels__{ck}.json"
        p.write_text(json.dumps(labels[ck]))
        out[ck] = p
    return out


def _write_target(cache_dir: Path, n_samples: int, seed: int) -> dict[str, Path]:
    """RAD-ChestCT-shaped target cache (1 scan/patient, identity key) + per-class label JSONs."""
    rng = np.random.default_rng(seed)
    emb = cache_dir / "embeddings" / "test"
    emb.mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {ck: {} for ck in CLASSES}
    for j in range(n_samples):
        vec = _base_vec(rng)
        acc = f"trn{10000 + j}"
        np.savez(emb / f"{acc}.npz", embedding=vec)
        for k, ck in enumerate(CLASSES):
            labels[ck][acc] = {
                "split": "test", "patient_id": acc,
                "qa_results": {"default_qa": [{TGT_Q[ck]: int(vec[k] > 0)}]},
            }
    out: dict[str, Path] = {}
    for ck in CLASSES:
        p = cache_dir / f"target_labels__{ck}.json"
        p.write_text(json.dumps(labels[ck]))
        out[ck] = p
    return out


def _deploy(src_cache, tgt_cache, src_lbl, tgt_lbl, ck, out_dir, head="linear"):
    # tiny-sample plumbing test: more epochs to converge, only exercises CV->deploy->aggregate
    return deploy_one_class(
        source_cache=src_cache, source_labels_json=src_lbl[ck],
        target_cache=tgt_cache, target_labels_json=tgt_lbl[ck],
        class_name=ck, out_dir=out_dir, head_spec={"kind": head},
        l2_normalize=True, source_cohort="ct_rate", target_cohort="radchestct",
        cv_folds=5, max_epochs=40, patience=12, batch_size=16, max_lr=1e-2,
        n_boot=50, device="cpu",
    )


def test_deploy_one_class_writes_layout_and_transfers(tmp_path):
    src_cache = tmp_path / "src"
    tgt_cache = tmp_path / "tgt"
    src_lbl = _write_source(src_cache, n_patients=40, seed=0)
    tgt_lbl = _write_target(tgt_cache, n_samples=50, seed=1)
    out = tmp_path / "cell"

    for ck in CLASSES:
        res = _deploy(src_cache, tgt_cache, src_lbl, tgt_lbl, ck, out)
        internal = out / "internal" / "per_class" / ck
        external = out / "external" / "per_class" / ck
        for d in (internal, external):
            assert (d / "summary.json").exists() and (d / "predictions_oof.json").exists()
        isum = json.loads((internal / "summary.json").read_text())
        esum = json.loads((external / "summary.json").read_text())
        assert isinstance(isum["pooled"]["auroc"], float) and 0.0 <= isum["pooled"]["auroc"] <= 1.0
        assert len(isum["pooled_auroc_ci_cluster"]) == 2
        assert isum["ci_resample_unit"] == "patient_cluster"
        assert isinstance(esum["pooled"]["auroc"], float)
        assert res["internal_auroc"] > 0.75, f"{ck} internal {res['internal_auroc']}"
        assert res["external_auroc"] > 0.70, f"{ck} external {res['external_auroc']}"
        assert isum["feature_l2_normalized"] is True and esum["feature_l2_normalized"] is True
        assert isum["n_patients"] == 40
        # external OOF = one row per target patient (50 identity keys)
        erows = json.loads((external / "predictions_oof.json").read_text())
        assert len(erows) == 50 and len({r["patient_id"] for r in erows}) == 50
        assert all(set(r) >= {"patient_id", "label", "score"} for r in erows)


def test_deploy_is_deterministic(tmp_path):
    src_cache = tmp_path / "src"
    tgt_cache = tmp_path / "tgt"
    src_lbl = _write_source(src_cache, n_patients=36, seed=3)
    tgt_lbl = _write_target(tgt_cache, n_samples=44, seed=4)
    r1 = _deploy(src_cache, tgt_cache, src_lbl, tgt_lbl, "c0", tmp_path / "a")
    r2 = _deploy(src_cache, tgt_cache, src_lbl, tgt_lbl, "c0", tmp_path / "b")
    assert r1["external_auroc"] == r2["external_auroc"]
    assert r1["internal_auroc"] == r2["internal_auroc"]


def test_macro_aggregate_consumes_both_sides(tmp_path):
    from experiments.studies import macro_aggregate

    src_cache = tmp_path / "src"
    tgt_cache = tmp_path / "tgt"
    src_lbl = _write_source(src_cache, n_patients=40, seed=5)
    tgt_lbl = _write_target(tgt_cache, n_samples=50, seed=6)
    out = tmp_path / "cell"
    for ck in CLASSES:
        _deploy(src_cache, tgt_cache, src_lbl, tgt_lbl, ck, out)

    for side, cohort in (("internal", "ct_rate"), ("external", "radchestct")):
        macro_out = out / side / "macro_summary.json"
        rc = macro_aggregate.main([
            "--per-class-dir", str(out / side / "per_class"),
            "--out", str(macro_out),
            "--expected-classes", str(len(CLASSES)),
            "--n-boot", "200", "--seed", "42", "--cohort", cohort,
        ])
        assert rc == 0
        d = json.loads(macro_out.read_text())
        for key in ("macro_auroc", "macro_f1", "macro_auprc"):
            assert isinstance(d[key], float) and not np.isnan(d[key]), f"{side} {key}={d[key]}"
        assert d["n_classes_used"] == len(CLASSES)
