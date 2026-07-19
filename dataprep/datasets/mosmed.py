"""MosMed (COVID19_1110) Kaggle stager + label builder; severity from CT-<n> studies/ subfolders, no label CSV."""
from __future__ import annotations

import random
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

_KAGGLE_SLUG = "mathurinache/mosmeddata-chest-ct-scans-with-covid19"
_TOP = "MosMedData Chest CT Scans with COVID-19 Related Findings COVID19_1110 1.0"
_N_STUDIES = 1110

_QA_MOW = "Does this CT show moderate-or-worse COVID-19 lung involvement? (MosMedData CT-2/3/4 vs CT-0/1)"
_QA_MOW_SEV = "Is the COVID severity moderate or worse (CT-2..CT-4)?"
_QA_COVID = "Does this CT show COVID-19 CT findings?"
_QA_SEVERE = "Does this CT show severe COVID-19 lung involvement (CT-3/CT-4)?"
_QA_ORDINAL = "What is the CT severity grade? (0=normal, 1=<25%, 2=25-50%, 3=50-75%, 4=>75%)"


def _studies_dir(staged_dir: Path) -> Path:
    return Path(staged_dir) / _TOP / "studies"


def _enumerate_studies(studies_dir: Path) -> dict[str, int]:
    """Map ``study_<nnnn>`` -> severity grade (0..4) from the ``CT-<n>`` subfolders."""
    studies: dict[str, int] = {}
    for grade in range(5):
        for f in (studies_dir / f"CT-{grade}").glob("study_*.nii"):
            studies[f.stem] = grade
    return studies


def stage(dest: Path = None, *, token: str | None = None, workers: int = 8,
          skip_download: bool = False) -> None:
    """Download the faithful Kaggle mirror into ``$DATA_ROOT/radiology/mosmed/`` (idempotent)."""
    if dest is None:
        dest = engine.expandvars("$DATA_ROOT/radiology/mosmed")
    dest = Path(dest)
    studies_dir = _studies_dir(dest)

    have = len(_enumerate_studies(studies_dir)) if studies_dir.is_dir() else 0
    if skip_download:
        print(f"[mosmed] skip_download; {have} studies present at {studies_dir}")
    elif have >= _N_STUDIES:
        print(f"[mosmed] already staged ({have} studies) at {studies_dir}")
    else:
        engine.kaggle_dataset_download(_KAGGLE_SLUG, dest, unzip=True)

    n = len(_enumerate_studies(studies_dir))
    if n < _N_STUDIES:
        raise RuntimeError(
            f"[mosmed] expected {_N_STUDIES} studies under {studies_dir}, found {n} "
            f"-- mirror incomplete; retry, or use another COVID19_1110 source."
        )
    print(f"[mosmed] staged {n} studies at {studies_dir}")


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 staged_dir: Path | None = None) -> None:
    """Re-derive every clinical label + the seed-42 split from the staged ``CT-<n>`` folders."""
    out_dir = Path(out_dir)
    if staged_dir is None:
        staged_dir = engine.expandvars("$DATA_ROOT/radiology/mosmed")
    studies_dir = _studies_dir(staged_dir)

    studies = _enumerate_studies(studies_dir)
    if len(studies) != _N_STUDIES:
        raise RuntimeError(
            f"[mosmed] expected {_N_STUDIES} staged studies under {studies_dir}, "
            f"found {len(studies)} -- run `--stage` first."
        )
    sorted_ids = sorted(studies)

    # seed-42 70/15/15 shuffle over the sorted ids.
    order = list(sorted_ids)
    random.seed(42)
    random.shuffle(order)
    n = len(order)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    split_of: dict[str, str] = {}
    for i, sid in enumerate(order):
        split_of[sid] = "train" if i < n_train else ("dev" if i < n_train + n_val else "test")

    def nii_path(sid: str) -> str:
        return engine.data_root_relative(studies_dir / f"CT-{studies[sid]}" / f"{sid}.nii")

    def rate_split(sid: str) -> str:
        return "valid" if split_of[sid] == "dev" else split_of[sid]

    # jsonl in shuffle order, one file per split.
    jsonl_dir = out_dir / "mosmed_jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for sid in order:
        buckets[split_of[sid]].append(
            {"sample_name": sid, "nii_path": nii_path(sid), "severity_class": studies[sid]}
        )
    for split, rows in buckets.items():
        engine.write_jsonl(rows, jsonl_dir / f"{split}.jsonl")

    # mosmed_labels.json uses the "valid" split label
    mow = {}
    for sid in order:
        g = studies[sid]
        tgt = 1 if g in (2, 3, 4) else 0
        mow[sid] = {
            "split": rate_split(sid),
            "qa_results": {"default_qa": [{_QA_MOW: tgt}]},
            "raw_severity_class": g,
            "target": tgt,
        }
    engine.write_json(mow, out_dir / "mosmed_labels.json")

    # per-task cell files (STOIC schema)
    def emit_task(name: str, qa: str, label_fn) -> None:
        d = {}
        for sid in order:
            d[sid] = {
                "split": split_of[sid],
                "qa_results": {"default_qa": [{qa: label_fn(studies[sid])}]},
                "patient_id": sid,
            }
        engine.write_json(d, out_dir / f"mosmed_labels__{name}.json")

    emit_task("severity_moderate_or_worse", _QA_MOW_SEV, lambda g: 1 if g in (2, 3, 4) else 0)
    emit_task("covid_findings", _QA_COVID, lambda g: 1 if g >= 1 else 0)
    emit_task("severe", _QA_SEVERE, lambda g: 1 if g >= 3 else 0)
    emit_task("severity_ordinal", _QA_ORDINAL, lambda g: g)

    pos = lambda pred: sum(1 for s in studies.values() if pred(s))
    print(f"[mosmed] labels for {n} studies -> {out_dir} "
          f"(covid+ {pos(lambda g: g>=1)}, mod+ {pos(lambda g: g>=2)}, severe {pos(lambda g: g>=3)})")


SPEC = DatasetSpec(
    name="mosmed",
    access=Access.KAGGLE,
    modality="chest_ct",
    role="descriptive",
    source=_KAGGLE_SLUG,
    token_env=None,  # kaggle CLI reads KAGGLE_API_TOKEN / ~/.kaggle/access_token
    committed_outputs=(
        "data/evaluation/mosmed_labels.json",
        "data/evaluation/mosmed_labels__severity_moderate_or_worse.json",
        "data/evaluation/mosmed_labels__covid_findings.json",
        "data/evaluation/mosmed_labels__severe.json",
        "data/evaluation/mosmed_labels__severity_ordinal.json",
        "data/evaluation/mosmed_jsonl/train.jsonl",
        "data/evaluation/mosmed_jsonl/dev.jsonl",
        "data/evaluation/mosmed_jsonl/test.jsonl",
    ),
    notes="Kaggle mirror mathurinache/mosmeddata-chest-ct-scans-with-covid19 (faithful "
          "COVID19_1110 release tree); severity from the CT-<n> studies/ subfolder; labels "
          "reproduce byte-for-byte via seed-42 70/15/15; five classification tasks "
          "(covid-findings/moderate-or-worse/severe/5-class-ordinal). Descriptive (underpowered).",
    stage=stage,
    build_labels=build_labels,
)
