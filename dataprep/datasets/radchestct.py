"""RAD-ChestCT (Zenodo record 6406114): staging + deterministic label-derivation waterfall."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

_ZENODO_RECORD_ID = "6406114"
_API = f"https://zenodo.org/api/records/{_ZENODO_RECORD_ID}"

# CT-RATE-aligned 17-class QA list (order matters); includes Mosaic, omits Interlobular
_CTRATE_CLASSES_17: tuple[str, ...] = (
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
)

# CT-RATE class -> Draelos abnormality token (Mosaic has no token; Interlobular -> septal_thickening)
_CTRATE_ABNORMALITY_MAP: dict[str, str] = {
    "Medical material":                   "hardware",
    "Arterial wall calcification":        "calcification",
    "Cardiomegaly":                       "cardiomegaly",
    "Pericardial effusion":               "pericardial_effusion",
    "Coronary artery wall calcification": "coronary_artery_disease",
    "Hiatal hernia":                      "hernia",
    "Lymphadenopathy":                    "lymphadenopathy",
    "Emphysema":                          "emphysema",
    "Atelectasis":                        "atelectasis",
    "Lung nodule":                        "nodule",
    "Lung opacity":                       "opacity",
    "Pulmonary fibrotic sequela":         "fibrosis",
    "Pleural effusion":                   "pleural_effusion",
    "Peribronchial thickening":           "bronchial_wall_thickening",
    "Consolidation":                      "consolidation",
    "Bronchiectasis":                     "bronchiectasis",
    "Interlobular septal thickening":     "septal_thickening",
}

# arterial + coronary merge into one calcification class in the 16-set
_ARTERIAL_SLUG = "arterial_wall_calcification"
_CORONARY_SLUG = "coronary_artery_wall_calcification"
_MERGED_SLUG = "calcification"
_SRC_PREFIX = "radchestct_labels__"
_OUT16_PREFIX = "radchestct16_labels__"

# paper-faithful 16-class correction labels (bring-your-own; skipped if absent)
_ANCHOR_LABELS = str(Path(__file__).resolve().parent / "radchestct_paper_labels.json")
_SEPTAL_NAME = "Interlobular septal thickening"
_SEPTAL_SLUG = "interlobular_septal_thickening"
_MOSAIC_SLUG = "mosaic_attenuation_pattern"


def _list_files(token: str) -> list[dict]:
    # InvenioRDM: restricted-record files live under /files, not the record's inline files[]
    with urllib.request.urlopen(f"{_API}/files?token={token}", timeout=60) as r:
        data = json.loads(r.read())
    return [{"key": e["key"], "size": e["size"]} for e in data.get("entries", [])]


def _download_one(file_info: dict, dest_dir: Path, token: str, retries: int = 6) -> tuple[str, str]:
    key, size = file_info["key"], file_info["size"]
    target = dest_dir / key
    if target.exists() and target.stat().st_size == size:
        return key, "SKIP"
    url = f"{_API}/files/{key}/content?token={token}"
    for attempt in range(retries):
        try:
            tmp = target.with_suffix(target.suffix + ".tmp")
            with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp.stat().st_size != size:
                tmp.unlink(missing_ok=True)
                raise IOError(f"size mismatch: {tmp.stat().st_size} vs expected {size}")
            tmp.rename(target)
            return key, "OK"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if attempt == retries - 1:
                    return key, f"FAIL HTTP 429 after {retries} retries"
                time.sleep(30 + 10 * attempt)
                continue
            if attempt == retries - 1:
                return key, f"FAIL HTTPError {e.code}: {str(e)[:120]}"
            time.sleep(min(2 ** attempt, 30))
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            if attempt == retries - 1:
                return key, f"FAIL {type(e).__name__}: {str(e)[:120]}"
            time.sleep(min(2 ** attempt, 30))
    return key, "FAIL exhausted"


def stage(dest: Path, *, token: str | None = None, workers: int = 8) -> None:
    """Download the 3,630 NPZ volumes + label CSVs into ``<dest>/{npz,labels}/`` (idempotent by size)."""
    token = token or engine.require_env(
        "ZENODO_RADCHESTCT_TOKEN", hint="Zenodo share-link JWT for record 6406114")
    dest = Path(dest)
    npz_dir, label_dir = dest / "npz", dest / "labels"
    npz_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    files = _list_files(token)
    todo = [{**f, "_dir": (npz_dir if f["key"].endswith(".npz") else label_dir)} for f in files]
    print(f"[radchestct] {len(todo)} files "
          f"({sum(f['size'] for f in files) / 1e9:.1f} GB) -> {dest}")

    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, f, f["_dir"], token): f for f in todo}
        for fut in as_completed(futs):
            key, status = fut.result()
            if status == "OK":
                n_ok += 1
            elif status == "SKIP":
                n_skip += 1
            else:
                n_fail += 1
                print(f"  FAIL {key}: {status}", file=sys.stderr)
    print(f"[radchestct] done ok={n_ok} skip={n_skip} fail={n_fail}")
    if n_fail:
        raise RuntimeError(f"[radchestct] {n_fail} downloads failed")


def _aggregate(row, abnormality: str) -> int:
    """OR a Draelos abnormality across all of its ``<abnormality>*<location>`` columns."""
    cols = [c for c in row.index if c.startswith(abnormality + "*")]
    if not cols:
        return 0
    return int(row[cols].astype(float).fillna(0).max() > 0)


def build_labels_17(labels_dir: Path, npz_dir: Path, eval_dir: Path, jsonl_dir: Path) -> None:
    """Derive ``radchestct_labels.json`` + ``radchestct_jsonl/`` from the Draelos CSVs."""
    import pandas as pd

    eval_dir, jsonl_dir = Path(eval_dir), Path(jsonl_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    csvs = {
        "train": Path(labels_dir) / "imgtrain_Abnormality_and_Location_Labels.csv",
        "dev":   Path(labels_dir) / "imgvalid_Abnormality_and_Location_Labels.csv",
        "test":  Path(labels_dir) / "imgtest_Abnormality_and_Location_Labels.csv",
    }
    all_labels: dict[str, dict] = {}
    for split_name, csv_path in csvs.items():
        if not csv_path.exists():
            raise SystemExit(f"missing label CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if "NoteAcc_DEID" not in df.columns:
            raise SystemExit(f"missing NoteAcc_DEID column in {csv_path}")
        tokens = sorted({c.split("*", 1)[0] for c in df.columns if "*" in c})
        rows: list[dict] = []
        for _, r in df.iterrows():
            acc = str(r["NoteAcc_DEID"]).strip()
            per_abn = {abn: _aggregate(r, abn) for abn in tokens}
            any_abn = int(any(v == 1 for v in per_abn.values()))
            present = [abn for abn, v in per_abn.items() if v == 1]
            ctrate_aligned = {ct: per_abn.get(tok, 0) for ct, tok in _CTRATE_ABNORMALITY_MAP.items()}
            all_labels[acc] = {
                "split": split_name,
                "any_abnormality": any_abn,
                "abnormalities_present": present,
                "ctrate_aligned": ctrate_aligned,
            }
            rows.append({
                "sample_name": acc,
                "npz_path": engine.data_root_relative(Path(npz_dir) / f"{acc}.npz"),
                "any_abnormality": any_abn,
                **{f"ctrate_{k.replace(' ', '_').lower()}": v for k, v in ctrate_aligned.items()},
            })
        engine.write_jsonl(rows, jsonl_dir / f"{split_name}.jsonl")
    engine.write_json(
        {"n_total": len(all_labels),
         "ctrate_class_to_radchestct_abnormality": _CTRATE_ABNORMALITY_MAP,
         "labels": all_labels},
        eval_dir / "radchestct_labels.json",
    )


def build_qa(labels_json: Path, out_path: Path) -> None:
    """``radchestct_labels.json`` -> ``radchestct_labels_qa.json`` (one entry, 17 QAs)."""
    raw = engine.read_json(labels_json)
    labels = raw.get("labels", raw)
    out: dict[str, dict] = {}
    for acc, entry in labels.items():
        aligned = entry.get("ctrate_aligned", {})
        qa_pairs = [{engine.qa_key(cls): int(aligned[cls])}
                    for cls in _CTRATE_CLASSES_17 if cls in aligned]
        out[acc] = {"split": entry.get("split"), "patient_id": acc,
                    "qa_results": {"default_qa": qa_pairs}}
    engine.write_json(out, out_path)


def build_per_class(qa_path: Path, out_dir: Path) -> None:
    """``radchestct_labels_qa.json`` -> 17 ``radchestct_labels__<slug>.json``."""
    source = engine.read_json(qa_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in _CTRATE_CLASSES_17:
        key = engine.qa_key(cls)
        out: dict[str, dict] = {}
        for acc, entry in source.items():
            value = next((p[key] for p in entry.get("qa_results", {}).get("default_qa", [])
                          if key in p), None)
            if value is None:
                continue
            out[acc] = {"split": entry.get("split"),
                        "patient_id": entry.get("patient_id", acc),
                        "qa_results": {"default_qa": [{key: int(value)}]}}
        engine.write_json(out, out_dir / f"{_SRC_PREFIX}{engine.slug(cls)}.json")


def build_merged16(eval_dir: Path, out_dir: Path) -> None:
    """17 per-class -> 16 ``radchestct16_labels__<slug>.json`` (Calcification = max(art, cor))."""
    eval_dir, out_dir = Path(eval_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = {f.name[len(_SRC_PREFIX):-len(".json")]: f
           for f in sorted(eval_dir.glob(f"{_SRC_PREFIX}*.json"))}
    for required in (_ARTERIAL_SLUG, _CORONARY_SLUG):
        if required not in src:
            raise SystemExit(f"missing required source class {required!r} under {eval_dir}")

    def _val(entry: dict) -> int:
        return int(next(iter(entry["qa_results"]["default_qa"][0].values())))

    # carry over the 15 non-calcification classes verbatim
    for slug_, path in src.items():
        if slug_ in (_ARTERIAL_SLUG, _CORONARY_SLUG):
            continue
        engine.write_json(engine.read_json(path), out_dir / f"{_OUT16_PREFIX}{slug_}.json")
    # merge arterial + coronary -> one calcification class (elementwise max)
    arterial, coronary = engine.read_json(src[_ARTERIAL_SLUG]), engine.read_json(src[_CORONARY_SLUG])
    merged_key = engine.qa_key(_MERGED_SLUG)
    merged: dict[str, dict] = {}
    for acc in sorted(set(arterial) | set(coronary)):
        a = _val(arterial[acc]) if acc in arterial else 0
        c = _val(coronary[acc]) if acc in coronary else 0
        ref = arterial.get(acc) or coronary[acc]
        merged[acc] = {"split": ref.get("split"),
                       "patient_id": ref.get("patient_id", acc),
                       "qa_results": {"default_qa": [{merged_key: max(a, c)}]}}
    engine.write_json(merged, out_dir / f"{_OUT16_PREFIX}{_MERGED_SLUG}.json")


def build_paperfaithful16(eval_dir: Path, anchor_path: Path = Path(_ANCHOR_LABELS)) -> None:
    """Correct the merged-16 set in place to the CT-CLIP paper-faithful 16 classes."""
    eval_dir = Path(eval_dir)
    anchor = engine.read_json(anchor_path)["labels"]  # {accession: {ClassName: 0/1}}

    # calcification: rebuild values from the anchor
    calc_path = eval_dir / f"{_OUT16_PREFIX}{_MERGED_SLUG}.json"
    calc = engine.read_json(calc_path)
    for acc, rec in calc.items():
        qa = rec["qa_results"]["default_qa"][0]
        (q,) = qa.keys()
        qa[q] = int(anchor[acc]["Calcification"])
    calc_path.write_text(json.dumps(calc, indent=2) + "\n")

    # interlobular_septal_thickening: clone calcification scaffolding, swap Q + values
    septal_q = engine.qa_key(_SEPTAL_NAME)
    septal = {}
    for acc, rec in calc.items():
        r = json.loads(json.dumps(rec))  # order-preserving deep copy
        r["qa_results"]["default_qa"][0] = {septal_q: int(anchor[acc][_SEPTAL_NAME])}
        septal[acc] = r
    (eval_dir / f"{_OUT16_PREFIX}{_SEPTAL_SLUG}.json").write_text(json.dumps(septal, indent=2) + "\n")

    # mosaic_attenuation_pattern: drop (no Draelos analogue)
    (eval_dir / f"{_OUT16_PREFIX}{_MOSAIC_SLUG}.json").unlink(missing_ok=True)


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 labels_dir: Path | None = None, npz_dir: Path | None = None,
                 jsonl_dir: Path = Path("data/radchestct_jsonl"),
                 anchor_path: Path = Path(_ANCHOR_LABELS)) -> None:
    """Run the full derivation waterfall; ``labels_dir`` + ``npz_dir`` rebuild the 17-class from CSVs first."""
    out_dir = Path(out_dir)
    if labels_dir and npz_dir:
        build_labels_17(labels_dir, npz_dir, out_dir, jsonl_dir)
    labels_json = out_dir / "radchestct_labels.json"
    if not labels_json.is_file():
        raise SystemExit(f"{labels_json} not found -- pass --labels-dir/--npz-dir to build it from CSVs")
    build_qa(labels_json, out_dir / "radchestct_labels_qa.json")
    build_per_class(out_dir / "radchestct_labels_qa.json", out_dir)
    build_merged16(out_dir, out_dir)
    if Path(anchor_path).is_file():
        build_paperfaithful16(out_dir, anchor_path)


_PER_CLASS_SLUGS = tuple(engine.slug(c) for c in _CTRATE_CLASSES_17)
_PAPERFAITHFUL16_SLUGS = tuple(
    s for s in _PER_CLASS_SLUGS if s not in (_ARTERIAL_SLUG, _CORONARY_SLUG, _MOSAIC_SLUG)
) + (_MERGED_SLUG, _SEPTAL_SLUG)

SPEC = DatasetSpec(
    name="radchestct",
    access=Access.ZENODO,
    modality="chest_ct",
    role="scored",
    source=_ZENODO_RECORD_ID,
    token_env="ZENODO_RADCHESTCT_TOKEN",
    committed_outputs=(
        "data/evaluation/radchestct_labels.json",
        "data/evaluation/radchestct_labels_qa.json",
        "data/radchestct_jsonl/train.jsonl",
        "data/radchestct_jsonl/dev.jsonl",
        "data/radchestct_jsonl/test.jsonl",
        *(f"data/evaluation/radchestct_labels__{s}.json" for s in _PER_CLASS_SLUGS),
        *(f"data/evaluation/radchestct16_labels__{s}.json" for s in _PAPERFAITHFUL16_SLUGS),
    ),
    notes="CT-CLIP needs an orientation fix (NPZs H<->W swapped + axial-reversed); "
          "paper-faithful 16-class set uses the Draelos-native Calcification token + "
          "interlobular_septal_thickening, drops mosaic.",
    stage=stage,
    build_labels=build_labels,
)
