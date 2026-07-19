"""CT-RATE V2 (HF ibrahimhamamci/CT-RATE) -- staging + label derivation."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

logger = logging.getLogger(__name__)

_HF_DATASET = "ibrahimhamamci/CT-RATE"

# real slope/intercept/spacing live in these CSVs (NIfTI headers are placeholders)
_METADATA_FILES: tuple[str, ...] = (
    "dataset/metadata/validation_metadata.csv",
    "dataset/metadata/train_metadata.csv",
    "dataset/metadata/no_chest_valid.txt",
    "dataset/metadata/no_chest_train.txt",
    "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
)

# 18 abnormalities; column order matters (per-class JSON names + 17-class CT-CLIP alignment)
_ABNORMALITIES: tuple[str, ...] = (
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
)


def _list_validation_files() -> list[str]:
    """List all validation NIfTI paths (repo-relative) on the HF dataset."""
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    api = HfApi()
    files = api.list_repo_files(_HF_DATASET, repo_type="dataset")
    return sorted(f for f in files if f.startswith("dataset/valid/") and f.endswith(".nii.gz"))


def _download_one(remote_path: str, local_root: Path, token: str | None) -> tuple[str, str, int]:
    """Download a single NIfTI; returns (remote_path, local_path, size_bytes)."""
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    local_path = hf_hub_download(
        _HF_DATASET,
        remote_path,
        repo_type="dataset",
        local_dir=str(local_root),
        token=token,
    )
    size = os.path.getsize(local_path)
    return remote_path, local_path, size


def _build_v2_valid_metadata(meta_dir: Path) -> None:
    """Derive validation_metadata_v2.csv (slope=1, int=0) from the shipped V1 CSV."""
    src = meta_dir / "validation_metadata.csv"
    dst = meta_dir / "validation_metadata_v2.csv"
    if not src.exists():
        logger.warning("validation_metadata.csv absent (%s); skipping V2 derivation", src)
        return
    import pandas as pd  # local import
    df = pd.read_csv(src)
    if "RescaleSlope" not in df or "RescaleIntercept" not in df:
        raise ValueError("validation_metadata.csv missing RescaleSlope/RescaleIntercept")
    df["RescaleSlope_orig"] = df["RescaleSlope"]
    df["RescaleIntercept_orig"] = df["RescaleIntercept"]
    df["RescaleSlope"] = 1.0
    df["RescaleIntercept"] = 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    logger.info("Wrote %s (%d rows; slope->1, int->0)", dst, len(df))


def stage(dest: Path, *, token: str | None = None, workers: int = 4,
          limit: int | None = None) -> None:
    """Download the CT-RATE validation split (~3,039 NIfTIs) + metadata CSVs into dest."""
    token = token or engine.require_env(
        "HF_TOKEN", hint="HuggingFace token for the gated ibrahimhamamci/CT-RATE repo")
    local_root = Path(dest)
    local_root.mkdir(parents=True, exist_ok=True)
    manifest_path = local_root / "manifest.csv"

    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    for meta_path in _METADATA_FILES:
        try:
            p = hf_hub_download(
                _HF_DATASET, meta_path, repo_type="dataset",
                local_dir=str(local_root), token=token,
            )
            logger.info("metadata: %s", p)
        except Exception as e:  # noqa: BLE001
            logger.warning("metadata download failed for %s: %s", meta_path, e)

    _build_v2_valid_metadata(local_root / "dataset" / "metadata")

    # valid_fixed staged -> V1 valid/ NIfTIs are deprecated upstream (download fails); skip
    if (local_root / "dataset" / "valid_fixed").exists():
        logger.info("valid_fixed present -> skipping deprecated V1 valid/ NIfTI download")
        return

    logger.info("Listing remote validation files on HF ...")
    remote_paths = _list_validation_files()
    logger.info("HF reports %d validation NIfTIs", len(remote_paths))
    if limit is not None:
        remote_paths = remote_paths[:limit]
        logger.info("Limited to %d for this run", len(remote_paths))

    rows: list[dict] = []
    n_done = n_failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one, rp, local_root, token): rp for rp in remote_paths}
        for fut in as_completed(futures):
            rp = futures[fut]
            try:
                remote, local, size = fut.result()
                rows.append({"remote_path": remote, "local_path": local, "size_bytes": size})
                n_done += 1
                if n_done % 50 == 0:
                    logger.info("Progress: %d/%d (failed=%d)", n_done, len(remote_paths), n_failed)
            except Exception as e:  # noqa: BLE001
                n_failed += 1
                logger.error("FAILED %s: %s", rp, e)
    logger.info("Download complete: ok=%d failed=%d", n_done, n_failed)

    rows.sort(key=lambda r: r["remote_path"])
    with manifest_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["remote_path", "local_path", "size_bytes"])
        w.writeheader()
        w.writerows(rows)
    total_gb = sum(r["size_bytes"] for r in rows) / 1024**3
    logger.info("Wrote manifest: %s (%d rows, %.1f GB total)", manifest_path, len(rows), total_gb)


def _split_for_patient(pid: int) -> str:
    """Deterministic 60/20/20 patient split (bespoke, not the canonical 70/15/15 stable_split)."""
    h = int(hashlib.sha256(f"ctrate_{pid}".encode()).hexdigest(), 16) % 100
    if h < 60:
        return "train"
    if h < 80:
        return "dev"
    return "test"


def _resolve_nii_path(volume_name: str, nii_root: Path) -> Path | None:
    """Map valid_1_a_1.nii.gz -> <nii_root>/valid_1/valid_1_a/valid_1_a_1.nii.gz."""
    stem = volume_name.replace(".nii.gz", "")
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "valid":
        return None
    pid = parts[1]
    scan = "_".join(parts[:3])  # valid_<pid>_<scan>
    return nii_root / f"valid_{pid}" / scan / volume_name


def _patient_id(volume_name: str) -> int:
    """Extract numeric patient id from valid_<pid>_<scan>_<recon>.nii.gz."""
    return int(volume_name.split("_")[1])


def build_v1_labels(ctrate_root: Path, eval_dir: Path, jsonl_dir: Path, *,
                    require_nii: bool = False) -> None:
    """Derive the 18 per-abnormality JSONs + ctrate_jsonl/ from the V1 labels CSV."""
    eval_dir, jsonl_dir = Path(eval_dir), Path(jsonl_dir)
    labels_csv = (Path(ctrate_root) / "dataset"
                  / "multi_abnormality_labels" / "valid_predicted_labels.csv")
    nii_root = Path(ctrate_root) / "dataset" / "valid"
    if not labels_csv.exists():
        raise SystemExit(f"Labels CSV not found: {labels_csv}")

    csv_rows: list[dict] = []
    with labels_csv.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for ab in _ABNORMALITIES:
            if ab not in cols:
                raise SystemExit(f"Abnormality column missing from CSV: {ab!r} (have: {cols})")
        for row in reader:
            csv_rows.append(row)
    logger.info("Loaded %d label rows from %s", len(csv_rows), labels_csv.name)

    # upstream excludes non-chest series from the chest-abnormality eval
    no_chest_path = Path(ctrate_root) / "dataset" / "metadata" / "no_chest_valid.txt"
    no_chest: set[str] = set()
    if no_chest_path.exists():
        no_chest = {Path(ln.strip()).name.replace(".nii.gz", "")
                    for ln in no_chest_path.read_text().splitlines() if ln.strip()}
        logger.info("no_chest exclusion: skipping %d non-chest volumes", len(no_chest))

    jsonl_payload: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    labels_payload: dict[str, dict] = {ab: {} for ab in _ABNORMALITIES}
    n_present = n_missing = n_no_chest = 0

    for row in csv_rows:
        vname = row["VolumeName"]
        if vname.replace(".nii.gz", "") in no_chest:
            n_no_chest += 1
            continue
        nii_path = _resolve_nii_path(vname, nii_root)
        if nii_path is None:
            logger.warning("Could not resolve volume name: %s", vname)
            continue
        if not nii_path.exists():
            n_missing += 1
            if require_nii:
                continue
        else:
            n_present += 1
        pid = _patient_id(vname)
        split = _split_for_patient(pid)
        accession = vname.replace(".nii.gz", "")  # e.g. valid_1_a_1
        jsonl_payload[split].append({
            "sample_name": accession,
            "nii_path": engine.data_root_relative(nii_path),
        })
        for ab in _ABNORMALITIES:
            try:
                lbl = int(row[ab])
            except ValueError:
                logger.warning("Non-integer label for %s/%s: %r", vname, ab, row[ab])
                continue
            qa_question = f"Is the abnormality '{ab}' present?"
            labels_payload[ab][accession] = {
                "split": split,
                "qa_results": {"default_qa": [{qa_question: lbl}]},
                "patient_id": pid,
            }

    logger.info("NIfTI resolution: %d present on disk, %d missing", n_present, n_missing)

    eval_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for ab in _ABNORMALITIES:
        engine.write_json(labels_payload[ab], eval_dir / f"ctrate_labels__{engine.slug(ab)}.json")
    logger.info("Wrote %d per-abnormality label JSONs in %s", len(_ABNORMALITIES), eval_dir)
    for split, items in jsonl_payload.items():
        engine.write_jsonl(items, jsonl_dir / f"{split}.jsonl")
        logger.info("Wrote %d rows -> %s/%s.jsonl", len(items), jsonl_dir, split)


def build_v2_valid_jsonl(v1_jsonl_dir: Path, v2_jsonl_dir: Path) -> None:
    """Rewrite the V1 valid JSONLs into V2 by swapping /valid/ -> /valid_fixed/ in nii_path."""
    v1_jsonl_dir, v2_jsonl_dir = Path(v1_jsonl_dir), Path(v2_jsonl_dir)
    v2_jsonl_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        v1_path = v1_jsonl_dir / f"{split}.jsonl"
        v2_path = v2_jsonl_dir / f"{split}.jsonl"
        if not v1_path.is_file():
            logger.warning("v1 JSONL missing: %s", v1_path)
            continue
        n_rewritten = 0
        with v1_path.open() as src, v2_path.open("w") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                p1 = row.get("nii_path")
                if not p1:
                    continue
                row["nii_path"] = p1.replace("/valid/", "/valid_fixed/")
                dst.write(json.dumps(row) + "\n")
                n_rewritten += 1
        logger.info("%s: rewrote %d rows (%s -> %s)", split, n_rewritten, v1_path, v2_path)


def build_v2_pretrain_jsonl(train_fixed_root: Path, out_dir: Path, *,
                            limit: int | None = None) -> None:
    """Enumerate the ~47K train_fixed/<pid>/<recon>/<series>.nii.gz tree -> KD-pretrain train.jsonl."""
    train_fixed_root, out_dir = Path(train_fixed_root), Path(out_dir)
    if not train_fixed_root.is_dir():
        raise FileNotFoundError(f"train_fixed root not found: {train_fixed_root}")
    paths = sorted(train_fixed_root.glob("*/*/*.nii.gz"))  # 3-level glob = HF hierarchy
    logger.info("Found %d NIfTI files under %s", len(paths), train_fixed_root)
    if limit is not None:
        paths = paths[:limit]
        logger.info("Capped to %d (smoke test)", len(paths))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"
    rows = [{"sample_name": p.name.removesuffix(".nii.gz"),
             "nii_path": engine.data_root_relative(p)} for p in paths]
    engine.write_jsonl(rows, out_path)
    logger.info("Wrote %d rows -> %s", len(rows), out_path)


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 ctrate_root: Path | None = None,
                 train_fixed_root: Path | None = None,
                 jsonl_dir: Path = Path("data/evaluation/ctrate_jsonl"),
                 v2_valid_jsonl_dir: Path = Path("data/evaluation/ctrate_v2_valid_jsonl"),
                 pretrain_jsonl_dir: Path = Path("data/ctrate_pretrain_v2_jsonl"),
                 require_nii: bool = False, limit: int | None = None) -> None:
    """Run the CT-RATE label/manifest derivation (V1 labels -> V2 valid JSONL; optional KD-pretrain JSONL)."""
    out_dir = Path(out_dir)
    if ctrate_root:
        build_v1_labels(ctrate_root, out_dir, jsonl_dir, require_nii=require_nii)
    if Path(jsonl_dir).is_dir():
        build_v2_valid_jsonl(jsonl_dir, v2_valid_jsonl_dir)
    if train_fixed_root:
        build_v2_pretrain_jsonl(train_fixed_root, pretrain_jsonl_dir, limit=limit)


_LABEL_SLUGS = tuple(engine.slug(ab) for ab in _ABNORMALITIES)

SPEC = DatasetSpec(
    name="ctrate",
    access=Access.HF,
    modality="chest_ct",
    role="kd-corpus",
    source=_HF_DATASET,
    token_env="HF_TOKEN",
    committed_outputs=(
        *(f"data/evaluation/ctrate_labels__{s}.json" for s in _LABEL_SLUGS),
        "data/evaluation/ctrate_jsonl/train.jsonl",
        "data/evaluation/ctrate_jsonl/dev.jsonl",
        "data/evaluation/ctrate_jsonl/test.jsonl",
        "data/evaluation/ctrate_v2_valid_jsonl/train.jsonl",
        "data/evaluation/ctrate_v2_valid_jsonl/dev.jsonl",
        "data/evaluation/ctrate_v2_valid_jsonl/test.jsonl",
        "data/ctrate_pretrain_v2_jsonl/train.jsonl",
    ),
    notes="CT-RATE V2: point profiles at train_fixed/valid_fixed (V1 uncorrected); "
          "double-rescale trap (regen validation_metadata_v2.csv slope=1/int=0) + "
          "~0.01% corrupt affines; 18-class label set (Mosaic dropped in the 17-class "
          "CT-CLIP alignment); bespoke 60/20/20 patient split + QA-question form.",
    stage=stage,
    build_labels=build_labels,
)
