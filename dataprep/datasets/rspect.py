"""RSPECT -- RSNA-STR Pulmonary Embolism Detection 2020 (Kaggle CTPA), the only contrast-enhanced cohort."""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

logger = logging.getLogger(__name__)

_COMPETITION_SLUG = "rsna-str-pulmonary-embolism-detection"

# required train.csv columns; endpoint panel depends on these exact names
_EXAM_LABEL_COLS = (
    "negative_exam_for_pe", "indeterminate", "rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1",
    "leftsided_pe", "rightsided_pe", "central_pe", "chronic_pe", "acute_and_chronic_pe",
)

# study-level QA keys; must byte-match the benchmark cell qa_key
_QA_PE = ("Is pulmonary embolism present? "
          "(RSPECT/RSNA-PE 2020 study-level, 1-negative_exam_for_pe)")
_QA_RVLV = ("Does this CT show right-heart strain (RV/LV diameter ratio >= 1)? "
            "(RSPECT/RSNA-PE 2020 study-level)")
_QA_CENTRAL = ("Is a central (main/saddle) pulmonary embolism present? "
               "(RSPECT/RSNA-PE 2020 study-level)")
_QA_RVLV_AMONG = ("Among PE-positive scans, is there right-heart strain (RV/LV ratio >= 1)? "
                  "(RSPECT/RSNA-PE 2020 study-level)")
_QA_CHRONIC = ("Are chronic thromboembolic features present (chronic or acute-and-chronic PE)? "
               "(RSPECT/RSNA-PE 2020 study-level)")
# materialized-only (committed, not scored)
_QA_BILATERAL = ("Is bilateral pulmonary embolism present (left and right sided)? "
                 "(RSPECT/RSNA-PE 2020 study-level)")
_QA_ACUTE = ("Among PE-positive scans, is the PE acute-only (no chronic features)? "
             "(RSPECT/RSNA-PE 2020 study-level)")


def _download(raw_dir: Path, *, skip_download: bool, kaggle_cli: str = "kaggle") -> None:
    """Fetch the Kaggle competition into raw_dir (left zipped for per-study streaming)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if skip_download:
        return
    if shutil.which(kaggle_cli) is None:
        raise RuntimeError(
            f"kaggle CLI not found ({kaggle_cli!r}); `uv pip install kaggle` and set "
            f"KAGGLE_API_TOKEN in jobs/secrets.env")
    logger.info("[rspect] downloading %s -> %s", _COMPETITION_SLUG, raw_dir)
    cmd = [kaggle_cli, "competitions", "download", "-c", _COMPETITION_SLUG, "-p", str(raw_dir)]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"kaggle competition download returned {rc}")


def _sitk_series_to_nifti(series_dir: Path, out: Path) -> bool:
    """SimpleITK ImageSeriesReader fallback when dicom2nifti refuses (largest GDCM series)."""
    try:
        import SimpleITK as sitk  # noqa: PLC0415
    except ImportError:  # pragma: no cover - environment-dependent
        return False
    reader = sitk.ImageSeriesReader()
    names: list[str] = []
    try:
        sids = reader.GetGDCMSeriesIDs(str(series_dir))
        if sids:
            names = max((reader.GetGDCMSeriesFileNames(str(series_dir), s) for s in sids), key=len)
    except Exception:  # noqa: BLE001
        names = []
    if not names:
        names = sorted(str(f) for f in series_dir.iterdir() if f.is_file())
    if not names:
        return False
    try:
        reader.SetFileNames(names)
        sitk.WriteImage(reader.Execute(), str(out))
    except Exception:  # noqa: BLE001
        return False
    return out.exists() and out.stat().st_size > 0


def _convert_series(series_dir: Path, out_path: Path) -> bool:
    """dicom2nifti(reorient_nifti=True) with SimpleITK fallback -> out_path (idempotent)."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    try:
        import dicom2nifti  # noqa: PLC0415
        dicom2nifti.dicom_series_to_nifti(str(series_dir), str(out_path), reorient_nifti=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("dicom2nifti %s: %s", series_dir.name, exc)
    if out_path.exists():
        out_path.unlink()
    return _sitk_series_to_nifti(series_dir, out_path)


def _valid_nifti(path: Path) -> bool:
    """Cheap validation before deleting raw DICOM: non-empty, 3-D, axial-first, finite HU."""
    if not (path.exists() and path.stat().st_size > 0):
        return False
    try:
        import numpy as np  # noqa: PLC0415
        import nibabel as nib  # noqa: PLC0415
        nii = nib.load(str(path))
        shape = tuple(int(s) for s in nii.shape[:3])
        if len(shape) < 3 or min(shape) < 8:
            return False
        mid = np.asarray(nii.dataobj[:, :, shape[2] // 2], dtype="float32")
        return bool(np.isfinite(mid).any())
    except Exception:  # noqa: BLE001
        return False


def _zip_study_index(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    """Map StudyInstanceUID -> [zip member names] for the train/ DICOM tree."""
    index: dict[str, list[str]] = defaultdict(list)
    for name in zf.namelist():
        parts = name.split("/")  # train/<study>/<series>/<sop>.dcm
        if len(parts) >= 4 and parts[0] == "train" and name.endswith(".dcm"):
            index[parts[1]].append(name)
    return index


def _largest_series_dir(study_dir: Path) -> Path | None:
    """The series dir with the most *.dcm (the label-time pick)."""
    series = [s for s in study_dir.iterdir() if s.is_dir()]
    if not series:
        return None
    return max(series, key=lambda s: sum(1 for _ in s.glob("*.dcm")))


def stage(dest: Path, *, token: str | None = None, skip_download: bool = False,
          keep_raw: bool = False, shard: int = 0, nshards: int = 1,
          kaggle_cli: str = "kaggle") -> None:
    """Download the Kaggle RSPECT competition and convert its train/ tree to per-study NIfTI."""
    dest = Path(dest)
    raw_dir = dest / "raw"
    out_root = dest / "nifti"
    out_root.mkdir(parents=True, exist_ok=True)

    _download(raw_dir, skip_download=skip_download, kaggle_cli=kaggle_cli)

    # stream from the zip when present; don't key on raw/train (concurrent shards race)
    extracted_train = raw_dir / "train"
    zips = sorted(raw_dir.glob("*.zip"), key=lambda p: -p.stat().st_size)
    use_zip = bool(zips)
    if not use_zip and not extracted_train.is_dir():
        raise RuntimeError(f"[rspect] no zip and no extracted train/ tree under {raw_dir}")

    zf = zipfile.ZipFile(zips[0]) if use_zip else None
    study_index: dict[str, list[str]] = {}
    if use_zip:
        study_index = _zip_study_index(zf)
        studies = sorted(study_index)
        logger.info("[rspect] %d studies in %s", len(studies), zips[0].name)
    else:
        studies = sorted(d.name for d in extracted_train.iterdir() if d.is_dir())
        logger.info("[rspect] %d studies in extracted tree %s", len(studies), extracted_train)

    studies = [s for i, s in enumerate(studies) if i % nshards == shard]

    n_ok = n_fail = n_skip = 0
    failed: list[str] = []
    try:
      for i, study in enumerate(studies):
        if use_zip:
            tmp_study = raw_dir / "train" / study
            if not tmp_study.exists():
                tmp_study.mkdir(parents=True, exist_ok=True)
                zf.extractall(raw_dir, members=study_index[study])
            study_dir = tmp_study
        else:
            study_dir = extracted_train / study

        largest = _largest_series_dir(study_dir)
        if largest is None:
            n_fail += 1
            failed.append(study)
            continue
        out_path = out_root / f"{study}__{largest.name}.nii.gz"
        if out_path.exists() and out_path.stat().st_size > 0:
            n_skip += 1
            largest_ok = True
        else:
            largest_ok = _convert_series(largest, out_path) and _valid_nifti(out_path)
            if largest_ok:
                n_ok += 1
            else:
                if out_path.exists():
                    out_path.unlink()
                n_fail += 1
                failed.append(study)
                print(f"  FAIL {study}: dicom2nifti + SimpleITK both produced no valid NIfTI",
                      file=sys.stderr)

        # validate-then-rm: drop raw DICOM only once NIfTI confirmed
        if not keep_raw and largest_ok:
            shutil.rmtree(study_dir, ignore_errors=True)
        if (i + 1) % 200 == 0:
            logger.info("[rspect] progress %d/%d  ok=%d skip=%d fail=%d",
                        i + 1, len(studies), n_ok, n_skip, n_fail)
    finally:
        if zf is not None:
            zf.close()

    total = n_ok + n_skip + n_fail
    logger.info("[rspect] convert done: ok=%d skip=%d fail=%d (of %d studies this shard)",
                n_ok, n_skip, n_fail, total)
    if failed:
        print(f"[rspect] {len(failed)} studies unconvertible (justified-QC drops): "
              f"{failed[:20]}{' ...' if len(failed) > 20 else ''}", file=sys.stderr)
    if total and n_fail > max(5, total // 20):
        raise RuntimeError(
            f"[rspect] catastrophic conversion failure: {n_fail}/{total} studies (>5%) -- "
            f"aborting before labels (bad converter/upgrade?).")


def _ifirst(g, col: str) -> int:
    """First value of an exam-level column within a study group, as 0/1 int."""
    return int(g[col].iloc[0])


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 rspect_root: Path = Path("$DATA_ROOT/radiology/rspect"),
                 nifti_root: Path | None = None) -> None:
    """Rederive the RSPECT anchor + sidecar labels JSONs and split manifests from train.csv."""
    import pandas as pd  # noqa: PLC0415

    rspect_root = engine.expandvars(str(rspect_root))
    if nifti_root is None:
        nifti_root = rspect_root / "nifti"
    nifti_root = Path(nifti_root)
    out_dir = Path(out_dir)
    jsonl_out = out_dir / "rspect_jsonl"
    labels_out = out_dir / "rspect_labels.json"

    csv_path = rspect_root / "extracted" / "train.csv"
    logger.info("[rspect] reading %s", csv_path)
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in ("StudyInstanceUID", "SeriesInstanceUID", *_EXAM_LABEL_COLS)
                    if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"[rspect] train.csv missing required columns {missing_cols}; "
                         f"got {list(df.columns)}")

    # exam labels constant within a study -> first row per study
    exam = df.groupby("StudyInstanceUID", sort=True).first().reset_index()

    # drop indeterminate studies
    n_indet = int(exam["indeterminate"].astype(int).sum())
    exam = exam[exam["indeterminate"].astype(int) == 0].reset_index(drop=True)
    logger.info("[rspect] dropped %d indeterminate studies; %d remain", n_indet, len(exam))

    # largest series per study (matches the stage-time pick)
    counts = (df.groupby(["StudyInstanceUID", "SeriesInstanceUID"]).size()
                .reset_index(name="n"))
    largest = (counts.sort_values(["StudyInstanceUID", "n"], ascending=[True, False])
                     .drop_duplicates("StudyInstanceUID", keep="first"))
    study_to_series = dict(zip(largest["StudyInstanceUID"], largest["SeriesInstanceUID"]))

    nifti_files = {p.name[:-len(".nii.gz")]: p for p in nifti_root.glob("*.nii.gz")}
    logger.info("[rspect] NIfTI on disk: %d", len(nifti_files))

    def _b(row, col: str) -> int:
        return int(row[col])

    # per-study endpoint derivation (phenotypes forced 0 for PE-negative)
    records = []
    missing = 0
    for _, r in exam.iterrows():
        study = r["StudyInstanceUID"]
        series = study_to_series.get(study)
        sample = f"{study}__{series}"
        nii = nifti_files.get(sample)
        if nii is None:
            missing += 1
            continue
        pe = 1 - _b(r, "negative_exam_for_pe")
        rvlv = _b(r, "rv_lv_ratio_gte_1") if pe else 0
        central = _b(r, "central_pe") if pe else 0
        chronic = int(bool(_b(r, "chronic_pe") or _b(r, "acute_and_chronic_pe"))) if pe else 0
        bilateral = int(bool(_b(r, "leftsided_pe") and _b(r, "rightsided_pe"))) if pe else 0
        acute = int(pe and not (_b(r, "chronic_pe") or _b(r, "acute_and_chronic_pe")))
        records.append({
            "sample_name": sample,
            "patient_id": study,
            "nii_path": engine.data_root_relative(nii),
            "split": engine.stable_split(study),
            "pe_present": pe, "rv_lv_strain": rvlv, "central_pe": central,
            "chronic_pe": chronic, "bilateral_pe": bilateral, "pe_positive_acute": acute,
        })
    matched = len(records)
    logger.info("[rspect] matched %d studies (missing NIfTI: %d)", matched, missing)
    if not records:
        raise SystemExit("[rspect] no matched studies -- did stage() run?")

    # coverage guard
    prior = {s: (sum(1 for _ in (jsonl_out / f"{s}.jsonl").open())
                 if (jsonl_out / f"{s}.jsonl").exists() else 0)
             for s in ("train", "dev", "test")}
    prior_total = sum(prior.values())
    # new total can dip below prior 7,111 (indeterminate drop + unrecovered)
    floor = max(6500, prior_total - 400 - 300) if prior_total else 6500
    if matched < floor:
        raise SystemExit(f"[rspect] COVERAGE FLOOR: matched {matched} < {floor} "
                         f"(prior committed {prior_total}); refusing -- check the nifti/ tree.")
    pe_prev = sum(r["pe_present"] for r in records) / matched
    if not (0.25 <= pe_prev <= 0.40):
        raise SystemExit(f"[rspect] PE prevalence {pe_prev:.3f} outside [0.25,0.40] -- "
                         f"schema/derivation drift?")

    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for r in records:
        by_split[r["split"]].append(r)
    for s in ("train", "dev", "test"):
        if not by_split[s]:
            raise SystemExit(f"[rspect] empty '{s}' split -- refusing to write.")

    # invariant: rv_lv_strain positives == among-PE positives (RSNA hierarchy)
    n_rvlv = sum(r["rv_lv_strain"] for r in records)
    n_rvlv_among = sum(r["rv_lv_strain"] for r in records if r["pe_present"])
    if n_rvlv != n_rvlv_among:
        raise SystemExit(f"[rspect] hierarchy violation: rv_lv_strain pos {n_rvlv} != "
                         f"among-PE pos {n_rvlv_among} (an RV-strain positive is not PE+?)")

    jsonl_out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "dev", "test"):
        rows = [{"sample_name": r["sample_name"], "nii_path": r["nii_path"],
                 "pe_present": r["pe_present"]} for r in by_split[s]]
        engine.write_jsonl(rows, jsonl_out / f"{s}.jsonl")
        pos = sum(x["pe_present"] for x in rows)
        logger.info("[rspect]   %s: %d rows  PE+=%d (%.1f%%)", s, len(rows), pos,
                    100 * pos / max(len(rows), 1))

    def _emit(path: Path, qa: str, field: str, *, pe_only: bool = False) -> None:
        d: dict[str, dict] = {}
        for r in records:
            if pe_only and not r["pe_present"]:
                continue
            tgt = int(r[field])
            d[r["sample_name"]] = {
                "split": "valid" if r["split"] == "dev" else r["split"],
                "qa_results": {"default_qa": [{qa: tgt}]},
                "patient_id": r["patient_id"],
                "target": tgt,
            }
        engine.write_json(d, path)
        pos = sum(v["target"] for v in d.values())
        logger.info("[rspect] wrote %s: %d entries, %d pos (%.1f%%)",
                    path.name, len(d), pos, 100 * pos / max(len(d), 1))

    ev = out_dir
    _emit(labels_out, _QA_PE, "pe_present")
    _emit(ev / "rspect_labels__rv_lv_strain.json", _QA_RVLV, "rv_lv_strain")
    _emit(ev / "rspect_labels__central_pe.json", _QA_CENTRAL, "central_pe")
    _emit(ev / "rspect_labels__rv_lv_strain_among_pe.json", _QA_RVLV_AMONG, "rv_lv_strain",
          pe_only=True)
    _emit(ev / "rspect_labels__chronic_pe.json", _QA_CHRONIC, "chronic_pe")
    # materialized-only (committed, not scored)
    _emit(ev / "rspect_labels__bilateral_pe.json", _QA_BILATERAL, "bilateral_pe")
    _emit(ev / "rspect_labels__pe_positive_acute.json", _QA_ACUTE, "pe_positive_acute",
          pe_only=True)
    logger.info("[rspect] build_labels done: %d studies, PE prevalence %.1f%%",
                matched, 100 * pe_prev)


SPEC = DatasetSpec(
    name="rspect",
    access=Access.KAGGLE,
    modality="chest_ct",
    role="scored",
    source="rsna-str-pulmonary-embolism-detection",
    token_env=None,
    committed_outputs=(
        "data/evaluation/rspect_labels.json",
        "data/evaluation/rspect_labels__rv_lv_strain.json",
        "data/evaluation/rspect_labels__central_pe.json",
        "data/evaluation/rspect_labels__rv_lv_strain_among_pe.json",
        "data/evaluation/rspect_labels__chronic_pe.json",
        "data/evaluation/rspect_labels__bilateral_pe.json",
        "data/evaluation/rspect_labels__pe_positive_acute.json",
        "data/evaluation/rspect_jsonl/train.jsonl",
        "data/evaluation/rspect_jsonl/dev.jsonl",
        "data/evaluation/rspect_jsonl/test.jsonl",
    ),
    notes="Only contrast-enhanced cohort (CTPA). Labels REDERIVED in full from train.csv "
          "(no committed reuse): indeterminate studies dropped, hash-based stable_split. "
          "5 scored endpoints (pe_present, rv_lv_strain, central_pe, rv_lv_strain_among_pe, "
          "chronic_pe) + 2 materialized (bilateral_pe, pe_positive_acute). Kaggle competition "
          "download + per-study streaming DICOM->NIfTI (largest series, dicom2nifti "
          "reorient_nifti=True + SimpleITK recovery, validate-then-rm).",
    stage=stage,
    build_labels=build_labels,
)
