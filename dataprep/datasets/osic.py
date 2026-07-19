"""OSIC Pulmonary Fibrosis Progression (Kaggle) -- staging + multi-task FVC-decline label panel."""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

_COMPETITION_SLUG = "osic-pulmonary-fibrosis-progression"

# anchor: >=10% relative FVC decline /52wk (du Bois 2011)
_QA = "Did this IPF patient experience >=10% FVC decline within 52 weeks of baseline?"

# registered = _QA + _QA_PPF + _QA_MOD; materialized = _QA_SEV / _QA_ABS / _QA_GRADE
_QA_PPF = "Did FVC%predicted decline by >=5 absolute points within 52 weeks (ATS/ERS 2022 PPF physiological progression)?"
_QA_MOD = "Is baseline FVC%predicted < 75% (GAP physiology-domain impairment)?"
_QA_SEV = "Is baseline FVC%predicted < 50% (severe restriction)?"
_QA_ABS = "Did FVC%predicted decline by >=10 absolute points within 52 weeks?"
_QA_GRADE = "du Bois FVC-decline grade (0=stable<5%, 1=marginal 5-10%, 2=significant>=10%)"


def _download_and_extract(raw_dir: Path, *, kaggle_cli: str = "kaggle",
                          skip_download: bool = False) -> None:
    """Download the competition zip into ``raw_dir`` and extract it in place."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{_COMPETITION_SLUG}.zip"

    if not skip_download:
        if shutil.which(kaggle_cli) is None:
            raise RuntimeError(
                f"kaggle CLI not found ({kaggle_cli}); install via "
                f"`uv pip install kaggle` and ensure ~/.kaggle/access_token "
                f"(KGAT_ format) is present per kaggle-auth.md memory")
        print(f"[osic] downloading {_COMPETITION_SLUG} to {raw_dir}")
        cmd = [kaggle_cli, "competitions", "download",
               "-c", _COMPETITION_SLUG, "-p", str(raw_dir)]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise RuntimeError(f"kaggle CLI returned {rc}")

    if not zip_path.exists():
        raise RuntimeError(f"expected {zip_path} after download; not found")

    print(f"[osic] extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    size_gb = sum(p.stat().st_size for p in raw_dir.rglob("*") if p.is_file()) / 1e9
    print(f"[osic] extracted to {raw_dir}; size={size_gb:.2f} GB")

    train_dir = raw_dir / "train"
    if train_dir.exists():
        n_patients = sum(1 for p in train_dir.iterdir() if p.is_dir())
        print(f"[osic] {n_patients} train patients with DICOM dirs at {train_dir}")


def _sitk_series_to_nifti(pdir: Path, out: Path) -> bool:
    """SimpleITK ``ImageSeriesReader`` fallback for DICOM dirs dicom2nifti refuses (irregular geometry)."""
    try:
        import SimpleITK as sitk  # noqa: PLC0415
    except ImportError:  # pragma: no cover - environment-dependent
        return False
    reader = sitk.ImageSeriesReader()
    names: list[str] = []
    try:
        sids = reader.GetGDCMSeriesIDs(str(pdir))
        if sids:
            names = max((reader.GetGDCMSeriesFileNames(str(pdir), s) for s in sids), key=len)
    except Exception:  # noqa: BLE001
        names = []
    if not names:
        names = sorted(str(f) for f in pdir.iterdir() if f.is_file())
    if not names:
        return False
    try:
        reader.SetFileNames(names)
        sitk.WriteImage(reader.Execute(), str(out))
    except Exception:  # noqa: BLE001
        return False
    return out.exists() and out.stat().st_size > 0


def _convert_dicom_to_nifti(dicom_root: Path, nifti_root: Path) -> list[str]:
    """Convert per-patient DICOM dirs to one NIfTI each (dicom2nifti + SimpleITK fallback, idempotent)."""
    try:
        import dicom2nifti  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise RuntimeError("dicom2nifti not installed; pip install dicom2nifti") from e

    nifti_root.mkdir(parents=True, exist_ok=True)
    patient_dirs = sorted(p for p in dicom_root.iterdir() if p.is_dir())
    print(f"[osic-convert] {len(patient_dirs)} patient DICOM dirs at {dicom_root}")

    ok = skip = recovered = 0
    failed: list[str] = []
    for pdir in patient_dirs:
        pid = pdir.name
        out = nifti_root / f"{pid}.nii.gz"
        if out.exists() and out.stat().st_size > 0:
            skip += 1
            continue
        # dicom2nifti writes a dir; use a per-patient temp then rename
        tmp_dir = nifti_root / f"_{pid}_tmp"
        tmp_dir.mkdir(exist_ok=True)
        produced = False
        try:
            dicom2nifti.convert_directory(str(pdir), str(tmp_dir),
                                          compression=True, reorient=False)
            niftis = sorted(tmp_dir.glob("*.nii.gz"), key=lambda p: -p.stat().st_size)
            if niftis:
                niftis[0].rename(out)
                produced = True
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  dicom2nifti failed {pid}: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if not produced:
            if _sitk_series_to_nifti(pdir, out):
                recovered += 1
                print(f"  RECOVERED {pid} via SimpleITK", file=sys.stderr)
            else:
                if out.exists():
                    out.unlink()
                failed.append(pid)
                print(f"  FAIL  {pid}: dicom2nifti + SimpleITK both produced no NIfTI", file=sys.stderr)
        if (ok + recovered + len(failed)) % 25 == 0:
            print(f"  progress: {ok} d2n, {recovered} sitk, {len(failed)} failed, {skip} skipped")
    print(f"[osic-convert] done: {ok} dicom2nifti + {recovered} SimpleITK-recovered "
          f"= {ok + recovered} converted, {skip} skipped, {len(failed)} unrecoverable")
    if failed:
        print(f"[osic-convert] unrecoverable PIDs (justified-QC drops): {failed}", file=sys.stderr)
    return failed


def stage(dest: Path, *, token: str | None = None, workers: int = 8,
          kaggle_cli: str = "kaggle", skip_download: bool = False) -> None:
    """Download + extract the Kaggle competition, then DICOM -> NIfTI (raw/ + nifti/ under ``dest``)."""
    dest = Path(dest)
    raw_dir = dest / "raw"
    _download_and_extract(raw_dir, kaggle_cli=kaggle_cli, skip_download=skip_download)
    failed = _convert_dicom_to_nifti(raw_dir / "train", dest / "nifti")
    if failed:
        print(f"[osic] {len(failed)} patients unconvertible after dicom2nifti + SimpleITK "
              f"(documented QC drops): {failed}", file=sys.stderr)


def _fvc_decline_binary(group, decline_threshold_pct: float = 10.0,
                        window_weeks: float = 52.0) -> int:
    """1 iff FVC drops by >= ``decline_threshold_pct`` within ``window_weeks`` of baseline (single-visit=0)."""
    g = group.sort_values("Weeks")
    if len(g) < 2:
        return 0
    baseline_fvc = g.iloc[0]["FVC"]
    baseline_week = g.iloc[0]["Weeks"]
    within = g[(g["Weeks"] - baseline_week) <= window_weeks]
    if within.empty:
        return 0
    min_fvc = within["FVC"].min()
    pct_decline = 100.0 * (baseline_fvc - min_fvc) / max(1.0, baseline_fvc)
    return int(pct_decline >= decline_threshold_pct)


def _fvc_pct_pred_decline(group, min_points: float, window_weeks: float = 52.0) -> int:
    """1 iff FVC%predicted drops by >= ``min_points`` ABSOLUTE points in the window (single-visit=0)."""
    g = group.sort_values("Weeks")
    if len(g) < 2:
        return 0
    base_pct = float(g.iloc[0].get("Percent", 0.0))
    base_week = g.iloc[0]["Weeks"]
    within = g[(g["Weeks"] - base_week) <= window_weeks]
    if within.empty:
        return 0
    return int((base_pct - float(within["Percent"].min())) >= min_points)


def _progression_grade(group, window_weeks: float = 52.0) -> int:
    """du Bois 2011 3-level FVC-decline ordinal (0=stable<5%, 1=marginal 5-10%, 2=significant>=10%)."""
    g = group.sort_values("Weeks")
    if len(g) < 2:
        return 0
    base_fvc = g.iloc[0]["FVC"]
    base_week = g.iloc[0]["Weeks"]
    within = g[(g["Weeks"] - base_week) <= window_weeks]
    if within.empty:
        return 0
    rel = 100.0 * (base_fvc - within["FVC"].min()) / max(1.0, base_fvc)
    return 2 if rel >= 10.0 else (1 if rel >= 5.0 else 0)


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 staged_dir: Path | None = None,
                 jsonl_dir: Path = Path("data/evaluation/osic_jsonl"),
                 train_csv: Path | None = None,
                 decline_threshold_pct: float = 10.0,
                 window_weeks: float = 52.0) -> None:
    """Derive ``osic_labels.json`` + ``osic_jsonl/`` from ``train.csv`` (only patients with a NIfTI on disk)."""
    import pandas as pd

    if staged_dir is None:
        staged_dir = Path(engine.expandvars("$DATA_ROOT/radiology/osic"))
    else:
        staged_dir = Path(staged_dir)
    train_csv = Path(train_csv) if train_csv else (staged_dir / "raw" / "train.csv")
    if not train_csv.exists():
        raise SystemExit(f"train.csv not found: {train_csv}")
    nifti_dir = staged_dir / "nifti"
    if not nifti_dir.exists():
        raise SystemExit(f"nifti dir not found: {nifti_dir}")

    out_dir, jsonl_dir = Path(out_dir), Path(jsonl_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    df.columns = [c.strip() for c in df.columns]
    qa_results: dict[str, dict] = {}
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    # multi-task sidecars: registered (ppf, moderate) + materialized (severe, abs10, grade)
    ppf_qa: dict[str, dict] = {}
    mod_qa: dict[str, dict] = {}
    sev_qa: dict[str, dict] = {}
    abs_qa: dict[str, dict] = {}
    grade_qa: dict[str, dict] = {}
    cand = {"anchor_rel10": 0, "ppf_abs5": 0, "moderate_lt75": 0, "severe_lt50": 0, "abs10": 0}
    grade_hist = {0: 0, 1: 0, 2: 0}

    # prior manifest row counts = coverage-regression floor
    prior = {s: (sum(1 for _ in (jsonl_dir / f"{s}.jsonl").open()) if (jsonl_dir / f"{s}.jsonl").exists() else 0)
             for s in ("train", "dev", "test")}
    prior_total = sum(prior.values())

    matched = 0
    for pid, group in df.groupby("Patient"):
        nii_files = list(nifti_dir.glob(f"{pid}.nii.gz")) + list(nifti_dir.glob(f"{pid}.nii"))
        if not nii_files:
            continue  # NIfTI not converted (QC drop)
        nii_path = engine.data_root_relative(nii_files[0])
        split = engine.stable_split(pid)
        first_row = group.sort_values("Weeks").iloc[0]
        base_pct = float(first_row.get("Percent", 0.0))

        label = _fvc_decline_binary(group, decline_threshold_pct, window_weeks)
        ppf = _fvc_pct_pred_decline(group, 5.0, window_weeks)
        moderate = int(base_pct < 75.0)
        severe = int(base_pct < 50.0)
        abs10 = _fvc_pct_pred_decline(group, 10.0, window_weeks)
        grade = _progression_grade(group, window_weeks)

        cand["anchor_rel10"] += label; cand["ppf_abs5"] += ppf
        cand["moderate_lt75"] += moderate; cand["severe_lt50"] += severe; cand["abs10"] += abs10
        grade_hist[grade] += 1

        qa_results[pid] = {
            "split": split,
            "qa_results": {"default_qa": [{_QA: label}]},
            "patient_id": pid,
            "baseline_fvc": float(first_row["FVC"]),
            "baseline_percent_predicted": base_pct,
            "age": int(first_row.get("Age", 0)),
            "sex": first_row.get("Sex", ""),
            "smoking_status": first_row.get("SmokingStatus", ""),
            "nii_path": nii_path,
        }
        ppf_qa[pid] = {"split": split, "qa_results": {"default_qa": [{_QA_PPF: ppf}]}, "patient_id": pid}
        mod_qa[pid] = {"split": split, "qa_results": {"default_qa": [{_QA_MOD: moderate}]}, "patient_id": pid}
        sev_qa[pid] = {"split": split, "qa_results": {"default_qa": [{_QA_SEV: severe}]}, "patient_id": pid}
        abs_qa[pid] = {"split": split, "qa_results": {"default_qa": [{_QA_ABS: abs10}]}, "patient_id": pid}
        grade_qa[pid] = {"split": split, "qa_results": {"default_qa": [{_QA_GRADE: grade}]},
                         "patient_id": pid, "raw_grade": grade}
        by_split[split].append({
            "sample_name": pid,
            "nii_path": nii_path,
            "fvc_decline_ge10pct_1y": label,
        })
        matched += 1

    n_total = matched

    def _pct(c: int) -> str:
        return f"{c}/{n_total} = {100 * c / max(1, n_total):.1f}%"

    print(f"  matched (NIfTI present): {matched}", file=sys.stderr)
    print(f"  REGISTERED: anchor rel>=10% {_pct(cand['anchor_rel10'])} | "
          f"ppf abs>=5pt {_pct(cand['ppf_abs5'])} | moderate <75% {_pct(cand['moderate_lt75'])}",
          file=sys.stderr)
    print(f"  MATERIALIZED: severe <50% {_pct(cand['severe_lt50'])} | abs>=10pt {_pct(cand['abs10'])} | "
          f"grade hist {grade_hist}", file=sys.stderr)
    print(f"  splits: train={len(by_split['train'])}, dev={len(by_split['dev'])}, "
          f"test={len(by_split['test'])}", file=sys.stderr)

    # coverage guard: matched must stay >= prior floor
    if prior_total and matched < prior_total:
        raise SystemExit(f"[osic] COVERAGE REGRESSION: matched {matched} < committed manifest {prior_total} "
                         f"({prior}). Refusing to overwrite -- check the staged nifti/ tree.")
    for s in ("train", "dev", "test"):
        if prior[s] and not by_split[s]:
            raise SystemExit(f"[osic] REFUSING to write empty '{s}' split over committed {prior[s]} rows.")

    engine.write_json(qa_results, out_dir / "osic_labels.json")
    for split, rows in by_split.items():
        engine.write_jsonl(rows, jsonl_dir / f"{split}.jsonl")
        print(f"  wrote {jsonl_dir / f'{split}.jsonl'} ({len(rows)} rows)", file=sys.stderr)
    engine.write_json(ppf_qa, out_dir / "osic_labels__ppf.json")
    engine.write_json(mod_qa, out_dir / "osic_labels__moderate.json")
    engine.write_json(sev_qa, out_dir / "osic_labels__severe.json")
    engine.write_json(abs_qa, out_dir / "osic_labels__abs10.json")
    engine.write_json(grade_qa, out_dir / "osic_labels__progression_grade.json")
    print("  wrote sidecars: ppf, moderate (registered); severe, abs10, progression_grade (materialized)",
          file=sys.stderr)


SPEC = DatasetSpec(
    name="osic",
    access=Access.KAGGLE,
    modality="chest_ct",
    role="candidate",
    source=_COMPETITION_SLUG,
    token_env=None,
    committed_outputs=(
        "data/evaluation/osic_labels.json",
        "data/evaluation/osic_labels__ppf.json",
        "data/evaluation/osic_labels__moderate.json",
        "data/evaluation/osic_labels__severe.json",
        "data/evaluation/osic_labels__abs10.json",
        "data/evaluation/osic_labels__progression_grade.json",
        "data/evaluation/osic_jsonl/train.jsonl",
        "data/evaluation/osic_jsonl/dev.jsonl",
        "data/evaluation/osic_jsonl/test.jsonl",
    ),
    notes="Kaggle osic-pulmonary-fibrosis-progression (rules pre-accepted); only public "
          "IPF cohort. Multi-task spirometry panel from train.csv: REGISTERED = anchor "
          "(>=10% relative FVC decline /52wk, du Bois 2011) + ppf (>=5-abs-pt FVC%pred decline, "
          "2022 ATS/ERS PPF) + moderate (baseline FVC%pred <75%, GAP); MATERIALIZED = severe "
          "(<50%, degenerate ~0.7%), abs10 (>=10-abs-pt), progression_grade (du Bois 3-level "
          "ordinal). DICOM->NIfTI via dicom2nifti (reorient=False) with a SimpleITK "
          "ImageSeriesReader fallback for OSIC's irregular slice geometry. Anchor grows with "
          "recovered patients (coverage-floor guard, not byte-identity).",
    stage=stage,
    build_labels=build_labels,
)
