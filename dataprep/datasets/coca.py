"""COCA -- Stanford AIMI Coronary Calcium and Chest CT (Redivis) staging + labels."""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

logger = logging.getLogger(__name__)

_REDIVIS_ORG = "AIMI"
_REDIVIS_DATASET = "coca_coronary_calcium_and_chest_ct_s"
_REDIVIS_VERSION = "v1_0"
_QUALIFIED_REF = "aimi.coca_coronary_calcium_and_chest_ct_s:1vm9:v1_0"

# Binary CAC-present QA (TANGERINE Supp Data 3): total Agatston > 0.
_QA_KEY = engine.qa_key("coronary artery calcification")


def _download(dest: Path, *, table: str = "both", limit: int | None = None,
              dl_workers: int = 16) -> None:
    """Mirror the Redivis file tree under <dest>/<table>/<f.path> (threaded, idempotent)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(exist_ok=True)

    import redivis  # noqa: PLC0415
    ds = redivis.organization(_REDIVIS_ORG).dataset(_REDIVIS_DATASET, version=_REDIVIS_VERSION)
    print(f"[coca] resolved: {ds.qualified_reference if hasattr(ds, 'qualified_reference') else 'COCA v1_0'}")

    target_tables = ["gated", "non-gated"] if table == "both" else [table]
    n_done = n_skip = n_err = 0
    bytes_done = 0
    t0 = time.time()

    def _dl_one(f, target_root: Path):
        # use f.path not f.name (basename collisions)
        rel_path = str(f.path)
        tgt = (dest / "labels" / "scores.xlsx") if rel_path == "scores.xlsx" else (target_root / rel_path)
        exp = int(getattr(f, "size", 0) or 0)
        if tgt.exists() and tgt.stat().st_size > 0 and (exp == 0 or tgt.stat().st_size == exp):
            return "skip", 0
        if tgt.exists():
            tgt.unlink()
        tgt.parent.mkdir(parents=True, exist_ok=True)
        # bounded retry on transient GCS/Redivis errors
        last_exc = None
        for _attempt in range(4):
            try:
                f.download(str(tgt.parent))
                return "ok", int(getattr(f, "size", 0) or 0)
            except Exception as e:
                # concurrent duplicate may have written it -- skip
                if "already exist" in str(e).lower() and tgt.exists() and tgt.stat().st_size > 0:
                    return "skip", 0
                last_exc = e
                if tgt.exists() and tgt.stat().st_size == 0:
                    tgt.unlink()
                time.sleep(1.5 * (_attempt + 1))
        raise last_exc

    for tname in target_tables:
        t = next(tt for tt in ds.list_tables() if tt.name == tname)
        target_root = dest / tname
        target_root.mkdir(parents=True, exist_ok=True)
        files = list(t.list_files(max_results=limit) if limit else t.list_files())
        print(f"[coca] {tname}: {len(files)} files, downloading with {dl_workers} threads", flush=True)
        with ThreadPoolExecutor(max_workers=dl_workers) as ex:
            futs = {ex.submit(_dl_one, f, target_root): f for f in files}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    status, sz = fut.result()
                    if status == "ok":
                        n_done += 1
                        bytes_done += sz
                    else:
                        n_skip += 1
                except Exception as e:
                    n_err += 1
                    if n_err <= 5:
                        print(f"  ERR on {futs[fut].path}: {e}", file=sys.stderr)
                if i % 500 == 0:
                    el = max(time.time() - t0, 0.001)
                    rate = bytes_done / el / 1e6
                    print(f"  progress {n_done + n_skip + n_err} files (ok={n_done} skip={n_skip} "
                          f"err={n_err}) -- {bytes_done/1e9:.2f} GB at {rate:.1f} MB/s, "
                          f"elapsed {el:.0f}s", flush=True)

    print(f"[coca] download DONE -- ok={n_done} skip={n_skip} err={n_err}, "
          f"{bytes_done/1e9:.2f} GB in {time.time()-t0:.0f}s")
    if n_err:
        raise RuntimeError(f"[coca] {n_err} downloads failed")


def _patient_id_from_filename(name: str) -> str | None:
    # gated convention: IM-<patient_id>-<slice_number>.dcm
    if not name.startswith("IM-") or not name.endswith(".dcm"):
        return None
    parts = name[3:-4].split("-")
    if len(parts) < 2:
        return None
    return parts[0]


def _convert_one_patient(pid: str, dicom_files: list[Path], dest_root: Path) -> tuple[str, int, str]:
    """Convert all DICOMs of one patient, picking the largest series as canonical scan.nii.gz."""
    import dicom2nifti  # noqa: PLC0415
    import pydicom  # noqa: PLC0415

    out_pdir = dest_root / pid
    out_pdir.mkdir(parents=True, exist_ok=True)
    if (out_pdir / "scan.nii.gz").exists():
        return pid, len(dicom_files), "SKIP"

    # group by SeriesInstanceUID; non-gated DICOMs lack one (single-bucket fallback)
    by_series: dict[str, list[Path]] = defaultdict(list)
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if series_uid is None:
                series_uid = "_no_series_uid"
            by_series[str(series_uid)].append(f)
        except Exception:
            continue

    if not by_series:
        return pid, len(dicom_files), "FAIL_NO_SERIES"

    largest_series = None
    largest_size = 0
    largest_path = None
    with tempfile.TemporaryDirectory(prefix=f"coca-{pid}-") as tmp_root:
        for sidx, (series_uid, files) in enumerate(by_series.items()):
            tmp_series = Path(tmp_root) / f"series_{sidx}"
            tmp_series.mkdir()
            for f in files:
                (tmp_series / f.name).symlink_to(f.resolve())
            tmp_out = Path(tmp_root) / f"series_{sidx}_out"
            tmp_out.mkdir()
            try:
                dicom2nifti.convert_directory(str(tmp_series), str(tmp_out), reorient=False)
            except Exception as e:
                logger.debug("convert_directory failed for %s series %s: %s", pid, sidx, e)
                continue
            for nii in tmp_out.glob("*.nii.gz"):
                final_path = out_pdir / f"series_{sidx}.nii.gz"
                shutil.move(str(nii), str(final_path))
                if final_path.stat().st_size > largest_size:
                    largest_size = final_path.stat().st_size
                    largest_path = final_path

    if largest_path is None:
        # SimpleITK fallback: more lenient, stitches by InstanceNumber
        try:
            import SimpleITK as sitk  # noqa: PLC0415
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames([str(f) for f in sorted(dicom_files, key=lambda p: p.name)])
            img = reader.Execute()
            fallback_path = out_pdir / "series_sitk.nii.gz"
            sitk.WriteImage(img, str(fallback_path))
            largest_path = fallback_path
            largest_size = fallback_path.stat().st_size
        except Exception as e:
            logger.debug("SimpleITK fallback failed for %s: %s", pid, e)
            return pid, len(dicom_files), "FAIL_CONVERT"
    canonical = out_pdir / "scan.nii.gz"
    canonical.symlink_to(largest_path.name)  # relative
    return pid, len(dicom_files), f"OK ({len(by_series)} series, largest {largest_size//1024} KB)"


def _convert(staged_dir: Path, dest: Path, *, workers: int = 4, limit: int | None = None) -> None:
    """Convert staged COCA DICOM slices to per-patient NIfTI volumes under dest."""
    staged_dir, dest = Path(staged_dir), Path(dest)
    if not staged_dir.exists():
        raise SystemExit(f"[coca] staged dir not found: {staged_dir}")
    dest.mkdir(parents=True, exist_ok=True)

    # two layouts: flat gated (pid from filename), hierarchical non-gated (pid = first numeric dir)
    by_pid: dict[str, list[Path]] = defaultdict(list)
    for f in staged_dir.iterdir():
        if f.is_file() and f.name.endswith(".dcm"):
            pid = _patient_id_from_filename(f.name)
            if pid:
                by_pid[pid].append(f)
    if not by_pid:
        for f in staged_dir.rglob("*.dcm"):
            if not f.is_file():
                continue
            rel = f.relative_to(staged_dir)
            top = rel.parts[0] if rel.parts else None
            if top and top.isdigit():
                # pad to 4 digits to match label-builder layout
                by_pid[f"{int(top):04d}"].append(f)
    logger.info("found %d patients (%d total DICOM files)", len(by_pid), sum(len(v) for v in by_pid.values()))

    pids = sorted(by_pid.keys())
    if limit:
        pids = pids[:limit]

    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_convert_one_patient, pid, by_pid[pid], dest): pid for pid in pids}
        for i, fut in enumerate(as_completed(futs)):
            pid, n_files, status = fut.result()
            if status == "OK" or status.startswith("OK "):
                n_ok += 1
            elif status == "SKIP":
                n_skip += 1
            else:
                n_fail += 1
                logger.warning("FAIL %s (%d files): %s", pid, n_files, status)
            if (i + 1) % 25 == 0:
                logger.info("progress: %d/%d (ok=%d skip=%d fail=%d)", i + 1, len(pids), n_ok, n_skip, n_fail)

    logger.info("DONE: ok=%d skip=%d fail=%d / %d patients", n_ok, n_skip, n_fail, len(pids))
    if n_fail:
        # coverage guard is the real gate; raise only on catastrophic (>5%) failure
        logger.warning("[coca] %d/%d patients failed conversion -- excluded (build_labels coverage "
                       "guard is the gate)", n_fail, len(pids))
        if n_fail > max(5, len(pids) // 20):
            raise RuntimeError(f"[coca] {n_fail}/{len(pids)} patients failed conversion (>5%, aborting)")


def stage(dest: Path, *, token: str | None = None, workers: int = 4,
          table: str = "non-gated", limit: int | None = None,
          nifti_subdir: str = "nifti_nongated") -> None:
    """Download COCA from Redivis, then convert its DICOMs to per-patient NIfTI (non-gated default)."""
    token = token or engine.require_env(
        "REDIVIS_API_TOKEN", hint="Redivis API token for Stanford AIMI COCA "
        f"({_QUALIFIED_REF})")
    dest = Path(dest)
    _download(dest, table=table, limit=limit)
    _convert(dest / table, dest / nifti_subdir, workers=workers, limit=limit)


# QA strings below MUST byte-match the BINARY_COHORTS qa_key entries in cross_cohort_benchmark.py
_QA_SIG = "Is the coronary calcium burden clinically significant (Agatston >= 100)?"  # ACC/AHA 2018 statin cut
_QA_HIGH = "Is the coronary calcium burden high/severe (Agatston >= 400)?"            # Rumberger 1999 high/severe
_QA_LAD = "Is calcification present in the LAD (left anterior descending) artery? (0=No, 1=Yes)"  # CAC-DRS 4-territory
_QA_STRICT = "Is any coronary calcification present (any vessel Agatston > 0)?"
_QA_DRS = "CAC-DRS Agatston grade (0=A0[0], 1=A1[1-99], 2=A2[100-300], 3=A3[>300])"   # Hecht 2018 (deferred, unregistered)

# 3 registered binary sidecars beyond the total>0 anchor
_CAC_TASKS = (("significant", _QA_SIG), ("high", _QA_HIGH), ("lad", _QA_LAD))


def _cac_drs_grade(total: float) -> int:
    """CAC-DRS Agatston grade (Hecht 2018): A0=0, A1=1-99, A2=100-300, A3=>300."""
    if total <= 0:
        return 0
    if total <= 99:
        return 1
    if total <= 300:
        return 2
    return 3


# shared CAC endpoint-panel emitter (reused by coca + coca_gated)

def _derive_cac_endpoints(records, *, out_dir: Path, prefix: str,
                          extra_tasks=(), extra_records=None):
    """Write the <prefix>_labels__{significant,high,lad}.json + strict + CAC-DRS sidecars."""
    records = list(records)
    task_qa: dict[str, dict[str, dict]] = {name: {} for name, _ in _CAC_TASKS}
    strict_qa: dict[str, dict] = {}
    drs_qa: dict[str, dict] = {}
    extra_qa: dict[str, dict[str, dict]] = {name: {} for name, _, _ in extra_tasks}
    tcount = {name: 0 for name, _ in _CAC_TASKS}
    ecount = {name: 0 for name, _, _ in extra_tasks}
    for rec in records:
        pid, split = rec["pid"], rec["split"]
        total, lad_v, vmax = rec["total"], rec["lad"], rec["vmax"]
        tlab = {"significant": int(total >= 100), "high": int(total >= 400), "lad": int(lad_v > 0)}
        for name, tqa in _CAC_TASKS:
            task_qa[name][pid] = {"split": split,
                                  "qa_results": {"default_qa": [{tqa: tlab[name]}]},
                                  "patient_id": pid}
            tcount[name] += tlab[name]
        strict_qa[pid] = {"split": split,
                          "qa_results": {"default_qa": [{_QA_STRICT: int(vmax > 0)}]},
                          "patient_id": pid}
        grade = rec.get("drs_grade", _cac_drs_grade(total))
        drs_qa[pid] = {"split": split,
                       "qa_results": {"default_qa": [{_QA_DRS: grade}]},
                       "patient_id": pid, "raw_grade": grade}
        for name, eqa, fn in extra_tasks:
            lab = int(fn(rec))
            extra_qa[name][pid] = {"split": split,
                                   "qa_results": {"default_qa": [{eqa: lab}]},
                                   "patient_id": pid}
            ecount[name] += lab
    for name, _ in _CAC_TASKS:
        engine.write_json(task_qa[name], out_dir / f"{prefix}_labels__{name}.json")
    engine.write_json(strict_qa, out_dir / f"{prefix}_labels__cac_present_strict.json")
    engine.write_json(drs_qa, out_dir / f"{prefix}_labels__drs_ordinal.json")  # deferred, unregistered
    for name, eqa, fn in extra_tasks:
        engine.write_json(extra_qa[name], out_dir / f"{prefix}_labels__{name}.json")
    tcount.update(ecount)
    return tcount


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 scores_xlsx: Path | None = None,
                 staged_dir: Path | None = None,
                 jsonl_dir: Path | None = None) -> None:
    """Derive coca_labels.json + coca_jsonl/{train,dev,test}.jsonl (CAC-positive iff total > 0)."""
    import pandas as pd  # noqa: PLC0415

    out_dir = Path(out_dir)
    jsonl_dir = Path(jsonl_dir) if jsonl_dir is not None else out_dir / "coca_jsonl"
    if scores_xlsx is None:
        scores_xlsx = engine.expandvars("$DATA_ROOT/radiology/coca/labels/scores.xlsx")
    else:
        scores_xlsx = Path(scores_xlsx)
    if staged_dir is None:
        staged_dir = engine.expandvars("$DATA_ROOT/radiology/coca/nifti_nongated")
    else:
        staged_dir = Path(staged_dir)

    if not scores_xlsx.exists():
        raise SystemExit(f"[coca] scores.xlsx not found: {scores_xlsx}")

    df = pd.read_excel(scores_xlsx, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    print(f"  {len(df)} score rows: cols={list(df.columns)}", file=sys.stderr)
    df["filename"] = df["filename"].astype(str).str.strip()
    df["cac_pos"] = (df["total"].fillna(0) > 0).astype(int)
    n_pos = int(df["cac_pos"].sum())
    print(f"  CAC-positive (total>0): {n_pos}/{len(df)} = {100*n_pos/len(df):.1f}%", file=sys.stderr)

    qa = _QA_KEY
    # row counts before overwrite = coverage-regression floor
    prior = {s: (sum(1 for _ in (jsonl_dir / f"{s}.jsonl").open()) if (jsonl_dir / f"{s}.jsonl").exists() else 0)
             for s in ("train", "dev", "test")}
    prior_total = sum(prior.values())

    qa_results: dict[str, dict] = {}
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    endpoint_records: list[dict] = []
    matched = 0
    for _, r in df.iterrows():
        pid = r["filename"]
        label = int(r["cac_pos"])
        split = engine.stable_split(pid)
        # pid_padded = zero-padded numeric prefix ('1A' -> '0001')
        pid_match = re.match(r"^(\d+)", pid)
        pid_padded = f"{int(pid_match.group(1)):04d}" if pid_match else pid
        candidate_paths = [
            staged_dir / pid_padded / "series_sitk.nii.gz",
            staged_dir / pid_padded / "scan.nii.gz",
            staged_dir / pid / "scan.nii.gz",
            staged_dir / pid,
        ]
        path = next((p for p in candidate_paths if p.exists()), None)
        qa_results[pid] = {
            "split": split,
            "qa_results": {"default_qa": [{qa: label}]},
            "patient_id": pid,
            "agatston_total": float(r.get("total", 0)),
            "agatston_lca": float(r.get("LCA", 0)),
            "agatston_lad": float(r.get("LAD", 0)),
            "agatston_lcx": float(r.get("LCX", 0)),
            "agatston_rca": float(r.get("RCA", 0)),
        }
        total = float(r.get("total", 0) or 0)
        lad_v = float(r.get("LAD", 0) or 0)
        vmax = max(float(r.get("LCA", 0) or 0), lad_v,
                   float(r.get("LCX", 0) or 0), float(r.get("RCA", 0) or 0))
        endpoint_records.append({"pid": pid, "split": split, "total": total,
                                 "lad": lad_v, "vmax": vmax})
        if path is not None:
            row = {"sample_name": pid,
                   "nii_path": engine.data_root_relative(path.resolve())}
            by_split[split].append(row)
            matched += 1
    print(f"  matched against staged data: {matched}/{len(df)}", file=sys.stderr)

    # integrity check (log, do not fail): total vs 4-vessel sum
    _v = df[["LCA", "LAD", "LCX", "RCA"]].fillna(0.0)
    _tot = df["total"].fillna(0.0)
    n_sum_mismatch = int(((_tot - _v.sum(axis=1)).abs() > 1.0).sum())
    n_zero_vessel = int(((_tot == 0) & (_v.max(axis=1) > 0)).sum())
    print(f"  integrity: {n_sum_mismatch}/{len(df)} rows |total-sum(vessels)|>1.0 "
          f"(~50 expected Agatston rounding, logged not failed); "
          f"{n_zero_vessel} rows total==0 but max(vessel)>0 (anchor stays total>0)", file=sys.stderr)

    # coverage / truncation guards
    if prior_total and matched < prior_total:
        raise SystemExit(f"[coca] COVERAGE REGRESSION: matched {matched} < committed manifest {prior_total} "
                         f"({prior}). Refusing to overwrite -- check the staged nifti_nongated tree.")
    for s in ("train", "dev", "test"):
        if prior[s] and not by_split[s]:
            raise SystemExit(f"[coca] REFUSING to write empty '{s}' split over committed {prior[s]} rows.")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    # compact to match the committed file -- not engine.write_json
    (out_dir / "coca_labels.json").write_text(json.dumps(qa_results))
    for s, rows in by_split.items():
        engine.write_jsonl(rows, jsonl_dir / f"{s}.jsonl")
        npos = sum(1 for r in rows if qa_results[r["sample_name"]]["qa_results"]["default_qa"][0][qa])
        print(f"  {s}: n={len(rows)} ({npos} pos, {len(rows)-npos} neg)", file=sys.stderr)

    tcount = _derive_cac_endpoints(endpoint_records, out_dir=out_dir, prefix="coca")
    print(f"  tasks: significant={tcount['significant']} high={tcount['high']} lad={tcount['lad']} "
          f"(+ deferred drs_ordinal, strict sidecars)", file=sys.stderr)


SPEC = DatasetSpec(
    name="coca",
    access=Access.REDIVIS,
    modality="chest_ct",
    role="descriptive",
    source=_QUALIFIED_REF,
    token_env="REDIVIS_API_TOKEN",
    committed_outputs=(
        "data/evaluation/coca_labels.json",
        "data/evaluation/coca_labels__significant.json",
        "data/evaluation/coca_labels__high.json",
        "data/evaluation/coca_labels__lad.json",
        "data/evaluation/coca_labels__cac_present_strict.json",
        "data/evaluation/coca_labels__drs_ordinal.json",
        "data/evaluation/coca_jsonl/train.jsonl",
        "data/evaluation/coca_jsonl/dev.jsonl",
        "data/evaluation/coca_jsonl/test.jsonl",
    ),
    notes="Smallest cell (n=35 test, wide CIs); stage NON-GATED full-chest series "
          "(table='non-gated', nifti_subdir='nifti_nongated') with f.path not f.name "
          "(basename collisions); non-gated DICOMs lack SeriesInstanceUID (SimpleITK "
          "fallback -> series_sitk.nii.gz). Anchor = CAC-present (total Agatston > 0, "
          "TANGERINE parity); clinical panel = significant (>=100), high (>=400), lad "
          "(LAD>0); deferred CAC-DRS ordinal materialized but not registered.",
    stage=stage,
    build_labels=build_labels,
)
