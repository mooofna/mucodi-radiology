"""DLCS24 (Duke Lung Cancer Screening 2024) -- Zenodo staging + label derivation."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

# part-1 13799069 (public, 1,120) + parts 2/3 12784601 / 14659131 (+493, restricted).
_DEFAULT_RECORD_IDS: tuple[str, ...] = ("13799069", "12784601", "14659131")

# Must match the eval --qa-key.
_QA_KEY = "Was lung cancer subsequently diagnosed (any timing)? (0=No, 1=Yes)"


def _list_files(record_id: str, token: str | None = None) -> list[dict]:
    # manifest at the /files endpoint (Zenodo v2)
    api = f"https://zenodo.org/api/records/{record_id}/files"
    if token:
        api = f"{api}?access_token={token}"
    with urllib.request.urlopen(api, timeout=60) as r:
        data = json.loads(r.read())
    return data.get("entries") or data.get("files") or []


def _download_stream(url: str, tmp: Path, sz: int, retries: int) -> None:
    """Single-connection whole-file download into ``tmp`` (small files / CSVs)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp.stat().st_size != sz:
                tmp.unlink(missing_ok=True)
                raise IOError(f"size mismatch: {tmp.stat().st_size} vs {sz}")
            return
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(30 + 10 * attempt)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
        except (urllib.error.URLError, IOError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))


def _download_one(file_info: dict, dest_dir: Path, token: str | None = None,
                  retries: int = 6, seg_workers: int = 32,
                  chunk_bytes: int = 256 * 1024 * 1024) -> tuple[str, str]:
    key = file_info["key"]
    sz = int(file_info["size"])
    target = dest_dir / key
    if target.exists() and target.stat().st_size == sz:
        return key, "SKIP"
    target.parent.mkdir(parents=True, exist_ok=True)
    # download url at links.content (older records used links.self)
    url = file_info["links"].get("content") or file_info["links"]["self"]
    if token:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}access_token={token}"
    tmp = target.with_suffix(target.suffix + ".tmp")

    # small files (CSVs): single stream
    if sz <= chunk_bytes:
        try:
            _download_stream(url, tmp, sz, retries)
        except urllib.error.HTTPError as e:
            return key, f"FAIL HTTP {e.code}: {str(e)[:120]}"
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            return key, f"FAIL {type(e).__name__}: {str(e)[:120]}"
        tmp.rename(target)
        return key, "OK"

    # large files: parallel Range segments (Zenodo honours 206 despite Accept-Ranges: none)
    print(f"[dlcs24] downloading {key} ({sz / 1e9:.1f} GB, {seg_workers}-seg parallel)", flush=True)
    with tmp.open("wb") as f:
        f.truncate(sz)
    ranges = [(s, min(s + chunk_bytes, sz) - 1) for s in range(0, sz, chunk_bytes)]
    failed: list[tuple[int, int]] = []

    def _fetch(rng: tuple[int, int]) -> None:
        s, e = rng
        want = e - s + 1
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={s}-{e}"})
                with urllib.request.urlopen(req, timeout=600) as r, tmp.open("r+b") as f:
                    f.seek(s)
                    got = 0
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                if got != want:
                    raise IOError(f"short segment {got} != {want}")
                return
            except urllib.error.HTTPError as ex:
                if ex.code == 429 and attempt < retries - 1:
                    time.sleep(30 + 10 * attempt)
                    continue
                if attempt == retries - 1:
                    failed.append(rng)
                    return
                time.sleep(min(2 ** attempt, 30))
            except (urllib.error.URLError, IOError, TimeoutError):
                if attempt == retries - 1:
                    failed.append(rng)
                    return
                time.sleep(min(2 ** attempt, 30))

    with ThreadPoolExecutor(max_workers=seg_workers) as ex:
        list(ex.map(_fetch, ranges))
    if failed:
        return key, f"FAIL {len(failed)}/{len(ranges)} segments"
    if tmp.stat().st_size != sz:
        tmp.unlink(missing_ok=True)
        return key, "FAIL size mismatch"
    tmp.rename(target)
    return key, "OK"


def _extract(raw_dir: Path, nifti_root: Path) -> None:
    """Unzip DLCS_subset*.zip into nifti_root, flattening to <pid>.nii.gz (via Info-ZIP; Deflate64)."""
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    nifti_root.mkdir(parents=True, exist_ok=True)
    zips = sorted(raw_dir.rglob("DLCS_subset*.zip"))
    workers = min(int(os.environ.get("DLCS24_EXTRACT_WORKERS", "8")), len(zips) or 1)
    print(f"[dlcs24-extract] {len(zips)} subset zip(s) -> {nifti_root} "
          f"(via unzip; Deflate64; {workers}-way parallel)", flush=True)

    def _unzip_one(zp: Path) -> tuple[str, str]:
        # each subset zip has a disjoint pid set (parallel-safe)
        r = subprocess.run(
            ["unzip", "-j", "-n", "-qq", str(zp), "*.nii.gz", "*.nii", "-d", str(nifti_root)],
            capture_output=True, text=True)
        if r.returncode not in (0, 11):  # 0 = extracted/skipped, 11 = no matching members
            return zp.name, f"FAIL rc={r.returncode}: {(r.stderr or r.stdout)[:200]}"
        return zp.name, "OK"

    fails = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_unzip_one, z) for z in zips]):
            name, status = fut.result()
            print(f"  {name}: {status}", flush=True)
            if status != "OK":
                fails.append((name, status))
    if fails:
        raise RuntimeError(f"[dlcs24] unzip failed: {fails}")
    n = len(list(nifti_root.glob("*.nii*")))
    print(f"[dlcs24-extract] nifti dir now has {n} volumes", flush=True)


def stage(dest: Path, *, token: str | None = None, workers: int = 1,
          seg_workers: int = 32,
          record_ids: tuple[str, ...] = _DEFAULT_RECORD_IDS) -> None:
    """Download the Zenodo record(s) into <dest>/raw/zenodo_<rid>/ then unzip to nifti/."""
    dest = Path(dest)
    raw_dir = dest / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[tuple[dict, Path]] = []
    for rid in record_ids:
        try:
            files = _list_files(rid, token)
        except urllib.error.HTTPError as e:
            print(f"ERROR: Zenodo record {rid} not accessible: HTTP {e.code}", file=sys.stderr)
            continue
        part_dir = raw_dir / f"zenodo_{rid}"
        part_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dlcs24] record {rid}: {len(files)} files "
              f"(total {sum(f['size'] for f in files) / 1e9:.2f} GB)")
        for f in files:
            all_files.append((f, part_dir))

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one, f, d, token, seg_workers=seg_workers): f["key"]
                   for f, d in all_files}
        for fut in as_completed(futures):
            key, status = fut.result()
            print(f"  {status:>5}  {key}")
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                fail += 1
    print(f"[dlcs24] done: {ok} downloaded, {skip} skipped, {fail} failed")
    if fail:
        raise RuntimeError(f"[dlcs24] {fail} downloads failed")
    _extract(raw_dir, dest / "nifti")


def _find_csv(staged_dir: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(staged_dir.rglob(pat))
        if hits:
            return hits[0]
    return None


def _find_col(cols, candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


# QA keys are byte-matched by the cross-cohort benchmark cell definitions.
_QA_LR3 = "Is this a Lung-RADS positive screen (category >=3)? (0=No, 1=Yes)"
_QA_LR4 = "Is this Lung-RADS 4 (4A/4B/4X, suspicious for malignancy)? (0=No, 1=Yes)"
_QA_SCREEN = "Was screening-detected / within-1-year lung cancer diagnosed? (0=No, 1=Yes)"
_QA_ORDINAL = "Lung-RADS category ordinal (0=1, 1=2, 2=3, 3=4A, 4=4B, 5=4X)"
_LR_ORDER = ["1", "2", "3", "4A", "4B", "4X"]  # ascending severity -> ordinal 0..5


def _lr_norm(lr) -> str | None:
    """Normalize a Lung-RADS cell to one of 1/2/3/4A/4B/4X (or None)."""
    if lr is None:
        return None
    s = str(lr).strip().upper().replace("LUNG-RADS", "").replace("CATEGORY", "").replace(" ", "")
    if s in _LR_ORDER:
        return s
    if s[:1] in ("1", "2", "3"):
        return s[0]
    if s.startswith("4"):
        return s if s in _LR_ORDER else "4A"
    return None


def _screendetected(cancer_any: int, timing) -> int:
    """1 iff a screening-detected / within-one-year lung cancer (heuristic over the timing text)."""
    if not cancer_any:
        return 0
    t = str(timing).strip().lower()
    return int(("screen" in t) or ("baseline" in t) or ("prevalent" in t)
               or ("within" in t and "year" in t and ("1" in t or "one" in t)))


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 staged_dir: Path | None = None,
                 jsonl_dir: Path = Path("data/evaluation/dlcs24_jsonl"),
                 metadata_csv: Path | None = None,
                 qa_key: str = _QA_KEY) -> None:
    """Derive ``dlcs24_labels.json`` + ``dlcs24_jsonl/{train,dev,test}.jsonl`` from the staged tree."""
    import pandas as pd

    if staged_dir is None:
        staged_dir = Path(engine.expandvars("$DATA_ROOT/radiology/dlcs24"))  # CLI --label default
    else:
        staged_dir = Path(staged_dir)
    out_dir = Path(out_dir)
    jsonl_dir = Path(jsonl_dir)

    meta_csv = metadata_csv or _find_csv(staged_dir, ["*metadata*.csv"])
    if meta_csv is None or not Path(meta_csv).exists():
        raise SystemExit(f"metadata CSV not found under {staged_dir}; pass --metadata-csv")
    meta_csv = Path(meta_csv)
    print(f"  metadata: {meta_csv}", file=sys.stderr)

    meta = pd.read_csv(meta_csv)
    meta.columns = [c.strip() for c in meta.columns]
    pid_col = _find_col(meta.columns, ["patient-id", "patient_id", "pid", "subject_id", "patient"])
    timing_col = _find_col(meta.columns, ["lung cancer diagnosis and timing, if applicable"])
    if timing_col is None:
        timing_col = next((c for c in meta.columns
                           if "diagnosis" in c.lower() and "timing" in c.lower()), None)
    lung_rads_col = _find_col(meta.columns, ["lung-rads score", "lung_rads_score", "lung-rads", "lung_rads"])
    if pid_col is None or timing_col is None:
        raise SystemExit(
            f"could not find patient-id / diagnosis-timing columns in {meta_csv.name}; "
            f"columns: {list(meta.columns)}")
    print(f"  cols: pid={pid_col!r} timing={timing_col!r} lung_rads={lung_rads_col!r}", file=sys.stderr)

    nifti_root = staged_dir / "nifti"
    if not nifti_root.exists():
        raise SystemExit(
            f"nifti dir not found at {nifti_root}; run extraction first "
            "(unzip raw/zenodo_13799069/DLCS_subset*.zip into nifti/)")

    # prior row counts = coverage-regression floor
    prior = {s: (sum(1 for _ in (jsonl_dir / f"{s}.jsonl").open()) if (jsonl_dir / f"{s}.jsonl").exists() else 0)
             for s in ("train", "dev", "test")}
    prior_total = sum(prior.values())

    qa_results: dict[str, dict] = {}
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    lr3_qa: dict[str, dict] = {}
    lr4_qa: dict[str, dict] = {}
    ord_qa: dict[str, dict] = {}
    screen_qa: dict[str, dict] = {}
    cand = {"cancer_any": 0, "lr3": 0, "lr4": 0, "screendetected": 0}
    ord_hist: dict[int, int] = {}
    matched = skipped_no_nifti = n_lr_missing = 0

    for _, r in meta.iterrows():
        pid = str(r[pid_col])
        nii = nifti_root / f"{pid}.nii.gz"
        if not nii.exists():
            hits = (list(nifti_root.glob(f"{pid}.nii.gz")) + list(nifti_root.glob(f"{pid}.nii"))
                    + list(nifti_root.glob(f"{pid}_*.nii.gz")))
            if not hits:
                skipped_no_nifti += 1
                continue
            nii = hits[0]
        timing = r[timing_col]
        label = int(pd.notna(timing) and str(timing).strip() != "")
        lr = _lr_norm(r[lung_rads_col]) if lung_rads_col and pd.notna(r[lung_rads_col]) else None
        lr3 = int(lr is not None and int(lr[0]) >= 3)
        lr4 = int(lr is not None and lr[0] == "4")
        lr_ord = _LR_ORDER.index(lr) if lr in _LR_ORDER else None
        screen = _screendetected(label, timing)
        if lr is None:
            n_lr_missing += 1
        split = engine.stable_split(pid)
        matched += 1
        cand["cancer_any"] += label; cand["lr3"] += lr3; cand["lr4"] += lr4
        cand["screendetected"] += screen
        if lr_ord is not None:
            ord_hist[lr_ord] = ord_hist.get(lr_ord, 0) + 1
        nii_path = engine.data_root_relative(nii)
        qa_results[pid] = {
            "split": split,
            "qa_results": {"default_qa": [{qa_key: label}]},
            "patient_id": pid,
            "lung_rads": (str(r[lung_rads_col]) if lung_rads_col and pd.notna(r[lung_rads_col]) else None),
            "nii_path": nii_path,
        }
        lr3_qa[pid] = {"split": split, "qa_results": {"default_qa": [{_QA_LR3: lr3}]}, "patient_id": pid}
        lr4_qa[pid] = {"split": split, "qa_results": {"default_qa": [{_QA_LR4: lr4}]}, "patient_id": pid}
        screen_qa[pid] = {"split": split, "qa_results": {"default_qa": [{_QA_SCREEN: screen}]}, "patient_id": pid}
        ord_qa[pid] = {"split": split, "qa_results": {"default_qa": [{_QA_ORDINAL: lr_ord}]},
                       "patient_id": pid, "lung_rads": lr}
        by_split[split].append({"sample_name": pid, "nii_path": nii_path, "cancer_any": label})

    def _pct(c: int) -> str:
        return f"{c}/{matched} = {100 * c / max(1, matched):.1f}%"

    print(f"  matched {matched} patients ({skipped_no_nifti} metadata rows without a staged NIfTI; "
          f"{n_lr_missing} without a parseable Lung-RADS)", file=sys.stderr)
    print(f"  REGISTERED: cancer_any {_pct(cand['cancer_any'])} | lr>=3 {_pct(cand['lr3'])} | "
          f"lr4 {_pct(cand['lr4'])}", file=sys.stderr)
    print(f"  MATERIALIZED: screendetected {_pct(cand['screendetected'])} "
          f"(~{cand['screendetected'] / 5:.0f} pos/fold) | lung_rads ordinal hist {dict(sorted(ord_hist.items()))}",
          file=sys.stderr)
    print("  splits: " + ", ".join(f"{k}={len(v)} (pos {sum(x['cancer_any'] for x in v)})"
                                    for k, v in by_split.items()), file=sys.stderr)

    # coverage-floor guard: never shrink or empty a populated split
    if prior_total and matched < prior_total:
        raise SystemExit(f"[dlcs24] COVERAGE REGRESSION: matched {matched} < committed {prior_total} "
                         f"({prior}). Refusing to overwrite -- check the staged nifti/ tree.")
    for s in ("train", "dev", "test"):
        if prior[s] and not by_split[s]:
            raise SystemExit(f"[dlcs24] REFUSING to write empty '{s}' split over committed {prior[s]} rows.")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    engine.write_json(qa_results, out_dir / "dlcs24_labels.json")
    for split, rows in by_split.items():
        engine.write_jsonl(rows, jsonl_dir / f"{split}.jsonl")
        print(f"  wrote {jsonl_dir / f'{split}.jsonl'} ({len(rows)} rows)", file=sys.stderr)
    engine.write_json(lr3_qa, out_dir / "dlcs24_labels__lungrads3.json")
    engine.write_json(lr4_qa, out_dir / "dlcs24_labels__lungrads4.json")
    engine.write_json(ord_qa, out_dir / "dlcs24_labels__lungrads_ordinal.json")
    engine.write_json(screen_qa, out_dir / "dlcs24_labels__screendetected.json")
    print("  wrote sidecars: lungrads3, lungrads4 (registered); lungrads_ordinal, screendetected (materialized)",
          file=sys.stderr)


SPEC = DatasetSpec(
    name="dlcs24",
    access=Access.ZENODO,
    modality="chest_ct",
    role="candidate",
    source="13799069",
    token_env="ZENODO_TOKEN",
    committed_outputs=(
        "data/evaluation/dlcs24_labels.json",
        "data/evaluation/dlcs24_labels__lungrads3.json",
        "data/evaluation/dlcs24_labels__lungrads4.json",
        "data/evaluation/dlcs24_labels__lungrads_ordinal.json",
        "data/evaluation/dlcs24_labels__screendetected.json",
        "data/evaluation/dlcs24_jsonl/train.jsonl",
        "data/evaluation/dlcs24_jsonl/dev.jsonl",
        "data/evaluation/dlcs24_jsonl/test.jsonl",
    ),
    notes="Zenodo part-1 (13799069) effectively public (DLCS_0001..1120, 1,120 of "
          "1,613); parts 2/3 (12784601/14659131) restricted -> ZENODO_TOKEN. NIfTI "
          "not DICOM; official benchmark_split unusable (no test pts) -> hash(pid)%100 "
          "70/15/15; binary any-timing lung-cancer dx.",
    stage=stage,
    build_labels=build_labels,
)
