"""LNDb (Zenodo 6613714) staging + author-GT Fleischner/nodule label panel."""
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
from dataprep.datasets.dlcs24 import _download_one as _seg_download_one  # 32-seg Zenodo dl

_ZENODO_RECORD_ID = "6613714"
_API = f"https://zenodo.org/api/records/{_ZENODO_RECORD_ID}"

# QA keys must byte-match the cross-cohort benchmark cell definitions
_QA_ACTIONABLE = ("Does this chest CT warrant nodule follow-up "
                  "(Fleischner 2017 category 1/2/3 vs 0)? (0=No, 1=Yes)")
_QA_URGENT = ("Is this the highest Fleischner follow-up urgency "
              "(category 3: 3-month CT / PET-CT / biopsy)? (0=No, 1=Yes)")
_QA_CATEGORY = "Fleischner 2017 follow-up category (0=no routine follow-up, 1, 2, 3=highest urgency)"
_QA_SUBSOLID = ("Does this chest CT contain a subsolid nodule "
                "(author non-solid/part-solid texture class, mean rating < 11/3)? (0=No, 1=Yes)")
_QA_LARGE = ("Does this chest CT contain a large nodule "
             "(author volume class, consolidated volume >= 250 mm3)? (0=No, 1=Yes)")
_QA_PRESENT = "Does this chest CT contain a pulmonary nodule (>= 1 consolidated true nodule)? (0=No, 1=Yes)"
_QA_AGREEMENT = ("Does this chest CT contain a multi-reader-consensus nodule "
                 "(>= 2 radiologists agreed)? (0=No, 1=Yes)")

# author-exact thresholds, verbatim from the LNDb reference scripts
_TEX_SOLID_THR = 11.0 / 3.0   # Text >= 11/3 -> solid, else subsolid
_VOL_LARGE_THR = 250.0        # consolidated volume mm^3, >= 250 -> "large"


def _list_files() -> list[dict]:
    with urllib.request.urlopen(_API, timeout=60) as r:
        return json.loads(r.read())["files"]


def _download_one(file_info: dict, dest_dir: Path, retries: int = 6) -> tuple[str, str]:
    key = file_info["key"]
    sz = file_info["size"]
    target = dest_dir / key
    if target.exists() and target.stat().st_size == sz:
        return key, "SKIP"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = file_info["links"]["self"]
    for attempt in range(retries):
        try:
            tmp = target.with_suffix(target.suffix + ".tmp")
            with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp.stat().st_size != sz:
                tmp.unlink(missing_ok=True)
                raise IOError(f"size mismatch: {tmp.stat().st_size} vs expected {sz}")
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


def _find_unrar() -> str:
    """Locate the `unrar` RAR extractor on PATH."""
    from shutil import which
    if which("unrar"):
        return "unrar"
    raise SystemExit(
        "[lndb] unrar not found. libarchive cannot decompress LNDb's compressed RAR4 CTs; "
        "install unrar (build it from rarlab.com's unrarsrc). unrar's license permits extraction.")


def _preflight_extractor() -> None:
    """Ensure the RAR extractor is present before a multi-GB download."""
    _find_unrar()


def _extract_archive(archive: Path, out_dir: Path) -> int:
    """Extract a .rar (bundled unrar; libarchive can't decode LNDb's RAR4) or .zip into out_dir."""
    import subprocess
    import zipfile
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    else:
        unrar = _find_unrar()
        r = subprocess.run([unrar, "x", "-idq", "-o+", str(archive), str(out_dir) + "/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"[lndb] unrar failed on {archive.name}: {(r.stderr or r.stdout)[-400:]}")
    return sum(1 for _ in out_dir.rglob("*") if _.is_file())


def _convert_mhd_to_nifti(raw_dir: Path, nifti_dir: Path) -> int:
    """SimpleITK-convert every CT ``LNDb-XXXX.mhd`` to ``nifti/LNDb-XXXX.nii.gz`` (skips mask volumes)."""
    import SimpleITK as sitk
    nifti_dir.mkdir(parents=True, exist_ok=True)
    mhds = [m for m in sorted(raw_dir.rglob("LNDb-*.mhd"))
            if "_rad" not in m.stem and "mask" not in str(m.parent).lower()]
    ok = fail = skip = 0
    for m in mhds:
        out = nifti_dir / (m.stem + ".nii.gz")
        if out.exists() and out.stat().st_size > 0:
            skip += 1
            continue
        try:
            sitk.WriteImage(sitk.ReadImage(str(m)), str(out))
            ok += 1
        except Exception as exc:  # pragma: no cover
            print(f"  [lndb] convert FAILED {m.name}: {exc}", file=sys.stderr)
            fail += 1
    print(f"[lndb] convert: {ok} written, {skip} skipped, {fail} failed "
          f"({len(mhds)} CT .mhd found)", file=sys.stderr)
    return ok + skip


def stage(dest: Path, *, token: str | None = None, workers: int = 4) -> None:
    """Download LNDb imaging + CSVs into ``<dest>/{nifti,labels,metadata}/`` (public, no token, idempotent)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    raw_dir = dest / "raw"
    nifti_dir = dest / "nifti"
    label_dir = dest / "labels"
    meta_dir = dest / "metadata"
    for d in (raw_dir, nifti_dir, label_dir, meta_dir):
        d.mkdir(exist_ok=True)
    _preflight_extractor()

    files = _list_files()
    def _route(key: str) -> Path:
        if key.endswith(".csv"):
            return label_dir
        if key.endswith((".rar", ".zip")):
            return raw_dir
        return meta_dir
    todo = [(f, _route(f["key"])) for f in files]
    print(f"[lndb] {len(files)} files (total {sum(f['size'] for f in files) / 1e9:.2f} GB)")
    ok = skip = fail = 0
    # big data*.rar fans out to 32 range-segments; small files single-stream
    for f, d in todo:
        key, status = _seg_download_one(f, d)
        print(f"  {status:>5}  {key}", flush=True)
        if status == "OK":
            ok += 1
        elif status == "SKIP":
            skip += 1
        else:
            fail += 1
    print(f"[lndb] download: {ok} downloaded, {skip} skipped, {fail} failed")
    if fail:
        raise RuntimeError(f"[lndb] {fail} downloads failed")

    # extract .rar (MetaImage) -> raw/, .zip -> labels/metadata
    for arc in sorted(raw_dir.glob("*.rar")):
        n = _extract_archive(arc, raw_dir)
        print(f"[lndb] extracted {arc.name} ({n} files under raw/)")
    for arc in sorted(raw_dir.glob("*.zip")):
        target = label_dir if "csv" in arc.name.lower() else meta_dir
        n = _extract_archive(arc, target)
        print(f"[lndb] extracted {arc.name} -> {target.name}/ ({n} files)")

    n_conv = _convert_mhd_to_nifti(raw_dir, nifti_dir)
    print(f"[lndb] staged: {n_conv} NIfTI volumes in {nifti_dir}")


def _true_nodule_groups(gt) -> dict[int, "object"]:
    """Map LNDbID -> the consolidated TRUE-nodule rows (Nodule==1) for that scan."""
    true_nod = gt[gt["Nodule"] == 1]
    return {int(k): v for k, v in true_nod.groupby("LNDbID")}


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 staged_dir: Path | None = None,
                 jsonl_dir: Path = Path("data/evaluation/lndb_jsonl")) -> None:
    """Derive the LNDb author-GT label panel + ``lndb_jsonl/{train,dev,test}.jsonl``."""
    import pandas as pd

    staged_dir = Path(staged_dir) if staged_dir is not None \
        else engine.expandvars("$DATA_ROOT/radiology/lndb")
    out_dir, jsonl_dir = Path(out_dir), Path(jsonl_dir)
    nifti_dir = staged_dir / "nifti"
    label_dir = staged_dir / "labels"
    if not nifti_dir.exists():
        raise SystemExit(f"nifti dir not found: {nifti_dir}")

    # author sub-challenge-C GT: per-scan Fleischner 2017 category 0-3
    fl_csv = label_dir / "trainFleischner.csv"
    if not fl_csv.exists():
        raise SystemExit(f"trainFleischner.csv not found in {label_dir}; LNDb release should ship it")
    fl = pd.read_csv(fl_csv)
    fl.columns = [c.strip() for c in fl.columns]
    fl_map = {int(i): int(c) for i, c in zip(fl["LNDbID"], fl["Fleischner"])}
    print(f"  Fleischner GT: {len(fl_map)} scans", file=sys.stderr)

    # consolidated per-nodule GT for the nodule-characteristic cells
    gt_csv = label_dir / "trainNodules_gt.csv"
    if not gt_csv.exists():
        raise SystemExit(f"trainNodules_gt.csv not found in {label_dir}; needed for nodule labels")
    gt = pd.read_csv(gt_csv)
    gt.columns = [c.strip() for c in gt.columns]
    nod_groups = _true_nodule_groups(gt)
    print(f"  consolidated nodule GT: {len(gt)} findings, "
          f"{len(nod_groups)} scans with >=1 true nodule", file=sys.stderr)

    # prior manifest floor -> coverage-regression guard on re-run
    prior = {s: (sum(1 for _ in (jsonl_dir / f"{s}.jsonl").open()) if (jsonl_dir / f"{s}.jsonl").exists() else 0)
             for s in ("train", "dev", "test")}
    prior_total = sum(prior.values())

    prim_qa: dict[str, dict] = {}
    urgent_qa: dict[str, dict] = {}
    category_qa: dict[str, dict] = {}
    subsolid_qa: dict[str, dict] = {}
    large_qa: dict[str, dict] = {}
    present_qa: dict[str, dict] = {}
    agreement_qa: dict[str, dict] = {}
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    cnt = {"actionable": 0, "urgent": 0, "subsolid": 0, "large": 0, "present": 0, "agreement": 0}
    cat_hist: dict[int, int] = {}
    matched = skipped_no_nifti = 0

    for lndb_id in sorted(fl_map):
        scan_id = f"LNDb-{lndb_id:04d}"
        nii_gz = nifti_dir / f"{scan_id}.nii.gz"
        if not nii_gz.exists():
            other = sorted(nifti_dir.glob(f"{scan_id}.*"))
            if not other:
                skipped_no_nifti += 1
                continue
            nii_gz = other[0]
        nii_path = engine.data_root_relative(nii_gz)
        split = engine.stable_split(scan_id)
        matched += 1

        fcat = fl_map[lndb_id]
        actionable = int(fcat in (1, 2, 3))
        urgent = int(fcat == 3)
        cat_hist[fcat] = cat_hist.get(fcat, 0) + 1

        sub = nod_groups.get(lndb_id)
        present = int(sub is not None and len(sub) > 0)
        if present:
            subsolid = int((sub["Text"] < _TEX_SOLID_THR).any())
            large = int((sub["Volume"] >= _VOL_LARGE_THR).any())
            agreement = int((sub["AgrLevel"] >= 2).any())
        else:
            subsolid = large = agreement = 0

        for k, v in (("actionable", actionable), ("urgent", urgent), ("subsolid", subsolid),
                     ("large", large), ("present", present), ("agreement", agreement)):
            cnt[k] += v

        prim_qa[scan_id] = {
            "split": split,
            "qa_results": {"default_qa": [{_QA_ACTIONABLE: actionable}]},
            "scan_id": scan_id, "lndb_id": lndb_id, "fleischner": fcat,
            "nii_path": nii_path,
        }
        urgent_qa[scan_id] = {"split": split, "qa_results": {"default_qa": [{_QA_URGENT: urgent}]},
                              "scan_id": scan_id}
        category_qa[scan_id] = {"split": split, "qa_results": {"default_qa": [{_QA_CATEGORY: fcat}]},
                                "scan_id": scan_id, "fleischner": fcat}
        subsolid_qa[scan_id] = {"split": split, "qa_results": {"default_qa": [{_QA_SUBSOLID: subsolid}]},
                                "scan_id": scan_id}
        large_qa[scan_id] = {"split": split, "qa_results": {"default_qa": [{_QA_LARGE: large}]},
                             "scan_id": scan_id}
        present_qa[scan_id] = {"split": split, "qa_results": {"default_qa": [{_QA_PRESENT: present}]},
                               "scan_id": scan_id}
        agreement_qa[scan_id] = {"split": split, "qa_results": {"default_qa": [{_QA_AGREEMENT: agreement}]},
                                 "scan_id": scan_id}
        by_split[split].append({
            "sample_name": scan_id, "nii_path": nii_path,
            "fleischner_actionable": actionable,
        })

    def _pct(c: int) -> str:
        return f"{c}/{matched} = {100 * c / max(1, matched):.1f}%"

    print(f"  matched {matched} scans ({skipped_no_nifti} Fleischner rows without a staged NIfTI)",
          file=sys.stderr)
    print(f"  PRIMARY  fleischner_actionable {_pct(cnt['actionable'])} | "
          f"fleischner_urgent {_pct(cnt['urgent'])} | category hist {dict(sorted(cat_hist.items()))}",
          file=sys.stderr)
    print(f"  NODULE   subsolid {_pct(cnt['subsolid'])} | large {_pct(cnt['large'])} | "
          f"present {_pct(cnt['present'])} | agreement {_pct(cnt['agreement'])}", file=sys.stderr)
    print("  splits: " + ", ".join(f"{k}={len(v)}" for k, v in by_split.items()), file=sys.stderr)

    if prior_total and matched < prior_total:
        raise SystemExit(f"[lndb] COVERAGE REGRESSION: matched {matched} < committed {prior_total} "
                         f"({prior}). Refusing to overwrite -- check the staged nifti/ tree.")
    for s in ("train", "dev", "test"):
        if prior[s] and not by_split[s]:
            raise SystemExit(f"[lndb] REFUSING to write empty '{s}' split over committed {prior[s]} rows.")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    engine.write_json(prim_qa, out_dir / "lndb_labels.json")
    engine.write_json(urgent_qa, out_dir / "lndb_labels__fleischner_urgent.json")
    engine.write_json(category_qa, out_dir / "lndb_labels__fleischner_category.json")
    engine.write_json(subsolid_qa, out_dir / "lndb_labels__subsolid.json")
    engine.write_json(large_qa, out_dir / "lndb_labels__large_nodule.json")
    engine.write_json(present_qa, out_dir / "lndb_labels__nodule_present.json")
    engine.write_json(agreement_qa, out_dir / "lndb_labels__nodule_agreement.json")
    print("  wrote lndb_labels.json (fleischner_actionable) + 6 sidecars "
          "(urgent, category[4-class], subsolid, large_nodule, nodule_present, nodule_agreement)",
          file=sys.stderr)
    for split, rows in by_split.items():
        engine.write_jsonl(rows, jsonl_dir / f"{split}.jsonl")
        print(f"  wrote {jsonl_dir / f'{split}.jsonl'} ({len(rows)} rows)", file=sys.stderr)


SPEC = DatasetSpec(
    name="lndb",
    access=Access.ZENODO,
    modality="chest_ct",
    role="candidate",
    source=_ZENODO_RECORD_ID,
    token_env=None,
    committed_outputs=(
        "data/evaluation/lndb_labels.json",
        "data/evaluation/lndb_labels__fleischner_urgent.json",
        "data/evaluation/lndb_labels__fleischner_category.json",
        "data/evaluation/lndb_labels__subsolid.json",
        "data/evaluation/lndb_labels__large_nodule.json",
        "data/evaluation/lndb_labels__nodule_present.json",
        "data/evaluation/lndb_labels__nodule_agreement.json",
        "data/evaluation/lndb_jsonl/train.jsonl",
        "data/evaluation/lndb_jsonl/dev.jsonl",
        "data/evaluation/lndb_jsonl/test.jsonl",
    ),
    notes="Zenodo 6613714 (CC BY-NC-ND 4.0, no token, eval-only); 236 training scans "
          "(58 test withheld); MetaImage .mhd/.raw in data*.rar -> SimpleITK -> nifti/; "
          "author-GT Fleischner panel (actionable/urgent/category) + nodule "
          "characteristics (subsolid/large/present/agreement) from trainNodules_gt.csv.",
    stage=stage,
    build_labels=build_labels,
)
