"""STOIC2021 (AWS Open Data s3://stoic2021-training) stager + label builder."""
from __future__ import annotations

import csv
import logging
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

logger = logging.getLogger(__name__)

_S3_BASE = "https://stoic2021-training.s3.amazonaws.com"
_S3_BUCKET = "s3://stoic2021-training"
_N_EXPECTED = 2000

# qa_key matched by exact-string dict lookup -- must byte-match benchmark cells
_QA_ANCHOR = "Does this CT show COVID-19? (STOIC2021 probCOVID target)"
_QA_COVID = "Is COVID-19 present?"
_QA_SEVERE = "Is the COVID-19 case severe?"
_QA_SEVERE_AMONG_COVID = "Is the COVID-19 case severe? (among RT-PCR+)"


def _read_reference_csv(path: str | Path) -> list[tuple[str, int, int]]:
    """Parse reference.csv -> [(PatientID, probCOVID, probSevere)] rows."""
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows or rows[0][:3] != ["PatientID", "probCOVID", "probSevere"]:
        raise SystemExit(f"[stoic2021] reference.csv header {rows[0] if rows else None!r} != "
                         f"['PatientID','probCOVID','probSevere']")
    out = []
    for r in rows[1:]:
        if r and r[0]:
            out.append((str(r[0]), int(float(r[1])), int(float(r[2]))))
    return out


def _head_size(url: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def _valid_mha(path: Path) -> bool:
    """Cheap validity: non-empty + SimpleITK reads a 3-D volume with finite HU."""
    try:
        if not (path.exists() and path.stat().st_size > 0):
            return False
        import SimpleITK as sitk  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        img = sitk.ReadImage(str(path))
        if img.GetDimension() < 3 or min(img.GetSize()) < 8:
            return False
        mid = sitk.GetArrayFromImage(img)[img.GetSize()[2] // 2]
        return bool(np.isfinite(mid).any())
    except Exception:  # noqa: BLE001
        return False


def stage(dest: Path, *, skip_download: bool = False, workers: int = 40,
          verify_mha: bool = False) -> None:
    """Download the 2,000 public STOIC2021 .mha + reference.csv (idempotent resume)."""
    dest = Path(dest)
    mha_dir = dest / "data" / "mha"
    meta_dir = dest / "metadata"
    mha_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    ref_path = meta_dir / "reference.csv"
    ref_path.write_bytes(urllib.request.urlopen(_S3_BASE + "/metadata/reference.csv", timeout=60).read())
    ids = [pid for pid, _c, _s in _read_reference_csv(ref_path)]
    if len(ids) != _N_EXPECTED:
        raise SystemExit(f"[stoic2021] reference.csv has {len(ids)} ids, expected {_N_EXPECTED}")
    logger.info("[stoic2021] reference.csv: %d ids", len(ids))

    def _dl(pid: str):
        url = f"{_S3_BASE}/data/mha/{pid}.mha"
        tgt = mha_dir / f"{pid}.mha"
        last = None
        for attempt in range(4):
            try:
                exp = _head_size(url)
                if tgt.exists() and exp > 0 and tgt.stat().st_size == exp:
                    return "skip", exp
                tmp = mha_dir / f"{pid}.mha.part"
                with urllib.request.urlopen(url, timeout=900) as r, open(tmp, "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                if exp and tmp.stat().st_size != exp:
                    sz = tmp.stat().st_size
                    tmp.unlink()
                    raise IOError(f"size {sz} != {exp}")
                tmp.rename(tgt)
                return "ok", tgt.stat().st_size
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        return "err", f"{pid}: {str(last)[:120]}"

    ok = skip = err = 0
    errs: list[str] = []
    if not skip_download:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_dl, pid): pid for pid in ids}
            for i, fut in enumerate(as_completed(futs), 1):
                st, info = fut.result()
                if st == "ok":
                    ok += 1
                elif st == "skip":
                    skip += 1
                else:
                    err += 1
                    errs.append(info)
                if i % 200 == 0:
                    logger.info("[stoic2021] download %d/%d ok=%d skip=%d err=%d",
                                i, len(ids), ok, skip, err)
        total = ok + skip + err
        if errs:
            print(f"[stoic2021] {len(errs)} download errors: {errs[:20]}", file=sys.stderr)
        if total and err > max(5, total // 20):
            raise RuntimeError(f"[stoic2021] catastrophic download failure: {err}/{total} (>5%)")

    n_disk = len(list(mha_dir.glob("*.mha")))
    logger.info("[stoic2021] staged .mha on disk: %d (ok=%d skip=%d err=%d)", n_disk, ok, skip, err)
    if n_disk != _N_EXPECTED:
        raise RuntimeError(f"[stoic2021] |staged|={n_disk} != {_N_EXPECTED} -- incomplete download")

    if verify_mha:
        bad = [p.name for p in sorted(mha_dir.glob("*.mha")) if not _valid_mha(p)]
        if bad:
            raise RuntimeError(f"[stoic2021] {len(bad)} unreadable .mha: {bad[:20]}")
        logger.info("[stoic2021] all %d .mha pass SimpleITK 3-D + finite-HU verify", n_disk)


def _committed_split_map(labels_json: Path) -> dict[str, str] | None:
    """Preserve the committed per-PatientID split (seed unrecoverable; normalizes 'valid'->'dev')."""
    p = Path(labels_json)
    if not p.exists():
        return None
    d = engine.read_json(p)
    return {pid: ("dev" if v.get("split") == "valid" else v["split"]) for pid, v in d.items()}


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 stoic2021_root: Path = Path("$DATA_ROOT/radiology/stoic2021")) -> None:
    """Rederive STOIC2021 labels from metadata/reference.csv (committed split preserved)."""
    stoic2021_root = engine.expandvars(str(stoic2021_root))
    mha_dir = Path(stoic2021_root) / "data" / "mha"
    ref_path = Path(stoic2021_root) / "metadata" / "reference.csv"
    out_dir = Path(out_dir)
    jsonl_out = out_dir / "stoic2021_jsonl"

    recs_csv = _read_reference_csv(ref_path)
    mha = {p.stem: p for p in mha_dir.glob("*.mha")}
    logger.info("[stoic2021] reference rows=%d ; .mha on disk=%d", len(recs_csv), len(mha))

    split_map = (_committed_split_map(out_dir / "stoic2021_labels__covid.json")
                 or _committed_split_map(out_dir / "stoic2021_labels.json"))
    if split_map is None:
        logger.warning("[stoic2021] no committed split found -- deriving via stable_split")
        split_map = {}

    records = []
    missing = 0
    for pid, covid, severe in recs_csv:
        if pid not in mha:
            missing += 1
            continue
        records.append({"pid": pid, "covid": covid, "severe": severe,
                        "split": split_map.get(pid) or engine.stable_split(pid),
                        "nii_path": engine.data_root_relative(mha[pid])})
    matched = len(records)
    logger.info("[stoic2021] matched %d (missing .mha: %d)", matched, missing)
    if matched != _N_EXPECTED:
        raise SystemExit(f"[stoic2021] matched {matched} != {_N_EXPECTED} -- did stage() run fully?")

    # probSevere is strictly conditional on COVID+ (Revel 2021)
    n_sev = sum(r["severe"] for r in records)
    n_sev_among = sum(r["severe"] for r in records if r["covid"])
    n_sev_noncovid = sum(r["severe"] for r in records if not r["covid"])
    if not (n_sev == n_sev_among and n_sev_noncovid == 0):
        raise SystemExit(f"[stoic2021] severity-conditionality violated: severe={n_sev} "
                         f"among_covid={n_sev_among} non_covid_severe={n_sev_noncovid}")
    n_covid = sum(r["covid"] for r in records)
    cov_prev = n_covid / matched
    if not (0.55 <= cov_prev <= 0.65):
        raise SystemExit(f"[stoic2021] COVID prevalence {cov_prev:.3f} out of expected range")

    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for r in records:
        by_split[r["split"]].append(r)
    for s in ("train", "dev", "test"):
        if not by_split[s]:
            raise SystemExit(f"[stoic2021] empty '{s}' split -- refusing to write.")

    jsonl_out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "dev", "test"):
        rows = [{"sample_name": r["pid"], "probCOVID": r["covid"], "probSevere": r["severe"],
                 "nii_path": r["nii_path"]} for r in by_split[s]]
        engine.write_jsonl(rows, jsonl_out / f"{s}.jsonl")
        pos = sum(x["probCOVID"] for x in rows)
        logger.info("[stoic2021]   %s: %d rows COVID+=%d (%.1f%%)", s, len(rows), pos,
                    100 * pos / max(len(rows), 1))

    out_dir.mkdir(parents=True, exist_ok=True)

    # anchor schema: dev->valid, includes target, no patient_id
    anchor = {r["pid"]: {
        "split": "valid" if r["split"] == "dev" else r["split"],
        "qa_results": {"default_qa": [{_QA_ANCHOR: r["covid"]}]},
        "probCOVID": r["covid"], "probSevere": r["severe"], "target": r["covid"],
    } for r in records}
    engine.write_json(anchor, out_dir / "stoic2021_labels.json")

    # sidecar schema: raw split + patient_id, no target
    def _sidecar(path: Path, qa: str, field: str, *, covid_only: bool = False) -> None:
        d = {}
        for r in records:
            if covid_only and not r["covid"]:
                continue
            d[r["pid"]] = {"split": r["split"],
                           "qa_results": {"default_qa": [{qa: r[field]}]},
                           "patient_id": r["pid"]}
        engine.write_json(d, path)
        pos = sum(next(iter(v["qa_results"]["default_qa"][0].values())) for v in d.values())
        logger.info("[stoic2021] wrote %s: %d entries, %d pos (%.1f%%)", Path(path).name,
                    len(d), pos, 100 * pos / max(len(d), 1))

    _sidecar(out_dir / "stoic2021_labels__covid.json", _QA_COVID, "covid")
    _sidecar(out_dir / "stoic2021_labels__severe.json", _QA_SEVERE, "severe")
    _sidecar(out_dir / "stoic2021_labels__severe_among_covid.json", _QA_SEVERE_AMONG_COVID,
             "severe", covid_only=True)
    logger.info("[stoic2021] build_labels done: %d scans | COVID %.1f%% | severe %.1f%% | "
                "severe|COVID+ %.1f%%", matched, 100 * cov_prev, 100 * n_sev / matched,
                100 * n_sev_among / max(n_covid, 1))


SPEC = DatasetSpec(
    name="stoic2021",
    access=Access.S3,
    modality="chest_ct",
    role="scored",
    source=_S3_BUCKET,
    token_env=None,
    committed_outputs=(
        "data/evaluation/stoic2021_labels.json",
        "data/evaluation/stoic2021_labels__covid.json",
        "data/evaluation/stoic2021_labels__severe.json",
        "data/evaluation/stoic2021_labels__severe_among_covid.json",
        "data/evaluation/stoic2021_jsonl/train.jsonl",
        "data/evaluation/stoic2021_jsonl/dev.jsonl",
        "data/evaluation/stoic2021_jsonl/test.jsonl",
    ),
    notes="Open-data S3 pull (urllib, --no-account) of 2,000 public .mha scans; hidden "
          "grand-challenge test set not released; probCOVID/probSevere targets rederived "
          "from metadata/reference.csv; committed split preserved.",
    stage=stage,
    build_labels=build_labels,
)
