"""LIDC-IDRI (NCI Imaging Data Commons) -- CT staging + malignancy labels."""
from __future__ import annotations

import csv
import hashlib
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

_LABELS_JSON = "data/evaluation/lidc_malignancy_labels.json"
_JSONL_DIR = "data/evaluation/lidc_chest_ct_jsonl"

_QA_QUESTION = "Does this scan contain a malignant nodule (median radiologist rating > 3)?"

_COLLECTION_ID = "lidc_idri"
_N_SERIES = 1018   # 1,010 patients; 8 multi-series
_XML_URL = ("https://wiki.cancerimagingarchive.net/download/attachments/1966254/"
            "LIDC-XML-only.zip?version=1&modificationDate=1530215018015&api=v2")
_XML_SHA256 = "644557a3aa305602609c718b0cae33a93be762e8901a80bcd9df735b0ec6ab90"
_MANIFEST_COLS = ["PatientID", "SeriesInstanceUID", "dicom_dir",
                  "nifti_path", "instanceCount", "series_size_MB"]

# non-uniform slice increment; excluded as imaging-QC
_EXCLUDE_UNCONVERTIBLE = frozenset({
    "LIDC-IDRI-0085", "LIDC-IDRI-0123", "LIDC-IDRI-0146", "LIDC-IDRI-0267", "LIDC-IDRI-0672",
})


def _committed_patient_ids() -> list[str]:
    """The label roster (union of ``sample_name`` over the 3 label JSONLs)."""
    ids: set[str] = set()
    for split in ("train", "dev", "test"):
        for row in engine.read_jsonl(Path(_JSONL_DIR) / f"{split}.jsonl"):
            ids.add(row["sample_name"])
    return sorted(ids)


def _select_series(patients: list[str] | None) -> list[dict]:
    """Per-patient CT series to stage (the series build_labels labels)."""
    import numpy as np

    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    _setup_pylidc_config(engine.expandvars("$DATA_ROOT/radiology/lidc"))
    import pylidc as pl

    query = pl.query(pl.Scan)
    if patients:
        query = query.filter(pl.Scan.patient_id.in_(list(patients)))
    chosen: dict[str, str] = {}
    for scan in query.all():  # last write wins
        chosen[scan.patient_id] = scan.series_instance_uid
    return [{"PatientID": p, "SeriesInstanceUID": u} for p, u in sorted(chosen.items())]


def _flatten_dicom(src: Path, dicom_dir: Path) -> None:
    """Move a patient's downloaded ``*.dcm`` (flat under ``src``) into ``dicom_dir``."""
    import shutil

    dicom_dir.mkdir(parents=True, exist_ok=True)
    dcms = list(src.rglob("*.dcm")) if src.exists() else []
    if not dcms:
        raise RuntimeError(f"[lidc] no DICOM downloaded under {src}")
    for f in dcms:
        target = dicom_dir / f.name
        if not target.exists():
            shutil.move(str(f), str(target))


def _convert_to_nifti(dicom_dir: Path, out_nii: Path) -> None:
    """DICOM dir -> single compressed NIfTI (largest slab), orientation preserved."""
    import shutil
    import tempfile

    import dicom2nifti
    import dicom2nifti.settings as _d2n
    from dicom2nifti.exceptions import ConversionValidationError
    import nibabel as nib

    out_nii.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(out_nii.parent)) as td:
        try:
            _d2n.enable_validate_slice_increment()
            dicom2nifti.convert_directory(str(dicom_dir), td, compression=True, reorient=False)
        except ConversionValidationError:
            print(f"[lidc] {dicom_dir.parent.name}: inconsistent slice increment "
                  f"(known LIDC quirk) -> converting with the check relaxed", flush=True)
            _d2n.disable_validate_slice_increment()
            try:
                dicom2nifti.convert_directory(str(dicom_dir), td, compression=True, reorient=False)
            finally:
                _d2n.enable_validate_slice_increment()
        niis = sorted(Path(td).glob("*.nii.gz"), key=lambda p: p.stat().st_size, reverse=True)
        if not niis:
            raise RuntimeError(f"[lidc] dicom2nifti produced no NIfTI for {dicom_dir}")
        shutil.move(str(niis[0]), str(out_nii))
    img = nib.load(str(out_nii))
    if img.ndim != 3 or min(img.shape) < 2:
        raise RuntimeError(f"[lidc] {out_nii}: unexpected shape {img.shape}")


def _ensure_xml(xml_dir: Path) -> None:
    """Download + SHA-256-verify + unpack ``LIDC-XML-only.zip`` (pylidc annotations); idempotent."""
    import hashlib
    import subprocess
    import zipfile

    xml_dir = Path(xml_dir)
    if xml_dir.exists() and any(xml_dir.rglob("*.xml")):
        print(f"[lidc] XML already present at {xml_dir}", flush=True)
        return
    xml_dir.mkdir(parents=True, exist_ok=True)
    zpath = xml_dir / "LIDC-XML-only.zip"
    print(f"[lidc] downloading LIDC-XML-only.zip -> {zpath}", flush=True)
    subprocess.run(["curl", "-fsSL", "-o", str(zpath), _XML_URL], check=True)
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if digest != _XML_SHA256:
        raise RuntimeError(f"[lidc] XML sha256 mismatch: {digest} != {_XML_SHA256}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(xml_dir)
    n = len(list(xml_dir.rglob("*.xml")))
    print(f"[lidc] unpacked {n} annotation XMLs", flush=True)


def _staged_ok(dest: Path) -> set[str]:
    """PatientIDs whose ``scan.nii.gz`` is already on disk (for idempotency)."""
    return {p.parent.name for p in Path(dest).glob("*/scan.nii.gz")}


def _write_manifest(manifest_csv: Path, dest: Path, sel: list[dict]) -> None:
    """One row per successfully-converted patient (``nifti_path`` ``${DATA_ROOT}``-relative)."""
    rows = []
    for r in sel:
        pid = r["PatientID"]
        nii = Path(dest) / pid / "scan.nii.gz"
        if not nii.exists():
            continue
        dicom_dir = Path(dest) / pid / "dicom"
        dcms = list(dicom_dir.glob("*.dcm")) if dicom_dir.exists() else []
        size_mb = round(sum(f.stat().st_size for f in dcms) / 1e6, 2) if dcms else ""
        rows.append({
            "PatientID": pid,
            "SeriesInstanceUID": r["SeriesInstanceUID"],
            "dicom_dir": str(dicom_dir),
            "nifti_path": engine.data_root_relative(nii),
            "instanceCount": len(dcms) if dcms else "",
            "series_size_MB": size_mb,
        })
    with Path(manifest_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)


def stage(dest: Path = None, *, scope: str = "all", patients: list[str] | None = None,
          limit: int | None = None, skip_download: bool = False,
          fetch_xml: bool = True) -> None:
    """Stage LIDC-IDRI: one CT series/patient from IDC -> <PID>/{dicom/,scan.nii.gz}."""
    if dest is None:
        dest = engine.expandvars("$DATA_ROOT/radiology/lidc")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if patients:
        roster: list[str] | None = list(patients)
    elif scope == "committed":
        roster = _committed_patient_ids()
    elif scope == "all":
        roster = None  # all patients
    else:
        raise ValueError(f"[lidc] unknown scope {scope!r} (all|committed)")

    sel = _select_series(roster)
    if limit is not None:
        sel = sel[:limit]
    if not sel:
        raise RuntimeError("[lidc] series selection returned 0 rows")

    have = _staged_ok(dest)
    todo = [r for r in sel if r["PatientID"] not in have]
    print(f"[lidc] {len(sel)} patients selected; {len(have)} already staged; "
          f"{len(todo)} to fetch", flush=True)

    if skip_download:
        print("[lidc] skip_download set -- not downloading", flush=True)
    elif todo:
        import shutil as _shutil
        raw = dest / "_idc_raw"
        # completeness gate against s5cmd silent partial transfers
        exp_ic: dict[str, int] = {}
        try:
            for _row in engine.idc_query(
                    "SELECT SeriesInstanceUID, instanceCount FROM idc_index "
                    f"WHERE collection_id='{_COLLECTION_ID}'"):
                exp_ic[_row["SeriesInstanceUID"]] = int(float(_row["instanceCount"]))
        except Exception as _e:  # pragma: no cover
            print(f"[lidc] WARN: IDC instanceCount query failed; completeness gate disabled ({_e})", flush=True)
        engine.idc_download([r["SeriesInstanceUID"] for r in todo], raw, dir_template="%PatientID")
        failed = []
        for r in todo:
            pid = r["PatientID"]
            try:
                _flatten_dicom(raw / pid, dest / pid / "dicom")
                exp = exp_ic.get(r["SeriesInstanceUID"])
                got = sum(1 for _ in (dest / pid / "dicom").glob("*.dcm"))
                if exp and got < exp:
                    _shutil.rmtree(dest / pid / "dicom", ignore_errors=True)
                    _shutil.rmtree(raw / pid, ignore_errors=True)
                    raise RuntimeError(f"truncated s5cmd transfer: {got}/{exp} DICOM files "
                                       f"(partials cleared; re-run to re-fetch)")
                _convert_to_nifti(dest / pid / "dicom", dest / pid / "scan.nii.gz")
            except Exception as e:
                print(f"[lidc] SKIP {pid}: {type(e).__name__}: {e}", flush=True)
                failed.append(pid)
        if failed:
            print(f"[lidc] {len(failed)} patients failed conversion: {failed[:10]}", flush=True)

    if fetch_xml:
        _ensure_xml(dest / "xml")

    _write_manifest(dest / "manifest.csv", dest, sel)

    staged = _staged_ok(dest)
    missing = [r["PatientID"] for r in sel if r["PatientID"] not in staged]
    if missing:
        print(f"[lidc] WARNING: {len(missing)} selected patients not staged (conversion "
              f"failures) -- the byte-identity gate is the backstop for the committed "
              f"roster; e.g. {missing[:5]}", flush=True)
    print(f"[lidc] staged {len(staged)} patients at {dest}", flush=True)


@dataclass
class _ScanLabel:
    patient_id: str
    label: int
    n_clusters: int
    cluster_medians: list[float]


def _split_for_patient(patient_id: str) -> str:
    """Deterministic 70/15/15 by the FULL sha1 digest of patient_id (NOT engine.stable_split's sha1[:8])."""
    h = int(hashlib.sha1(patient_id.encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    elif h < 85:
        return "dev"
    return "test"


def _aggregate_scan(scan, *, drop_ambiguous: bool = True) -> _ScanLabel | None:
    """Cluster annotations per scan and return a binary malignancy label."""
    clusters = scan.cluster_annotations(verbose=False)
    if not clusters:
        return None

    medians: list[float] = []
    for cluster in clusters:
        mals = [a.malignancy for a in cluster]
        medians.append(float(statistics.median(mals)))

    if drop_ambiguous:
        non_ambig = [m for m in medians if m != 3.0]
        if not non_ambig:
            return None
        label = int(any(m > 3 for m in non_ambig))
    else:
        label = int(any(m > 3 for m in medians))

    return _ScanLabel(
        patient_id=scan.patient_id,
        label=label,
        n_clusters=len(clusters),
        cluster_medians=medians,
    )


# scan-level binary morphology tasks: positive iff any cluster median meets the cut
_TASK_CHARS = ("spiculation", "lobulation", "texture", "calcification",
               "subtlety", "margin", "sphericity")
_TASKS = [
    # (name, characteristic, op, threshold, QA question)
    ("spiculated",        "spiculation",   ">=", 2, "Does this scan contain a spiculated nodule (median radiologist rating >= 2)?"),
    ("spiculated_marked", "spiculation",   ">=", 4, "Does this scan contain a markedly spiculated nodule (median rating >= 4)?"),
    ("lobulated",         "lobulation",    ">=", 2, "Does this scan contain a lobulated nodule (median rating >= 2)?"),
    ("lobulated_marked",  "lobulation",    ">=", 4, "Does this scan contain a markedly lobulated nodule (median rating >= 4)?"),
    ("subsolid",          "texture",       "<=", 3, "Does this scan contain a subsolid (non-solid / part-solid) nodule (median texture rating <= 3)?"),
    ("calcified",         "calcification", "<=", 5, "Does this scan contain a calcified nodule (median calcification rating <= 5, i.e. not absent)?"),
    ("subtle",            "subtlety",      "<=", 3, "Does this scan contain a subtle nodule (median subtlety rating <= 3)?"),
    ("poorly_marginated", "margin",        "<=", 3, "Does this scan contain a poorly-marginated nodule (median margin rating <= 3)?"),
    ("non_spherical",     "sphericity",    "<=", 3, "Does this scan contain a non-spherical nodule (median sphericity rating <= 3)?"),
]


def _task_positive(op: str, thresh: int, medians: list[float]) -> int:
    """Scan-positive iff any cluster median meets the (op, threshold) cut."""
    if op == ">=":
        return int(any(m >= thresh for m in medians))
    return int(any(m <= thresh for m in medians))


def _aggregate_scan_all(scan, *, drop_ambiguous: bool = True):
    """Cluster once; return (malignancy _ScanLabel or None, {char: [per-cluster medians]})."""
    clusters = scan.cluster_annotations(verbose=False)
    if not clusters:
        return None, {}
    mal_medians = [float(statistics.median([a.malignancy for a in cl])) for cl in clusters]
    if drop_ambiguous:
        non_ambig = [m for m in mal_medians if m != 3.0]
        mal = (None if not non_ambig
               else _ScanLabel(scan.patient_id, int(any(m > 3 for m in non_ambig)),
                               len(clusters), mal_medians))
    else:
        mal = _ScanLabel(scan.patient_id, int(any(m > 3 for m in mal_medians)),
                         len(clusters), mal_medians)
    char_medians = {c: [float(statistics.median([getattr(a, c) for a in cl])) for cl in clusters]
                    for c in _TASK_CHARS}
    return mal, char_medians


def _staged_patients(manifest_csv: Path) -> set[str]:
    """PatientIDs whose NIfTI is on disk per the LIDC manifest (empty -> no filter)."""
    if not manifest_csv.exists():
        return set()
    with manifest_csv.open() as f:
        return {row["PatientID"] for row in csv.DictReader(f)}


def _read_manifest_paths(manifest_csv: Path) -> dict[str, str]:
    """Return ``{PatientID: nifti_path}`` from the LIDC manifest CSV."""
    if not manifest_csv.exists():
        return {}
    with manifest_csv.open() as f:
        return {row["PatientID"]: row["nifti_path"] for row in csv.DictReader(f)}


def _setup_pylidc_config(dicom_root: Path) -> None:
    """Write a minimal ``.pylidcrc`` to a temp HOME so we don't pollute the user's home."""
    import os

    pyhome = Path("/tmp") / f"pylidc_home_{os.getuid()}"
    pyhome.mkdir(parents=True, exist_ok=True)
    rc = pyhome / ".pylidcrc"
    rc.write_text(f"[dicom]\npath = {dicom_root}\n")
    os.environ["HOME"] = str(pyhome)


def _write_split_jsonls(entries: dict[str, dict], out_dir: Path) -> None:
    """Emit train/dev/test JSONLs (``sample_name`` + ``${DATA_ROOT}``-relative ``nii_path``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for pid, entry in entries.items():
        nii_path = f"${{DATA_ROOT}}/radiology/lidc/{pid}/scan.nii.gz"
        by_split[entry["split"]].append({"sample_name": pid, "nii_path": nii_path})

    for split, rows in by_split.items():
        engine.write_jsonl(rows, out_dir / f"{split}.jsonl")


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 dicom_root: Path | None = None,
                 limit: int | None = None,
                 keep_ambiguous: bool = False,
                 patients: list[str] | None = None) -> None:
    """(Re)derive the full LIDC label panel from pylidc's bundled annotation DB (no staged data needed)."""
    import numpy as np

    # pylidc 0.2.3 references np.int (removed in numpy 2.x); restore for clustering.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    warnings.filterwarnings("ignore", category=UserWarning)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_root = Path(dicom_root) if dicom_root else engine.expandvars("$DATA_ROOT/radiology/lidc")
    _setup_pylidc_config(dicom_root)
    import pylidc as pl  # after the np.int patch + pylidcrc setup

    query = pl.query(pl.Scan)
    if patients:
        query = query.filter(pl.Scan.patient_id.in_(list(patients)))
    scans_iter = query.all() if limit is None else query.limit(limit).all()

    entries: dict[str, dict] = {}
    task_labels: dict[str, dict[str, int]] = {name: {} for name, *_ in _TASKS}
    for scan in scans_iter:
        try:
            mal, char_medians = _aggregate_scan_all(scan, drop_ambiguous=not keep_ambiguous)
        except Exception:
            continue
        if mal is None:
            continue
        pid = mal.patient_id
        if pid in _EXCLUDE_UNCONVERTIBLE:
            continue
        entries[pid] = {
            "split": _split_for_patient(pid),
            "qa_results": {"default_qa": [{_QA_QUESTION: mal.label}]},
            "n_clusters": mal.n_clusters,
            "cluster_medians": mal.cluster_medians,
        }
        for name, attr, op, thresh, _qa in _TASKS:
            task_labels[name][pid] = _task_positive(op, thresh, char_medians[attr])

    engine.write_json(entries, out_dir / "lidc_malignancy_labels.json")
    for name, attr, op, thresh, qa in _TASKS:
        d = {pid: {"split": entries[pid]["split"],
                   "qa_results": {"default_qa": [{qa: task_labels[name][pid]}]},
                   "patient_id": pid}
             for pid in entries}
        engine.write_json(d, out_dir / f"lidc_labels__{name}.json")

    _write_split_jsonls(entries, out_dir / "lidc_chest_ct_jsonl")
    n_mal = sum(e["qa_results"]["default_qa"][0][_QA_QUESTION] for e in entries.values())
    print(f"[lidc] {len(entries)} evaluable scans -> {out_dir}: malignancy {n_mal} pos; "
          + ", ".join(f"{n}={sum(task_labels[n].values())}" for n, *_ in _TASKS), flush=True)


SPEC = DatasetSpec(
    name="lidc",
    access=Access.IDC,
    modality="chest_ct",
    role="scored",
    source="lidc_idri",
    token_env=None,
    committed_outputs=(
        _LABELS_JSON,
        *(f"data/evaluation/lidc_labels__{name}.json" for name, *_ in _TASKS),
        f"{_JSONL_DIR}/train.jsonl",
        f"{_JSONL_DIR}/dev.jsonl",
        f"{_JSONL_DIR}/test.jsonl",
    ),
    notes="Production IDC stager: idc-index run ISOLATED (uv run --no-project --with "
          "idc-index) so its pandas<=2.2.4 never enters the venv; per-patient series = the "
          "one build_labels labels (pylidc-chosen, so NIfTI<->label aligned incl. the 8 "
          "multi-series patients), DICOM->NIfTI, SHA-256 LIDC-XML fetch. Labels via pylidc "
          "(bundled DB, patches numpy.int): label=1 iff any "
          "non-ambiguous nodule-cluster median malignancy > 3, no-cluster + all-ambiguous "
          "scans dropped. Split uses the FULL sha1 digest (NOT engine.stable_split's sha1[:8]).",
    stage=stage,
    build_labels=build_labels,
)
