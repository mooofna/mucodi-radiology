"""COCA gated subset -- ECG-gated cardiac calcium-scoring CT, Agatston from segmentation."""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataprep.datasets import coca, engine
from dataprep.datasets.spec import Access, DatasetSpec

logger = logging.getLogger(__name__)

_QUALIFIED_REF = coca._QUALIFIED_REF
_TABLE = "gated"
_NIFTI_SUBDIR = "nifti_gated"

# mirror the non-gated coca anchor QA string
_QA_KEY = engine.qa_key("coronary artery calcification")

# Vessel Name -> 4-territory code (off-vocabulary coerces to LCA)
_VESSEL = {
    "Left Anterior Descending Artery": "LAD",
    "Left Circumflex Artery": "LCX",
    "Right Coronary Artery": "RCA",
    "Left Coronary Artery": "LCA",
}
_VESSELS = ("LAD", "LCX", "RCA", "LCA")

# per-vessel + multivessel QA strings; byte-matched by BINARY_COHORTS
_QA_LAD = "Is calcification present in the LAD (left anterior descending) artery? (0=No, 1=Yes)"
_QA_LCX = "Is calcification present in the LCX (left circumflex) artery? (0=No, 1=Yes)"
_QA_RCA = "Is calcification present in the RCA (right coronary) artery? (0=No, 1=Yes)"
_QA_LCA = "Is calcification present in the LCA (left main / left coronary) artery? (0=No, 1=Yes)"
_QA_MULTIVESSEL = "Is multivessel coronary calcification present (>=2 of LAD/LCX/RCA/LCA)? (0=No, 1=Yes)"


def _weight(peak_hu: float) -> int:
    if peak_hu >= 400:
        return 4
    if peak_hu >= 300:
        return 3
    if peak_hu >= 200:
        return 2
    if peak_hu >= 130:
        return 1
    return 0


def _parse_point_px(roi: dict) -> list[tuple[float, float]]:
    """Parse OsiriX ``Point_px`` = list of ``"(col, row)"`` = (x, y) pixel coords."""
    pts: list[tuple[float, float]] = []
    n_bad = 0
    for s in roi.get("Point_px", []):
        m = re.match(r"\(\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\)", str(s))
        if m:
            try:
                pts.append((float(m.group(1)), float(m.group(2))))
                continue
            except ValueError:
                pass
        n_bad += 1
    if n_bad:
        logger.warning("[coca_gated] _parse_point_px: %d Point_px token(s) failed to parse", n_bad)
    return pts


def _poly_mask(pts, ny: int, nx: int):
    """Rasterize an OsiriX ROI polygon to a boolean pixel mask (skimage.draw.polygon)."""
    import numpy as np  # noqa: PLC0415
    from skimage.draw import polygon as _sk_polygon  # noqa: PLC0415

    mask = np.zeros((ny, nx), bool)
    if len(pts) < 3:
        for (x, y) in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < ny and 0 <= xi < nx:
                mask[yi, xi] = True
        return mask
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    rr, cc = _sk_polygon(ys, xs, shape=(ny, nx))  # row=y, col=x
    mask[rr, cc] = True
    return mask


def _cc_components(binary):
    """4-connectivity connected components (bounding-box-restricted, pure numpy, no scipy)."""
    import numpy as np  # noqa: PLC0415

    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return []
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    sub = binary[y0:y1 + 1, x0:x1 + 1]
    sy, sx = sub.shape
    labels = np.zeros((sy, sx), np.int32)
    parent = {0: 0}

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for i in range(sy):
        for j in range(sx):
            if not sub[i, j]:
                continue
            up = labels[i - 1, j] if i > 0 else 0
            lf = labels[i, j - 1] if j > 0 else 0
            if up and lf:
                labels[i, j] = min(up, lf)
                union(up, lf)
            elif up:
                labels[i, j] = up
            elif lf:
                labels[i, j] = lf
            else:
                labels[i, j] = nxt
                parent[nxt] = nxt
                nxt += 1
    comps: dict[int, list[tuple[int, int]]] = {}
    for i in range(sy):
        for j in range(sx):
            lab = labels[i, j]
            if lab:
                comps.setdefault(find(lab), []).append((i + y0, j + x0))
    return list(comps.values())


def _read_gated_slices(dicom_dir: Path):
    """Read one gated patient's DICOM slices in ascending ImagePositionPatient z (ImageIndex order)."""
    import re as _re  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pydicom  # noqa: PLC0415
    from collections import Counter as _Counter  # noqa: PLC0415

    def _fn_ord(name):
        m = _re.search(r"IM-\d+-(\d+)\.dcm$", name)
        return int(m.group(1)) if m else None

    def _series(name):
        m = _re.match(r"IM-(\d+)-\d+\.dcm$", name)
        return m.group(1) if m else "?"

    rows = []
    broken = []  # (fn, name, series)
    for p in sorted(Path(dicom_dir).rglob("*.dcm")):
        name = p.name
        if name.startswith(".") or not name.startswith("IM-"):
            continue  # skip Azure temp cruft
        try:
            d = pydicom.dcmread(str(p))
        except Exception:
            broken.append((_fn_ord(name), name, _series(name)))
            continue
        if "PixelData" not in d or getattr(d, "ImagePositionPatient", None) is None:
            broken.append((_fn_ord(name), name, _series(name)))
            continue
        arr = d.pixel_array.astype(np.float32)
        slope = float(getattr(d, "RescaleSlope", 1) or 1)
        inter = float(getattr(d, "RescaleIntercept", 0) or 0)
        ps = [float(x) for x in getattr(d, "PixelSpacing", [1.0, 1.0])]
        rows.append(dict(z=float(d.ImagePositionPatient[2]), hu=arr * slope + inter,
                         px_area=ps[0] * ps[1],
                         thick=float(getattr(d, "SliceThickness", 0) or 0),
                         fn=_fn_ord(name), series=_series(name)))
    if not rows:
        if broken:
            raise ValueError("[coca_gated] %s: all %d slice(s) unreadable/header-only; refusing "
                             "to score (silent false-negative risk)" % (dicom_dir, len(broken)))
        return []
    counts = _Counter(r["series"] for r in rows)
    keep = counts.most_common(1)[0][0]
    if len(counts) > 1:
        logger.warning("[coca_gated] %s: %d DICOM series present %s; keeping dominant %s, "
                       "dropping rest", dicom_dir, len(counts), dict(counts), keep)
        rows = [r for r in rows if r["series"] == keep]
        broken = [b for b in broken if b[2] == keep]
    if broken:
        fit_rows = [r for r in rows if r["fn"] is not None]
        if len(fit_rows) >= 2 and all(fn is not None for fn, _nm, _s in broken):
            ny, nx = rows[0]["hu"].shape
            fns = np.array([r["fn"] for r in fit_rows], dtype=float)
            zsv = np.array([r["z"] for r in fit_rows], dtype=float)
            a, b = np.polyfit(fns, zsv, 1)
            for fn, _nm, _s in broken:
                rows.append(dict(z=float(a * fn + b),
                                 hu=np.full((ny, nx), -1024.0, dtype=np.float32),
                                 px_area=rows[0]["px_area"], thick=rows[0]["thick"],
                                 fn=fn, series=keep))
            logger.warning("[coca_gated] %s: re-placed %d header-only stub slice(s) by "
                           "filename-ordinal z-interpolation (ImageIndex order preserved)",
                           dicom_dir, len(broken))
        else:
            raise ValueError("[coca_gated] %s: %d unplaceable stub slice(s) - ImageIndex "
                             "mapping would be corrupted" % (dicom_dir, len(broken)))
    rows.sort(key=lambda r: (r["z"], -(r["fn"] if r["fn"] is not None else 0)))
    return rows


def _agatston_from_xml(xml_path: Path, dicom_dir_or_slices) -> dict:
    """Classic Agatston 1990 per-vessel score from an OsiriX calcium_xml + raw gated DICOM."""
    import plistlib  # noqa: PLC0415

    if isinstance(dicom_dir_or_slices, (str, Path)):
        slices = _read_gated_slices(Path(dicom_dir_or_slices))
    else:
        slices = list(dicom_dir_or_slices)
    per_vessel = {v: 0.0 for v in _VESSELS}
    if not slices:
        return {"total": 0.0, **per_vessel, "n_slices_used": 0}
    n_sl = len(slices)
    ny, nx = slices[0]["hu"].shape
    thick0 = slices[0]["thick"]
    if thick0 and abs(thick0 - 3.0) > 0.6:
        logger.warning("[coca_gated] %s: slice thickness %.2f mm (expected 3.0 gated)",
                       xml_path, thick0)
    with Path(xml_path).open("rb") as fh:
        pl = plistlib.load(fh)
    used = set()
    n_oor = 0
    off_vocab: set = set()
    for im in pl.get("Images", []):
        idx = im.get("ImageIndex")
        if idx is None:
            continue
        if not (0 <= idx < n_sl):
            n_oor += 1
            continue
        s = slices[idx]
        for roi in im.get("ROIs", []):
            pts = _parse_point_px(roi)
            if len(pts) < 3:
                continue
            _rname = str(roi.get("Name", ""))
            vessel = _VESSEL.get(_rname, "LCA")
            if _rname not in _VESSEL:
                off_vocab.add(_rname)
            mask = _poly_mask(pts, ny, nx)
            calc = mask & (s["hu"] >= 130)
            if not calc.any():
                continue
            for comp in _cc_components(calc):
                area_mm2 = len(comp) * s["px_area"]
                if area_mm2 < 1.0:  # >=1 mm^2 minimum-lesion floor
                    continue
                peak = max(s["hu"][i, j] for (i, j) in comp)
                per_vessel[vessel] += area_mm2 * _weight(peak)
            used.add(idx)
    if n_oor:
        logger.warning("[coca_gated] %s: %d image(s) reference an out-of-range ImageIndex "
                       "(n_slices=%d); slice-count/z-order mapping is suspect", xml_path, n_oor, n_sl)
    if off_vocab:
        logger.warning("[coca_gated] %s: off-vocabulary ROI Name(s) coerced to LCA: %s",
                       xml_path, sorted(off_vocab))
    return {"total": float(sum(per_vessel.values())), **per_vessel, "n_slices_used": len(used)}


def _convert_one_gated(dir_id: str, dicom_files: list[Path], dest_root: Path) -> tuple[str, int, str]:
    """Convert one gated patient's DICOMs (grouped by the ``patient/<dir_id>`` dir) to ``<dir_id>/scan.nii.gz``."""
    import dicom2nifti  # noqa: PLC0415

    out_pdir = dest_root / dir_id
    out_pdir.mkdir(parents=True, exist_ok=True)
    if (out_pdir / "scan.nii.gz").exists():
        return dir_id, len(dicom_files), "SKIP"
    # keep only the dominant series (match the scored series)
    import re as _re2
    from collections import Counter as _Counter2
    _val = [f for f in dicom_files if _re2.match(r"IM-(\d+)-\d+\.dcm$", f.name)]
    if _val:
        _dom = _Counter2(_re2.match(r"IM-(\d+)-\d+\.dcm$", f.name).group(1) for f in _val).most_common(1)[0][0]
        dicom_files = [f for f in _val if _re2.match(r"IM-(\d+)-\d+\.dcm$", f.name).group(1) == _dom]
    largest_size = 0
    largest_path = None
    with tempfile.TemporaryDirectory(prefix=f"cocag-{dir_id}-") as tmp_root:
        tmp_in = Path(tmp_root) / "in"
        tmp_in.mkdir()
        for f in dicom_files:
            (tmp_in / f.name).symlink_to(f.resolve())
        tmp_out = Path(tmp_root) / "out"
        tmp_out.mkdir()
        try:
            dicom2nifti.convert_directory(str(tmp_in), str(tmp_out), reorient=False)
        except Exception as e:
            logger.debug("convert_directory failed for %s: %s", dir_id, e)
        for nii in Path(tmp_out).glob("*.nii.gz"):
            final_path = out_pdir / nii.name
            shutil.move(str(nii), str(final_path))
            if final_path.stat().st_size > largest_size:
                largest_size = final_path.stat().st_size
                largest_path = final_path
    if largest_path is None:
        # SimpleITK fallback for series dicom2nifti rejects
        try:
            import SimpleITK as sitk  # noqa: PLC0415
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames([str(f) for f in sorted(dicom_files, key=lambda p: p.name)])
            img = reader.Execute()
            largest_path = out_pdir / "series_sitk.nii.gz"
            sitk.WriteImage(img, str(largest_path))
            largest_size = largest_path.stat().st_size
        except Exception as e:
            logger.debug("SimpleITK fallback failed for %s: %s", dir_id, e)
            return dir_id, len(dicom_files), "FAIL_CONVERT"
    (out_pdir / "scan.nii.gz").symlink_to(largest_path.name)  # relative
    return dir_id, len(dicom_files), f"OK (largest {largest_size // 1024} KB)"


def _convert_gated(staged_table_dir: Path, dest: Path, *, workers: int = 4,
                   limit: int | None = None) -> None:
    """Convert the staged gated DICOM tree to per-patient NIfTI under ``dest/<n>/scan.nii.gz``."""
    staged_table_dir, dest = Path(staged_table_dir), Path(dest)
    if not staged_table_dir.exists():
        raise SystemExit(f"[coca_gated] staged dir not found: {staged_table_dir}")
    dest.mkdir(parents=True, exist_ok=True)

    by_dir: dict[str, list[Path]] = defaultdict(list)
    pat_root = staged_table_dir / "patient"
    search_root = pat_root if pat_root.exists() else staged_table_dir
    for f in search_root.rglob("*.dcm"):
        if not f.is_file():
            continue
        name = f.name
        if name.startswith(".") or not name.startswith("IM-"):
            continue  # Azure temp cruft
        rel = f.relative_to(search_root)
        dir_id = rel.parts[0] if rel.parts else None
        if dir_id and dir_id.isdigit():
            by_dir[dir_id].append(f)
    logger.info("[coca_gated] found %d patient dirs (%d DICOM files)",
                len(by_dir), sum(len(v) for v in by_dir.values()))

    dir_ids = sorted(by_dir, key=lambda d: int(d))
    if limit:
        dir_ids = dir_ids[:limit]
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_convert_one_gated, d, by_dir[d], dest): d for d in dir_ids}
        for i, fut in enumerate(as_completed(futs)):
            dir_id, n_files, status = fut.result()
            if status.startswith("OK"):
                n_ok += 1
            elif status == "SKIP":
                n_skip += 1
            else:
                n_fail += 1
                logger.warning("FAIL %s (%d files): %s", dir_id, n_files, status)
            if (i + 1) % 25 == 0:
                logger.info("progress: %d/%d (ok=%d skip=%d fail=%d)",
                            i + 1, len(dir_ids), n_ok, n_skip, n_fail)
    logger.info("[coca_gated] DONE: ok=%d skip=%d fail=%d / %d", n_ok, n_skip, n_fail, len(dir_ids))
    if n_fail > max(5, len(dir_ids) // 20):
        raise RuntimeError(f"[coca_gated] {n_fail}/{len(dir_ids)} conversions failed (>5%, aborting)")


def stage(dest: Path, *, token: str | None = None, workers: int = 4,
          limit: int | None = None) -> None:
    """Download the COCA **gated** table from Redivis and convert to ``<dest>/nifti_gated/<n>/scan.nii.gz``."""
    _ = token or engine.require_env(
        "REDIVIS_API_TOKEN", hint="Redivis API token for Stanford AIMI COCA "
        f"({_QUALIFIED_REF})")
    dest = Path(dest)
    coca._download(dest, table=_TABLE, limit=limit)
    _convert_gated(dest / _TABLE, dest / _NIFTI_SUBDIR, workers=workers, limit=limit)


def _cac_drs_grade(total: float) -> int:
    """CAC-DRS Agatston grade (Hecht-2018): A0=0, A1=1-99, A2=100-299, A3>=300."""
    if total <= 0:
        return 0
    if total < 100:
        return 1
    if total < 300:
        return 2
    return 3


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 xml_dir: Path | None = None,
                 dicom_dir: Path | None = None,
                 staged_dir: Path | None = None,
                 jsonl_dir: Path | None = None,
                 limit: int | None = None) -> None:
    """Derive ``coca_gated_labels.json`` + jsonl + endpoint sidecars from the gated segmentations."""
    out_dir = Path(out_dir)
    jsonl_dir = Path(jsonl_dir) if jsonl_dir is not None else out_dir / "coca_gated_jsonl"
    if dicom_dir is None:
        dicom_dir = engine.expandvars("$DATA_ROOT/radiology/coca_gated/gated/patient")
    else:
        dicom_dir = Path(dicom_dir)
    if xml_dir is None:
        xml_dir = engine.expandvars("$DATA_ROOT/radiology/coca_gated/gated/calcium_xml")
    else:
        xml_dir = Path(xml_dir)
    if staged_dir is None:
        staged_dir = engine.expandvars("$DATA_ROOT/radiology/coca_gated/nifti_gated")
    else:
        staged_dir = Path(staged_dir)

    if not dicom_dir.exists():
        raise SystemExit(f"[coca_gated] DICOM patient dir not found: {dicom_dir}")

    # full gated patient set = every integer-named patient/<n> dir
    dir_ids = sorted((p.name for p in dicom_dir.iterdir() if p.is_dir() and p.name.isdigit()),
                     key=lambda d: int(d))
    if limit:
        dir_ids = dir_ids[:limit]
    print(f"  {len(dir_ids)} gated patient dirs under {dicom_dir}", file=sys.stderr)

    # coverage floor from committed manifests
    prior = {s: (sum(1 for _ in (jsonl_dir / f"{s}.jsonl").open()) if (jsonl_dir / f"{s}.jsonl").exists() else 0)
             for s in ("train", "dev", "test")}
    prior_total = sum(prior.values())

    qa = _QA_KEY
    qa_results: dict[str, dict] = {}
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    endpoint_records: list[dict] = []
    matched = 0
    dropped: list = []
    n_pos = n_xml = 0
    for dir_id in dir_ids:
        split = engine.stable_split(dir_id)
        xml_path = xml_dir / f"{dir_id}.xml"
        if xml_path.exists():
            n_xml += 1
            ag = _agatston_from_xml(xml_path, dicom_dir / dir_id)
        else:
            ag = {"total": 0.0, "LAD": 0.0, "LCX": 0.0, "RCA": 0.0, "LCA": 0.0, "n_slices_used": 0}
        total = ag["total"]
        label = int(total > 0)
        n_pos += label
        qa_results[dir_id] = {
            "split": split,
            "qa_results": {"default_qa": [{qa: label}]},
            "patient_id": dir_id,
            "agatston_total": total,
            "agatston_lca": ag["LCA"],
            "agatston_lad": ag["LAD"],
            "agatston_lcx": ag["LCX"],
            "agatston_rca": ag["RCA"],
        }
        vmax = max(ag["LAD"], ag["LCX"], ag["RCA"], ag["LCA"])
        n_vessels = sum(1 for v in _VESSELS if ag[v] > 0)
        endpoint_records.append({"pid": dir_id, "split": split, "total": total,
                                 "lad": ag["LAD"], "vmax": vmax,
                                 "lcx": ag["LCX"], "rca": ag["RCA"], "lca": ag["LCA"],
                                 "n_vessels": n_vessels,
                                 "drs_grade": _cac_drs_grade(total)})
        candidate_paths = [
            staged_dir / dir_id / "scan.nii.gz",
            staged_dir / dir_id / "series_sitk.nii.gz",
        ]
        path = next((p for p in candidate_paths if p.exists()), None)
        if path is not None:
            by_split[split].append({"sample_name": dir_id,
                                     "nii_path": engine.data_root_relative(path.resolve())})
            matched += 1
        else:
            dropped.append((dir_id, total, label))
    print(f"  Agatston: {n_xml}/{len(dir_ids)} patients segmented (rest calcium-negative); "
          f"CAC-positive (total>0): {n_pos}/{len(dir_ids)} = {100*n_pos/max(len(dir_ids),1):.1f}%",
          file=sys.stderr)
    print(f"  matched against staged NIfTI: {matched}/{len(dir_ids)}", file=sys.stderr)
    for did, tot, lab in dropped:
        print(f"  DROPPED (no NIfTI): pid={did} total={tot:.1f} label={lab}", file=sys.stderr)
    _pos_dropped = [d for d in dropped if d[2]]
    if _pos_dropped:
        logger.warning("[coca_gated] %d CAC-POSITIVE patient(s) dropped from eval splits (no NIfTI): %s",
                       len(_pos_dropped), [d[0] for d in _pos_dropped])

    # coverage / truncation guards
    if prior_total and matched < prior_total:
        raise SystemExit(f"[coca_gated] COVERAGE REGRESSION: matched {matched} < committed manifest "
                         f"{prior_total} ({prior}). Refusing to overwrite -- check nifti_gated.")
    for s in ("train", "dev", "test"):
        if prior[s] and not by_split[s]:
            raise SystemExit(f"[coca_gated] REFUSING to write empty '{s}' split over committed {prior[s]} rows.")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    # compact JSON to match the committed file
    (out_dir / "coca_gated_labels.json").write_text(json.dumps(qa_results))
    for s, rows in by_split.items():
        engine.write_jsonl(rows, jsonl_dir / f"{s}.jsonl")
        npos = sum(1 for r in rows if qa_results[r["sample_name"]]["qa_results"]["default_qa"][0][qa])
        print(f"  {s}: n={len(rows)} ({npos} pos, {len(rows)-npos} neg)", file=sys.stderr)

    # per-vessel lcx/rca/lca + multivessel beyond the shared panel
    extra_tasks = (
        ("lcx", _QA_LCX, lambda r: int(r["lcx"] > 0)),
        ("rca", _QA_RCA, lambda r: int(r["rca"] > 0)),
        ("lca", _QA_LCA, lambda r: int(r["lca"] > 0)),
        ("multivessel", _QA_MULTIVESSEL, lambda r: int(r["n_vessels"] >= 2)),
    )
    tcount = coca._derive_cac_endpoints(endpoint_records, out_dir=out_dir,
                                        prefix="coca_gated", extra_tasks=extra_tasks)
    print(f"  tasks: significant={tcount['significant']} high={tcount['high']} lad={tcount['lad']} "
          f"lcx={tcount['lcx']} rca={tcount['rca']} lca={tcount['lca']} "
          f"multivessel={tcount['multivessel']} (+ deferred drs_ordinal, strict sidecars)",
          file=sys.stderr)


SPEC = DatasetSpec(
    name="coca_gated",
    access=Access.REDIVIS,
    modality="chest_ct",
    role="candidate",
    source=_QUALIFIED_REF,
    token_env="REDIVIS_API_TOKEN",
    committed_outputs=(
        "data/evaluation/coca_gated_labels.json",
        "data/evaluation/coca_gated_labels__significant.json",
        "data/evaluation/coca_gated_labels__high.json",
        "data/evaluation/coca_gated_labels__lad.json",
        "data/evaluation/coca_gated_labels__lcx.json",
        "data/evaluation/coca_gated_labels__rca.json",
        "data/evaluation/coca_gated_labels__lca.json",
        "data/evaluation/coca_gated_labels__multivessel.json",
        "data/evaluation/coca_gated_labels__cac_present_strict.json",
        "data/evaluation/coca_gated_labels__drs_ordinal.json",
        "data/evaluation/coca_gated_jsonl/train.jsonl",
        "data/evaluation/coca_gated_jsonl/dev.jsonl",
        "data/evaluation/coca_gated_jsonl/test.jsonl",
    ),
    notes="COCA GATED table (ECG-gated 3mm cardiac CT, Agatston reference-standard). "
          "Separate cohort from non-gated coca. 797 patient/<n> DICOM dirs (n=0..789); "
          "451 calcium_xml/<n>.xml segmentations map DIRECTLY to patient/<n> (geometrically "
          "verified); xml-less dirs = calcium-negative (total=0). Agatston 1990 from Point_px "
          "on the raw DICOM (ImageIndex=z-ascending slice, >=130 HU, area*weight, >=1mm2 floor). "
          "Anchor=CAC-present; panel=significant/high + per-vessel lad/lcx/rca/lca + multivessel; "
          "deferred CAC-DRS ordinal (corrected A2=100-299/A3>=300 boundary).",
    stage=stage,
    build_labels=build_labels,
)
