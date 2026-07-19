"""MIDRC-RICORD-1A/1B (RSNA COVID-19 Open Radiology DB): staging + labels."""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.spec import Access, DatasetSpec

_COLLECTIONS = (("1a", "midrc_ricord_1a"), ("1b", "midrc_ricord_1b"))
_MDAI_URL = ("https://www.cancerimagingarchive.net/wp-content/uploads/"
             "MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json_.zip")
_CLINICAL_URL = {
    "1a": "https://www.cancerimagingarchive.net/wp-content/uploads/"
          "MIDRC-RICORD-1a-Clinical-Data-Jan-13-2021.xlsx",
    "1b": "https://www.cancerimagingarchive.net/wp-content/uploads/"
          "MIDRC-RICORD-1b-Clinical-Data-Feb-2021.xlsx",
}
_APPEARANCE = ("Typical", "Indeterminate", "Atypical", "Negative for pneumonia")
# non-diagnostic series descriptions (fallback path only)
_EXCL_DESC = re.compile(r"scout|localiz|\bmip\b|bone\s*alg|detail\s*alg|cor\s*\dx\d|sag\s*\dx\d", re.I)

# QA keys must byte-match the cross-cohort benchmark cell definitions.
_QA_COVID = ("Is COVID-19 present on this chest CT (RT-PCR reference standard; "
             "MIDRC-RICORD 1A vs 1B)? (0=No, 1=Yes)")
_QA_TYPICAL = ("Is this chest CT an RSNA-typical appearance for COVID-19 pneumonia "
               "(Typical vs Indeterminate/Atypical/Negative; higher-specificity cut)? (0=No, 1=Yes)")
_QA_SUGGESTIVE = ("Is this chest CT suggestive of COVID-19 pneumonia "
                  "(RSNA Typical or Indeterminate vs Atypical or Negative; higher-sensitivity cut)? (0=No, 1=Yes)")
_QA_APPEARANCE = "RSNA COVID-19 CT appearance (0=Negative, 1=Atypical, 2=Indeterminate, 3=Typical)"
_QA_INDETERMINATE = "Is this chest CT an RSNA-indeterminate appearance for COVID-19? (0=No, 1=Yes)"
_QA_ATYPICAL = "Is this chest CT an RSNA-atypical appearance for COVID-19? (0=No, 1=Yes)"
_QA_NEGATIVE = "Is this chest CT RSNA-negative for pneumonia (imaging-negative)? (0=No, 1=Yes)"
_QA_INFECTIOUS = "Does this chest CT show infectious lung disease (RICORD reader)? (0=No, 1=Yes)"
_QA_OPACITY = "Does this chest CT contain a segmented infectious opacity (RICORD polygon)? (0=No, 1=Yes)"
_QA_OPACITY_BURDEN = "Number of segmented infectious-opacity polygons on this chest CT (RICORD)"
_QA_EFFUSION = "Does this chest CT show a pleural effusion (RICORD reader)? (0=No, 1=Yes)"
_QA_LYMPH = "Does this chest CT show lymphadenopathy (RICORD reader)? (0=No, 1=Yes)"
_QA_EMPHYSEMA = "Does this chest CT show emphysema (RICORD reader)? (0=No, 1=Yes)"
_QA_INADEQUATE = ("Did the RICORD reader flag this chest CT as QA-inadequate "
                  "(motion/breathing/inspiration/coverage artifact)? (0=No, 1=Yes)")
_APPEARANCE_ORD = {"Negative for pneumonia": 0, "Atypical": 1, "Indeterminate": 2, "Typical": 3}


def _download(url: str, target: Path, retries: int = 5) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
            target.write_bytes(data)
            return target
        except Exception:  # pragma: no cover
            if attempt == retries - 1:
                raise
    return target


def _fetch_mdai(ann_dir: Path) -> Path:
    """Download + unzip the RICORD-1A MD.ai annotation JSON into ann_dir; return the .json path."""
    out = ann_dir / "MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json"
    if out.exists() and out.stat().st_size > 0:
        return out
    ann_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_MDAI_URL, timeout=300) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    name = next(n for n in zf.namelist() if n.endswith(".json"))
    out.write_bytes(zf.read(name))
    return out


def _parse_mdai(json_path: Path) -> dict:
    """Parse the MD.ai export -> per-study appearance / findings / annotated-series / opacity-count."""
    mdai = json.loads(Path(json_path).read_text())
    label_name: dict[str, str] = {}
    for lg in mdai.get("labelGroups", []):
        for lab in lg.get("labels", []):
            label_name[lab["id"]] = lab["name"]
    appearance: dict[str, str] = {}
    findings: dict[str, set] = defaultdict(set)
    annotated_series: dict[str, set] = defaultdict(set)
    opacity_count: dict[str, int] = defaultdict(int)
    for a in mdai["datasets"][0]["annotations"]:
        su, se = a.get("StudyInstanceUID"), a.get("SeriesInstanceUID")
        nm = label_name.get(a.get("labelId"))
        if su and se:
            annotated_series[su].add(se)
        if su and nm:
            findings[su].add(nm)
            if nm in _APPEARANCE:
                appearance[su] = nm
            if nm == "Infectious opacity" and a.get("data"):
                opacity_count[su] += 1
    return {"appearance": appearance, "findings": findings,
            "annotated_series": annotated_series, "opacity_count": opacity_count}


def _series_metadata(collection: str) -> list[dict]:
    rows = engine.idc_query(f"""
        SELECT PatientID, StudyInstanceUID, SeriesInstanceUID,
               CAST(SeriesNumber AS VARCHAR) AS SeriesNumber,
               SeriesDescription, instanceCount
        FROM idc_index
        WHERE collection_id='{collection}' AND Modality='CT'""")
    for r in rows:
        r["instanceCount"] = int(float(r["instanceCount"]))
    return rows


def _series_number(r: dict) -> int:
    try:
        return int(float(r["SeriesNumber"]))
    except (TypeError, ValueError):
        return 1 << 30


def _series_dir(dicom_root: Path, r: dict) -> Path:
    return dicom_root / r["StudyInstanceUID"] / r["SeriesInstanceUID"]


def _have_count(dicom_root: Path, r: dict) -> int:
    d = _series_dir(dicom_root, r)
    return sum(1 for _ in d.glob("*.dcm")) if d.exists() else 0


def _download_verified(sel: list[dict], dicom_root: Path, *, max_retries: int = 5) -> None:
    """Download every series to completeness (local .dcm count >= IDC instanceCount), clearing partial dirs and retrying."""
    import shutil
    for attempt in range(max_retries):
        need = [r for r in sel if _have_count(dicom_root, r) < r["instanceCount"]]
        if not need:
            print(f"[midrc]   all {len(sel)} series complete (file count >= instanceCount)", flush=True)
            return
        print(f"[midrc]   {len(need)}/{len(sel)} series incomplete -> (re)download (attempt {attempt + 1}/{max_retries})",
              flush=True)
        for r in need:
            d = _series_dir(dicom_root, r)
            if d.exists():
                shutil.rmtree(d)
        engine.idc_download([r["SeriesInstanceUID"] for r in need], dicom_root,
                            dir_template="%StudyInstanceUID/%SeriesInstanceUID")
    still = [r for r in sel if _have_count(dicom_root, r) < r["instanceCount"]]
    if still:
        raise RuntimeError(f"[midrc] {len(still)} series still incomplete after {max_retries} retries: "
                           f"{[(r['SeriesInstanceUID'][-10:], _have_count(dicom_root, r), r['instanceCount']) for r in still[:5]]}")


def _select_series(rows: list[dict], annotated_series: dict[str, set]) -> list[dict]:
    """One primary series per study: label-aligned when annotated (1A), else largest non-scout series."""
    by_study: dict[str, list] = defaultdict(list)
    for r in rows:
        by_study[r["StudyInstanceUID"]].append(r)
    selected = []
    for su, series in by_study.items():
        annot = annotated_series.get(su, set()) & {r["SeriesInstanceUID"] for r in series}
        if annot:
            cand = [r for r in series if r["SeriesInstanceUID"] in annot]
        else:
            cand = [r for r in series if not (r["SeriesDescription"] and _EXCL_DESC.search(r["SeriesDescription"]))] \
                or series
        selected.append(sorted(cand, key=lambda r: (-r["instanceCount"], _series_number(r)))[0])
    return selected


def _convert_one(args) -> str:
    dcm_dir, out_path = args
    import shutil
    import dicom2nifti
    import dicom2nifti.settings as d2n
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"
    dcm_dir = Path(dcm_dir)
    if not dcm_dir.is_dir():
        # idc-index dir_template leaf == SeriesInstanceUID; locate it if nested differently
        hits = [d for d in dcm_dir.parent.parent.rglob(dcm_dir.name) if d.is_dir()] \
            if dcm_dir.parent.parent.exists() else []
        if not hits:
            return f"FAIL no-dicom-dir {out_path.name}"
        dcm_dir = hits[0]
    # irregular z-spacing: dicom2nifti silently skips unless the increment check is off
    d2n.disable_validate_slice_increment()
    with tempfile.TemporaryDirectory() as td:
        try:
            dicom2nifti.convert_directory(str(dcm_dir), td, compression=True, reorient=False)
        except Exception as exc:  # pragma: no cover
            return f"FAIL convert {out_path.name}: {type(exc).__name__}"
        niis = sorted(Path(td).glob("*.nii.gz"), key=lambda p: p.stat().st_size, reverse=True)
        if not niis:
            return _convert_sitk(dcm_dir, out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(niis[0]), str(out_path))
    return "ok"


def _convert_sitk(dcm_dir: "Path", out_path: "Path") -> str:
    """SimpleITK fallback: explicit ImagePositionPatient-z sort + duplicate-z dedupe."""
    try:
        import SimpleITK as sitk
        import pydicom
        zpos: dict[float, str] = {}
        for f in Path(dcm_dir).glob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                z = round(float(ds.ImagePositionPatient[2]), 3)
                zpos.setdefault(z, str(f))
            except Exception:
                continue
        ordered = [zpos[z] for z in sorted(zpos)]
        if len(ordered) < 10:
            return f"FAIL sitk-too-few-slices({len(ordered)}) {out_path.name}"
        r = sitk.ImageSeriesReader()
        r.SetFileNames(ordered)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(r.Execute(), str(out_path))
        return "ok-sitk"
    except Exception as exc:  # pragma: no cover
        return f"FAIL no-nifti {out_path.name}: {type(exc).__name__}"


def stage(dest: Path, *, token: str | None = None, workers: int = 8, convert: bool = True) -> None:
    """Stage MIDRC-RICORD-1A/1B: TCIA annotations + IDC imaging -> <dest>/{1a,1b}/{dicom,nifti}/."""
    dest = Path(dest)
    ann_dir = dest / "labels" / "annotations"
    print("[midrc] downloading TCIA annotations (MD.ai JSON + clinical XLSX)", flush=True)
    mdai = _parse_mdai(_fetch_mdai(ann_dir))
    for tag in ("1a", "1b"):
        _download(_CLINICAL_URL[tag], ann_dir / f"MIDRC-RICORD-{tag}-Clinical-Data.xlsx")

    manifest: list[dict] = []
    for tag, coll in _COLLECTIONS:
        rows = _series_metadata(coll)
        sel = _select_series(rows, mdai["annotated_series"])
        if tag == "1a":
            aligned = sum(1 for r in sel
                          if r["SeriesInstanceUID"] in mdai["annotated_series"].get(r["StudyInstanceUID"], set()))
            print(f"[midrc] {tag}: {len(sel)} studies, label-aligned {aligned}/{len(sel)} "
                  f"(rest single-series fallback)", flush=True)
        else:
            print(f"[midrc] {tag}: {len(sel)} studies (single-series)", flush=True)

        dicom_root = dest / tag / "dicom"
        nifti_root = dest / tag / "nifti"
        _download_verified(sel, dicom_root)
        jobs = []
        for r in sel:
            sid = f'{r["PatientID"]}__{r["StudyInstanceUID"]}'
            dcm_dir = dicom_root / r["StudyInstanceUID"] / r["SeriesInstanceUID"]
            out_path = nifti_root / f"{sid}.nii.gz"
            jobs.append((str(dcm_dir), str(out_path)))
            manifest.append({"PatientID": r["PatientID"], "StudyInstanceUID": r["StudyInstanceUID"],
                             "SeriesInstanceUID": r["SeriesInstanceUID"], "collection": tag,
                             "sid": sid, "nii_path": engine.data_root_relative(out_path)})
        if convert:
            print(f"[midrc] {tag}: converting {len(jobs)} series (dicom2nifti, {workers}-way)", flush=True)
            ok = 0
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for res in ex.map(_convert_one, jobs):
                    if res.startswith("ok") or res == "skip":
                        ok += 1
                    else:
                        print(f"  {res}", file=sys.stderr)
            print(f"[midrc] {tag}: converted {ok}/{len(jobs)}", flush=True)

    man_path = dest / "manifest.csv"
    with man_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["PatientID", "StudyInstanceUID", "SeriesInstanceUID",
                                          "collection", "sid", "nii_path"])
        w.writeheader()
        w.writerows(manifest)
    print(f"[midrc] staged {len(manifest)} volumes; manifest -> {man_path}", flush=True)


def convert_staged(dest: Path, *, workers: int = 16) -> None:
    """Parallel DICOM->NIfTI over the already-downloaded series in manifest.csv (idempotent)."""
    dest = Path(dest)
    rows = list(csv.DictReader((dest / "manifest.csv").open()))
    jobs = [(str(dest / r["collection"] / "dicom" / r["StudyInstanceUID"] / r["SeriesInstanceUID"]),
             str(dest / r["collection"] / "nifti" / f'{r["sid"]}.nii.gz')) for r in rows]
    ok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_convert_one, jobs):
            if res.startswith("ok") or res == "skip":
                ok += 1
            else:
                print(f"  {res}", file=sys.stderr)
    print(f"[midrc] convert_staged: {ok}/{len(jobs)} NIfTI", flush=True)


def _read_rtpcr(ann_dir: Path) -> dict[str, int]:
    """StudyInstanceUID -> COVID label from the clinical XLSX RT-PCR Result column."""
    import pandas as pd
    out: dict[str, int] = {}
    for tag in ("1a", "1b"):
        p = ann_dir / f"MIDRC-RICORD-{tag}-Clinical-Data.xlsx"
        if not p.exists():
            continue
        df = pd.read_excel(p)
        df.columns = [str(c).strip() for c in df.columns]
        uid_col = next(c for c in df.columns if "Study UID" in c or "StudyInstanceUID" in c)
        res_col = next(c for c in df.columns if c.strip().lower() == "result")
        for _, row in df.iterrows():
            uid = str(row[uid_col]).strip()
            res = str(row[res_col]).strip().upper()
            out[uid] = int("DETECT" in res and "NOT" not in res)
    return out


def _emit(out_dir: Path, name: str, key: str, sid_label: dict) -> None:
    engine.write_json({sid: {"split": split, "qa_results": {"default_qa": [{key: lab}]}, "sample_name": sid}
                       for sid, (lab, split) in sid_label.items()},
                      out_dir / name)


def build_labels(out_dir: Path = Path("data/evaluation"), *,
                 staged_dir: Path | None = None,
                 jsonl_dir: Path = Path("data/evaluation/midrc_ricord_jsonl")) -> None:
    """Derive the MIDRC-RICORD label panel from the staged manifest + TCIA annotations."""
    staged_dir = Path(staged_dir) if staged_dir is not None \
        else engine.expandvars("$DATA_ROOT/radiology/midrc_ricord")
    out_dir, jsonl_dir = Path(out_dir), Path(jsonl_dir)
    ann_dir = staged_dir / "labels" / "annotations"
    man_path = staged_dir / "manifest.csv"
    if not man_path.exists():
        raise SystemExit(f"manifest.csv not found: {man_path} -- run stage() first")

    rows = list(csv.DictReader(man_path.open()))
    mdai = _parse_mdai(_fetch_mdai(ann_dir))
    rtpcr = _read_rtpcr(ann_dir)

    covid_qa: dict[str, dict] = {}
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    # per-sid (label, split) maps for each 1A appearance/finding cell
    typ, sug, ind, atyp, neg, appear, infl, opac, opac_n = ({} for _ in range(9))
    eff, lymph, emph, inadeq = {}, {}, {}, {}
    n_pos = n_1a = 0
    for r in rows:
        sid, su, coll = r["sid"], r["StudyInstanceUID"], r["collection"]
        split = engine.stable_split(r["PatientID"])            # subject-level
        nii_path = r["nii_path"]
        covid = rtpcr.get(su, 1 if coll == "1a" else 0)
        assert covid == (1 if coll == "1a" else 0), f"RT-PCR/collection mismatch for {su}"
        n_pos += covid
        covid_qa[sid] = {"split": split, "qa_results": {"default_qa": [{_QA_COVID: covid}]},
                         "sample_name": sid, "collection": coll.upper(), "nii_path": nii_path}
        by_split[split].append({"sample_name": sid, "nii_path": nii_path, "covid_positive": covid})
        if coll != "1a":
            continue
        n_1a += 1
        app = mdai["appearance"].get(su)
        find = mdai["findings"].get(su, set())
        oc = mdai["opacity_count"].get(su, 0)
        typ[sid] = (int(app == "Typical"), split)
        sug[sid] = (int(app in ("Typical", "Indeterminate")), split)
        ind[sid] = (int(app == "Indeterminate"), split)
        atyp[sid] = (int(app == "Atypical"), split)
        neg[sid] = (int(app == "Negative for pneumonia"), split)
        appear[sid] = (_APPEARANCE_ORD.get(app, -1), split)
        infl[sid] = (int("Infectious lung disease" in find), split)
        opac[sid] = (int(oc > 0), split)
        opac_n[sid] = (oc, split)
        eff[sid] = (int("Effusion" in find), split)
        lymph[sid] = (int("Lymphadenopathy" in find), split)
        emph[sid] = (int("Emphysema" in find), split)
        inadeq[sid] = (int(any("inadequate" in f.lower() for f in find)), split)

    print(f"  COVID+: {n_pos}/{len(rows)} = {100*n_pos/max(1,len(rows)):.1f}% | 1A appearance studies: {n_1a}",
          file=sys.stderr)
    print(f"  typical {sum(v for v,_ in typ.values())}/{n_1a} | suggestive {sum(v for v,_ in sug.values())}/{n_1a} | "
          f"opacity {sum(v for v,_ in opac.values())}/{n_1a}", file=sys.stderr)
    print("  splits: " + ", ".join(f"{k}={len(v)}" for k, v in by_split.items()), file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    engine.write_json(covid_qa, out_dir / "midrc_ricord_labels.json")
    _emit(out_dir, "midrc_ricord_labels__typical.json", _QA_TYPICAL, typ)
    _emit(out_dir, "midrc_ricord_labels__suggestive.json", _QA_SUGGESTIVE, sug)
    _emit(out_dir, "midrc_ricord_labels__appearance_4way.json", _QA_APPEARANCE, appear)
    _emit(out_dir, "midrc_ricord_labels__indeterminate.json", _QA_INDETERMINATE, ind)
    _emit(out_dir, "midrc_ricord_labels__atypical.json", _QA_ATYPICAL, atyp)
    _emit(out_dir, "midrc_ricord_labels__negative_for_pneumonia.json", _QA_NEGATIVE, neg)
    _emit(out_dir, "midrc_ricord_labels__infectious_lung_disease.json", _QA_INFECTIOUS, infl)
    _emit(out_dir, "midrc_ricord_labels__opacity_present.json", _QA_OPACITY, opac)
    _emit(out_dir, "midrc_ricord_labels__opacity_burden.json", _QA_OPACITY_BURDEN, opac_n)
    _emit(out_dir, "midrc_ricord_labels__effusion.json", _QA_EFFUSION, eff)
    _emit(out_dir, "midrc_ricord_labels__lymphadenopathy.json", _QA_LYMPH, lymph)
    _emit(out_dir, "midrc_ricord_labels__emphysema.json", _QA_EMPHYSEMA, emph)
    _emit(out_dir, "midrc_ricord_labels__qa_inadequate.json", _QA_INADEQUATE, inadeq)
    for split, rws in by_split.items():
        engine.write_jsonl(rws, jsonl_dir / f"{split}.jsonl")
    print(f"  wrote midrc_ricord_labels.json (COVID) + 13 sidecars + {len(by_split)} JSONLs", file=sys.stderr)


SPEC = DatasetSpec(
    name="midrc_ricord",
    access=Access.IDC,
    modality="chest_ct",
    role="candidate",
    source="midrc_ricord_1a + midrc_ricord_1b",
    token_env=None,
    committed_outputs=(
        "data/evaluation/midrc_ricord_labels.json",
        "data/evaluation/midrc_ricord_labels__typical.json",
        "data/evaluation/midrc_ricord_labels__suggestive.json",
        "data/evaluation/midrc_ricord_labels__appearance_4way.json",
        "data/evaluation/midrc_ricord_labels__indeterminate.json",
        "data/evaluation/midrc_ricord_labels__atypical.json",
        "data/evaluation/midrc_ricord_labels__negative_for_pneumonia.json",
        "data/evaluation/midrc_ricord_labels__infectious_lung_disease.json",
        "data/evaluation/midrc_ricord_labels__opacity_present.json",
        "data/evaluation/midrc_ricord_labels__opacity_burden.json",
        "data/evaluation/midrc_ricord_labels__effusion.json",
        "data/evaluation/midrc_ricord_labels__lymphadenopathy.json",
        "data/evaluation/midrc_ricord_labels__emphysema.json",
        "data/evaluation/midrc_ricord_labels__qa_inadequate.json",
        "data/evaluation/midrc_ricord_jsonl/train.jsonl",
        "data/evaluation/midrc_ricord_jsonl/dev.jsonl",
        "data/evaluation/midrc_ricord_jsonl/test.jsonl",
    ),
    notes="TCIA/IDC (CC-BY-NC 4.0, eval-only); 240 studies / 227 patients (1A COVID+ 120 / 1B "
          "COVID- 120), 50%; IDC DICOM (label-aligned 1A series selection) -> dicom2nifti; "
          "RICORD-1A MD.ai appearance panel (typical/suggestive scored) from TCIA; RT-PCR COVID "
          "from clinical XLSX; subject-grouped (sid=PatientID__StudyUID) 5-fold CV.",
    stage=stage,
    build_labels=build_labels,
)
