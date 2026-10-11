"""Demo scans: download from the NCI Imaging Data Commons, convert, and derive the web tiers.

Only cohorts that allow redistribution and were external in the thesis are used (LIDC-IDRI,
CC BY 3.0; MIDRC-RICORD, CC BY-NC 4.0). CT-RATE is never bundled: its gated terms forbid
redistributing any part of it.

Per case the web app gets:

* ``<id>.view.nii.gz``: canonical RAS int16 HU for the viewer (in-plane halved above 384 px).
* ``<id>.grid.nii.gz``: the exact 128x256x256 model-grid HU volume (HU x 10 as int16) with the
  world affine of the resize, so the browser only has to window it.
* ``<id>.full.nii.gz`` (hero case only): the native-resolution volume, for the complete live
  pipeline (decode, reorient, resize, window) in the browser.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import numpy.typing as npt

from mucodi_demo import reference

HU_SCALE = 10  # model-grid volumes store round(HU * 10) as int16


@dataclass(frozen=True)
class CaseSpec:
    id: str
    collection: str
    patient: str
    series_uid: str
    licence: str
    title: str
    note: str
    hero: bool = False


def download_series(spec: CaseSpec, dest: Path) -> Path:
    """Fetch the DICOM series from IDC (public buckets, no login) into ``dest/<id>/dicom``."""
    from idc_index import index

    target = dest / spec.id / "dicom"
    if target.exists() and any(target.rglob("*.dcm")):
        return target
    target.mkdir(parents=True, exist_ok=True)
    client = index.IDCClient()
    client.download_from_selection(
        seriesInstanceUID=[spec.series_uid], downloadDir=str(target), dirTemplate=None, quiet=True
    )
    return target


def convert(dicom_dir: Path, out: Path) -> Path:
    """DICOM series -> NIfTI with dcm2niix (same release as the WASM build used in the browser)."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.nii*"):
        old.unlink()
    subprocess.run(
        ["dcm2niix", "-z", "n", "-b", "y", "-f", "native", "-o", str(out), str(dicom_dir)],
        check=True,
        capture_output=True,
    )
    files = sorted(out.glob("native*.nii"))
    if len(files) != 1:
        raise RuntimeError(f"dcm2niix produced {len(files)} volumes in {out}: {files}")
    return files[0]


def grid_affine(
    canonical: npt.NDArray[np.float64], shape_xyz: tuple[int, int, int]
) -> npt.NDArray[np.float64]:
    """World affine of the model grid stored in NIfTI voxel order (i, j, k) = (W, H, D).

    That order makes the stored (Fortran) data identical to the model's (D, H, W) C-order array,
    so the browser can use it without reordering. Model voxel (d, h, w) samples canonical voxel
    ((h + 0.5) sx - 0.5, (w + 0.5) sy - 0.5, (d + 0.5) sz - 0.5), the ``align_corners=False``
    mapping of ``F.interpolate``.
    """
    sx = shape_xyz[0] / reference.TARGET_SHAPE[1]
    sy = shape_xyz[1] / reference.TARGET_SHAPE[2]
    sz = shape_xyz[2] / reference.TARGET_SHAPE[0]
    scale: npt.NDArray[np.float64] = np.array(
        [
            [0, sx, 0, 0.5 * sx - 0.5],
            [sy, 0, 0, 0.5 * sy - 0.5],
            [0, 0, sz, 0.5 * sz - 0.5],
            [0, 0, 0, 1],
        ]
    )
    return canonical @ scale


def _save(
    data_xyz: npt.NDArray[np.int16],
    affine: npt.NDArray[np.float64],
    path: Path,
    scl_slope: float = 1.0,
) -> None:
    img = nib.Nifti1Image(data_xyz, affine)
    img.header.set_data_dtype(data_xyz.dtype)
    img.header.set_xyzt_units("mm")
    img.header["scl_slope"] = scl_slope
    img.header["scl_inter"] = 0.0
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    raw = path.with_suffix("")  # strip .gz
    nib.save(img, str(raw))
    with raw.open("rb") as src, gzip.open(path, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    raw.unlink()


def build_case(spec: CaseSpec, downloads: Path, out_dir: Path) -> dict[str, object]:
    dicom = download_series(spec, downloads)
    native = convert(dicom, downloads / spec.id / "nifti")
    out_dir.mkdir(parents=True, exist_ok=True)

    hu = reference.read_hu(native)  # (D, H, W) canonical RAS
    canon = reference.canonical_affine(native)
    hu_xyz = hu.transpose(1, 2, 0)  # back to canonical (x, y, z)
    zooms = np.sqrt((canon[:3, :3] ** 2).sum(axis=0))

    # Viewer volume: in-plane halved (2x2 mean) above 384 px, z untouched.
    view = hu_xyz
    view_affine = canon.copy()
    if hu_xyz.shape[0] > 384 and hu_xyz.shape[1] > 384:
        x, y = hu_xyz.shape[0] // 2 * 2, hu_xyz.shape[1] // 2 * 2
        view = hu_xyz[:x, :y].reshape(x // 2, 2, y // 2, 2, -1).mean(axis=(1, 3))
        half = np.diag([2.0, 2.0, 1.0, 1.0])
        half[:3, 3] = [0.5, 0.5, 0.0]
        view_affine = canon @ half
    _save(
        np.clip(np.rint(view), -32768, 32767).astype(np.int16),
        view_affine,
        out_dir / f"{spec.id}.view.nii.gz",
    )

    # Model grid: exact thesis resize, stored as HU x 10.
    resized = reference.resize(hu)  # (D, H, W) = (128, 256, 256)
    grid_ijk = resized.transpose(
        2, 1, 0
    )  # NIfTI (i, j, k) = (W, H, D): Fortran order == (D, H, W) C order
    grid = np.clip(np.rint(grid_ijk * HU_SCALE), -32768, 32767).astype(np.int16)
    _save(
        grid,
        grid_affine(canon, cast(tuple[int, int, int], hu_xyz.shape)),
        out_dir / f"{spec.id}.grid.nii.gz",
        1.0 / HU_SCALE,
    )

    files = {"view": f"{spec.id}.view.nii.gz", "grid": f"{spec.id}.grid.nii.gz"}
    if spec.hero:
        _save(
            np.clip(np.rint(hu_xyz), -32768, 32767).astype(np.int16),
            canon,
            out_dir / f"{spec.id}.full.nii.gz",
        )
        files["full"] = f"{spec.id}.full.nii.gz"

    sidecar = (
        json.loads(native.with_suffix(".json").read_text())
        if native.with_suffix(".json").exists()
        else {}
    )
    info: dict[str, object] = {
        **asdict(spec),
        "files": files,
        "native_shape_xyz": list(hu_xyz.shape),
        "spacing_mm": [round(float(z), 4) for z in zooms],
        "extent_mm": [round(float(z * n), 1) for z, n in zip(zooms, hu_xyz.shape, strict=True)],
        "manufacturer": sidecar.get("Manufacturer"),
        "convolution_kernel": sidecar.get("ConvolutionKernel"),
        "slice_thickness_mm": sidecar.get("SliceThickness"),
    }
    return info


LIDC_ATTRIBUTION = (
    "The authors acknowledge the National Cancer Institute and the Foundation for the National "
    "Institutes of Health, and their critical role in the creation of the free publicly available "
    "LIDC/IDRI Database used in this study."
)

CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        id="lidc-0001",
        collection="LIDC-IDRI",
        patient="LIDC-IDRI-0001",
        series_uid="1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192",
        licence="CC BY 3.0",
        title="Chest CT, lung nodule collection",
        note="LIDC-IDRI, a public collection of thoracic CT scans with radiologist-marked "
        "lung nodules. External cohort in the thesis benchmark.",
        hero=True,
    ),
    CaseSpec(
        id="ricord-000082",
        collection="MIDRC-RICORD-1A",
        patient="MIDRC-RICORD-1A-419639-000082",
        series_uid="1.2.826.0.1.3680043.10.474.419639.403650391453800318566197197100",
        licence="CC BY-NC 4.0",
        title="Chest CT, COVID-19 positive collection",
        note="MIDRC-RICORD-1a, the RSNA International COVID-19 Open Radiology Database, COVID-19 "
        "positive cohort. External cohort in the thesis benchmark.",
    ),
    CaseSpec(
        id="ricord-000800",
        collection="MIDRC-RICORD-1A",
        patient="MIDRC-RICORD-1A-419639-000800",
        series_uid="1.2.826.0.1.3680043.10.474.419639.689197686214923485781691870218",
        licence="CC BY-NC 4.0",
        title="Chest CT, COVID-19 positive collection",
        note="MIDRC-RICORD-1a, the RSNA International COVID-19 Open Radiology Database, COVID-19 "
        "positive cohort. External cohort in the thesis benchmark.",
    ),
)


def build_cases(downloads: Path, out_dir: Path) -> list[dict[str, object]]:
    infos = [build_case(spec, downloads, out_dir) for spec in CASES]
    (out_dir / "cases.json").write_text(
        json.dumps({"lidc_attribution": LIDC_ATTRIBUTION, "cases": infos}, indent=2) + "\n"
    )
    return infos
