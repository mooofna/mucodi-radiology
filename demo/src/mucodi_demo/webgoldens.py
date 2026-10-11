"""Golden vectors that pin the browser preprocessing to nibabel and the thesis code.

Writes small, committed fixtures into ``web/src/lib/volume/__fixtures__``:

* ``orientation.json``: nibabel io_orientation and as_closest_canonical affines for all 48
  axis permutations and flips plus oblique and sheared affines.
* ``resize.json``: torch trilinear resizes (thesis code path) of small random volumes.
* ``windows.json``: the thesis windowing on a sweep of HU values.
* ``nifti/*.nii(.gz)`` + ``nifti.json``: tiny NIfTI files across datatypes, endianness, qform
  and sform codes, scaling and NIfTI-2, with the canonical float32 arrays nibabel produces.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt

from mucodi_demo import reference

FIXTURES = Path(__file__).resolve().parents[3] / "web" / "src" / "lib" / "volume" / "__fixtures__"


def _rot(axis: int, angle_deg: float) -> npt.NDArray[np.float64]:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    r = np.eye(3)
    i, j = [k for k in range(3) if k != axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def orientation_cases(rng: np.random.Generator) -> list[dict[str, object]]:
    out = []
    shape = (7, 5, 3)
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            zooms = rng.uniform(0.5, 3.0, 3)
            for i, ax in enumerate(perm):
                m[ax, i] = flips[i] * zooms[i]
            aff = np.eye(4)
            aff[:3, :3] = m
            aff[:3, 3] = rng.uniform(-200, 200, 3)
            out.append(aff)
    for k in range(12):
        m = _rot(k % 3, rng.uniform(-35, 35)) @ _rot((k + 1) % 3, rng.uniform(-20, 20))
        m = m @ np.diag(rng.uniform(0.5, 2.5, 3) * rng.choice([-1, 1], 3))
        if k % 4 == 3:
            m = m @ np.array([[1, 0.15, 0], [0, 1, 0.1], [0, 0, 1]])  # shear
        aff = np.eye(4)
        aff[:3, :3] = m
        aff[:3, 3] = rng.uniform(-100, 100, 3)
        out.append(aff)
    cases = []
    for aff in out:
        img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), aff)
        canon = nib.as_closest_canonical(img)
        cases.append(
            {
                "shape": list(shape),
                "affine": aff.tolist(),
                "ornt": nib.orientations.io_orientation(aff).astype(int).tolist(),
                "canonical_shape": list(canon.shape),
                "canonical_affine": np.asarray(canon.affine).tolist(),
            }
        )
    return cases


def resize_cases(rng: np.random.Generator) -> list[dict[str, object]]:
    specs = [
        ((9, 7, 5), (4, 6, 8)),
        ((5, 6, 7), (8, 9, 10)),
        ((13, 11, 17), (6, 7, 5)),
        ((4, 4, 4), (4, 4, 4)),
    ]
    cases = []
    for shape_xyz, target_dhw in specs:
        vol = rng.uniform(-1200, 1500, shape_xyz).astype(np.float32)
        hu = vol.transpose(2, 0, 1)  # rate_eval reader: (x, y, z) -> (D, H, W)
        res = reference.resize(hu, target_dhw)
        cases.append(
            {
                "shape_xyz": list(shape_xyz),
                "data_xyz_fortran": vol.ravel(order="F").tolist(),
                "target_dhw": list(target_dhw),
                "expected_dhw": res.ravel(order="C").tolist(),
            }
        )
    return cases


def window_cases() -> dict[str, object]:
    hu = np.concatenate(
        [np.linspace(-1500, 1600, 311), [-1350, -600, -150, -125, 0, 50, 225, 250, 1300]]
    ).astype(np.float32)
    x = hu.reshape(len(hu), 1, 1)
    out = reference.window(x)[0]  # (4, n, 1, 1)
    return {"hu": hu.tolist(), "expected": out.reshape(4, -1).tolist()}


def nifti_cases(rng: np.random.Generator, folder: Path) -> list[dict[str, object]]:
    folder.mkdir(parents=True, exist_ok=True)
    shape = (6, 5, 4)
    base = np.diag([0.8, 0.9, 2.5, 1.0])
    base[:3, 3] = [-100.0, 50.0, -300.0]
    flipped = base @ np.array(
        [[-1, 0, 0, 5], [0, 1, 0, 0], [0, 0, -1, 3], [0, 0, 0, 1]], dtype=float
    )
    oblique = np.eye(4)
    oblique[:3, :3] = _rot(2, 12) @ np.diag([0.7, 0.7, 1.25])
    specs = [
        ("int16_sform", np.int16, base, "sform", 1.0, -1024.0, False, 1),
        ("int16_qform_only", np.int16, flipped, "qform", 1.0, 0.0, False, 1),
        ("float32_both", np.float32, oblique, "both", 1.0, 0.0, False, 1),
        ("uint16_scaled", np.uint16, flipped, "sform", 0.5, -1000.0, False, 1),
        ("int16_no_codes", np.int16, base, "none", 1.0, 0.0, False, 1),
        ("int16_bigendian", np.int16, flipped, "both", 1.0, -1024.0, True, 1),
        ("float64_nifti2", np.float64, oblique, "sform", 1.0, 0.0, False, 2),
        ("uint8_gz", np.uint8, base, "sform", 4.0, -1000.0, False, 1),
    ]
    out = []
    for name, dtype, aff, codes, slope, inter, big, version in specs:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(cast(type[np.integer[Any]], dtype))
            data = rng.integers(max(info.min, -2000), min(info.max, 3000), shape).astype(dtype)
        else:
            data = rng.uniform(-1000, 1000, shape).astype(dtype)
        cls: type[nib.Nifti1Image] = nib.Nifti2Image if version == 2 else nib.Nifti1Image
        img = cls(data, aff)
        hdr = img.header
        hdr.set_data_dtype(data.dtype.newbyteorder(">") if big else data.dtype)
        img.set_qform(
            aff if codes in ("qform", "both") else None, code=1 if codes in ("qform", "both") else 0
        )
        img.set_sform(
            aff if codes in ("sform", "both") else None, code=1 if codes in ("sform", "both") else 0
        )
        hdr["scl_slope"] = slope
        hdr["scl_inter"] = inter
        if big:
            img = cls(
                data.astype(data.dtype.newbyteorder(">")), img.affine, hdr.as_byteswapped(">")
            )
        suffix = ".nii.gz" if name.endswith("_gz") else ".nii"
        path = folder / f"{name}{suffix}"
        nib.save(img, str(path))
        loaded = cast(nib.Nifti1Image, nib.load(str(path)))
        canon = nib.as_closest_canonical(loaded)
        expected = np.asarray(canon.get_fdata(), dtype=np.float32)
        out.append(
            {
                "file": path.name,
                "best_affine": np.asarray(loaded.affine).tolist(),
                "canonical_shape": list(expected.shape),
                "canonical_affine": np.asarray(canon.affine).tolist(),
                "canonical_fortran": expected.ravel(order="F").tolist(),
            }
        )
    return out


def write_web_goldens(target: Path = FIXTURES) -> None:
    rng = np.random.default_rng(7)
    target.mkdir(parents=True, exist_ok=True)
    (target / "orientation.json").write_text(json.dumps(orientation_cases(rng)))
    (target / "resize.json").write_text(json.dumps(resize_cases(rng)))
    (target / "windows.json").write_text(json.dumps(window_cases()))
    (target / "nifti.json").write_text(json.dumps(nifti_cases(rng, target / "nifti")))
