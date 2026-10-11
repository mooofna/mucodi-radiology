"""Release archives: the split between core and large files, and byte-for-byte reproducibility."""

from __future__ import annotations

import tarfile
from pathlib import Path

from mucodi_demo.bundle import CORE, LARGE, build_bundles


def make_assets(root: Path) -> None:
    for name in (
        "models/models.json",
        "models/tmin.core.onnx",
        "models/b0.taps.onnx",
        "models/b2.core.onnx",
        "models/b6.core.onnx.data1",
        "cases/lidc-0001.grid.nii.gz",
        "evidence/curves/lidc.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode() * 3)


def names(archive: Path) -> list[str]:
    with tarfile.open(archive) as tar:
        return tar.getnames()


def test_core_has_everything_but_the_large_models(tmp_path: Path) -> None:
    make_assets(tmp_path / "assets")
    build_bundles(tmp_path / "assets", tmp_path / "out")
    assert names(tmp_path / "out" / LARGE) == ["models/b2.core.onnx", "models/b6.core.onnx.data1"]
    assert names(tmp_path / "out" / CORE) == [
        "cases/lidc-0001.grid.nii.gz",
        "evidence/curves/lidc.json",
        "models/b0.taps.onnx",
        "models/models.json",
        "models/tmin.core.onnx",
    ]
    sums = (tmp_path / "out" / "SHA256SUMS").read_text().splitlines()
    assert [line.split()[1] for line in sums] == [CORE, LARGE]


def test_archives_are_reproducible(tmp_path: Path) -> None:
    make_assets(tmp_path / "assets")
    first = [p.read_bytes() for p in build_bundles(tmp_path / "assets", tmp_path / "a")]
    second = [p.read_bytes() for p in build_bundles(tmp_path / "assets", tmp_path / "b")]
    assert first == second
