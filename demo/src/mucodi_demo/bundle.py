"""Release bundles of the built demo assets, for GitHub Pages and CI.

The site serves everything under ``assets/``. Two archives go to the ``demo-assets`` release:
``web-assets-core.tar.gz`` holds what the tests need (the t-line and B0 models, the scans, the
evidence and the figures); ``web-assets-large.tar`` holds the B2 to B6 models, which only the
deployed site serves. Model weights are float32 and barely compress, so the large archive is not
gzipped. Archives are deterministic: sorted entries, zeroed owners and timestamps.
"""

from __future__ import annotations

import gzip
import hashlib
import tarfile
from pathlib import Path

LARGE_RUNGS = ("b2", "b3", "b4", "b5", "b6")
CORE = "web-assets-core.tar.gz"
LARGE = "web-assets-large.tar"


def is_large(relative: Path) -> bool:
    """True for the model files of the B2 to B6 students (``models/b5.core.onnx.data0`` etc.)."""
    return relative.parts[0] == "models" and relative.name.split(".")[0] in LARGE_RUNGS


def _write_tar(files: list[tuple[Path, str]], path: Path, compress: bool) -> None:
    raw = path.open("wb")
    # mtime=0 keeps the gzip header free of the build time.
    stream = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) if compress else raw
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for source, name in files:
            info = tarfile.TarInfo(name)
            info.size = source.stat().st_size
            info.mode = 0o644
            with source.open("rb") as handle:
                tar.addfile(info, handle)
    if compress:
        stream.close()
    raw.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bundles(assets: Path, out_dir: Path) -> list[Path]:
    """Write the two archives and a ``SHA256SUMS`` file; return the paths written."""
    files = sorted(p for p in assets.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no assets under {assets}")
    entries = [(p, p.relative_to(assets).as_posix()) for p in files]
    out_dir.mkdir(parents=True, exist_ok=True)
    core = [(p, n) for p, n in entries if not is_large(Path(n))]
    large = [(p, n) for p, n in entries if is_large(Path(n))]
    outputs = [out_dir / CORE, out_dir / LARGE]
    _write_tar(core, outputs[0], compress=True)
    _write_tar(large, outputs[1], compress=False)
    sums = "".join(f"{_sha256(p)}  {p.name}\n" for p in outputs)
    (out_dir / "SHA256SUMS").write_text(sums)
    return [*outputs, out_dir / "SHA256SUMS"]
