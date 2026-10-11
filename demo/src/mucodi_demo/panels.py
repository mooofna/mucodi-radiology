"""Panels and the MuCoDi overview figure for the method page, from a redistributable scan.

The thesis figure (defense/tikz/mucodi_overview_standalone.tex) embeds panels rendered from a
CT-RATE volume, which may not be republished. This module renders the same eight panels, with
the code path of the thesis's kd_student_sweep/analysis/visualize_data_pipeline.py (cache
resize, anisotropic random-resized crop, four windows), from the bundled LIDC-IDRI case and
compiles the unchanged TikZ source against them.
"""

from __future__ import annotations

import base64
import io
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Rectangle
from PIL import Image

from mucodi_demo import reference

CACHE = (160, 320, 320)
TARGET = reference.TARGET_SHAPE
HU_DISPLAY = (-1000.0, 400.0)
V1_COLOR, V2_COLOR = "#EE7733", "#4477AA"
SEED = 0
TIKZ_FILES = (
    "mucodi_overview_standalone.tex",
    "mucodi-defs-full.tex",
    "mucodi-styles.tex",
    "pipeline-full.tex",
    "hsphere-deck.tex",
)


def _crop(
    rng: np.random.Generator, vol: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], tuple[slice, slice, slice]]:
    """Anisotropic random-resized crop: axial scale in [0.95, 1], in-plane in [0.85, 1]."""
    sizes = []
    for n, lo in zip(vol.shape, (0.95, 0.85, 0.85), strict=True):
        sizes.append(max(1, int(round(n * rng.uniform(lo, 1.0)))))  # noqa: RUF046, as in the thesis
    slices = []
    for n, s in zip(vol.shape, sizes, strict=True):
        start = int(rng.integers(0, n - s + 1))
        slices.append(slice(start, start + s))
    sd, sh, sw = slices
    return reference.resize(vol[sd, sh, sw], TARGET), (sd, sh, sw)


def _export(
    arr: npt.NDArray[np.float32],
    path: Path,
    vmin: float,
    vmax: float,
    boxes: list[tuple[slice, slice, str]] | None = None,
) -> None:
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_axis_off()
    for sh, sw, col in boxes or []:
        ax.add_patch(
            Rectangle(
                (sw.start, sh.start),
                sw.stop - sw.start,
                sh.stop - sh.start,
                fill=False,
                edgecolor=col,
                lw=3.0,
            )
        )
    fig.savefig(path)
    plt.close(fig)


def render_panels(native: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    hu = reference.read_hu(native)
    cache = reference.resize(hu, CACHE)
    rng = np.random.default_rng(SEED)
    v1, b1 = _crop(rng, cache)
    v2, b2 = _crop(rng, cache)
    win = reference.window(v1)[0]
    z_src, z_view = CACHE[0] // 2, TARGET[0] // 2
    lo, hi = HU_DISPLAY
    _export(cache[z_src], out / "cached_raw.png", lo, hi)
    _export(
        cache[z_src],
        out / "cached_with_boxes.png",
        lo,
        hi,
        [(b1[1], b1[2], V1_COLOR), (b2[1], b2[2], V2_COLOR)],
    )
    _export(v1[z_view], out / "view1.png", lo, hi)
    _export(v2[z_view], out / "view2.png", lo, hi)
    for ch, name in enumerate(w[0] for w in reference.WINDOWS):
        _export(win[ch, z_view], out / f"{name}.png", 0.0, 1.0)


PANELS = ("", "offbox", "pipebox", "m1box", "objbox", "m2box")


def build_figure(tikz_dir: Path, panels: Path, work: Path, out_dir: Path) -> list[Path]:
    """Compile the thesis TikZ overview with the replacement panels, plain and with each of the
    five panels (a) to (e) highlighted as in the defense tour, and convert the PDFs to SVG."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for name in TIKZ_FILES:
        shutil.copy(tikz_dir / name, work / name)
    shutil.copytree(panels, work / "mucodi_panels")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for hl in PANELS:
        job = f"overview_{hl or 'full'}"
        subprocess.run(
            [
                "lualatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-jobname={job}",
                rf"\def\panelhl{{{hl}}}\input{{mucodi_overview_standalone}}",
            ],
            cwd=work,
            check=True,
            capture_output=True,
        )
        svg = out_dir / f"mucodi_{hl or 'full'}.svg"
        subprocess.run(["pdftocairo", "-svg", str(work / f"{job}.pdf"), str(svg)], check=True)
        slim_svg(svg)
        outputs.append(svg)
    return outputs


_IMAGE = re.compile(r'<image id="([^"]+)"([^>]*?)xlink:href="data:image/png;base64,([^"]+)"\s*/>')


def slim_svg(svg: Path, max_px: int = 320) -> None:
    """Shrink the embedded panel bitmaps (900 px, shown at under 100 px on screen) and drop
    duplicates. Opaque panels become JPEG; the soft masks stay PNG. Each <image> keeps its
    user-space size, so the layout is unchanged."""
    text = svg.read_text()
    first: dict[str, str] = {}
    alias: dict[str, str] = {}

    def shrink(m: re.Match[str]) -> str:
        ident, attrs, payload = m.groups()
        if payload in first:
            alias[ident] = first[payload]
            return ""
        first[payload] = ident
        img: Image.Image = Image.open(io.BytesIO(base64.b64decode(payload)))
        if img.mode not in ("L", "LA", "RGB", "RGBA"):
            img = img.convert("RGBA")
        img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        if img.mode == "RGB":
            img.save(buf, format="JPEG", quality=88)
            mime = "jpeg"
        else:
            img.save(buf, format="PNG", optimize=True)
            mime = "png"
        data = base64.b64encode(buf.getvalue()).decode()
        return f'<image id="{ident}"{attrs}xlink:href="data:image/{mime};base64,{data}"/>'

    text = _IMAGE.sub(shrink, text)
    for dup, keep in alias.items():
        text = text.replace(f'"#{dup}"', f'"#{keep}"')
    svg.write_text(text)
