"""Command line entry point: ``uv run mucodi-demo <command>``."""

from __future__ import annotations

import typer

from mucodi_demo.config import load_paths
from mucodi_demo.students import RUNGS

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Build the assets of the MuCoDi in-browser chest CT demo."""


@app.command()
def models(rungs: str = typer.Option(",".join(RUNGS), help="Comma-separated rungs")) -> None:
    """Export the student backbones to ONNX (core + taps) with parity checks and models.json."""
    from mucodi_demo.models_build import build_models

    paths = load_paths()
    build_models(
        paths.checkpoints, paths.assets / "models", tuple(r for r in rungs.split(",") if r)
    )


if __name__ == "__main__":
    app()


@app.command()
def cases() -> None:
    """Download, convert and tier the bundled demo scans (cases.json)."""
    from mucodi_demo.cases import build_cases

    paths = load_paths()
    for info in build_cases(paths.downloads, paths.assets / "cases"):
        print(info["id"], info["native_shape_xyz"], info["spacing_mm"], info["files"])


@app.command()
def goldens() -> None:
    """PyTorch golden embeddings for every bundled case and rung (goldens.json)."""
    from mucodi_demo.goldens import build_goldens

    paths = load_paths()
    build_goldens(paths.checkpoints, paths.assets / "cases", paths.downloads)


@app.command("web-goldens")
def web_goldens() -> None:
    """Golden vectors for the browser preprocessing tests (web/src/lib/volume/__fixtures__)."""
    from mucodi_demo.webgoldens import write_web_goldens

    write_web_goldens()


@app.command()
def evidence() -> None:
    """Evidence JSON for the web page from the thesis artifacts (with provenance checks)."""
    from mucodi_demo.evidence import build_evidence

    paths = load_paths()
    build_evidence(paths, paths.assets / "evidence")
