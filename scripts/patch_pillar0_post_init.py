"""Idempotently append Pillar-0's missing self.post_init() to its cached HF custom code."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_modeling_files(hf_home: Path) -> list[Path]:
    base = hf_home / "modules" / "transformers_modules"
    if not base.is_dir():
        return []
    # YalaLab/Pillar0-ChestCT cached as .../YalaLab/Pillar0*<...>/<sha>/modeling_clip_mmatlas.py
    return sorted(base.glob("**/Pillar0*/**/modeling_clip_mmatlas.py"))


def _patch_one(path: Path) -> str:
    src = path.read_text()
    if "self.post_init()" in src:
        return "already-patched"

    lines = src.splitlines()
    # locate the class
    cls_i = next((i for i, l in enumerate(lines) if l.lstrip().startswith("class CLIPMultimodalAtlas")), None)
    if cls_i is None:
        return "skip(no CLIPMultimodalAtlas class)"
    cls_indent = len(lines[cls_i]) - len(lines[cls_i].lstrip())

    # find its __init__
    init_i = None
    for i in range(cls_i + 1, len(lines)):
        stripped = lines[i].lstrip()
        indent = len(lines[i]) - len(stripped)
        if stripped and indent <= cls_indent and not stripped.startswith(("#", '"', "'")):
            break
        if stripped.startswith("def __init__"):
            init_i = i
            break
    if init_i is None:
        return "skip(no __init__)"
    method_indent = len(lines[init_i]) - len(lines[init_i].lstrip())
    body_indent = method_indent + 4

    # end of __init__ body
    end_i = len(lines)
    for i in range(init_i + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent <= method_indent:
            end_i = i
            break

    # insert self.post_init() as the last statement of __init__
    insert_at = end_i
    while insert_at > init_i + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, " " * body_indent + "self.post_init()")
    path.write_text("\n".join(lines) + "\n")
    return "patched"


def main() -> int:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("ERROR: HF_HOME is not set (source jobs/env.sh first).", file=sys.stderr)
        return 2
    files = _find_modeling_files(Path(hf_home))
    if not files:
        print(f"No Pillar-0 modeling_clip_mmatlas.py under {hf_home}/modules/transformers_modules/ -- "
              "download the Pillar-0 weights first (the model is fetched on first use), then re-run.")
        return 1
    rc = 0
    for f in files:
        status = _patch_one(f)
        print(f"[{status}] {f}")
        if status == "patched":
            pyc = f.parent / "__pycache__"
            for p in pyc.glob("modeling_clip_mmatlas.*.pyc"):
                p.unlink()
                print(f"  wiped {p}")
        elif status.startswith("skip"):
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
