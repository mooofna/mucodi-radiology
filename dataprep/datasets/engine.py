"""Shared, pure dataset-curation primitives that ``dataprep.datasets`` specs compose."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

# indent=2, no trailing newline: matches committed label files byte-for-byte
_JSON_KW = dict(indent=2)


def stable_split(key: str, *, train_frac: float = 0.70, dev_frac: float = 0.15) -> str:
    """Deterministic patient-level split via ``sha1(key)[:8] % 100``; returns train/dev/test."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 100
    if h < train_frac * 100:
        return "train"
    if h < (train_frac + dev_frac) * 100:
        return "dev"
    return "test"


def qa_key(class_name: str) -> str:
    """Per-class QA question: ``'Is <name> present? (0=No, 1=Yes)'``."""
    return f"Is {class_name.lower()} present? (0=No, 1=Yes)"


def slug(name: str) -> str:
    """Filesystem-safe slug: lowercase, runs of non-alphanumerics -> single ``_``."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(obj: Any, path: str | Path) -> None:
    """Write JSON exactly like the committed label files (indent=2, no trailing newline)."""
    Path(path).write_text(json.dumps(obj, **_JSON_KW))


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def require_env(name: str, *, hint: str = "") -> str:
    val = os.environ.get(name)
    if not val:
        suffix = f" -- {hint}" if hint else ""
        raise RuntimeError(f"required environment variable {name} is unset{suffix}")
    return val


def kaggle_dataset_download(
    slug: str,
    dest: str | Path,
    *,
    unzip: bool = True,
    competition: bool = False,
    kaggle_cli: str = "kaggle",
) -> Path:
    """Download a Kaggle dataset (or competition) into ``dest`` via the ``kaggle`` CLI."""
    if shutil.which(kaggle_cli) is None:
        raise RuntimeError(
            f"kaggle CLI not found ({kaggle_cli!r}); `uv pip install kaggle` and set "
            f"KAGGLE_API_TOKEN in jobs/secrets.env (or ~/.kaggle/access_token)"
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if competition:
        cmd = [kaggle_cli, "competitions", "download", "-c", slug, "-p", str(dest)]
    else:
        cmd = [kaggle_cli, "datasets", "download", slug, "-p", str(dest)]
    if unzip:
        cmd.append("--unzip")
    kind = "competition" if competition else "dataset"
    print(f"[kaggle] downloading {kind} {slug} -> {dest} (unzip={unzip})", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"kaggle {kind} download for {slug!r} returned {rc}")
    return dest


# idc-index pins pandas<=2.2.4: run isolated via `uv run --no-project`, not in the venv

_IDC_QUERY_MARKER = "IDC_QUERY_JSON:"

_IDC_QUERY_DRIVER = r"""
import json, os, duckdb
from idc_index import index
df = index.IDCClient().index
con = duckdb.connect(); con.register("idc_index", df)
rows = con.sql(os.environ["IDC_SQL"]).df().to_dict(orient="records")
print("IDC_QUERY_JSON:" + json.dumps(rows, default=str))
"""

_IDC_DOWNLOAD_DRIVER = r"""
import json, os
from idc_index import index
index.IDCClient().download_from_selection(
    downloadDir=os.environ["IDC_DEST"],
    seriesInstanceUID=json.loads(os.environ["IDC_UIDS"]),
    dirTemplate=os.environ["IDC_DIRTEMPLATE"], quiet=True)
"""


def _idc_isolated(driver: str, env_extra: dict, *, capture: bool, uv: str = "uv"):
    """Run ``driver`` (on stdin) in an isolated ``uv run --no-project --with idc-index`` env."""
    if shutil.which(uv) is None:
        raise RuntimeError(
            "uv not found on PATH; the isolated idc-index env needs `uv` "
            "(idc-index is deliberately NOT in the venv -- pandas pin conflict)"
        )
    proc = subprocess.run(
        [uv, "run", "--no-project", "--with", "idc-index", "python", "-"],
        input=driver, text=True, env={**os.environ, **env_extra}, capture_output=capture,
    )
    if proc.returncode != 0:
        detail = f"\n{proc.stderr[-2000:]}" if capture and proc.stderr else ""
        raise RuntimeError(f"isolated idc-index run failed (rc={proc.returncode}){detail}")
    return proc


def idc_query(sql: str, *, uv: str = "uv") -> list[dict]:
    """Read-only DuckDB query over the IDC index (SQL against the ``idc_index`` table); rows as dicts."""
    proc = _idc_isolated(_IDC_QUERY_DRIVER, {"IDC_SQL": sql}, capture=True, uv=uv)
    for line in proc.stdout.splitlines():
        if line.startswith(_IDC_QUERY_MARKER):
            return json.loads(line[len(_IDC_QUERY_MARKER):])
    raise RuntimeError(f"idc_query: no result marker in output:\n{proc.stdout[-2000:]}")


def idc_download(series_uids, dest: str | Path, *,
                 dir_template: str = "%PatientID", uv: str = "uv") -> Path:
    """Download IDC DICOM series into ``dest`` (isolated idc-index, anonymous S3 mirror)."""
    uids = list(series_uids)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[idc] downloading {len(uids)} series -> {dest} (isolated idc-index)", flush=True)
    _idc_isolated(
        _IDC_DOWNLOAD_DRIVER,
        {"IDC_UIDS": json.dumps(uids), "IDC_DEST": str(dest), "IDC_DIRTEMPLATE": dir_template},
        capture=False, uv=uv,
    )
    return dest


def expandvars(path_template: str) -> Path:
    """Expand ``$VAR`` / ``${VAR}`` in a committed manifest path for on-disk use."""
    return Path(os.path.expandvars(path_template))


def data_root_relative(path: str | Path) -> str:
    """Render ``path`` under ``$DATA_ROOT`` as a portable ``${DATA_ROOT}/...`` template; else raw str."""
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        try:
            rel = Path(path).resolve().relative_to(Path(data_root).resolve())
            return f"${{DATA_ROOT}}/{rel.as_posix()}"
        except ValueError:
            pass
    return str(path)
