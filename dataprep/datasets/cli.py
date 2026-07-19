"""python -m dataprep.datasets -- stage / label / verify one dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from dataprep.datasets import engine
from dataprep.datasets.registry import REGISTRY, get

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fwd_kwargs(fn, args, names: tuple[str, ...]) -> dict:
    """Forward only the CLI options fn accepts."""
    import inspect

    params = inspect.signature(fn).parameters
    return {k: getattr(args, k) for k in names
            if getattr(args, k) is not None and k in params}


def _verify(spec) -> int:
    """Check the dataset's committed outputs exist (no download)."""
    missing = [rel for rel in spec.committed_outputs if not (_REPO_ROOT / rel).is_file()]
    if missing:
        print(f"[{spec.name}] MISSING {len(missing)}/{len(spec.committed_outputs)} committed outputs:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print(f"[{spec.name}] OK -- {len(spec.committed_outputs)} committed outputs present")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dataprep.datasets", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="dataset name (omit with --list)")
    ap.add_argument("--list", action="store_true", help="list registered datasets")
    ap.add_argument("--stage", action="store_true", help="download + convert raw imaging")
    ap.add_argument("--label", action="store_true", help="(re)build committed labels + manifests")
    ap.add_argument("--all", action="store_true", help="stage then label")
    ap.add_argument("--verify", action="store_true", help="check committed outputs exist")
    ap.add_argument("--dest", type=Path, default=None,
                    help="stage destination (default $DATA_ROOT/radiology/<name>)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/evaluation"),
                    help="label output dir (default data/evaluation)")
    ap.add_argument("--scope", choices=["all", "committed"], default=None,
                    help="stager scope where supported (lidc: all ~1010 vs committed 660)")
    ap.add_argument("--patients", nargs="+", default=None,
                    help="stage only these IDs (tiny verify sample; mutually exclusive with --scope)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of items staged/labelled (verify sample)")
    args = ap.parse_args(argv)

    if args.list or not args.name:
        for name in sorted(REGISTRY):
            print(f"  {REGISTRY[name]}")
        return 0

    try:
        spec = get(args.name)
    except KeyError as e:
        raise SystemExit(str(e))

    if args.verify:
        return _verify(spec)
    if args.stage or args.all:
        if spec.stage is None:
            raise SystemExit(f"[{spec.name}] has no programmatic stager "
                             "(download manually)")
        if args.scope and args.patients:
            raise SystemExit("--scope and --patients are mutually exclusive")
        spec.stage(args.dest or engine.expandvars(f"$DATA_ROOT/radiology/{spec.name}"),
                   **_fwd_kwargs(spec.stage, args, ("scope", "patients", "limit")))
    if args.label or args.all:
        if spec.build_labels is None:
            raise SystemExit(f"[{spec.name}] has no label builder")
        spec.build_labels(args.out_dir, **_fwd_kwargs(spec.build_labels, args, ("limit",)))
    if not (args.stage or args.label or args.all):
        print(f"{spec}\n  {len(spec.committed_outputs)} committed outputs "
              "(use --stage / --label / --all / --verify)")
    return 0
