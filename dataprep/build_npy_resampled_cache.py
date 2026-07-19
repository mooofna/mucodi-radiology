"""Build a pre-resampled `.npy` cache for any NIfTI-backed corpus."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
from rate_eval.datasets.affine_repair import safe_canonical
import numpy as np
import torch
import torch.nn.functional as F
import yaml


DEFAULT_TARGET_SHAPE = (160, 320, 320)


def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _resample_one(
    sample_name: str,
    nii_path: Path,
    dst_path: Path,
    target_shape: tuple[int, int, int],
    out_dtype: str,
) -> tuple[str, str]:
    """Resample one volume to `.npy`; returns (sample_name, status) with status ok/skip/err:<reason>."""
    if dst_path.is_file():
        return sample_name, "skip"
    if not nii_path.is_file():
        return sample_name, f"err:MissingNii:{nii_path}"
    try:
        # keep byte-for-byte in sync with loader _load_volume/_resample
        nii = safe_canonical(str(nii_path))
        data = nii.get_fdata().astype(np.float32)
        data = np.transpose(data, (2, 0, 1))  # (H, W, D) -> (D, H, W)
        x = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
        out = x.squeeze(0).squeeze(0).numpy()
        assert out.shape == target_shape, f"unexpected shape {out.shape} for {sample_name}"
        # HU range [-1024, +3071] fits in fp16
        if out_dtype == "float16":
            out = out.astype(np.float16)
        elif out_dtype == "float32":
            out = out.astype(np.float32)
        else:
            raise ValueError(f"unsupported dtype {out_dtype!r}")
        tmp = dst_path.parent / (dst_path.stem + ".tmp.npy")
        np.save(tmp, out)
        os.replace(tmp, dst_path)
        return sample_name, "ok"
    except Exception as e:
        return sample_name, f"err:{type(e).__name__}:{e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--corpus",
        required=True,
        choices=["ctrate_v2", "lidc", "coca"],
        help="Corpus name (drives cache_meta provenance)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Path to the corpus's train.jsonl manifest (sample_name + nii_path per row)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination cache directory (will be created)",
    )
    parser.add_argument(
        "--target-shape",
        nargs=3,
        type=int,
        default=list(DEFAULT_TARGET_SHAPE),
        help="Resample target shape (default: 160 320 320)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Output dtype (default: float16). fp16 halves disk; HU range fits.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Process workers for parallel resampling (default: 24)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of volumes (for dev / smoke testing)",
    )
    args = parser.parse_args()

    target_shape = tuple(args.target_shape)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build-cache] corpus: {args.corpus}")
    print(f"[build-cache] jsonl:  {args.jsonl}")
    print(f"[build-cache] output dir: {out_dir}")
    print(f"[build-cache] target shape: {target_shape}")
    print(f"[build-cache] dtype: {args.dtype}")
    print(f"[build-cache] workers: {args.workers}")

    entries = _load_jsonl(args.jsonl)
    manifest: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for entry in entries:
        name = entry["sample_name"]
        nii = Path(os.path.expandvars(entry["nii_path"]))
        if name in seen:
            continue
        seen.add(name)
        manifest.append((name, nii))

    if args.limit:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"[build-cache] total unique volumes in manifest: {total}")

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    errors: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _resample_one,
                name,
                src,
                out_dir / f"{name}.npy",
                target_shape,
                args.dtype,
            ): name
            for name, src in manifest
        }
        for i, fut in enumerate(as_completed(futures), 1):
            name, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                errors.append((name, status))
            if i % 100 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (total - i) / rate if rate > 0 else 0.0
                print(
                    f"  [{i}/{total}] ok={n_ok} skip={n_skip} err={n_err} "
                    f"rate={rate:.1f}/s eta={eta/60:.1f}min"
                )

    elapsed = time.time() - t0
    print(f"[build-cache] done: ok={n_ok} skip={n_skip} err={n_err} in {elapsed/60:.1f} min")
    if errors:
        print(f"[build-cache] first {min(10, len(errors))} errors:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    meta = {
        "schema_version": "1.0",
        "kind": f"{args.corpus}_resampled_npy",
        "target_shape": list(target_shape),
        "interpolation": "trilinear",
        "align_corners": False,
        "dtype": args.dtype,
        "source": {
            "jsonl": str(args.jsonl),
            "corpus": args.corpus,
        },
        "counts": {
            "total_in_manifest": total,
            "written_or_skipped": n_ok + n_skip,
            "ok_this_run": n_ok,
            "skip_pre_existing": n_skip,
            "errors": n_err,
        },
        "build": {
            "datetime": dt.datetime.now(dt.timezone.utc).isoformat(),
            "workers": args.workers,
            "elapsed_sec": int(elapsed),
            "host": os.uname().nodename,
        },
    }
    meta_path = out_dir / "cache_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"[build-cache] provenance: {meta_path}")

    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
