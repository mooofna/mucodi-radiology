"""Single-corpus 3D KD DataLoader (CT-RATE V2): yields (v1, v2, {teacher: feat}, {"sample_names": name})."""
from __future__ import annotations

import functools
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import set_track_meta
from monai.transforms import Compose, RandFlip, RandScaleCrop, Resize
from torch.utils.data import DataLoader, Dataset, RandomSampler

_LOG = logging.getLogger(__name__)

# skip MetaTensor wrapping (augment cost; metadata unused)
set_track_meta(False)

_RADIANS_PER_DEGREE = math.pi / 180.0


def _load_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_anisotropic_rrc_flip_augmentation(target_shape: tuple[int, int, int], flip: bool = True) -> Compose:
    """Anisotropic random-resized-crop + optional L-R mirror KD augmentation."""
    transforms = [
        RandScaleCrop(
            roi_scale=[0.95, 0.85, 0.85],
            max_roi_scale=[1.0, 1.0, 1.0],
            random_center=True,
            random_size=True,
        ),
        Resize(spatial_size=target_shape, mode="trilinear", align_corners=False),
    ]
    if flip:
        transforms.append(RandFlip(spatial_axis=1, prob=0.5))  # axis 1 = Left-Right
    return Compose(transforms)


class KDVolumeDataset(Dataset):
    """Per-sample two augmented views + frozen per-teacher features for one corpus."""

    def __init__(
        self,
        corpus_name: str,
        jsonl_path: str | Path,
        cache_template: str,
        teacher_names: list[str],
        *,
        pre_resampled_root: str | Path | None = None,
        target_shape: tuple[int, int, int] = (256, 256, 256),
        pre_crop_shape: tuple[int, int, int] | None = (288, 288, 288),
        cache_split_subdir: str = "train",
        aug_strength: str = "anisotropic_rrc_flip",
    ) -> None:
        self.corpus_name = corpus_name
        self.jsonl_path = Path(jsonl_path)
        self.teacher_names = list(teacher_names)
        self.target_shape = tuple(target_shape)
        # None => skip pre-resample; RRC samples native resolution
        self.pre_crop_shape = tuple(pre_crop_shape) if pre_crop_shape is not None else None
        self.cache_split_subdir = cache_split_subdir

        if aug_strength not in ("anisotropic_rrc", "anisotropic_rrc_flip"):
            raise ValueError(
                f"[{corpus_name}] unsupported aug_strength={aug_strength!r}; supported: "
                "'anisotropic_rrc' (B8: L-R flip dropped) | 'anisotropic_rrc_flip'"
            )
        self.augment = build_anisotropic_rrc_flip_augmentation(
            self.target_shape, flip=(aug_strength == "anisotropic_rrc_flip"))

        self.pre_resampled_root: Path | None = None
        if pre_resampled_root is not None:
            root = Path(pre_resampled_root)
            if root.is_dir():
                self.pre_resampled_root = root
            else:
                _LOG.warning(
                    "[%s] pre_resampled_root not found (%s); using on-the-fly NIfTI "
                    "loading (build the cache via dataprep/build_npy_resampled_cache.py "
                    "for full throughput).",
                    corpus_name, root,
                )

        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"[{corpus_name}] JSONL not found: {self.jsonl_path}")
        self.entries = _load_jsonl(self.jsonl_path)
        if not self.entries:
            raise RuntimeError(f"[{corpus_name}] empty JSONL: {self.jsonl_path}")

        self.teacher_cache_subdirs: dict[str, Path] = {}
        for teacher in self.teacher_names:
            subdir = Path(cache_template.format(teacher=teacher)) / self.cache_split_subdir
            if not subdir.is_dir():
                raise FileNotFoundError(
                    f"[{corpus_name}] teacher cache subdir not found for {teacher!r}: {subdir}"
                )
            self.teacher_cache_subdirs[teacher] = subdir

        self.entries = self._filter_to_cached(self.entries)

    def _filter_to_cached(self, entries: list[dict]) -> list[dict]:
        """Drop samples missing a cached feature for any teacher."""
        if os.environ.get("KDSWEEP_SKIP_CACHE_FILTER") == "1":
            _LOG.info(
                "[%s] KDSWEEP_SKIP_CACHE_FILTER=1 -> trusting JSONL; skipped per-sample "
                "cache stat (%d samples).", self.corpus_name, len(entries),
            )
            return list(entries)

        kept = [
            e for e in entries
            if all((self.teacher_cache_subdirs[t] / f"{e['sample_name']}.npz").exists()
                   for t in self.teacher_names)
        ]
        if not kept:
            raise RuntimeError(
                f"[{self.corpus_name}] no entries with cached features for all teachers "
                f"{self.teacher_names} under "
                f"{[str(p) for p in self.teacher_cache_subdirs.values()]}"
            )
        if len(kept) < len(entries):
            _LOG.info(
                "[%s] dropped %d / %d entries missing teacher features; using %d samples.",
                self.corpus_name, len(entries) - len(kept), len(entries), len(kept),
            )
        return kept

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _resample(vol_1ch: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
        x = vol_1ch.unsqueeze(0)
        x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
        return x.squeeze(0)

    def _load_teacher_feature(self, teacher: str, sample_name: str) -> torch.Tensor:
        feat_path = self.teacher_cache_subdirs[teacher] / f"{sample_name}.npz"
        with np.load(feat_path) as f_npz:
            arr = f_npz[list(f_npz.files)[0]]
        return torch.from_numpy(np.squeeze(arr).astype(np.float32))

    def _load_volume(self, entry: dict) -> torch.Tensor:
        """Return a (1, D, H, W) float32 HU tensor (cache fast path or NIfTI fallback)."""
        if self.pre_resampled_root is not None:
            cache_path = self.pre_resampled_root / f"{entry['sample_name']}.npy"
            if cache_path.is_file():
                arr = np.load(cache_path)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                return torch.from_numpy(arr).unsqueeze(0)
        # repair degenerate affines that would SVD-fail in a worker
        from rate_eval.datasets.affine_repair import safe_canonical
        path = Path(os.path.expandvars(entry["nii_path"]))
        nii = safe_canonical(str(path))
        data = nii.get_fdata().astype(np.float32)
        data = np.transpose(data, (2, 0, 1))  # -> (D, H, W)
        return torch.from_numpy(data).unsqueeze(0)

    def __getitem__(self, idx: int):
        # substitute a deterministically-offset sample on read failure; give up after 8
        n = len(self.entries)
        last_err: Exception | None = None
        for attempt in range(8):
            j = idx if attempt == 0 else (idx + attempt * 7919) % n
            try:
                return self._get_one(j)
            except Exception as e:  # noqa: BLE001
                last_err = e
                _LOG.warning(
                    "[%s] sample %r (idx %d) failed to load (%s: %s); substituting another sample",
                    self.corpus_name, self.entries[j].get("sample_name"), j, type(e).__name__, e,
                )
        raise RuntimeError(
            f"[{self.corpus_name}] 8 consecutive sample loads failed near idx {idx}; "
            f"last error: {type(last_err).__name__}: {last_err}"
        )

    def _get_one(self, idx: int):
        entry = self.entries[idx]
        sample_name = entry["sample_name"]
        vol_pre = self._load_volume(entry)
        if self.pre_crop_shape is not None and tuple(vol_pre.shape[-3:]) != self.pre_crop_shape:
            vol_pre = self._resample(vol_pre, self.pre_crop_shape)
        v1 = self.augment(vol_pre)
        v2 = self.augment(vol_pre)
        if not isinstance(v1, torch.Tensor):
            v1 = torch.as_tensor(v1)
        if not isinstance(v2, torch.Tensor):
            v2 = torch.as_tensor(v2)
        # clone strips residual MetaTensor before DataLoader IPC
        v1 = v1.detach().to(dtype=torch.float32, copy=False).clone()
        v2 = v2.detach().to(dtype=torch.float32, copy=False).clone()
        feats = {t: self._load_teacher_feature(t, sample_name) for t in self.teacher_names}
        return v1, v2, feats, {"sample_names": sample_name}


def collate_3d(batch):
    """Stack a batch into (V1, V2, {teacher: [B, d_t]}, {"sample_names": names}) per the engine contract."""
    v1s, v2s = [], []
    feats: dict[str, list[torch.Tensor]] = {}
    sample_names: list[str] = []
    for v1, v2, feat_dict, meta in batch:
        v1s.append(v1)
        v2s.append(v2)
        sample_names.append(meta["sample_names"])
        for name, val in feat_dict.items():
            feats.setdefault(name, []).append(val)
    return (
        torch.stack(v1s),
        torch.stack(v2s),
        {name: torch.stack(vals) for name, vals in feats.items()},
        {"sample_names": sample_names},
    )


def _kd_worker_init(worker_id, seed, rank):
    # per-(seed,rank,worker) seed; else forked workers share RNG and augment identically
    s = (int(seed) * 100003 + int(rank) * 997 + int(worker_id)) & 0x7FFFFFFF
    try:
        from monai.utils import set_determinism
        set_determinism(seed=s)
    except Exception:
        import numpy as _np, random as _r
        _np.random.seed(s); _r.seed(s); torch.manual_seed(s)


def get_volume_loader(args, split: str = "train") -> DataLoader:
    """Build the single-corpus KD DataLoader from a CT-RATE V2 KD profile."""
    from dataprep.config import load_profile

    profile = load_profile(args.dataset)
    teacher_names = list(profile.get("teacher_dims", {}).keys())
    if not teacher_names:
        raise ValueError(f"Profile {args.dataset!r} has no teachers")
    target_shape = tuple(profile.get("target_shape", [256, 256, 256]))
    _pcs = profile.get("pre_crop_shape", [288, 288, 288])
    pre_crop_shape = tuple(_pcs) if _pcs is not None else None  # None => native RRC
    aug_strength = profile.get("aug_strength", "anisotropic_rrc_flip")

    corpora = profile.get("corpora")
    if not corpora:
        raise ValueError(f"Profile {args.dataset!r} missing required 'corpora' field")
    if len(corpora) != 1:
        raise ValueError(
            f"Profile {args.dataset!r} has {len(corpora)} corpora; this is the "
            "single-corpus loader (CT-RATE V2). Use exactly one corpus."
        )
    corpus_name, spec = next(iter(corpora.items()))

    reader = spec.get("reader", "npy_mmap")
    if reader not in ("npy_mmap", "nifti"):
        raise NotImplementedError(
            f"[{corpus_name}] reader={reader!r} not supported; expected 'npy_mmap' or 'nifti'."
        )

    project_root = Path(__file__).resolve().parents[1]

    def _resolve(value: str | None) -> str | None:
        if value is None:
            return None
        p = Path(value)
        return str(p if p.is_absolute() else project_root / p)

    pre_resampled_root = _resolve(spec.get("pre_resampled_root")) if reader == "npy_mmap" else None

    dataset = KDVolumeDataset(
        corpus_name=corpus_name,
        jsonl_path=_resolve(spec["jsonl"]),
        cache_template=spec["cache_template"],
        teacher_names=teacher_names,
        pre_resampled_root=pre_resampled_root,
        target_shape=target_shape,
        pre_crop_shape=pre_crop_shape,
        cache_split_subdir=spec.get("cache_split_subdir", "train"),
        aug_strength=aug_strength,
    )
    if getattr(args, "rank", 0) == 0:
        _LOG.info(
            "[%s] single-corpus KD loader: %d samples, teachers=%s, target=%s",
            corpus_name, len(dataset), teacher_names, target_shape,
        )

    # each rank draws independently with replacement; no DistributedSampler
    n_samples = max(1, int(getattr(args, "samples_per_epoch", len(dataset))))
    _seed = int(getattr(args, "seed", 0) or 0)
    _rank = int(getattr(args, "rank", 0) or 0)
    # per-rank seed => ranks see different samples
    _samp_gen = torch.Generator().manual_seed(_seed + _rank)
    sampler = RandomSampler(dataset, replacement=True, num_samples=n_samples, generator=_samp_gen)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
        collate_fn=collate_3d,
        drop_last=(split == "train"),
        # partial keeps worker_init_fn picklable under spawn
        worker_init_fn=functools.partial(_kd_worker_init, seed=_seed, rank=_rank),
    )
