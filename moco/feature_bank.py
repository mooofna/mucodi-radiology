"""Per-teacher full-corpus frozen-teacher feature bank for contrastive KD."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_DTYPES = {"float16": torch.float16, "float32": torch.float32}

# split stem -> cache subdir (mirrors dataprep/loader.py)
_SPLIT_TO_CACHE_SUBDIR = {
    "train": "train",
    "dev": "valid",
    "valid": "valid",
    "test": "test",
}


def _load_npz_vector(path: Path) -> np.ndarray:
    """Read one cached teacher embedding as a 1-D float32 vector."""
    with np.load(path) as f_npz:
        arr = f_npz[list(f_npz.files)[0]]
    return np.squeeze(arr).astype(np.float32)


def _default_study_key(name: str) -> str:
    """Group key for false-negative ``study`` masking (first two ``_`` tokens)."""
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] in ("train", "valid", "test"):
        return "_".join(parts[:2])
    return name


def _teacher_cache_dirs(profile: dict, teacher: str, project_root: Path,
                        split: str = "train") -> list[Path]:
    """Resolve the on-disk embedding dir(s) for one teacher, mirroring the loaders."""
    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else project_root / path

    dirs: list[Path] = []
    if "corpora" in profile:
        for spec in profile["corpora"].values():
            subdir = spec.get("cache_split_subdir", "train")
            dirs.append(_abs(spec["cache_template"].format(teacher=teacher)) / subdir)
    else:
        cache_root = _abs(profile["cache_root"])
        dirs.append(cache_root / _SPLIT_TO_CACHE_SUBDIR.get(split, "train"))
    return dirs


class FrozenTeacherBank:
    """Per-teacher full-corpus bank of L2-normalized frozen teacher keys."""

    def __init__(self, rows: dict[str, torch.Tensor], name_to_row: dict[str, int],
                 study_codes: torch.Tensor) -> None:
        self.rows = rows
        self.name_to_row = name_to_row
        self.study_codes = study_codes
        any_t = next(iter(rows.values()))
        self.device = any_t.device
        self.N = any_t.size(0)

    def indices_for(self, names: list[str]) -> torch.Tensor:
        """``[B]`` long tensor of bank rows for the given sample names."""
        return torch.tensor([self.name_to_row[n] for n in names],
                            device=self.device, dtype=torch.long)

    def sample_indices(self, m: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """``[M]`` random bank rows (with replacement) -- the shared negatives."""
        return torch.randint(0, self.N, (m,), device=self.device, generator=generator)

    def gather(self, teacher: str, idx: torch.Tensor) -> torch.Tensor:
        """Rows ``idx`` of ``teacher`` -> ``[len(idx), d_t]`` (already normalized)."""
        return self.rows[teacher].index_select(0, idx)

    def false_neg_bias(self, pos_idx: torch.Tensor, neg_idx: torch.Tensor,
                       mode: str, out_dtype: torch.dtype) -> torch.Tensor | None:
        """Additive ``[B, B+M]`` logit bias masking false-negative columns."""
        if mode == "off":
            return None
        B, M = pos_idx.size(0), neg_idx.size(0)
        if mode == "self":
            key_pos, key_neg = pos_idx, neg_idx
        elif mode == "study":
            key_pos = self.study_codes[pos_idx]
            key_neg = self.study_codes[neg_idx]
        else:
            raise ValueError(f"unknown false-negative mask mode: {mode!r}")
        neg_coll = key_pos.unsqueeze(1) == key_neg.unsqueeze(0)            # [B, M]
        neg_bias = torch.zeros(B, M, device=self.device, dtype=out_dtype).masked_fill(
            neg_coll, float("-inf"))
        # mask same-key collisions in the positive block, never the diagonal
        pos_coll = key_pos.unsqueeze(1) == key_pos.unsqueeze(0)           # [B, B]
        pos_coll.fill_diagonal_(False)
        pos_bias = torch.zeros(B, B, device=self.device, dtype=out_dtype).masked_fill(
            pos_coll, float("-inf"))
        return torch.cat([pos_bias, neg_bias], dim=1)                     # [B, B+M]


def _finalize(per_teacher_vecs: dict[str, dict[str, np.ndarray]],
              dtype: torch.dtype, device: str | int) -> FrozenTeacherBank:
    """Intersect names across teachers, stack + L2-normalize, build the bank."""
    teachers = list(per_teacher_vecs.keys())
    common = set(per_teacher_vecs[teachers[0]])
    for t in teachers[1:]:
        common &= set(per_teacher_vecs[t])
    if not common:
        raise RuntimeError(
            f"feature bank: no sample names common to all teachers {teachers}")
    names = sorted(common)
    name_to_row = {n: i for i, n in enumerate(names)}

    rows: dict[str, torch.Tensor] = {}
    for t in teachers:
        mat = np.stack([per_teacher_vecs[t][n] for n in names], axis=0)
        x = torch.from_numpy(mat)
        x = F.normalize(x, dim=-1)
        rows[t] = x.to(dtype=dtype, device=device, copy=False)

    study_strs = [_default_study_key(n) for n in names]
    uniq = {k: i for i, k in enumerate(dict.fromkeys(study_strs))}
    study_codes = torch.tensor([uniq[k] for k in study_strs],
                               device=device, dtype=torch.long)
    return FrozenTeacherBank(rows, name_to_row, study_codes)


def build_frozen_bank(profile: dict, teacher_dims: dict[str, int],
                      dtype: str = "float16", device: str | int = "cuda",
                      rank: int = 0, project_root: Path | None = None,
                      split: str = "train",
                      expected_n: int | None = None) -> FrozenTeacherBank:
    """Load the full-corpus frozen-teacher bank from the on-disk caches."""
    torch_dtype = _DTYPES[dtype]
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    t0 = time.time()

    per_teacher_vecs: dict[str, dict[str, np.ndarray]] = {}
    for teacher in teacher_dims:
        vecs: dict[str, np.ndarray] = {}
        for cache_dir in _teacher_cache_dirs(profile, teacher, project_root, split):
            if not cache_dir.is_dir():
                raise FileNotFoundError(
                    f"feature bank: teacher cache dir not found for {teacher!r}: {cache_dir}")
            for npz_path in sorted(cache_dir.glob("*.npz")):
                try:
                    vecs[npz_path.stem] = _load_npz_vector(npz_path)
                except Exception as e:
                    if rank == 0:
                        print(f"[feature_bank] WARN: skipping {npz_path} ({type(e).__name__}: {e})")
        if not vecs:
            raise RuntimeError(f"feature bank: no .npz embeddings found for teacher {teacher!r}")
        per_teacher_vecs[teacher] = vecs

    # integrity gate: a silently-undersized/mismatched bank is invisible downstream
    per_teacher_counts = {t: len(v) for t, v in per_teacher_vecs.items()}
    n_common = len(set.intersection(*(set(v) for v in per_teacher_vecs.values())))
    if rank == 0:
        dropped = {t: c - n_common for t, c in per_teacher_counts.items()}
        print(f"[feature_bank] integrity: per-teacher counts={per_teacher_counts}, "
              f"|common|={n_common}, dropped_by_intersection={dropped}")
        if len(set(per_teacher_counts.values())) != 1:
            print(f"[feature_bank] WARN: per-teacher npz counts differ {per_teacher_counts} -- "
                  f"incomplete/mismatched extraction; only the {n_common} common samples are used.")
    if expected_n is not None and n_common != expected_n:
        raise RuntimeError(
            f"feature_bank: expected {expected_n} common samples across teachers "
            f"{list(per_teacher_vecs)}, got {n_common} (per-teacher {per_teacher_counts}). "
            f"Refusing to train on a silently-undersized/mismatched bank.")

    bank = _finalize(per_teacher_vecs, torch_dtype, device)
    if rank == 0:
        n_bytes = sum(r.numel() * r.element_size() for r in bank.rows.values())
        print(
            f"[feature_bank] built bank: N={bank.N} samples, teachers={list(teacher_dims)}, "
            f"dtype={dtype}, {n_bytes / 1e6:.0f} MB on device, "
            f"n_studies={int(bank.study_codes.max().item()) + 1}, {time.time() - t0:.1f}s")
    return bank
