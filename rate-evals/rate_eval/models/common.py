import torch
import rve
from typing import Optional, Union, List


def batch_apply_mr_windowing(
    batch: torch.Tensor,
    mr_window_type: str = "all",
    modality: str = "MR",
    normalize_mean: Optional[float] = None,
    normalize_std: Optional[float] = None,
    per_sample: bool = True,
) -> torch.Tensor:
    """Apply windowing and normalization transforms to MR volumes."""

    B, C, D, H, W = batch.shape
    device = batch.device
    if per_sample:
        if mr_window_type in ["high_contrast", "znorm"]:
            num_output_channels = C
            windowed = torch.zeros(
                (B, num_output_channels, D, H, W), device=device, dtype=batch.dtype
            )
            for b in range(B):
                for cidx in range(C):
                    sample = batch[b : b + 1, cidx : cidx + 1]
                    windowed[b, cidx] = rve.apply_windowing(sample, mr_window_type, modality)
            batch = windowed
        else:
            raise ValueError(f"Invalid window type: {mr_window_type}")
    else:
        if mr_window_type in ["high_contrast", "znorm"]:
            windowed = torch.zeros((B, C, D, H, W), device=device, dtype=batch.dtype)

            for cidx in range(C):
                windowed[:, cidx] = rve.apply_windowing(
                    batch[:, cidx : cidx + 1], mr_window_type, modality
                )
            batch = windowed
        else:
            raise ValueError(f"Invalid window type: {mr_window_type}")

    assert (normalize_mean is None) == (
        normalize_std is None
    ), "Either both or none of normalize_mean and normalize_std must be provided"
    if normalize_mean is not None or normalize_std is not None:
        normalize_mean = torch.tensor(normalize_mean, device=device)
        normalize_std = torch.tensor(normalize_std, device=device)
        batch = (batch - normalize_mean) / normalize_std

    return batch


def _vectorized_ct_windowing_all(batch: torch.Tensor, window_names=None) -> torch.Tensor:
    """Vectorized ct_window_type='all': (B,1,D,H,W) HU -> (B,len(windows),D,H,W)."""
    device = batch.device
    out_dtype = batch.dtype
    # float() for bitwise parity with apply_windowing
    volume = batch.squeeze(1).float()
    B, D, H, W = volume.shape

    if window_names is None:
        window_names = rve.get_available_windows("CT")
    anatomical = rve.ANATOMICAL_WINDOWS.get("CT", {})

    out = torch.empty((B, len(window_names), D, H, W), device=device, dtype=out_dtype)
    for i, name in enumerate(window_names):
        if name == "minmax":
            mins = volume.amin(dim=(-3, -2, -1), keepdim=True)
            maxs = volume.amax(dim=(-3, -2, -1), keepdim=True)
            windowed = (volume - mins) / (maxs - mins + 1e-8)
            windowed = torch.clamp(windowed, 0.0, 1.0)
        elif name in anatomical:
            params = anatomical[name]
            center = float(params["center"])
            width = float(params["width"])
            low = center - width / 2.0
            # match apply_anatomical_window
            windowed = (volume - low) / (width + 1e-8)
            windowed = torch.clamp(windowed, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported CT window: {name!r}")
        out[:, i] = windowed.to(out_dtype)
    return out


def batch_apply_ct_windowing(
    batch: torch.Tensor,
    ct_window_type=None,
    modality: str = "CT",
    normalize_mean: Optional[float] = None,
    normalize_std: Optional[float] = None,
    per_sample: bool = True,
) -> torch.Tensor:
    """Apply windowing and normalization transforms to CT volumes."""
    B, C, D, H, W = batch.shape
    device = batch.device

    if per_sample:
        if ct_window_type == "all" and modality == "CT" and C == 1:
            batch = _vectorized_ct_windowing_all(batch)
        elif (
            isinstance(ct_window_type, list)
            and modality == "CT"
            and C == 1
            and len(ct_window_type) > 0
            and all(isinstance(n, str) for n in ct_window_type)
        ):
            batch = _vectorized_ct_windowing_all(batch, window_names=ct_window_type)
        elif ct_window_type == "all":
            # loop fallback for non-CT or multi-channel
            windows = rve.get_available_windows(modality)
            windowed = torch.zeros((B, len(windows), D, H, W), device=device, dtype=batch.dtype)
            for b in range(B):
                sample = batch[b : b + 1]
                for i, window in enumerate(windows):
                    if C == 1:
                        windowed[b, i] = rve.apply_windowing(
                            sample.squeeze(0).squeeze(0), window, modality
                        )
                    else:
                        windowed[b, i] = rve.apply_windowing(sample[0, 0], window, modality)
            batch = windowed
        elif isinstance(ct_window_type, list):
            windowed = torch.zeros(
                (B, len(ct_window_type), D, H, W), device=device, dtype=batch.dtype
            )
            for b in range(B):
                sample = batch[b : b + 1]
                for i, window in enumerate(ct_window_type):
                    if C == 1:
                        windowed[b, i] = rve.apply_windowing(
                            sample.squeeze(0).squeeze(0), window, modality
                        )
                    else:
                        windowed[b, i] = rve.apply_windowing(sample[0, 0], window, modality)
            batch = windowed
        else:
            windowed = torch.zeros((B, 1, D, H, W), device=device, dtype=batch.dtype)
            for b in range(B):
                sample = batch[b : b + 1]
                if C == 1:
                    windowed[b, 0] = rve.apply_windowing(
                        sample.squeeze(0).squeeze(0), ct_window_type, modality
                    )
                else:
                    windowed[b, 0] = rve.apply_windowing(sample[0, 0], ct_window_type, modality)
            batch = windowed
    else:
        # batch-wise: stats span the whole batch
        if ct_window_type == "all":
            windows = rve.get_available_windows(modality)
            windowed = torch.zeros((B, len(windows), D, H, W), device=device, dtype=batch.dtype)

            for i, window in enumerate(windows):
                if C == 1:
                    windowed[:, i] = rve.apply_windowing(batch.squeeze(1), window, modality)
                else:
                    windowed[:, i] = rve.apply_windowing(batch[:, 0], window, modality)
            batch = windowed
        elif isinstance(ct_window_type, list):
            windowed = torch.zeros(
                (B, len(ct_window_type), D, H, W), device=device, dtype=batch.dtype
            )
            for i, window in enumerate(ct_window_type):
                if C == 1:
                    windowed[:, i] = rve.apply_windowing(batch.squeeze(1), window, modality)
                else:
                    windowed[:, i] = rve.apply_windowing(batch[:, 0], window, modality)
            batch = windowed
        else:
            if C == 1:
                batch = rve.apply_windowing(batch.squeeze(1), ct_window_type, modality).unsqueeze(1)
            else:
                batch = rve.apply_windowing(batch[:, 0], ct_window_type, modality).unsqueeze(1)

    assert (normalize_mean is None) == (
        normalize_std is None
    ), "Either both or none of normalize_mean and normalize_std must be provided"
    if normalize_mean is not None or normalize_std is not None:
        normalize_mean = torch.tensor(normalize_mean, device=device)
        normalize_std = torch.tensor(normalize_std, device=device)
        batch = (batch - normalize_mean) / normalize_std

    return batch


def batch_apply_normalization(
    batch: torch.Tensor,
    normalize_mean: Union[float, List[float]],
    normalize_std: Union[float, List[float]],
) -> torch.Tensor:
    """Apply normalization to a batch of volumes."""
    assert isinstance(normalize_mean, (float, list)) and isinstance(
        normalize_std, (float, list)
    ), f"normalize_mean and normalize_std must be either float or list, got {type(normalize_mean)} and {type(normalize_std)}"

    if isinstance(normalize_mean, float):
        # shared mean/std across channels
        if len(batch.shape) == 5:  # (B, C, D, H, W)
            mean = torch.tensor(normalize_mean, device=batch.device).view(1, -1, 1, 1, 1)
            std = torch.tensor(normalize_std, device=batch.device).view(1, -1, 1, 1, 1)
        elif len(batch.shape) == 4:  # (B, C, H, W)
            mean = torch.tensor(normalize_mean, device=batch.device).view(1, -1, 1, 1)
            std = torch.tensor(normalize_std, device=batch.device).view(1, -1, 1, 1)
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {batch.shape}")
    else:
        assert (
            len(normalize_mean) == len(normalize_std) == 3
        ), f"Expected 3 channels, got {len(normalize_mean)} and {len(normalize_std)}"
        if len(batch.shape) == 5:  # (B, C, D, H, W)
            assert batch.shape[1] == 1, f"Expected 1 channel, got {batch.shape[1]}"
            mean = torch.tensor(normalize_mean, device=batch.device).view(1, -1, 1, 1, 1)
            std = torch.tensor(normalize_std, device=batch.device).view(1, -1, 1, 1, 1)
            batch = batch.repeat(1, 3, 1, 1, 1)
        elif len(batch.shape) == 4:  # (B, C, H, W)
            assert batch.shape[1] == 1, f"Expected 1 channel, got {batch.shape[1]}"
            mean = torch.tensor(normalize_mean, device=batch.device).view(1, -1, 1, 1)
            std = torch.tensor(normalize_std, device=batch.device).view(1, -1, 1, 1)
            batch = batch.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {batch.shape}")

    batch = (batch - mean) / std

    return batch
