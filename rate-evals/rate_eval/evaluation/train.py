"""Plain-PyTorch per-fold trainer for the linear / MLP probe (AdamW + OneCycleLR + early-stop on val AUROC)."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from .data import FeatureDataset


logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Per-fold training hyperparameters."""

    max_epochs: int = 32
    patience: int = 16
    batch_size: int = 64
    max_lr: float = 1e-4
    div_factor: float = 25.0
    weight_decay: float = 0.01
    use_class_balance: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    device: Optional[str] = None


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_logits_and_labels(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    head.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=False)
            logits = head(x).squeeze(-1)
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(y.detach().cpu().numpy())
    return np.concatenate(logits_list), np.concatenate(labels_list)


def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def train_one_fold(
    head: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    config: Optional[TrainConfig] = None,
) -> Tuple[nn.Module, dict]:
    """Train `head` (logits shape (B, 1) or (B,)), early-stop on val AUROC; returns best-weights head + stats dict."""
    cfg = config or TrainConfig()
    device = _resolve_device(cfg.device)

    head = head.to(device)

    train_ds = FeatureDataset(train_features, train_labels)
    val_ds = FeatureDataset(val_features, val_labels)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(cfg.batch_size, 256), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    n_pos = int((train_labels == 1).sum().item())
    n_neg = int((train_labels == 0).sum().item())
    if cfg.use_class_balance and n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    else:
        pos_weight = None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.max_lr / cfg.div_factor, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        epochs=cfg.max_epochs,
        steps_per_epoch=steps_per_epoch,
        div_factor=cfg.div_factor,
        anneal_strategy="cos",
    )

    best_state = copy.deepcopy(head.state_dict())
    best_val_auroc = -float("inf")
    best_epoch = -1
    train_loss_history: list[float] = []
    val_auroc_history: list[float] = []
    epochs_since_improvement = 0

    for epoch in range(cfg.max_epochs):
        head.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            logits = head(x).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= max(1, n_batches)
        train_loss_history.append(epoch_loss)

        val_logits, val_y = _eval_logits_and_labels(head, val_loader, device)
        val_auroc = _binary_auroc(val_y, val_logits)
        val_auroc_history.append(val_auroc)

        improved = val_auroc > best_val_auroc
        if improved:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        logger.debug(
            "epoch %d: train_loss=%.4f val_auroc=%.4f%s",
            epoch, epoch_loss, val_auroc, "  (best)" if improved else "",
        )

        if epochs_since_improvement >= cfg.patience:
            logger.info(
                "early stop at epoch %d (no val improvement for %d epochs; best epoch %d, val_auroc=%.4f)",
                epoch, cfg.patience, best_epoch, best_val_auroc,
            )
            break

    head.load_state_dict(best_state)
    return head, {
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_val_auroc),
        "n_epochs_run": len(train_loss_history),
        "train_loss_history": train_loss_history,
        "val_auroc_history": val_auroc_history,
        "n_train_pos": n_pos,
        "n_train_neg": n_neg,
    }


def predict_proba(
    head: nn.Module,
    features: torch.Tensor,
    *,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return per-sample sigmoid probabilities of class 1, shape (N,)."""
    dev = _resolve_device(device)
    head = head.to(dev).eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            x = features[start:start + batch_size].to(dev).float()
            logits = head(x).squeeze(-1)
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out)


# multi-class sibling of the binary path; class weighting off by default (Curia Sec 4.5.6)


def _eval_logits_multiclass(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    head.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=False)
            logits = head(x)
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(y.detach().cpu().numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list)


def _val_balanced_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    pred = np.argmax(logits, axis=1)
    return float(balanced_accuracy_score(labels, pred))


def train_one_fold_multiclass(
    head: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    *,
    config: Optional[TrainConfig] = None,
    use_class_weights: bool = False,
) -> Tuple[nn.Module, dict]:
    """Train multi-class `head` (logits (B, num_classes)), early-stop on val balanced accuracy."""
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    cfg = config or TrainConfig()
    device = _resolve_device(cfg.device)

    head = head.to(device)

    train_ds = FeatureDataset(train_features, train_labels)
    val_ds = FeatureDataset(val_features, val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(cfg.batch_size, 256), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    weight_t: Optional[torch.Tensor] = None
    if use_class_weights:
        labels_np = train_labels.detach().cpu().numpy().astype(int)
        per_class = np.bincount(labels_np, minlength=num_classes).astype(np.float64)
        # inverse-frequency, normalized to sum to num_classes
        with np.errstate(divide="ignore"):
            inv = np.where(per_class > 0, 1.0 / per_class, 0.0)
        if inv.sum() > 0:
            inv = inv * (num_classes / inv.sum())
        weight_t = torch.tensor(inv, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_t)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=cfg.max_lr / cfg.div_factor,
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        epochs=cfg.max_epochs,
        steps_per_epoch=steps_per_epoch,
        div_factor=cfg.div_factor,
        anneal_strategy="cos",
    )

    best_state = copy.deepcopy(head.state_dict())
    best_val_bacc = -float("inf")
    best_epoch = -1
    train_loss_history: list[float] = []
    val_bacc_history: list[float] = []
    epochs_since_improvement = 0

    for epoch in range(cfg.max_epochs):
        head.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= max(1, n_batches)
        train_loss_history.append(epoch_loss)

        val_logits, val_y = _eval_logits_multiclass(head, val_loader, device)
        val_bacc = _val_balanced_accuracy(val_logits, val_y)
        val_bacc_history.append(val_bacc)

        improved = val_bacc > best_val_bacc
        if improved:
            best_val_bacc = val_bacc
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        logger.debug(
            "[mc] epoch %d: train_loss=%.4f val_bacc=%.4f%s",
            epoch, epoch_loss, val_bacc, "  (best)" if improved else "",
        )

        if epochs_since_improvement >= cfg.patience:
            logger.info(
                "[mc] early stop at epoch %d (no val improvement for %d epochs; best epoch %d, val_bacc=%.4f)",
                epoch, cfg.patience, best_epoch, best_val_bacc,
            )
            break

    head.load_state_dict(best_state)
    n_per_class = np.bincount(
        train_labels.detach().cpu().numpy().astype(int), minlength=num_classes,
    ).tolist()
    return head, {
        "best_epoch": best_epoch,
        "best_val_bacc": float(best_val_bacc),
        "n_epochs_run": len(train_loss_history),
        "train_loss_history": train_loss_history,
        "val_bacc_history": val_bacc_history,
        "n_train_per_class": n_per_class,
        "num_classes": num_classes,
    }


def predict_proba_multiclass(
    head: nn.Module,
    features: torch.Tensor,
    *,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return per-sample softmax probabilities, shape (N, C)."""
    dev = _resolve_device(device)
    head = head.to(dev).eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            x = features[start:start + batch_size].to(dev).float()
            logits = head(x)
            out.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(out, axis=0)
