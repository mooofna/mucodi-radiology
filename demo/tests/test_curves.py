"""ROC and PR summaries computed from out-of-fold predictions for the evidence page."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mucodi_demo.evidence_results import GRID, _roc_pr


def mann_whitney_auc(labels: npt.NDArray[np.int64], scores: npt.NDArray[np.float64]) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


@pytest.fixture
def predictions() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    scores = np.round(rng.normal(0.8 * labels, 1.0), 1)  # rounding creates tied scores
    return labels, scores


def test_auroc_equals_mann_whitney_with_ties(
    predictions: tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]],
) -> None:
    labels, scores = predictions
    out = _roc_pr(labels, scores)
    assert out["auroc"] == pytest.approx(mann_whitney_auc(labels, scores), abs=1e-12)
    assert (out["n_pos"], out["n_neg"]) == (int(labels.sum()), int((1 - labels).sum()))


def test_curves_live_on_the_grid_and_are_monotone(
    predictions: tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]],
) -> None:
    out = _roc_pr(*predictions)
    tpr, precision = np.array(out["roc_tpr"]), np.array(out["pr_precision"])
    assert len(tpr) == len(precision) == len(GRID) == 201
    assert np.all(np.diff(tpr) >= 0) and tpr[-1] == 1.0
    assert np.all(np.diff(precision) <= 0)  # interpolated precision never rises with recall
    ops = {o["name"]: o for o in out["ops"]}
    assert ops["sens90"]["sens"] >= 0.9
    assert ops["spec90"]["spec"] >= 0.9
    youden = ops["youden"]["sens"] + ops["youden"]["spec"]
    assert all(youden >= o["sens"] + o["spec"] - 1e-9 for o in out["ops"])
