from __future__ import annotations

import pytest

from mucodi_demo.evidence_results import detex


@pytest.mark.parametrize(
    ("raw", "plain"),
    [
        (r"Lung-RADS category $\geq 3$", "Lung-RADS category ≥ 3"),
        (r"LR$\geq$3", "LR≥3"),
        (r"RV/LV$\vert$PE", "RV/LV|PE"),
        (r"median margin $\leq 3$", "median margin ≤ 3"),
        (r"involvement (CT-3/CT-4, $>50\%$)", "involvement (CT-3/CT-4, >50%)"),
        (r"baseline FVC\%pred $< \SI{75}{\percent}$", "baseline FVC%pred < 75 %"),
        (r"abs.\ FVC decline $\geq 5$ pts within 52\,wk", "abs. FVC decline ≥ 5 pts within 52 wk"),
        ("plain text stays", "plain text stays"),
    ],
)
def test_detex(raw: str, plain: str) -> None:
    assert detex(raw) == plain
