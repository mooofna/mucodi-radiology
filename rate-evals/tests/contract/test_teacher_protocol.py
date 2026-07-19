"""Every registered teacher wrapper satisfies the method-only `TeacherWrapper` protocol."""

from __future__ import annotations

from importlib import import_module

import pytest

from rate_eval.components import _MODEL_REGISTRY
from rate_eval.models.base import TeacherWrapper


def _import_wrapper_class(module_path: str, class_name: str):
    """Import the wrapper class lazily; skip the test if an optional dep (e.g. flash-attn) makes it unimportable."""
    try:
        module = import_module(module_path)
    except ImportError as exc:  # pragma: no cover -- env-dependent
        pytest.skip(f"wrapper module {module_path} not importable in this env: {exc}")
    if not hasattr(module, class_name):  # pragma: no cover -- registry drift
        pytest.fail(f"{module_path} has no class {class_name} (registry drift)")
    return getattr(module, class_name)


@pytest.mark.parametrize(
    "registry_name,module_path,class_name",
    [
        (name, mod, cls)
        for name, (mod, cls) in _MODEL_REGISTRY.items()
    ],
)
def test_wrapper_satisfies_teacher_protocol(
    registry_name: str, module_path: str, class_name: str
):
    wrapper_cls = _import_wrapper_class(module_path, class_name)
    assert issubclass(wrapper_cls, TeacherWrapper), (
        f"{registry_name} (-> {module_path}.{class_name}) does not satisfy the "
        f"TeacherWrapper protocol. Missing methods? Expected: extract_features, eval."
    )
