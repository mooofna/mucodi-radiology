"""Pure-function orchestration layer (Request -> Result, no argparse) behind the `rate_eval.cli` shims."""

from .extract import ExtractRequest, ExtractResult, SplitResult, run as run_extract

__all__ = [
    "ExtractRequest",
    "ExtractResult",
    "SplitResult",
    "run_extract",
]
