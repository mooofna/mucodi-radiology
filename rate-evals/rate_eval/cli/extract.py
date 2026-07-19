"""CLI for `rate-extract`: parse argparse -> build `ExtractRequest` -> delegate to `pipelines.extract`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .. import setup_pipeline
from ..config import apply_cli_overrides
from ..config.hydra import (
    create_hydra_compatible_cli,
    get_hydra_overrides_from_env,
    load_config_with_hydra,
    parse_hydra_overrides_from_args,
)
from ..core.logging import get_logger, setup_logging
from ..pipelines.extract import ExtractRequest, run as run_extract

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse surface (separate so tests can introspect it without running the CLI)."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract embeddings from radiology foundation models with graceful resume. "
            "Supports Hydra-style overrides: key.subkey=value"
        ),
        epilog=(
            "Examples:\n"
            "  rate-extract --model curia2 --dataset radchestct\n"
            "  rate-extract --model curia2 --dataset radchestct hardware.batch_size_per_gpu=32\n"
            "  rate-extract --model ctclip_zero_shot --dataset radchestct/ctclip_zero_shot model.checkpoint_path=..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'curia2', 'ctclip_zero_shot')")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., 'radchestct', 'lidc_chest_ct')"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Process all available splits (train, valid, test)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for embeddings (default: cache/{model}_{dataset})",
    )
    parser.add_argument(
        "--skip-existing-cache",
        action="store_true",
        default=False,
        help="Ignore previously cached embeddings and process every sample from scratch",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size per GPU (overrides config)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use for single GPU mode"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG level logging for detailed tracing (helps debug NaN issues)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help=(
            "Continue extraction even when errors occur (logs error but doesn't stop). "
            "WARNING: May result in incomplete or corrupted data!"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader workers per GPU (overrides config, default: 4)",
    )
    parser.add_argument(
        "--check-nan",
        action="store_true",
        default=False,
        help="Enable detailed NaN checking and logging (may impact performance)",
    )
    parser.add_argument(
        "--model-repo-id",
        type=str,
        default=None,
        help="Override model repository ID (e.g., 'YalaLab/model_name')",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help="Override model revision (e.g., 'epoch_16')",
    )
    parser.add_argument(
        "--ct-window-type",
        type=str,
        default=None,
        help=(
            "CT window type for CT windowing (e.g., 'minmax', 'lung', 'all') "
            "(default: use config default)"
        ),
    )
    parser.add_argument(
        "--pool-op",
        type=str,
        default=None,
        help="Pooling operation to use (e.g., 'mean', 'max', 'cls') (default: use config default)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        help=(
            "Modality to use for feature extraction (e.g., 'abdomen_ct', 'chest_xray_two_view') "
            "(default: use dataset default)"
        ),
    )
    return parser


def extract_embeddings_cli() -> None:
    """CLI entry point -- argparse + delegate to `pipelines.extract.run`."""
    parser = _build_parser()

    # peel Hydra-style trailing key=value overrides before argparse
    regular_args, cli_hydra_overrides = parse_hydra_overrides_from_args(sys.argv[1:])
    args = parser.parse_args(regular_args)

    # log-level precedence: env var > --debug > INFO
    log_level = os.environ.get("RATE_LOG_LEVEL", "INFO").upper()
    if args.debug:
        log_level = "DEBUG"
    setup_logging(level=log_level, rank=0, world_size=1, debug=args.debug)
    if log_level == "DEBUG":
        logger.debug("Debug mode active")

    if args.continue_on_error:
        logger.warning("!  WARNING: --continue-on-error flag is enabled!")
        logger.warning("Extraction will continue even when errors occur.")
        logger.warning("This may result in incomplete or corrupted data being saved.")
        logger.warning("Use this flag only for debugging purposes.")

    # default cache dir cache/<model>_<dataset>; sanitize '/' from nested dataset name
    if args.output_dir is None:
        safe_dataset = args.dataset.replace("/", "_")
        args.output_dir = f"cache/{args.model}_{safe_dataset}"

    splits = ["train", "valid", "test"] if args.all_splits else [args.split]
    if args.all_splits:
        logger.info("Processing all splits: train, valid, test")

    # combine env-set Hydra overrides with CLI-parsed ones
    hydra_overrides = list(get_hydra_overrides_from_env())
    for override in cli_hydra_overrides:
        if override not in hydra_overrides:
            hydra_overrides.append(override)

    if hydra_overrides:
        logger.info(f"Applying Hydra overrides: {hydra_overrides}")
        config = load_config_with_hydra(
            config_name="config" if args.config is None else Path(args.config).stem,
            model_name=args.model,
            dataset_name=args.dataset,
            overrides=hydra_overrides,
        )
    else:
        config = setup_pipeline(config_path=args.config, model=args.model, dataset=args.dataset)

    if log_level == "DEBUG":
        from omegaconf import OmegaConf

        logger.debug("=" * 80)
        logger.debug("FULL CONFIGURATION:")
        logger.debug("=" * 80)
        logger.debug(OmegaConf.to_yaml(config))
        logger.debug("=" * 80)

    applied = apply_cli_overrides(config, args)
    if applied:
        logger.debug(f"Applied CLI overrides: {applied}")

    request = ExtractRequest(
        model=args.model,
        dataset=args.dataset,
        splits=splits,
        output_dir=Path(args.output_dir),
        config=config,
        max_samples=args.max_samples,
        skip_existing_cache=args.skip_existing_cache,
        continue_on_error=args.continue_on_error,
        check_nan=args.check_nan,
        debug=args.debug,
        num_gpus=args.num_gpus,
        # also on config via apply_cli_overrides; kept for traceability
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    run_extract(request)


def main() -> None:
    """Main entry point that supports Hydra overrides via the decorator wrapper."""
    hydra_cli = create_hydra_compatible_cli(extract_embeddings_cli)
    hydra_cli()


if __name__ == "__main__":
    main()
