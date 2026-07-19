"""CLI for one cross-cohort ``crossval -> deploy`` cell (one abnormality class); see `evaluation.deploy`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _run(args: argparse.Namespace) -> None:
    # lazy imports keep import cheap
    from ..core.logging import setup_logging
    from ..evaluation.deploy import deploy_one_class

    setup_logging(level=args.log_level)
    head_spec = {"kind": args.head}
    if args.head_config_json:
        head_spec.update(json.loads(args.head_config_json))

    result = deploy_one_class(
        source_cache=Path(args.source_checkpoint_dir),
        source_labels_json=Path(args.source_labels_json),
        target_cache=Path(args.target_checkpoint_dir),
        target_labels_json=Path(args.target_labels_json),
        class_name=args.class_name,
        out_dir=Path(args.output_dir),
        head_spec=head_spec,
        l2_normalize=args.l2_normalize,
        source_cohort=args.source_cohort,
        target_cohort=args.target_cohort,
        cv_folds=args.cv_folds,
        cv_seed=args.cv_seed,
        seed=args.seed,
        val_fraction=args.val_fraction,
        n_boot=args.n_boot,
        alpha=args.alpha,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    def _fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    print(
        f"[deploy] {result['class_name']}: internal AUROC={_fmt(result['internal_auroc'])} "
        f"external AUROC={_fmt(result['external_auroc'])} "
        f"(n_src={result['n_source']} n_tgt={result['n_target']})"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One cross-cohort crossval->deploy cell (source CV + fold-ensemble target deploy).",
    )
    p.add_argument("--source-checkpoint-dir", required=True,
                   help="Source (CT-RATE) rate-extract cache dir (embeddings/{split}/*.npz).")
    p.add_argument("--source-labels-json", required=True, help="Source per-class label JSON.")
    p.add_argument("--target-checkpoint-dir", required=True,
                   help="Target (RAD-ChestCT) rate-extract cache dir.")
    p.add_argument("--target-labels-json", required=True, help="Target per-class label JSON.")
    p.add_argument("--class-name", required=True,
                   help="Abnormality slug; names the per-class subdir under internal/ and external/.")
    p.add_argument("--output-dir", required=True,
                   help="Cell dir; writes {internal,external}/per_class/<class-name>/ beneath it.")
    p.add_argument("--head", default="mlp", choices=["linear", "mlp"])
    p.add_argument("--head-config-json", default=None)
    p.add_argument("--l2-normalize", action="store_true",
                   help="L2-normalize each feature vector at eval time (both source and target).")
    p.add_argument("--source-cohort", default="ct_rate",
                   help="Patient-grouping key for the source CV (default ct_rate).")
    p.add_argument("--target-cohort", default="radchestct",
                   help="Patient-grouping key for the target deploy (default radchestct = identity).")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-epochs", type=int, default=32)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    p.add_argument("--log-level", default="INFO")
    p.set_defaults(func=_run)
    return p


def deploy_embeddings_cli() -> None:
    """rate-deploy console-script entry."""
    args = _build_parser().parse_args()
    args.func(args)


def main() -> None:
    deploy_embeddings_cli()


if __name__ == "__main__":
    main()
