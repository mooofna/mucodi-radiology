"""3D radiology knowledge-distillation training entrypoint."""
from __future__ import annotations

import os
import signal
import builtins
import random
import datetime as dt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

import warnings
warnings.filterwarnings("ignore", message="Overwriting .* in registry")

from utils import options
from utils import engine
from dataprep.config import load_profile, get_teacher_dims_str
from dataprep import loader
from models.student import MultiTeacherStudent
from models.efficientnet3d import create_3d_backbone, probe_feat_dim


def init_wandb(args, run_name):
    extra = {"git_sha": os.environ.get("GIT_SHA", "unknown")}
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "radiology-fm-distillation"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_name,
        resume="allow",
        config={**vars(args), **extra},
        settings=wandb.Settings(quiet=True),
    )


def _parse_betas(s: str) -> tuple[float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--betas expected 'b1,b2', got {s!r}")
    return float(parts[0]), float(parts[1])


def derive_window_spec(profile: dict) -> tuple:
    """Resolve ``(ct_window_type, in_channels)`` from a profile's windowing config."""
    win_cfg = profile.get("windowing", {})
    ct_window_type = win_cfg.get("type")
    if ct_window_type == "all":
        in_channels = 11
    elif isinstance(ct_window_type, list):
        in_channels = len(ct_window_type)
    else:
        in_channels = 1
    return ct_window_type, in_channels


def main():
    parser = options.get_args_parser()
    args = parser.parse_args()

    # resolve before mp.spawn so all ranks inherit args.resume
    if getattr(args, "auto_resume", False) and not args.resume:
        import glob as _glob
        _cks = sorted(_glob.glob(os.path.join(args.save_dir, "step_*.pth.tar")))
        if _cks:
            args.resume = _cks[-1]
            print(f"[auto-resume] resuming from latest checkpoint: {args.resume}")
        else:
            print(f"[auto-resume] no checkpoint in {args.save_dir}; starting fresh")

    # cap CPU threads vs dataloader workers
    torch.set_num_threads(1)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    profile = load_profile(args.dataset)
    if "teacher_dims" not in profile:
        raise ValueError(f"profile {args.dataset!r} missing required 'teacher_dims' field")
    if not args.teacher_dims:
        args.teacher_dims = get_teacher_dims_str(profile)

    ngpus_per_node = torch.cuda.device_count()
    if args.dist_url == "env://" and not args.multiprocessing_distributed:
        # torchrun set RANK/LOCAL_RANK/WORLD_SIZE
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        args.rank = int(os.environ.get("RANK", 0))
        args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        print(
            f"[main] torchrun env:// -- rank {args.rank}/{args.world_size} "
            f"local_rank {args.gpu}"
        )
        main_worker(args.gpu, ngpus_per_node, args)
    elif args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        print(
            f"Launching {ngpus_per_node} processes per node, "
            f"for a total of {args.world_size} processes."
        )
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.multiprocessing_distributed and (gpu != 0 or args.rank != 0):
        builtins.print = lambda *args, **kwargs: None

    print(
        f"[main] arch={args.arch} dataset={args.dataset} "
        f"per-GPU batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps}"
    )
    print(f"Learning rate: {args.lr}, optimizer: AdamW")

    if getattr(args, "amp", False):
        args.amp_dtype = torch.bfloat16
        print("AMP enabled with dtype=bfloat16")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            device_id=torch.device(f"cuda:{args.gpu}") if args.gpu is not None else None,
            timeout=dt.timedelta(hours=4),
        )
        torch.distributed.barrier()

    stamp = dt.datetime.now().strftime("%y%m%d-%H%M")
    run_name = os.environ.get("WANDB_NAME") or f"{args.arch}|{args.dataset}|{stamp}"

    if args.rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    # stash ct_window_type for utils/engine.py
    profile = load_profile(args.dataset)
    args.ct_window_type, in_channels = derive_window_spec(profile)

    backbone = create_3d_backbone(args.arch, in_channels=in_channels)

    print(
        f"\n[Backbone] {args.arch} (in_channels={in_channels}) "
        f"with {sum(p.numel() for p in backbone.parameters()):,} parameters\n"
    )

    student_dim = probe_feat_dim(backbone, in_channels=in_channels, probe_shape=(8, 8, 8))
    print(f"[Backbone] feat_dim={student_dim}")

    teacher_dims_dict = options.parse_teacher_dims(args.teacher_dims)
    if len(teacher_dims_dict) != 1:
        print(
            f"[main] note: {len(teacher_dims_dict)} teachers configured; "
            "single-teacher mode is the validated first-run path."
        )

    # teacher_order must match --teacher-dims order or heads/losses misalign
    profile_teacher_order = profile.get("teacher_order")
    if profile_teacher_order is not None:
        derived = list(teacher_dims_dict.keys())
        if derived != list(profile_teacher_order):
            raise RuntimeError(
                f"teacher order mismatch between profile {args.dataset!r} "
                f"`teacher_order` field ({list(profile_teacher_order)}) and "
                f"--teacher-dims-derived dict insertion order ({derived}). "
                f"This would silently misalign the per-teacher heads/losses. "
                f"Fix the YAML profile or --teacher-dims CLI arg."
            )

    student_mt = MultiTeacherStudent(
        backbone,
        feat_dim=student_dim,
        teacher_dims=teacher_dims_dict,
        hidden=student_dim,
        num_layers=args.num_layers,
        projector_arch=args.projector_arch,
        dropout=args.projector_dropout,
    )

    if args.distributed:
        student_mt = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_mt)
        torch.cuda.set_device(args.gpu)
        student_mt.cuda(args.gpu)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    else:
        student_mt.cuda(args.gpu)

    model = student_mt

    if args.rank == 0:
        init_wandb(args, run_name)
        wandb.watch(model, log="all", log_freq=100)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            bucket_cap_mb=50,
        )

    param_groups = [{"params": model.parameters(), "name": "student"}]

    betas = _parse_betas(args.betas)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=betas,
    )

    n_optim_tensors = sum(len(g["params"]) for g in optimizer.param_groups)
    n_backbone_tensors = sum(1 for _ in backbone.parameters())
    if n_optim_tensors < n_backbone_tensors:
        raise RuntimeError(
            f"Optimizer has only {n_optim_tensors} param tensors; expected "
            f">={n_backbone_tensors} (the backbone alone). A model wrapper is hiding "
            f"submodules from .parameters()."
        )
    if args.rank == 0:
        print(f"[main] optimizer scope: {n_optim_tensors} param tensors")

    if args.resume and os.path.isfile(args.resume):
        loc = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
        checkpoint = torch.load(args.resume, map_location=loc, weights_only=False)
        args.start_step = checkpoint["step"] + 1
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"=> Loaded checkpoint '{args.resume}' (step {checkpoint['step']})")

    torch.set_float32_matmul_precision("high")
    cudnn.benchmark = True
    cudnn.deterministic = False

    train_loader = loader.get_volume_loader(args, split="train")

    # frozen-teacher feature bank for InfoNCE negatives; see moco/feature_bank.py
    bank = None
    if getattr(args, "neg_source", "gather") != "gather":
        from moco.feature_bank import build_frozen_bank
        bank_device = args.gpu if args.gpu is not None else (
            "cuda" if torch.cuda.is_available() else "cpu")
        # expected corpus size so the bank-build gate catches an undersized bank
        _expected_n = (len(train_loader.dataset)
                       if os.environ.get("KDSWEEP_SKIP_CACHE_FILTER") == "1"
                       and hasattr(train_loader.dataset, "__len__") else None)
        bank = build_frozen_bank(
            profile,
            teacher_dims_dict,
            dtype=args.neg_bank_dtype,
            device=bank_device,
            rank=args.rank,
            expected_n=_expected_n,
        )

    engine.train_one_epoch(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        args=args,
        teacher_dims_dict=teacher_dims_dict,
        start_step=args.start_step,
        bank=bank,
    )

    if args.rank == 0:
        signal.signal(signal.SIGALRM, lambda *_: os._exit(0))
        signal.alarm(120)
        wandb.finish(quiet=True, exit_code=0)
        signal.alarm(0)


if __name__ == "__main__":
    main()
