"""3D training loop with gradient accumulation for radiology KD."""
from __future__ import annotations

import time
from contextlib import nullcontext

import torch
import wandb

from utils import utils
from moco.loss import KDInfoNCELoss
from rate_eval.models.common import batch_apply_ct_windowing


def _compute_per_teacher_losses(out, gathered, kd_criterion, teacher_order):
    """Per-teacher InfoNCE losses as (name, loss) pairs in teacher_order."""
    out_dict = out
    pairs = []
    for t_name in teacher_order:
        if t_name not in out_dict or t_name not in gathered:
            continue
        k_all, labels = gathered[t_name]
        pairs.append(
            (t_name, kd_criterion.forward_single(out_dict[t_name], k_all, labels))
        )
    return pairs


def _compute_per_teacher_losses_bank(out, bank_cache, kd_criterion, teacher_order,
                                     logit_bias=None):
    """Feature-bank analog of _compute_per_teacher_losses."""
    out_dict = out
    pairs = []
    for t_name in teacher_order:
        if t_name not in out_dict or t_name not in bank_cache:
            continue
        pos_rows, neg_rows = bank_cache[t_name]
        pairs.append(
            (t_name, kd_criterion.forward_bank(out_dict[t_name], pos_rows, neg_rows,
                                               logit_bias=logit_bias))
        )
    return pairs


def _losses_only(per_teacher_pairs):
    return [l for _, l in per_teacher_pairs]


def _dispatch_backward(per_teacher_pairs, scale):
    """Sum the per-teacher InfoNCE losses (uniform) and backward once."""
    if not per_teacher_pairs:
        return
    scaled = [l * scale for l in _losses_only(per_teacher_pairs)]
    total = scaled[0]
    for l in scaled[1:]:
        total = total + l
    total.backward()


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    args,
    teacher_dims_dict,
    start_step: int = 0,
    bank=None,
):
    """K-step gradient-accumulation training loop."""
    K = max(1, int(getattr(args, "grad_accum_steps", 1)))
    teacher_order = list(teacher_dims_dict.keys())

    # neg source: "gather" all_gathers batch keys; "bank"/"bank-full" sample the frozen bank
    neg_source = getattr(args, "neg_source", "gather")
    neg_bank_size = int(getattr(args, "neg_bank_size", 4096))
    neg_mask_mode = getattr(args, "neg_mask_false_negatives", "study")
    using_bank = neg_source != "gather"
    if using_bank and bank is None:
        raise RuntimeError(
            f"neg_source={neg_source!r} requires a frozen feature bank but bank is None "
            "(main.py should build it via moco.feature_bank.build_frozen_bank)")
    neg_gen = None
    if using_bank:
        # same seed across ranks; re-seeded per step so negatives are reproducible
        seed = int(getattr(args, "seed", None) or 0)
        neg_gen = torch.Generator(device=bank.device).manual_seed(seed)
    # main.py stashes profile["windowing"]["type"] here; "all" -> 11 windows.
    ct_window_type = getattr(args, "ct_window_type", "all") or "all"

    batch_time = utils.AverageMeter("Time", ":6.3f")
    data_time = utils.AverageMeter("Data", ":6.3f")
    learning_rates = utils.AverageMeter("LR", ":.4e")
    losses_meter = utils.AverageMeter("Loss", ":.4e")
    vols_meter = utils.ValueMeter("Vols", ":d")

    progress = utils.ProgressMeter(
        [batch_time, data_time, learning_rates, losses_meter, vols_meter],
        prefix="Step",
    )

    amp_enabled = getattr(args, "amp", False)
    amp_dtype = torch.bfloat16
    if amp_enabled and args.rank == 0 and start_step == 0:
        print(f"AMP enabled with dtype={amp_dtype}")

    if args.rank == 0:
        print(
            f"Gradient accumulation: K={K} micro-batches per effective step "
            f"(effective batch ~= {args.batch_size * K * max(1, args.world_size)} samples)"
        )
        print(f"Loss aggregation: uniform sum; teacher_order={teacher_order}")
        if using_bank:
            n_neg = bank.N if neg_source == "bank-full" else neg_bank_size
            print(
                f"Negative source: {neg_source} (M={n_neg} negatives/step from a frozen "
                f"bank of N={bank.N}; mask_false_negatives={neg_mask_mode}; no all_gather)"
            )
        else:
            print("Negative source: gather (all_gather of the current batch's teacher keys)")

    model.train()
    kd_criterion = KDInfoNCELoss(temperature=args.moco_t).cuda(args.gpu)

    end = time.time()
    step_idx = start_step
    total_steps = args.train_steps
    loader_iter = iter(train_loader)

    while step_idx < total_steps:
        lr = utils.adjust_learning_rate_steps(optimizer, step_idx, args)
        learning_rates.update(lr)

        optimizer.zero_grad(set_to_none=True)

        # keep losses on GPU; defer .item() pulls past the K-loop to avoid per-micro sync
        accum_log_tensors: dict[str, torch.Tensor] = {}
        accum_loss_tensor: torch.Tensor | None = None
        accum_vols = 0
        no_sync = model.no_sync if hasattr(model, "no_sync") else nullcontext

        # K micro-batches under no_sync; only the last one syncs DDP gradients.
        for micro_idx in range(K):
            try:
                im_q1, im_q2, feats_dict, meta = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                im_q1, im_q2, feats_dict, meta = next(loader_iter)

            data_time.update(time.time() - end)

            # windowing on GPU to keep CPU prefetch light
            if args.gpu is not None:
                im_q1 = im_q1.cuda(args.gpu, non_blocking=True)
                im_q2 = im_q2.cuda(args.gpu, non_blocking=True)
                im_q1 = batch_apply_ct_windowing(
                    im_q1, ct_window_type=ct_window_type, modality="CT", per_sample=True
                )
                im_q2 = batch_apply_ct_windowing(
                    im_q2, ct_window_type=ct_window_type, modality="CT", per_sample=True
                )
                teacher_features_gpu = {}
                for name in teacher_dims_dict.keys():
                    if name in feats_dict:
                        teacher_features_gpu[name] = (
                            feats_dict[name].cuda(args.gpu, non_blocking=True).float().detach()
                        )
            else:
                im_q1 = batch_apply_ct_windowing(
                    im_q1, ct_window_type=ct_window_type, modality="CT", per_sample=True
                )
                im_q2 = batch_apply_ct_windowing(
                    im_q2, ct_window_type=ct_window_type, modality="CT", per_sample=True
                )
                teacher_features_gpu = {
                    name: feats_dict[name].float().detach()
                    for name in teacher_dims_dict.keys()
                    if name in feats_dict
                }

            # build negatives + per-view loss closure
            if not using_bank:
                gathered = kd_criterion.gather_keys_multi(teacher_features_gpu)
                def compute_losses(out, _g=gathered):
                    return _compute_per_teacher_losses(out, _g, kd_criterion, teacher_order)
            else:
                pos_idx = bank.indices_for(meta["sample_names"])                       # [B]
                if neg_source == "bank-full":
                    neg_idx = torch.arange(bank.N, device=bank.device)        # [N]
                else:
                    if neg_gen is not None:
                        neg_gen.manual_seed(int(getattr(args, "seed", 0) or 0) * 1_000_003
                                            + step_idx * 131 + micro_idx)
                    neg_idx = bank.sample_indices(neg_bank_size, generator=neg_gen)  # [M]
                bank_cache = {
                    t: (bank.gather(t, pos_idx), bank.gather(t, neg_idx))
                    for t in teacher_order if t in teacher_features_gpu
                }
                # fp32 bias; -inf masks false-negative columns
                logit_bias = bank.false_neg_bias(pos_idx, neg_idx, neg_mask_mode, torch.float32)
                def compute_losses(out, _bc=bank_cache, _lb=logit_bias):
                    return _compute_per_teacher_losses_bank(
                        out, _bc, kd_criterion, teacher_order, logit_bias=_lb)

            is_last_micro = micro_idx == (K - 1)
            sync_ctx = nullcontext() if is_last_micro else no_sync()

            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    out1 = model(im_q1)
                # InfoNCE objective in fp32 (backbone forward stays bf16)
                with torch.amp.autocast("cuda", enabled=False):
                    per_teacher_v1 = compute_losses({t: q.float() for t, q in out1.items()})
                for t_name, ell in per_teacher_v1:
                    key = f"train/loss_{t_name}_v1"
                    ell_d = ell.detach()
                    accum_log_tensors[key] = (
                        ell_d if key not in accum_log_tensors
                        else accum_log_tensors[key] + ell_d
                    )
                _dispatch_backward(per_teacher_v1, scale=1.0 / (2 * K))

                # view 2: symmetric
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    out2 = model(im_q2)
                with torch.amp.autocast("cuda", enabled=False):
                    per_teacher_v2 = compute_losses({t: q.float() for t, q in out2.items()})
                for t_name, ell in per_teacher_v2:
                    key = f"train/loss_{t_name}_v2"
                    ell_d = ell.detach()
                    accum_log_tensors[key] = (
                        ell_d if key not in accum_log_tensors
                        else accum_log_tensors[key] + ell_d
                    )
                _dispatch_backward(per_teacher_v2, scale=1.0 / (2 * K))

            step_loss_tensor = sum(
                l.detach() for _, l in per_teacher_v1
            ) + sum(l.detach() for _, l in per_teacher_v2)
            accum_loss_tensor = (
                step_loss_tensor if accum_loss_tensor is None
                else accum_loss_tensor + step_loss_tensor
            )
            accum_vols += int(im_q1.size(0))

        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        optimizer.step()

        # pull loss scalars to CPU once per step
        accum_loss = (
            accum_loss_tensor.item() if accum_loss_tensor is not None else 0.0
        )
        accum_log = {k: v.item() for k, v in accum_log_tensors.items()}

        losses_meter.update(accum_loss / K, accum_vols)
        world_size = max(1, getattr(args, "world_size", 1))
        global_vols = accum_vols * world_size
        vols_meter.update(global_vols)

        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0:
            wandb_log = {
                "step": step_idx,
                "train/loss_total": accum_loss / K,
                "train/lr": float(lr),
                "timing/data_sec": data_time.val,
                "data/global_num_vols": global_vols,
                "grad_accum/K": K,
            }
            for k, v in accum_log.items():
                wandb_log[k] = v / K
            wandb.log(wandb_log, step=step_idx)

            if step_idx % args.print_freq == 0:
                progress.display(step_idx)

        # only rank 0 writes checkpoints (avoid tmp-file race)
        right_rank = args.rank == 0
        right_step = ((step_idx % args.save_every == 0) or (step_idx == args.train_steps // 2)) and (step_idx > start_step)
        if right_rank and right_step:
            utils.save_checkpoint(model, optimizer, step_idx, args)

        step_idx += 1

    if args.rank == 0:
        utils.save_checkpoint(model, optimizer, total_steps, args)
        progress.display_summary()
