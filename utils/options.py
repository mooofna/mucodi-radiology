import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="MuCoDi: 3D radiology multi-teacher knowledge-distillation training"
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet3d_b0',
                        help='3D EfficientNet backbone; see the registry in models/efficientnet3d.py')
    model_group.add_argument('--teacher-dims', type=str, default='',
                        help="teachers and dims as 'name:dim,...' (default: from the dataset profile)")
    model_group.add_argument('--dataset', type=str, default='ctrate_kd',
                        help='dataset profile name from dataprep/datasets.yaml')
    model_group.add_argument('--num-layers', type=int, default=1,
                        help='number of per-teacher projection layers')
    model_group.add_argument('--projector-arch', type=str, default='linear',
                        choices=['linear', 'mlp_dt'],
                        help='per-teacher head: "linear", or "mlp_dt" (2-layer MLP with hidden=d_t)')
    model_group.add_argument('--projector-dropout', type=float, default=0.0,
                        help='dropout on the pooled backbone feature before the per-teacher heads')
    model_group.add_argument('--moco-t', default=0.2, type=float, help='InfoNCE softmax temperature')
    model_group.add_argument('--neg-source', default='bank', type=str,
                        choices=['gather', 'bank', 'bank-full'],
                        help="InfoNCE negatives: 'gather' (in-batch all_gather), 'bank' (sample "
                             "--neg-bank-size from the frozen teacher bank), or 'bank-full' (whole corpus)")
    model_group.add_argument('--neg-bank-size', default=16384, type=int,
                        help='negatives sampled per micro-batch from the frozen bank (--neg-source bank)')
    model_group.add_argument('--neg-bank-dtype', default='float32', type=str,
                        choices=['float16', 'float32'],
                        help='dtype of the in-GPU frozen key bank')
    model_group.add_argument('--neg-mask-false-negatives', default='study', type=str,
                        choices=['off', 'self', 'study'],
                        help="mask sampled negatives colliding with a query's positive: 'self' (same "
                             "sample), 'study' (same-study scans), or 'off'")

    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='base learning rate', dest='lr')
    opt_group.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                        metavar='W', help='weight decay', dest='weight_decay')
    opt_group.add_argument("--train-steps", type=int, default=73700,
                        help="total number of optimizer steps (global budget)")
    opt_group.add_argument("--start-step", type=int, default=0, help="starting step (for restarts)")
    opt_group.add_argument('--warmup-steps', type=int, default=3685,
                        help='number of warmup steps (absolute count)')
    opt_group.add_argument('--clip-grad', default=1.0, type=float,
                        help='gradient clipping max norm (0 disables)')
    opt_group.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                        help='automatic mixed precision (bfloat16)')
    opt_group.add_argument('--grad-accum-steps', default=1, type=int,
                        help='gradient accumulation steps (effective batch = batch_size * grad_accum_steps * world_size)')
    opt_group.add_argument('--betas', default='0.9,0.95', type=str,
                        help='AdamW betas, comma-separated')

    sys_group = parser.add_argument_group("System & Data")
    sys_group.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data-loading workers')
    sys_group.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='per-GPU mini-batch size')
    sys_group.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint to resume from')
    sys_group.add_argument('--auto-resume', action='store_true', default=False,
                        help='if --resume is empty, resume from the newest step_*.pth.tar in --save-dir')
    sys_group.add_argument("--save-every", type=int, default=500,
                        help="save a checkpoint every this many steps")
    sys_group.add_argument("--save-dir", type=str, default="checkpoints",
                        help="checkpoint directory")
    sys_group.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                        help='print frequency')
    sys_group.add_argument('--seed', default=42, type=int, help='random seed')
    sys_group.add_argument('--gpu', default=None, type=int, help='GPU id to use')

    dist_group = parser.add_argument_group("Distributed")
    dist_group.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    dist_group.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    dist_group.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    dist_group.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    dist_group.add_argument('--multiprocessing-distributed', action='store_true',
                        help='launch one process per GPU per node')

    return parser


def parse_teacher_dims(dims_str):
    """Parse 'pillar0_chest_ct:1152,curia1:768' into {'pillar0_chest_ct': 1152, ...}."""
    d = {}
    for pair in dims_str.split(','):
        if not pair.strip():
            continue
        k, v = pair.split(':')
        d[k.strip()] = int(v)
    return d
