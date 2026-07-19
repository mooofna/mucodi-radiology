import torch
import os
import math

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, meters, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def display(self, step_idx):
        entries = [f"{self.prefix} [{step_idx}]"] + [str(m) for m in self.meters]
        print("\t".join(entries), flush=True)

    def display_summary(self):
        entries = ["Summary:"] + [m.summary() for m in self.meters]
        print(" ".join(entries), flush=True)

class ValueMeter:
    def __init__(self, name, fmt=":d"):
        self.name, self.fmt, self.val = name, fmt, 0
    def update(self, v): self.val = int(v)
    def __str__(self): return f"{self.name} {self.val}"
    def summary(self): return f"{self.name} {self.val}"

def adjust_learning_rate_steps(optimizer, step, args):
    warmup_len = max(1, args.warmup_steps)
    if step < warmup_len:
        lr = args.lr * (step / warmup_len)
    else:
        curr_step = step - warmup_len
        total_steps = args.train_steps - warmup_len
        progress = min(1.0, max(0.0, curr_step / max(1, total_steps)))        
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

def save_checkpoint(model, optimizer, step_idx, args):
    state_dict = model.state_dict()
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    import monai  # stamp MONAI version for eval
    ckpt = {
        "step": step_idx,
        "arch": args.arch,
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "monai_version": monai.__version__,
    }
    os.makedirs(args.save_dir, exist_ok=True)

    filename = os.path.join(args.save_dir, f"step_{step_idx:07d}.pth.tar")
    # atomic write: .tmp then os.replace
    tmp = filename + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, filename)
    print(f"[checkpoint] saved {filename}", flush=True)
