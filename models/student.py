import torch
import torch.nn as nn

def make_projector(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Module:
    """Per-teacher projection head (linear if num_layers<=1, else MoCo-v3 MLP)."""
    if num_layers <= 1:
        return nn.Linear(in_dim, out_dim, bias=True)

    layers = []
    layers.extend([
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True)
    ])

    for _ in range(num_layers - 2):
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        ])

    # final projection: no BN/ReLU
    layers.append(nn.Linear(hidden_dim, out_dim, bias=True))
    
    return nn.Sequential(*layers)

class MultiTeacherStudent(nn.Module):
    """Shared backbone + one projection head per teacher for multi-teacher KD."""
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        teacher_dims: dict,
        hidden: int,
        num_layers: int,
        projector_arch: str = "linear",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # ModuleDict registers each head for DDP + optimizer
        self.projectors = nn.ModuleDict()

        for name, t_dim in teacher_dims.items():
            if projector_arch == "mlp_dt":
                proj = make_projector(
                    in_dim=feat_dim,
                    hidden_dim=t_dim,
                    out_dim=t_dim,
                    num_layers=2,
                )
            elif projector_arch == "linear":
                proj = make_projector(
                    in_dim=feat_dim,
                    hidden_dim=hidden,
                    out_dim=t_dim,
                    num_layers=num_layers,
                )
            else:
                raise ValueError(
                    f"Unknown projector_arch={projector_arch!r}. "
                    f"Expected one of: 'linear', 'mlp_dt'."
                )
            self.projectors[name] = proj

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass -> {teacher_name: [B, teacher_dim]} (normalized in the loss)."""
        f = self.backbone(x)

        f = self.dropout(f)

        outputs = {}
        for name, proj in self.projectors.items():
            # heads emit unnormalized vectors; the loss normalizes
            outputs[name] = proj(f)

        return outputs