"""MLP benchmark: Linear -> GELU -> LayerNorm, 5 layers."""

import torch
import torch.nn as nn


DIM = 1024
HIDDEN = 2048


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, DIM),
        )

    def forward(self, x):
        return self.net(x)


def run(device, batches_per_epoch=50, batch_size=256, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, DIM, device=device),
         torch.randn(batch_size, DIM, device=device))
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("mlp", model, batches, device, optimizer, **kwargs)
