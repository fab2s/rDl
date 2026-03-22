"""MLP benchmark: Linear -> GELU -> LayerNorm, 3 layers."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        return self.net(x)


def run(device, batches_per_epoch=100, batch_size=128, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, 256, device=device),
         torch.randn(batch_size, 256, device=device))
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("mlp", model, batches, device, optimizer, **kwargs)
