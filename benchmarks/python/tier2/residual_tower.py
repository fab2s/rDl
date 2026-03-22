"""Residual tower benchmark: 8 layers with skip connections.

PyTorch equivalent of flodl's `.also()` — manual `x = x + block(x)`.
"""

import torch
import torch.nn as nn


DIM = 256
NUM_BLOCKS = 8


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.GELU(),
            nn.LayerNorm(DIM),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Linear(DIM, DIM)
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(NUM_BLOCKS)])
        self.output = nn.Linear(DIM, DIM)

    def forward(self, x):
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def run(device, batches_per_epoch=100, batch_size=128, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = ResidualTower().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, DIM, device=device),
         torch.randn(batch_size, DIM, device=device))
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("residual_tower", model, batches, device, optimizer, **kwargs)
