"""Iterative refinement benchmark: encoder -> refinement loop (8x) -> decoder.

PyTorch equivalent of flodl's `.loop_body().for_n()` — a Python for-loop
in forward() calling the same submodule repeatedly.
"""

import torch
import torch.nn as nn


DIM = 1024
REFINE_STEPS = 8


class RefineBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.GELU(),
            nn.LayerNorm(DIM),
        )

    def forward(self, x):
        return x + self.net(x)


class IterativeRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.GELU(),
            nn.LayerNorm(DIM),
        )
        self.refine = RefineBlock()
        self.decoder = nn.Linear(DIM, DIM)

    def forward(self, x):
        x = self.encoder(x)
        for _ in range(REFINE_STEPS):
            x = self.refine(x)
        return self.decoder(x)


def run(device, batches_per_epoch=50, batch_size=256, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = IterativeRefine().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, DIM, device=device),
         torch.randn(batch_size, DIM, device=device))
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("iterative_refine", model, batches, device, optimizer, **kwargs)
