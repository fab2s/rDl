"""Gated routing (MoE) benchmark: soft routing with 4 expert MLPs.

PyTorch equivalent of flodl's `.gate()` — manual softmax gating
and expert stacking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


DIM = 128
NUM_EXPERTS = 4


class ExpertBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.GELU(),
            nn.LayerNorm(DIM),
        )

    def forward(self, x):
        return self.net(x)


class GatedRouting(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.GELU(),
        )
        self.router = nn.Linear(DIM, NUM_EXPERTS)
        self.experts = nn.ModuleList([ExpertBlock() for _ in range(NUM_EXPERTS)])
        self.output = nn.Linear(DIM, DIM)

    def forward(self, x):
        x = self.entry(x)
        # Soft routing: all experts run, weighted by router output
        weights = F.softmax(self.router(x), dim=-1)  # [B, num_experts]
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, num_experts, DIM]
        x = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [B, DIM]
        return self.output(x)


def run(device, batches_per_epoch=100, batch_size=128, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = GatedRouting().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, DIM, device=device),
         torch.randn(batch_size, DIM, device=device))
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("gated_routing", model, batches, device, optimizer, **kwargs)
