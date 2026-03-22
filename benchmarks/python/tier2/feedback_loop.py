"""Feedback loop benchmark: encoder -> adaptive loop with learned halt -> decoder.

PyTorch equivalent of flodl's `.loop_body().until_cond()` — a while-loop
in forward() with a halt network, manually threading state and managing
gradient flow through iterations.

This is the pattern that motivated flodl's existence: recursive feedback
loops are painful to write correctly in PyTorch.
"""

import torch
import torch.nn as nn


DIM = 128
MAX_ITER = 8


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


class HaltNet(nn.Module):
    """Learned halt condition: projects to scalar, positive = halt."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(DIM, 1)

    def forward(self, x):
        return self.linear(x).mean()  # scalar halt signal


class FeedbackLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.GELU(),
            nn.LayerNorm(DIM),
        )
        self.refine = RefineBlock()
        self.halt = HaltNet()
        self.decoder = nn.Linear(DIM, DIM)

    def forward(self, x):
        x = self.encoder(x)

        # Do-while: body runs at least once, halt checked after
        for _ in range(MAX_ITER):
            x = self.refine(x)
            halt_signal = self.halt(x)
            if halt_signal.item() > 0:
                break

        return self.decoder(x)


def run(device, batches_per_epoch=100, batch_size=64, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = FeedbackLoop().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, DIM, device=device),
         torch.randn(batch_size, DIM, device=device))
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("feedback_loop", model, batches, device, optimizer, **kwargs)
