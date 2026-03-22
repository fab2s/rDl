"""ConvNet benchmark: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Linear.

Deeper network on 64x64 images with wider channels.
"""

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 64x64 -> 32x32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32x32 -> 16x16
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 16x16 -> 8x8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 8x8 -> 4x4
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def run(device, batches_per_epoch=50, batch_size=128, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = ConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, 3, 64, 64, device=device),
         torch.randn(batch_size, 100, device=device))
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("convnet", model, batches, device, optimizer, **kwargs)
