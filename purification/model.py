import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 4
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class PurificationNet(nn.Module):
    """
    Small residual image-to-image denoiser for CIFAR-10.

    Input and output are RGB tensors in pixel space [0, 1].
    The network predicts a correction rather than reconstructing
    the complete image from scratch. Zero-initializing the final
    layer makes the initial model an identity mapping.
    """

    def __init__(self, channels=48, blocks=4):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)]
        )
        self.head = nn.Conv2d(channels, 3, 3, padding=1)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.stem(x)
        correction = self.head(self.body(features))
        return torch.clamp(x + correction, 0.0, 1.0)
