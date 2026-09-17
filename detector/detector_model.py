import torch
import torch.nn as nn


class AdversarialDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # One raw binary logit.
            nn.Linear(64, 1),
        )

    def forward(self, images):
        features = self.features(images)
        logits = self.classifier(features)

        return logits.squeeze(1)


if __name__ == "__main__":
    model = AdversarialDetector()

    sample = torch.randn(
        8,
        3,
        32,
        32,
    )

    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)

    assert output.shape == (8,)

    print("Detector model test passed.")