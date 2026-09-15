import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18_cifar10():
    model = resnet18(weights=None)

    # Modification for 32×32 CIFAR-10 images
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove the original ImageNet max-pooling layer
    model.maxpool = nn.Identity()

    # CIFAR-10 contains 10 classes
    model.fc = nn.Linear(
        model.fc.in_features,
        10,
    )

    return model