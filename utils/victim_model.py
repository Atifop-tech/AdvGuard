from pathlib import Path

import torch

from model import create_resnet18_cifar10


def load_victim_model(
    checkpoint_path,
    device,
):
    """
    Load and freeze the trained CIFAR-10 ResNet-18.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = create_resnet18_cifar10()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )
    else:
        state_dict = checkpoint

    first_key = next(iter(state_dict))

    # Handle DataParallel checkpoints.
    if first_key.startswith("module."):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Freeze the victim model.
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model