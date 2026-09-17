import torch
import torch.nn.functional as F


def fgsm_attack(
    model,
    images,
    labels,
    epsilon=8 / 255,
    mean=None,
    std=None,
):
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    if mean is None or std is None:
        raise ValueError("mean and std are required")

    if torch.any(std <= 0):
        raise ValueError(
            "standard deviation values must be positive"
        )

    normalized_images = (
        images.detach().clone().requires_grad_(True)
    )

    predictions = model(normalized_images)

    loss = F.cross_entropy(
        predictions,
        labels,
    )

    gradient = torch.autograd.grad(
        loss,
        normalized_images,
        only_inputs=True,
    )[0]

    clean_pixels = (
        normalized_images.detach() * std + mean
    )

    clean_pixels = torch.clamp(
        clean_pixels,
        0.0,
        1.0,
    )

    adversarial_pixels = (
        clean_pixels
        + epsilon * gradient.sign()
    )

    adversarial_pixels = torch.clamp(
        adversarial_pixels,
        0.0,
        1.0,
    )

    adversarial_images = (
        adversarial_pixels - mean
    ) / std

    return adversarial_images.detach()