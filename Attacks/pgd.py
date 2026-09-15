import torch
import torch.nn.functional as F


def pgd_attack(
    model,
    images,
    labels,
    epsilon=8 / 255,
    alpha=2 / 255,
    steps=10,
    mean=None,
    std=None,
    random_start=True,
):
    """
    Untargeted L-infinity PGD attack.

    images:
        Normalized images given to the classifier.

    epsilon:
        Maximum allowed perturbation in pixel space.

    alpha:
        Perturbation added during each PGD iteration.

    steps:
        Number of attack iterations.

    random_start:
        Start from a random point inside the epsilon region.
    """

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if steps < 1:
        raise ValueError("steps must be at least 1")

    if mean is None or std is None:
        raise ValueError("mean and std are required")

    if torch.any(std <= 0):
        raise ValueError("standard deviation values must be positive")

    # Convert normalized images back to pixel range [0, 1].
    clean_pixels = images.detach() * std + mean
    clean_pixels = torch.clamp(clean_pixels, 0.0, 1.0)

    # Random initialization inside the epsilon region.
    if random_start and epsilon > 0:
        random_noise = torch.empty_like(clean_pixels).uniform_(
            -epsilon,
            epsilon,
        )

        adversarial_pixels = clean_pixels + random_noise
        adversarial_pixels = torch.clamp(
            adversarial_pixels,
            0.0,
            1.0,
        )
    else:
        adversarial_pixels = clean_pixels.clone()

    for _ in range(steps):
        adversarial_pixels.requires_grad_(True)

        # Normalize before giving the image to the classifier.
        normalized_images = (
            adversarial_pixels - mean
        ) / std

        predictions = model(normalized_images)

        loss = F.cross_entropy(
            predictions,
            labels,
        )

        gradient = torch.autograd.grad(
            loss,
            adversarial_pixels,
            only_inputs=True,
        )[0]

        with torch.no_grad():
            # Move in the direction that increases classification loss.
            adversarial_pixels = (
                adversarial_pixels
                + alpha * gradient.sign()
            )

            # Project perturbation back into the epsilon region.
            perturbation = torch.clamp(
                adversarial_pixels - clean_pixels,
                min=-epsilon,
                max=epsilon,
            )

            # Keep the final image inside the valid pixel range.
            adversarial_pixels = torch.clamp(
                clean_pixels + perturbation,
                0.0,
                1.0,
            )

    # Return normalized adversarial images.
    adversarial_images = (
        adversarial_pixels - mean
    ) / std

    return adversarial_images.detach()