import torch
import torch.nn.functional as F


def inverse_tanh(x):
    """
    Numerically stable inverse tanh.
    """
    x = torch.clamp(x, -0.999999, 0.999999)

    return 0.5 * torch.log(
        (1 + x) / (1 - x)
    )


def cw_l2_attack(
    model,
    images,
    labels,
    mean,
    std,
    c=1.0,
    kappa=0.0,
    steps=50,
    learning_rate=0.01,
):
    """
    Untargeted Carlini and Wagner L2 attack.

    Parameters
    ----------
    model:
        Victim image-classification model.

    images:
        Normalized model inputs.

    labels:
        Ground-truth labels.

    mean, std:
        Normalization tensors with shape [1, 3, 1, 1].

    c:
        Controls the balance between image similarity
        and attack success.

    kappa:
        Required confidence margin for misclassification.

    steps:
        Number of optimization iterations.

    learning_rate:
        Adam optimizer learning rate.

    Returns
    -------
    Normalized adversarial images.
    """

    if c <= 0:
        raise ValueError("c must be positive")

    if kappa < 0:
        raise ValueError("kappa cannot be negative")

    if steps < 1:
        raise ValueError("steps must be at least 1")

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive"
        )

    if torch.any(std <= 0):
        raise ValueError(
            "standard deviation must be positive"
        )

    device = images.device
    batch_size = images.size(0)

    # Convert normalized model inputs to pixel space.
    clean_pixels = images.detach() * std + mean
    clean_pixels = torch.clamp(
        clean_pixels,
        0.0,
        1.0,
    )

    # Convert image to tanh space.
    #
    # tanh(w) gives values in [-1, 1].
    # We then convert them to [0, 1].
    w = inverse_tanh(
        clean_pixels * 2 - 1
    ).detach()

    w.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [w],
        lr=learning_rate,
    )

    # Store the lowest-distortion successful
    # adversarial image for every sample.
    best_adversarial = clean_pixels.clone()

    best_l2 = torch.full(
        (batch_size,),
        float("inf"),
        device=device,
    )

    attack_found = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    final_adversarial = clean_pixels.clone()

    for _ in range(steps):
        # Convert optimization variable back to pixels.
        adversarial_pixels = (
            torch.tanh(w) + 1
        ) / 2

        normalized_adversarial = (
            adversarial_pixels - mean
        ) / std

        logits = model(normalized_adversarial)

        # L2 distance for every image.
        l2_distance = (
            adversarial_pixels - clean_pixels
        ).pow(2).flatten(1).sum(dim=1)

        # Logit belonging to the correct class.
        correct_logits = logits.gather(
            1,
            labels.view(-1, 1),
        ).squeeze(1)

        # Find the strongest incorrect-class logit.
        one_hot_labels = F.one_hot(
            labels,
            num_classes=logits.size(1),
        ).bool()

        incorrect_logits = logits.masked_fill(
            one_hot_labels,
            float("-inf"),
        ).max(dim=1).values

        # Untargeted C&W objective:
        #
        # correct_logit should become smaller than
        # the strongest incorrect logit.
        classification_loss = torch.clamp(
            correct_logits
            - incorrect_logits
            + kappa,
            min=0,
        )

        total_loss = (
            l2_distance
            + c * classification_loss
        ).sum()

        optimizer.zero_grad()

        gradient = torch.autograd.grad(
            total_loss,
            w,
            only_inputs=True,
        )[0]

        w.grad = gradient
        optimizer.step()

        with torch.no_grad():
            final_adversarial = (
                torch.tanh(w) + 1
            ) / 2

            final_normalized = (
                final_adversarial - mean
            ) / std

            predictions = model(
                final_normalized
            ).argmax(dim=1)

            successful = predictions.ne(labels)

            current_l2 = (
                final_adversarial - clean_pixels
            ).pow(2).flatten(1).sum(dim=1)

            improved = (
                successful
                & (current_l2 < best_l2)
            )

            best_l2[improved] = current_l2[improved]

            best_adversarial[improved] = (
                final_adversarial[improved]
            )

            attack_found |= successful

    # If C&W failed for a sample, return the final
    # optimized candidate instead of the clean image.
    result_pixels = torch.where(
        attack_found.view(-1, 1, 1, 1),
        best_adversarial,
        final_adversarial,
    )

    result_pixels = torch.clamp(
        result_pixels,
        0.0,
        1.0,
    )

    result_normalized = (
        result_pixels - mean
    ) / std

    return result_normalized.detach()