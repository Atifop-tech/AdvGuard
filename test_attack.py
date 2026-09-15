import os

import torch
import torch.nn as nn

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "./models/resnet18_cifar10.pth"

EPSILON = 8 / 255


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# CIFAR-10 normalization values
MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465]
).view(1, 3, 1, 1).to(DEVICE)

STD = torch.tensor(
    [0.2470, 0.2435, 0.2616]
).view(1, 3, 1, 1).to(DEVICE)


# ============================================================
# MODEL
# ============================================================

def load_model():

    model = models.resnet18(weights=None)

    # Same ResNet modification used during training
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(image):

    return (image - MEAN) / STD


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image):

    with torch.no_grad():

        output = model(
            normalize(image)
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return (
        prediction.item(),
        confidence.item()
    )


# ============================================================
# FGSM ATTACK
# ============================================================

def fgsm_attack(
    model,
    image,
    label,
    epsilon
):

    # Make a copy and enable gradients
    adversarial_image = image.clone().detach()

    adversarial_image.requires_grad = True


    # Run image through the victim model
    output = model(
        normalize(adversarial_image)
    )


    # Calculate classification loss
    loss = nn.CrossEntropyLoss()(
        output,
        label
    )


    # Remove old gradients
    model.zero_grad()


    # Calculate gradient
    loss.backward()


    # Direction that increases the classification loss
    gradient_sign = (
        adversarial_image.grad.sign()
    )


    # Add small perturbation
    adversarial_image = (
        adversarial_image
        + epsilon * gradient_sign
    )


    # Ensure image remains valid
    adversarial_image = torch.clamp(
        adversarial_image,
        0,
        1
    )


    return adversarial_image.detach()


# ============================================================
# DISPLAY FUNCTION
# ============================================================

def show_image(
    image,
    title
):

    image = (
        image.squeeze(0)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print(" FGSM Attack on ResNet-18 / CIFAR-10")
    print("========================================")

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"FGSM epsilon: {EPSILON:.6f}"
    )


    # --------------------------------------------------------
    # Load victim model
    # --------------------------------------------------------

    model = load_model()

    print(
        "\nVictim model loaded successfully."
    )


    # --------------------------------------------------------
    # Load CIFAR-10
    # --------------------------------------------------------

    transform = transforms.ToTensor()

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )


    # --------------------------------------------------------
    # Find an image the model initially classifies correctly
    # --------------------------------------------------------

    selected_image = None
    selected_label = None
    selected_prediction = None
    selected_confidence = None


    for index in range(len(test_dataset)):

        image, label = test_dataset[index]

        image = image.unsqueeze(0).to(DEVICE)

        prediction, confidence = predict(
            model,
            image
        )

        if prediction == label:

            selected_image = image

            selected_label = torch.tensor(
                [label],
                device=DEVICE
            )

            selected_prediction = prediction

            selected_confidence = confidence

            break


    if selected_image is None:

        raise RuntimeError(
            "Could not find a correctly classified image."
        )


    print()
    print(
        "Original class:",
        CLASS_NAMES[
            selected_prediction
        ]
    )

    print(
        "Original confidence:",
        f"{selected_confidence * 100:.2f}%"
    )


    # --------------------------------------------------------
    # FGSM ATTACK
    # --------------------------------------------------------

    adversarial_image = fgsm_attack(
        model=model,
        image=selected_image,
        label=selected_label,
        epsilon=EPSILON,
    )


    # --------------------------------------------------------
    # Predict adversarial image
    # --------------------------------------------------------

    adversarial_prediction, adversarial_confidence = (
        predict(
            model,
            adversarial_image
        )
    )


    print()
    print(
        "After FGSM:",
        CLASS_NAMES[
            adversarial_prediction
        ]
    )

    print(
        "Confidence:",
        f"{adversarial_confidence * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Attack result
    # --------------------------------------------------------

    attack_successful = (
        adversarial_prediction
        != selected_label.item()
    )


    print()

    if attack_successful:

        print(
            "ATTACK SUCCESSFUL!"
        )

    else:

        print(
            "Attack did not change the prediction."
        )


    # --------------------------------------------------------
    # Display images
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 4)
    )


    plt.subplot(
        1,
        2,
        1
    )

    show_image(
        selected_image,
        (
            f"Original\n"
            f"{CLASS_NAMES[selected_prediction]}"
            f" ({selected_confidence * 100:.1f}%)"
        )
    )


    plt.subplot(
        1,
        2,
        2
    )

    show_image(
        adversarial_image,
        (
            f"FGSM\n"
            f"{CLASS_NAMES[adversarial_prediction]}"
            f" ({adversarial_confidence * 100:.1f}%)"
        )
    )


    plt.tight_layout()

    plt.show()