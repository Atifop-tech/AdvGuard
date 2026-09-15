import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from attacks import cw_l2_attack
from model import create_resnet18_cifar10


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
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

    if first_key.startswith("module."):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)


def evaluate_cw(
    model,
    data_loader,
    device,
    c,
    kappa,
    steps,
    learning_rate,
    max_samples=None,
):
    mean = torch.tensor(
        CIFAR10_MEAN,
        device=device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        CIFAR10_STD,
        device=device,
    ).view(1, 3, 1, 1)

    total_samples = 0
    clean_correct = 0
    adversarial_correct = 0
    successful_attacks = 0

    total_clean_confidence = 0.0
    total_adversarial_confidence = 0.0
    total_l2 = 0.0

    progress_bar = tqdm(
        data_loader,
        desc="Running C&W attack",
    )

    for images, labels in progress_bar:
        if max_samples is not None:
            remaining = max_samples - total_samples

            if remaining <= 0:
                break

            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device)
        labels = labels.to(device)

        # Clean predictions
        with torch.no_grad():
            clean_outputs = model(images)

            clean_probabilities = torch.softmax(
                clean_outputs,
                dim=1,
            )

            (
                clean_confidences,
                clean_predictions,
            ) = clean_probabilities.max(dim=1)

        # Generate C&W adversarial images
        adversarial_images = cw_l2_attack(
            model=model,
            images=images,
            labels=labels,
            mean=mean,
            std=std,
            c=c,
            kappa=kappa,
            steps=steps,
            learning_rate=learning_rate,
        )

        # Adversarial predictions
        with torch.no_grad():
            adversarial_outputs = model(
                adversarial_images
            )

            adversarial_probabilities = (
                torch.softmax(
                    adversarial_outputs,
                    dim=1,
                )
            )

            (
                adversarial_confidences,
                adversarial_predictions,
            ) = adversarial_probabilities.max(
                dim=1
            )

        initially_correct = (
            clean_predictions.eq(labels)
        )

        attack_succeeded = (
            initially_correct
            & adversarial_predictions.ne(labels)
        )

        clean_pixels = images * std + mean

        adversarial_pixels = (
            adversarial_images * std + mean
        )

        l2_values = (
            adversarial_pixels - clean_pixels
        ).pow(2).flatten(1).sum(dim=1).sqrt()

        batch_size = labels.size(0)

        total_samples += batch_size

        clean_correct += (
            initially_correct.sum().item()
        )

        adversarial_correct += (
            adversarial_predictions.eq(labels)
            .sum()
            .item()
        )

        successful_attacks += (
            attack_succeeded.sum().item()
        )

        total_clean_confidence += (
            clean_confidences.sum().item()
        )

        total_adversarial_confidence += (
            adversarial_confidences.sum().item()
        )

        total_l2 += l2_values.sum().item()

    if total_samples == 0:
        raise ValueError(
            "No samples were evaluated"
        )

    clean_accuracy = (
        clean_correct / total_samples
    )

    adversarial_accuracy = (
        adversarial_correct / total_samples
    )

    if clean_correct > 0:
        attack_success_rate = (
            successful_attacks / clean_correct
        )
    else:
        attack_success_rate = 0.0

    return {
        "c": c,
        "kappa": kappa,
        "steps": steps,
        "learning_rate": learning_rate,
        "evaluated_samples": total_samples,
        "clean_correct": clean_correct,
        "adversarial_correct":
            adversarial_correct,
        "successful_attacks":
            successful_attacks,
        "clean_accuracy":
            clean_accuracy,
        "adversarial_accuracy":
            adversarial_accuracy,
        "attack_success_rate":
            attack_success_rate,
        "mean_clean_confidence":
            total_clean_confidence / total_samples,
        "mean_adversarial_confidence":
            total_adversarial_confidence
            / total_samples,
        "mean_l2":
            total_l2 / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate C&W L2 attack on CIFAR-10"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cw_metrics.csv"),
    )

    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--kappa",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    if args.c <= 0:
        raise ValueError("c must be positive")

    if args.kappa < 0:
        raise ValueError(
            "kappa cannot be negative"
        )

    if args.steps < 1:
        raise ValueError(
            "steps must be at least 1"
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "learning rate must be positive"
        )

    if (
        args.max_samples is not None
        and args.max_samples < 1
    ):
        raise ValueError(
            "max-samples must be positive"
        )

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_resnet18_cifar10().to(
        device
    )

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model.eval()

    metrics = evaluate_cw(
        model=model,
        data_loader=test_loader,
        device=device,
        c=args.c,
        kappa=args.kappa,
        steps=args.steps,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metrics.keys(),
        )

        writer.writeheader()
        writer.writerow(metrics)

    print("\nC&W evaluation completed")
    print("------------------------")

    print(
        f"Evaluated samples: "
        f"{metrics['evaluated_samples']}"
    )

    print(
        f"Clean accuracy: "
        f"{metrics['clean_accuracy']:.2%}"
    )

    print(
        f"Adversarial accuracy: "
        f"{metrics['adversarial_accuracy']:.2%}"
    )

    print(
        f"Successful attacks: "
        f"{metrics['successful_attacks']}"
    )

    print(
        f"Attack Success Rate: "
        f"{metrics['attack_success_rate']:.2%}"
    )

    print(
        f"Mean clean confidence: "
        f"{metrics['mean_clean_confidence']:.2%}"
    )

    print(
        f"Mean adversarial confidence: "
        f"{metrics['mean_adversarial_confidence']:.2%}"
    )

    print(
        f"Mean L2 perturbation: "
        f"{metrics['mean_l2']:.6f}"
    )

    print(
        f"\nResults saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()