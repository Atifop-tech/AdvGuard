import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from protection.protection_dataset import (
    ProtectionDataset,
    ATTACK_NAMES,
)
from utils.victim_model import load_victim_model


def calculate_accuracy(correct, total):
    if total == 0:
        return 0.0

    return correct / total


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch in progress:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{correct / total:.4f}",
        )

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


@torch.inference_mode()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    group_correct = {
        attack_id: 0
        for attack_id in ATTACK_NAMES
    }

    group_total = {
        attack_id: 0
        for attack_id in ATTACK_NAMES
    }

    successful_correct = 0
    successful_total = 0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for batch in progress:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        attack_ids = batch["attack_id"].to(device)
        attack_success = batch[
            "attack_success"
        ].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        batch_correct = (
            predictions == labels
        )

        correct += batch_correct.sum().item()
        total += batch_size

        for attack_id in ATTACK_NAMES:
            mask = attack_ids == attack_id

            group_total[attack_id] += (
                mask.sum().item()
            )

            group_correct[attack_id] += (
                batch_correct[mask]
                .sum()
                .item()
            )

        successful_mask = (
            (attack_ids != 0)
            & attack_success
        )

        successful_total += (
            successful_mask.sum().item()
        )

        successful_correct += (
            batch_correct[successful_mask]
            .sum()
            .item()
        )

    group_accuracy = {
        attack_id: calculate_accuracy(
            group_correct[attack_id],
            group_total[attack_id],
        )
        for attack_id in ATTACK_NAMES
    }

    robust_correct = sum(
        group_correct[attack_id]
        for attack_id in ATTACK_NAMES
        if attack_id != 0
    )

    robust_total = sum(
        group_total[attack_id]
        for attack_id in ATTACK_NAMES
        if attack_id != 0
    )

    return {
        "loss": running_loss / total,
        "overall_accuracy": (
            correct / total
        ),
        "clean_accuracy": (
            group_accuracy[0]
        ),
        "fgsm_accuracy": (
            group_accuracy[1]
        ),
        "pgd_accuracy": (
            group_accuracy[2]
        ),
        "cw_accuracy": (
            group_accuracy[3]
        ),
        "robust_accuracy": (
            calculate_accuracy(
                robust_correct,
                robust_total,
            )
        ),
        "successful_attack_accuracy": (
            calculate_accuracy(
                successful_correct,
                successful_total,
            )
        ),
    }


def print_validation_metrics(metrics):
    print(
        f"Validation loss: "
        f"{metrics['loss']:.6f}"
    )

    print(
        f"Overall accuracy: "
        f"{metrics['overall_accuracy']:.2%}"
    )

    print(
        f"Clean accuracy: "
        f"{metrics['clean_accuracy']:.2%}"
    )

    print(
        f"FGSM robust accuracy: "
        f"{metrics['fgsm_accuracy']:.2%}"
    )

    print(
        f"PGD robust accuracy: "
        f"{metrics['pgd_accuracy']:.2%}"
    )

    print(
        f"C&W robust accuracy: "
        f"{metrics['cw_accuracy']:.2%}"
    )

    print(
        f"Combined robust accuracy: "
        f"{metrics['robust_accuracy']:.2%}"
    )

    print(
        f"Successful-attack accuracy: "
        f"{metrics['successful_attack_accuracy']:.2%}"
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "validation_metrics": metrics,
        "model_type": (
            "adversarially_trained_resnet18"
        ),
        "classes": 10,
    }

    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune ResNet-18 using clean "
            "and adversarial CIFAR-10 images."
        )
    )

    parser.add_argument(
        "--train-dataset",
        default=(
            "adversarial_dataset/train.pt"
        ),
    )

    parser.add_argument(
        "--validation-dataset",
        default=(
            "adversarial_dataset/"
            "validation.pt"
        ),
    )

    parser.add_argument(
        "--victim-checkpoint",
        default=(
            "model/resnet18_cifar10.pth"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "model/"
            "protected_resnet18_best.pth"
        ),
    )

    parser.add_argument(
        "--history",
        default=(
            "results/"
            "protected_model_history.csv"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    train_dataset = ProtectionDataset(
        args.train_dataset,
        training=True,
    )

    validation_dataset = ProtectionDataset(
        args.validation_dataset,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    model = load_victim_model(
    args.victim_checkpoint,
    device,
    )

    optimizer = AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    )



    model = model.to(device)

    # The victim-model loader freezes parameters for attack generation.
    # Unfreeze them because this model must now be fine-tuned.
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    if trainable_parameters == 0:
        raise RuntimeError(
            "The protected model has no trainable parameters."
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = Path(args.history)
    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []

    print("\nINITIAL MODEL EVALUATION")
    print("-" * 60)

    initial_metrics = evaluate(
        model,
        validation_loader,
        criterion,
        device,
    )

    print_validation_metrics(
        initial_metrics
    )

    for epoch in range(1, args.epochs + 1):
        print(
            f"\nEPOCH {epoch}/{args.epochs}"
        )

        print("-" * 60)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )

        print(
            f"Training loss: "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Training accuracy: "
            f"{train_metrics['accuracy']:.2%}"
        )

        print_validation_metrics(
            validation_metrics
        )

        # Balanced score between clean and
        # combined adversarial accuracy.
        selection_score = (
            validation_metrics[
                "clean_accuracy"
            ]
            + validation_metrics[
                "robust_accuracy"
            ]
        ) / 2.0

        scheduler.step(selection_score)

        history_rows.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_accuracy": (
                train_metrics["accuracy"]
            ),
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_overall_accuracy": (
                validation_metrics[
                    "overall_accuracy"
                ]
            ),
            "validation_clean_accuracy": (
                validation_metrics[
                    "clean_accuracy"
                ]
            ),
            "validation_fgsm_accuracy": (
                validation_metrics[
                    "fgsm_accuracy"
                ]
            ),
            "validation_pgd_accuracy": (
                validation_metrics[
                    "pgd_accuracy"
                ]
            ),
            "validation_cw_accuracy": (
                validation_metrics[
                    "cw_accuracy"
                ]
            ),
            "validation_robust_accuracy": (
                validation_metrics[
                    "robust_accuracy"
                ]
            ),
            "successful_attack_accuracy": (
                validation_metrics[
                    "successful_attack_accuracy"
                ]
            ),
            "selection_score": (
                selection_score
            ),
        })

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                output_path,
                model,
                optimizer,
                epoch,
                validation_metrics,
            )

            print(
                "Saved new best protected "
                "model."
            )

        else:
            epochs_without_improvement += 1

            print(
                "Epochs without improvement: "
                f"{epochs_without_improvement}"
            )

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    history_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(history_rows)

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "\nEarly stopping activated."
            )
            break

    print("\nTRAINING COMPLETED")
    print("-" * 60)
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best selection score: "
        f"{best_score:.2%}"
    )
    print(f"Best model: {output_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()