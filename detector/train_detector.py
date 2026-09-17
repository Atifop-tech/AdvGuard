import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from detector.detector_dataset import (
    AdversarialDetectorDataset,
)

from detector.detector_model import (
    AdversarialDetector,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(
    labels,
    probabilities,
    threshold=0.5,
):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    true_negative = matrix[0, 0]
    false_positive = matrix[0, 1]
    false_negative = matrix[1, 0]
    true_positive = matrix[1, 1]

    false_positive_rate = (
        false_positive
        / (false_positive + true_negative)
        if false_positive + true_negative > 0
        else 0.0
    )

    false_negative_rate = (
        false_negative
        / (false_negative + true_positive)
        if false_negative + true_positive > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "false_positive_rate":
            false_positive_rate,
        "false_negative_rate":
            false_negative_rate,
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_probabilities = []

    progress_bar = tqdm(
        data_loader,
        desc="Training",
        leave=False,
    )

    for batch in progress_bar:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(logits)

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .tolist()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss / len(data_loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
    )

    return epoch_loss, metrics


@torch.no_grad()
def validate(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_probabilities = []

    progress_bar = tqdm(
        data_loader,
        desc="Validation",
        leave=False,
    )

    for batch in progress_bar:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        probabilities = torch.sigmoid(logits)

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().tolist()
        )

    validation_loss = (
        running_loss / len(data_loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
    )

    return (
        validation_loss,
        metrics,
        all_labels,
        all_probabilities,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the adversarial-image detector"
        )
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path(
            "adversarial_dataset/train.pt"
        ),
    )

    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path(
            "adversarial_dataset/validation.pt"
        ),
    )

    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path(
            "model/adversarial_detector.pth"
        ),
    )

    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path(
            "results/detector_training_history.csv"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
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
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    train_dataset = (
        AdversarialDetectorDataset(
            args.train_data
        )
    )

    validation_dataset = (
        AdversarialDetectorDataset(
            args.validation_data
        )
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(validation_dataset),
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    model = AdversarialDetector().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    args.output_model.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.history_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_auc = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = (
            train_one_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        (
            validation_loss,
            validation_metrics,
            validation_labels,
            validation_probabilities,
        ) = validate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        validation_auc = (
            validation_metrics["roc_auc"]
        )

        scheduler.step(validation_auc)

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "learning_rate":
                current_learning_rate,
            "train_loss": train_loss,
            "train_accuracy":
                train_metrics["accuracy"],
            "train_f1":
                train_metrics["f1"],
            "train_roc_auc":
                train_metrics["roc_auc"],
            "validation_loss":
                validation_loss,
            "validation_accuracy":
                validation_metrics["accuracy"],
            "validation_precision":
                validation_metrics["precision"],
            "validation_recall":
                validation_metrics["recall"],
            "validation_f1":
                validation_metrics["f1"],
            "validation_roc_auc":
                validation_auc,
            "validation_false_positive_rate":
                validation_metrics[
                    "false_positive_rate"
                ],
            "validation_false_negative_rate":
                validation_metrics[
                    "false_negative_rate"
                ],
        }

        history.append(row)

        print(
            f"\nEpoch {epoch:02d}/{args.epochs}"
        )

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_metrics['accuracy']:.2%} | "
            f"Train AUC: "
            f"{train_metrics['roc_auc']:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f} | "
            f"Validation accuracy: "
            f"{validation_metrics['accuracy']:.2%}"
        )

        print(
            f"Precision: "
            f"{validation_metrics['precision']:.2%} | "
            f"Recall: "
            f"{validation_metrics['recall']:.2%} | "
            f"F1: "
            f"{validation_metrics['f1']:.2%} | "
            f"AUC: {validation_auc:.4f}"
        )

        print(
            f"False positive rate: "
            f"{validation_metrics['false_positive_rate']:.2%} | "
            f"False negative rate: "
            f"{validation_metrics['false_negative_rate']:.2%}"
        )

        if validation_auc > best_auc:
            best_auc = validation_auc
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),
                "best_validation_auc":
                    best_auc,
                "epoch": epoch,
                "threshold": 0.5,
                "architecture":
                    "AdversarialDetector",
            }

            torch.save(
                checkpoint,
                args.output_model,
            )

            print(
                "Best detector checkpoint saved."
            )

        else:
            epochs_without_improvement += 1

            print(
                "Epochs without improvement:",
                epochs_without_improvement,
            )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "\nEarly stopping activated."
            )
            break

    with args.history_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)

    print("\nTraining completed.")

    print(
        f"Best validation ROC-AUC: "
        f"{best_auc:.4f}"
    )

    print(
        f"Best model saved to: "
        f"{args.output_model}"
    )

    print(
        f"Training history saved to: "
        f"{args.history_output}"
    )


if __name__ == "__main__":
    main()