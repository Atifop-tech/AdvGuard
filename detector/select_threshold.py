import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from detector.detector_dataset import (
    AdversarialDetectorDataset,
)

from detector.detector_model import (
    AdversarialDetector,
)


@torch.no_grad()
def collect_probabilities(
    model,
    data_loader,
    device,
):
    model.eval()

    labels = []
    probabilities = []

    for batch in data_loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        batch_labels = batch["label"]

        logits = model(images)
        batch_probabilities = torch.sigmoid(logits)

        labels.extend(
            batch_labels.cpu().numpy().tolist()
        )

        probabilities.extend(
            batch_probabilities
            .cpu()
            .numpy()
            .tolist()
        )

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities),
    )


def calculate_threshold_metrics(
    labels,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    true_positive = int(
        ((predictions == 1) & (labels == 1)).sum()
    )

    true_negative = int(
        ((predictions == 0) & (labels == 0)).sum()
    )

    false_positive = int(
        ((predictions == 1) & (labels == 0)).sum()
    )

    false_negative = int(
        ((predictions == 0) & (labels == 1)).sum()
    )

    accuracy = (
        true_positive + true_negative
    ) / len(labels)

    precision = (
        true_positive
        / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

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

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    youden_j = recall - false_positive_rate

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate":
            false_positive_rate,
        "false_negative_rate":
            false_negative_rate,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "youden_j": youden_j,
    }


def print_result(title, result):
    print(f"\n{title}")
    print("-" * 50)

    print(
        f"Threshold: {result['threshold']:.2f}"
    )

    print(
        f"Accuracy: {result['accuracy']:.2%}"
    )

    print(
        f"Precision: {result['precision']:.2%}"
    )

    print(
        f"Recall: {result['recall']:.2%}"
    )

    print(
        f"F1 score: {result['f1']:.2%}"
    )

    print(
        f"False-positive rate: "
        f"{result['false_positive_rate']:.2%}"
    )

    print(
        f"False-negative rate: "
        f"{result['false_negative_rate']:.2%}"
    )

    print(
        "Confusion matrix:"
    )

    print(
        f"TN={result['true_negative']}  "
        f"FP={result['false_positive']}  "
        f"FN={result['false_negative']}  "
        f"TP={result['true_positive']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select an adversarial detector threshold"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "model/adversarial_detector.pth"
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
        "--output",
        type=Path,
        default=Path(
            "results/threshold_analysis.csv"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Detector checkpoint not found: "
            f"{args.checkpoint}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = AdversarialDetectorDataset(
        args.validation_data
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model = AdversarialDetector().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    labels, probabilities = collect_probabilities(
        model,
        data_loader,
        device,
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    thresholds = np.arange(
        0.05,
        0.951,
        0.01,
    )

    results = [
        calculate_threshold_metrics(
            labels,
            probabilities,
            threshold,
        )
        for threshold in thresholds
    ]

    default_result = min(
        results,
        key=lambda row: abs(
            row["threshold"] - 0.50
        ),
    )

    best_f1_result = max(
        results,
        key=lambda row: row["f1"],
    )

    best_youden_result = max(
        results,
        key=lambda row: row["youden_j"],
    )

    eligible_results = [
        row
        for row in results
        if row["false_positive_rate"]
        <= args.max_fpr
    ]

    if not eligible_results:
        raise RuntimeError(
            "No threshold satisfies the maximum "
            "false-positive-rate constraint"
        )

    # Security-oriented recommendation:
    # maximize recall while respecting max FPR.
    recommended_result = max(
        eligible_results,
        key=lambda row: (
            row["recall"],
            row["f1"],
            row["precision"],
        ),
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
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nValidation ROC-AUC: {roc_auc:.6f}"
    )

    print_result(
        "DEFAULT THRESHOLD",
        default_result,
    )

    print_result(
        "BEST F1 THRESHOLD",
        best_f1_result,
    )

    print_result(
        "BEST YOUDEN J THRESHOLD",
        best_youden_result,
    )

    print_result(
        (
            "RECOMMENDED SECURITY THRESHOLD "
            f"(FPR <= {args.max_fpr:.2%})"
        ),
        recommended_result,
    )

    print(
        f"\nFull threshold analysis saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()