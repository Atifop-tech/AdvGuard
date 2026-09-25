import argparse
import csv
from pathlib import Path

import numpy as np
import torch

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


ATTACK_NAMES = {
    0: "clean",
    1: "fgsm",
    2: "pgd",
    3: "cw",
}


@torch.no_grad()
def collect_predictions(
    model,
    data_loader,
    device,
):
    model.eval()

    labels = []
    probabilities = []
    attack_ids = []
    attack_success = []

    progress_bar = tqdm(
        data_loader,
        desc="Evaluating detector",
    )

    for batch in progress_bar:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)
        batch_probabilities = torch.sigmoid(logits)

        labels.extend(
            batch["label"].cpu().numpy().tolist()
        )

        probabilities.extend(
            batch_probabilities
            .cpu()
            .numpy()
            .tolist()
        )

        attack_ids.extend(
            batch["attack_id"]
            .cpu()
            .numpy()
            .tolist()
        )

        attack_success.extend(
            batch["attack_success"]
            .cpu()
            .numpy()
            .tolist()
        )

    return {
        "labels": np.asarray(
            labels,
            dtype=np.int64,
        ),

        "probabilities": np.asarray(
            probabilities,
            dtype=np.float64,
        ),

        "attack_ids": np.asarray(
            attack_ids,
            dtype=np.int64,
        ),

        "attack_success": np.asarray(
            attack_success,
            dtype=np.bool_,
        ),
    }


def calculate_overall_metrics(
    labels,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

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
        "threshold": threshold,

        "accuracy": accuracy_score(
            labels,
            predictions,
        ),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            labels,
            probabilities,
        ),

        "false_positive_rate":
            false_positive_rate,

        "false_negative_rate":
            false_negative_rate,

        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def calculate_group_metrics(
    probabilities,
    threshold,
    group_name,
):
    if len(probabilities) == 0:
        return {
            "group": group_name,
            "samples": 0,
            "mean_adversarial_probability": 0.0,
            "detected": 0,
            "missed": 0,
            "detection_rate": 0.0,
        }

    predictions = probabilities >= threshold

    detected = int(predictions.sum())
    missed = int(len(predictions) - detected)

    return {
        "group": group_name,
        "samples": len(probabilities),

        "mean_adversarial_probability":
            float(probabilities.mean()),

        "detected": detected,
        "missed": missed,

        "detection_rate":
            detected / len(probabilities),
    }


def print_overall_metrics(metrics):
    print("\nOVERALL TEST RESULTS")
    print("-" * 55)

    print(
        f"Threshold: {metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy: {metrics['accuracy']:.2%}"
    )

    print(
        f"Precision: {metrics['precision']:.2%}"
    )

    print(
        f"Recall: {metrics['recall']:.2%}"
    )

    print(
        f"F1 score: {metrics['f1']:.2%}"
    )

    print(
        f"ROC-AUC: {metrics['roc_auc']:.6f}"
    )

    print(
        f"False-positive rate: "
        f"{metrics['false_positive_rate']:.2%}"
    )

    print(
        f"False-negative rate: "
        f"{metrics['false_negative_rate']:.2%}"
    )

    print("\nConfusion matrix:")

    print(
        f"TN={metrics['true_negative']}  "
        f"FP={metrics['false_positive']}"
    )

    print(
        f"FN={metrics['false_negative']}  "
        f"TP={metrics['true_positive']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the adversarial detector "
            "on the independent test dataset"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "model/adversarial_detector_final.pth"
        ),
    )

    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path(
            "adversarial_dataset/test.pt"
        ),
    )

    parser.add_argument(
        "--overall-output",
        type=Path,
        default=Path(
            "results/detector_test_metrics.csv"
        ),
    )

    parser.add_argument(
        "--group-output",
        type=Path,
        default=Path(
            "results/attack_specific_metrics.csv"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    threshold = float(
        checkpoint.get("threshold", 0.5)
    )

    print(
        "Loaded threshold:",
        threshold,
    )

    model = AdversarialDetector().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    dataset = AdversarialDetectorDataset(
        args.test_data
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    collected = collect_predictions(
        model,
        data_loader,
        device,
    )

    labels = collected["labels"]
    probabilities = collected["probabilities"]
    attack_ids = collected["attack_ids"]
    attack_success = collected["attack_success"]

    overall_metrics = calculate_overall_metrics(
        labels,
        probabilities,
        threshold,
    )

    print_overall_metrics(
        overall_metrics
    )

    group_results = []

    # Clean images: "detected" means falsely flagged.
    clean_mask = attack_ids == 0

    clean_result = calculate_group_metrics(
        probabilities[clean_mask],
        threshold,
        "clean_false_positive",
    )

    group_results.append(clean_result)

    print("\nGROUP RESULTS")
    print("-" * 55)

    print(
        f"Clean falsely flagged: "
        f"{clean_result['detected']}/"
        f"{clean_result['samples']} "
        f"({clean_result['detection_rate']:.2%})"
    )

    # Attack-specific detection.
    for attack_id in [1, 2, 3]:
        attack_name = ATTACK_NAMES[attack_id]

        mask = attack_ids == attack_id

        result = calculate_group_metrics(
            probabilities[mask],
            threshold,
            attack_name,
        )

        group_results.append(result)

        print(
            f"{attack_name.upper()} detected: "
            f"{result['detected']}/"
            f"{result['samples']} "
            f"({result['detection_rate']:.2%})"
        )

    adversarial_mask = labels == 1

    successful_mask = (
        adversarial_mask & attack_success
    )

    failed_mask = (
        adversarial_mask & ~attack_success
    )

    successful_result = calculate_group_metrics(
        probabilities[successful_mask],
        threshold,
        "successful_attacks",
    )

    failed_result = calculate_group_metrics(
        probabilities[failed_mask],
        threshold,
        "failed_attacks",
    )

    group_results.extend([
        successful_result,
        failed_result,
    ])

    print(
        f"Successful attacks detected: "
        f"{successful_result['detected']}/"
        f"{successful_result['samples']} "
        f"({successful_result['detection_rate']:.2%})"
    )

    print(
        f"Failed attacks detected: "
        f"{failed_result['detected']}/"
        f"{failed_result['samples']} "
        f"({failed_result['detection_rate']:.2%})"
    )

    args.overall_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.overall_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=overall_metrics.keys(),
        )

        writer.writeheader()
        writer.writerow(overall_metrics)

    with args.group_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=group_results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(group_results)

    print(
        f"\nOverall results saved to: "
        f"{args.overall_output}"
    )

    print(
        f"Group results saved to: "
        f"{args.group_output}"
    )


if __name__ == "__main__":
    main()
    