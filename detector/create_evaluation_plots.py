import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from detector.detector_dataset import (
    AdversarialDetectorDataset,
)

from detector.detector_model import (
    AdversarialDetector,
)


@torch.no_grad()
def collect_predictions(
    model,
    data_loader,
    device,
):
    model.eval()

    labels = []
    probabilities = []

    progress_bar = tqdm(
        data_loader,
        desc="Collecting predictions",
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

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities),
    )


def create_confusion_matrix(
    labels,
    probabilities,
    threshold,
    output_path,
):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    class_names = [
        "Clean",
        "Adversarial",
    ]

    axis.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Actual class",
        title=(
            "Adversarial Detector Confusion Matrix\n"
            f"Threshold = {threshold:.2f}"
        ),
    )

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    threshold_color = matrix.max() / 2

    for row in range(2):
        for column in range(2):
            count = matrix[row, column]

            percentage = (
                count / row_totals[row, 0]
                if row_totals[row, 0] > 0
                else 0.0
            )

            axis.text(
                column,
                row,
                (
                    f"{count}\n"
                    f"{percentage:.1%}"
                ),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                color=(
                    "white"
                    if count > threshold_color
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_roc_curve(
    labels,
    probabilities,
    threshold,
    output_path,
):
    false_positive_rates, true_positive_rates, _ = (
        roc_curve(
            labels,
            probabilities,
        )
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    true_negative = matrix[0, 0]
    false_positive = matrix[0, 1]
    false_negative = matrix[1, 0]
    true_positive = matrix[1, 1]

    operating_fpr = (
        false_positive
        / (false_positive + true_negative)
    )

    operating_tpr = (
        true_positive
        / (true_positive + false_negative)
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.plot(
        false_positive_rates,
        true_positive_rates,
        color="#1F77B4",
        linewidth=2.5,
        label=f"Detector ROC (AUC = {roc_auc:.4f})",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Random classifier",
    )

    axis.scatter(
        operating_fpr,
        operating_tpr,
        color="#D62728",
        s=90,
        zorder=5,
        label=(
            f"Threshold {threshold:.2f} "
            f"(FPR={operating_fpr:.2%}, "
            f"TPR={operating_tpr:.2%})"
        ),
    )

    axis.set_xlim(
        -0.01,
        1.0,
    )

    axis.set_ylim(
        0.0,
        1.01,
    )

    axis.set_xlabel(
        "False Positive Rate"
    )

    axis.set_ylabel(
        "True Positive Rate"
    )

    axis.set_title(
        "ROC Curve for Adversarial Attack Detection"
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend(
        loc="lower right",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create detector confusion matrix "
            "and ROC curve"
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
        "--output-dir",
        type=Path,
        default=Path("results"),
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

    labels, probabilities = collect_predictions(
        model,
        data_loader,
        device,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_path = (
        args.output_dir
        / "detector_confusion_matrix.png"
    )

    roc_path = (
        args.output_dir
        / "detector_roc_curve.png"
    )

    create_confusion_matrix(
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
        output_path=confusion_path,
    )

    create_roc_curve(
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
        output_path=roc_path,
    )

    print(
        "Confusion matrix saved to:",
        confusion_path,
    )

    print(
        "ROC curve saved to:",
        roc_path,
    )


if __name__ == "__main__":
    main()