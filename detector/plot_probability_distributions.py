import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from detector.detector_dataset import (
    AdversarialDetectorDataset,
)

from detector.detector_model import (
    AdversarialDetector,
)


GROUPS = {
    0: {
        "name": "Clean",
        "color": "#2CA02C",
    },
    1: {
        "name": "FGSM",
        "color": "#1F77B4",
    },
    2: {
        "name": "PGD",
        "color": "#FF7F0E",
    },
    3: {
        "name": "C&W",
        "color": "#D62728",
    },
}


@torch.no_grad()
def collect_probabilities(
    model,
    data_loader,
    device,
):
    model.eval()

    probabilities = []
    attack_ids = []

    progress_bar = tqdm(
        data_loader,
        desc="Collecting probabilities",
    )

    for batch in progress_bar:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        batch_probabilities = torch.sigmoid(
            logits
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

    return (
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        np.asarray(
            attack_ids,
            dtype=np.int64,
        ),
    )


def create_individual_distributions(
    probabilities,
    attack_ids,
    threshold,
    output_path,
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
    )

    axes = axes.flatten()

    bins = np.linspace(
        0.0,
        1.0,
        41,
    )

    for axis, attack_id in zip(
        axes,
        GROUPS.keys(),
    ):
        group = GROUPS[attack_id]

        group_probabilities = probabilities[
            attack_ids == attack_id
        ]

        axis.hist(
            group_probabilities,
            bins=bins,
            color=group["color"],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )

        axis.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=(
                f"Threshold = {threshold:.2f}"
            ),
        )

        below_threshold = int(
            (
                group_probabilities
                < threshold
            ).sum()
        )

        above_threshold = int(
            (
                group_probabilities
                >= threshold
            ).sum()
        )

        if attack_id == 0:
            subtitle = (
                f"Accepted: {below_threshold} | "
                f"False alarms: {above_threshold}"
            )
        else:
            subtitle = (
                f"Detected: {above_threshold} | "
                f"Missed: {below_threshold}"
            )

        axis.set_title(
            (
                f"{group['name']} Images\n"
                f"{subtitle}"
            )
        )

        axis.set_ylabel(
            "Number of images"
        )

        axis.grid(
            alpha=0.2,
        )

        axis.legend()

    axes[2].set_xlabel(
        "Predicted adversarial probability"
    )

    axes[3].set_xlabel(
        "Predicted adversarial probability"
    )

    figure.suptitle(
        (
            "Adversarial Detector Probability "
            "Distributions"
        ),
        fontsize=16,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_overlay_distribution(
    probabilities,
    attack_ids,
    threshold,
    output_path,
):
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    bins = np.linspace(
        0.0,
        1.0,
        51,
    )

    for attack_id, group in GROUPS.items():
        group_probabilities = probabilities[
            attack_ids == attack_id
        ]

        axis.hist(
            group_probabilities,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=group["color"],
            label=group["name"],
        )

    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}",
    )

    axis.axvspan(
        0,
        threshold,
        color="#2CA02C",
        alpha=0.05,
        label="Predicted clean region",
    )

    axis.axvspan(
        threshold,
        1,
        color="#D62728",
        alpha=0.05,
        label="Predicted adversarial region",
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.set_xlabel(
        "Predicted adversarial probability"
    )

    axis.set_ylabel(
        "Probability density"
    )

    axis.set_title(
        (
            "Clean and Attack Probability "
            "Distributions"
        )
    )

    axis.grid(
        alpha=0.2,
    )

    axis.legend(
        loc="upper center",
        ncol=3,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_summary(
    probabilities,
    attack_ids,
    threshold,
    output_path,
):
    rows = []

    for attack_id, group in GROUPS.items():
        group_probabilities = probabilities[
            attack_ids == attack_id
        ]

        above_threshold = int(
            (
                group_probabilities
                >= threshold
            ).sum()
        )

        below_threshold = int(
            (
                group_probabilities
                < threshold
            ).sum()
        )

        rows.append({
            "group": group["name"],
            "samples": len(
                group_probabilities
            ),
            "mean_probability": float(
                group_probabilities.mean()
            ),
            "median_probability": float(
                np.median(
                    group_probabilities
                )
            ),
            "minimum_probability": float(
                group_probabilities.min()
            ),
            "maximum_probability": float(
                group_probabilities.max()
            ),
            "percentile_10": float(
                np.percentile(
                    group_probabilities,
                    10,
                )
            ),
            "percentile_90": float(
                np.percentile(
                    group_probabilities,
                    90,
                )
            ),
            "below_threshold":
                below_threshold,
            "above_threshold":
                above_threshold,
            "above_threshold_rate":
                above_threshold
                / len(group_probabilities),
        })

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot detector probability "
            "distributions by attack"
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
        checkpoint.get(
            "threshold",
            0.5,
        )
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

    probabilities, attack_ids = (
        collect_probabilities(
            model,
            data_loader,
            device,
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    individual_path = (
        args.output_dir
        / "detector_probability_distributions.png"
    )

    overlay_path = (
        args.output_dir
        / "detector_probability_overlay.png"
    )

    summary_path = (
        args.output_dir
        / "detector_probability_summary.csv"
    )

    create_individual_distributions(
        probabilities,
        attack_ids,
        threshold,
        individual_path,
    )

    create_overlay_distribution(
        probabilities,
        attack_ids,
        threshold,
        overlay_path,
    )

    rows = create_summary(
        probabilities,
        attack_ids,
        threshold,
        summary_path,
    )

    print("\nPROBABILITY SUMMARY")
    print("-" * 65)

    for row in rows:
        print(
            f"{row['group']:6s} | "
            f"mean={row['mean_probability']:.4f} | "
            f"median={row['median_probability']:.4f} | "
            f"above threshold="
            f"{row['above_threshold_rate']:.2%}"
        )

    print(
        "\nIndividual distributions:",
        individual_path,
    )

    print(
        "Overlay distribution:",
        overlay_path,
    )

    print(
        "Summary CSV:",
        summary_path,
    )


if __name__ == "__main__":
    main()
    