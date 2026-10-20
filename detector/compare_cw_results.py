import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from detector.detector_model import AdversarialDetector


CIFAR10_MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465]
).view(1, 3, 1, 1)

CIFAR10_STD = torch.tensor(
    [0.2470, 0.2435, 0.2616]
).view(1, 3, 1, 1)


def load_detector(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
        threshold = float(checkpoint.get("threshold", 0.26))
    else:
        state_dict = checkpoint
        threshold = 0.26

    model = AdversarialDetector().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, threshold


def calculate_statistics(name, probabilities, l2_values, linf_values):
    count = len(probabilities)

    if count == 0:
        return {
            "group": name,
            "count": 0,
            "mean_probability": np.nan,
            "median_probability": np.nan,
            "minimum_probability": np.nan,
            "maximum_probability": np.nan,
            "mean_l2": np.nan,
            "median_l2": np.nan,
            "minimum_l2": np.nan,
            "maximum_l2": np.nan,
            "mean_linf": np.nan,
            "median_linf": np.nan,
            "minimum_linf": np.nan,
            "maximum_linf": np.nan,
        }

    return {
        "group": name,
        "count": count,
        "mean_probability": float(np.mean(probabilities)),
        "median_probability": float(np.median(probabilities)),
        "minimum_probability": float(np.min(probabilities)),
        "maximum_probability": float(np.max(probabilities)),
        "mean_l2": float(np.mean(l2_values)),
        "median_l2": float(np.median(l2_values)),
        "minimum_l2": float(np.min(l2_values)),
        "maximum_l2": float(np.max(l2_values)),
        "mean_linf": float(np.mean(linf_values)),
        "median_linf": float(np.median(linf_values)),
        "minimum_linf": float(np.min(linf_values)),
        "maximum_linf": float(np.max(linf_values)),
    }


def print_statistics(statistics):
    print("\nC&W COMPARISON RESULTS")
    print("-" * 78)

    for result in statistics:
        print(f"\n{result['group']}")
        print("-" * 45)
        print(f"Samples:                 {result['count']}")

        if result["count"] == 0:
            continue

        print(
            f"Detector probability:    "
            f"mean={result['mean_probability']:.6f}, "
            f"median={result['median_probability']:.6f}"
        )
        print(
            f"Probability range:       "
            f"{result['minimum_probability']:.6f} "
            f"to {result['maximum_probability']:.6f}"
        )
        print(
            f"L2 distortion:           "
            f"mean={result['mean_l2']:.6f}, "
            f"median={result['median_l2']:.6f}"
        )
        print(
            f"L2 range:                "
            f"{result['minimum_l2']:.6f} "
            f"to {result['maximum_l2']:.6f}"
        )
        print(
            f"Linf distortion:         "
            f"mean={result['mean_linf']:.6f}, "
            f"median={result['median_linf']:.6f}"
        )
        print(
            f"Linf range:              "
            f"{result['minimum_linf']:.6f} "
            f"to {result['maximum_linf']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare detected and missed C&W adversarial samples."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the independent detector test dataset.",
    )

    parser.add_argument(
        "--checkpoint",
        default="model/adversarial_detector_final.pth",
        help="Path to the final detector checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/cw_comparison",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.dataset, map_location="cpu")

    images = data["images"].float()
    binary_labels = data["binary_labels"].long()
    attack_ids = data["attack_ids"].long()
    attack_success = data["attack_success"].bool()
    source_indices = data["source_indices"].long()
    l2_values = data["l2"].float()
    linf_values = data["linf"].float()

    model, threshold = load_detector(
        args.checkpoint,
        device,
    )

    print(f"Loaded threshold: {threshold:.2f}")

    indexed_dataset = TensorDataset(
        images,
        torch.arange(len(images)),
    )

    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    probabilities = torch.empty(
        len(images),
        dtype=torch.float32,
    )

    mean = CIFAR10_MEAN.to(device)
    std = CIFAR10_STD.to(device)

    with torch.inference_mode():
        for batch_images, batch_indices in loader:
            batch_images = batch_images.to(device)

            normalized_images = (
                batch_images - mean
            ) / std

            logits = model(normalized_images).flatten()
            batch_probabilities = torch.sigmoid(logits)

            probabilities[batch_indices] = (
                batch_probabilities.cpu()
            )

    # Dataset convention:
    # CLEAN = 0
    # FGSM = 1
    # PGD = 2
    # C&W = 3
    cw_mask = (
        (binary_labels == 1)
        & (attack_ids == 3)
    )

    detected_mask = probabilities >= threshold
    missed_mask = probabilities < threshold

    successful_detected_mask = (
        cw_mask
        & attack_success
        & detected_mask
    )

    successful_missed_mask = (
        cw_mask
        & attack_success
        & missed_mask
    )

    failed_detected_mask = (
        cw_mask
        & (~attack_success)
        & detected_mask
    )

    failed_missed_mask = (
        cw_mask
        & (~attack_success)
        & missed_mask
    )

    groups = {
        "Successful detected C&W": successful_detected_mask,
        "Successful missed C&W": successful_missed_mask,
        "Failed detected C&W": failed_detected_mask,
        "Failed missed C&W": failed_missed_mask,
    }

    statistics = []

    for group_name, group_mask in groups.items():
        indices = torch.where(group_mask)[0]

        group_probabilities = (
            probabilities[indices].numpy()
        )

        group_l2 = (
            l2_values[indices].numpy()
        )

        group_linf = (
            linf_values[indices].numpy()
        )

        statistics.append(
            calculate_statistics(
                group_name,
                group_probabilities,
                group_l2,
                group_linf,
            )
        )

    print_statistics(statistics)

    total_cw = int(cw_mask.sum())
    detected_cw = int((cw_mask & detected_mask).sum())
    missed_cw = int((cw_mask & missed_mask).sum())

    successful_cw = int(
        (cw_mask & attack_success).sum()
    )

    successful_detected = int(
        successful_detected_mask.sum()
    )

    successful_missed = int(
        successful_missed_mask.sum()
    )

    print("\nOVERALL C&W SUMMARY")
    print("-" * 78)
    print(f"Total C&W samples:                 {total_cw}")
    print(f"All C&W detected:                  {detected_cw}")
    print(f"All C&W missed:                    {missed_cw}")
    print(f"Successful C&W attacks:            {successful_cw}")
    print(f"Successful C&W detected:           {successful_detected}")
    print(f"Successful C&W missed:             {successful_missed}")

    if successful_cw > 0:
        successful_detection_rate = (
            successful_detected / successful_cw
        )

        print(
            f"Successful C&W detection rate:     "
            f"{successful_detection_rate:.2%}"
        )

    summary_path = output_dir / "cw_comparison_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=statistics[0].keys(),
        )

        writer.writeheader()
        writer.writerows(statistics)

    details_path = output_dir / "cw_sample_details.csv"

    with details_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "dataset_index",
            "source_index",
            "attack_success",
            "detected",
            "group",
            "detector_probability",
            "threshold",
            "l2",
            "linf",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        cw_indices = torch.where(cw_mask)[0]

        for index in cw_indices.tolist():
            success = bool(attack_success[index])
            detected = bool(detected_mask[index])

            if success and detected:
                group = "successful_detected"
            elif success and not detected:
                group = "successful_missed"
            elif not success and detected:
                group = "failed_detected"
            else:
                group = "failed_missed"

            writer.writerow({
                "dataset_index": index,
                "source_index": int(source_indices[index]),
                "attack_success": success,
                "detected": detected,
                "group": group,
                "detector_probability": float(
                    probabilities[index]
                ),
                "threshold": threshold,
                "l2": float(l2_values[index]),
                "linf": float(linf_values[index]),
            })

    successful_detected_indices = torch.where(
        successful_detected_mask
    )[0]

    successful_missed_indices = torch.where(
        successful_missed_mask
    )[0]

    detected_probabilities = probabilities[
        successful_detected_indices
    ].numpy()

    missed_probabilities = probabilities[
        successful_missed_indices
    ].numpy()

    detected_l2 = l2_values[
        successful_detected_indices
    ].numpy()

    missed_l2 = l2_values[
        successful_missed_indices
    ].numpy()

    detected_linf = linf_values[
        successful_detected_indices
    ].numpy()

    missed_linf = linf_values[
        successful_missed_indices
    ].numpy()

    # Probability distribution
    plt.figure(figsize=(10, 6))

    bins = np.linspace(0, 1, 31)

    plt.hist(
        detected_probabilities,
        bins=bins,
        alpha=0.70,
        label=f"Detected successful C&W ({len(detected_probabilities)})",
        color="tab:blue",
        edgecolor="black",
    )

    plt.hist(
        missed_probabilities,
        bins=bins,
        alpha=0.80,
        label=f"Missed successful C&W ({len(missed_probabilities)})",
        color="tab:red",
        edgecolor="black",
    )

    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}",
    )

    plt.xlabel("Predicted adversarial probability")
    plt.ylabel("Number of samples")
    plt.title(
        "Detector probabilities for successful C&W attacks"
    )
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    probability_plot_path = (
        output_dir / "cw_probability_comparison.png"
    )

    plt.savefig(
        probability_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # Distortion boxplots
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].boxplot(
        [detected_l2, missed_l2],
        tick_labels=["Detected", "Missed"],
        showmeans=True,
    )

    axes[0].set_title(
        "L2 distortion of successful C&W attacks"
    )

    axes[0].set_ylabel("L2 distortion")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].boxplot(
        [detected_linf, missed_linf],
        tick_labels=["Detected", "Missed"],
        showmeans=True,
    )

    axes[1].set_title(
        "L∞ distortion of successful C&W attacks"
    )

    axes[1].set_ylabel("L∞ distortion")
    axes[1].grid(axis="y", alpha=0.2)

    figure.tight_layout()

    distortion_plot_path = (
        output_dir / "cw_distortion_comparison.png"
    )

    figure.savefig(
        distortion_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Detector probability against L2 distortion
    plt.figure(figsize=(10, 6))

    plt.scatter(
        detected_l2,
        detected_probabilities,
        color="tab:blue",
        alpha=0.70,
        label="Detected successful C&W",
    )

    plt.scatter(
        missed_l2,
        missed_probabilities,
        color="tab:red",
        marker="x",
        s=70,
        linewidth=2,
        label="Missed successful C&W",
    )

    plt.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}",
    )

    plt.xlabel("L2 distortion")
    plt.ylabel("Predicted adversarial probability")

    plt.title(
        "C&W distortion versus detector probability"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    scatter_plot_path = (
        output_dir / "cw_l2_probability_scatter.png"
    )

    plt.savefig(
        scatter_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("\nFILES CREATED")
    print("-" * 78)
    print(f"Summary:             {summary_path}")
    print(f"Sample details:      {details_path}")
    print(f"Probability plot:    {probability_plot_path}")
    print(f"Distortion plot:     {distortion_plot_path}")
    print(f"L2 scatter plot:     {scatter_plot_path}")


if __name__ == "__main__":
    main()