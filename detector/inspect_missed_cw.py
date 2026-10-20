import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from detector.detector_model import AdversarialDetector


CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    threshold = float(checkpoint.get("threshold", 0.26)) if isinstance(checkpoint, dict) else 0.26
    model = AdversarialDetector().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, threshold


def as_numpy_image(tensor):
    return tensor.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Inspect C&W false negatives.")
    parser.add_argument("--dataset", default="data/detector_test.pt")
    parser.add_argument("--checkpoint", default="model/adversarial_detector_final.pth")
    parser.add_argument("--output-dir", default="results/missed_cw")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.dataset, map_location="cpu")
    images = data["images"].float()
    labels = data["binary_labels"].long()
    attack_ids = data["attack_ids"].long()
    source_indices = data["source_indices"].long()
    true_labels = data["true_labels"].long()
    attack_success = data["attack_success"].bool()
    clean_predictions = data["clean_predictions"].long()
    adversarial_predictions = data["adversarial_predictions"].long()
    linf = data["linf"].float()
    l2 = data["l2"].float()

    model, threshold = load_checkpoint(args.checkpoint, device)
    dataset = TensorDataset(images, torch.arange(len(images)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    probabilities = torch.empty(len(images), dtype=torch.float32)
    mean = CIFAR10_MEAN.to(device)
    std = CIFAR10_STD.to(device)
    with torch.inference_mode():
        for batch_images, batch_indices in loader:
            batch_images = batch_images.to(device)
            logits = model((batch_images - mean) / std).flatten()
            probabilities[batch_indices] = torch.sigmoid(logits).cpu()

    # Dataset convention: CLEAN=0, FGSM=1, PGD=2, C&W=3.
    missed_mask = (labels == 1) & (attack_ids == 3) & (probabilities < threshold)
    missed_indices = torch.where(missed_mask)[0]

    clean_by_source = {
        int(source_indices[i]): i
        for i in torch.where(attack_ids == 0)[0].tolist()
    }

    rows = []
    for rank, adv_index in enumerate(missed_indices.tolist(), start=1):
        source_index = int(source_indices[adv_index])
        clean_index = clean_by_source.get(source_index)
        rows.append({
            "rank": rank,
            "dataset_index": adv_index,
            "source_index": source_index,
            "detector_probability": float(probabilities[adv_index]),
            "threshold": threshold,
            "attack_success": bool(attack_success[adv_index]),
            "true_class": CIFAR10_CLASSES[int(true_labels[adv_index])],
            "clean_prediction": CIFAR10_CLASSES[int(clean_predictions[adv_index])],
            "adversarial_prediction": CIFAR10_CLASSES[int(adversarial_predictions[adv_index])],
            "linf": float(linf[adv_index]),
            "l2": float(l2[adv_index]),
            "clean_dataset_index": "" if clean_index is None else clean_index,
        })

    csv_path = output_dir / "missed_cw_examples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["rank"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        columns = max(1, args.columns)
        figure_rows = math.ceil(len(rows) / columns)
        figure, axes = plt.subplots(figure_rows, columns * 2,
                                    figsize=(3.4 * columns, 3.0 * figure_rows),
                                    squeeze=False)
        for axis in axes.flat:
            axis.axis("off")

        for item, row in enumerate(rows):
            adv_index = row["dataset_index"]
            clean_index = row["clean_dataset_index"]
            left = axes[item // columns, 2 * (item % columns)]
            right = axes[item // columns, 2 * (item % columns) + 1]

            if clean_index != "":
                clean_image = images[clean_index]
                left.imshow(as_numpy_image(clean_image))
                left.set_title(f"Clean: {row['true_class']}", fontsize=9)
                perturbation = images[adv_index] - clean_image
                amplified = perturbation.abs().max(dim=0).values
                right.imshow(as_numpy_image(images[adv_index]))
                right.set_title(
                    f"C&W missed  p={row['detector_probability']:.3f}\n"
                    f"L2={row['l2']:.3f}  max|d|={amplified.max():.3f}",
                    fontsize=8,
                )
            else:
                left.imshow(as_numpy_image(images[adv_index]))
                left.set_title("Clean pair unavailable", fontsize=9)
                right.imshow(as_numpy_image(images[adv_index]))
                right.set_title(f"C&W missed  p={row['detector_probability']:.3f}", fontsize=8)
            left.axis("off")
            right.axis("off")

        figure.suptitle(
            f"Missed C&W examples: {len(rows)} false negatives at threshold {threshold:.2f}",
            fontsize=14,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(output_dir / "missed_cw_examples.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

        successful = sum(row["attack_success"] for row in rows)
        probabilities_missed = np.array([row["detector_probability"] for row in rows])
        l2_missed = np.array([row["l2"] for row in rows])
        print(f"Missed C&W examples: {len(rows)}")
        print(f"Successful attacks among misses: {successful}/{len(rows)}")
        print(f"Detector probability: mean={probabilities_missed.mean():.4f}, "
              f"min={probabilities_missed.min():.4f}, max={probabilities_missed.max():.4f}")
        print(f"L2 distortion: mean={l2_missed.mean():.4f}, "
              f"min={l2_missed.min():.4f}, max={l2_missed.max():.4f}")
    else:
        print("No missed C&W examples were found at the loaded threshold.")

    print(f"CSV saved to: {csv_path}")
    print(f"Figure saved to: {output_dir / 'missed_cw_examples.png'}")


if __name__ == "__main__":
    main()
