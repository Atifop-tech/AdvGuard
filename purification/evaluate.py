import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from purification.dataset import PurificationPairDataset, ATTACK_NAMES
from purification.model import PurificationNet
from utils.victim_model import load_victim_model


MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)


def load_classifier(checkpoint_path, device):
    return load_victim_model(checkpoint_path, device)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate purification on the untouched AdvGuard test split."
    )
    parser.add_argument("--test-dataset", default="adversarial_dataset/test.pt")
    parser.add_argument("--purifier-checkpoint", default="model/purifier_best.pth")
    parser.add_argument("--victim-checkpoint", default="model/resnet18_cifar10.pth")
    parser.add_argument("--protected-checkpoint", default="model/protected_resnet18_continued_best.pth")
    parser.add_argument("--output-dir", default="results/purification")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PurificationPairDataset(
        args.test_dataset, training=False, include_identity=True
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(
        args.purifier_checkpoint, map_location=device, weights_only=True
    )
    channels = int(checkpoint.get("channels", 48))
    blocks = int(checkpoint.get("blocks", 4))
    purifier = PurificationNet(channels=channels, blocks=blocks).to(device)
    purifier.load_state_dict(checkpoint["model_state_dict"])
    purifier.eval()

    victim = load_classifier(args.victim_checkpoint, device)
    protected = None
    protected_path = Path(args.protected_checkpoint)
    if protected_path.is_file():
        protected = load_classifier(protected_path, device)
        print(f"Protected classifier loaded: {protected_path}")
    else:
        print(f"Protected classifier not found; skipping it: {protected_path}")

    mean = MEAN.to(device)
    std = STD.to(device)
    classifiers = {"victim": victim}
    if protected is not None:
        classifiers["protected"] = protected

    rows = []
    grouped = defaultdict(lambda: {
        "count": 0, "raw_correct": defaultdict(int),
        "purified_correct": defaultdict(int), "mae": 0.0, "mse": 0.0,
        "raw_confidence": defaultdict(float),
        "purified_confidence": defaultdict(float),
    })

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        purified = purifier(images)
        labels = torch.tensor(
            [0] * len(images), device=device
        )
        # The paired dataset does not need class labels for reconstruction.
        # CIFAR-10 labels are loaded separately from the original test tensor.
        del labels

        batch_size = images.size(0)
        attack_ids = batch["attack_id"].tolist()
        is_clean = batch["is_clean"].tolist()
        success_flags = batch["attack_success"].tolist()
        source_ids = batch["source_id"].tolist()

        mae = (purified - targets).abs().flatten(1).mean(1)
        mse = (purified - targets).pow(2).flatten(1).mean(1)

        raw_normalized = (images - mean) / std
        purified_normalized = (purified - mean) / std

        batch_outputs = {}
        for name, classifier in classifiers.items():
            raw_logits = classifier(raw_normalized)
            purified_logits = classifier(purified_normalized)
            batch_outputs[name] = (
                raw_logits.argmax(1),
                purified_logits.argmax(1),
                raw_logits.softmax(1).amax(1),
                purified_logits.softmax(1).amax(1),
            )

        # Recover class labels from the paired dataset's source tensor.
        # The dataset is extended below to expose labels in each batch.
        if "label" not in batch:
            raise RuntimeError(
                "PurificationPairDataset must return source class labels; "
                "update purification/dataset.py before evaluation."
            )
        true_labels = batch["label"].to(device)

        for i in range(batch_size):
            attack_id = int(attack_ids[i])
            group = "clean" if is_clean[i] else ATTACK_NAMES[attack_id]
            record = {
                "source_id": int(source_ids[i]),
                "group": group,
                "is_clean": bool(is_clean[i]),
                "attack_success": bool(success_flags[i]),
                "mae": float(mae[i].item()),
                "mse": float(mse[i].item()),
            }
            for name, (raw_pred, pur_pred, raw_conf, pur_conf) in batch_outputs.items():
                record[f"{name}_raw_prediction"] = int(raw_pred[i].item())
                record[f"{name}_purified_prediction"] = int(pur_pred[i].item())
                record[f"{name}_raw_correct"] = bool(raw_pred[i].item() == true_labels[i].item())
                record[f"{name}_purified_correct"] = bool(pur_pred[i].item() == true_labels[i].item())
                record[f"{name}_raw_confidence"] = float(raw_conf[i].item())
                record[f"{name}_purified_confidence"] = float(pur_conf[i].item())

            rows.append(record)
            stats = grouped[group]
            stats["count"] += 1
            stats["mae"] += record["mae"]
            stats["mse"] += record["mse"]
            for name in classifiers:
                stats["raw_correct"][name] += int(record[f"{name}_raw_correct"])
                stats["purified_correct"][name] += int(record[f"{name}_purified_correct"])
                stats["raw_confidence"][name] += record[f"{name}_raw_confidence"]
                stats["purified_confidence"][name] += record[f"{name}_purified_confidence"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = output_dir / "purification_test_predictions.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for group, stats in grouped.items():
        count = stats["count"]
        row = {
            "group": group,
            "samples": count,
            "mean_mae": stats["mae"] / count,
            "mean_mse": stats["mse"] / count,
        }
        for name in classifiers:
            row[f"{name}_raw_accuracy"] = stats["raw_correct"][name] / count
            row[f"{name}_purified_accuracy"] = stats["purified_correct"][name] / count
            row[f"{name}_raw_confidence"] = stats["raw_confidence"][name] / count
            row[f"{name}_purified_confidence"] = stats["purified_confidence"][name] / count
        summary_rows.append(row)

    summary_path = output_dir / "purification_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nPURIFICATION TEST SUMMARY")
    for row in summary_rows:
        print(
            f"{row['group']:>8}: n={row['samples']:4d}, "
            f"MAE={row['mean_mae']:.5f}, "
            + ", ".join(
                f"{name} raw={row[f'{name}_raw_accuracy']:.2%} "
                f"purified={row[f'{name}_purified_accuracy']:.2%}"
                for name in classifiers
            )
        )
    print(f"Per-sample results: {per_sample_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
