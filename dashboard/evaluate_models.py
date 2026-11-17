"""Evaluate the trained AdvGuard models on the held-out paired test split."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .model_loader import CLASSES, DEVICE, MEAN, ROOT, STD, ModelRegistry

ATTACK_NAMES = {0: "clean", 1: "FGSM", 2: "PGD", 3: "C&W"}
COLORS = {"victim": "#44c8ed", "protected": "#a38bff", "detector": "#51d6aa", "purified": "#f5ad61"}


def _save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(matrix, title, path, labels):
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")
    image = ax.imshow(matrix, cmap="magma", interpolation="nearest")
    ax.set_title(title, color="#edf3fb", fontsize=14, pad=15, weight="bold")
    ax.set_xlabel("Predicted class", color="#aebdd0")
    ax.set_ylabel("True class", color="#aebdd0")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", color="#aebdd0", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, color="#aebdd0", fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#29364a")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="#aebdd0")
    vmax = matrix.max() if matrix.size else 1
    for row in range(matrix.shape[0]):
        total = max(int(matrix[row].sum()), 1)
        for col in range(matrix.shape[1]):
            count = int(matrix[row, col])
            rate = count / total
            ax.text(col, row, f"{count}\n{rate:.0%}", ha="center", va="center",
                    color="#fff" if count > vmax * 0.45 else "#e8eaf0", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _predict(model, pixels, batch_size=128):
    all_logits = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            batch = pixels[start:start + batch_size].to(DEVICE)
            logits = model((batch - MEAN) / STD)
            all_logits.append(logits.cpu())
    logits = torch.cat(all_logits)
    probabilities = logits.softmax(dim=1).numpy()
    return probabilities.argmax(axis=1), probabilities.max(axis=1), probabilities


def _classification_rows(name, labels, predictions, confidences, attack_ids):
    rows = []
    for group_id, group_name in [(-1, "all"), (-2, "all_adversarial"), *[(i, n) for i, n in ATTACK_NAMES.items()]]:
        mask = np.ones(len(labels), dtype=bool) if group_id == -1 else (
            attack_ids != 0 if group_id == -2 else attack_ids == group_id
        )
        if not mask.any():
            continue
        rows.append({
            "model": name,
            "subset": group_name,
            "samples": int(mask.sum()),
            "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
            "mean_top1_confidence": float(confidences[mask].mean()),
        })
    return rows


def evaluate(output: Path, dataset_path: Path, batch_size: int):
    output.mkdir(parents=True, exist_ok=True)
    raw = torch.load(dataset_path, map_location="cpu", weights_only=True)
    required = {"images", "binary_labels", "true_labels", "attack_ids", "source_indices"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Test dataset is missing required fields: {sorted(missing)}")
    pixels = raw["images"].float().clamp(0, 1)
    labels = raw["true_labels"].long().numpy()
    binary_labels = raw["binary_labels"].long().numpy()
    attack_ids = raw["attack_ids"].long().numpy()
    source_ids = raw["source_indices"].long().numpy()

    registry = ModelRegistry()
    for needed in ("victim", "protected", "purifier", "detector"):
        if needed not in registry.models:
            raise RuntimeError(f"Required trained model '{needed}' is unavailable: {registry.errors.get(needed)}")

    report = []
    class_rows = []
    predictions = {}
    confidence = {}
    probabilities = {}
    for model_id in ("victim", "protected"):
        pred, conf, probs = _predict(registry.models[model_id], pixels, batch_size)
        predictions[model_id], confidence[model_id], probabilities[model_id] = pred, conf, probs
        report.extend(_classification_rows(model_id, labels, pred, conf, attack_ids))
        for subset, mask in (("clean", attack_ids == 0), ("adversarial", attack_ids != 0)):
            matrix = confusion_matrix(labels[mask], pred[mask], labels=list(range(10)))
            _plot_confusion(matrix, f"{model_id.title()} classifier · {subset} test images",
                            output / f"{model_id}_confusion_{subset}.png", CLASSES)
        class_report = classification_report(labels, pred, labels=list(range(10)), target_names=CLASSES,
                                             output_dict=True, zero_division=0)
        for class_name in CLASSES:
            report_row = class_report[class_name]
            class_rows.append({"model": model_id, "class": class_name, "support": int(report_row["support"]),
                               "precision": float(report_row["precision"]), "recall": float(report_row["recall"]),
                               "f1": float(report_row["f1-score"])})
            report.append({"model": model_id, "subset": f"class:{class_name}", "samples": int(report_row["support"]),
                           "accuracy": float(report_row["recall"]), "mean_top1_confidence": ""})

    # The purifier is scored against each row's paired clean source image.
    source_to_clean = {}
    for index, source_id in enumerate(source_ids):
        if binary_labels[index] == 0:
            source_to_clean[int(source_id)] = index
    clean_indices = np.asarray([source_to_clean[int(source_id)] for source_id in source_ids])
    targets = pixels[torch.from_numpy(clean_indices)]
    purified_batches = []
    purifier = registry.models["purifier"]
    purifier.eval()
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            purified_batches.append(purifier(pixels[start:start + batch_size].to(DEVICE)).cpu())
    purified = torch.cat(purified_batches).clamp(0, 1)
    abs_error = (purified - targets).abs().flatten(1).mean(1).numpy()
    sq_error = (purified - targets).square().flatten(1).mean(1).numpy()
    purified_preds = {}
    purified_confidence = {}
    for model_id in ("victim", "protected"):
        pred, conf, _ = _predict(registry.models[model_id], purified, batch_size)
        purified_preds[model_id], purified_confidence[model_id] = pred, conf
    purification_rows = []
    for attack_id, group in ATTACK_NAMES.items():
        mask = attack_ids == attack_id
        if not mask.any():
            continue
        row = {
            "subset": group,
            "samples": int(mask.sum()),
            "mean_mae": float(abs_error[mask].mean()),
            "mean_mse": float(sq_error[mask].mean()),
        }
        for model_id in ("victim", "protected"):
            row[f"{model_id}_raw_accuracy"] = float((predictions[model_id][mask] == labels[mask]).mean())
            row[f"{model_id}_purified_accuracy"] = float((purified_preds[model_id][mask] == labels[mask]).mean())
            row[f"{model_id}_raw_confidence"] = float(confidence[model_id][mask].mean())
            row[f"{model_id}_purified_confidence"] = float(purified_confidence[model_id][mask].mean())
        purification_rows.append(row)
    _save_csv(output / "classification_metrics.csv", report)
    _save_csv(output / "classification_report.csv", class_rows)
    _save_csv(output / "purification_metrics.csv", purification_rows)

    # Detector metrics and scores from the independent test split.
    detector = registry.models["detector"]
    detector_probs = []
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            batch = pixels[start:start + batch_size].to(DEVICE)
            logits = detector((batch - MEAN) / STD)
            detector_probs.extend(logits.sigmoid().cpu().tolist())
    detector_probs = np.asarray(detector_probs, dtype=float)
    threshold = registry.threshold
    detector_pred = detector_probs >= threshold
    tn, fp, fn, tp = confusion_matrix(binary_labels, detector_pred.astype(int), labels=[0, 1]).ravel()
    detector_metrics = [{
        "test_samples": len(binary_labels), "threshold": threshold,
        "accuracy": float(accuracy_score(binary_labels, detector_pred)),
        "precision": float(precision_score(binary_labels, detector_pred, zero_division=0)),
        "recall": float(recall_score(binary_labels, detector_pred, zero_division=0)),
        "f1": float(f1_score(binary_labels, detector_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(binary_labels, detector_probs)),
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
    }]
    detector_group_metrics = []
    for attack_id, group in ATTACK_NAMES.items():
        mask = attack_ids == attack_id
        if mask.any():
            detector_group_metrics.append({"subset": group, "samples": int(mask.sum()),
                                           "detection_rate": float(detector_pred[mask].mean()),
                                           "mean_adversarial_probability": float(detector_probs[mask].mean())})
    _save_csv(output / "detector_metrics.csv", detector_metrics)
    _save_csv(output / "detector_metrics_by_attack.csv", detector_group_metrics)
    _plot_confusion(np.array([[tn, fp], [fn, tp]]), f"Detector confusion matrix · threshold {threshold:.2f}",
                    output / "detector_confusion_matrix.png", ["Clean", "Adversarial"])

    fpr, tpr, _ = roc_curve(binary_labels, detector_probs)
    auc = roc_auc_score(binary_labels, detector_probs)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")
    ax.plot(fpr, tpr, color=COLORS["detector"], lw=2.6, label=f"Detector ROC · AUC {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#718096", lw=1, label="Random baseline")
    ax.scatter([fp / max(fp + tn, 1)], [tp / max(tp + fn, 1)], color="#f5ad61", s=58,
               label=f"Operating threshold {threshold:.2f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate",
           title="Adversarial detector · held-out test ROC")
    ax.grid(alpha=.18, color="#95a4ba")
    ax.tick_params(colors="#aebdd0")
    for spine in ax.spines.values(): spine.set_color("#29364a")
    ax.title.set_color("#edf3fb")
    ax.xaxis.label.set_color("#aebdd0"); ax.yaxis.label.set_color("#aebdd0")
    leg = ax.legend(facecolor="#111a28", edgecolor="#29364a")
    for text in leg.get_texts(): text.set_color("#d8e2ef")
    fig.tight_layout()
    fig.savefig(output / "detector_roc_curve.png", dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # Compact classifier comparison plot.
    groups = ["clean", "FGSM", "PGD", "C&W"]
    x = np.arange(len(groups))
    width = .35
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")
    for offset, model_id in ((-width / 2, "victim"), (width / 2, "protected")):
        vals = [next(row["accuracy"] for row in report if row.get("model") == model_id and row.get("subset") == group) * 100 for group in groups]
        bars = ax.bar(x + offset, vals, width, color=COLORS[model_id], label=model_id.title(), edgecolor="#0b1220")
        ax.bar_label(bars, fmt="%.1f", padding=3, color="#c5d2e0", fontsize=8)
    ax.set_xticks(x, groups, color="#aebdd0")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Classification accuracy (%)", color="#aebdd0")
    ax.set_title("Victim and protected classifier · test accuracy by input group", color="#edf3fb", weight="bold", pad=14)
    ax.grid(axis="y", alpha=.17, color="#95a4ba")
    ax.tick_params(axis="y", colors="#aebdd0")
    for spine in ax.spines.values(): spine.set_color("#29364a")
    leg = ax.legend(facecolor="#111a28", edgecolor="#29364a")
    for text in leg.get_texts(): text.set_color("#d8e2ef")
    fig.tight_layout()
    fig.savefig(output / "classifier_accuracy_by_attack.png", dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # Purification metrics: pixel error and classification behavior.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0b1220")
    for ax in axes: ax.set_facecolor("#0b1220")
    labels_group = [row["subset"] for row in purification_rows]
    axes[0].bar(labels_group, [row["mean_mae"] for row in purification_rows], color=COLORS["purified"])
    axes[0].set_title("Purified image · mean pixel MAE", color="#edf3fb", weight="bold")
    axes[0].set_ylabel("MAE in [0, 1] pixel space", color="#aebdd0")
    x2 = np.arange(len(labels_group)); w2 = .19
    series = [("victim_raw_accuracy", "Victim · raw", COLORS["victim"]),
              ("victim_purified_accuracy", "Victim · purified", "#8ce2f5"),
              ("protected_raw_accuracy", "Protected · raw", COLORS["protected"]),
              ("protected_purified_accuracy", "Protected · purified", "#c4b5fd")]
    for i, (key, name, color) in enumerate(series):
        axes[1].bar(x2 + (i - 1.5) * w2, [row[key] * 100 for row in purification_rows], w2, label=name, color=color)
    axes[1].set_title("Classifier accuracy before and after purification", color="#edf3fb", weight="bold")
    axes[1].set_ylabel("Accuracy (%)", color="#aebdd0")
    axes[1].set_xticks(x2, labels_group, color="#aebdd0")
    for ax in axes:
        ax.tick_params(axis="y", colors="#aebdd0")
        ax.grid(axis="y", alpha=.17, color="#95a4ba")
        for spine in ax.spines.values(): spine.set_color("#29364a")
    axes[1].set_ylim(0, 105)
    leg = axes[1].legend(fontsize=7, facecolor="#111a28", edgecolor="#29364a")
    for text in leg.get_texts(): text.set_color("#d8e2ef")
    fig.tight_layout()
    fig.savefig(output / "purification_evaluation.png", dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # Per-row predictions are useful for reproducible drill-down.
    prediction_rows = []
    for i in range(len(labels)):
        prediction_rows.append({
            "source_index": int(source_ids[i]), "attack": ATTACK_NAMES[int(attack_ids[i])],
            "true_class": CLASSES[int(labels[i])], "binary_label": int(binary_labels[i]),
            "victim_prediction": CLASSES[int(predictions["victim"][i])],
            "victim_confidence": float(confidence["victim"][i]),
            "victim_correct": bool(predictions["victim"][i] == labels[i]),
            "protected_prediction": CLASSES[int(predictions["protected"][i])],
            "protected_confidence": float(confidence["protected"][i]),
            "protected_correct": bool(predictions["protected"][i] == labels[i]),
            "detector_probability": float(detector_probs[i]),
            "detector_prediction": "adversarial" if detector_pred[i] else "clean",
            "purification_mae": float(abs_error[i]), "purification_mse": float(sq_error[i]),
        })
    _save_csv(output / "test_predictions.csv", prediction_rows)

    summary_path = output / "README.md"
    det = detector_metrics[0]
    victim_clean = next(row["accuracy"] for row in report if row.get("model") == "victim" and row.get("subset") == "clean")
    protected_clean = next(row["accuracy"] for row in report if row.get("model") == "protected" and row.get("subset") == "clean")
    adv = next(row for row in purification_rows if row["subset"] != "clean")
    attack_table = []
    for attack_id, group in ATTACK_NAMES.items():
        victim_row = next(row for row in report if row.get("model") == "victim" and row.get("subset") == group)
        protected_row = next(row for row in report if row.get("model") == "protected" and row.get("subset") == group)
        detection_row = next(row for row in detector_group_metrics if row["subset"] == group)
        purification_row = next(row for row in purification_rows if row["subset"] == group)
        attack_table.append(
            f"| {group} | {victim_row['samples']} | {victim_row['accuracy']:.2%} | "
            f"{protected_row['accuracy']:.2%} | {detection_row['detection_rate']:.2%} | "
            f"{purification_row['victim_purified_accuracy']:.2%} | "
            f"{purification_row['protected_purified_accuracy']:.2%} |"
        )
    summary_path.write_text(
        "# AdvGuard model evaluation\n\n"
        "This report was generated from the held-out `adversarial_dataset/test.pt` split using the local trained checkpoints. "
        "The detector was selected on validation data; these figures are test results. The adversarial examples were generated "
        "against the victim classifier, so classifier comparisons are non-adaptive.\n\n"
        f"- Test rows: {len(labels)} ({int((binary_labels == 0).sum())} clean, {int((binary_labels == 1).sum())} adversarial).\n"
        f"- Detector threshold: {threshold:.2f}; accuracy {det['accuracy']:.2%}; precision {det['precision']:.2%}; "
        f"recall {det['recall']:.2%}; F1 {det['f1']:.2%}; ROC-AUC {det['roc_auc']:.4f}.\n"
        f"- Detector confusion counts: TN {tn}, FP {fp}, FN {fn}, TP {tp}.\n"
        f"- Clean classifier accuracy: victim {victim_clean:.2%}; protected {protected_clean:.2%}.\n"
        f"- Adversarial purification mean MAE {adv['mean_mae']:.5f}, MSE {adv['mean_mse']:.5f} in pixel space.\n\n"
        "## Test accuracy by input group\n\n"
        "| Group | Samples | Victim raw | Protected raw | Detector rate | Victim purified | Protected purified |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(attack_table) + "\n\n"
        "## Visual summary\n\n"
        "### Detector performance\n\n"
        "![Detector ROC curve](detector_roc_curve.png)\n\n"
        "![Detector confusion matrix](detector_confusion_matrix.png)\n\n"
        "### Classifiers and purification\n\n"
        "![Classifier accuracy by attack](classifier_accuracy_by_attack.png)\n\n"
        "![Purification evaluation](purification_evaluation.png)\n\n"
        "## Artifacts\n\n"
        "- `detector_roc_curve.png` — detector operating point and ROC curve.\n"
        "- `detector_confusion_matrix.png` — binary test confusion matrix, with counts and row rates.\n"
        "- `classifier_accuracy_by_attack.png` — victim/protected accuracy by attack group.\n"
        "- `victim_confusion_*.png`, `protected_confusion_*.png` — CIFAR-10 class confusion matrices.\n"
        "- `purification_evaluation.png` — reconstruction MAE and classifier effects.\n"
        "- `classification_report.csv` — per-class precision, recall, F1, and support.\n"
        "- Other CSV files — test metrics, per-row predictions, and purification measurements.\n\n"
        "Confidence is not accuracy. Classifier accuracy uses the test split's provided true labels. Reconstruction MAE/MSE are "
        "pixel-space errors against each paired clean source image.\n",
        encoding="utf-8",
    )
    print(f"Evaluation report written to {output}")
    print(f"Detector: accuracy={det['accuracy']:.3%} AUC={det['roc_auc']:.5f} threshold={threshold:.2f} TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Clean accuracy: victim={victim_clean:.3%}, protected={protected_clean:.3%}")
    print(f"Adversarial reconstruction MAE={adv['mean_mae']:.6f}, MSE={adv['mean_mse']:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=Path, default=ROOT / "adversarial_dataset/test.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/model_evaluation")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    evaluate(args.output_dir, args.test_data, args.batch_size)


if __name__ == "__main__":
    main()
