
import os
import torch
import pandas as pd

from model import create_resnet18_cifar10


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_PATH = "adversarial_dataset/test.pt"
METADATA_PATH = "adversarial_dataset/test_metadata.csv"

VICTIM_PATH = "model/resnet18_cifar10.pth"
PROTECTED_PATH = "model/protected_resnet18_continued_best.pth"

OUTPUT_DIR = "results"
BATCH_SIZE = 64

# CIFAR-10 normalization used by the project's classifier.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_model(checkpoint_path, protected=False):
    model = create_resnet18_cifar10()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if protected:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def main():
    global device

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the stored test examples and their true labels.
    test = torch.load(
        TEST_PATH,
        map_location="cpu",
        weights_only=False
    )

    images = test["images"].float()
    labels = test["true_labels"].long()

    metadata = pd.read_csv(METADATA_PATH)

    if len(images) != len(labels) or len(images) != len(metadata):
        raise ValueError(
            "Images, labels, and metadata have different lengths."
        )

    # Apply the same CIFAR-10 normalization to both models.
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    images = (images - mean) / std

    print("Test samples:", len(images))
    print("Attack categories:")
    print(metadata["attack_name"].value_counts().to_string())

    victim = load_model(VICTIM_PATH, protected=False)
    protected = load_model(PROTECTED_PATH, protected=True)

    models = {
        "victim": victim,
        "protected": protected
    }

    # Save each model's predictions on exactly the same examples.
    predictions = {}

    for model_name, model in models.items():
        all_predictions = []

        with torch.inference_mode():
            for start in range(0, len(images), BATCH_SIZE):
                batch = images[start:start + BATCH_SIZE].to(device)
                logits = model(batch)
                predicted = logits.argmax(dim=1)
                all_predictions.extend(predicted.cpu().tolist())

        predictions[model_name] = all_predictions

        print(f"\n{model_name.upper()} evaluation completed.")

    results = []

    # Use the metadata's attack names rather than assuming attack IDs.
    attack_names = metadata["attack_name"].astype(str)

    for model_name, predicted in predictions.items():
        predicted = torch.tensor(predicted)
        correct = predicted.eq(labels)

        for attack_name in ["clean", "fgsm", "pgd", "cw"]:
            mask = torch.tensor(
                (attack_names == attack_name).to_numpy(),
                dtype=torch.bool
            )

            count = int(mask.sum())

            if count == 0:
                print(f"Warning: no samples found for {attack_name}")
                continue

            accuracy = correct[mask].float().mean().item()

            results.append({
                "model": model_name,
                "category": attack_name,
                "samples": count,
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100
            })

        # Accuracy over the adversarial subset only.
        adversarial_mask = torch.tensor(
            (attack_names != "clean").to_numpy(),
            dtype=torch.bool
        )

        results.append({
            "model": model_name,
            "category": "all_adversarial",
            "samples": int(adversarial_mask.sum()),
            "accuracy": correct[adversarial_mask].float().mean().item(),
            "accuracy_percent":
                correct[adversarial_mask].float().mean().item() * 100
        })

        results.append({
            "model": model_name,
            "category": "overall",
            "samples": len(labels),
            "accuracy": correct.float().mean().item(),
            "accuracy_percent": correct.float().mean().item() * 100
        })

    results_df = pd.DataFrame(results)

    # Human-readable comparison table.
    comparison = results_df.pivot(
        index="category",
        columns="model",
        values="accuracy_percent"
    )

    comparison["protected_minus_victim_pp"] = (
        comparison["protected"] - comparison["victim"]
    )

    print("\n========== TEST RESULTS (%) ==========")
    print(comparison.round(2).to_string())

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "victim_vs_protected_test.csv"),
        index=False
    )

    comparison.to_csv(
        os.path.join(OUTPUT_DIR, "victim_vs_protected_comparison.csv")
    )

    prediction_df = metadata.copy()
    prediction_df["victim_prediction"] = predictions["victim"]
    prediction_df["protected_prediction"] = predictions["protected"]
    prediction_df["true_label"] = labels.tolist()
    prediction_df["victim_correct"] = (
        prediction_df["victim_prediction"]
        == prediction_df["true_label"]
    )
    prediction_df["protected_correct"] = (
        prediction_df["protected_prediction"]
        == prediction_df["true_label"]
    )

    prediction_df.to_csv(
        os.path.join(OUTPUT_DIR, "test_predictions_comparison.csv"),
        index=False
    )

    print("\nSaved evaluation files:")
    print("1. results/victim_vs_protected_test.csv")
    print("2. results/victim_vs_protected_comparison.csv")
    print("3. results/test_predictions_comparison.csv")


if __name__ == "__main__":
    main()
