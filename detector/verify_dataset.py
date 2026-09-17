from pathlib import Path

import torch


ATTACK_NAMES = {
    0: "clean",
    1: "fgsm",
    2: "pgd",
    3: "cw",
}


def load_split(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def verify_split(split_name, data):
    required_keys = {
        "images",
        "binary_labels",
        "true_labels",
        "attack_ids",
        "source_indices",
        "clean_correct",
        "attack_success",
        "clean_predictions",
        "adversarial_predictions",
        "linf",
        "l2",
    }

    missing_keys = required_keys - set(data.keys())

    if missing_keys:
        raise ValueError(
            f"{split_name} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    images = data["images"]
    binary_labels = data["binary_labels"]
    attack_ids = data["attack_ids"]

    number_of_samples = images.size(0)

    print(f"\n{split_name.upper()} SPLIT")
    print("-" * 50)

    print("Image shape:", tuple(images.shape))
    print("Image dtype:", images.dtype)
    print("Minimum pixel:", images.min().item())
    print("Maximum pixel:", images.max().item())
    print("Contains NaN:", torch.isnan(images).any().item())
    print("Contains infinity:", torch.isinf(images).any().item())

    clean_count = (
        binary_labels == 0
    ).sum().item()

    adversarial_count = (
        binary_labels == 1
    ).sum().item()

    print("Total samples:", number_of_samples)
    print("Clean samples:", clean_count)
    print("Adversarial samples:", adversarial_count)

    print("\nAttack distribution:")

    for attack_id, attack_name in ATTACK_NAMES.items():
        count = (
            attack_ids == attack_id
        ).sum().item()

        print(
            f"  {attack_name.upper()}: {count}"
        )

    clean_mask = binary_labels == 0
    adversarial_mask = binary_labels == 1

    successful_attacks = (
        data["attack_success"][adversarial_mask]
        .sum()
        .item()
    )

    adversarial_total = (
        adversarial_mask.sum().item()
    )

    attack_success_rate = (
        successful_attacks / adversarial_total
        if adversarial_total > 0
        else 0.0
    )

    print("\nAttack information:")
    print(
        "Successful adversarial samples:",
        successful_attacks,
    )

    print(
        "Attack success percentage:",
        f"{attack_success_rate:.2%}",
    )

    if adversarial_total > 0:
        print(
            "Mean adversarial Linf:",
            data["linf"][adversarial_mask]
            .mean()
            .item(),
        )

        print(
            "Mean adversarial L2:",
            data["l2"][adversarial_mask]
            .mean()
            .item(),
        )

    assert images.ndim == 4
    assert images.shape[1:] == (3, 32, 32)

    assert number_of_samples == len(binary_labels)
    assert number_of_samples == len(attack_ids)

    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0

    assert not torch.isnan(images).any()
    assert not torch.isinf(images).any()

    assert set(binary_labels.unique().tolist()).issubset(
        {0.0, 1.0}
    )

    assert clean_count == adversarial_count

    assert torch.all(
        attack_ids[clean_mask] == 0
    )

    assert torch.all(
        attack_ids[adversarial_mask] > 0
    )

    print(
        f"\n{split_name} verification passed."
    )


def check_split_leakage(train_data, validation_data):
    train_sources = set(
        train_data["source_indices"]
        .unique()
        .tolist()
    )

    validation_sources = set(
        validation_data["source_indices"]
        .unique()
        .tolist()
    )

    overlap = (
        train_sources & validation_sources
    )

    print("\nSPLIT LEAKAGE CHECK")
    print("-" * 50)

    print(
        "Unique training source images:",
        len(train_sources),
    )

    print(
        "Unique validation source images:",
        len(validation_sources),
    )

    print(
        "Overlapping source images:",
        len(overlap),
    )

    if overlap:
        raise ValueError(
            "Data leakage detected between "
            "training and validation splits."
        )

    print("No train-validation leakage detected.")


def main():
    dataset_directory = Path(
        "adversarial_dataset"
    )

    train_data = load_split(
        dataset_directory / "train.pt"
    )

    validation_data = load_split(
        dataset_directory / "validation.pt"
    )

    verify_split(
        "train",
        train_data,
    )

    verify_split(
        "validation",
        validation_data,
    )

    check_split_leakage(
        train_data,
        validation_data,
    )

    print(
        "\nDataset verification completed successfully."
    )


if __name__ == "__main__":
    main()