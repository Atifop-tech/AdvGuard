import argparse

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import RandomCrop, RandomHorizontalFlip


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)

ATTACK_NAMES = {
    0: "CLEAN",
    1: "FGSM",
    2: "PGD",
    3: "CW",
}


class ProtectionDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        training=False,
    ):
        self.dataset_path = dataset_path
        self.training = training

        data = torch.load(
            dataset_path,
            map_location="cpu",
        )

        required_keys = [
            "images",
            "true_labels",
            "binary_labels",
            "attack_ids",
            "source_indices",
        ]

        for key in required_keys:
            if key not in data:
                raise KeyError(
                    f"Required key '{key}' was not "
                    f"found in {dataset_path}"
                )

        self.images = data["images"].float()
        self.true_labels = data[
            "true_labels"
        ].long()

        self.binary_labels = data[
            "binary_labels"
        ].long()

        self.attack_ids = data[
            "attack_ids"
        ].long()

        self.source_indices = data[
            "source_indices"
        ].long()

        self.attack_success = data.get(
            "attack_success",
            torch.zeros(
                len(self.images),
                dtype=torch.bool,
            ),
        ).bool()

        if len(self.images) != len(
            self.true_labels
        ):
            raise ValueError(
                "The number of images and labels "
                "does not match."
            )

        self.random_crop = RandomCrop(
            size=32,
            padding=4,
            padding_mode="reflect",
        )

        self.random_flip = (
            RandomHorizontalFlip(p=0.5)
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].clone()

        if self.training:
            image = self.random_crop(image)
            image = self.random_flip(image)

        image = TF.normalize(
            image,
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        )

        return {
            "image": image,
            "label": self.true_labels[index],
            "binary_label": (
                self.binary_labels[index]
            ),
            "attack_id": self.attack_ids[index],
            "source_index": (
                self.source_indices[index]
            ),
            "attack_success": (
                self.attack_success[index]
            ),
        }


def inspect_dataset(name, dataset):
    print(f"\n{name.upper()} SPLIT")
    print("-" * 60)

    images = dataset.images
    labels = dataset.true_labels
    binary_labels = dataset.binary_labels
    attack_ids = dataset.attack_ids
    source_indices = dataset.source_indices

    print(f"Dataset path: {dataset.dataset_path}")
    print(f"Image shape: {tuple(images.shape)}")
    print(f"Image dtype: {images.dtype}")
    print(f"Minimum pixel: {images.min().item()}")
    print(f"Maximum pixel: {images.max().item()}")
    print(f"Contains NaN: {torch.isnan(images).any().item()}")
    print(
        f"Contains infinity: "
        f"{torch.isinf(images).any().item()}"
    )
    print(f"Total samples: {len(dataset)}")
    print(
        f"Unique source images: "
        f"{source_indices.unique().numel()}"
    )

    clean_count = int(
        (binary_labels == 0).sum().item()
    )

    adversarial_count = int(
        (binary_labels == 1).sum().item()
    )

    print(f"Clean samples: {clean_count}")
    print(
        f"Adversarial samples: "
        f"{adversarial_count}"
    )

    print("\nAttack distribution:")

    for attack_id, attack_name in (
        ATTACK_NAMES.items()
    ):
        count = int(
            (attack_ids == attack_id)
            .sum()
            .item()
        )

        print(f"  {attack_name}: {count}")

    print("\nClass distribution:")

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    for class_id, class_name in enumerate(
        class_names
    ):
        count = int(
            (labels == class_id)
            .sum()
            .item()
        )

        print(
            f"  {class_id} - "
            f"{class_name}: {count}"
        )

    if images.ndim != 4:
        raise ValueError(
            "Images must have shape "
            "[N, C, H, W]."
        )

    if images.shape[1:] != (3, 32, 32):
        raise ValueError(
            f"Unexpected image shape: "
            f"{tuple(images.shape[1:])}"
        )

    if images.min() < 0 or images.max() > 1:
        raise ValueError(
            "Images must be stored in the "
            "[0,1] pixel range."
        )

    if torch.isnan(images).any():
        raise ValueError(
            "Dataset contains NaN values."
        )

    if torch.isinf(images).any():
        raise ValueError(
            "Dataset contains infinite values."
        )

    if labels.min() < 0 or labels.max() > 9:
        raise ValueError(
            "True labels must be in the "
            "range 0 to 9."
        )

    print(f"\n{name} verification passed.")


def check_source_leakage(
    train_dataset,
    validation_dataset,
):
    train_sources = set(
        train_dataset.source_indices.tolist()
    )

    validation_sources = set(
        validation_dataset
        .source_indices
        .tolist()
    )

    overlap = (
        train_sources
        & validation_sources
    )

    print("\nSOURCE LEAKAGE CHECK")
    print("-" * 60)
    print(
        f"Training source images: "
        f"{len(train_sources)}"
    )
    print(
        f"Validation source images: "
        f"{len(validation_sources)}"
    )
    print(
        f"Overlapping source images: "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Train-validation source leakage "
            "was detected."
        )

    print(
        "No train-validation source leakage "
        "detected."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify datasets for training "
            "the protected classifier."
        )
    )

    parser.add_argument(
        "--train",
        default=(
            "adversarial_dataset/train.pt"
        ),
    )

    parser.add_argument(
        "--validation",
        default=(
            "adversarial_dataset/"
            "validation.pt"
        ),
    )

    args = parser.parse_args()

    train_dataset = ProtectionDataset(
        args.train,
        training=True,
    )

    validation_dataset = ProtectionDataset(
        args.validation,
        training=False,
    )

    inspect_dataset(
        "train",
        train_dataset,
    )

    inspect_dataset(
        "validation",
        validation_dataset,
    )

    check_source_leakage(
        train_dataset,
        validation_dataset,
    )

    sample = train_dataset[0]

    print("\nTRANSFORMED SAMPLE")
    print("-" * 60)
    print(
        f"Image shape: "
        f"{tuple(sample['image'].shape)}"
    )
    print(
        f"Image dtype: "
        f"{sample['image'].dtype}"
    )
    print(
        f"True label: "
        f"{sample['label'].item()}"
    )
    print(
        f"Attack type: "
        f"{ATTACK_NAMES[int(sample['attack_id'])]}"
    )

    print(
        "\nProtection dataset verification "
        "completed successfully."
    )


if __name__ == "__main__":
    main()