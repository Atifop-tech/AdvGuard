from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


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


class AdversarialDetectorDataset(Dataset):
    def __init__(self, dataset_path):
        dataset_path = Path(dataset_path)

        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        data = torch.load(
            dataset_path,
            map_location="cpu",
            weights_only=True,
        )

        self.images = data["images"]
        self.binary_labels = data["binary_labels"]
        self.attack_ids = data["attack_ids"]
        self.attack_success = data["attack_success"]
        self.source_indices = data["source_indices"]

        self.normalize = transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        )

        if len(self.images) != len(
            self.binary_labels
        ):
            raise ValueError(
                "Images and labels have different lengths"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Stored tensors are float16 in raw pixel space.
        image = self.images[index].float()

        # Detector receives normalized images.
        image = self.normalize(image)

        label = self.binary_labels[index].float()

        return {
            "image": image,
            "label": label,
            "attack_id": self.attack_ids[index],
            "attack_success":
                self.attack_success[index],
            "source_index":
                self.source_indices[index],
        }