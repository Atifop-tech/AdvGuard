from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


ATTACK_NAMES = {
    0: "clean",
    1: "fgsm",
    2: "pgd",
    3: "cw",
}


class PurificationPairDataset(Dataset):
    """
    Builds source-aligned (adversarial, clean) pairs from the
    existing AdvGuard tensor format. Each source contributes one
    adversarial-to-clean example and one clean identity example.
    """

    def __init__(self, dataset_path, training=False, include_identity=True):
        self.dataset_path = Path(dataset_path)
        self.training = training
        data = torch.load(self.dataset_path, map_location="cpu")

        required = ("images", "source_indices", "attack_ids", "binary_labels")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing keys in {self.dataset_path}: {missing}")

        images = data["images"].float()
        source_ids = data["source_indices"].long()
        attack_ids = data["attack_ids"].long()
        binary_labels = data["binary_labels"].long()
        attack_success = data.get(
            "attack_success", torch.zeros(len(images), dtype=torch.bool)
        ).bool()

        groups = {}
        for idx, source_id in enumerate(source_ids.tolist()):
            groups.setdefault(source_id, {})[int(binary_labels[idx].item())] = idx

        self.pairs = []
        for source_id, members in groups.items():
            if 0 not in members or 1 not in members:
                raise ValueError(
                    f"Source {source_id} must have one clean and one adversarial row."
                )
            clean_idx = members[0]
            adv_idx = members[1]
            attack_id = int(attack_ids[adv_idx].item())
            if attack_id not in (1, 2, 3):
                raise ValueError(
                    f"Unexpected attack id {attack_id} for source {source_id}."
                )
            self.pairs.append((
                images[adv_idx],
                images[clean_idx],
                attack_id,
                bool(attack_success[adv_idx].item()),
                source_id,
            ))

        if not self.pairs:
            raise ValueError(f"No clean/adversarial pairs found in {self.dataset_path}")

        self.images = images
        self.include_identity = include_identity

    def __len__(self):
        return len(self.pairs) * (2 if self.include_identity else 1)

    def __getitem__(self, index):
        pair_index = index % len(self.pairs)
        adversarial, clean, attack_id, attack_success, source_id = self.pairs[pair_index]

        is_clean = self.include_identity and index >= len(self.pairs)
        image = clean.clone() if is_clean else adversarial.clone()
        target = clean.clone()

        if self.training:
            # Apply identical spatial augmentation to input and target.
            stacked = torch.cat((image, target), dim=0)
            stacked = F.pad(stacked, (4, 4, 4, 4), mode="reflect")
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            stacked = stacked[:, top:top + 32, left:left + 32]
            if torch.rand(()) < 0.5:
                stacked = torch.flip(stacked, dims=(2,))
            image, target = stacked[:3], stacked[3:]

        return {
            "image": image,
            "target": target,
            "attack_id": 0 if is_clean else attack_id,
            "attack_success": False if is_clean else attack_success,
            "source_id": source_id,
            "is_clean": is_clean,
        }
