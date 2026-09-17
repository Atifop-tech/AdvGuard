import argparse
import csv
import random
from pathlib import Path

import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from attacks import (
    fgsm_attack,
    pgd_attack,
    cw_l2_attack,
)

from utils.victim_model import load_victim_model


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

ATTACK_NAMES = [
    "fgsm",
    "pgd",
    "cw",
]


def create_attack_assignments(
    number_of_images,
    seed,
):
    """
    Assign FGSM, PGD and C&W approximately equally.
    """

    attacks = []

    for index in range(number_of_images):
        attacks.append(
            ATTACK_NAMES[index % len(ATTACK_NAMES)]
        )

    generator = random.Random(seed)
    generator.shuffle(attacks)

    return attacks


def generate_attack(
    attack_name,
    model,
    images,
    labels,
    mean,
    std,
):
    """
    Generate one type of adversarial attack.
    """

    if attack_name == "fgsm":
        return fgsm_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=8 / 255,
            mean=mean,
            std=std,
        )

    if attack_name == "pgd":
        return pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=8 / 255,
            alpha=2 / 255,
            steps=10,
            mean=mean,
            std=std,
            random_start=True,
        )

    if attack_name == "cw":
        return cw_l2_attack(
            model=model,
            images=images,
            labels=labels,
            mean=mean,
            std=std,
            c=1.0,
            kappa=0.0,
            steps=50,
            learning_rate=0.01,
        )

    raise ValueError(
        f"Unsupported attack: {attack_name}"
    )


def process_split(
    split_name,
    source_indices,
    source_dataset,
    victim_model,
    device,
    output_directory,
    batch_size,
    seed,
):
    """
    Create clean and adversarial samples for one split.

    Every source image creates:
        1 clean sample
        1 adversarial sample
    """

    attack_assignments = create_attack_assignments(
        len(source_indices),
        seed,
    )

    mean = torch.tensor(
        CIFAR10_MEAN,
        device=device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        CIFAR10_STD,
        device=device,
    ).view(1, 3, 1, 1)

    stored_images = []
    binary_labels = []
    true_labels = []
    attack_ids = []
    source_ids = []
    attack_success_values = []
    clean_correct_values = []
    clean_predictions_list = []
    adversarial_predictions_list = []
    linf_values = []
    l2_values = []

    metadata_rows = []

    attack_to_id = {
        "clean": 0,
        "fgsm": 1,
        "pgd": 2,
        "cw": 3,
    }

    progress = tqdm(
        range(0, len(source_indices), batch_size),
        desc=f"Generating {split_name}",
    )

    for batch_start in progress:
        batch_indices = source_indices[
            batch_start:batch_start + batch_size
        ]

        normalized_images = []
        labels = []

        for source_index in batch_indices:
            image, label = source_dataset[source_index]

            normalized_images.append(image)
            labels.append(label)

        normalized_images = torch.stack(
            normalized_images
        ).to(device)

        labels = torch.tensor(
            labels,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            clean_outputs = victim_model(
                normalized_images
            )

            clean_predictions = (
                clean_outputs.argmax(dim=1)
            )

        clean_correct = clean_predictions.eq(labels)

        # Convert clean images to raw pixel space.
        clean_pixels = (
            normalized_images * std + mean
        )

        clean_pixels = torch.clamp(
            clean_pixels,
            0.0,
            1.0,
        )

        batch_attack_names = attack_assignments[
            batch_start:
            batch_start + len(batch_indices)
        ]

        adversarial_normalized = torch.empty_like(
            normalized_images
        )

        # Process each attack type separately.
        for attack_name in ATTACK_NAMES:
            positions = [
                position
                for position, name
                in enumerate(batch_attack_names)
                if name == attack_name
            ]

            if not positions:
                continue

            position_tensor = torch.tensor(
                positions,
                dtype=torch.long,
                device=device,
            )

            selected_images = normalized_images[
                position_tensor
            ]

            selected_labels = labels[
                position_tensor
            ]

            generated = generate_attack(
                attack_name=attack_name,
                model=victim_model,
                images=selected_images,
                labels=selected_labels,
                mean=mean,
                std=std,
            )

            adversarial_normalized[
                position_tensor
            ] = generated

        with torch.no_grad():
            adversarial_outputs = victim_model(
                adversarial_normalized
            )

            adversarial_predictions = (
                adversarial_outputs.argmax(dim=1)
            )

        adversarial_pixels = (
            adversarial_normalized * std + mean
        )

        adversarial_pixels = torch.clamp(
            adversarial_pixels,
            0.0,
            1.0,
        )

        attack_success = (
            clean_correct
            & adversarial_predictions.ne(labels)
        )

        perturbation = (
            adversarial_pixels - clean_pixels
        )

        batch_linf = (
            perturbation
            .abs()
            .flatten(1)
            .max(dim=1)
            .values
        )

        batch_l2 = (
            perturbation
            .pow(2)
            .flatten(1)
            .sum(dim=1)
            .sqrt()
        )

        # Store the clean images.
        for local_index, source_index in enumerate(
            batch_indices
        ):
            clean_tensor_index = len(stored_images)

            stored_images.append(
                clean_pixels[local_index]
                .detach()
                .cpu()
                .half()
            )

            binary_labels.append(0)
            true_labels.append(
                labels[local_index].item()
            )

            attack_ids.append(
                attack_to_id["clean"]
            )

            source_ids.append(source_index)

            attack_success_values.append(False)

            clean_correct_values.append(
                clean_correct[local_index].item()
            )

            clean_prediction = (
                clean_predictions[local_index].item()
            )

            clean_predictions_list.append(
                clean_prediction
            )

            adversarial_predictions_list.append(
                clean_prediction
            )

            linf_values.append(0.0)
            l2_values.append(0.0)

            metadata_rows.append({
                "split": split_name,
                "tensor_index": clean_tensor_index,
                "source_index": source_index,
                "binary_label": 0,
                "attack_name": "clean",
                "true_label":
                    labels[local_index].item(),
                "clean_prediction":
                    clean_prediction,
                "adversarial_prediction":
                    clean_prediction,
                "clean_correct":
                    clean_correct[local_index].item(),
                "attack_success": False,
                "linf": 0.0,
                "l2": 0.0,
            })

        # Store the adversarial images.
        for local_index, source_index in enumerate(
            batch_indices
        ):
            adversarial_tensor_index = len(
                stored_images
            )

            attack_name = batch_attack_names[
                local_index
            ]

            stored_images.append(
                adversarial_pixels[local_index]
                .detach()
                .cpu()
                .half()
            )

            binary_labels.append(1)

            true_labels.append(
                labels[local_index].item()
            )

            attack_ids.append(
                attack_to_id[attack_name]
            )

            source_ids.append(source_index)

            attack_success_values.append(
                attack_success[local_index].item()
            )

            clean_correct_values.append(
                clean_correct[local_index].item()
            )

            clean_predictions_list.append(
                clean_predictions[local_index].item()
            )

            adversarial_predictions_list.append(
                adversarial_predictions[
                    local_index
                ].item()
            )

            linf_values.append(
                batch_linf[local_index].item()
            )

            l2_values.append(
                batch_l2[local_index].item()
            )

            metadata_rows.append({
                "split": split_name,
                "tensor_index":
                    adversarial_tensor_index,
                "source_index": source_index,
                "binary_label": 1,
                "attack_name": attack_name,
                "true_label":
                    labels[local_index].item(),
                "clean_prediction":
                    clean_predictions[
                        local_index
                    ].item(),
                "adversarial_prediction":
                    adversarial_predictions[
                        local_index
                    ].item(),
                "clean_correct":
                    clean_correct[
                        local_index
                    ].item(),
                "attack_success":
                    attack_success[
                        local_index
                    ].item(),
                "linf":
                    batch_linf[
                        local_index
                    ].item(),
                "l2":
                    batch_l2[
                        local_index
                    ].item(),
            })

    split_data = {
        # Raw pixel-space images in [0, 1].
        "images": torch.stack(stored_images),

        # 0 = clean, 1 = adversarial.
        "binary_labels": torch.tensor(
            binary_labels,
            dtype=torch.float32,
        ),

        "true_labels": torch.tensor(
            true_labels,
            dtype=torch.long,
        ),

        # 0=clean, 1=FGSM, 2=PGD, 3=C&W.
        "attack_ids": torch.tensor(
            attack_ids,
            dtype=torch.long,
        ),

        "source_indices": torch.tensor(
            source_ids,
            dtype=torch.long,
        ),

        "clean_correct": torch.tensor(
            clean_correct_values,
            dtype=torch.bool,
        ),

        "attack_success": torch.tensor(
            attack_success_values,
            dtype=torch.bool,
        ),

        "clean_predictions": torch.tensor(
            clean_predictions_list,
            dtype=torch.long,
        ),

        "adversarial_predictions": torch.tensor(
            adversarial_predictions_list,
            dtype=torch.long,
        ),

        "linf": torch.tensor(
            linf_values,
            dtype=torch.float32,
        ),

        "l2": torch.tensor(
            l2_values,
            dtype=torch.float32,
        ),
    }

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_path = (
        output_directory / f"{split_name}.pt"
    )

    torch.save(
        split_data,
        split_path,
    )

    metadata_path = (
        output_directory
        / f"{split_name}_metadata.csv"
    )

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metadata_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print(
        f"\nSaved {split_name} data: {split_path}"
    )

    print(
        f"Number of samples: "
        f"{len(stored_images)}"
    )

    print(
        f"Clean samples: "
        f"{binary_labels.count(0)}"
    )

    print(
        f"Adversarial samples: "
        f"{binary_labels.count(1)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate clean and adversarial data "
            "for detector training"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "model/resnet18_cifar10.pth"
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "adversarial_dataset"
        ),
    )

    parser.add_argument(
        "--source-images",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.source_images < 3:
        raise ValueError(
            "source-images must be at least 3"
        )

    if not 0 < args.validation_ratio < 1:
        raise ValueError(
            "validation-ratio must be between 0 and 1"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    victim_model = load_victim_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])

    source_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    if args.source_images > len(source_dataset):
        raise ValueError(
            "source-images is larger than "
            "the CIFAR-10 training set"
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    selected_indices = torch.randperm(
        len(source_dataset),
        generator=generator,
    )[:args.source_images].tolist()

    validation_count = int(
        args.source_images
        * args.validation_ratio
    )

    training_count = (
        args.source_images - validation_count
    )

    training_indices = selected_indices[
        :training_count
    ]

    validation_indices = selected_indices[
        training_count:
    ]

    print(
        f"Training source images: "
        f"{len(training_indices)}"
    )

    print(
        f"Validation source images: "
        f"{len(validation_indices)}"
    )

    process_split(
        split_name="train",
        source_indices=training_indices,
        source_dataset=source_dataset,
        victim_model=victim_model,
        device=device,
        output_directory=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    process_split(
        split_name="validation",
        source_indices=validation_indices,
        source_dataset=source_dataset,
        victim_model=victim_model,
        device=device,
        output_directory=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )

    print(
        "\nDetector dataset generation completed."
    )


if __name__ == "__main__":
    main()