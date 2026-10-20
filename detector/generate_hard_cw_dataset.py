import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from attacks import cw_l2_attack
from utils.victim_model import load_victim_model


CW_ATTACK_ID = 3

CIFAR10_MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465],
    dtype=torch.float32,
).view(1, 3, 1, 1)

CIFAR10_STD = torch.tensor(
    [0.2470, 0.2435, 0.2616],
    dtype=torch.float32,
).view(1, 3, 1, 1)


def select_clean_training_samples(dataset):
    images = dataset["images"].float()
    binary_labels = dataset["binary_labels"].long()
    attack_ids = dataset["attack_ids"].long()
    true_labels = dataset["true_labels"].long()
    source_indices = dataset["source_indices"].long()

    clean_mask = (
        (binary_labels == 0)
        & (attack_ids == 0)
    )

    clean_indices = torch.where(clean_mask)[0]

    return (
        images[clean_indices],
        true_labels[clean_indices],
        source_indices[clean_indices],
    )


def normalize_images(images, mean, std):
    return (images - mean) / std


def calculate_distortions(clean_images, adversarial_images):
    perturbations = adversarial_images - clean_images

    linf = (
        perturbations.abs()
        .flatten(start_dim=1)
        .max(dim=1)
        .values
    )

    l2 = (
        perturbations
        .flatten(start_dim=1)
        .norm(p=2, dim=1)
    )

    return linf, l2


def run_cw_attack(
    model,
    images,
    labels,
    mean,
    std,
    c,
    kappa,
    steps,
    learning_rate,
):
    adversarial_images = cw_l2_attack(
        model=model,
        images=images,
        labels=labels,
        mean=mean,
        std=std,
        c=c,
        kappa=kappa,
        steps=steps,
        learning_rate=learning_rate,
    )

    return adversarial_images.detach()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate successful low-L2 C&W attacks "
            "using only detector training images."
        )
    )

    parser.add_argument(
        "--train-dataset",
        default="adversarial_dataset/train.pt",
        help="Path to the existing detector training dataset.",
    )

    parser.add_argument(
        "--victim-checkpoint",
        default="model/resnet18_cifar10.pth",
        help="Path to the trained ResNet-18 checkpoint.",
    )

    parser.add_argument(
        "--output",
        default="adversarial_dataset/hard_cw_train.pt",
        help="Output path for the generated dataset.",
    )

    parser.add_argument(
        "--target-samples",
        type=int,
        default=100,
        help="Maximum number of hard C&W samples to retain.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--minimum-l2",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--maximum-l2",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    mean = CIFAR10_MEAN.to(device)
    std = CIFAR10_STD.to(device)

    print(f"Device: {device}")
    print(f"Training dataset: {args.train_dataset}")
    print(f"Victim checkpoint: {args.victim_checkpoint}")
    print(
        f"Required L2 range: "
        f"{args.minimum_l2} to {args.maximum_l2}"
    )

    train_data = torch.load(
        args.train_dataset,
        map_location="cpu",
    )

    (
        clean_images,
        true_labels,
        source_indices,
    ) = select_clean_training_samples(train_data)

    print(
        f"Available clean training images: "
        f"{len(clean_images)}"
    )

    victim_model = load_victim_model(
        args.victim_checkpoint,
        device,
    )

    victim_model.eval()

    for parameter in victim_model.parameters():
        parameter.requires_grad_(False)

    # Keep only images that the victim model classifies correctly.
    clean_predictions = []

    clean_loader = DataLoader(
        TensorDataset(
            clean_images,
            true_labels,
        ),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    with torch.inference_mode():
        for batch_images, batch_labels in clean_loader:
            batch_images = batch_images.to(device)

            normalized_images = normalize_images(
                batch_images,
                mean,
                std,
            )

            clean_logits = victim_model(
                normalized_images
            )

            batch_predictions = clean_logits.argmax(
                dim=1
            )

            clean_predictions.append(
                batch_predictions.cpu()
            )

    clean_predictions = torch.cat(
        clean_predictions
    )

    clean_correct_mask = (
        clean_predictions == true_labels
    )

    clean_images = clean_images[
        clean_correct_mask
    ]

    true_labels = true_labels[
        clean_correct_mask
    ]

    source_indices = source_indices[
        clean_correct_mask
    ]

    clean_predictions = clean_predictions[
        clean_correct_mask
    ]

    print(
        f"Correctly classified clean images: "
        f"{len(clean_images)}"
    )

    if len(clean_images) == 0:
        raise RuntimeError(
            "No correctly classified clean images were found. "
            "Check the victim-model normalization and checkpoint."
        )

    # Shuffle eligible images deterministically.
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    permutation = torch.randperm(
        len(clean_images),
        generator=generator,
    )

    clean_images = clean_images[
        permutation
    ]

    true_labels = true_labels[
        permutation
    ]

    source_indices = source_indices[
        permutation
    ]

    clean_predictions = clean_predictions[
        permutation
    ]

    # C&W parameter combinations.
    configurations = [
        {
            "c": 1.0,
            "kappa": 0.0,
            "steps": 50,
            "learning_rate": 0.01,
        },
    ]

    retained_images = []
    retained_true_labels = []
    retained_source_indices = []
    retained_clean_predictions = []
    retained_adversarial_predictions = []
    retained_linf = []
    retained_l2 = []
    retained_c = []
    retained_kappa = []
    retained_steps = []
    retained_learning_rate = []

    used_sources = set()

    source_dataset = TensorDataset(
        clean_images,
        true_labels,
        source_indices,
        clean_predictions,
    )

    for configuration in configurations:
        if len(retained_images) >= args.target_samples:
            break

        print("\nRunning configuration")
        print("-" * 50)
        print(f"c: {configuration['c']}")
        print(f"kappa: {configuration['kappa']}")
        print(f"steps: {configuration['steps']}")
        print(
            f"learning_rate: "
            f"{configuration['learning_rate']}"
        )

        loader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        progress = tqdm(
            loader,
            desc=f"C&W c={configuration['c']}",
        )

        for (
            batch_images,
            batch_labels,
            batch_sources,
            batch_clean_predictions,
        ) in progress:

            if len(retained_images) >= args.target_samples:
                break

            unused_mask = torch.tensor(
                [
                    int(source.item())
                    not in used_sources
                    for source in batch_sources
                ],
                dtype=torch.bool,
            )

            if not unused_mask.any():
                continue

            batch_images = batch_images[
                unused_mask
            ]

            batch_labels = batch_labels[
                unused_mask
            ]

            batch_sources = batch_sources[
                unused_mask
            ]

            batch_clean_predictions = (
                batch_clean_predictions[
                    unused_mask
                ]
            )

            batch_images_device = (
                batch_images.to(device)
            )

            batch_labels_device = (
                batch_labels.to(device)
            )

            adversarial_images = run_cw_attack(
                model=victim_model,
                images=batch_images_device,
                labels=batch_labels_device,
                mean=mean,
                std=std,
                c=configuration["c"],
                kappa=configuration["kappa"],
                steps=configuration["steps"],
                learning_rate=configuration[
                    "learning_rate"
                ],
            )

            adversarial_images = (
                adversarial_images.clamp(0.0, 1.0)
            )

            with torch.inference_mode():
                normalized_adversarial_images = (
                    normalize_images(
                        adversarial_images,
                        mean,
                        std,
                    )
                )

                adversarial_logits = victim_model(
                    normalized_adversarial_images
                )

                adversarial_predictions = (
                    adversarial_logits.argmax(dim=1)
                )

            attack_success = (
                adversarial_predictions
                != batch_labels_device
            )

            linf, l2 = calculate_distortions(
                batch_images_device,
                adversarial_images,
            )

            hard_attack_mask = (
                attack_success
                & (l2 >= args.minimum_l2)
                & (l2 <= args.maximum_l2)
            )

            selected_indices = torch.where(
                hard_attack_mask
            )[0]

            for selected_index in (
                selected_indices.tolist()
            ):
                if (
                    len(retained_images)
                    >= args.target_samples
                ):
                    break

                source_index = int(
                    batch_sources[
                        selected_index
                    ].item()
                )

                if source_index in used_sources:
                    continue

                used_sources.add(source_index)

                retained_images.append(
                    adversarial_images[
                        selected_index
                    ].cpu().half()
                )

                retained_true_labels.append(
                    batch_labels[
                        selected_index
                    ].cpu()
                )

                retained_source_indices.append(
                    batch_sources[
                        selected_index
                    ].cpu()
                )

                retained_clean_predictions.append(
                    batch_clean_predictions[
                        selected_index
                    ].cpu()
                )

                retained_adversarial_predictions.append(
                    adversarial_predictions[
                        selected_index
                    ].cpu()
                )

                retained_linf.append(
                    linf[
                        selected_index
                    ].cpu()
                )

                retained_l2.append(
                    l2[
                        selected_index
                    ].cpu()
                )

                retained_c.append(
                    configuration["c"]
                )

                retained_kappa.append(
                    configuration["kappa"]
                )

                retained_steps.append(
                    configuration["steps"]
                )

                retained_learning_rate.append(
                    configuration[
                        "learning_rate"
                    ]
                )

            progress.set_postfix(
                retained=len(retained_images)
            )

    if len(retained_images) == 0:
        raise RuntimeError(
            "No successful C&W attacks matched the selected "
            "L2 range. Try increasing --maximum-l2 or using "
            "more attack steps."
        )

    images_tensor = torch.stack(
        retained_images
    )

    true_labels_tensor = torch.stack(
        retained_true_labels
    ).long()

    source_indices_tensor = torch.stack(
        retained_source_indices
    ).long()

    clean_predictions_tensor = torch.stack(
        retained_clean_predictions
    ).long()

    adversarial_predictions_tensor = torch.stack(
        retained_adversarial_predictions
    ).long()

    linf_tensor = torch.stack(
        retained_linf
    ).float()

    l2_tensor = torch.stack(
        retained_l2
    ).float()

    sample_count = len(images_tensor)

    output_data = {
        "images": images_tensor,
        "binary_labels": torch.ones(
            sample_count,
            dtype=torch.long,
        ),
        "true_labels": true_labels_tensor,
        "attack_ids": torch.full(
            (sample_count,),
            CW_ATTACK_ID,
            dtype=torch.long,
        ),
        "source_indices": source_indices_tensor,
        "clean_correct": torch.ones(
            sample_count,
            dtype=torch.bool,
        ),
        "attack_success": torch.ones(
            sample_count,
            dtype=torch.bool,
        ),
        "clean_predictions": (
            clean_predictions_tensor
        ),
        "adversarial_predictions": (
            adversarial_predictions_tensor
        ),
        "linf": linf_tensor,
        "l2": l2_tensor,
        "cw_c": torch.tensor(
            retained_c,
            dtype=torch.float32,
        ),
        "cw_kappa": torch.tensor(
            retained_kappa,
            dtype=torch.float32,
        ),
        "cw_steps": torch.tensor(
            retained_steps,
            dtype=torch.long,
        ),
        "cw_learning_rate": torch.tensor(
            retained_learning_rate,
            dtype=torch.float32,
        ),
        "metadata": {
            "purpose": (
                "Successful low-L2 C&W detector training"
            ),
            "source_dataset": args.train_dataset,
            "minimum_l2": args.minimum_l2,
            "maximum_l2": args.maximum_l2,
            "target_samples": args.target_samples,
            "actual_samples": sample_count,
            "seed": args.seed,
        },
    }

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        output_data,
        output_path,
    )

    unique_sources = len(
        set(source_indices_tensor.tolist())
    )

    print("\nHARD C&W DATASET")
    print("-" * 60)
    print(f"Saved samples: {sample_count}")
    print(
        f"Image shape: "
        f"{tuple(images_tensor.shape)}"
    )
    print(f"Image dtype: {images_tensor.dtype}")
    print(
        f"Minimum pixel: "
        f"{images_tensor.min().item():.6f}"
    )
    print(
        f"Maximum pixel: "
        f"{images_tensor.max().item():.6f}"
    )
    print(
        f"Mean L2: "
        f"{l2_tensor.mean().item():.6f}"
    )
    print(
        f"Minimum L2: "
        f"{l2_tensor.min().item():.6f}"
    )
    print(
        f"Maximum L2: "
        f"{l2_tensor.max().item():.6f}"
    )
    print(
        f"Mean L∞: "
        f"{linf_tensor.mean().item():.6f}"
    )
    print(
        f"Unique source images: "
        f"{unique_sources}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()