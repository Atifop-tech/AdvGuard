import argparse
import random
from pathlib import Path

import torch
from torchvision import datasets, transforms

from detector.generate_dataset import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    process_split,
)

from utils.victim_model import load_victim_model


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an independent detector "
            "test dataset"
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
        default=1000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()

    if args.source_images < 3:
        raise ValueError(
            "source-images must be at least 3"
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

    # Important: train=False loads the independent
    # CIFAR-10 test split.
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    if args.source_images > len(test_dataset):
        raise ValueError(
            "source-images exceeds the size "
            "of the CIFAR-10 test set"
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    selected_indices = torch.randperm(
        len(test_dataset),
        generator=generator,
    )[:args.source_images].tolist()

    print(
        "Selected independent source images:",
        len(selected_indices),
    )

    process_split(
        split_name="test",
        source_indices=selected_indices,
        source_dataset=test_dataset,
        victim_model=victim_model,
        device=device,
        output_directory=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(
        "\nIndependent test dataset generated."
    )

    print(
        "Test tensor:",
        args.output_dir / "test.pt",
    )

    print(
        "Test metadata:",
        args.output_dir
        / "test_metadata.csv",
    )


if __name__ == "__main__":
    main()