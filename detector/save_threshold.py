from pathlib import Path

import torch


INPUT_CHECKPOINT = Path(
    "model/adversarial_detector.pth"
)

OUTPUT_CHECKPOINT = Path(
    "model/adversarial_detector_final.pth"
)

SELECTED_THRESHOLD = 0.26


def main():
    if not INPUT_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{INPUT_CHECKPOINT}"
        )

    checkpoint = torch.load(
        INPUT_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint["threshold"] = (
        SELECTED_THRESHOLD
    )

    checkpoint["threshold_selection"] = {
        "source": "validation_set",
        "criterion": (
            "maximum recall with "
            "false-positive rate <= 2%"
        ),
        "validation_recall": 0.985,
        "validation_false_positive_rate":
            0.013333,
        "validation_false_negative_rate":
            0.015,
        "validation_f1": 0.9858,
        "validation_roc_auc": 0.997444,
    }

    torch.save(
        checkpoint,
        OUTPUT_CHECKPOINT,
    )

    print(
        "Selected threshold:",
        checkpoint["threshold"],
    )

    print(
        "Final detector saved to:",
        OUTPUT_CHECKPOINT,
    )


if __name__ == "__main__":
    main()