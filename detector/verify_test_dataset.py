from pathlib import Path

from detector.verify_dataset import (
    load_split,
    verify_split,
)


def main():
    test_path = Path(
        "adversarial_dataset/test.pt"
    )

    test_data = load_split(test_path)

    verify_split(
        "independent test",
        test_data,
    )

    print(
        "\nIndependent test dataset "
        "verification completed successfully."
    )


if __name__ == "__main__":
    main()