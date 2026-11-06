
import torch
import pandas as pd

from model import create_resnet18_cifar10

TEST_PATH = "adversarial_dataset/test.pt"
METADATA_PATH = "adversarial_dataset/test_metadata.csv"
VICTIM_PATH = "model/resnet18_cifar10.pth"

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load(TEST_PATH, map_location="cpu", weights_only=False)
metadata = pd.read_csv(METADATA_PATH)

model = create_resnet18_cifar10()
state = torch.load(VICTIM_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(state, strict=True)
model.to(device).eval()

images = data["images"].float()
labels = data["true_labels"].long()

# Use the same normalization as the previous evaluation.
images = (images - MEAN) / STD

predictions = []

with torch.inference_mode():
    for start in range(0, len(images), 64):
        batch = images[start:start + 64].to(device)
        predictions.extend(model(batch).argmax(1).cpu().tolist())

predictions = pd.Series(predictions)
labels_series = pd.Series(labels.tolist())
names = metadata["attack_name"].astype(str)

print("=== STORED VS RECOMPUTED PREDICTIONS ===")
for column in ["clean_prediction", "adversarial_prediction"]:
    if column not in metadata.columns:
        continue

    stored = metadata[column]
    matches = predictions.eq(stored)

    print(f"\nCompared with {column}:")
    print(f"Exact prediction agreement: {matches.mean() * 100:.2f}%")

    for attack in ["clean", "fgsm", "pgd", "cw"]:
        mask = names.eq(attack)
        if mask.any():
            print(
                f"  {attack}: "
                f"{matches[mask].mean() * 100:.2f}% agreement"
            )

print("\n=== VICTIM ACCURACY BY ATTACK ===")
for attack in ["clean", "fgsm", "pgd", "cw"]:
    mask = names.eq(attack)
    accuracy = predictions[mask].eq(labels_series[mask]).mean()
    print(f"{attack}: {accuracy * 100:.2f}%")

print("\n=== STORED ATTACK SUCCESS COUNTS ===")
print(pd.crosstab(
    metadata["attack_name"],
    metadata["attack_success"],
    margins=True
))

print("\n=== CLEAN-CORRECT COUNTS ===")
print(pd.crosstab(
    metadata["attack_name"],
    metadata["clean_correct"],
    margins=True
))

