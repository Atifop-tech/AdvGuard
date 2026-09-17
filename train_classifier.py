import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001

DATA_DIR = "./data"
MODEL_DIR = "./models"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# CIFAR-10 NORMALIZATION
# --------------------------------------------------

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)


# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean, std),
    ]
)


test_transform = transforms.Compose(
    [
        transforms.ToTensor(),

        transforms.Normalize(mean, std),
    ]
)


# --------------------------------------------------
# LOAD CIFAR-10
# --------------------------------------------------

train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform,
)


test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=test_transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)


# --------------------------------------------------
# CREATE RESNET-18
# --------------------------------------------------

model = models.resnet18(weights=None)


# CIFAR-10 images are only 32x32.
# Standard ResNet starts with a large 7x7 convolution,
# which is unnecessary for CIFAR-10.

model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
)


# Remove the first max pooling layer.
model.maxpool = nn.Identity()


# CIFAR-10 has 10 classes.
model.fc = nn.Linear(
    model.fc.in_features,
    10,
)


model = model.to(DEVICE)


# --------------------------------------------------
# LOSS + OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)


# --------------------------------------------------
# TRAIN FUNCTION
# --------------------------------------------------

def train_one_epoch(epoch):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            dim=1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


        accuracy = 100 * correct / total


        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{accuracy:.2f}%"
        )


    average_loss = (
        running_loss
        / len(train_loader)
    )


    return average_loss, accuracy


# --------------------------------------------------
# TEST FUNCTION
# --------------------------------------------------

def evaluate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    accuracy = 100 * correct / total

    return accuracy


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("===================================")
    print(" Adversarial Defense Project")
    print(" Training ResNet-18 on CIFAR-10")
    print("===================================")

    print(f"\nDevice: {DEVICE}\n")


    best_accuracy = 0


    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            epoch
        )


        test_accuracy = evaluate()


        print(
            f"\nEpoch {epoch + 1}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        print(
            f"Test Accuracy: {test_accuracy:.2f}%"
        )


        # Save best model.

        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            torch.save(
                model.state_dict(),
                os.path.join(
                    MODEL_DIR,
                    "resnet18_cifar10.pth"
                )
            )

            print(
                "Best model saved."
            )


        scheduler.step()


    print("\nTraining finished.")

    print(
        f"Best Test Accuracy: "
        f"{best_accuracy:.2f}%"
    )   