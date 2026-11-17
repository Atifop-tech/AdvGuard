import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from purification.dataset import PurificationPairDataset
from purification.model import PurificationNet


def gradient_loss(prediction, target):
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def reconstruction_loss(prediction, target):
    pixel = F.smooth_l1_loss(prediction, target)
    edges = gradient_loss(prediction, target)
    return pixel + 0.10 * edges


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    clean_abs = 0.0
    clean_count = 0
    adv_abs = 0.0
    adv_count = 0
    attack_counts = {1: [0.0, 0], 2: [0.0, 0], 3: [0.0, 0]}

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        output = model(images)
        loss = reconstruction_loss(output, targets)
        count = images.size(0)
        total_loss += loss.item() * count
        total += count

        per_image_mae = (output - targets).abs().flatten(1).mean(1)
        clean_mask = batch["is_clean"].bool()
        if clean_mask.any():
            clean_abs += per_image_mae[clean_mask].sum().item()
            clean_count += int(clean_mask.sum().item())
        adv_mask = ~clean_mask
        if adv_mask.any():
            adv_abs += per_image_mae[adv_mask].sum().item()
            adv_count += int(adv_mask.sum().item())
            for attack_id in (1, 2, 3):
                mask = adv_mask & (batch["attack_id"] == attack_id)
                if mask.any():
                    attack_counts[attack_id][0] += per_image_mae[mask].sum().item()
                    attack_counts[attack_id][1] += int(mask.sum().item())

    return {
        "val_loss": total_loss / max(total, 1),
        "clean_mae": clean_abs / max(clean_count, 1),
        "adversarial_mae": adv_abs / max(adv_count, 1),
        "fgsm_mae": attack_counts[1][0] / max(attack_counts[1][1], 1),
        "pgd_mae": attack_counts[2][0] / max(attack_counts[2][1], 1),
        "cw_mae": attack_counts[3][0] / max(attack_counts[3][1], 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the AdvGuard image purification network.")
    parser.add_argument("--train-dataset", default="adversarial_dataset/train.pt")
    parser.add_argument("--validation-dataset", default="adversarial_dataset/validation.pt")
    parser.add_argument("--output", default="model/purifier_best.pth")
    parser.add_argument("--history", default="results/purifier_training_history.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data = PurificationPairDataset(args.train_dataset, training=True)
    val_data = PurificationPairDataset(args.validation_dataset, training=False)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = PurificationNet(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    output_path = Path(args.output)
    history_path = Path(args.history)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    stale_epochs = 0
    history_exists = history_path.exists() and history_path.stat().st_size > 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = reconstruction_loss(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_count = images.size(0)
            running_loss += loss.item() * batch_count
            seen += batch_count
            progress.set_postfix(loss=f"{loss.item():.5f}")

        metrics = validate(model, val_loader, device)
        train_loss = running_loss / max(seen, 1)
        scheduler.step(metrics["val_loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            **metrics,
        }
        print(
            f"Epoch {epoch}: train={train_loss:.6f}, "
            f"val={metrics['val_loss']:.6f}, "
            f"clean_MAE={metrics['clean_mae']:.6f}, "
            f"adv_MAE={metrics['adversarial_mae']:.6f}, "
            f"FGSM={metrics['fgsm_mae']:.6f}, "
            f"PGD={metrics['pgd_mae']:.6f}, "
            f"CW={metrics['cw_mae']:.6f}"
        )

        with history_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            if not history_exists:
                writer.writeheader()
                history_exists = True
            writer.writerow(row)

        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_metrics": metrics,
                "model_type": "advguard_residual_image_purifier",
                "channels": args.channels,
                "blocks": args.blocks,
                "input_range": "[0,1]",
            }, output_path)
            print(f"Saved best purifier: {output_path}")
        else:
            stale_epochs += 1
            print(f"No validation improvement: {stale_epochs}/{args.patience}")

        if stale_epochs >= args.patience:
            print("Early stopping.")
            break

    print(f"Training complete. Best validation loss: {best_loss:.6f}")
    print(f"Checkpoint: {output_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
