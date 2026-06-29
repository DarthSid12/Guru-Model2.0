"""
train.py

Train a single network (model.py, unchanged from the familiar-faces branch) on
the combined faces + houses + objects data under one unified softmax head.

Example:
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --categories faces objects \
        --lr 1e-3 --epochs 50 --variant lp

The classifier head (fc2) is just the training signal; the Yin/NIMBLE
simulation operates on the shared 256-d binary code `h` from fc1.
"""

import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Model
from datasets import make_datasets


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["faces", "objects"],
                    help="categories to train on jointly")
    ap.add_argument("--variant", choices=["lp", "cnn"], default="lp",
                    help="lp = log-polar/foveated, cnn = plain crop")
    ap.add_argument("--processed-root", default="processed_data")
    ap.add_argument("--num-fixations", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained-path", default=None,
                    help="optional checkpoint to warm-start from (loaded strict=False)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, num_fixations, num_classes, device):
    """Sum per-fixation logits over a base image's fixations, then argmax."""
    model.eval()
    accs = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels
        B, n, C, H, W = inputs.shape
        logits = model(inputs.reshape(-1, C, H, W))
        logits = logits.reshape(B, n, -1).sum(dim=1)
        preds = logits.argmax(dim=1)
        accs.append((preds == label_ids).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print("Device:", device, "| categories:", args.categories, "| variant:", args.variant)

    out_dir = args.output_dir or (
        f"runs/{'_'.join(args.categories)}_{args.variant}_"
        f"{args.num_fixations}fix_lr{args.lr}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------- Data -----------------------------
    datasets, label_map = make_datasets(
        categories=args.categories,
        processed_root=args.processed_root,
        variant=args.variant,
        num_salient_points=args.num_fixations,
    )
    num_classes = len(label_map)
    print(f"Unified label space: {num_classes} classes across {len(args.categories)} categories")
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    valid_batch_size = max(1, args.batch_size // args.num_fixations)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(datasets["valid"], batch_size=valid_batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(datasets["test"], batch_size=valid_batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ----------------------------- Model ----------------------------
    model = Model(size=180, num_classes=num_classes, pretrained=False, T=args.temperature).to(device)
    if args.pretrained_path:
        state = torch.load(args.pretrained_path, map_location=device)
        missing = model.load_state_dict(state, strict=False)
        print(f"Warm-started from {args.pretrained_path} ({missing})")
    model.stochastic = False  # deterministic expectation during training

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    ce_criterion = torch.nn.CrossEntropyLoss()

    best_val = -1.0
    patience_counter = 0
    best_path = os.path.join(out_dir, "best_model.pth")
    history = []

    # ----------------------------- Train ----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        epoch_losses = []
        pbar = tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch}/{args.epochs}", unit="img")
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels

            optimizer.zero_grad()
            logits = model(inputs)
            loss = ce_criterion(logits, label_ids)
            loss.backward()
            optimizer.step()

            correct += (logits.argmax(dim=1) == label_ids).sum().item()
            total += label_ids.size(0)
            epoch_losses.append(loss.item())
            pbar.update(inputs.size(0))
            pbar.set_postfix(acc=f"{correct/total*100:.2f}%", ce=f"{loss.item():.3f}")
        pbar.close()

        train_acc = correct / max(total, 1)
        train_loss = float(np.mean(epoch_losses))
        valid_acc, valid_std = evaluate(model, valid_loader, args.num_fixations, num_classes, device)
        test_acc, test_std = evaluate(model, test_loader, args.num_fixations, num_classes, device)
        print(f"-> Epoch {epoch}: train {train_acc*100:.2f}% | "
              f"valid {valid_acc*100:.2f}% | test(inv) {test_acc*100:.2f}% | loss {train_loss:.4f}")

        history.append({"epoch": epoch, "train_acc": train_acc, "train_loss": train_loss,
                        "valid_acc": valid_acc, "valid_std": valid_std,
                        "test_acc": test_acc, "test_std": test_std})

        if valid_acc > best_val:
            best_val = valid_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"   saved new best (valid {valid_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"   early stopping {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # ----------------------------- Save -----------------------------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(history).to_csv(os.path.join(out_dir, f"training_history_{ts}.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, f"final_model_{ts}.pth"))

    epochs = [h["epoch"] for h in history]
    plt.figure()
    plt.plot(epochs, [h["train_acc"] for h in history], label="train")
    plt.plot(epochs, [h["valid_acc"] for h in history], label="valid (upright)")
    plt.plot(epochs, [h["test_acc"] for h in history], label="test (inverted)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Training")
    plt.savefig(os.path.join(out_dir, "accuracy.png")); plt.close()
    print(f"\nDone. Best valid acc {best_val*100:.2f}%. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
