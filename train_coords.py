"""
train_coords.py

Train a single network (model.py, unchanged from the familiar-faces branch) on
the combined faces + houses + objects data under one unified softmax head.

Two data modes:
  - packed (default when ./fixation_data exists): raw images + precomputed
    Gabor-saliency fixation coords (preprocess_fixations.py); crops are cut on
    the CPU and rotated/foveated/log-polar-transformed on the GPU on the fly.
    ~10 GB-scale sequential I/O per epoch instead of millions of PNG reads.
  - png: legacy pre-rendered crops from preprocess.py under ./processed_data.

Example:
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --categories faces objects houses \
        --lr 1e-3 --epochs 50 --variant lp

Each run writes into its output dir:
    config.json      the exact input configuration of the run
    summary.json     concise input + output (best/final metrics, wall time)
    best_model.pth / final_model_<ts>.pth / label_map.json /
    training_history_<ts>.csv / accuracy.png

The classifier head (fc2) is just the training signal; the Yin/NIMBLE
simulation operates on the shared 256-d binary code `h` from fc1.
"""

import argparse
import datetime
import json
import os
import socket
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_coords import ModelCoords
from datasets_coords import make_datasets, make_packed_datasets
from salience_trans import OnTheFlyTransform

# On DHONI, faces/objects/houses are already downloaded + preprocessed here.
# If no local ./processed_data exists, fall back to the shared copy instead
# of forcing a fresh (slow, disk-heavy) download.py + preprocess.py run.
DHONI_PROCESSED_ROOT = "/home/siagrawal/combined_lpnet/processed_data"
DHONI_FIXATION_ROOT = "/home/siagrawal/combined_lpnet/fixation_data"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["faces", "objects"],
                    help="categories to train on jointly")
    ap.add_argument("--variant", choices=["lp", "cnn"], default="lp",
                    help="lp = log-polar/foveated, cnn = plain crop")
    ap.add_argument("--data-mode", choices=["auto", "packed", "png"], default="auto",
                    help="packed = fixation_data (fast, on-the-fly transforms); "
                         "png = legacy pre-rendered processed_data crops; "
                         "auto = packed if fixation_data exists, else png")
    ap.add_argument("--fixation-root", default=None,
                    help="packed data root (default: ./fixation_data, falling back to "
                         f"{DHONI_FIXATION_ROOT} on DHONI)")
    ap.add_argument("--processed-root", default=None,
                    help="png-mode data root (default: ./processed_data, falling back to "
                         f"{DHONI_PROCESSED_ROOT} on DHONI)")
    ap.add_argument("--num-fixations", type=int, default=16)
    ap.add_argument("--max-images-per-class", nargs="+", default=[],
                    help="optional per-category cap on training base images, e.g. "
                         "--max-images-per-class objects=200 (train split only; "
                         "valid/test are unaffected)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                    help="cosine = CosineAnnealingLR decaying to 0 over --epochs")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW weight decay (0 = plain Adam behaviour)")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="bf16 autocast for model forward/backward (--no-amp to disable)")
    ap.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                    help="channels_last memory format for model + inputs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained-path", default=None,
                    help="optional checkpoint to warm-start from (loaded strict=False)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-tag", default=None,
                    help="suffix appended to the auto-named output dir (used by run_experiments.py)")
    ap.add_argument("--device", default="auto",
                    help="cuda:N | cpu | auto (auto picks the CUDA device with the most free memory)")
    return ap.parse_args()


def pick_free_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True)
        rows = [tuple(int(v) for v in line.split(",")) for line in out.strip().splitlines()]
        # nvidia-smi enumerates physical GPUs; respect CUDA_VISIBLE_DEVICES if set
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            allowed = [int(v) for v in visible.split(",") if v.strip() != ""]
            rows = [(allowed.index(i), m) for i, m in rows if i in allowed]
        idx = min(rows, key=lambda r: r[1])[0]
        return f"cuda:{idx}"
    except Exception:
        return "cuda:0"


@torch.no_grad()
def evaluate(model, loader, device, transform=None, amp=False, channels_last=False, invert_coords=False):
    """
    Coordinate fixation-summing evaluation.

    Dataset:
        inputs: (B,N,C,H,W)
        coords: (B,N,4) = x,y,dx,dy

    We discard dx,dy and only use:
        x,y

    Model:
        image + coordinate -> fixation logits

    Fixation logits are summed over N fixations.
    """

    model.eval()
    accs = []

    for inputs, coords, labels in loader:

        inputs = inputs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if invert_coords:
            coords = 1 - coords

        label_ids = (
            labels.argmax(dim=1)
            if labels.dim() > 1
            else labels
        )

        B, N, C, H, W = inputs.shape

        # remove delta coordinates
        coords = coords[:, :, :2]

        # flatten fixations
        inputs = inputs.reshape(B*N, C, H, W)
        coords = coords.reshape(B*N, 2)

        if transform is not None:
            inputs = transform(inputs)

        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda"
        ):

            logits = model(inputs, coords)

        # restore fixation dimension
        logits = logits.float().reshape(B, N, -1)

        # sum evidence over fixations
        logits = logits.sum(dim=1)

        preds = logits.argmax(dim=1)

        accs.append(
            (preds == label_ids).float().mean().item()
        )

    return float(np.mean(accs)), float(np.std(accs))


def resolve_root(explicit, local_default, dhoni_fallback):
    if explicit is not None:
        return explicit
    if not os.path.isdir(local_default) and "dhoni" in socket.gethostname().lower() \
            and os.path.isdir(dhoni_fallback):
        print(f"[info] No local ./{local_default} found; using shared DHONI data at {dhoni_fallback}.")
        return dhoni_fallback
    return local_default


def main():
    args = parse_args()
    args.processed_root = resolve_root(args.processed_root, "processed_data", DHONI_PROCESSED_ROOT)
    args.fixation_root = resolve_root(args.fixation_root, "fixation_data", DHONI_FIXATION_ROOT)
    if args.data_mode == "auto":
        args.data_mode = "packed" if os.path.isdir(args.fixation_root) else "png"
        print(f"[info] --data-mode auto -> {args.data_mode}")
    if args.device == "auto":
        args.device = pick_free_device()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # fixed 180x180 input -> autotune kernels
        torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+
    max_images_per_class = {}
    for spec in args.max_images_per_class:
        category, _, n = spec.partition("=")
        if not n:
            raise ValueError(f"--max-images-per-class expects category=N pairs, got {spec!r}")
        max_images_per_class[category] = int(n)
    print("Device:", device, "| categories:", args.categories, "| variant:", args.variant,
          "| data-mode:", args.data_mode, "| max_images_per_class:", max_images_per_class or "none")

    out_dir = args.output_dir or (
        f"runs/{'_'.join(args.categories)}_{args.variant}_"
        f"{args.num_fixations}fix_lr{args.lr}"
        + ("_" + "_".join(f"{c}{n}img" for c, n in sorted(max_images_per_class.items()))
           if max_images_per_class else "")
        + (f"_{args.run_tag}" if args.run_tag else "")
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "hostname": socket.gethostname(),
                   "started": datetime.datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)

    # ----------------------------- Data -----------------------------
    if args.data_mode == "packed":
        datasets, label_map = make_packed_datasets(
            categories=args.categories,
            packed_root=args.fixation_root,
            num_salient_points=args.num_fixations,
            max_images_per_class=max_images_per_class,
        )
        transforms = {
            "train": OnTheFlyTransform("train", args.variant, device).to(device),
            "valid": OnTheFlyTransform("valid", args.variant, device).to(device),
            "test": OnTheFlyTransform("test", args.variant, device).to(device),
        }
    else:
        datasets, label_map = make_datasets(
            categories=args.categories,
            processed_root=args.processed_root,
            variant=args.variant,
            num_salient_points=args.num_fixations,
            max_images_per_class=max_images_per_class,
        )
        transforms = {"train": None, "valid": None, "test": None}

    num_classes = len(label_map)
    print(f"Unified label space: {num_classes} classes across {len(args.categories)} categories")
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    valid_batch_size = max(1, args.batch_size // args.num_fixations)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(datasets["valid"], batch_size=valid_batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(datasets["test"], batch_size=valid_batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ----------------------------- Model ----------------------------
    model = ModelCoords(size=180, num_classes=num_classes, pretrained=False, T=args.temperature).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.pretrained_path:
        state = torch.load(args.pretrained_path, map_location=device)
        missing = model.load_state_dict(state, strict=False)
        print(f"Warm-started from {args.pretrained_path} ({missing})")
    model.stochastic = False  # deterministic expectation during training

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.lr_schedule == "cosine" else None)
    ce_criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = args.amp and device.type == "cuda"

    best_val = -1.0
    best_epoch = 0
    patience_counter = 0
    best_path = os.path.join(out_dir, "best_model.pth")
    history = []
    t_start = time.time()

    # ----------------------------- Train ----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        epoch_losses = []
        pbar = tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch}/{args.epochs}", unit="img")
        for inputs, coords, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            B, C, H, W = inputs.shape

            # remove dx/dy
            coords = coords[:, :2]

            if transforms["train"] is not None:
                with torch.no_grad():
                    inputs = transforms["train"](inputs)
            if args.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels

            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(inputs, coords)
                loss = ce_criterion(logits, label_ids)

            loss.backward()
            optimizer.step()

            correct += (logits.argmax(dim=1) == label_ids).sum().item()
            total += label_ids.size(0)
            epoch_losses.append(loss.item())
            pbar.update(B)
            pbar.set_postfix(acc=f"{correct/total*100:.2f}%", ce=f"{loss.item():.3f}")
        pbar.close()

        train_acc = correct / max(total, 1)
        train_loss = float(np.mean(epoch_losses))
        valid_acc, valid_std = evaluate(model, valid_loader, device, transforms["valid"],
                                        amp=use_amp, channels_last=args.channels_last)
        test_acc, test_std = evaluate(model, test_loader, device, transforms["test"],
                                      amp=use_amp, channels_last=args.channels_last, invert_coords=True)
        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        print(f"-> Epoch {epoch}: train {train_acc*100:.2f}% | "
              f"valid {valid_acc*100:.2f}% | test(inv) {test_acc*100:.2f}% | "
              f"loss {train_loss:.4f} | lr {cur_lr:.2e}")

        history.append({"epoch": epoch, "lr": cur_lr,
                        "train_acc": train_acc, "train_loss": train_loss,
                        "valid_acc": valid_acc, "valid_std": valid_std,
                        "test_acc": test_acc, "test_std": test_std,
                        "elapsed_sec": round(time.time() - t_start, 1)})

        if valid_acc > best_val:
            best_val = valid_acc
            best_epoch = epoch
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
    hist_csv = os.path.join(out_dir, f"training_history_{ts}.csv")
    pd.DataFrame(history).to_csv(hist_csv, index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, f"final_model_{ts}.pth"))

    epochs = [h["epoch"] for h in history]
    plt.figure()
    plt.plot(epochs, [h["train_acc"] for h in history], label="train")
    plt.plot(epochs, [h["valid_acc"] for h in history], label="valid (upright)")
    plt.plot(epochs, [h["test_acc"] for h in history], label="test (inverted)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Training")
    plt.savefig(os.path.join(out_dir, "accuracy.png")); plt.close()

    best_row = next(h for h in history if h["epoch"] == best_epoch) if history else {}
    summary = {
        "config": vars(args),
        "hostname": socket.gethostname(),
        "num_classes": num_classes,
        "dataset_sizes": {k: len(v) for k, v in datasets.items()},
        "results": {
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_valid_acc": best_val,
            "test_acc_at_best_epoch": best_row.get("test_acc"),
            "final_train_acc": history[-1]["train_acc"] if history else None,
            "final_valid_acc": history[-1]["valid_acc"] if history else None,
            "final_test_acc": history[-1]["test_acc"] if history else None,
            "wall_time_sec": round(time.time() - t_start, 1),
            "sec_per_epoch": round((time.time() - t_start) / max(len(history), 1), 1),
        },
        "artifacts": {
            "best_model": best_path,
            "history_csv": hist_csv,
            "label_map": os.path.join(out_dir, "label_map.json"),
        },
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Best valid acc {best_val*100:.2f}% (epoch {best_epoch}). Artifacts in {out_dir}")


if __name__ == "__main__":
    main()