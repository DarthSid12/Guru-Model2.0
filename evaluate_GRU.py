import torch
import argparse

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
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="bf16 autocast for model forward/backward (--no-amp to disable)")
    ap.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                    help="channels_last memory format for model + inputs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default="auto",
                    help="cuda:N | cpu | auto (auto picks the CUDA device with the most free memory)")

    ap.add_argument("--cnn-full-image", action="store_true", help="train fully classical cnn without fixations")
    # this is useful for the thatcherization experiment, but less so for yin

    ap.add_argument("--checkpoint", required=True)

    return ap.parse_args()

@torch.no_grad()
def evaluate(model, loader, device, transform=None, amp=False, channels_last=False, inverted=False):
    """
    Evaluate one prediction per image.

    Dataset returns:
        inputs : (B, N, C, H, W)
        coords : (B, N, 4)
        labels : one-hot or class indices

    Model returns:
        logits : (B, num_classes)
    """

    model.eval()
    accs = []

    for inputs, coords, labels in loader:
        inputs = inputs.to(device, non_blocking=True)

        coords = coords.to(device, non_blocking=True)

        if inverted:
            coords *= -1

        labels = labels.to(device, non_blocking=True)

        label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels

        # Apply crop transform to every fixation independently
        if transform is not None:
            B, N, C, H, W = inputs.shape
            inputs = inputs.reshape(B * N, C, H, W)
            inputs = transform(inputs)
            inputs = inputs.reshape(B, N, C, H, W)

        if channels_last:
            B, N, C, H, W = inputs.shape
            inputs = (
                inputs.reshape(B * N, C, H, W)
                      .contiguous(memory_format=torch.channels_last)
                      .reshape(B, N, C, H, W)
            )

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(inputs, coords)

        preds = logits.argmax(dim=1)
        accs.append((preds == label_ids).float().mean().item())

    return float(np.mean(accs)), float(np.std(accs))

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

    # ----------------------------- Data -----------------------------
    if args.data_mode == "packed":
        datasets, label_map = make_packed_datasets_coords(
            categories=args.categories,
            packed_root=args.fixation_root,
            num_salient_points=args.num_fixations,
            max_images_per_class=max_images_per_class,
            full_image = args.cnn_full_image
        )

        # in case we don't want cnn fixations
        if args.variant == "cnn" and args.cnn_full_image:
            transforms = {
                "valid": None,
                "test": None,
            }
        else:
            transforms = {
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
            full_image = args.cnn_full_image
        )
        transforms = {"valid": None, "test": None}

    num_classes = len(label_map)
    print(f"Unified label space: {num_classes} classes across {len(args.categories)} categories")
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    if args.variant == "cnn" and args.cnn_full_image:
        valid_batch_size = args.batch_size
    else:
        valid_batch_size = max(1, args.batch_size // args.num_fixations)

    valid_loader = DataLoader(datasets["valid"], batch_size=valid_batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(datasets["test"], batch_size=valid_batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ----------------------------- Model ----------------------------
    model = ModelCoordsGRU(size=180, num_classes=num_classes, pretrained=False, T=args.temperature).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.stochastic = False  

    use_amp = args.amp and device.type == "cuda"

    valid_acc, valid_std = evaluate(model, valid_loader, device, transforms["valid"],
                                        amp=use_amp, channels_last=args.channels_last, inverted=False)
    test_acc, test_std = evaluate(model, test_loader, device, transforms["test"],
                                      amp=use_amp, channels_last=args.channels_last, inverted=True)
    print(f"valid {valid_acc*100:.2f}% | test(inv) {test_acc*100:.2f}% | ")

if __name__ == "__main__":
    main()