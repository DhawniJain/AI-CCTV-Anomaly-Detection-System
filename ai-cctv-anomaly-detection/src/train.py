"""Train the convolutional autoencoder on NORMAL frames only.

    python -m src.train                      # real dataset (paths.dataset_root in config.yaml)
    python -m src.train --smoke              # tiny/fast run on the real dataset (checks the pipeline)
    python -m src.train --synthetic          # software test on the SYNTHETIC fixture (not real results)
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.autoencoder import ConvAutoencoder, save_checkpoint
from src.dataset import get_manifest, make_loader, select_normal_frames, summarize_manifest
from src.preprocessing import tensor_to_bgr
from src.utils import ProjectError, ensure_dir, get_device, get_logger, load_config, resolve_path, save_json, set_seed

log = get_logger()


def _run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, _, _ in loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / max(1, n)


def save_reconstruction_grid(model, val_loader, device, path, n=8):
    model.eval()
    x, _, _ = next(iter(val_loader))
    x = x[:n].to(device)
    with torch.no_grad():
        x_hat, _ = model(x)
    top = np.hstack([tensor_to_bgr(a.cpu().numpy()) for a in x])
    bot = np.hstack([tensor_to_bgr(a.cpu().numpy()) for a in x_hat])
    cv2.imwrite(str(path), np.vstack([top, bot]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device", help="auto|cuda|mps|cpu")
    ap.add_argument("--rebuild-cache", action="store_true", help="Re-extract frames and rebuild the manifest")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, smoke=args.smoke, synthetic=args.synthetic)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    set_seed(cfg["data"]["seed"])
    device = get_device(args.device or cfg["training"]["device"])
    log.info("Device: %s | smoke=%s synthetic=%s", device, cfg["mode"]["smoke"], cfg["mode"]["synthetic"])
    if cfg["mode"]["synthetic"]:
        log.warning("SYNTHETIC fixture in use: results are for software testing only, never for the report.")

    df = get_manifest(cfg, force=args.rebuild_cache)
    counts = summarize_manifest(df)
    log.info("Split summary: %s", counts)
    train_df, val_df = select_normal_frames(df, "train", cfg), select_normal_frames(df, "val", cfg)
    if cfg["mode"]["smoke"] and not cfg["mode"]["synthetic"]:
        train_df, val_df = train_df.sample(min(256, len(train_df)), random_state=0), val_df.sample(min(64, len(val_df)), random_state=0)
    if len(train_df) == 0:
        raise ProjectError("No NORMAL training frames. Either label some data as normal, or set "
                           "data.assume_unlabeled_normal: true if the unlabeled footage is normal activity.")
    if len(val_df) == 0:
        raise ProjectError("No normal VALIDATION frames (needed for early stopping and threshold calibration). "
                           "Provide more videos/folders, or adjust data.split ratios / strategy.")

    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    model = ConvAutoencoder(1 if d["grayscale"] else 3, d["image_size"], m["base_channels"], m["latent_dim"]).to(device)
    log.info("Model parameters: %.2fM | train frames: %d | val frames: %d",
             sum(p.numel() for p in model.parameters()) / 1e6, len(train_df), len(val_df))
    train_loader = make_loader(train_df, cfg, shuffle=True)
    val_loader = make_loader(val_df, cfg, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    out_dir, models_dir = ensure_dir(cfg["paths"]["outputs_dir"]), ensure_dir(cfg["paths"]["models_dir"])
    ckpt_path = models_dir / "autoencoder.pt"
    history, best, bad, t0 = [], float("inf"), 0, time.time()
    for epoch in range(1, int(t["epochs"]) + 1):
        tr = _run_epoch(model, train_loader, device, opt)
        va = _run_epoch(model, val_loader, device)
        sched.step(va)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "lr": opt.param_groups[0]["lr"]})
        log.info("epoch %02d/%02d  train_mse=%.6f  val_mse=%.6f", epoch, t["epochs"], tr, va)
        if va < best - 1e-9:
            best, bad = va, 0
            save_checkpoint(model, ckpt_path, {
                "epoch": epoch, "val_loss": va, "image_size": d["image_size"], "grayscale": d["grayscale"],
                "synthetic_fixture": cfg["mode"]["synthetic"], "trained_at": datetime.now().isoformat(timespec="seconds")})
        else:
            bad += 1
            if bad >= t["early_stopping_patience"]:
                log.info("Early stopping (no val improvement for %d epochs).", bad)
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train")
        plt.plot(hist_df["epoch"], hist_df["val_loss"], label="validation")
        plt.xlabel("epoch"); plt.ylabel("MSE reconstruction loss"); plt.legend(); plt.grid(alpha=.3)
        plt.title("Autoencoder training (normal frames only)")
        plt.tight_layout(); plt.savefig(out_dir / "training_curve.png", dpi=150); plt.close()
    except Exception as e:  # plotting must never kill a finished training run
        log.warning("Could not save training curve: %s", e)

    # reload the BEST weights for the sample grid
    from src.autoencoder import load_checkpoint
    best_model, _ = load_checkpoint(ckpt_path, device)
    save_reconstruction_grid(best_model, val_loader, device, out_dir / "reconstruction_samples.png")
    save_json({"device": str(device), "epochs_run": len(history), "best_val_mse": best, "seconds": round(time.time() - t0, 1),
               "split_counts": counts, "n_train_normal": len(train_df), "n_val_normal": len(val_df),
               "synthetic_fixture": cfg["mode"]["synthetic"], "smoke": cfg["mode"]["smoke"],
               "config_snapshot": {"data": d, "model": m, "training": t}}, out_dir / "train_info.json")
    log.info("Done. Best val MSE %.6f. Checkpoint: %s", best, ckpt_path)
    log.info("NEXT: python -m src.calibrate_threshold%s", " --synthetic" if cfg["mode"]["synthetic"] else
             (" --smoke" if cfg["mode"]["smoke"] else ""))


if __name__ == "__main__":
    main()
