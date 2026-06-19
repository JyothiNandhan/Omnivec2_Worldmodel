"""
OmniVec2 Stage 1 — Main training script.

Interleaves RGB and LiDAR batches through the shared encoder each epoch.

Usage:
    python train.py --dataroot /path/to/nuscenes --output_dir ./results
    python train.py --help
"""
import os
import sys
import time
import random
import numpy as np
import torch
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure package imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .config import parse_args
    from .checkpointing import (
        load_training_checkpoint,
        make_checkpoint_state,
        save_stage_checkpoint_bundle,
    )
    from .model import OmniVec2Stage1
    from .data.build import build_dataloaders
    from .shared.losses import masked_mse_loss, reconstruction_psnr
    from .rgb.visualize import (
        visualize_rgb_reconstruction,
        visualize_patch_grid,
        visualize_token_embeddings,
        visualize_positional_encoding,
        visualize_patch_norms,
    )
    from .lidar.visualize import (
        visualize_lidar,
        visualize_fps_centers,
        visualize_patch_groups,
        visualize_lidar_token_similarity,
        visualize_lidar_token_norms_3d,
    )
except ImportError:
    from config import parse_args
    from checkpointing import (
        load_training_checkpoint,
        make_checkpoint_state,
        save_stage_checkpoint_bundle,
    )
    from model import OmniVec2Stage1
    from data.build import build_dataloaders
    from shared.losses import masked_mse_loss, reconstruction_psnr
    from rgb.visualize import (
        visualize_rgb_reconstruction,
        visualize_patch_grid,
        visualize_token_embeddings,
        visualize_positional_encoding,
        visualize_patch_norms,
    )
    from lidar.visualize import (
        visualize_lidar,
        visualize_fps_centers,
        visualize_patch_groups,
        visualize_lidar_token_similarity,
        visualize_lidar_token_norms_3d,
    )

from nuscenes.nuscenes import NuScenes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────── Reproducibility ────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_val_loss(rgb_val_loss, lidar_val_loss):
    return rgb_val_loss + lidar_val_loss


def maybe_resume_path(args):
    if args.resume:
        return args.resume
    auto_path = os.path.join(args.checkpoints_dir, "stage1_last.pth")
    if args.auto_resume and os.path.exists(auto_path):
        return auto_path
    return ""


# ─────────────────── Per-Epoch Functions ────────────────
def _train_rgb_batch(model, imgs, optimizer, mask_ratio, max_grad_norm):
    imgs = imgs.to(DEVICE)
    pred, target, mask = model.forward_rgb(imgs, mask_ratio)
    loss = masked_mse_loss(pred, target, mask)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item(), reconstruction_psnr(pred, target, mask), imgs.size(0)


def _train_lidar_batch(model, points, optimizer, mask_ratio, max_grad_norm):
    points = points.to(DEVICE)
    pred, target, mask, _, _ = model.forward_lidar(points, mask_ratio)
    loss = masked_mse_loss(pred, target, mask)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item(), reconstruction_psnr(pred, target, mask), points.size(0)


def train_one_epoch_interleaved(epoch, model, rgb_dl, lidar_dl, optimizer,
                                mask_ratio_rgb, mask_ratio_lidar, max_grad_norm,
                                rgb_batches_per_lidar=6):
    model.train()
    rgb_iter = iter(rgb_dl)
    lidar_iter = iter(lidar_dl)

    rgb_loss, rgb_psnr, rgb_n = 0.0, 0.0, 0
    lidar_loss, lidar_psnr, lidar_n = 0.0, 0.0, 0
    rgb_batches_done, lidar_batches_done = 0, 0
    total_batches = len(rgb_dl) + len(lidar_dl)

    pbar = tqdm(
        total=total_batches,
        desc=f"Ep{epoch} | Stage1 train 6RGB:1LiDAR",
        leave=False,
        file=sys.stdout,
        miniters=max(1, total_batches // 100),
    )

    while rgb_batches_done < len(rgb_dl) or lidar_batches_done < len(lidar_dl):
        for _ in range(rgb_batches_per_lidar):
            if rgb_batches_done >= len(rgb_dl):
                break
            imgs = next(rgb_iter)
            loss, psnr, n = _train_rgb_batch(
                model, imgs, optimizer, mask_ratio_rgb, max_grad_norm
            )
            rgb_loss += loss * n
            rgb_psnr += psnr * n
            rgb_n += n
            rgb_batches_done += 1
            pbar.update(1)
            pbar.set_postfix(
                rgb=f"{rgb_loss / max(rgb_n, 1):.4f}",
                lidar=f"{lidar_loss / max(lidar_n, 1):.4f}",
            )

        if lidar_batches_done < len(lidar_dl):
            points = next(lidar_iter)
            loss, psnr, n = _train_lidar_batch(
                model, points, optimizer, mask_ratio_lidar, max_grad_norm
            )
            lidar_loss += loss * n
            lidar_psnr += psnr * n
            lidar_n += n
            lidar_batches_done += 1
            pbar.update(1)
            pbar.set_postfix(
                rgb=f"{rgb_loss / max(rgb_n, 1):.4f}",
                lidar=f"{lidar_loss / max(lidar_n, 1):.4f}",
            )

    pbar.close()
    return (
        rgb_loss / max(rgb_n, 1),
        rgb_psnr / max(rgb_n, 1),
        lidar_loss / max(lidar_n, 1),
        lidar_psnr / max(lidar_n, 1),
    )


def train_one_epoch_rgb(epoch, model, dl, optimizer, mask_ratio, max_grad_norm):
    model.train()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    pbar = tqdm(dl, desc=f"Epoch{epoch} | RGB train", leave=False, file=sys.stdout, miniters=max(1, len(dl)//100))
    for imgs in pbar:
        imgs = imgs.to(DEVICE)
        pred, target, mask = model.forward_rgb(imgs, mask_ratio)
        loss = masked_mse_loss(pred, target, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_psnr += reconstruction_psnr(pred, target, mask) * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / max(n, 1):.4f}",
            psnr=f"{total_psnr / max(n, 1):.2f}",
        )
    return total_loss / max(n, 1), total_psnr / max(n, 1)


def train_one_epoch_lidar(epoch, model, dl, optimizer, mask_ratio, max_grad_norm):
    model.train()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    pbar = tqdm(dl, desc=f"Ep{epoch} | LiDAR train", leave=False, file=sys.stdout, miniters=max(1, len(dl)//100))
    for points in pbar:
        points = points.to(DEVICE)
        pred, target, mask, _, _ = model.forward_lidar(points, mask_ratio)
        loss = masked_mse_loss(pred, target, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * points.size(0)
        total_psnr += reconstruction_psnr(pred, target, mask) * points.size(0)
        n += points.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / max(n, 1):.4f}",
            psnr=f"{total_psnr / max(n, 1):.2f}",
        )
    return total_loss / max(n, 1), total_psnr / max(n, 1)


@torch.no_grad()
def validate_rgb(epoch, model, dl, mask_ratio):
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    pbar = tqdm(dl, desc=f"Ep{epoch} | RGB val", leave=False, file=sys.stdout, miniters=max(1, len(dl)//100))
    for imgs in pbar:
        imgs = imgs.to(DEVICE)
        pred, target, mask = model.forward_rgb(imgs, mask_ratio)
        loss = masked_mse_loss(pred, target, mask)
        total_loss += loss.item() * imgs.size(0)
        total_psnr += reconstruction_psnr(pred, target, mask) * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / max(n, 1):.4f}",
            psnr=f"{total_psnr / max(n, 1):.2f}",
        )
    return total_loss / max(n, 1), total_psnr / max(n, 1)


@torch.no_grad()
def validate_lidar(epoch, model, dl, mask_ratio):
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    pbar = tqdm(dl, desc=f"Ep{epoch} | LiDAR val", leave=False, file=sys.stdout, miniters=max(1, len(dl)//100))
    for points in pbar:
        points = points.to(DEVICE)
        pred, target, mask, _, _ = model.forward_lidar(points, mask_ratio)
        loss = masked_mse_loss(pred, target, mask)
        total_loss += loss.item() * points.size(0)
        total_psnr += reconstruction_psnr(pred, target, mask) * points.size(0)
        n += points.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / max(n, 1):.4f}",
            psnr=f"{total_psnr / max(n, 1):.2f}",
        )
    return total_loss / max(n, 1), total_psnr / max(n, 1)


# ─────────────────── Plot Training Curves ───────────────
def plot_curves(history, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    configs = [
        (0, 0, "rgb_train_loss",   "rgb_val_loss",   "RGB Loss (MSE)",   "Loss"),
        (0, 1, "rgb_train_psnr",   "rgb_val_psnr",   "RGB PSNR (dB)",    "PSNR"),
        (1, 0, "lidar_train_loss", "lidar_val_loss", "LiDAR Loss (MSE)", "Loss"),
        (1, 1, "lidar_train_psnr", "lidar_val_psnr", "LiDAR PSNR (dB)",  "PSNR")]
    for r, c, tk, vk, title, ylabel in configs:
        axes[r, c].plot(history[tk], label="Train", marker="o", ms=3)
        axes[r, c].plot(history[vk], label="Val",   marker="o", ms=3)
        axes[r, c].set_title(title)
        axes[r, c].set_xlabel("Epoch")
        axes[r, c].set_ylabel(ylabel)
        axes[r, c].legend()
        axes[r, c].grid(True, alpha=0.3)
    plt.suptitle("OmniVec2 Stage 1 — Multimodal Pretraining", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves → {path}")


# ─────────────────── Full Training Loop ─────────────────
def run_training(model, train_rgb_dl, val_rgb_dl,
                 train_lidar_dl, val_lidar_dl, args):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {k: [] for k in [
        "rgb_train_loss", "rgb_val_loss", "rgb_train_psnr", "rgb_val_psnr",
        "lidar_train_loss", "lidar_val_loss", "lidar_train_psnr", "lidar_val_psnr",
        "joint_val_loss"]}
    best_val_loss = float("inf")
    start_epoch = 1

    resume_path = maybe_resume_path(args)
    if resume_path:
        checkpoint = load_training_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=DEVICE,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        history = checkpoint.get("history", history)
        print(f"Resumed training from {resume_path} at epoch {checkpoint['epoch']}.")

    print("\n========== STAGE 1 — MULTIMODAL MASKED PRETRAINING ==========\n")

    if start_epoch > args.epochs:
        print("Requested epochs already completed in the resume checkpoint.")
        return history

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        rt_l, rt_p, lt_l, lt_p = train_one_epoch_interleaved(
            epoch,
            model,
            train_rgb_dl,
            train_lidar_dl,
            optimizer,
            args.mask_ratio_rgb,
            args.mask_ratio_lidar,
            args.max_grad_norm,
            rgb_batches_per_lidar=6,
        )
        rv_l, rv_p = validate_rgb(epoch, model, val_rgb_dl, args.mask_ratio_rgb)
        lv_l, lv_p = validate_lidar(epoch, model, val_lidar_dl, args.mask_ratio_lidar)

        scheduler.step()
        elapsed = time.time() - t0

        history["rgb_train_loss"].append(rt_l)
        history["rgb_val_loss"].append(rv_l)
        history["rgb_train_psnr"].append(rt_p)
        history["rgb_val_psnr"].append(rv_p)
        history["lidar_train_loss"].append(lt_l)
        history["lidar_val_loss"].append(lv_l)
        history["lidar_train_psnr"].append(lt_p)
        history["lidar_val_psnr"].append(lv_p)
        history["joint_val_loss"].append(aggregate_val_loss(rv_l, lv_l))

        joint_val_loss = history["joint_val_loss"][-1]
        is_best = joint_val_loss < best_val_loss
        if is_best:
            best_val_loss = joint_val_loss

        print(f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.0f}s) | "
              f"RGB: {rt_l:.4f}/{rv_l:.4f} ({rt_p:.1f}/{rv_p:.1f}dB) | "
              f"LiDAR: {lt_l:.4f}/{lv_l:.4f} ({lt_p:.1f}/{lv_p:.1f}dB) | "
              f"Joint val: {joint_val_loss:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")
        if epoch % args.save_every == 0:
            state = make_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                history=history,
                args=args,
            )
            save_stage_checkpoint_bundle(
                checkpoint_dir=args.checkpoints_dir,
                prefix="stage1",
                state=state,
                is_best=is_best,
                keep_last_n=args.keep_last_n_checkpoints,
            )
        sys.stdout.flush()

    print("\n✓ Multimodal pretraining complete.\n")
    return history


# ─────────────────── Main ───────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device      : {DEVICE}")
    print(f"NuScenes    : {args.version} @ {args.dataroot}")
    print(f"Output      : {args.output_dir}")
    print(f"Scene limit : {args.scene_limit if args.scene_limit > 0 else 'all'}")
    print(f"Train split : {args.train_split_ratio:.2f}")
    print(f"Epochs      : {args.epochs}  |  LR: {args.lr}  |  BS: {args.batch_size}")
    print(f"Checkpoints : {args.checkpoints_dir}")
    print(f"Exports     : {args.exports_dir}")
    print()
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.exports_dir, exist_ok=True)

    # ── Load NuScenes ──
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # ── Dataloaders ──
    train_rgb_dl, val_rgb_dl, train_lidar_dl, val_lidar_dl = \
        build_dataloaders(
            nusc,
            args.dataroot,
            args.batch_size,
            args.num_workers,
            split_ratio=args.train_split_ratio,
            scene_limit=args.scene_limit,
            seed=args.seed,
        )

    # ── Model ──
    model = OmniVec2Stage1().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: {n_params:.2f}M parameters")
    for name, module in [
        ("Shared encoder",  model.encoder),
        ("RGB tokenizer",   model.rgb_tokenizer),
        ("RGB decoder",     model.rgb_decoder),
        ("LiDAR tokenizer", model.lidar_patch_encoder),
        ("LiDAR pos embed", model.lidar_pos_embed),
        ("LiDAR decoder",   model.lidar_decoder)]:
        print(f"  {name:18s}: "
              f"{sum(p.numel() for p in module.parameters()) / 1e6:.2f}M")
    sys.stdout.flush()

    # ────────────────────────────────────────────────────
    #  PRE-TRAINING VISUALIZATIONS  (post-tokenizer)
    # ────────────────────────────────────────────────────
    if args.skip_pretraining_visuals:
        print("\n── Skipping pre-training visualizations ──")
    else:
        print("\n── Post-Tokenizer Visualizations ──")

        # RGB
        visualize_patch_grid(val_rgb_dl, args.output_dir)
        visualize_token_embeddings(model, val_rgb_dl, args.output_dir)
        visualize_positional_encoding(model, args.output_dir)
        visualize_patch_norms(model, val_rgb_dl, args.output_dir)

        # LiDAR
        visualize_fps_centers(val_lidar_dl, args.output_dir)
        visualize_patch_groups(val_lidar_dl, args.output_dir)
        visualize_lidar_token_similarity(model, val_lidar_dl, args.output_dir)
        visualize_lidar_token_norms_3d(model, val_lidar_dl, args.output_dir)

    # ────────────────────────────────────────────────────
    #  TRAINING
    # ────────────────────────────────────────────────────
    try:
        history = run_training(model, train_rgb_dl, val_rgb_dl,
                               train_lidar_dl, val_lidar_dl, args)
    except KeyboardInterrupt:
        print("\nTraining interrupted. The latest saved checkpoint can be used to resume.")
        raise

    # ────────────────────────────────────────────────────
    #  POST-TRAINING VISUALIZATIONS (reconstruction)
    # ────────────────────────────────────────────────────
    if args.skip_posttraining_visuals:
        print("\n── Skipping post-training visualizations ──")
    else:
        print("\n── Post-Training Visualizations ──")
        plot_curves(history, args.output_dir)
        visualize_rgb_reconstruction(model, val_rgb_dl, args.output_dir, args.mask_ratio_rgb)
        visualize_lidar(model, val_lidar_dl, args.output_dir, args.mask_ratio_lidar)

    # ────────────────────────────────────────────────────
    #  SAVE WEIGHTS
    # ────────────────────────────────────────────────────
    model.save_pretrained(
        os.path.join(args.exports_dir, "omnivec2_stage1_rgb_lidar.pth"))
    torch.save(model.state_dict(),
               os.path.join(args.exports_dir, "omnivec2_stage1_full.pth"))
    print("\n✔  All done!")


if __name__ == "__main__":
    main()
