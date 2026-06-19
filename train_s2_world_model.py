"""
OmniVec2 Stage 2 → World Model Training Script.

Pipeline:
  Stage 2 (frozen) → TemporalTokenPredictor (TRAINED)

The Stage 2 checkpoint already contains all Stage 1 weights inside it
(rgb_tokenizer, f/encoder, lidar_patch_encoder, etc.), so only the Stage 2
checkpoint is needed. No separate Stage 1 checkpoint required.

For each history frame, the input is encoded through the full Stage 2 pipeline
to produce g(.) tokens. The TemporalTokenPredictor then predicts the g(.) tokens
for the next frame. Loss = MSE(predicted g(.) vs real g(.)).

Example (HiperGator):
    sbatch run_s2_world_model.sh

Example (local test — 2 scenes, 1 epoch):
    python train_s2_world_model.py \\
        --dataroot /orange/iruchkin/isen/nsfull \\
        --stage2_checkpoint ./runs/stage2/checkpoints/stage2_best.pth \\
        --output_dir ./runs/s2_world_model \\
        --scene_limit 2 --epochs 1
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Make the package root importable ─────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIRNAME,
    EPOCHS,
    FC_HIDDEN,
    LR,
    NUSCENES_DATAROOT,
    NUSCENES_VERSION,
    NUM_WORKERS,
    SCENE_LIMIT,
    SEED,
    TRAIN_SPLIT_RATIO,
    WEIGHT_DECAY,
)
from model import OmniVec2Stage1  # noqa: E402
from stage2 import OmniVec2Stage2  # noqa: E402
from s2_world_model.temporal_dataset import build_world_model_dataloaders  # noqa: E402
from s2_world_model.world_model_s2 import (  # noqa: E402
    OmniVec2Stage2WorldModel,
    TemporalTokenPredictor,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────── Arguments ───────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OmniVec2 Stage 2 → World Model")
    p.add_argument("--dataroot",          type=str,   default=NUSCENES_DATAROOT)
    p.add_argument("--version",           type=str,   default=NUSCENES_VERSION)
    p.add_argument("--stage2_checkpoint", type=str,   required=True,
                   help="Stage 2 .pth checkpoint — already contains all Stage 1 weights.")
    p.add_argument("--output_dir",        type=str,   default="./runs/s2_world_model")
    p.add_argument("--checkpoints_dir",   type=str,   default=None)
    p.add_argument("--epochs",            type=int,   default=EPOCHS)
    p.add_argument("--lr",                type=float, default=LR)
    p.add_argument("--weight_decay",      type=float, default=WEIGHT_DECAY)
    p.add_argument("--batch_size",        type=int,   default=BATCH_SIZE)
    p.add_argument("--num_workers",       type=int,   default=NUM_WORKERS)
    p.add_argument("--scene_limit",       type=int,   default=SCENE_LIMIT)
    p.add_argument("--train_split_ratio", type=float, default=TRAIN_SPLIT_RATIO)
    p.add_argument("--seed",              type=int,   default=SEED)
    p.add_argument("--history",           type=int,   default=4,
                   help="Number of past frames fed as context.")
    p.add_argument("--steps_ahead",       type=int,   default=2,
                   help="Number of steps after last input T to predict. "
                        "At nuScenes 2 Hz (0.5s/step), steps_ahead=2 predicts T+2 = 1.0s ahead of T. ")
    p.add_argument("--temporal_layers",   type=int,   default=4)
    p.add_argument("--temporal_heads",    type=int,   default=4)
    p.add_argument("--max_grad_norm",     type=float, default=1.0)
    p.add_argument("--save_every",        type=int,   default=1)
    p.add_argument("--resume",            type=str,   default="")
    p.add_argument("--auto_resume",       action="store_true")
    p.add_argument("--num_vis_samples",   type=int,   default=4)
    p.add_argument("--skip_visualizations", action="store_true")
    args = p.parse_args()
    if args.checkpoints_dir is None:
        args.checkpoints_dir = os.path.join(args.output_dir, CHECKPOINT_DIRNAME)
    return args


# ─────────────────────────── Utilities ───────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stage2_checkpoint(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        return
    model.load_state_dict(state)


def maybe_resume_path(args):
    if args.resume:
        return args.resume
    auto = os.path.join(args.checkpoints_dir, "wm_s2_last.pth")
    return auto if (args.auto_resume and os.path.exists(auto)) else ""


def save_checkpoint(args, state, is_best):
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    torch.save(state, os.path.join(args.checkpoints_dir, "wm_s2_last.pth"))
    if is_best:
        torch.save(state, os.path.join(args.checkpoints_dir, "wm_s2_best.pth"))


# ─────────────────────────── Training ────────────────────────────────────────

def run_epoch(model, loader, optimizer, args, train: bool):
    model.train(train)
    tot = tot_rgb = tot_lid = 0.0
    steps = 0
    pbar = tqdm(loader, desc="Train" if train else "Val  ", leave=False, file=sys.stdout)

    for batch in pbar:
        rgb_seq   = batch["rgb_sequence"].to(DEVICE)
        lid_seq   = batch["lidar_sequence"].to(DEVICE)
        rgb_tgt   = batch["rgb_target"].to(DEVICE)
        lid_tgt   = batch["lidar_target"].to(DEVICE)

        with torch.set_grad_enabled(train):
            out = model(rgb_seq, lid_seq)
            with torch.no_grad():
                tgt_rgb_g, tgt_lid_g = model.encode_target(rgb_tgt, lid_tgt)
                # Extract frame T tokens to act as negative samples for contrastive loss
                base_rgb_g, base_lid_g = model.encode_target(rgb_seq[:, -1], lid_seq[:, -1])

            loss_rgb_mse = F.mse_loss(out["pred_rgb_tokens"], tgt_rgb_g)
            loss_lid_mse = F.mse_loss(out["pred_lidar_tokens"], tgt_lid_g)
            
            # Ground truth color statistics for the future frame
            from rgb.patches import patchify
            tgt_rgb_patches = patchify(rgb_tgt)
            true_mean = tgt_rgb_patches.mean(dim=-1, keepdim=True)
            true_var  = tgt_rgb_patches.var(dim=-1, keepdim=True)
            
            # Color prediction losses
            loss_color_mean = F.mse_loss(out["pred_rgb_mean"], true_mean)
            loss_color_var  = F.mse_loss(out["pred_rgb_var"], true_var)

            # Combine MSE and Color
            loss_rgb = loss_rgb_mse + loss_color_mean + loss_color_var
            loss_lid = loss_lid_mse
            loss = loss_rgb + loss_lid

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                optimizer.step()

        tot     += loss.item()
        tot_rgb += loss_rgb.item()
        tot_lid += loss_lid.item()
        steps   += 1
        pbar.set_postfix(
            loss=f"{tot/steps:.4f}",
            rgb=f"{tot_rgb/steps:.4f}",
            lidar=f"{tot_lid/steps:.4f}",
        )

    d = max(steps, 1)
    return tot / d, tot_rgb / d, tot_lid / d


def train(model, train_dl, val_dl, args):
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    log = {k: [] for k in [
        "train_loss", "train_rgb_loss", "train_lidar_loss",
        "val_loss",   "val_rgb_loss",   "val_lidar_loss",
    ]}
    best_val  = float("inf")
    start_ep  = 1

    resume = maybe_resume_path(args)
    if resume:
        try:
            ckpt = torch.load(resume, map_location=DEVICE, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_ep = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        log      = ckpt.get("history", log)
        print(f"Resumed from {resume} at epoch {ckpt['epoch']}.")

    print(f"\n{'='*50}")
    print("  OmniVec2  Stage 2 → World Model")
    print(f"  Stage 2 backbone : FROZEN")
    print(f"  Trainable params : {sum(p.numel() for p in params):,}")
    print(f"  Epochs           : {args.epochs}")
    print(f"{'='*50}\n")

    for ep in range(start_ep, args.epochs + 1):
        t0 = time.time()
        tr,  tr_r,  tr_l  = run_epoch(model, train_dl, optimizer, args, train=True)
        val, val_r, val_l = run_epoch(model, val_dl,   optimizer, args, train=False)

        log["train_loss"].append(tr);   log["train_rgb_loss"].append(tr_r)
        log["train_lidar_loss"].append(tr_l)
        log["val_loss"].append(val);    log["val_rgb_loss"].append(val_r)
        log["val_lidar_loss"].append(val_l)

        is_best = val < best_val
        if is_best:
            best_val = val

        print(
            f"Ep {ep:03d}/{args.epochs} ({time.time()-t0:.0f}s) | "
            f"train {tr:.4f} (rgb {tr_r:.4f} lid {tr_l:.4f}) | "
            f"val {val:.4f} (rgb {val_r:.4f} lid {val_l:.4f})"
            + (" ★ best" if is_best else "")
        )

        if ep % args.save_every == 0:
            save_checkpoint(args, {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": ep, "best_val_loss": best_val,
                "history": log, "args": vars(args),
            }, is_best)

    return log


# ─────────────────────────── Visualization ───────────────────────────────────

def _vis_dir(output_dir):
    p = os.path.join(output_dir, "visualizations")
    os.makedirs(p, exist_ok=True)
    return p


def plot_curves(log, output_dir):
    vd = _vis_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        ("train_loss",       "val_loss",       "Total g(.) Token Loss"),
        ("train_rgb_loss",   "val_rgb_loss",   "RGB g(.) Token Loss"),
        ("train_lidar_loss", "val_lidar_loss", "LiDAR g(.) Token Loss"),
    ]
    for ax, (tk, vk, title) in zip(axes, pairs):
        ax.plot(log.get(tk, []), label="train", marker="o", ms=3)
        ax.plot(log.get(vk, []), label="val",   marker="o", ms=3)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("OmniVec2 Stage 2 → World Model — Training Curves")
    fig.tight_layout()
    out = os.path.join(vd, "s2_wm_training_curves.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[VIS] Training curves → {out}")


@torch.no_grad()
def visualize(model, val_dl, args):
    """
    Honest diagnostic visualizations with naive copy-last-frame baseline.

    Every metric is shown alongside the baseline so the model's actual
    contribution to prediction quality is clear.
    Ground-truth images are labelled explicitly as reference, NOT predictions.
    The predicted tokens are also decoded back to pixel-space for visual inspection.
    """
    import math
    try:
        from sklearn.decomposition import PCA; HAS_PCA = True
    except ImportError:
        HAS_PCA = False

    from rgb.patches import patchify, unpatchify

    vd = _vis_dir(args.output_dir)
    model.eval()
    batch = next(iter(val_dl))
    rgb_seq = batch["rgb_sequence"].to(DEVICE)
    lid_seq = batch["lidar_sequence"].to(DEVICE)
    rgb_tgt = batch["rgb_target"].to(DEVICE)
    lid_tgt = batch["lidar_target"].to(DEVICE)

    # Encode history into g(.) space
    rgb_g_seq, lid_g_seq = model.encode_sequence(rgb_seq, lid_seq)

    # Model prediction
    out      = model.temporal(rgb_g_seq, lid_g_seq)
    pred_rgb = out["pred_rgb_tokens"]
    pred_lid = out["pred_lidar_tokens"]

    # Naive baseline: copy last history frame's tokens
    base_rgb = rgb_g_seq[:, -1].clone()
    base_lid = lid_g_seq[:, -1].clone()

    # Ground truth T+1
    tgt_rgb_g, tgt_lid_g = model.encode_target(rgb_tgt, lid_tgt)

    # Errors and cosine
    m_rgb_err = ((pred_rgb - tgt_rgb_g)**2).mean(-1)
    b_rgb_err = ((base_rgb - tgt_rgb_g)**2).mean(-1)
    m_lid_err = ((pred_lid - tgt_lid_g)**2).mean(-1)
    b_lid_err = ((base_lid - tgt_lid_g)**2).mean(-1)
    m_rgb_cos = F.cosine_similarity(pred_rgb, tgt_rgb_g, dim=-1)
    b_rgb_cos = F.cosine_similarity(base_rgb, tgt_rgb_g, dim=-1)
    m_lid_cos = F.cosine_similarity(pred_lid, tgt_lid_g, dim=-1)
    b_lid_cos = F.cosine_similarity(base_lid, tgt_lid_g, dim=-1)

    m_rgb_mse = F.mse_loss(pred_rgb, tgt_rgb_g).item()
    b_rgb_mse = F.mse_loss(base_rgb, tgt_rgb_g).item()
    m_lid_mse = F.mse_loss(pred_lid, tgt_lid_g).item()
    b_lid_mse = F.mse_loss(base_lid, tgt_lid_g).item()

    n = min(args.num_vis_samples, rgb_seq.shape[0])
    T = rgb_seq.shape[1]
    N = pred_rgb.shape[1]
    grid = int(math.isqrt(N))

    # ── Decode predicted tokens back to an image ─────────────────────────────
    # We decode purely from the predicted g(.) tokens.
    rgb_refined_pred = model.stage2.ca_rgb_back.norm_out(pred_rgb)
    ids_restore = torch.arange(N, device=DEVICE).unsqueeze(0).expand(rgb_seq.shape[0], N)
    pred_patches = model.stage2.rgb_decoder(rgb_refined_pred, ids_restore)

    # 1. RAW STRUCTURAL PREDICTION (ignores untrained color heads)
    # This allows us to see the movement physics clearly while the color heads are still learning.
    raw_pred_imgs = unpatchify(pred_patches)
    raw_pred_imgs = (raw_pred_imgs + 2.0) / 4.0  # Scale N(0,1) to [0, 1]

    # 2. COLORIZED PREDICTION (uses predicted color statistics)
    # This will look like static during early epochs until the heads learn to predict color!
    pred_mean = out["pred_rgb_mean"]
    pred_var  = out["pred_rgb_var"]
    color_pixels = pred_patches * (pred_var + 1e-6).sqrt() + pred_mean
    color_pred_imgs = unpatchify(color_pixels)

    # ── 1. Context strip with Model Prediction ───────────────────────────────
    for i in range(n):
        # We now need T + 3 columns: History (T), Raw Structure (1), Colorized (1), Ground Truth (1)
        fig, axes = plt.subplots(1, T + 3, figsize=(3.2*(T+3), 3.6))
        
        for t in range(T):
            img = rgb_seq[i, t].permute(1,2,0).cpu().numpy()
            axes[t].imshow(np.clip(img, 0, 1))
            
            # Label frames as T-3, T-2, T-1, T (where T is the last input frame)
            offset = T - 1 - t
            title_str = f"History T-{offset}" if offset > 0 else "History T (Last Input)"
            axes[t].set_title(title_str, fontsize=9)
            axes[t].axis("off")
            
        pred_frame_label = args.steps_ahead              # e.g., 2 -> T+2
        time_ahead_s     = args.steps_ahead * 0.5        # 2 × 0.5s = 1.0s
        
        # Plot Raw Structure Prediction
        raw_img = raw_pred_imgs[i].permute(1,2,0).cpu().numpy()
        axes[-3].imshow(np.clip(raw_img, 0, 1))
        axes[-3].set_title(
            f"RAW STRUCTURE T+{pred_frame_label}\n"
            f"(Physics only - no color)",
            fontsize=9, color="purple", fontweight="bold"
        )
        axes[-3].axis("off")
        for spine in axes[-3].spines.values():
            spine.set_edgecolor("purple"); spine.set_linewidth(3)

        # Plot Colorized Prediction
        color_img = color_pred_imgs[i].permute(1,2,0).cpu().numpy()
        axes[-2].imshow(np.clip(color_img, 0, 1))
        axes[-2].set_title(
            f"COLOR PREDICTION T+{pred_frame_label}\n"
            f"(Learning to paint...)",
            fontsize=9, color="#2878b5", fontweight="bold"
        )
        axes[-2].axis("off")
        for spine in axes[-2].spines.values():
            spine.set_edgecolor("#2878b5"); spine.set_linewidth(3)
            
        # Ground Truth
        tgt_img = rgb_tgt[i].permute(1,2,0).cpu().numpy()
        axes[-1].imshow(np.clip(tgt_img, 0, 1))
        axes[-1].set_title(
            f"GROUND TRUTH T+{pred_frame_label}\n"
            f"({time_ahead_s:.1f}s ahead — reference only)",
            fontsize=9, color="darkred", fontweight="bold")
        axes[-1].axis("off")
        for spine in axes[-1].spines.values():
            spine.set_edgecolor("red"); spine.set_linewidth(3)
            
        fig.suptitle(f"Sample {i} — Model Future Prediction vs Ground Truth")
        fig.tight_layout()
        fig.savefig(os.path.join(vd, f"s2_wm_gt_context_strip_{i}.png"),
                    dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] Prediction context strips saved.")

    # ── 2. RGB error heatmaps: Baseline | Model | Improvement ────────────────
    fig, axes = plt.subplots(n, 3, figsize=(13, 4*n))
    if n == 1: axes = axes[np.newaxis, :]
    for i in range(n):
        bh  = b_rgb_err[i].reshape(grid,grid).cpu().numpy()
        mh  = m_rgb_err[i].reshape(grid,grid).cpu().numpy()
        dif = bh - mh  # positive = model better
        vmax = max(bh.max(), mh.max()) + 1e-9
        for col, (data, cmap, label) in enumerate([
            (bh,  "magma",  "Baseline MSE"),
            (mh,  "magma",  "Model MSE"),
            (dif, "RdYlGn", "Improvement (green=model wins)"),
        ]):
            vr = (-vmax/2, vmax/2) if col==2 else (0, vmax)
            im = axes[i, col].imshow(data, cmap=cmap, vmin=vr[0], vmax=vr[1])
            axes[i, col].axis("off")
            fig.colorbar(im, ax=axes[i,col], fraction=0.046, pad=0.04)
        if i==0:
            for col, lbl in enumerate(["Baseline (copy T)", "Model Prediction",
                                        "Improvement (baseline−model)"]):
                axes[0, col].set_title(lbl, fontsize=10)
    fig.suptitle("RGB g(.) Token Error — Baseline vs Model", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(vd, "s2_wm_model_vs_baseline_heatmap.png"),
                dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] Model vs baseline heatmaps saved.")

    # ── 3. LiDAR side-by-side bars ───────────────────────────────────────────
    fig, axes = plt.subplots(n, 1, figsize=(14, 3*n))
    if n == 1: axes = [axes]
    for i in range(n):
        me = m_lid_err[i].cpu().numpy(); be = b_lid_err[i].cpu().numpy()
        xs = np.arange(len(me)); w = 0.38
        axes[i].bar(xs-w/2, be, width=w, label="Baseline", color="#e07070", alpha=0.85)
        axes[i].bar(xs+w/2, me, width=w, label="Model",    color="#2878b5", alpha=0.85)
        axes[i].set_title(f"Sample {i}", fontsize=9)
        axes[i].set_xlabel("LiDAR group"); axes[i].set_ylabel("MSE")
        axes[i].legend(fontsize=8); axes[i].grid(True, axis="y", alpha=0.3)
    fig.suptitle("LiDAR g(.) Token Error — Baseline vs Model", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(vd, "s2_wm_lidar_model_vs_baseline.png"),
                dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] LiDAR baseline vs model bars saved.")

    # ── 4. Cosine histograms: Baseline (red) vs Model (blue) ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(0.6, 1.01, 50)
    for ax, (bc, mc, title) in zip(axes, [
        (b_rgb_cos, m_rgb_cos, "RGB g(.) Cosine Similarity"),
        (b_lid_cos, m_lid_cos, "LiDAR g(.) Cosine Similarity"),
    ]):
        ax.hist(bc.cpu().reshape(-1).numpy(), bins=bins,
                alpha=0.6, color="#e07070", label="Baseline (copy T)")
        ax.hist(mc.cpu().reshape(-1).numpy(), bins=bins,
                alpha=0.6, color="#2878b5", label="Model prediction")
        ax.axvline(1.0, color="green", lw=1.5, ls="--", label="Perfect=1.0")
        ax.set_title(title); ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Count")
        ax.legend(); ax.grid(True, alpha=0.3)
    pred_lbl = args.steps_ahead
    fig.suptitle(f"Cosine Similarity vs Real T+{pred_lbl} ({args.steps_ahead*0.5:.1f}s ahead)  |  Model must shift RIGHT of baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(vd, "s2_wm_cosine_similarity.png"),
                dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] Cosine similarity histograms saved.")

    # ── 5. PCA: 3-way (last frame / model prediction / real T+1) ─────────────
    if HAS_PCA:
        combined = np.concatenate([
            base_rgb[0].cpu().numpy(),
            pred_rgb[0].cpu().numpy(),
            tgt_rgb_g[0].cpu().numpy(),
        ], axis=0)
        proj = PCA(n_components=2).fit_transform(combined)
        fig, ax = plt.subplots(figsize=(8, 7))
        _pl = args.steps_ahead
        ax.scatter(proj[:N,0],     proj[:N,1],     s=14, alpha=0.55,
                   color="#9b59b6", label="Last history frame T (= baseline)")
        ax.scatter(proj[N:2*N,0],  proj[N:2*N,1],  s=14, alpha=0.55,
                   color="#2878b5", marker="^", label=f"Model predicted T+{_pl} ({args.steps_ahead*0.5:.1f}s ahead)")
        ax.scatter(proj[2*N:3*N,0],proj[2*N:3*N,1],s=14, alpha=0.55,
                   color="#e07070", marker="x", label=f"Real T+{_pl} ground truth")
        ax.set_title("PCA of g(.) RGB Tokens — 3-way comparison\n"
                     f"Model (blue ^) should cluster closer to Real T+{_pl} (red x) than baseline (purple).")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(vd, "s2_wm_pca_token_space.png"),
                    dpi=160, bbox_inches="tight"); plt.close(fig)
        print("[VIS] PCA token space saved.")

    # ── 6. Metric summary bar chart ───────────────────────────────────────────
    def pct(model_v, base_v, lower_better=True):
        d = (base_v - model_v) if lower_better else (model_v - base_v)
        return d / (abs(base_v) + 1e-9) * 100

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (title, bv, mv, lb) in zip(axes, [
        ("RGB MSE ↓",       b_rgb_mse, m_rgb_mse, True),
        ("LiDAR MSE ↓",     b_lid_mse, m_lid_mse, True),
        ("RGB Cosine ↑",    b_rgb_cos.mean().item(), m_rgb_cos.mean().item(), False),
        ("LiDAR Cosine ↑",  b_lid_cos.mean().item(), m_lid_cos.mean().item(), False),
    ]):
        bars = ax.bar(["Baseline\n(copy T)", "Model"], [bv, mv],
                      color=["#e07070","#2878b5"], width=0.5, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, [bv, mv]):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+ax.get_ylim()[1]*0.01,
                    f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
        imp = pct(mv, bv, lb)
        ax.set_xlabel(f"{'▼' if imp>0 else '▲'} {abs(imp):.1f}% vs baseline",
                      color="green" if imp>0 else "red", fontsize=9, fontweight="bold")
    fig.suptitle("Metric Summary — Model vs Naive Baseline\n"
                 "Green improvement = model learned something beyond copying the present state.")
    fig.tight_layout()
    fig.savefig(os.path.join(vd, "s2_wm_metric_summary.png"),
                dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] Metric summary chart saved.")

    # ── 7. Text report ────────────────────────────────────────────────────────
    rgb_imp  = pct(m_rgb_mse, b_rgb_mse)
    lid_imp  = pct(m_lid_mse, b_lid_mse)
    rpt = os.path.join(vd, "s2_wm_report.txt")
    with open(rpt, "w") as f:
        f.write("OmniVec2 Stage 2 → World Model — Honest Prediction Report\n")
        f.write("="*60 + "\n\n")
        # At nuScenes 2 Hz: 1 step = 0.5s
        # Last input = T.  Target = T + steps_ahead
        future_label = args.steps_ahead                  # e.g. 2  → "T+2"
        time_ahead   = args.steps_ahead * 0.5            # e.g. 2×0.5 = 1.0s from T
        f.write(f"Predicts Stage 2 g(.) tokens for T+{future_label} ({time_ahead:.1f}s ahead of last input T)\n")
        f.write(f"  NuScenes rate : 2 Hz (0.5s per frame)\n")
        f.write(f"  Input window  : [T-{args.history-1}, ..., T]  ({args.history} frames = {args.history*0.5:.1f}s of history)\n")
        f.write(f"  Target        : T + {args.steps_ahead} steps = T+{future_label} = {time_ahead:.1f}s in the future\n")
        f.write("  Token space   : Stage 2 g(.) tokens\n")
        f.write("NOTE: The decoded image is generated purely from the predicted g(.) tokens\n")
        f.write("without any high-frequency f(.) leakage from previous frames, ensuring an\n")
        f.write("honest visualization of the predicted movement.\n\n")
        f.write(f"{'Metric':<34}{'Baseline':>10}{'Model':>10}{'Improvement':>14}\n")
        f.write("-"*68+"\n")
        for name, bv, mv, lb in [
            ("RGB g(.) MSE (lower=better)",   b_rgb_mse, m_rgb_mse, True),
            ("LiDAR g(.) MSE (lower=better)", b_lid_mse, m_lid_mse, True),
            ("RGB Cosine (higher=better)",    b_rgb_cos.mean().item(), m_rgb_cos.mean().item(), False),
            ("LiDAR Cosine (higher=better)",  b_lid_cos.mean().item(), m_lid_cos.mean().item(), False),
        ]:
            imp = pct(mv, bv, lb)
            f.write(f"{name:<34}{bv:>10.6f}{mv:>10.6f}{imp:>+13.2f}%\n")
        verdict = "PASSES" if (rgb_imp > 0 and lid_imp > 0) else "DOES NOT PASS"
        f.write(f"\nVERDICT: Model {verdict} the baseline check.\n")
        f.write("\nOUTPUT FILES:\n")
        f.write("  s2_wm_training_curves.png          - train/val loss\n")
        f.write("  s2_wm_gt_context_strip_*.png        - ground-truth reference frames\n")
        f.write("  s2_wm_model_vs_baseline_heatmap.png - per-patch MSE comparison\n")
        f.write("  s2_wm_lidar_model_vs_baseline.png   - LiDAR group MSE comparison\n")
        f.write("  s2_wm_cosine_similarity.png         - cosine distributions\n")
        f.write("  s2_wm_pca_token_space.png           - 3-way PCA\n")
        f.write("  s2_wm_metric_summary.png            - summary bar chart\n")
    print(f"[VIS] Report → {rpt}")
    print(f"[RESULT] RGB  — Baseline:{b_rgb_mse:.4f} Model:{m_rgb_mse:.4f} ({rgb_imp:+.1f}%)")
    print(f"[RESULT] LiDAR — Baseline:{b_lid_mse:.4f} Model:{m_lid_mse:.4f} ({lid_imp:+.1f}%)")
    print(f"\n[VIS] All files saved to: {vd}")



# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir,    exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    print(f"Device         : {DEVICE}")
    print(f"NuScenes       : {args.version} @ {args.dataroot}")
    print(f"Stage 2 ckpt   : {args.stage2_checkpoint}")
    print(f"Output         : {args.output_dir}")
    print(f"History frames : {args.history}")
    print(f"Epochs         : {args.epochs}")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    train_dl, val_dl = build_world_model_dataloaders(
        nusc=nusc, dataroot=args.dataroot,
        batch_size=args.batch_size, num_workers=args.num_workers,
        split_ratio=args.train_split_ratio, scene_limit=args.scene_limit,
        seed=args.seed, history=args.history, steps_ahead=args.steps_ahead,
    )

    # ── Build Stage 2 (contains all Stage 1 weights internally) ─────────────
    print("\nBuilding Stage 2 model and loading checkpoint...")
    stage1 = OmniVec2Stage1().to(DEVICE)   # shell needed for Stage2 constructor
    stage2 = OmniVec2Stage2(stage1).to(DEVICE)
    load_stage2_checkpoint(stage2, args.stage2_checkpoint)
    stage2.eval()
    print("Stage 2 checkpoint loaded (includes Stage 1 weights). Frozen.\n")

    # ── World model ───────────────────────────────────────────────────────────
    temporal = TemporalTokenPredictor(
        embed_dim=FC_HIDDEN,
        num_heads=args.temporal_heads,
        num_layers=args.temporal_layers,
        max_history=args.history,
    )
    model = OmniVec2Stage2WorldModel(stage2, temporal, freeze_stage2=True).to(DEVICE)

    # ── Train ─────────────────────────────────────────────────────────────────
    history_log = train(model, train_dl, val_dl, args)

    # ── Visualize ─────────────────────────────────────────────────────────────
    if not args.skip_visualizations:
        print("\nGenerating visualizations...")
        plot_curves(history_log, args.output_dir)
        visualize(model, val_dl, args)

    # ── Export ────────────────────────────────────────────────────────────────
    export = os.path.join(args.output_dir, "s2_wm_temporal_predictor.pth")
    torch.save(model.temporal.state_dict(), export)
    print(f"\nExported temporal predictor weights → {export}")
    print("Done.")


if __name__ == "__main__":
    main()
