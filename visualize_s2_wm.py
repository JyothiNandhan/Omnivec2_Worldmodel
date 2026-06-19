"""
Standalone Visualization Script — OmniVec2 Stage 2 World Model.

Loads an existing wm_s2_best.pth (or wm_s2_last.pth) checkpoint from an
ongoing or completed training run and generates all visualizations WITHOUT
touching or resuming training.

Usage (on HiperGator login/compute node):
    python visualize_s2_wm.py \\
        --stage2_checkpoint ./runs/stage2_first500/checkpoints/stage2_best.pth \\
        --wm_checkpoint     ./runs/s2_world_model/checkpoints/wm_s2_best.pth \\
        --output_dir        ./runs/s2_world_model \\
        --dataroot          /orange/iruchkin/isen/nsfull

Or submit as a short SLURM job (see bottom of this file).
"""
import argparse
import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (
    BATCH_SIZE, CHECKPOINT_DIRNAME, FC_HIDDEN,
    NUSCENES_DATAROOT, NUSCENES_VERSION,
    NUM_WORKERS, SCENE_LIMIT, SEED, TRAIN_SPLIT_RATIO,
)
from model import OmniVec2Stage1
from stage2 import OmniVec2Stage2
from s2_world_model_2.temporal_dataset import build_world_model_dataloaders
from s2_world_model_2.world_model_s2 import OmniVec2Stage2WorldModel, TemporalTokenPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────── Args ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize Stage 2 World Model checkpoint")
    p.add_argument("--stage2_checkpoint", type=str, required=True,
                   help="Path to the trained Stage 2 .pth checkpoint.")
    p.add_argument("--wm_checkpoint", type=str, default=None,
                   help="Path to world model checkpoint (wm_s2_best.pth or wm_s2_last.pth). "
                        "If not given, searches in --output_dir/checkpoints/.")
    p.add_argument("--output_dir",   type=str, default="./runs/s2_world_model")
    p.add_argument("--dataroot",     type=str, default=NUSCENES_DATAROOT)
    p.add_argument("--version",      type=str, default=NUSCENES_VERSION)
    p.add_argument("--scene_limit",  type=int, default=SCENE_LIMIT)
    p.add_argument("--batch_size",   type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers",  type=int, default=NUM_WORKERS)
    p.add_argument("--seed",         type=int, default=SEED)
    p.add_argument("--history",      type=int, default=4)
    p.add_argument("--steps_ahead",  type=int, default=2)
    p.add_argument("--temporal_layers", type=int, default=4)
    p.add_argument("--temporal_heads",  type=int, default=4)
    p.add_argument("--num_vis_samples", type=int, default=4)
    p.add_argument("--train_split_ratio", type=float, default=TRAIN_SPLIT_RATIO)
    args = p.parse_args()

    # Auto-find wm checkpoint
    if args.wm_checkpoint is None:
        best = os.path.join(args.output_dir, CHECKPOINT_DIRNAME, "wm_s2_best.pth")
        last = os.path.join(args.output_dir, CHECKPOINT_DIRNAME, "wm_s2_last.pth")
        if os.path.exists(best):
            args.wm_checkpoint = best
            print(f"[Auto] Using best checkpoint: {best}")
        elif os.path.exists(last):
            args.wm_checkpoint = last
            print(f"[Auto] Using last checkpoint: {last}")
        else:
            raise FileNotFoundError(
                f"No wm_s2_best.pth or wm_s2_last.pth found in "
                f"{os.path.join(args.output_dir, CHECKPOINT_DIRNAME)}. "
                "Pass --wm_checkpoint explicitly."
            )
    return args


# ─────────────────────────── Loaders ─────────────────────────────────────────

def load_stage2(path):
    stage1 = OmniVec2Stage1().to(DEVICE)
    stage2 = OmniVec2Stage2(stage1).to(DEVICE)
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    if "model_state_dict" in state:
        stage2.load_state_dict(state["model_state_dict"])
    else:
        stage2.load_state_dict(state)
    stage2.eval()
    return stage2


def load_world_model(stage2, wm_path, args):
    temporal = TemporalTokenPredictor(
        embed_dim=FC_HIDDEN,
        num_heads=args.temporal_heads,
        num_layers=args.temporal_layers,
        max_history=args.history,
    )
    model = OmniVec2Stage2WorldModel(stage2, temporal, freeze_stage2=True).to(DEVICE)
    try:
        ckpt = torch.load(wm_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(wm_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    history_log = ckpt.get("history", {})
    print(f"Loaded world model checkpoint — epoch {epoch}")
    model.eval()
    return model, epoch, history_log


# ─────────────────────────── Visualization ───────────────────────────────────

def _vis_dir(output_dir):
    p = os.path.join(output_dir, "visualizations")
    os.makedirs(p, exist_ok=True)
    return p


def plot_curves(log, output_dir, epoch):
    """Training/validation loss curves up to the current epoch."""
    if not log:
        print("[VIS] No training history in checkpoint — skipping loss curves.")
        return
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
    fig.suptitle(f"OmniVec2 Stage 2 → World Model — Training Curves (epoch {epoch})")
    fig.tight_layout()
    out = os.path.join(vd, "s2_wm_training_curves.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[VIS] Training curves → {out}")


@torch.no_grad()
def visualize(model, val_dl, args, epoch):
    """All 7 diagnostic visualizations against a naive copy-last-frame baseline."""
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
    ego_seq = batch["ego_sequence"].to(DEVICE)
    rgb_tgt = batch["rgb_target"].to(DEVICE)
    lid_tgt = batch["lidar_target"].to(DEVICE)

    # Encode history → g(.) space
    rgb_g_seq, lid_g_seq = model.encode_sequence(rgb_seq, lid_seq)

    # Model prediction
    out      = model.temporal(rgb_g_seq, lid_g_seq, ego_seq)
    pred_rgb = out["pred_rgb_tokens"]
    pred_lid = out["pred_lidar_tokens"]

    # Naive baseline: copy last history frame's tokens
    base_rgb = rgb_g_seq[:, -1].clone()
    base_lid = lid_g_seq[:, -1].clone()

    # Ground truth
    tgt_rgb_g, tgt_lid_g = model.encode_target(rgb_tgt, lid_tgt)

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

    # ── Decode predicted tokens → pixel image ────────────────────────────────
    out_last = model.stage2(rgb_seq[:, -1], lid_seq[:, -1], mask_ratio_rgb=0.0, mask_ratio_lidar=0.0)
    rgb_f_last = out_last["rgb_f"]
    rgb_refined_pred = model.stage2.ca_rgb_back(pred_rgb, rgb_f_last)
    ids_restore = torch.arange(N, device=DEVICE).unsqueeze(0).expand(rgb_seq.shape[0], N)
    pred_patches = model.stage2.rgb_decoder(rgb_refined_pred, ids_restore)
    # Use the model's predicted color statistics!
    mean = out["pred_rgb_mean"]
    var  = out["pred_rgb_var"]
    pred_pixels = pred_patches * (var + 1e-6).sqrt() + mean
    pred_imgs = unpatchify(pred_pixels)

    # ── 1. Context strip ──────────────────────────────────────────────────────
    for i in range(n):
        fig, axes = plt.subplots(1, T + 2, figsize=(3.2*(T+2), 3.6))
        for t in range(T):
            img = rgb_seq[i, t].permute(1,2,0).cpu().numpy()
            axes[t].imshow(np.clip(img, 0, 1))
            axes[t].set_title(f"History T-{T-t}", fontsize=9); axes[t].axis("off")
        pred_img = pred_imgs[i].permute(1,2,0).cpu().numpy()
        axes[-2].imshow(np.clip(pred_img, 0, 1))
        axes[-2].set_title(f"MODEL PREDICTION\nT+{args.steps_ahead}", fontsize=9,
                           color="#2878b5", fontweight="bold"); axes[-2].axis("off")
        for sp in axes[-2].spines.values():
            sp.set_edgecolor("#2878b5"); sp.set_linewidth(3)
        tgt_img = rgb_tgt[i].permute(1,2,0).cpu().numpy()
        axes[-1].imshow(np.clip(tgt_img, 0, 1))
        axes[-1].set_title(f"GROUND TRUTH\nT+{args.steps_ahead}", fontsize=9,
                           color="darkred", fontweight="bold"); axes[-1].axis("off")
        for sp in axes[-1].spines.values():
            sp.set_edgecolor("red"); sp.set_linewidth(3)
        fig.suptitle(f"Sample {i} — Model Prediction vs Ground Truth  [Epoch {epoch}]")
        fig.tight_layout()
        fig.savefig(os.path.join(vd, f"s2_wm_gt_context_strip_{i}.png"),
                    dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] Prediction context strips saved.")

    # ── 2. RGB error heatmaps ─────────────────────────────────────────────────
    fig, axes = plt.subplots(n, 3, figsize=(13, 4*n))
    if n == 1: axes = axes[np.newaxis, :]
    for i in range(n):
        bh = b_rgb_err[i].reshape(grid,grid).cpu().numpy()
        mh = m_rgb_err[i].reshape(grid,grid).cpu().numpy()
        dif = bh - mh
        vmax = max(bh.max(), mh.max()) + 1e-9
        for col, (data, cmap, lbl) in enumerate([
            (bh,  "magma",  "Baseline MSE"),
            (mh,  "magma",  "Model MSE"),
            (dif, "RdYlGn", "Improvement (green=model wins)"),
        ]):
            vr = (-vmax/2, vmax/2) if col==2 else (0, vmax)
            im = axes[i,col].imshow(data, cmap=cmap, vmin=vr[0], vmax=vr[1])
            axes[i,col].axis("off")
            fig.colorbar(im, ax=axes[i,col], fraction=0.046, pad=0.04)
        if i == 0:
            for col, lbl in enumerate(["Baseline (copy T)", "Model Prediction",
                                        "Improvement (baseline−model)"]):
                axes[0,col].set_title(lbl, fontsize=10)
    fig.suptitle(f"RGB g(.) Token Error — Baseline vs Model  [Epoch {epoch}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(vd, "s2_wm_model_vs_baseline_heatmap.png"),
                dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] Model vs baseline heatmaps saved.")

    # ── 3. LiDAR bars ────────────────────────────────────────────────────────
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
    fig.suptitle(f"LiDAR g(.) Token Error — Baseline vs Model  [Epoch {epoch}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(vd, "s2_wm_lidar_model_vs_baseline.png"),
                dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] LiDAR bars saved.")

    # ── 4. Cosine similarity histograms ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(0.6, 1.01, 50)
    for ax, (bc, mc, title) in zip(axes, [
        (b_rgb_cos, m_rgb_cos, "RGB g(.) Cosine Similarity"),
        (b_lid_cos, m_lid_cos, "LiDAR g(.) Cosine Similarity"),
    ]):
        ax.hist(bc.cpu().reshape(-1).numpy(), bins=bins, alpha=0.6,
                color="#e07070", label="Baseline (copy T)")
        ax.hist(mc.cpu().reshape(-1).numpy(), bins=bins, alpha=0.6,
                color="#2878b5", label="Model prediction")
        ax.axvline(1.0, color="green", lw=1.5, ls="--", label="Perfect=1.0")
        ax.set_title(title); ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Count")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Cosine Similarity vs Real T+{args.steps_ahead}  [Epoch {epoch}]  |  "
                 "Model must shift RIGHT of baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(vd, "s2_wm_cosine_similarity.png"),
                dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] Cosine similarity histograms saved.")

    # ── 5. PCA 3-way ──────────────────────────────────────────────────────────
    if HAS_PCA:
        combined = np.concatenate([
            base_rgb[0].cpu().numpy(),
            pred_rgb[0].cpu().numpy(),
            tgt_rgb_g[0].cpu().numpy(),
        ], axis=0)
        proj = __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(n_components=2).fit_transform(combined)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(proj[:N,0],     proj[:N,1],     s=14, alpha=0.55,
                   color="#9b59b6", label="Last history frame T (baseline)")
        ax.scatter(proj[N:2*N,0],  proj[N:2*N,1],  s=14, alpha=0.55,
                   color="#2878b5", marker="^", label=f"Model predicted T+{args.steps_ahead}")
        ax.scatter(proj[2*N:3*N,0],proj[2*N:3*N,1],s=14, alpha=0.55,
                   color="#e07070", marker="x", label=f"Real T+{args.steps_ahead} (ground truth)")
        ax.set_title(f"PCA of g(.) RGB Tokens — 3-way  [Epoch {epoch}]\n"
                     f"Model (blue ^) should cluster closer to Real T+{args.steps_ahead} (red x) than baseline (purple).")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(vd, "s2_wm_pca_token_space.png"),
                    dpi=160, bbox_inches="tight"); plt.close(fig)
        print("[VIS] PCA token space saved.")

    # ── 6. Metric summary bar chart ───────────────────────────────────────────
    def pct(mv, bv, lower_better=True):
        d = (bv - mv) if lower_better else (mv - bv)
        return d / (abs(bv) + 1e-9) * 100

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (title, bv, mv, lb) in zip(axes, [
        ("RGB MSE ↓",      b_rgb_mse, m_rgb_mse, True),
        ("LiDAR MSE ↓",    b_lid_mse, m_lid_mse, True),
        ("RGB Cosine ↑",   b_rgb_cos.mean().item(), m_rgb_cos.mean().item(), False),
        ("LiDAR Cosine ↑", b_lid_cos.mean().item(), m_lid_cos.mean().item(), False),
    ]):
        bars = ax.bar(["Baseline\n(copy T)", "Model"], [bv, mv],
                      color=["#e07070","#2878b5"], width=0.5, edgecolor="white")
        ax.set_title(title, fontweight="bold"); ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, [bv, mv]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ax.get_ylim()[1]*0.01,
                    f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
        imp = pct(mv, bv, lb)
        ax.set_xlabel(f"{'▼' if imp>0 else '▲'} {abs(imp):.1f}% vs baseline",
                      color="green" if imp>0 else "red", fontsize=9, fontweight="bold")
    fig.suptitle(f"Metric Summary — Model vs Naive Baseline  [Epoch {epoch}]\n"
                 "Green = model learned something beyond copying the present state.")
    fig.tight_layout()
    fig.savefig(os.path.join(vd, "s2_wm_metric_summary.png"),
                dpi=160, bbox_inches="tight"); plt.close(fig)
    print("[VIS] Metric summary chart saved.")

    # ── 7. Text report ────────────────────────────────────────────────────────
    rgb_imp = pct(m_rgb_mse, b_rgb_mse)
    lid_imp = pct(m_lid_mse, b_lid_mse)
    rpt = os.path.join(vd, "s2_wm_report.txt")
    with open(rpt, "w") as f:
        f.write(f"OmniVec2 Stage 2 World Model — Checkpoint Epoch {epoch}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Predicts Stage 2 g(.) tokens for frame T+{args.steps_ahead} ({args.steps_ahead*0.5:.1f}s ahead)\n\n")
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
    print(f"[VIS] Report → {rpt}")
    print(f"[RESULT] RGB   — Baseline:{b_rgb_mse:.4f}  Model:{m_rgb_mse:.4f}  ({rgb_imp:+.1f}%)")
    print(f"[RESULT] LiDAR — Baseline:{b_lid_mse:.4f}  Model:{m_lid_mse:.4f}  ({lid_imp:+.1f}%)")
    print(f"\n[VIS] All visualizations saved to: {_vis_dir(args.output_dir)}")


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Device       : {DEVICE}")
    print(f"WM checkpoint: {args.wm_checkpoint}")
    print(f"Stage2 ckpt  : {args.stage2_checkpoint}")
    print(f"Output dir   : {args.output_dir}")

    # Build val dataloader
    print("\nLoading NuScenes...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    _, val_dl = build_world_model_dataloaders(
        nusc=nusc, dataroot=args.dataroot,
        batch_size=args.batch_size, num_workers=max(1, args.num_workers // 2),
        split_ratio=args.train_split_ratio, scene_limit=args.scene_limit,
        seed=args.seed, history=args.history, steps_ahead=args.steps_ahead,
    )

    # Load models
    print("\nLoading Stage 2 backbone...")
    stage2 = load_stage2(args.stage2_checkpoint)
    print("Loading World Model...")
    model, epoch, history_log = load_world_model(stage2, args.wm_checkpoint, args)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_curves(history_log, args.output_dir, epoch)
    visualize(model, val_dl, args, epoch)
    print("\nDone!")


if __name__ == "__main__":
    main()
