"""
LiDAR Visualizations
====================
1. Post-training:  Masking & reconstruction 3D plots (HTML)
2. Post-tokenizer: FPS centers, patch groups, embedding similarity, token norms
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ..config import NUM_GROUP, GROUP_SIZE
    from ..shared.masking import random_mask
    from ..data.lidar_helpers import make_patches_batch
except ImportError:
    from config import NUM_GROUP, GROUP_SIZE
    from shared.masking import random_mask
    from data.lidar_helpers import make_patches_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════
#  POST-TRAINING: masking + reconstruction 3D plots
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_lidar(model, dataloader, output_dir, mask_ratio):
    """Save LiDAR masking + reconstruction as interactive HTML."""
    model.eval()
    points = next(iter(dataloader))[:1].to(DEVICE)

    # ── Masking plot ──
    patches, centers = make_patches_batch(points, NUM_GROUP, GROUP_SIZE)
    tokens = model.lidar_patch_encoder(patches) + model.lidar_pos_embed(centers)
    _, _, _, mask = random_mask(tokens, mask_ratio)

    patches_np = patches.squeeze(0).cpu().numpy()
    centers_np = centers.squeeze(0).cpu().numpy()
    mask_1d    = mask.squeeze(0).cpu().numpy()

    fig = go.Figure()
    for i in range(NUM_GROUP):
        pts = patches_np[i] + centers_np[i]
        color = "red" if mask_1d[i] else "blue"
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker=dict(size=2, color=color), showlegend=False))
    fig.update_layout(scene_aspectmode="data", width=800, height=600,
                      title="LiDAR: Visible (blue) vs Masked (red)")
    path1 = os.path.join(output_dir, "lidar_masking.html")
    fig.write_html(path1)
    print(f"  ✓ LiDAR masking → {path1}")

    # ── Reconstruction plot ──
    pred, target, mask2, patches2, centers2 = model.forward_lidar(
        points, mask_ratio)
    patches_np2 = patches2.squeeze(0).cpu().numpy()
    centers_np2 = centers2.squeeze(0).cpu().numpy()
    mask_1d2    = mask2.squeeze(0).cpu().numpy()

    target_flat = patches2.reshape(1, NUM_GROUP, -1)
    t_mean = target_flat.mean(dim=-1, keepdim=True)
    t_var  = target_flat.var(dim=-1, keepdim=True)
    pred_unnorm = pred * (t_var + 1e-6).sqrt() + t_mean
    pred_pts = pred_unnorm.squeeze(0).reshape(
        NUM_GROUP, GROUP_SIZE, 3).cpu().numpy()

    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Original", "Reconstructed (masked = green)"],
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]])
    for i in range(NUM_GROUP):
        pts_orig = patches_np2[i] + centers_np2[i]
        c1 = "red" if mask_1d2[i] else "blue"
        fig2.add_trace(go.Scatter3d(
            x=pts_orig[:, 0], y=pts_orig[:, 1], z=pts_orig[:, 2],
            mode="markers", marker=dict(size=2, color=c1),
            showlegend=False), row=1, col=1)
        if mask_1d2[i]:
            pts_rec = pred_pts[i] + centers_np2[i]
            fig2.add_trace(go.Scatter3d(
                x=pts_rec[:, 0], y=pts_rec[:, 1], z=pts_rec[:, 2],
                mode="markers", marker=dict(size=2, color="green"),
                showlegend=False), row=1, col=2)
        else:
            fig2.add_trace(go.Scatter3d(
                x=pts_orig[:, 0], y=pts_orig[:, 1], z=pts_orig[:, 2],
                mode="markers", marker=dict(size=2, color="blue"),
                showlegend=False), row=1, col=2)
    fig2.update_layout(width=1200, height=600,
                       title_text="LiDAR Reconstruction")
    path2 = os.path.join(output_dir, "lidar_reconstruction.html")
    fig2.write_html(path2)
    print(f"  ✓ LiDAR reconstruction → {path2}")


# ══════════════════════════════════════════════════════════
#  POST-TOKENIZER: what the model "sees" before masking
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_fps_centers(dataloader, output_dir):
    """3D plot: full point cloud + FPS-selected center points (highlighted).
    Shows how the patching algorithm selects representative points.
    """
    points = next(iter(dataloader))[:1].to(DEVICE)
    patches, centers = make_patches_batch(points, NUM_GROUP, GROUP_SIZE)

    pts_all = points.squeeze(0).cpu().numpy()       # (N, 3)
    ctrs    = centers.squeeze(0).cpu().numpy()       # (G, 3)

    fig = go.Figure()
    # Full cloud — grey, small
    fig.add_trace(go.Scatter3d(
        x=pts_all[:, 0], y=pts_all[:, 1], z=pts_all[:, 2],
        mode="markers", marker=dict(size=1, color="lightgray", opacity=0.4),
        name=f"Full cloud ({pts_all.shape[0]} pts)"))
    # FPS centers — large red
    fig.add_trace(go.Scatter3d(
        x=ctrs[:, 0], y=ctrs[:, 1], z=ctrs[:, 2],
        mode="markers", marker=dict(size=6, color="red"),
        name=f"FPS centers ({NUM_GROUP})"))
    fig.update_layout(scene_aspectmode="data", width=900, height=600,
                      title=f"FPS Center Selection — {NUM_GROUP} centers from {pts_all.shape[0]} points")
    path = os.path.join(output_dir, "lidar_fps_centers.html")
    fig.write_html(path)
    print(f"  ✓ FPS centers → {path}")


@torch.no_grad()
def visualize_patch_groups(dataloader, output_dir):
    """3D plot: each kNN patch group in a different color.
    Shows how the point cloud is partitioned into local patches.
    """
    points = next(iter(dataloader))[:1].to(DEVICE)
    patches, centers = make_patches_batch(points, NUM_GROUP, GROUP_SIZE)

    patches_np = patches.squeeze(0).cpu().numpy()   # (G, K, 3)
    centers_np = centers.squeeze(0).cpu().numpy()    # (G, 3)

    # Generate distinct colors for each group
    colors = plt.cm.tab20(np.linspace(0, 1, min(NUM_GROUP, 20)))
    colors = np.tile(colors, (NUM_GROUP // 20 + 1, 1))[:NUM_GROUP]

    fig = go.Figure()
    for i in range(NUM_GROUP):
        pts = patches_np[i] + centers_np[i]  # un-center
        c = f"rgb({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)})"
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", marker=dict(size=2, color=c),
            name=f"Patch {i}", showlegend=False))
    # Center markers
    fig.add_trace(go.Scatter3d(
        x=centers_np[:, 0], y=centers_np[:, 1], z=centers_np[:, 2],
        mode="markers", marker=dict(size=5, color="black", symbol="diamond"),
        name="Centers"))
    fig.update_layout(scene_aspectmode="data", width=900, height=600,
                      title=f"LiDAR Patch Groups — {NUM_GROUP} patches × {GROUP_SIZE} points")
    path = os.path.join(output_dir, "lidar_patch_groups.html")
    fig.write_html(path)
    print(f"  ✓ Patch groups → {path}")


@torch.no_grad()
def visualize_lidar_token_similarity(model, dataloader, output_dir):
    """Pairwise cosine similarity between LiDAR patch token embeddings.
    Shows which 3D regions produce similar representations.
    """
    model.eval()
    points = next(iter(dataloader))[:1].to(DEVICE)
    patches, centers = make_patches_batch(points, NUM_GROUP, GROUP_SIZE)
    tokens = model.lidar_patch_encoder(patches) + model.lidar_pos_embed(centers)
    tok_np = tokens.squeeze(0).cpu().numpy()  # (G, D)

    sim_matrix = cosine_similarity(tok_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Similarity matrix
    im = axes[0].imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Patch Embedding Similarity (cosine)")
    axes[0].set_xlabel("Patch ID")
    axes[0].set_ylabel("Patch ID")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Embedding norm per patch
    norms = np.linalg.norm(tok_np, axis=1)
    axes[1].bar(range(NUM_GROUP), norms, color="steelblue", alpha=0.7)
    axes[1].set_title("Token L2 Norm per Patch")
    axes[1].set_xlabel("Patch ID")
    axes[1].set_ylabel("L2 Norm")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("LiDAR Post-Tokenizer — Patch Embeddings", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "lidar_token_similarity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Token similarity → {path}")


@torch.no_grad()
def visualize_lidar_token_norms_3d(model, dataloader, output_dir):
    """3D colored by token embedding norm — shows which spatial regions
    produce stronger activations in the tokenizer.
    """
    model.eval()
    points = next(iter(dataloader))[:1].to(DEVICE)
    patches, centers = make_patches_batch(points, NUM_GROUP, GROUP_SIZE)
    tokens = model.lidar_patch_encoder(patches) + model.lidar_pos_embed(centers)

    norms = tokens.squeeze(0).norm(dim=-1).cpu().numpy()  # (G,)
    patches_np = patches.squeeze(0).cpu().numpy()
    centers_np = centers.squeeze(0).cpu().numpy()

    # Normalize norms to [0, 1] for colormap
    norm_min, norm_max = norms.min(), norms.max()
    norm_scaled = (norms - norm_min) / (norm_max - norm_min + 1e-8)

    fig = go.Figure()
    for i in range(NUM_GROUP):
        pts = patches_np[i] + centers_np[i]
        val = norm_scaled[i]
        # Blue → Red colormap
        r = int(255 * val)
        b = int(255 * (1 - val))
        c = f"rgb({r},50,{b})"
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", marker=dict(size=2, color=c),
            name=f"Patch {i} (norm={norms[i]:.1f})", showlegend=False))
    fig.update_layout(
        scene_aspectmode="data", width=900, height=600,
        title="LiDAR Token Norms — Blue (low) → Red (high activation)")
    path = os.path.join(output_dir, "lidar_token_norms_3d.html")
    fig.write_html(path)
    print(f"  ✓ Token norms 3D → {path}")
