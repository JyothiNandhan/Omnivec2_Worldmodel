"""
RGB Visualizations
==================
1. Post-training:  Masked → Reconstructed image grid
2. Post-tokenizer: Patch grid overlay, embedding similarity, positional encoding heatmap
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ..config import IMG_SIZE, PATCH_SIZE
    from .patches import patchify, unpatchify
except ImportError:
    from config import IMG_SIZE, PATCH_SIZE
    from rgb.patches import patchify, unpatchify

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════
#  POST-TRAINING: masked reconstruction
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_rgb_reconstruction(model, dataloader, output_dir,
                                 mask_ratio, n_samples=4):
    """Save Original | Masked | Reconstruction grid (PNG)."""
    model.eval()
    imgs = next(iter(dataloader))[:n_samples].to(DEVICE)
    pred, target, mask = model.forward_rgb(imgs, mask_ratio)

    raw_patches = patchify(imgs)
    mean = raw_patches.mean(dim=-1, keepdim=True)
    var  = raw_patches.var(dim=-1, keepdim=True)
    pred_pixels = pred * (var + 1e-6).sqrt() + mean

    recon_patches = raw_patches.clone()
    mask_exp = mask.unsqueeze(-1).expand_as(recon_patches).bool()
    recon_patches[mask_exp] = pred_pixels[mask_exp]
    recon_imgs = unpatchify(recon_patches)

    masked_patches = raw_patches.clone()
    masked_patches[mask_exp] = 0.5
    masked_imgs = unpatchify(masked_patches)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    titles = ["Original", "Masked", "Reconstruction"]
    for i in range(n_samples):
        for j, img_t in enumerate([imgs[i], masked_imgs[i], recon_imgs[i]]):
            img_np = img_t.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            ax = axes[i, j] if n_samples > 1 else axes[j]
            ax.imshow(img_np)
            ax.set_title(titles[j])
            ax.axis("off")
    plt.suptitle("RGB Masked Reconstruction", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "rgb_reconstruction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ RGB reconstruction → {path}")


# ══════════════════════════════════════════════════════════
#  POST-TOKENIZER: what the model "sees" before masking
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_patch_grid(dataloader, output_dir, n_samples=3):
    """Show how the image is divided into patches with a grid overlay."""
    imgs = next(iter(dataloader))[:n_samples]
    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))
    if n_samples == 1:
        axes = [axes]
    for i, img in enumerate(imgs):
        img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i].imshow(img_np)
        # Draw grid lines
        for x in range(0, IMG_SIZE + 1, PATCH_SIZE):
            axes[i].axvline(x - 0.5, color="cyan", linewidth=0.8, alpha=0.7)
            axes[i].axhline(x - 0.5, color="cyan", linewidth=0.8, alpha=0.7)
        # Label patches
        grid = IMG_SIZE // PATCH_SIZE
        for r in range(grid):
            for c in range(grid):
                idx = r * grid + c
                axes[i].text(c * PATCH_SIZE + PATCH_SIZE // 2,
                             r * PATCH_SIZE + PATCH_SIZE // 2,
                             str(idx), ha="center", va="center",
                             fontsize=5, color="yellow", alpha=0.8)
        axes[i].set_title(f"Image {i+1} — {grid}×{grid} = {grid*grid} patches")
        axes[i].axis("off")
    plt.suptitle(f"RGB Patch Grid (P={PATCH_SIZE}, img={IMG_SIZE}×{IMG_SIZE})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "rgb_patch_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Patch grid → {path}")


@torch.no_grad()
def visualize_token_embeddings(model, dataloader, output_dir, n_samples=2):
    """Visualize pairwise cosine similarity between patch token embeddings.
    Shows which patches produce similar representations.
    """
    model.eval()
    imgs = next(iter(dataloader))[:n_samples].to(DEVICE)
    tokens = model.rgb_tokenizer(imgs)  # (B, N, D)

    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 5 * n_samples),
                             gridspec_kw={"width_ratios": [1, 1.2]})
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Original image
        img_np = imgs[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        # Cosine similarity matrix
        tok_np = tokens[i].cpu().numpy()  # (N, D)
        sim_matrix = cosine_similarity(tok_np)
        im = axes[i, 1].imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[i, 1].set_title("Patch Embedding Similarity (cosine)")
        axes[i, 1].set_xlabel("Patch ID")
        axes[i, 1].set_ylabel("Patch ID")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)

    plt.suptitle("RGB Token Embeddings — Post-Tokenizer", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "rgb_token_similarity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Token similarity → {path}")


@torch.no_grad()
def visualize_positional_encoding(model, output_dir):
    """Heatmap of the fixed 2D sinusoidal positional encoding."""
    pos = model.rgb_tokenizer.pos_embed.squeeze(0).cpu().numpy()  # (N, D)
    grid = IMG_SIZE // PATCH_SIZE

    # Show first 16 dims as spatial heatmaps
    n_show = min(16, pos.shape[1])
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    for d in range(n_show):
        r, c = divmod(d, cols)
        spatial = pos[:, d].reshape(grid, grid)
        im = axes[r, c].imshow(spatial, cmap="viridis")
        axes[r, c].set_title(f"dim {d}", fontsize=9)
        axes[r, c].axis("off")
        plt.colorbar(im, ax=axes[r, c], fraction=0.046)
    plt.suptitle("RGB Positional Encoding — Spatial patterns per dimension", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "rgb_positional_encoding.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Positional encoding → {path}")


@torch.no_grad()
def visualize_patch_norms(model, dataloader, output_dir, n_samples=3):
    """Show per-patch token magnitude as a heatmap over the image.
    Highlights which spatial regions produce stronger activations.
    """
    model.eval()
    imgs = next(iter(dataloader))[:n_samples].to(DEVICE)
    tokens = model.rgb_tokenizer(imgs)  # (B, N, D)
    norms = tokens.norm(dim=-1).cpu().numpy()  # (B, N)

    grid = IMG_SIZE // PATCH_SIZE
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        img_np = imgs[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        heatmap = norms[i].reshape(grid, grid)
        im = axes[i, 1].imshow(heatmap, cmap="hot", interpolation="nearest")
        axes[i, 1].set_title("Token L2 Norm (activation strength)")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
        axes[i, 1].axis("off")

    plt.suptitle("RGB Post-Tokenizer — Patch activation magnitudes", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "rgb_token_norms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Token norms → {path}")
