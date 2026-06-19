"""
OmniVec2 Stage 2 masked pretraining up to g(.), implemented inside the package.

Paper-faithful flow:
tokenizer -> f(.) -> cross-attention -> shared g(.)
-> cross-attention with modality-specific features -> decoder
"""
import os
import sys

import torch
import torch.nn as nn
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BATCH_SIZE,
    CA_HEADS,
    EPOCHS,
    FC_HIDDEN,
    G_LAYERS,
    G_NUM_HEADS,
    GROUP_SIZE,
    LR,
    MASK_RATIO_LIDAR,
    MASK_RATIO_RGB,
    MAX_GRAD_NORM,
    NUM_GROUP,
    WEIGHT_DECAY,
)
from checkpointing import (
    load_training_checkpoint,
    make_checkpoint_state,
    save_stage_checkpoint_bundle,
)
from data.lidar_helpers import make_patches_batch
from lidar.decoder import LidarDecoder
from rgb.decoder import RGBDecoder
from rgb.patches import normalize_patches, patchify
from shared.losses import masked_mse_loss
from shared.masking import random_mask


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossAttentionBlock(nn.Module):
    """Cross-attention with pre-norm and residual connection."""

    def __init__(self, embed_dim=FC_HIDDEN, num_heads=CA_HEADS):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, query_tokens, context_tokens):
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(context_tokens)
        attn_out, _ = self.attn(q, kv, kv)
        return self.norm_out(query_tokens + attn_out)


class SecondTransformerG(nn.Module):
    """Shared second transformer g(.) with the same hidden width as f(.) output."""

    def __init__(self, embed_dim=FC_HIDDEN, num_heads=G_NUM_HEADS, num_layers=G_LAYERS):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.transformer(x))


class OmniVec2Stage2(nn.Module):
    """
    Stage 2 model built from a pretrained Stage 1 model.

    Stage 1 decoders are discarded and fresh Stage 2 decoders are attached
    only to provide masked reconstruction losses during pretraining.
    """

    def __init__(self, stage1_model):
        super().__init__()
        self.rgb_tokenizer = stage1_model.rgb_tokenizer
        self.lidar_patch_encoder = stage1_model.lidar_patch_encoder
        self.lidar_pos_embed = stage1_model.lidar_pos_embed
        self.f = stage1_model.encoder

        self.ca_rgb_from_lidar = CrossAttentionBlock()
        self.ca_lidar_from_rgb = CrossAttentionBlock()
        self.g = SecondTransformerG()
        self.ca_rgb_back = CrossAttentionBlock()
        self.ca_lidar_back = CrossAttentionBlock()

        self.rgb_decoder = RGBDecoder()
        self.lidar_decoder = LidarDecoder()

    def _lidar_tokens(self, points):
        patches, centers = make_patches_batch(points, NUM_GROUP, GROUP_SIZE)
        lidar_tokens = self.lidar_patch_encoder(patches) + self.lidar_pos_embed(centers)
        return lidar_tokens, patches, centers

    def forward(self, imgs, points, mask_ratio_rgb=MASK_RATIO_RGB, mask_ratio_lidar=MASK_RATIO_LIDAR):
        rgb_tokens = self.rgb_tokenizer(imgs)
        lidar_tokens, lidar_patches, lidar_centers = self._lidar_tokens(points)

        rgb_vis, rgb_ids_restore, _, rgb_mask = random_mask(rgb_tokens, mask_ratio_rgb)
        lidar_vis, lidar_ids_restore, _, lidar_mask = random_mask(lidar_tokens, mask_ratio_lidar)

        rgb_f = self.f(rgb_vis)
        lidar_f = self.f(lidar_vis)

        rgb_mixed = self.ca_rgb_from_lidar(rgb_f, lidar_f)
        lidar_mixed = self.ca_lidar_from_rgb(lidar_f, rgb_f)

        rgb_g = self.g(rgb_mixed)
        lidar_g = self.g(lidar_mixed)

        rgb_refined = self.ca_rgb_back(rgb_g, rgb_f)
        lidar_refined = self.ca_lidar_back(lidar_g, lidar_f)

        pred_rgb = self.rgb_decoder(rgb_refined, rgb_ids_restore)
        pred_lidar = self.lidar_decoder(lidar_refined, lidar_ids_restore, lidar_centers)

        with torch.no_grad():
            target_rgb, _, _ = normalize_patches(patchify(imgs))
            target_lidar, _, _ = normalize_patches(
                lidar_patches.reshape(lidar_patches.shape[0], NUM_GROUP, -1)
            )

        loss_rgb = masked_mse_loss(pred_rgb, target_rgb, rgb_mask)
        loss_lidar = masked_mse_loss(pred_lidar, target_lidar, lidar_mask)
        loss = loss_rgb + loss_lidar

        return {
            "loss": loss,
            "loss_rgb": loss_rgb,
            "loss_lidar": loss_lidar,
            "rgb_f": rgb_f,
            "lidar_f": lidar_f,
            "rgb_g": rgb_g,
            "lidar_g": lidar_g,
            "rgb_refined": rgb_refined,
            "lidar_refined": lidar_refined,
        }

    def export_pretrained_fg(self):
        """Paper-faithful Stage 2 export: tokenizers + f(.) + g(.) + cross-attention blocks."""
        return {
            "encoder": self.f.state_dict(),
            "g": self.g.state_dict(),
            "rgb_tokenizer": self.rgb_tokenizer.state_dict(),
            "lidar_patch_encoder": self.lidar_patch_encoder.state_dict(),
            "lidar_pos_embed": self.lidar_pos_embed.state_dict(),
            "ca_rgb_from_lidar": self.ca_rgb_from_lidar.state_dict(),
            "ca_lidar_from_rgb": self.ca_lidar_from_rgb.state_dict(),
            "ca_rgb_back": self.ca_rgb_back.state_dict(),
            "ca_lidar_back": self.ca_lidar_back.state_dict(),
        }


def paired_batch_iterator(dl_a, dl_b):
    """Yield modality pairs for as many steps as the longer dataloader."""
    if len(dl_a) >= len(dl_b):
        main_dl, aux_dl = dl_a, dl_b
        swap = False
    else:
        main_dl, aux_dl = dl_b, dl_a
        swap = True

    aux_iter = iter(aux_dl)
    for main_batch in main_dl:
        try:
            aux_batch = next(aux_iter)
        except StopIteration:
            aux_iter = iter(aux_dl)
            aux_batch = next(aux_iter)
        yield (aux_batch, main_batch) if swap else (main_batch, aux_batch)


def maybe_resume_path(args):
    if getattr(args, "resume", ""):
        return args.resume
    auto_path = os.path.join(args.checkpoints_dir, "stage2_last.pth")
    if getattr(args, "auto_resume", False) and os.path.exists(auto_path):
        return auto_path
    return ""


def progress_miniters(dl_a, dl_b):
    return max(1, max(len(dl_a), len(dl_b)) // 100)


def train_stage2(model, train_rgb_dl, val_rgb_dl, train_lidar_dl, val_lidar_dl, args):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(args, "lr", LR),
        weight_decay=getattr(args, "weight_decay", WEIGHT_DECAY),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=getattr(args, "epochs", EPOCHS),
        eta_min=1e-6,
    )

    history = {
        "train_loss": [],
        "train_loss_rgb": [],
        "train_loss_lidar": [],
        "val_loss": [],
        "val_loss_rgb": [],
        "val_loss_lidar": [],
    }
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
        print(f"Resumed Stage 2 from {resume_path} at epoch {checkpoint['epoch']}.")

    total_epochs = getattr(args, "epochs", EPOCHS)
    save_every = getattr(args, "save_every", 1)
    keep_last_n = getattr(args, "keep_last_n_checkpoints", 5)
    max_grad_norm = getattr(args, "max_grad_norm", MAX_GRAD_NORM)
    batch_size = getattr(args, "batch_size", BATCH_SIZE)
    mask_ratio_rgb = getattr(args, "mask_ratio_rgb", MASK_RATIO_RGB)
    mask_ratio_lidar = getattr(args, "mask_ratio_lidar", MASK_RATIO_LIDAR)

    print("\n========== STAGE 2 — MULTIMODAL MASKED PRETRAINING UP TO g(.) ==========\n")
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        train_total = 0.0
        train_rgb_total = 0.0
        train_lidar_total = 0.0
        train_steps = 0

        train_pbar = tqdm(
            paired_batch_iterator(train_rgb_dl, train_lidar_dl),
            total=max(len(train_rgb_dl), len(train_lidar_dl)),
            desc=f"Ep{epoch} | Stage2 train",
            leave=False,
            file=sys.stdout,
            miniters=progress_miniters(train_rgb_dl, train_lidar_dl),
        )
        for imgs, points in train_pbar:
            if imgs.size(0) != batch_size or points.size(0) != batch_size:
                continue

            imgs = imgs.to(DEVICE)
            points = points.to(DEVICE)
            out = model(imgs, points, mask_ratio_rgb=mask_ratio_rgb, mask_ratio_lidar=mask_ratio_lidar)

            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_total += out["loss"].item()
            train_rgb_total += out["loss_rgb"].item()
            train_lidar_total += out["loss_lidar"].item()
            train_steps += 1
            train_pbar.set_postfix(
                loss=f"{train_total / max(train_steps, 1):.4f}",
                rgb=f"{train_rgb_total / max(train_steps, 1):.4f}",
                lidar=f"{train_lidar_total / max(train_steps, 1):.4f}",
            )

        model.eval()
        val_total = 0.0
        val_rgb_total = 0.0
        val_lidar_total = 0.0
        val_steps = 0
        with torch.no_grad():
            val_pbar = tqdm(
                paired_batch_iterator(val_rgb_dl, val_lidar_dl),
                total=max(len(val_rgb_dl), len(val_lidar_dl)),
                desc=f"Ep{epoch} | Stage2 val",
                leave=False,
                file=sys.stdout,
                miniters=progress_miniters(val_rgb_dl, val_lidar_dl),
            )
            for imgs, points in val_pbar:
                if imgs.size(0) != batch_size or points.size(0) != batch_size:
                    continue

                imgs = imgs.to(DEVICE)
                points = points.to(DEVICE)
                out = model(imgs, points, mask_ratio_rgb=mask_ratio_rgb, mask_ratio_lidar=mask_ratio_lidar)

                val_total += out["loss"].item()
                val_rgb_total += out["loss_rgb"].item()
                val_lidar_total += out["loss_lidar"].item()
                val_steps += 1
                val_pbar.set_postfix(
                    loss=f"{val_total / max(val_steps, 1):.4f}",
                    rgb=f"{val_rgb_total / max(val_steps, 1):.4f}",
                    lidar=f"{val_lidar_total / max(val_steps, 1):.4f}",
                )

        scheduler.step()

        train_loss = train_total / max(train_steps, 1)
        train_loss_rgb = train_rgb_total / max(train_steps, 1)
        train_loss_lidar = train_lidar_total / max(train_steps, 1)
        val_loss = val_total / max(val_steps, 1)
        val_loss_rgb = val_rgb_total / max(val_steps, 1)
        val_loss_lidar = val_lidar_total / max(val_steps, 1)

        history["train_loss"].append(train_loss)
        history["train_loss_rgb"].append(train_loss_rgb)
        history["train_loss_lidar"].append(train_loss_lidar)
        history["val_loss"].append(val_loss)
        history["val_loss_rgb"].append(val_loss_rgb)
        history["val_loss_lidar"].append(val_loss_lidar)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        print(
            f"Epoch {epoch:03d}/{total_epochs} | "
            f"train {train_loss:.4f} (rgb {train_loss_rgb:.4f}, lidar {train_loss_lidar:.4f}) | "
            f"val {val_loss:.4f} (rgb {val_loss_rgb:.4f}, lidar {val_loss_lidar:.4f})"
        )

        if epoch % save_every == 0:
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
                prefix="stage2",
                state=state,
                is_best=is_best,
                keep_last_n=keep_last_n,
            )

    return history
