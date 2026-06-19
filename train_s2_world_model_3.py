"""
OmniVec2 Stage 2 -> World Model 3 (training script, currently v5)

Architecture changes in v5 (run_s2_world_model_4.sh):
  - TemporalTokenPredictor scaled up: inner_dim=256, 6 layers, 8 heads, FFN=1024 (~5.3M params)
  - Input/output projections bridge Stage-2 token dim (128) to inner_dim (256)
  - Pixel head detach removed: pixel reconstruction loss now co-supervises temporal encoder

Loss design (v5 = v4 losses, bigger model):
  - Huber (Smooth L1, beta=0.1) for token magnitude
  - Per-token cosine loss (dim=-1, 128-d space) for direction
  - Direction margin loss: prediction must be closer to future than to present
  - VICReg variance + covariance — anti-collapse
  - spatial_diversity_loss: penalise within-image patch collapse (weight=0.3)
  - patch_nce_loss: per-position InfoNCE across batch (weight=0.05)
  - RGB patch reconstruction: L1 + Huber delta + edge loss (now with gradients to encoder)

History of fixes:
  v2→v3: VICReg applied to deltas (not total tokens); detach pixel head to protect tokens
  v3→v4: spatial_diversity_loss + patch_nce_loss to fix PCA clustering
  v4→v5: remove detach (pixel loss now helps encoder); 3× wider inner dim

Usage:
    sbatch run_s2_world_model_4.sh
"""
# ── Re-use everything from wm2 except the patched functions ──────────────────
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import importlib
_wm2 = importlib.import_module("train_s2_world_model_2")

import time
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from rgb.patches import patchify, unpatchify

DEVICE = _wm2.DEVICE


# ── Loss helpers ──────────────────────────────────────────────────────────────

def variance_loss(z, gamma=1.0, eps=1e-4):
    """VICReg variance: force per-dimension std >= gamma across the batch."""
    z_flat = z.reshape(-1, z.shape[-1])           # (B*N, C)
    std = torch.sqrt(z_flat.var(dim=0) + eps)      # (C,)
    return F.relu(gamma - std).mean()


def covariance_loss(z):
    """VICReg covariance: penalise off-diagonal entries → decorrelate features."""
    z_flat = z.reshape(-1, z.shape[-1])            # (B*N, C)
    z_flat = z_flat - z_flat.mean(dim=0)
    N, D = z_flat.shape
    cov = (z_flat.T @ z_flat) / max(N - 1, 1)     # (C, C)
    off_diag = cov.masked_fill(
        torch.eye(D, device=z.device, dtype=torch.bool), 0.0
    )
    return (off_diag ** 2).sum() / D


def token_cosine_loss(pred, target):
    """Per-token cosine loss in C-dim space.

    pred, target: (B, N, C)
    Computes cosine similarity per token (dim=-1) then averages.
    This is the correct way — NOT flattening to (B, N*C) which
    loses all structure in high-dimensional space.
    """
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def image_edge_loss(pred_patches, target_patches):
    """Preserve spatial gradients (edges) in the predicted image."""
    pred   = unpatchify(pred_patches)
    target = unpatchify(target_patches)
    pred_dx   = pred[:, :, :, 1:]  - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy   = pred[:, :, 1:, :]  - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def image_ssim_loss(pred_patches, target_patches, window=8):
    """Sliding-window SSIM loss (disabled by default, ssim_weight=0.0)."""
    pred_img   = unpatchify(pred_patches)
    target_img = unpatchify(target_patches)
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    def local_stats(x):
        return F.avg_pool2d(x, kernel_size=window, stride=window)

    mu1, mu2 = local_stats(pred_img), local_stats(target_img)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = (local_stats(pred_img   * pred_img)   - mu1_sq).clamp(min=0)
    sigma2_sq = (local_stats(target_img * target_img) - mu2_sq).clamp(min=0)
    sigma12   =  local_stats(pred_img   * target_img) - mu12
    ssim_map  = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
                 (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


def spatial_diversity_loss(tokens):
    """Penalise predicted patches collapsing to identical representations within
    the same image.

    VICReg measures variance across the combined B*N population, which can be
    satisfied even when all N patches in image i share the same delta — as long
    as different images provide between-image variance.  This loss closes that
    gap by directly penalising the mean off-diagonal cosine similarity among the
    N patch vectors within each image.  At full collapse (all patches identical)
    the loss equals 1.0; at perfect diversity it approaches 0.

    tokens : (B, N, C)
    """
    normed = F.normalize(tokens, dim=-1)                             # (B, N, C)
    sim    = torch.bmm(normed, normed.transpose(1, 2))               # (B, N, N)
    N      = tokens.shape[1]
    off    = ~torch.eye(N, dtype=torch.bool, device=tokens.device)   # (N, N) mask
    return sim[:, off].clamp(min=0).mean()


def patch_nce_loss(pred, target, temperature=0.2):
    """Per-patch-position InfoNCE across the batch.

    For each spatial position j, pred[i, j] must identify its own future
    target[i, j] among all target[k, j] in the batch (k = 0..B-1).
    This forces the model to produce position-aware predictions rather than a
    single global-average token broadcast to every patch.

    pred, target : (B, N, C)  — L2-normalised inside this function.
    Returns scalar loss; skipped when B < 2 (no negatives available).
    """
    B, N, C = pred.shape
    if B < 2:
        return pred.new_tensor(0.0)
    pred_n = F.normalize(pred,   dim=-1).permute(1, 0, 2)   # (N, B, C)
    tgt_n  = F.normalize(target, dim=-1).permute(1, 0, 2)   # (N, B, C)
    # (N, B, B): similarity of each pred to every target, per position
    logits = torch.bmm(pred_n, tgt_n.transpose(1, 2)) / temperature
    labels = torch.arange(B, device=pred.device).unsqueeze(0).expand(N, -1)
    return F.cross_entropy(logits.reshape(N * B, B), labels.reshape(N * B))


# ── 1. Argument parser ────────────────────────────────────────────────────────
def parse_args():
    args = _wm2.parse_args()

    if args.contrastive_weight is None:
        args.contrastive_weight = 0.1
    if not hasattr(args, "ssim_weight"):
        args.ssim_weight = 0.0
    if not hasattr(args, "lidar_weight") or args.lidar_weight == 1.0:
        args.lidar_weight = 1.5
    if not hasattr(args, "variance_weight"):
        args.variance_weight = 0.2
    # WM3/4 default: wider inner dim. Only override if user left it at the WM2 default (128).
    if getattr(args, "temporal_inner_dim", 128) == 128:
        args.temporal_inner_dim = 256

    if "s2_world_model_2" in args.output_dir:
        args.output_dir = args.output_dir.replace("s2_world_model_2", "s2_world_model_3")
        args.checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    return args


# ── 2. Training epoch ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, args, train: bool, scaler=None):
    model.train(train)
    tot = tot_rgb = tot_lid = tot_ctr = tot_spa = 0.0
    steps = 0
    pbar = tqdm(loader, desc="Train" if train else "Val  ", leave=False, file=sys.stdout)
    use_amp = torch.cuda.is_available()

    for batch in pbar:
        rgb_seq = batch["rgb_sequence"].to(DEVICE)
        lid_seq = batch["lidar_sequence"].to(DEVICE)
        ego_seq = batch["ego_sequence"].to(DEVICE)
        rgb_tgt = batch["rgb_target"].to(DEVICE)
        lid_tgt = batch["lidar_target"].to(DEVICE)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(rgb_seq, lid_seq, ego_seq)
                with torch.no_grad():
                    tgt_rgb_g,  tgt_lid_g  = model.encode_target(rgb_tgt,         lid_tgt)
                    base_rgb_g, base_lid_g = model.encode_target(rgb_seq[:, -1],  lid_seq[:, -1])

                # ── Token losses: magnitude (Huber) + direction (cosine) ──────
                # Huber is robust to outlier residuals; cosine ensures directional alignment.
                # Both computed per-token in 128-d space — NOT flattened (avoids high-dim curse).
                loss_rgb_mse = F.smooth_l1_loss(out["pred_rgb_tokens"],   tgt_rgb_g,  beta=0.1)
                loss_lid_mse = F.smooth_l1_loss(out["pred_lidar_tokens"], tgt_lid_g,  beta=0.1)
                loss_rgb_cos = token_cosine_loss(out["pred_rgb_tokens"],   tgt_rgb_g)
                loss_lid_cos = token_cosine_loss(out["pred_lidar_tokens"], tgt_lid_g)

                # ── RGB patch reconstruction ──────────────────────────────────
                tgt_patches        = patchify(rgb_tgt)
                last_patches       = patchify(rgb_seq[:, -1])
                target_patch_delta = tgt_patches - last_patches
                true_mean = tgt_patches.mean(dim=-1, keepdim=True)
                true_var  = tgt_patches.var( dim=-1, keepdim=True)

                loss_color = 0.1 * (
                    F.mse_loss(out["pred_rgb_mean"], true_mean)
                    + F.mse_loss(out["pred_rgb_var"],  true_var)
                )
                ssim_weight = getattr(args, "ssim_weight", 0.0)
                loss_patch = (
                    F.l1_loss(out["pred_rgb_patches"], tgt_patches)
                    + 0.5  * F.smooth_l1_loss(out["pred_rgb_patch_delta"], target_patch_delta)
                    + 0.25 * image_edge_loss(out["pred_rgb_patches"], tgt_patches)
                    + ssim_weight * image_ssim_loss(out["pred_rgb_patches"], tgt_patches)
                )

                # Combine modality losses (cosine weight 0.3 balances with Huber scale)
                loss_rgb = loss_rgb_mse + 0.3 * loss_rgb_cos + loss_color + args.rgb_patch_weight * loss_patch
                loss_lid = loss_lid_mse + 0.3 * loss_lid_cos

                # ── Direction margin loss ─────────────────────────────────────
                # Penalise when prediction is MORE similar to the current frame
                # than to the future frame. Prevents copy-present collapse.
                # Uses per-token cosine in 128-d — correct.
                rgb_future_sim  = F.cosine_similarity(out["pred_rgb_tokens"],   tgt_rgb_g,  dim=-1)
                rgb_present_sim = F.cosine_similarity(out["pred_rgb_tokens"],   base_rgb_g, dim=-1)
                lid_future_sim  = F.cosine_similarity(out["pred_lidar_tokens"], tgt_lid_g,  dim=-1)
                lid_present_sim = F.cosine_similarity(out["pred_lidar_tokens"], base_lid_g, dim=-1)

                loss_contrast = 0.5 * (
                    F.relu(rgb_present_sim - rgb_future_sim + 0.1).mean()
                    + F.relu(lid_present_sim - lid_future_sim + 0.1).mean()
                )

                # ── VICReg anti-collapse ──────────────────────────────────────
                # Apply to DELTAS, not total tokens. Total tokens = last_frame +
                # delta; the frozen Stage 2 last_frame already has std ≥ 1 so
                # variance_loss on total tokens is ≈ 0 and gives zero gradient to
                # the delta predictor. Targeting the deltas with gamma=0.1 ensures
                # the predicted changes are diverse across batch/patches.
                loss_var = (
                    variance_loss(out["pred_rgb_token_delta"],   gamma=0.1)
                    + variance_loss(out["pred_lidar_token_delta"], gamma=0.1)
                )
                loss_cov = (
                    covariance_loss(out["pred_rgb_token_delta"])
                    + covariance_loss(out["pred_lidar_token_delta"])
                )

                # ── Spatial diversity: force patches to differ within each image ─
                # VICReg enforces across-batch diversity but allows every patch in
                # one image to collapse to the same delta. spatial_diversity_loss
                # directly penalises that by measuring mean off-diagonal cosine
                # similarity of the N deltas within each image (root cause of the
                # PCA cluster seen in visualisations).
                loss_spatial = (
                    spatial_diversity_loss(out["pred_rgb_token_delta"])
                    + spatial_diversity_loss(out["pred_lidar_token_delta"])
                )

                # ── Patch NCE: position-aware contrastive across the batch ───────
                # For each spatial position j, pred[i,j] must rank above all
                # target[k,j] (k≠i) in the batch. Complements the direction-margin
                # loss which only operates at the image level.
                loss_patch_nce = patch_nce_loss(out["pred_rgb_tokens"], tgt_rgb_g)

                lidar_weight    = getattr(args, "lidar_weight",    1.5)
                variance_weight = getattr(args, "variance_weight", 0.2)
                loss = (
                    loss_rgb
                    + lidar_weight    * loss_lid
                    + args.contrastive_weight * loss_contrast
                    + variance_weight * loss_var
                    + 0.1             * loss_cov
                    + 0.3             * loss_spatial
                    + 0.05            * loss_patch_nce
                )

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm)
                    optimizer.step()

        tot     += loss.item()
        tot_rgb += loss_rgb.item()
        tot_lid += loss_lid.item()
        tot_ctr += loss_contrast.item()
        tot_spa += loss_spatial.item()
        steps   += 1
        pbar.set_postfix(
            loss=f"{tot/steps:.4f}",
            rgb=f"{tot_rgb/steps:.4f}",
            lid=f"{tot_lid/steps:.4f}",
            ctr=f"{tot_ctr/steps:.4f}",
            spa=f"{tot_spa/steps:.4f}",
        )

    d = max(steps, 1)
    return tot / d, tot_rgb / d, tot_lid / d


# ── 3. Train loop (cosine LR scheduler) ──────────────────────────────────────
def train(model, train_dl, val_dl, args):
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    log = {k: [] for k in [
        "train_loss", "train_rgb_loss", "train_lidar_loss",
        "val_loss",   "val_rgb_loss",   "val_lidar_loss",
    ]}
    best_val = float("inf")
    start_ep = 1

    resume = _wm2.maybe_resume_path(args)
    ckpt   = None
    resume_training_state = False
    if resume:
        try:
            ckpt = torch.load(resume, map_location=DEVICE, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume, map_location=DEVICE)
        incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if incompatible.missing_keys:
            print(f"Warm-start missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Warm-start ignored keys: {incompatible.unexpected_keys}")
        architecture_changed = bool(incompatible.missing_keys or incompatible.unexpected_keys)
        if architecture_changed:
            print("Architecture changed: resetting optimizer, scheduler, history.")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            resume_training_state = True
        start_ep = int(ckpt["epoch"]) + 1
        if resume_training_state:
            best_val = float(ckpt.get("best_val_loss", best_val))
            log = ckpt.get("history", log)
        print(f"Resumed from {resume} at epoch {ckpt['epoch']}.")

    if start_ep > args.epochs:
        raise ValueError(
            f"Checkpoint already at epoch {start_ep - 1} but --epochs={args.epochs}. "
            "Increase --epochs or start fresh."
        )

    remaining = max(args.epochs - start_ep + 1, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=args.lr * 0.01)
    if resume_training_state and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if resume_training_state and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    lidar_weight    = getattr(args, "lidar_weight",    1.5)
    variance_weight = getattr(args, "variance_weight", 0.2)

    inner_dim = getattr(args, "temporal_inner_dim", 256)
    print(f"\n{'='*60}")
    print("  OmniVec2  Stage 2 -> World Model  (v5)")
    print(f"  Arch            : inner_dim={inner_dim}, layers={args.temporal_layers}, heads={args.temporal_heads}")
    print(f"  Pixel head      : connected (no detach — co-supervises temporal encoder)")
    print(f"  Token loss      : Huber(beta=0.1) + 0.3*cosine (per-token)")
    print(f"  LiDAR weight    : {lidar_weight}")
    print(f"  RGB patch weight: {args.rgb_patch_weight}")
    print(f"  Contrastive wt  : {args.contrastive_weight}  (direction margin, image-level)")
    print(f"  Variance wt     : {variance_weight}  (VICReg anti-collapse, B*N-level)")
    print(f"  Covariance wt   : 0.1  (VICReg feature decorrelation)")
    print(f"  Spatial div wt  : 0.3  (within-image patch diversity)")
    print(f"  Patch NCE wt    : 0.05 (per-position contrastive across batch)")
    print(f"  LR schedule     : CosineAnnealing (T={remaining})")
    print(f"  Trainable params: {sum(p.numel() for p in params):,}")
    print(f"  Epochs          : {start_ep} -> {args.epochs}")
    print(f"{'='*60}\n")

    for ep in range(start_ep, args.epochs + 1):
        t0 = time.time()
        tr,  tr_r, tr_l = run_epoch(model, train_dl, optimizer, args, train=True,  scaler=scaler)
        val, vr,   vl   = run_epoch(model, val_dl,   optimizer, args, train=False, scaler=None)
        scheduler.step()

        log["train_loss"].append(tr);      log["train_rgb_loss"].append(tr_r)
        log["train_lidar_loss"].append(tr_l)
        log["val_loss"].append(val);       log["val_rgb_loss"].append(vr)
        log["val_lidar_loss"].append(vl)

        is_best = val < best_val
        if is_best:
            best_val = val

        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"Ep {ep:03d}/{args.epochs} ({time.time()-t0:.0f}s) lr={cur_lr:.2e} | "
            f"train {tr:.4f} (rgb {tr_r:.4f} lid {tr_l:.4f}) | "
            f"val {val:.4f} (rgb {vr:.4f} lid {vl:.4f})"
            + (" ★ best" if is_best else "")
        )

        if ep % args.save_every == 0:
            _wm2.save_checkpoint(args, {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "epoch": ep, "best_val_loss": best_val,
                "history": log, "args": vars(args),
            }, is_best)

    return log


# ── 4. Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    _wm2.set_seed(args.seed)
    os.makedirs(args.output_dir,      exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    print(f"Device         : {DEVICE}")
    print(f"NuScenes       : {args.version} @ {args.dataroot}")
    print(f"Stage 2 ckpt   : {args.stage2_checkpoint}")
    print(f"Output         : {args.output_dir}")

    from nuscenes.nuscenes import NuScenes
    from s2_world_model_2.temporal_dataset import build_world_model_dataloaders
    from s2_world_model_2.world_model_s2 import OmniVec2Stage2WorldModel, TemporalTokenPredictor
    from model import OmniVec2Stage1
    from stage2 import OmniVec2Stage2
    from config import FC_HIDDEN

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    train_dl, val_dl = build_world_model_dataloaders(
        nusc=nusc, dataroot=args.dataroot,
        batch_size=args.batch_size, num_workers=args.num_workers,
        split_ratio=args.train_split_ratio, scene_limit=args.scene_limit,
        seed=args.seed, history=args.history, steps_ahead=args.steps_ahead,
    )

    stage1 = OmniVec2Stage1().to(DEVICE)
    stage2 = OmniVec2Stage2(stage1).to(DEVICE)
    _wm2.load_stage2_checkpoint(stage2, args.stage2_checkpoint)
    stage2.eval()

    temporal = TemporalTokenPredictor(
        embed_dim=FC_HIDDEN,
        inner_dim=getattr(args, "temporal_inner_dim", 256),
        num_heads=args.temporal_heads,
        num_layers=args.temporal_layers,
        max_history=args.history,
    )
    model = OmniVec2Stage2WorldModel(stage2, temporal, freeze_stage2=True).to(DEVICE)

    history_log = train(model, train_dl, val_dl, args)

    if not args.skip_visualizations:
        print("\nGenerating visualizations...")
        _wm2.plot_curves(history_log, args.output_dir)
        _wm2.visualize(model, val_dl, args)

    export = os.path.join(args.output_dir, "s2_wm3_temporal_predictor.pth")
    torch.save(model.temporal.state_dict(), export)
    print(f"\nExported -> {export}")
    print("Done.")


if __name__ == "__main__":
    main()
