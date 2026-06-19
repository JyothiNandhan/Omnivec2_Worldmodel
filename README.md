# OmniVec2 — Multimodal MAE + World Model (RGB + LiDAR, PATCH_SIZE=8)

**Paper:** OmniVec2 – A Novel Multimodal Multitask Network

This is the **reduced-patch-size** variant (`PATCH_SIZE=8`, down from the paper's 16), giving
784 patches per image (28×28) instead of 196.  It runs on the University of Florida
**HiPerGator** SLURM cluster against the nuScenes dataset at `/orange/iruchkin/isen/nsfull`.

---

## Project Structure

```
omnivec2_reduced_PatchSize/
│
├── config.py                  ← All hyperparameters & CLI flags (single source of truth)
├── model.py                   ← OmniVec2Stage1 (RGB + LiDAR masked autoencoder)
├── stage2.py                  ← OmniVec2Stage2 (cross-modal fusion, g(.) transformer)
├── train.py                   ← Stage 1 training entry point
├── train_stage2.py            ← Stage 2 training entry point
├── train_s2_world_model_2.py  ← World Model v2 training (base)
├── train_s2_world_model_3.py  ← World Model v3 training (current — extends WM2)
├── checkpointing.py           ← Resume / best / rolling-checkpoint helpers
├── requirements.txt           ← Python dependencies
│
├── run_full_stage1_first500.sh  ← SLURM: Stage 1, first 500 scenes
├── run_full_stage2_first500.sh  ← SLURM: Stage 2, first 500 scenes
├── run_s2_world_model_3.sh      ← SLURM: World Model v3 (active)
├── run_stage1.sh / run_stage2.sh ← Standalone SLURM scripts
│
├── data/                      ← Data loading (both modalities)
│   ├── rgb_dataset.py         #   NuScenes camera dataset
│   ├── lidar_dataset.py       #   NuScenes LIDAR_TOP dataset
│   ├── lidar_helpers.py       #   FPS + kNN patching (Point-BERT style)
│   └── build.py               #   build_dataloaders() → 4 loaders
│
├── rgb/                       ← Image model components
│   ├── tokenizer.py           #   Conv2d patch embedding (PATCH_SIZE=8)
│   ├── decoder.py             #   Transformer → pixel patches
│   ├── patches.py             #   patchify / unpatchify / normalize
│   └── visualize.py           #   Patch grid, token similarity, norms, pos encoding
│
├── lidar/                     ← Point-cloud model components
│   ├── tokenizer.py           #   PointNet-style patch encoder
│   ├── decoder.py             #   Transformer → 3D patches
│   └── visualize.py           #   FPS centers, patch groups, token similarity, norms
│
├── shared/                    ← Shared by both modalities
│   ├── encoder.py             #   Shared encoder f(·)
│   ├── masking.py             #   MAE-style random masking
│   ├── losses.py              #   MSE loss + PSNR metric
│   └── positional.py          #   Sinusoidal (2D) + learned (3D) positional encodings
│
└── s2_world_model_2/          ← World Model components (WM2 & WM3 share this)
    ├── world_model_s2.py      #   OmniVec2Stage2WorldModel + TemporalTokenPredictor
    └── temporal_dataset.py    #   NuScenesWorldModelDataset (history windows)
```

---

## Key Dimensions (`config.py`)

| Name | Value | Role |
|------|-------|------|
| `PATCH_SIZE` | **8** | RGB patch size (half the paper's 16) |
| `IMG_SIZE` | 224 | Input image resolution |
| `NUM_PATCHES` | **784** | (224 / 8)² patches per image |
| `EMBED_DIM` | 256 | Transformer hidden dim into f(.) |
| `FC_HIDDEN` | 128 | Output dim of f(.) / token dim throughout Stage 2 + WM |
| `NUM_GROUP` | 64 | LiDAR FPS center count |
| `GROUP_SIZE` | 32 | Points per LiDAR group (kNN) |

---

## Architecture

### Stage 1 (`model.py` → `OmniVec2Stage1`)

- **RGB path**: `RGBPatchEmbedding` (Conv2d, `PATCH_SIZE=8`) → 75 % random masking → `SharedEncoder f(.)` → `RGBDecoder`
- **LiDAR path**: `LidarPatchEncoder` (PointNet-style) + `LidarPositionalEmbedding` → masking → same `SharedEncoder f(.)` → `LidarDecoder`
- Training interleaves 6 RGB batches per 1 LiDAR batch.
- Loss: masked MSE on normalized patches (both modalities).

### Stage 2 (`stage2.py` → `OmniVec2Stage2`)

Adds cross-modal fusion on top of f(.):

```
tokenizer → f(.) → CrossAttentionBlock → g(.) → cross-attn-back → decoder
```

Each modality's f(.) features attend to the other, pass through shared `SecondTransformerG g(.)`,
then attend back to their own f(.) features.  Stage 1 decoders are replaced with fresh Stage 2
decoders for the masked reconstruction loss.

### World Model — WM2 / WM3 (active development)

Located in `s2_world_model_2/`, trained by `train_s2_world_model_2.py` (base) and
`train_s2_world_model_3.py` (current).  WM3 imports WM2 as a module and overrides only
`parse_args`, `run_epoch`, and `train`.

**Data flow** (`temporal_dataset.py`):
- `NuScenesWorldModelDataset` builds windows of `history + steps_ahead` consecutive frames.
- Each sample: 4 history frames (RGB + LiDAR + ego) → predict 1 target frame.
- NuScenes runs at 2 Hz; `steps_ahead=2` → 1 second ahead.
- Ego-motion (7D: translation + quaternion) is zero-centred per window.

**Architecture** (`world_model_s2.py`):
- Frozen Stage 2 backbone (`OmniVec2Stage2`) + trainable `TemporalTokenPredictor`.
- `TemporalTokenPredictor`: encodes history g(.) token sequences for RGB, LiDAR, and ego
  (via MLP), concatenates them, runs joint spatio-temporal attention `(B, T×N_total, C)`,
  and outputs **residual deltas** added to the last observed frame — prevents copy-last collapse.

**WM3 additions over WM2**:
- Batch InfoNCE contrastive loss to prevent PCA/token collapse
- Edge loss on predicted RGB patches (preserves roads, vehicles, buildings)
- Cosine LR schedule
- Residual delta prediction (predict future change, not absolute tokens)

---

## Environment Setup

Two conda environments are used on HiPerGator:

| Env | Used by |
|-----|---------|
| `omnivec2` | Stage 1, Stage 2 |
| `omnivec2_fix` | World Model (WM2, WM3) |

```bash
module load conda
conda activate omnivec2        # for Stage 1 / Stage 2
conda activate omnivec2_fix    # for World Model
```

### Create from scratch (conda — HiPerGator)

```bash
module load conda
conda create -n omnivec2 python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omnivec2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Stage 1 — local smoke test
python train.py --dataroot /orange/iruchkin/isen/nsfull --output_dir ./results --epochs 5

# World Model — local smoke test (2 scenes, 1 epoch)
python train_s2_world_model_2.py \
    --dataroot /orange/iruchkin/isen/nsfull \
    --stage2_checkpoint ./runs/stage2_first500/checkpoints/stage2_best.pth \
    --output_dir ./runs/s2_world_model_2 \
    --scene_limit 2 --epochs 1
```

---

## Running on HiPerGator

### Submit the full pipeline

```bash
sbatch run_full_stage1_first500.sh
sbatch --dependency=afterok:<stage1_job_id> run_full_stage2_first500.sh
sbatch --dependency=afterok:<stage2_job_id> run_s2_world_model_3.sh
```

### Monitor

```bash
squeue -u YOUR_GATORLINK
tail -f omnivec2_s2_wm3_<JOB_ID>.log
```

### Resuming

All training scripts support `--auto_resume` (picks up `<prefix>_last.pth` automatically):

```bash
python train.py --output_dir ./runs/stage1_nuscenes --auto_resume
```

---

## Checkpoint Layout

```
runs/
  stage1_first500/
    checkpoints/   → stage1_best.pth, stage1_last.pth, stage1_checkpoint1..4.pth
    exports/       → omnivec2_stage1_rgb_lidar.pth, omnivec2_stage1_full.pth
  stage2_first500/
    checkpoints/   → stage2_best.pth, stage2_last.pth
    exports/       → omnivec2_stage2_fg.pth  (tokenizers + f + g + cross-attn)
  s2_world_model_3/
    checkpoints/   → wm_s2_best.pth, wm_s2_last.pth
                   → s2_wm3_temporal_predictor.pth  (temporal predictor only)
```

Stage 2 checkpoints embed all Stage 1 weights — only a Stage 2 checkpoint is needed to run
the World Model.

Checkpoint policy:
- `*_best.pth`: lowest validation loss so far (full restartable state)
- `*_last.pth`: latest full training state
- `*_checkpoint1..4.pth`: rolling recent checkpoints

---

## Output Visualizations

### Post-tokenizer (before training)
| File | What it shows |
|------|---------------|
| `rgb_patch_grid.png` | Image divided into 784 patches with grid overlay |
| `rgb_token_similarity.png` | Cosine similarity between patch embeddings |
| `rgb_positional_encoding.png` | Spatial patterns in sinusoidal pos encoding |
| `rgb_token_norms.png` | Per-patch activation strength heatmap |
| `lidar_fps_centers.html` | FPS-selected center points in full cloud |
| `lidar_patch_groups.html` | Each kNN group in a different color |
| `lidar_token_similarity.png` | Patch embedding similarity + norm bar chart |
| `lidar_token_norms_3d.html` | 3D cloud colored by token activation |

### Post-training
| File | What it shows |
|------|---------------|
| `training_curves.png` | Loss + PSNR for both modalities |
| `rgb_reconstruction.png` | Original → Masked → Reconstructed images |
| `lidar_masking.html` | Visible (blue) vs masked (red) patches |
| `lidar_reconstruction.html` | Original vs reconstructed point cloud |
