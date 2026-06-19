# OmniVec2 Full Standalone Folder

This folder is self-contained. Copy `omnivec2_full/` to HiperGator, `cd` into it,
and run the scripts from this folder.

## What Trains Where

- Stage 1: first 500 nuScenes scenes
- Stage 2: first 500 nuScenes scenes, initialized from Stage 1
- Stage 3 detection: next 500 scenes, scene indices `500:1000`

All outputs are written inside this copied folder:

```text
runs/stage1_first500/
runs/stage2_first500/
runs/stage3_det_next500/
```

Important outputs:

```text
runs/stage1_first500/checkpoints/stage1_best.pth
runs/stage2_first500/exports/omnivec2_stage2_fg.pth
runs/stage3_det_next500/checkpoints/stage3_det_best.pth
runs/stage3_det_next500/det_3d_boxes.png
runs/stage3_det_next500/det_bev_predictions.png
```

## Submit Full Pipeline

```bash
cd omnivec2_full
bash submit_full_pipeline.sh
```

This submits Stage 1, then Stage 2 after Stage 1 succeeds, then Stage 3 detection
after Stage 2 succeeds.

## Submit Manually

```bash
sbatch run_full_stage1_first500.sh
sbatch --dependency=afterok:<stage1_job_id> run_full_stage2_first500.sh
sbatch --dependency=afterok:<stage2_job_id> run_full_stage3_det_next500.sh
```

## Common Overrides

```bash
DATAROOT=/orange/iruchkin/isen/nsfull
VERSION=v1.0-trainval
CONDA_ENV_NAME=omnivec2
SCENE_LIMIT=500
DET_START_SCENE=500
DET_SCENE_COUNT=500
BATCH_SIZE=32
NUM_WORKERS=8
```

Example:

```bash
DATAROOT=/orange/iruchkin/isen/nsfull BATCH_SIZE=32 bash submit_full_pipeline.sh
```

## L4 GPU Resource Starting Point

For one HiperGator L4 GPU, start with:

```text
Stage 1: 1 GPU, 8 CPUs, 64 GB RAM, 72 hours, batch size 32
Stage 2: 1 GPU, 8 CPUs, 64 GB RAM, 72 hours, batch size 32
Stage 3 detection: 1 GPU, 8 CPUs, 64+ GB RAM, 120 hours, batch size 32
```

Stage 3 detection is the most likely to run out of memory or need more time. If
it fails with CUDA OOM, use `BATCH_SIZE=4`, `BATCH_SIZE=2`, or `BATCH_SIZE=1`.
