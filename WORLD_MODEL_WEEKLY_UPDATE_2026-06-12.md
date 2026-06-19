# OmniVec2 World Model: Weekly Update

## Goal

Build a world model for nuScenes that uses OmniVec2 Stage 2 features to predict
RGB and LiDAR one second into the future.

## Work Completed

- Connected the frozen OmniVec2 Stage 2 encoder to a temporal transformer.
- Used four history frames, RGB, LiDAR, and ego-motion as input.
- Added RGB and LiDAR future-token losses.
- Added contrastive learning and cosine learning-rate scheduling.
- Fixed checkpoint resume, epoch handling, and World Model 3 training errors.
- Added comparisons against a copy-last-frame baseline.

## Results

The model learned useful future hidden features:

- RGB token MSE improved by approximately **50%** over the baseline.
- LiDAR token MSE improved by approximately **50%** over the baseline.

However, the future RGB image was almost completely gray and the predicted
tokens collapsed into a small PCA cluster. This showed that good hidden-token
metrics did not automatically produce a useful future image.

## Latest Improvements

We redesigned the model to:

- start from the last real frame instead of generating an image from zero;
- predict only the future change in RGB, RGB tokens, and LiDAR tokens;
- use edge loss to preserve roads, vehicles, and buildings;
- use batch InfoNCE to reduce PCA/token collapse;
- report real RGB pixel MAE and MSE;
- show the predicted image and its change map in visualizations.

The previous epoch-10 model, code, and results were backed up before these
changes.

## Next Step

Train the redesigned model from scratch for 10 epochs and evaluate:

- whether the future RGB image is no longer gray;
- whether it beats copy-last-frame in pixel metrics;
- whether PCA predictions move toward the future target distribution;
- whether RGB and LiDAR validation losses continue improving.
