# Photometric Mitigation Experiment: pan_2 / sharp_2

Date: 2026-06-04

## Goal

Evaluate five ways to reduce bias from exposure, ISO, aperture, and general intensity differences between the panning frame and the registered sharp reference during reference-based blur-kernel estimation.

Manual reference blur length used for comparison: `140 px`.

## EXIF

| Image | Exposure | ISO | Aperture | Focal length | Photometric value |
|---|---:|---:|---:|---:|---:|
| pan_2 | 0.008000 s | 160 | f/16.0 | 38.0 mm | 0.005000 |
| sharp_2 | 0.000500 s | 320 | f/5.6 | 45.0 mm | 0.005102 |

The EXIF photometric ratio used by `exif_linear` is `(pan exposure * pan ISO / pan f_number^2) / (sharp exposure * sharp ISO / sharp f_number^2) = 0.98`. The exposure time differs strongly, but ISO and aperture nearly cancel it, so EXIF-linear scaling is close to neutral for this pair.

## Implemented Options

| Mode | Option | Implementation |
|---|---|---|
| `patch_affine` | 1. Patch-level affine photometric fit | For every candidate blur length, solve `blurry_patch ~= a * reblurred_sharp_patch + c`, then minimize the residual MSE. |
| `global_affine` | 2. Global background exposure matching | Fit one `a, c` over valid non-car background pixels after registration, transform the registered sharp image once, then run the normal pixel loss. |
| `robust_norm` | 3. Robust patch normalization | Normalize each patch by median and robust percentile scale `(p95 - p5) / 2` before spectral/pixel comparison. |
| `gradient` | 4. Gradient-domain fitting | Compare Sobel gradient stacks with an optimal scalar gain for each candidate blur length. |
| `exif_linear` | 6. EXIF-based linear exposure normalization | Scale the registered sharp image by the EXIF photometric exposure ratio before fitting. |

The raw baseline was also run for comparison. All modes used the same existing registration, car mask, valid mask, patch grid, trajectory RANSAC settings, and `manual_blur_px=140`.

## Results

| Mode | Global blur px | Delta vs 140 px | RANSAC RMSE px | Inliers | Kernel median px | Common residual RMSE | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `raw` | 170.70 | +30.70 | 8.47 | 55/75 | 173.96 | 24.76 | Original behavior; chooses high-blur cluster. |
| `patch_affine` | 143.54 | +3.54 | 6.88 | 33/75 | 154.79 | 24.75 | Strong improvement; close to manual reference. |
| `global_affine` | 134.30 | -5.70 | 6.40 | 34/75 | 152.82 | 24.81 | Good RMSE; slightly undershoots manual reference. |
| `robust_norm` | 139.27 | -0.73 | 6.23 | 38/75 | 137.74 | 24.77 | Best balanced result; closest to manual and low RMSE. |
| `gradient` | 174.92 | +34.92 | 5.10 | 34/75 | 153.88 | 24.78 | Lowest RMSE but locks onto high-blur cluster. |
| `exif_linear` | 168.20 | +28.20 | 10.01 | 55/75 | 170.70 | 24.76 | Near raw because EXIF photometric ratio is only 0.98. |

## Best Option

`robust_norm` performed best overall for this pair. It estimated a global blur of `139.27 px`, only `-0.73 px` from the manual `140 px` reference, with `6.23 px` trajectory weighted RMSE and `38/75` RANSAC inliers.

`gradient` had the lowest trajectory RMSE (`5.10 px`) but selected the same high-blur family as the old raw pipeline (`174.92 px`). That makes it internally consistent but likely wrong for the blur-length question we are trying to solve.

`patch_affine` was the second-best practical option: `143.54 px`, `+3.54 px` from manual, and lower RMSE than raw. It is useful if we want local exposure matching while preserving raw image contrast in each patch.

## Artifacts

- Results JSON: `outputs/photometric_compare_pan_2__sharp_2/comparison_results.json`
- Results CSV: `outputs/photometric_compare_pan_2__sharp_2/comparison_results.csv`
- `raw` outputs: `outputs/photometric_compare_pan_2__sharp_2/raw/`
- `patch_affine` outputs: `outputs/photometric_compare_pan_2__sharp_2/patch_affine/`
- `global_affine` outputs: `outputs/photometric_compare_pan_2__sharp_2/global_affine/`
- `robust_norm` outputs: `outputs/photometric_compare_pan_2__sharp_2/robust_norm/`
- `gradient` outputs: `outputs/photometric_compare_pan_2__sharp_2/gradient/`
- `exif_linear` outputs: `outputs/photometric_compare_pan_2__sharp_2/exif_linear/`

## Pipeline Decision

The default reference-based kernel estimator now uses `--photometric_mode robust_norm`. Other modes remain available for ablations: `raw`, `patch_affine`, `global_affine`, `gradient`, and `exif_linear`.
