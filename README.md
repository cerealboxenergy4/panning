# Panning Shot Analysis — Blur Kernel Estimation & Speed Recovery

Recovers the camera angular velocity and F1 car ground speed from a single
panning photograph paired with a sharp reference image of the same car.

The core idea: motion blur in a panning shot is **signal, not degradation**.
The length and direction of the blur kernel encode the camera trajectory during
the exposure. By comparing the blurry frame to a registered sharp reference,
we can extract that kernel without any deconvolution.

---

## Pipeline Overview

```
pan_1.jpg  ──────────────────────────────────────────────────────────────────┐
           │                                                                  │
           ▼                                                                  ▼
   [ Stage 1 ]                                                        [ Stage 3 ]
  segment_car.py                                                  kernel_estimation.py
  Grounded SAM2                                                   ├─ global φ from full image
  → car_mask.png                                                  ├─ per-patch sinc² spectral fit
           │                                                      ├─ pixel-domain MSE refinement
           │                  sharp_1.jpg                         └─ kernel_map.npz / .png
           │                       │                                         │
           ▼                       ▼                                         ▼
   [ Stage 2 ]                                                        [ Stage 4 ]
  register_reference.py                                          trajectory_fitting.py
  LoFTR + RANSAC                                                  WLS: b_i = Bx·cosφ + By·sinφ
  → sharp_registered.png ────────────────────────────────────►   → trajectory.json
    sharp_registered_valid.png
```

---

## Stages

### Stage 1 — Car Segmentation (`segment_car.py`)

Isolates the F1 car so it is excluded from all background kernel estimates.

- **Grounding DINO** (text-prompted open-vocabulary detector) produces bounding
  boxes from a natural-language prompt (`"formula 1 racing car . f1 car . race car ."`).
- **SAM 2** refines those boxes into pixel-accurate instance masks.

Outputs:
- `outputs/car_mask.png` — binary mask (255 = car, 0 = background)
- `outputs/car_detection.png` — overlay visualization

```bash
python segment_car.py --image pan_1.jpg
```

---

### Stage 2 — Reference Registration (`register_reference.py`)

Warps `sharp_1.jpg` into the coordinate frame of `pan_1.jpg` via a projective
homography so that each background pixel in the blurry image has a corresponding
sharp pixel from the reference.

- **LoFTR** is the default matcher for the homography baseline. SIFT and ORB
  remain available with `--matcher sift` or `--matcher orb`.
- Images are resized to `long_side=840 px` before matching, then matches are
  scaled back to full resolution before RANSAC.
- Add `--depth_warp --depth_provider da3` or `--depth_warp --depth_provider vggt`
  to also save a depth-based unprojection/reprojection warp for comparison.
- Car-region matches are suppressed using the Stage 1 mask before homography
  estimation, since the car occupies a different depth plane.
- A validity mask tracks which pixels of the warped image contain real data
  vs. zero-padded fill, so Stage 3 can skip boundary patches.

Outputs:
- `outputs/sharp_registered.png` — sharp reference in blurry image frame
- `outputs/sharp_registered_valid.png` — coverage mask (white = valid)
- `outputs/homography.npy` — 3×3 homography H (maps sharp → blurry)
- `outputs/registration_debug.png` — inlier match visualization

```bash
python register_reference.py --blurry pan_1.jpg --sharp sharp_1.jpg
```

---

### Stage 3 — Kernel Estimation (`kernel_estimation.py`)

Estimates the blur kernel length `b` (in pixels) for each background patch using
the registered sharp image as a reference. The blur direction `φ` is treated as
a **global** constant (camera motion direction) and estimated once from the full
blurry image before any per-patch fitting.

#### Global blur direction

A weighted Sobel gradient histogram is built from the **entire blurry image**
(car pixels masked out). The dominant gradient direction + 90° gives `φ`. Using
the full image instead of per-patch estimates avoids direction errors caused by
patches where local scene gradients dominate over the blur signature.

```
φ = argmax_θ Σ_pixels |∇I| · [angle(∇I) == θ] + 90°
```

#### Per-patch kernel length

For each eligible background patch (not car, fully covered by the registered
sharp, sufficient gradient energy in the sharp reference):

**Primary estimate — pixel-domain MSE (`b_px_pixel`)**

Before pixel comparison, the sharp patch is aligned to the blurry patch in the
**perpendicular-to-blur direction** using phase cross-correlation (aligning along
the blur axis would confound with `b`). Then `b` is found by minimising
pixel-domain reconstruction error over the full search range with no prior:

```
b_px_pixel = argmin_{b ∈ [1, 0.45·P]}  ||blur_patch − sharp_aligned ⊛ k(b, φ)||²
```

Empirically this yields more accurate kernel estimates than initialising from the
spectral fit, because the spectral ratio is sensitive to registration noise and
JPEG compression artefacts in the frequency domain. `trajectory_fitting.py` and
the uniform trajectory baseline both use `b_px_pixel`.

**Auxiliary estimate — spectral sinc² fit (`b_px_spec`)**

Kept for diagnostics and side-by-side comparison in `kernel_patch_grid.png`.
Rotate both patches so `φ` is horizontal, compute the marginal power spectrum
ratio, and fit:

```
P_blur(fx) / P_sharp(fx)  ≈  sinc²(b · fx)
```

The ratio is median-filtered and the fit is done in the log domain with
low-frequency weighting (higher SNR at low f). Stored in `kernel_map.npz` as
`b_px_spec` but not used downstream.

#### Patch quality filters

| Filter | Criterion |
|--------|-----------|
| `skip_car` | patch car fraction > 10% |
| `skip_invalid_region` | valid mask coverage < 95% |
| `skip_flat_sharp` | Sobel energy in sharp patch < threshold (kernel underdetermined) |

#### Uniform trajectory baseline

A single global kernel `(b_global, φ)` is also computed as the
confidence-weighted mean of all valid `b_px_pixel` estimates and applied to the
**full registered sharp image**, producing a direct residual comparison against
`pan_1.jpg` without trajectory fitting.

Outputs:
- `outputs/kernel_map.npz` — per-patch arrays: `b_px`, `b_px_spec`, `b_px_pixel`, `phi_deg`, `confidence`, `status`
- `outputs/kernel_map.csv` — same as CSV
- `outputs/kernel_map.png` — overlay: arrows show blur direction, colour encodes `b_px_pixel`
- `outputs/kernel_patch_grid.png` — 8×6 diagnostic grid (4 near-mean + 4 outlier patches)
- `outputs/uniform_traj.json` — global `(Bx, By, b, φ)` from weighted mean
- `outputs/uniform_trajectory_residual.png` — blurry | re-blurred | |residual|

```bash
python kernel_estimation.py --blurry pan_1.jpg \
                             --sharp_reg outputs/sharp_registered.png \
                             --car_mask  outputs/car_mask.png
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--patch_size` | 400 | Patch edge length in pixels |
| `--grad_energy_thres` | 100 | Min Sobel energy in sharp patch |
| `--global_phi` | — | Hard-override blur direction (skips pre-pass) |
| `--valid_thres` | 0.95 | Min fraction of patch covered by registration |

---

### Stage 4 — Trajectory Fitting (`trajectory_fitting.py`)

Fits a global 2D displacement vector **B = (Bx, By)** in pixel space from all
inlier patch estimates. Under constant-velocity panning, each patch's blur
length satisfies:

```
b_i = Bx · cos(φ_i) + By · sin(φ_i)
```

This is solved as weighted least squares (confidence = Sobel energy of sharp
patch). One round of outlier rejection drops patches with |residual| > 2.5σ.

Because all φ_i are nearly identical (pure panning), the system is nearly
rank-1. When the weighted spread of φ is < 5°, a 1D scalar fit is used instead
of the full 2D WLS to avoid numerical amplification of noise into the
perpendicular component.

#### Physical conversion

```
pixel pitch   p = sensor_width / image_width_px
focal length  f_px = focal_mm / p

angular displacement  α = |B| · p / focal_mm   [rad]
angular velocity      ω = α / t_exposure        [rad/s]
```

Outputs:
- `outputs/trajectory.json` — fitted parameters + physical quantities (ω in deg/s and rad/s)
- `outputs/trajectory_residual.png` — blurry | re-blurred sharp | |residual|
- `outputs/trajectory_scatter.png` — measured vs predicted `b` per patch + spatial residual map

```bash
python trajectory_fitting.py --kernel_map outputs/kernel_map.npz \
                              --sharp_reg  outputs/sharp_registered.png \
                              --blurry     pan_1.jpg
```

---

## Running the Full Pipeline

```bash
conda activate tttnvs

python run_pipeline.py --blurry pan_1.jpg --sharp sharp_1.jpg
```

The runner pins child processes to GPU 4 by default and writes all artifacts to
`outputs/`. It skips stages whose expected outputs already exist; add `--force`
to rerun from scratch.

Useful variants:

```bash
# Print the commands without running them.
python run_pipeline.py --dry-run

# Rerun only kernel estimation and trajectory fitting.
python run_pipeline.py --start-at kernel --force

# Run a fresh pipeline into a separate output directory.
python run_pipeline.py --out_dir outputs_test --force
```

The stages can still be run individually for debugging:

```bash
python segment_car.py --image pan_1.jpg
python register_reference.py --blurry pan_1.jpg --sharp sharp_1.jpg
python kernel_estimation.py --blurry pan_1.jpg --sharp_reg outputs/sharp_registered.png --car_mask outputs/car_mask.png
python trajectory_fitting.py --kernel_map outputs/kernel_map.npz --sharp_reg outputs/sharp_registered.png --blurry pan_1.jpg
```

Camera constants (focal length, sensor size, exposure time) are hard-coded at
the top of `trajectory_fitting.py` — edit these to match your EXIF data before
running.

---

## Output Summary

| File | Stage | Description |
|------|-------|-------------|
| `car_mask.png` | 1 | Binary car segmentation |
| `sharp_registered.png` | 2 | Sharp reference in blurry frame |
| `sharp_registered_valid.png` | 2 | Registration coverage mask |
| `kernel_map.npz` | 3 | Per-patch kernel estimates |
| `kernel_map.png` | 3 | Kernel map overlay |
| `kernel_patch_grid.png` | 3 | Patch diagnostic grid (spectral vs pixel) |
| `uniform_traj.json` | 3 | Single-kernel baseline |
| `uniform_trajectory_residual.png` | 3 | Baseline residual image |
| `trajectory.json` | 4 | Fitted B, φ, ω |
| `trajectory_residual.png` | 4 | Full-image re-blur residual |
| `trajectory_scatter.png` | 4 | Measured vs predicted scatter |

---

## Dependencies

```
torch           # GPU inference for LoFTR, SAM 2, and optional depth providers
kornia          # LoFTR matcher
transformers    # Grounding DINO + SAM 2 via HuggingFace
opencv-python   # SIFT/ORB, homography, warping
scipy           # Sobel, rotation, optimization
matplotlib
Pillow
```

Tested under the `tttnvs` conda environment.
