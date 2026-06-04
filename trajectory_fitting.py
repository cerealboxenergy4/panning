"""
Stage 3 — Global camera trajectory fitting.

Models the camera motion during the exposure as a constant-velocity 2D
displacement (B_x, B_y) in pixel space. Every background patch's blur is a
1D box kernel whose length equals the projection of (B_x, B_y) onto the local
blur direction:

    b_i  ≈  B_x · cos(φ_i)  +  B_y · sin(φ_i)

This is solved as a weighted least-squares problem, with one round of
outlier rejection (patches whose residual exceeds N·σ are dropped).

Physical conversion:
    α  = |B| · pixel_pitch / focal_length          (total angular displacement)
    ω  = α / exposure_time                          (camera angular velocity)

Sanity check: the registered sharp reference is re-blurred with the fitted
kernel and compared to the observed blurry image. The residual should look like
white noise if the model is correct.

Outputs:
  outputs/trajectory.json          — fitted parameters + physical quantities
  outputs/trajectory_residual.png  — blurry | re-blurred | |residual| comparison
  outputs/trajectory_scatter.png   — measured vs predicted b per patch
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from scipy.ndimage import rotate as nd_rotate, uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Camera constants (from EXIF) ─────────────────────────────────────────────
FOCAL_MM    = 31.0
SENSOR_W_MM = 23.5
IMAGE_W_PX  = 6000
PIXEL_PITCH = SENSOR_W_MM / IMAGE_W_PX   # mm/px ≈ 3.917e-3
EXPOSURE_S  = 1 / 125


# ── Weighted least squares ────────────────────────────────────────────────────

def fit_trajectory_wls(b_arr, phi_arr, w_arr, phi_spread_threshold_deg=5.0):
    """
    Find (B_x, B_y) minimising  Σ w_i · (B_x·cos φ_i + B_y·sin φ_i − b_i)².

    When all φ_i agree within phi_spread_threshold_deg the system is rank-1:
    only the blur magnitude along the common direction is identifiable.
    In that case we fit a scalar b along the weighted-mean direction instead
    of solving the ill-conditioned 2D system, which would amplify numerical
    noise ~1/sin²(spread) into the perpendicular component.
    """
    W = w_arr / w_arr.sum()
    phi_mean = float(np.dot(W, phi_arr))
    phi_spread = float(np.sqrt(np.dot(W, (phi_arr - phi_mean) ** 2)))

    phi_r = np.radians(phi_arr)
    phi_mean_r = np.radians(phi_mean)

    if phi_spread < phi_spread_threshold_deg:
        # 1-D fit: project each b_i onto the mean blur direction
        # b_i_proj = b_i * cos(phi_i - phi_mean) ≈ b_i for small spread
        proj = b_arr * np.cos(phi_r - phi_mean_r)
        b_scalar = float(np.dot(W, proj))
        B_x = b_scalar * np.cos(phi_mean_r)
        B_y = b_scalar * np.sin(phi_mean_r)
        print(f"  [1-D fit] φ spread={phi_spread:.2f}° < {phi_spread_threshold_deg}° "
              f"→ scalar fit: b={b_scalar:.1f} px, φ={phi_mean:.2f}°")
    else:
        # 2-D WLS — system is well-conditioned when directions vary
        A = np.stack([np.cos(phi_r), np.sin(phi_r)], axis=1)
        ATWA = (A * W[:, None]).T @ A
        ATWb = (A * W[:, None]).T @ b_arr
        B = np.linalg.solve(ATWA, ATWb)
        B_x, B_y = float(B[0]), float(B[1])
        print(f"  [2-D WLS] φ spread={phi_spread:.2f}°")

    A = np.stack([np.cos(phi_r), np.sin(phi_r)], axis=1)
    B_vec = np.array([B_x, B_y])
    residuals = A @ B_vec - b_arr
    wmse = float(np.dot(W, residuals ** 2))
    return B_x, B_y, residuals, np.sqrt(wmse)


# ── Motion blur application ───────────────────────────────────────────────────

def apply_motion_blur(gray_float, b_px, phi_deg):
    """1D box blur of length b_px in direction phi_deg (degrees)."""
    rotated = nd_rotate(gray_float, -phi_deg, reshape=False, mode='reflect')
    b = max(1, round(b_px))
    blurred_rot = uniform_filter1d(rotated, size=b, axis=1, mode='reflect')
    return nd_rotate(blurred_rot, phi_deg, reshape=False, mode='reflect')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kernel_map', default='outputs/kernel_map.npz')
    p.add_argument('--sharp_reg', default='outputs/sharp_registered.png')
    p.add_argument('--blurry', default='pan_1.jpg')
    p.add_argument('--out_dir', default='outputs')
    p.add_argument('--outlier_sigma', type=float, default=2.5,
                   help='Drop patches with |residual| > this many σ after first fit')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Load kernel map ───────────────────────────────────────────────────────
    data = np.load(args.kernel_map, allow_pickle=True)
    ok = data['status'] == 'ok'

    b_arr = data['b_px_pixel'][ok].astype(float)
    phi_arr = data['phi_deg'][ok].astype(float)
    conf_arr = data['confidence'][ok].astype(float)
    cx_arr = data['cx'][ok]
    cy_arr = data['cy'][ok]

    if len(b_arr) < 3:
        raise ValueError(f"Only {len(b_arr)} valid patches — need ≥ 3 for fitting.")

    print(f"Fitting from {len(b_arr)} patches  "
          f"(b: {b_arr.mean():.1f} ± {b_arr.std():.1f} px, "
          f"φ: {phi_arr.mean():.1f} ± {phi_arr.std():.1f}°)")

    # ── First fit + outlier rejection ─────────────────────────────────────────
    B_x, B_y, res1, wrmse1 = fit_trajectory_wls(b_arr, phi_arr, conf_arr)
    sigma1 = float(np.std(res1))
    keep = np.abs(res1) < args.outlier_sigma * sigma1
    n_out = int((~keep).sum())
    print(f"Outliers: {n_out} / {len(b_arr)} (|res| > {args.outlier_sigma}σ,  σ={sigma1:.2f} px)")

    # ── Refit on inliers ──────────────────────────────────────────────────────
    if keep.sum() >= 3:
        B_x, B_y, res2, wrmse2 = fit_trajectory_wls(
            b_arr[keep], phi_arr[keep], conf_arr[keep])
        b_used, phi_used, conf_used = b_arr[keep], phi_arr[keep], conf_arr[keep]
        cx_used, cy_used = cx_arr[keep], cy_arr[keep]
        final_res, final_wrmse = res2, wrmse2
    else:
        print("[warn] Too few inliers — using all patches.")
        b_used, phi_used, conf_used = b_arr, phi_arr, conf_arr
        cx_used, cy_used = cx_arr, cy_arr
        final_res, final_wrmse = res1, wrmse1

    # ── Derived quantities ────────────────────────────────────────────────────
    b_total = float(np.sqrt(B_x ** 2 + B_y ** 2))
    phi_fit = float(np.degrees(np.arctan2(B_y, B_x)) % 180.0)
    f_px = FOCAL_MM / PIXEL_PITCH                   # focal length in pixels
    alpha_rad = b_total / f_px                      # total angular displacement
    omega_rad_s = alpha_rad / EXPOSURE_S            # angular velocity

    print(f"\n=== Trajectory Fit ===")
    print(f"  B_x = {B_x:+.2f} px,  B_y = {B_y:+.2f} px")
    print(f"  |B| = {b_total:.2f} px   (blur kernel length)")
    print(f"  φ   = {phi_fit:.2f}°   (blur direction)")
    print(f"  α   = {alpha_rad * 1e3:.3f} mrad  ({np.degrees(alpha_rad) * 60:.3f} arcmin)")
    print(f"  ω   = {np.degrees(omega_rad_s):.2f} °/s  =  {omega_rad_s:.4f} rad/s")
    print(f"  Weighted RMSE: {final_wrmse:.2f} px  (over {len(b_used)} inlier patches)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        'B_x_px': B_x, 'B_y_px': B_y,
        'b_total_px': b_total, 'phi_deg': phi_fit,
        'alpha_mrad': float(alpha_rad * 1e3),
        'omega_deg_s': float(np.degrees(omega_rad_s)),
        'omega_rad_s': float(omega_rad_s),
        'n_patches_total': int(ok.sum()),
        'n_patches_inlier': int(len(b_used)),
        'n_outliers': n_out,
        'weighted_rmse_px': float(final_wrmse),
        'exposure_s': EXPOSURE_S,
        'focal_mm': FOCAL_MM,
        'sensor_w_mm': SENSOR_W_MM,
        'pixel_pitch_mm': float(PIXEL_PITCH),
        'focal_px': float(f_px),
    }
    traj_path = out_dir / 'trajectory.json'
    with open(traj_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {traj_path}")

    # ── Sanity check: re-blur sharp_registered → compare to blurry ───────────
    def load_gray(path):
        return np.array(
            ImageOps.exif_transpose(Image.open(path)).convert('L'),
            dtype=np.float32)

    sharp_gray = load_gray(args.sharp_reg)
    blur_gray = load_gray(args.blurry)

    reblurred = apply_motion_blur(sharp_gray, b_total, phi_fit)
    residual = np.abs(reblurred - blur_gray)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(blur_gray, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Observed blurry image')
    axes[1].imshow(reblurred, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Re-blurred sharp ref  (b={b_total:.1f} px, φ={phi_fit:.1f}°)')
    im = axes[2].imshow(residual, cmap='hot', vmin=0, vmax=60)
    axes[2].set_title(
        f'|Residual|   mean={residual.mean():.1f}  std={residual.std():.1f}')
    fig.colorbar(im, ax=axes[2], fraction=0.025)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    res_path = out_dir / 'trajectory_residual.png'
    fig.savefig(res_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {res_path}")

    # ── Scatter: measured vs predicted b ─────────────────────────────────────
    phi_r = np.radians(phi_used)
    b_pred = B_x * np.cos(phi_r) + B_y * np.sin(phi_r)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: measured vs predicted
    ax = axes2[0]
    sc = ax.scatter(b_used, b_pred, c=conf_used, cmap='viridis',
                    alpha=0.75, s=50, edgecolors='none')
    lim = [min(b_used.min(), b_pred.min()) * 0.9,
           max(b_used.max(), b_pred.max()) * 1.1]
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='ideal (y=x)')
    ax.set_xlabel('Measured b (px)')
    ax.set_ylabel('Predicted b from trajectory (px)')
    ax.set_title('Global consistency: measured vs predicted blur length')
    ax.legend()
    fig2.colorbar(sc, ax=ax).set_label('Patch confidence')

    # Spatial map of residuals
    ax2 = axes2[1]
    H_img, W_img = blur_gray.shape
    sc2 = ax2.scatter(cx_used, cy_used, c=final_res, cmap='RdBu',
                      vmin=-3 * float(np.std(final_res)),
                      vmax=+3 * float(np.std(final_res)),
                      s=200, marker='s', alpha=0.8, edgecolors='none')
    ax2.set_xlim(0, W_img)
    ax2.set_ylim(H_img, 0)
    ax2.set_aspect('equal')
    ax2.set_title('Spatial residual map  (blue=under, red=over predicted)')
    fig2.colorbar(sc2, ax=ax2).set_label('Residual b (px)')

    plt.tight_layout()
    scatter_path = out_dir / 'trajectory_scatter.png'
    fig2.savefig(scatter_path, dpi=130, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {scatter_path}")


if __name__ == '__main__':
    main()
