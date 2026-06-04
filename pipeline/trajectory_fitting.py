"""
Stage 3 — Global camera trajectory fitting.

Models the camera motion during the exposure as a constant-velocity 2D
displacement (B_x, B_y) in pixel space. Every background patch's blur is a
1D box kernel whose length equals the projection of (B_x, B_y) onto the local
blur direction:

    b_i  ≈  B_x · cos(φ_i)  +  B_y · sin(φ_i)

This is solved with weighted RANSAC followed by weighted least squares on
the consensus inlier set, so bad patch-level kernel estimates do not dominate
the global trajectory.

Physical conversion:
    α  = |B| · pixel_pitch / focal_length          (total angular displacement)
    ω  = α / exposure_time                          (camera angular velocity)

Sanity check: the registered sharp reference is re-blurred with the fitted
kernel and compared to the observed blurry image. The residual should look like
white noise if the model is correct.

Outputs:
  outputs/<blurry_stem>/trajectory.json          — fitted parameters + physical quantities
  outputs/<blurry_stem>/trajectory_residual.png  — blurry | re-blurred | |residual| comparison
  outputs/<blurry_stem>/trajectory_scatter.png   — measured vs predicted b per patch
  outputs/<blurry_stem>/kernel_contribution_map.png — per-patch contribution to final b
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
from matplotlib import colors
from matplotlib.patches import Rectangle

try:
    from .exif_utils import (
        DEFAULT_EXPOSURE_S,
        DEFAULT_FOCAL_MM,
        DEFAULT_SENSOR_W_MM,
        compare_exif_metadata,
        read_image_exif_metadata,
        resolve_camera_calibration,
    )
except ImportError:  # pragma: no cover - supports `python pipeline/trajectory_fitting.py`
    from exif_utils import (
        DEFAULT_EXPOSURE_S,
        DEFAULT_FOCAL_MM,
        DEFAULT_SENSOR_W_MM,
        compare_exif_metadata,
        read_image_exif_metadata,
        resolve_camera_calibration,
    )

# Fallback calibration used only when the panning image has no usable EXIF.
FALLBACK_FOCAL_MM = DEFAULT_FOCAL_MM
FALLBACK_SENSOR_W_MM = DEFAULT_SENSOR_W_MM
FALLBACK_EXPOSURE_S = DEFAULT_EXPOSURE_S


# ── Weighted robust fitting ───────────────────────────────────────────────────

def normalize_weights(w_arr):
    """Return non-negative weights normalized to sum to 1."""
    w = np.asarray(w_arr, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    if w.size == 0:
        return w
    total = float(w.sum())
    if total <= 0:
        return np.full_like(w, 1.0 / len(w), dtype=float)
    return w / total


def axial_angle_mean_and_spread(phi_arr, w_arr):
    """Weighted mean/spread for 180-degree axial angles."""
    W = normalize_weights(w_arr)
    theta = 2.0 * np.radians(phi_arr)
    mean = 0.5 * np.degrees(np.arctan2(np.dot(W, np.sin(theta)), np.dot(W, np.cos(theta))))
    phi_mean = float(mean % 180.0)
    diff = ((phi_arr - phi_mean + 90.0) % 180.0) - 90.0
    spread = float(np.sqrt(np.dot(W, diff ** 2)))
    return phi_mean, spread


def trajectory_residuals(B_x, B_y, b_arr, phi_arr):
    phi_r = np.radians(phi_arr)
    A = np.stack([np.cos(phi_r), np.sin(phi_r)], axis=1)
    return A @ np.array([B_x, B_y]) - b_arr


def weighted_rmse(residuals, w_arr):
    W = normalize_weights(w_arr)
    if residuals.size == 0:
        return float('nan')
    return float(np.sqrt(np.dot(W, residuals ** 2)))


def fit_trajectory_wls(b_arr, phi_arr, w_arr, phi_spread_threshold_deg=5.0, verbose=True):
    """
    Find (B_x, B_y) minimising  Σ w_i · (B_x·cos φ_i + B_y·sin φ_i − b_i)².

    When all φ_i agree within phi_spread_threshold_deg the system is rank-1:
    only the blur magnitude along the common direction is identifiable.
    In that case we fit a scalar b along the weighted-mean direction instead
    of solving the ill-conditioned 2D system, which would amplify numerical
    noise ~1/sin²(spread) into the perpendicular component.
    """
    W = normalize_weights(w_arr)
    phi_mean, phi_spread = axial_angle_mean_and_spread(phi_arr, W)

    phi_r = np.radians(phi_arr)
    phi_mean_r = np.radians(phi_mean)

    if phi_spread < phi_spread_threshold_deg:
        # 1-D fit: project each b_i onto the mean blur direction
        # b_i_proj = b_i * cos(phi_i - phi_mean) ≈ b_i for small spread
        proj = b_arr * np.cos(phi_r - phi_mean_r)
        b_scalar = float(np.dot(W, proj))
        B_x = b_scalar * np.cos(phi_mean_r)
        B_y = b_scalar * np.sin(phi_mean_r)
        if verbose:
            print(f"  [1-D fit] φ spread={phi_spread:.2f}° < {phi_spread_threshold_deg}° "
                  f"→ scalar fit: b={b_scalar:.1f} px, φ={phi_mean:.2f}°")
    else:
        # 2-D WLS — system is well-conditioned when directions vary.
        A = np.stack([np.cos(phi_r), np.sin(phi_r)], axis=1)
        sqrt_w = np.sqrt(W)
        Aw = A * sqrt_w[:, None]
        bw = b_arr * sqrt_w
        B, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        B_x, B_y = float(B[0]), float(B[1])
        if verbose:
            print(f"  [2-D WLS] φ spread={phi_spread:.2f}°")

    residuals = trajectory_residuals(B_x, B_y, b_arr, phi_arr)
    return B_x, B_y, residuals, weighted_rmse(residuals, w_arr)


def ransac_trajectory(
    b_arr,
    phi_arr,
    w_arr,
    residual_thres_px,
    n_iter=256,
    min_samples=3,
    seed=0,
):
    """Weighted RANSAC for the global blur trajectory."""
    n = len(b_arr)
    sample_size = max(1, min(int(min_samples), n))
    rng = np.random.default_rng(seed)
    W = normalize_weights(w_arr)
    weighted_sampling = int(np.count_nonzero(W > 0)) >= sample_size

    best_key = None
    best_mask = None

    for _ in range(max(1, int(n_iter))):
        try:
            sample_idx = rng.choice(
                n,
                size=sample_size,
                replace=False,
                p=W if weighted_sampling else None,
            )
            B_x, B_y, _, _ = fit_trajectory_wls(
                b_arr[sample_idx], phi_arr[sample_idx], w_arr[sample_idx], verbose=False)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

        residuals = trajectory_residuals(B_x, B_y, b_arr, phi_arr)
        mask = np.abs(residuals) <= residual_thres_px
        if int(mask.sum()) < sample_size:
            continue

        inlier_weight = float(W[mask].sum())
        inlier_rmse = weighted_rmse(residuals[mask], w_arr[mask])
        key = (inlier_weight, int(mask.sum()), -inlier_rmse)
        if best_key is None or key > best_key:
            best_key = key
            best_mask = mask

    if best_mask is None:
        print('[warn] RANSAC found no consensus set — falling back to all patches.')
        best_mask = np.ones(n, dtype=bool)

    # Refit and polish the mask once under the final model.
    min_inliers = max(1, min(sample_size, n))
    for _ in range(2):
        B_x, B_y, _, _ = fit_trajectory_wls(
            b_arr[best_mask], phi_arr[best_mask], w_arr[best_mask], verbose=False)
        residuals = trajectory_residuals(B_x, B_y, b_arr, phi_arr)
        polished = np.abs(residuals) <= residual_thres_px
        if int(polished.sum()) < min_inliers or np.array_equal(polished, best_mask):
            break
        best_mask = polished

    B_x, B_y, _, _ = fit_trajectory_wls(
        b_arr[best_mask], phi_arr[best_mask], w_arr[best_mask], verbose=True)
    all_residuals = trajectory_residuals(B_x, B_y, b_arr, phi_arr)
    final_wrmse = weighted_rmse(all_residuals[best_mask], w_arr[best_mask])
    diagnostics = {
        'ransac_iterations': int(n_iter),
        'ransac_min_samples': int(sample_size),
        'ransac_residual_thres_px': float(residual_thres_px),
        'ransac_inlier_weight': float(W[best_mask].sum()),
    }
    return B_x, B_y, all_residuals, final_wrmse, best_mask, diagnostics


# ── Motion blur application ───────────────────────────────────────────────────

def apply_motion_blur(gray_float, b_px, phi_deg):
    """1D box blur of length b_px in direction phi_deg (degrees)."""
    rotated = nd_rotate(gray_float, -phi_deg, reshape=False, mode='reflect')
    b = max(1, round(b_px))
    blurred_rot = uniform_filter1d(rotated, size=b, axis=1, mode='reflect')
    return nd_rotate(blurred_rot, phi_deg, reshape=False, mode='reflect')



def save_kernel_contribution_map(
    blur_rgb,
    kernel_data,
    ok_mask,
    b_arr,
    phi_arr,
    weights,
    keep,
    residuals,
    B_x,
    B_y,
    b_total,
    phi_fit,
    final_wrmse,
    weight_field,
    out_path,
    manual_blur_px=None,
):
    """
    Draw how each patch contributed to the final fitted blur length.

    Contribution is the normalized inlier weight times the measured patch blur
    projected onto the final fitted blur direction. Rejected patches are shown
    but contribute zero to the final length.
    """
    del B_x, B_y  # The scalar contribution view is along the fitted |B| axis.
    H, W = blur_rgb.shape[:2]
    n_all = len(kernel_data['status'])
    ok_idx = np.flatnonzero(ok_mask)

    x0_all = kernel_data['x0'].astype(float)
    y0_all = kernel_data['y0'].astype(float)
    cx_all = kernel_data['cx'].astype(float)
    cy_all = kernel_data['cy'].astype(float)
    if 'patch_size' in kernel_data.files:
        patch_sizes = kernel_data['patch_size'].astype(float)
    else:
        half_sizes = np.concatenate([cx_all - x0_all, cy_all - y0_all])
        half_sizes = half_sizes[np.isfinite(half_sizes) & (half_sizes > 0)]
        fallback_size = float(np.median(half_sizes) * 2.0) if half_sizes.size else 400.0
        patch_sizes = np.full(n_all, fallback_size, dtype=float)

    inlier_w = np.zeros_like(b_arr, dtype=float)
    if np.any(keep):
        inlier_w[keep] = normalize_weights(weights[keep])

    projected = b_arr * np.cos(np.radians(phi_arr - phi_fit))
    contribution = np.zeros_like(b_arr, dtype=float)
    contribution[keep] = inlier_w[keep] * projected[keep]

    fig_w = 16
    fig_h = max(7.0, fig_w * H / max(W, 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(blur_rgb)
    ax.add_patch(Rectangle((0, 0), W, H, facecolor='black', edgecolor='none', alpha=0.18))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    ax.set_title(
        f'Patch contributions to final blur kernel | fit={b_total:.1f}px',
        fontsize=15,
        pad=12,
    )

    inlier_contrib = contribution[keep] if np.any(keep) else np.array([0.0])
    vmax = max(float(np.nanmax(inlier_contrib)) if inlier_contrib.size else 0.0, 1.0)
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap('turbo')

    # Faint skipped patch rectangles keep this aligned with the original kernel_map.png grid.
    for i in range(n_all):
        if ok_mask[i]:
            continue
        ax.add_patch(Rectangle(
            (x0_all[i], y0_all[i]), patch_sizes[i], patch_sizes[i],
            fill=False, edgecolor='white', linewidth=0.45, alpha=0.18,
        ))

    for local_i, all_i in enumerate(ok_idx):
        x0 = x0_all[all_i]
        y0 = y0_all[all_i]
        P = patch_sizes[all_i]
        cx = cx_all[all_i]
        cy = cy_all[all_i]
        phi_r = np.radians(phi_arr[local_i])
        dx = np.cos(phi_r) * P * 0.24
        dy = np.sin(phi_r) * P * 0.24

        if keep[local_i]:
            color = cmap(norm(contribution[local_i]))
            ax.add_patch(Rectangle(
                (x0, y0), P, P,
                facecolor=color, edgecolor='white',
                linewidth=1.4, alpha=0.34,
            ))
            ax.annotate(
                '',
                xy=(cx + dx, cy + dy),
                xytext=(cx - dx, cy - dy),
                arrowprops=dict(
                    arrowstyle='->',
                    color='white',
                    linewidth=2.1,
                    shrinkA=0,
                    shrinkB=0,
                ),
            )
            label = (
                f"b {b_arr[local_i]:.0f}px\n"
                f"w {inlier_w[local_i] * 100:.1f}%\n"
                f"+{contribution[local_i]:.1f}px"
            )
            ax.text(
                cx,
                cy,
                label,
                ha='center',
                va='center',
                color='white',
                fontsize=8,
                linespacing=1.0,
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.62, pad=2.4),
            )
        else:
            ax.add_patch(Rectangle(
                (x0, y0), P, P,
                fill=False, edgecolor='#ff4d4d',
                linewidth=1.6, alpha=0.85,
            ))
            ax.plot(
                [x0 + P * 0.18, x0 + P * 0.82],
                [y0 + P * 0.18, y0 + P * 0.82],
                color='#ff4d4d',
                linewidth=1.5,
                alpha=0.85,
            )
            ax.plot(
                [x0 + P * 0.82, x0 + P * 0.18],
                [y0 + P * 0.18, y0 + P * 0.82],
                color='#ff4d4d',
                linewidth=1.5,
                alpha=0.85,
            )
            ax.text(
                cx,
                cy,
                f"rejected\nb {b_arr[local_i]:.0f}px\nres {residuals[local_i]:+.0f}px",
                ha='center',
                va='center',
                color='white',
                fontsize=7,
                linespacing=1.0,
                bbox=dict(facecolor='#240000', edgecolor='none', alpha=0.68, pad=2.2),
            )

    summary = [
        f'global blur: {b_total:.1f} px',
        f'direction: {phi_fit:.1f} deg',
        f'inliers: {int(keep.sum())}/{len(b_arr)}',
        f'weighted RMSE: {final_wrmse:.1f} px',
        f'weight field: {weight_field}',
    ]
    if manual_blur_px is not None:
        summary.append(f'manual reference: {manual_blur_px:.1f} px')
        summary.append(f'fit - manual: {b_total - manual_blur_px:+.1f} px')
    ax.text(
        W * 0.012,
        H * 0.035,
        '\n'.join(summary),
        ha='left',
        va='top',
        color='white',
        fontsize=11,
        linespacing=1.2,
        bbox=dict(facecolor='black', edgecolor='white', linewidth=0.5, alpha=0.68, pad=6.0),
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label('Weighted projected contribution to final b (px)')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kernel_map', default=None,
                   help='Kernel map path; defaults to <out_dir>/kernel_map.npz')
    p.add_argument('--sharp_reg', default=None,
                   help='Registered sharp image; defaults to <out_dir>/sharp_registered.png')
    p.add_argument('--skip_residual_image', action='store_true',
                   help='Skip re-blurred-sharp residual image output for blind runs.')
    p.add_argument('--blurry', default='pan_1.jpg')
    p.add_argument('--sharp_image', default=None,
                   help='Original sharp reference image, used for EXIF comparison metadata only.')
    p.add_argument('--output_root', default='outputs',
                   help='Parent directory for per-image output subdirectories')
    p.add_argument('--out_dir', default=None,
                   help='Explicit output directory; defaults to <output_root>/<blurry_stem>')
    p.add_argument('--outlier_sigma', type=float, default=2.5,
                   help='Fallback WLS sigma-clipping cutoff when --disable_ransac is used')
    p.add_argument('--disable_ransac', action='store_true',
                   help='Disable RANSAC and use one-pass sigma-clipped weighted least squares')
    p.add_argument('--ransac_iters', type=int, default=256,
                   help='Number of weighted RANSAC hypotheses')
    p.add_argument('--ransac_min_samples', type=int, default=3,
                   help='Number of patches per RANSAC hypothesis')
    p.add_argument('--ransac_residual_thres', type=float, default=15.0,
                   help='Patch is a RANSAC inlier if |predicted_b - measured_b| is below this many px')
    p.add_argument('--ransac_seed', type=int, default=0,
                   help='Random seed for reproducible RANSAC sampling')
    p.add_argument('--weight_field', default='confidence',
                   help='Kernel-map field to use as WLS/RANSAC weights, e.g. confidence or grad_mag_var')
    p.add_argument('--manual_blur_px', type=float, default=None,
                   help='Optional manual blur length reference annotated on kernel_contribution_map.png')
    p.add_argument('--focal_px', type=float, default=None,
                   help='Override panning-image focal length in pixels for angular velocity conversion.')
    p.add_argument('--focal_mm', type=float, default=None,
                   help='Override panning-image focal length in millimeters.')
    p.add_argument('--sensor_width_mm', type=float, default=None,
                   help='Sensor width used with --focal_mm or EXIF focal length when focal-plane metadata is missing.')
    p.add_argument('--exposure_s', type=float, default=None,
                   help='Override panning-image exposure time in seconds.')
    args = p.parse_args()

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.kernel_map is not None:
        out_dir = Path(args.kernel_map).parent
    else:
        out_dir = Path(args.output_root) / Path(args.blurry).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.kernel_map is None:
        args.kernel_map = str(out_dir / 'kernel_map.npz')
    if args.sharp_reg is None and not args.skip_residual_image:
        args.sharp_reg = str(out_dir / 'sharp_registered.png')

    calibration = resolve_camera_calibration(
        args.blurry,
        focal_px=args.focal_px,
        focal_mm=args.focal_mm,
        sensor_width_mm=args.sensor_width_mm,
        exposure_s=args.exposure_s,
        fallback_focal_mm=FALLBACK_FOCAL_MM,
        fallback_sensor_width_mm=FALLBACK_SENSOR_W_MM,
        fallback_exposure_s=FALLBACK_EXPOSURE_S,
    )
    sharp_exif = read_image_exif_metadata(args.sharp_image) if args.sharp_image else None
    exif_comparison = compare_exif_metadata(calibration['exif'], sharp_exif)
    if exif_comparison and exif_comparison.get('warnings'):
        for warning in exif_comparison['warnings']:
            print(f"[warn] EXIF comparison: {warning}")

    # ── Load kernel map ───────────────────────────────────────────────────────
    data = np.load(args.kernel_map, allow_pickle=True)
    ok = data['status'] == 'ok'

    b_arr = data['b_px_pixel'][ok].astype(float)
    phi_arr = data['phi_deg'][ok].astype(float)
    weight_field = args.weight_field if args.weight_field in data.files else 'confidence'
    if weight_field != args.weight_field:
        print(f"[warn] weight field '{args.weight_field}' not found; using 'confidence'")
    conf_arr = np.nan_to_num(data[weight_field][ok].astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    conf_arr = np.clip(conf_arr, 0.0, None)
    if float(conf_arr.sum()) <= 0:
        print(f"[warn] weight field '{weight_field}' is all zero; using uniform weights")
        conf_arr = np.ones_like(b_arr, dtype=float)
    cx_arr = data['cx'][ok]
    cy_arr = data['cy'][ok]

    if len(b_arr) < 3:
        raise ValueError(f"Only {len(b_arr)} valid patches — need ≥ 3 for fitting.")

    print(f"Fitting from {len(b_arr)} patches  "
          f"(b: {b_arr.mean():.1f} ± {b_arr.std():.1f} px, "
          f"φ: {phi_arr.mean():.1f} ± {phi_arr.std():.1f}°)")

    print(f"Weight field: {weight_field}")

    # ── Robust fit + outlier rejection ────────────────────────────────────────
    ransac_info = {}
    if args.disable_ransac:
        print("RANSAC disabled; using sigma-clipped weighted least squares.")
        B_x, B_y, res1, wrmse1 = fit_trajectory_wls(b_arr, phi_arr, conf_arr)
        sigma1 = max(float(np.std(res1)), 1e-6)
        keep = np.abs(res1) < args.outlier_sigma * sigma1
        if int(keep.sum()) < 3:
            print("[warn] Too few sigma-clipped inliers — using all patches.")
            keep = np.ones_like(keep, dtype=bool)
        B_x, B_y, _, _ = fit_trajectory_wls(
            b_arr[keep], phi_arr[keep], conf_arr[keep])
        all_res = trajectory_residuals(B_x, B_y, b_arr, phi_arr)
        final_wrmse = weighted_rmse(all_res[keep], conf_arr[keep])
        fit_method = 'sigma_clipped_wls'
        print(f"Outliers: {int((~keep).sum())} / {len(b_arr)} "
              f"(|res| > {args.outlier_sigma}σ, σ={sigma1:.2f} px)")
    else:
        B_x, B_y, all_res, final_wrmse, keep, ransac_info = ransac_trajectory(
            b_arr,
            phi_arr,
            conf_arr,
            residual_thres_px=args.ransac_residual_thres,
            n_iter=args.ransac_iters,
            min_samples=args.ransac_min_samples,
            seed=args.ransac_seed,
        )
        fit_method = 'weighted_ransac_wls'
        print(f"RANSAC inliers: {int(keep.sum())} / {len(b_arr)} "
              f"(weight={ransac_info.get('ransac_inlier_weight', float('nan')):.3f}, "
              f"threshold={args.ransac_residual_thres:.1f} px)")

    n_out = int((~keep).sum())
    b_used, phi_used, conf_used = b_arr[keep], phi_arr[keep], conf_arr[keep]
    cx_used, cy_used = cx_arr[keep], cy_arr[keep]
    final_res = all_res[keep]

    # ── Derived quantities ────────────────────────────────────────────────────
    b_total = float(np.sqrt(B_x ** 2 + B_y ** 2))
    phi_fit = float(np.degrees(np.arctan2(B_y, B_x)) % 180.0)
    f_px = float(calibration['focal_px'])             # panning-image focal length in pixels
    exposure_s = float(calibration['exposure_s'])
    alpha_rad = b_total / f_px                      # total angular displacement
    omega_rad_s = alpha_rad / exposure_s            # angular velocity

    print(f"\n=== Trajectory Fit ===")
    print(f"  B_x = {B_x:+.2f} px,  B_y = {B_y:+.2f} px")
    print(f"  |B| = {b_total:.2f} px   (blur kernel length)")
    print(f"  φ   = {phi_fit:.2f}°   (blur direction)")
    print(f"  α   = {alpha_rad * 1e3:.3f} mrad  ({np.degrees(alpha_rad) * 60:.3f} arcmin)")
    print(f"  ω   = {np.degrees(omega_rad_s):.2f} °/s  =  {omega_rad_s:.4f} rad/s")
    print(
        f"  camera = f_px {f_px:.1f} ({calibration['focal_source']}), "
        f"exposure {exposure_s:.6f}s ({calibration['exposure_source']})"
    )
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
        'fit_method': fit_method,
        'weight_field': weight_field,
        **ransac_info,
        'exposure_s': exposure_s,
        'exposure_source': calibration['exposure_source'],
        'focal_mm': calibration.get('focal_mm'),
        'sensor_w_mm': calibration.get('sensor_width_mm'),
        'pixel_pitch_mm': calibration.get('pixel_pitch_mm'),
        'focal_px': float(f_px),
        'focal_source': calibration['focal_source'],
        'camera_metadata': {
            'blurry': calibration['exif'],
            'sharp_reference': sharp_exif,
            'comparison': exif_comparison,
        },
    }
    if args.manual_blur_px is not None:
        result['manual_blur_px'] = float(args.manual_blur_px)
        result['manual_vs_fitted_delta_px'] = float(b_total - args.manual_blur_px)

    traj_path = out_dir / 'trajectory.json'
    with open(traj_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {traj_path}")

    # ── Optional sanity check: re-blur sharp_registered → compare to blurry ───
    def load_gray(path):
        return np.array(
            ImageOps.exif_transpose(Image.open(path)).convert('L'),
            dtype=np.float32)

    blur_gray = load_gray(args.blurry)
    blur_rgb = np.array(ImageOps.exif_transpose(Image.open(args.blurry)).convert('RGB'))

    contrib_path = out_dir / 'kernel_contribution_map.png'
    save_kernel_contribution_map(
        blur_rgb, data, ok, b_arr, phi_arr, conf_arr, keep, all_res, B_x, B_y,
        b_total, phi_fit, final_wrmse, weight_field, contrib_path,
        manual_blur_px=args.manual_blur_px,
    )
    print(f"Saved: {contrib_path}")

    if args.skip_residual_image:
        print('Skipping trajectory_residual.png because --skip_residual_image was supplied.')
    elif args.sharp_reg is None or not Path(args.sharp_reg).exists():
        print(f"[warn] {args.sharp_reg} not found; skipping trajectory_residual.png")
    else:
        sharp_gray = load_gray(args.sharp_reg)
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
    b_pred_all = B_x * np.cos(np.radians(phi_arr)) + B_y * np.sin(np.radians(phi_arr))

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: measured vs predicted
    ax = axes2[0]
    sc = ax.scatter(b_used, b_pred, c=conf_used, cmap='viridis',
                    alpha=0.75, s=50, edgecolors='none', label='inlier')
    if n_out:
        ax.scatter(b_arr[~keep], b_pred_all[~keep], color='0.25', marker='x',
                   s=60, linewidths=1.5, label='rejected')
    lim = [min(b_arr.min(), b_pred_all.min()) * 0.9,
           max(b_arr.max(), b_pred_all.max()) * 1.1]
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='ideal (y=x)')
    ax.set_xlabel('Measured b (px)')
    ax.set_ylabel('Predicted b from trajectory (px)')
    ax.set_title('Global consistency: measured vs predicted blur length')
    ax.legend()
    fig2.colorbar(sc, ax=ax).set_label(f'Patch weight ({weight_field})')

    # Spatial map of residuals
    ax2 = axes2[1]
    H_img, W_img = blur_gray.shape
    res_scale = max(1.0, 3 * float(np.std(final_res)))
    sc2 = ax2.scatter(cx_used, cy_used, c=final_res, cmap='RdBu',
                      vmin=-res_scale, vmax=+res_scale,
                      s=200, marker='s', alpha=0.8, edgecolors='none')
    if n_out:
        ax2.scatter(cx_arr[~keep], cy_arr[~keep], color='black', marker='x',
                    s=80, linewidths=1.5, label='rejected')
        ax2.legend(loc='lower right')
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
