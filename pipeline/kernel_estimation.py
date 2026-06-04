"""
Stage 2 — Dense reference-based blur kernel estimation.

For each background patch (non-car, non-flat):
  1. Estimate blur direction from Sobel histogram of the *blurry* patch.
  2. Rotate both blurry and registered-sharp patches so blur is horizontal.
  3. Fit sinc²(b·fx) to the ratio  P_blurry(fx) / P_sharp(fx)  → kernel length b.
     This avoids the isotropy assumption of the blind approach: we just divide the
     blurry marginal spectrum by the sharp marginal spectrum directly.
  4. Confidence weight = sharp-patch texture score: Sobel energy plus
     gradient-magnitude variance (high texture → well-conditioned estimate).

Outputs:
  outputs/<blurry_stem>/kernel_map.npz  — per-patch arrays: b_px, phi_deg, confidence, texture metrics, status
  outputs/<blurry_stem>/kernel_map.csv  — same as CSV for inspection
  outputs/<blurry_stem>/kernel_map.png  — overlay visualization
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from scipy.ndimage import sobel, rotate as nd_rotate, uniform_filter1d, uniform_filter, median_filter, shift as nd_shift
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle


# ── Blur direction (from blurry patch Sobel) ─────────────────────────────────

def estimate_blur_direction(gray, n_bins=360, downsample=2, weight_mask=None):
    """
    Returns (blur_dir_deg, peak_score) or (None, 0) if no gradient.
    weight_mask: optional bool/float array same shape as gray; False/0 pixels
                 are excluded from the gradient histogram (e.g. car region).
    """
    h = gray[::downsample, ::downsample]
    gx = sobel(h, axis=1)
    gy = sobel(h, axis=0)
    mag = np.hypot(gx, gy)
    if weight_mask is not None:
        mag = mag * weight_mask[::downsample, ::downsample].astype(float)
    if mag.sum() < 1e-6:
        return None, 0.0
    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    hist, edges = np.histogram(angle, bins=n_bins, range=(0, 180), weights=mag)
    centers = (edges[:-1] + edges[1:]) / 2.0
    peaks, props = find_peaks(hist, height=hist.max() * 0.3)
    peak_idx = (int(peaks[np.argmax(props['peak_heights'])]) if len(peaks)
                else int(np.argmax(hist)))
    gradient_peak = float(centers[peak_idx])
    blur_dir = (gradient_peak + 90.0) % 180.0
    peak_score = float(hist[peak_idx]) / (float(hist.mean()) + 1e-12)
    return blur_dir, peak_score


# ── Reference-based sinc² fit ─────────────────────────────────────────────────

def reference_sinc2_fit(blur_patch, sharp_patch, blur_dir_deg,
                        f_min=0.005, f_max=0.20):
    """
    Estimate blur kernel length b (px) from the ratio of marginal power spectra:
      P_blurry(fx) / P_sharp(fx) ≈ sinc²(b · fx)
    after rotating so the blur direction is horizontal.

    Fit is done in the log domain with low-frequency weighting and
    median pre-smoothing to suppress spectral noise.

    Returns b_px (float).
    """
    rot_blur = nd_rotate(blur_patch, -blur_dir_deg, reshape=False, mode='reflect')
    rot_sharp = nd_rotate(sharp_patch, -blur_dir_deg, reshape=False, mode='reflect')

    N = min(rot_blur.shape)
    N -= N % 2
    cy, cx = np.array(rot_blur.shape) // 2
    sl = slice(cy - N // 2, cy + N // 2), slice(cx - N // 2, cx + N // 2)

    win2d = np.outer(np.hanning(N), np.hanning(N))
    P_blur = np.abs(np.fft.fft2(rot_blur[sl] * win2d)) ** 2
    P_sharp = np.abs(np.fft.fft2(rot_sharp[sl] * win2d)) ** 2

    freqs = np.fft.fftfreq(N)
    pos = freqs > 0
    f_pos = freqs[pos]

    # Marginal along x: sum over all fy, keep positive fx
    P_blur_x = P_blur[:, pos].sum(axis=0)
    P_sharp_x = P_sharp[:, pos].sum(axis=0)

    mask = (f_pos >= f_min) & (f_pos <= f_max)
    f_fit = f_pos[mask]
    R = P_blur_x[mask] / (P_sharp_x[mask] + 1e-30)
    R_norm = R / (R.max() + 1e-30)

    # Median-smooth to suppress spectral noise before fitting
    R_smooth = median_filter(R_norm, size=5)

    valleys, _ = find_peaks(-R_smooth)
    b_init = float(1.0 / f_fit[valleys[0]]) if len(valleys) else 30.0

    # Log-domain fit: treats multiplicative spectral noise as additive
    def log_sinc2(f, b, c):
        return np.log(np.sinc(b * f) ** 2 + 1e-6) + c

    # Weight low-frequency bins more heavily (higher SNR)
    weights = 1.0 / (f_fit + 0.01)

    popt, _ = curve_fit(
        log_sinc2, f_fit, np.log(R_smooth + 1e-6),
        p0=[b_init, 0.0],
        bounds=([1.0, -10.0], [float(N), 10.0]),
        sigma=1.0 / weights,
        maxfev=10000)
    return float(popt[0])


def apply_motion_blur(patch, b_px, phi_deg):
    """1D box blur of length b_px in direction phi_deg (degrees)."""
    rotated = nd_rotate(patch, -phi_deg, reshape=False, mode='reflect')
    b = max(1, round(b_px))
    blurred = uniform_filter1d(rotated, size=b, axis=1, mode='reflect')
    return nd_rotate(blurred, phi_deg, reshape=False, mode='reflect')


def align_perpendicular(blur_p, sharp_p, phi_deg, max_shift_px=8):
    """
    Correct residual translational misalignment between patch pair using
    phase cross-correlation, but only in the direction perpendicular to blur.

    Aligning along the blur axis would confound with the kernel length, so we
    rotate both patches, apply a Y-only shift, then rotate back. This corrects
    depth-parallax and homography residuals without disturbing the blur estimate.

    Returns shifted sharp_p (same shape).
    """
    rot_blur = nd_rotate(blur_p, -phi_deg, reshape=False, mode='reflect')
    rot_sharp = nd_rotate(sharp_p, -phi_deg, reshape=False, mode='reflect')

    # Phase cross-correlation in rotated frame
    F_blur = np.fft.fft2(rot_blur)
    F_sharp = np.fft.fft2(rot_sharp)
    R = F_blur * np.conj(F_sharp)
    R /= np.abs(R) + 1e-30
    cc = np.real(np.fft.ifft2(R))

    peak = np.unravel_index(np.argmax(cc), cc.shape)
    dy, dx = int(peak[0]), int(peak[1])
    H, W = blur_p.shape
    if dy > H // 2: dy -= H
    if dx > W // 2: dx -= W

    # Clamp and zero out the along-blur (X) component — only correct perpendicular
    dy = int(np.clip(dy, -max_shift_px, max_shift_px))

    rot_sharp_aligned = nd_shift(rot_sharp, [dy, 0], mode='reflect')
    return nd_rotate(rot_sharp_aligned, phi_deg, reshape=False, mode='reflect')


# ── Main ──────────────────────────────────────────────────────────────────────

def load_gray_rgb(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    rgb = np.array(img, dtype=np.uint8)
    gray = np.array(img.convert('L'), dtype=np.float32)
    return rgb, gray


def patch_structure_metrics(gray_patch, harris_block_size=5, harris_k=0.04):
    """Return texture metrics from the registered sharp patch.

    Gradient energy measures how much signal is available; gradient-magnitude
    variance favors high-contrast, non-uniform structure over flat or repetitive
    regions. The Harris score is optional at selection time, but always saved so
    weak-corner patches can be inspected after a run.
    """
    gx = sobel(gray_patch, axis=1)
    gy = sobel(gray_patch, axis=0)
    grad_sq = gx ** 2 + gy ** 2
    grad_mag = np.sqrt(grad_sq)

    block = max(1, int(harris_block_size))
    ix2 = uniform_filter(gx * gx, size=block, mode='reflect')
    iy2 = uniform_filter(gy * gy, size=block, mode='reflect')
    ixy = uniform_filter(gx * gy, size=block, mode='reflect')
    det = ix2 * iy2 - ixy ** 2
    trace = ix2 + iy2
    harris = det - harris_k * trace ** 2
    harris_pos = harris[harris > 0]

    grad_energy = float(np.mean(grad_sq))
    grad_mag_var = float(np.var(grad_mag))
    texture_score = grad_energy + grad_mag_var
    return {
        'grad_energy': grad_energy,
        'grad_mag_var': grad_mag_var,
        'grad_p95': float(np.percentile(grad_mag, 95)),
        'harris_max': float(harris_pos.max()) if harris_pos.size else 0.0,
        'harris_mean': float(harris_pos.mean()) if harris_pos.size else 0.0,
        'texture_score': float(texture_score),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--blurry', default='pan_1.jpg')
    p.add_argument('--sharp_reg', default=None,
                   help='Registered sharp image; defaults to <out_dir>/sharp_registered.png')
    p.add_argument('--car_mask', default=None,
                   help='Car mask path; defaults to <out_dir>/car_mask.png')
    p.add_argument('--patch_size', type=int, default=400)
    p.add_argument('--peak_thres', type=float, default=2.5,
                   help='Min peak/mean Sobel histogram ratio (direction confidence)')
    p.add_argument('--grad_energy_thres', type=float, default=100.0,
                   help='Min mean squared Sobel gradient in *sharp* patch (identifiability)')
    p.add_argument('--grad_var_thres', type=float, default=0.0,
                   help='Min variance of sharp-patch gradient magnitudes; 0 disables this gate')
    p.add_argument('--harris_thres', type=float, default=0.0,
                   help='Min max Harris corner response in sharp patch; 0 disables this gate')
    p.add_argument('--harris_block_size', type=int, default=5,
                   help='Window size used for Harris corner response')
    p.add_argument('--harris_k', type=float, default=0.04,
                   help='Harris corner detector k parameter')
    p.add_argument('--valid_mask', default=None,
                   help='Validity mask from register_reference.py; defaults to <out_dir>/sharp_registered_valid.png')
    p.add_argument('--car_overlap_thres', type=float, default=0.10,
                   help='Skip patch if >this fraction overlaps car mask')
    p.add_argument('--valid_thres', type=float, default=0.95,
                   help='Skip patch if <this fraction is covered by the registered sharp image')
    p.add_argument('--global_phi', type=float, default=None,
                   help='Override blur direction in degrees; skips the pre-pass direction estimate')
    p.add_argument('--output_root', default='outputs',
                   help='Parent directory for per-image output subdirectories')
    p.add_argument('--out_dir', default=None,
                   help='Explicit output directory; defaults to <output_root>/<blurry_stem>')
    args = p.parse_args()

    blurry_path = Path(args.blurry)
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.output_root) / blurry_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.sharp_reg is None:
        args.sharp_reg = str(out_dir / 'sharp_registered.png')
    if args.car_mask is None:
        args.car_mask = str(out_dir / 'car_mask.png')
    if args.valid_mask is None:
        args.valid_mask = str(out_dir / 'sharp_registered_valid.png')

    blur_rgb, blur_gray = load_gray_rgb(args.blurry)
    _, sharp_gray = load_gray_rgb(args.sharp_reg)
    car_mask = np.array(Image.open(args.car_mask).convert('L')) > 127

    valid_mask_path = Path(args.valid_mask)
    if valid_mask_path.exists():
        valid_mask = np.array(Image.open(valid_mask_path).convert('L')) > 127
    else:
        print(f"[warn] {args.valid_mask} not found — treating all pixels as valid")
        valid_mask = np.ones(blur_gray.shape, dtype=bool)

    H, W = blur_gray.shape
    P = args.patch_size
    n_rows, n_cols = H // P, W // P
    print(f"Image: {W}×{H}  patch: {P}px  grid: {n_cols}×{n_rows}")

    # ── Phase 1: estimate global blur direction from full image ───────────────
    if args.global_phi is not None:
        global_phi = args.global_phi
        print(f"Using supplied global blur direction: {global_phi:.2f}°")
    else:
        bg_mask = ~car_mask  # exclude car gradients from the histogram
        global_phi, peak_score = estimate_blur_direction(blur_gray, weight_mask=bg_mask)
        if global_phi is None:
            raise ValueError("Could not estimate blur direction from full image.")
        print(f"Global blur direction: {global_phi:.2f}°  (full-image estimate, peak_score={peak_score:.1f})")

    # ── Phase 2: kernel fitting with global phi ───────────────────────────────
    records = []
    for row in range(n_rows):
        for col in range(n_cols):
            y0, x0 = row * P, col * P
            blur_p  = blur_gray[y0:y0 + P, x0:x0 + P]
            sharp_p = sharp_gray[y0:y0 + P, x0:x0 + P]
            car_p   = car_mask[y0:y0 + P, x0:x0 + P]
            valid_p = valid_mask[y0:y0 + P, x0:x0 + P]

            rec = dict(row=row, col=col, x0=x0, y0=y0,
                       cx=x0 + P / 2.0, cy=y0 + P / 2.0, patch_size=P,
                       car_frac=float(car_p.mean()),
                       b_px=None, b_px_spec=None, b_px_pixel=None,
                       phi_deg=global_phi, confidence=None,
                       grad_energy=None, grad_mag_var=None, grad_p95=None,
                       harris_max=None, harris_mean=None, texture_score=None,
                       status='pending')

            # Skip car patches
            if rec['car_frac'] > args.car_overlap_thres:
                rec['status'] = 'skip_car'
                records.append(rec)
                continue

            # Skip patches not fully covered by the registered sharp image
            if float(valid_p.mean()) < args.valid_thres:
                rec['status'] = 'skip_invalid_region'
                records.append(rec)
                continue

            # Identifiability: structural content of the registered sharp reference.
            metrics = patch_structure_metrics(
                sharp_p,
                harris_block_size=args.harris_block_size,
                harris_k=args.harris_k,
            )
            rec.update(metrics)
            rec['confidence'] = metrics['texture_score']

            if metrics['grad_energy'] < args.grad_energy_thres:
                rec['status'] = 'skip_flat_sharp'
                records.append(rec)
                continue

            if metrics['grad_mag_var'] < args.grad_var_thres:
                rec['status'] = 'skip_low_texture'
                records.append(rec)
                continue

            if args.harris_thres > 0 and metrics['harris_max'] < args.harris_thres:
                rec['status'] = 'skip_no_corner'
                records.append(rec)
                continue

            # Kernel length: spectral fit + parallel pixel-domain fit
            try:
                b_spec = reference_sinc2_fit(blur_p, sharp_p, global_phi)
                sharp_p_aligned = align_perpendicular(blur_p, sharp_p, global_phi)

                def mse(b):
                    return float(np.mean(
                        (blur_p - apply_motion_blur(sharp_p_aligned, b, global_phi)) ** 2))

                # Spectral-seeded pixel refinement (narrow bounds around b_spec)
                res_spec = minimize_scalar(
                    mse, bounds=(max(1.0, b_spec * 0.5), b_spec * 2.0),
                    method='bounded')

                # Independent pixel-domain estimate (wide bounds, no spectral seed)
                res_pixel = minimize_scalar(
                    mse, bounds=(1.0, P * 0.45),
                    method='bounded')

                rec['b_px_spec']  = b_spec
                rec['b_px']       = res_spec.x   # spectral→pixel (trajectory fitting uses this)
                rec['b_px_pixel'] = res_pixel.x
                rec['status'] = 'ok'
            except Exception as e:
                rec['status'] = f'fit_failed:{e}'
            records.append(rec)

    valid = [r for r in records if r['status'] == 'ok']
    skipped = len(records) - len(valid)
    print(f"Patches: {len(records)} total | {len(valid)} valid | {skipped} skipped")

    if valid:
        b_arr = np.array([r['b_px'] for r in valid])
        b_spec_arr = np.array([r['b_px_spec'] for r in valid])
        b_pixel_arr = np.array([r['b_px_pixel'] for r in valid])
        phi_arr = np.array([r['phi_deg'] for r in valid])
        w_arr = np.array([r['confidence'] for r in valid])
        tex_arr = np.array([r['texture_score'] for r in valid])
        grad_var_arr = np.array([r['grad_mag_var'] for r in valid])
        harris_arr = np.array([r['harris_max'] for r in valid])
        w_arr_n = w_arr / w_arr.sum()
        print(f"b_px_pixel (used downstream): mean={b_pixel_arr.mean():.1f}  "
              f"std={b_pixel_arr.std():.1f}  "
              f"w_mean={float(np.dot(w_arr_n, b_pixel_arr)):.1f}")
        print(f"b_px spectral→pixel diagnostic: mean={b_arr.mean():.1f}  "
              f"std={b_arr.std():.1f}  "
              f"w_mean={float(np.dot(w_arr_n, b_arr)):.1f}")
        print(f"b_px_spec diagnostic: mean={b_spec_arr.mean():.1f}  "
              f"std={b_spec_arr.std():.1f}  "
              f"w_mean={float(np.dot(w_arr_n, b_spec_arr)):.1f}")
        print(f"phi_deg: mean={phi_arr.mean():.2f}  std={phi_arr.std():.2f}  "
              f"w_mean={float(np.dot(w_arr_n, phi_arr)):.2f}")
        print(f"texture_score: mean={tex_arr.mean():.1f}  "
              f"grad_var_mean={grad_var_arr.mean():.1f}  "
              f"harris_max_mean={harris_arr.mean():.1f}")

    # ── Save NPZ ──────────────────────────────────────────────────────────────
    npz_path = out_dir / 'kernel_map.npz'
    np.savez(
        npz_path,
        row=np.array([r['row'] for r in records]),
        col=np.array([r['col'] for r in records]),
        x0=np.array([r['x0'] for r in records]),
        y0=np.array([r['y0'] for r in records]),
        cx=np.array([r['cx'] for r in records]),
        cy=np.array([r['cy'] for r in records]),
        patch_size=np.array([r['patch_size'] for r in records]),
        b_px=np.array([r['b_px']       if r['b_px']       is not None else np.nan for r in records]),
        b_px_spec=np.array([r['b_px_spec'] if r['b_px_spec'] is not None else np.nan for r in records]),
        b_px_pixel=np.array([r['b_px_pixel'] if r['b_px_pixel'] is not None else np.nan for r in records]),
        phi_deg=np.array([r['phi_deg'] if r['phi_deg'] is not None else np.nan
                          for r in records]),
        confidence=np.array([r['confidence'] if r['confidence'] is not None else 0.0
                             for r in records]),
        grad_energy=np.array([r['grad_energy'] if r['grad_energy'] is not None else 0.0
                              for r in records]),
        grad_mag_var=np.array([r['grad_mag_var'] if r['grad_mag_var'] is not None else 0.0
                               for r in records]),
        grad_p95=np.array([r['grad_p95'] if r['grad_p95'] is not None else 0.0
                           for r in records]),
        harris_max=np.array([r['harris_max'] if r['harris_max'] is not None else 0.0
                             for r in records]),
        harris_mean=np.array([r['harris_mean'] if r['harris_mean'] is not None else 0.0
                              for r in records]),
        texture_score=np.array([r['texture_score'] if r['texture_score'] is not None else 0.0
                                for r in records]),
        car_frac=np.array([r['car_frac'] for r in records]),
        status=np.array([r['status'] for r in records]))
    print(f"Saved: {npz_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / 'kernel_map.csv'
    fields = ['row', 'col', 'x0', 'y0', 'cx', 'cy', 'patch_size',
              'car_frac', 'grad_energy', 'grad_mag_var', 'grad_p95',
              'harris_max', 'harris_mean', 'texture_score',
              'b_px', 'b_px_spec', 'b_px_pixel', 'phi_deg', 'confidence', 'status']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fields})
    print(f"Saved: {csv_path}")

    # ── Overlay visualization ─────────────────────────────────────────────────
    fig_w = 14
    fig_h = max(6, fig_w * H / max(W, 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(blur_rgb)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    ax.set_title(f'Reference-based kernel estimates | patch={P}px')

    b_valid = np.array([r['b_px_pixel'] for r in valid]) if valid else np.array([0.0, 1.0])
    norm = colors.Normalize(vmin=b_valid.min(), vmax=b_valid.max())
    cmap = plt.get_cmap('turbo')
    arrow_len = P * 0.34

    status_color = {
        'ok': 'white',
        'skip_car': 'red',
        'skip_invalid_region': 'magenta',
        'skip_flat_sharp': '0.4',
        'skip_low_texture': '0.55',
        'skip_no_corner': 'cyan',
        'skip_no_gradient': '0.4',
        'skip_low_peak': '0.6',
    }
    for r in records:
        ec = status_color.get(r['status'], 'orange')
        ax.add_patch(Rectangle((r['x0'], r['y0']), P, P,
                               fill=False, edgecolor=ec, linewidth=0.7, alpha=0.5))
        if r['status'] != 'ok':
            continue
        phi_r = np.radians(r['phi_deg'])
        dx = np.cos(phi_r) * arrow_len / 2
        dy = np.sin(phi_r) * arrow_len / 2
        color = cmap(norm(r['b_px_pixel']))
        ax.annotate('', xy=(r['cx'] + dx, r['cy'] + dy),
                    xytext=(r['cx'] - dx, r['cy'] - dy),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   linewidth=2.0, shrinkA=0, shrinkB=0))
        ax.text(r['cx'], r['cy'] + P * 0.22, f"{r['b_px_pixel']:.0f}px",
                color='white', fontsize=7, ha='center', va='center',
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.4, pad=1.5))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02).set_label('Blur length b (px)')
    plt.tight_layout()
    png_path = out_dir / 'kernel_map.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png_path}")

    # ── Patch diagnostic grid (8 patches × 4 columns) ────────────────────────
    if len(valid) >= 4:
        b_vals = np.array([r['b_px_pixel'] for r in valid])
        b_mean = float(b_vals.mean())
        dist = np.abs(b_vals - b_mean)
        order = np.argsort(dist)

        n_near = min(4, len(valid) // 2)
        n_out  = min(4, len(valid) - n_near)
        near_recs    = [valid[i] for i in order[:n_near]]
        outlier_recs = [valid[i] for i in order[-n_out:]]
        grid_recs    = near_recs + outlier_recs
        group_labels = ['near-mean'] * n_near + ['outlier'] * n_out
        n_rows = len(grid_recs)

        col_titles = ['Aligned sharp',
                      'Re-blur (spectral)',  'Re-blur (pixel)',
                      'Original blur',
                      '|Residual| spec',    '|Residual| pixel']
        fig, axes = plt.subplots(n_rows, 6, figsize=(6 * 2.8, n_rows * 2.8))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=8, fontweight='bold', pad=4)

        res_max = 0.0  # compute common residual scale across all patches/methods
        patch_cache = []
        for rec in grid_recs:
            y0, x0 = rec['y0'], rec['x0']
            blur_p   = blur_gray[y0:y0 + P, x0:x0 + P]
            sharp_p  = sharp_gray[y0:y0 + P, x0:x0 + P]
            phi_p    = rec['phi_deg']
            b_spec_p = rec['b_px_spec'] if rec['b_px_spec'] is not None else rec['b_px']
            b_pix_p  = rec['b_px_pixel'] if rec['b_px_pixel'] is not None else rec['b_px']
            sharp_al    = align_perpendicular(blur_p, sharp_p, phi_p)
            reblu_spec  = apply_motion_blur(sharp_al, b_spec_p, phi_p)
            reblu_pixel = apply_motion_blur(sharp_al, b_pix_p,  phi_p)
            res_spec    = np.abs(reblu_spec  - blur_p)
            res_pixel   = np.abs(reblu_pixel - blur_p)
            res_max = max(res_max, float(res_spec.max()), float(res_pixel.max()))
            patch_cache.append((blur_p, sharp_al, reblu_spec, reblu_pixel, res_spec, res_pixel))

        res_clim = max(1.0, res_max * 0.9)

        for row, (rec, label, (blur_p, sharp_al, reblu_spec, reblu_pixel, res_spec, res_pixel)) \
                in enumerate(zip(grid_recs, group_labels, patch_cache)):
            panels = [sharp_al, reblu_spec, reblu_pixel, blur_p, res_spec, res_pixel]
            for col, patch in enumerate(panels):
                ax = axes[row, col]
                if col >= 4:
                    ax.imshow(patch, cmap='hot', vmin=0, vmax=res_clim)
                else:
                    ax.imshow(patch, cmap='gray', vmin=0, vmax=255)
                ax.axis('off')

            b_spec_v  = rec['b_px_spec']  or float('nan')
            b_pixel_v = rec['b_px_pixel'] or float('nan')
            axes[row, 0].set_ylabel(
                f"{label}\nspec={b_spec_v:.0f}px  pix={b_pixel_v:.0f}px\n"
                f"φ={rec['phi_deg']:.1f}°  patch({rec['col']},{rec['row']})",
                fontsize=7, rotation=0, ha='right', va='center', labelpad=80)

        # Divider between near-mean and outlier groups
        if n_near > 0 and n_out > 0:
            y_div = n_near / n_rows
            fig.add_artist(
                plt.Line2D([0.01, 0.99], [1 - y_div, 1 - y_div],
                           transform=fig.transFigure,
                           color='steelblue', linewidth=1.5, linestyle='--'))
            fig.text(0.5, 1 - y_div + 0.005, '── outliers below ──',
                     ha='center', va='bottom', fontsize=8, color='steelblue',
                     transform=fig.transFigure)

        fig.suptitle(
            f'Patch diagnostics  |  b mean={b_mean:.1f} px  |  '
            f'top {n_near} near-mean  +  {n_out} largest outliers',
            fontsize=10, y=1.01)
        plt.tight_layout()
        grid_path = out_dir / 'kernel_patch_grid.png'
        fig.savefig(grid_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {grid_path}")

    # ── Uniform trajectory: single kernel applied to the full image ───────────
    if valid:
        b_vals   = np.array([r['b_px_pixel']  for r in valid])
        conf_arr = np.array([r['confidence']  for r in valid])
        global_b = float(np.average(b_vals, weights=conf_arr))

        print(f"\nUniform kernel: b={global_b:.2f} px  φ={global_phi:.2f}°")

        reblurred_full = apply_motion_blur(sharp_gray, global_b, global_phi)
        residual_full  = np.abs(reblurred_full - blur_gray)

        B_x = global_b * np.cos(np.radians(global_phi))
        B_y = global_b * np.sin(np.radians(global_phi))
        utraj = {
            'B_x_px': float(B_x), 'B_y_px': float(B_y),
            'b_total_px': global_b, 'phi_deg': global_phi,
            'n_patches_used': len(valid),
            'method': 'uniform_kernel_weighted_mean',
        }
        utraj_path = out_dir / 'uniform_traj.json'
        with open(utraj_path, 'w') as f:
            json.dump(utraj, f, indent=2)
        print(f"Saved: {utraj_path}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(blur_gray,      cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original blurry')
        axes[1].imshow(reblurred_full, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f'Re-blurred (uniform b={global_b:.1f} px, φ={global_phi:.1f}°)')
        im = axes[2].imshow(residual_full, cmap='hot', vmin=0, vmax=60)
        axes[2].set_title(
            f'|Residual|   mean={residual_full.mean():.1f}  std={residual_full.std():.1f}')
        fig.colorbar(im, ax=axes[2], fraction=0.025)
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        ures_path = out_dir / 'uniform_trajectory_residual.png'
        fig.savefig(ures_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {ures_path}")


if __name__ == '__main__':
    main()
