"""
Patchwise blur direction and kernel-length estimation.

For each non-overlapping square patch:
1. Estimate dominant gradient orientation.
2. Convert it to blur direction by rotating 90 degrees.
3. Skip patches whose orientation histogram peak is too flat.
4. Fit the anisotropic power-spectrum ratio to sinc^2(b*f) to estimate b.
5. Save an overlay with arrows and a CSV table of per-patch estimates.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import sobel
from scipy.signal import find_peaks

from estimate_blur import power_spectrum_ratio_fit


def safe_name_part(text):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_'
                   for ch in text).strip('_')


def safe_value_part(value):
    return safe_name_part(f'{value:g}')


def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    rgb = np.array(img, dtype=np.uint8)
    gray = np.array(img.convert('L'), dtype=np.float32)
    return rgb, gray


def estimate_patch_direction(patch, n_bins=360, downsample=2):
    """
    Estimate blur direction and orientation peak score for one gray patch.

    peak_score = max_hist_bin / mean_hist_bin.
    A flat orientation histogram is near 1; stronger directional structure is
    larger. Returns None for angle fields if the patch has no gradient energy.
    """
    h = patch[::downsample, ::downsample]
    gx = sobel(h, axis=1)
    gy = sobel(h, axis=0)
    mag = np.hypot(gx, gy)
    total = float(mag.sum())
    mean_mag = float(mag.mean())
    if total <= 1e-6:
        return None, None, 0.0, total, mean_mag

    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    hist, edges = np.histogram(angle, bins=n_bins, range=(0, 180),
                               weights=mag)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mean_height = float(hist.mean()) + 1e-12
    peaks, props = find_peaks(hist, height=hist.max() * 0.3)
    if len(peaks):
        peak_idx = int(peaks[np.argmax(props['peak_heights'])])
    else:
        peak_idx = int(np.argmax(hist))

    gradient_peak = float(centers[peak_idx])
    blur_direction = (gradient_peak + 90.0) % 180.0
    peak_score = float(hist[peak_idx] / mean_height)
    return blur_direction, gradient_peak, peak_score, total, mean_mag


def iter_patches(gray, patch_size):
    H, W = gray.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * patch_size
            x0 = col * patch_size
            patch = gray[y0:y0 + patch_size, x0:x0 + patch_size]
            yield row, col, x0, y0, patch


def estimate_patchwise(gray, patch_size, peak_thres, grad_thres):
    results = []
    for row, col, x0, y0, patch in iter_patches(gray, patch_size):
        blur_angle, gradient_peak, peak_score, grad_sum, grad_mean = (
            estimate_patch_direction(patch))
        rec = {
            'row': row,
            'col': col,
            'x0': x0,
            'y0': y0,
            'cx': x0 + patch_size / 2.0,
            'cy': y0 + patch_size / 2.0,
            'patch_size': patch_size,
            'peak_score': peak_score,
            'gradient_sum': grad_sum,
            'gradient_mean': grad_mean,
            'gradient_peak_deg': gradient_peak,
            'blur_angle_deg': blur_angle,
            'b_px': None,
            'status': 'skip_low_peak',
        }

        if blur_angle is None:
            rec['status'] = 'skip_no_gradient'
            results.append(rec)
            continue

        if grad_mean < grad_thres:
            rec['status'] = 'skip_low_gradient'
            results.append(rec)
            continue

        if peak_score < peak_thres:
            results.append(rec)
            continue

        try:
            b_px, *_ = power_spectrum_ratio_fit(
                patch, blur_angle, f_min=0.005, f_max=0.2)
            rec['b_px'] = float(b_px)
            rec['status'] = 'ok'
        except Exception as exc:
            rec['status'] = f'fit_failed:{exc}'
        results.append(rec)

    return results


def save_results_csv(results, path):
    fieldnames = [
        'row', 'col', 'x0', 'y0', 'cx', 'cy', 'patch_size',
        'peak_score', 'gradient_sum', 'gradient_mean', 'gradient_peak_deg',
        'blur_angle_deg', 'b_px', 'status',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {path}")


def save_overlay(rgb, results, patch_size, peak_thres, grad_thres, path):
    valid = [r for r in results if r['status'] == 'ok' and r['b_px'] is not None]
    b_vals = np.array([r['b_px'] for r in valid], dtype=np.float32)
    norm = colors.Normalize(vmin=float(b_vals.min()), vmax=float(b_vals.max())) if len(b_vals) else None
    cmap = plt.get_cmap('turbo')

    H, W = rgb.shape[:2]
    fig_w = 12
    fig_h = max(6, fig_w * H / max(W, 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgb)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    ax.set_title(
        f'Patchwise blur estimates | patch={patch_size}px | '
        f'peak_thres={peak_thres:g} | grad_thres={grad_thres:g}')

    arrow_len = patch_size * 0.34
    for r in results:
        x0, y0 = r['x0'], r['y0']
        rect_color = 'white' if r['status'] == 'ok' else '0.65'
        ax.add_patch(Rectangle((x0, y0), patch_size, patch_size,
                               fill=False, edgecolor=rect_color,
                               linewidth=0.7, alpha=0.55))

        if r['status'] != 'ok':
            continue

        angle = np.radians(r['blur_angle_deg'])
        dx = np.cos(angle) * arrow_len / 2.0
        dy = np.sin(angle) * arrow_len / 2.0
        color = cmap(norm(r['b_px'])) if norm is not None else 'red'
        ax.annotate(
            '',
            xy=(r['cx'] + dx, r['cy'] + dy),
            xytext=(r['cx'] - dx, r['cy'] - dy),
            arrowprops=dict(arrowstyle='->', color=color, linewidth=2.0,
                            shrinkA=0, shrinkB=0),
        )
        ax.text(r['cx'], r['cy'] + patch_size * 0.22, f"{r['b_px']:.0f}px",
                color='white', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.45,
                          pad=1.5))

    if norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Estimated blur length b (px)')

    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--patch_size', type=int, default=400)
    p.add_argument('--peak_thres', type=float, default=3.0,
                   help='Minimum peak/mean histogram ratio; flat histograms are near 1')
    p.add_argument('--grad_thres', type=float, default=8.0,
                   help='Minimum mean Sobel gradient magnitude before fitting')
    p.add_argument('--out_dir', type=str, default='outputs')
    args = p.parse_args()

    if args.patch_size < 16:
        raise ValueError('--patch_size must be at least 16 px')

    rgb, gray = load_image(args.image)
    H, W = gray.shape
    n_rows = H // args.patch_size
    n_cols = W // args.patch_size
    if n_rows == 0 or n_cols == 0:
        raise ValueError('Image is smaller than one patch')

    ignored_h = H - n_rows * args.patch_size
    ignored_w = W - n_cols * args.patch_size
    print(f"Image: {args.image} ({W}x{H}px, EXIF applied)")
    print(f"Patches: {n_cols} cols x {n_rows} rows, size={args.patch_size}px")
    if ignored_w or ignored_h:
        print(f"Ignoring right/bottom remainder: {ignored_w}px x {ignored_h}px")
    print(f"Peak threshold: {args.peak_thres:g} (peak/mean histogram ratio)")
    print(f"Gradient threshold: {args.grad_thres:g} (mean Sobel magnitude)")

    results = estimate_patchwise(gray, args.patch_size, args.peak_thres,
                                 args.grad_thres)
    valid = [r for r in results if r['status'] == 'ok']
    skipped = len(results) - len(valid)
    print(f"Valid patches: {len(valid)}/{len(results)}  skipped={skipped}")
    if valid:
        b_vals = np.array([r['b_px'] for r in valid], dtype=np.float32)
        angles = np.array([r['blur_angle_deg'] for r in valid], dtype=np.float32)
        print(f"b px: mean={b_vals.mean():.1f}, std={b_vals.std():.1f}, "
              f"min={b_vals.min():.1f}, max={b_vals.max():.1f}")
        print(f"angle deg: mean={angles.mean():.2f}, std={angles.std():.2f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name_part(Path(args.image).stem) or 'image'
    prefix = (
        f"{stem}_patchwise_ps_{args.patch_size}"
        f"_peak_{safe_value_part(args.peak_thres)}"
        f"_grad_{safe_value_part(args.grad_thres)}")
    overlay_path = out_dir / f"{prefix}_overlay.png"
    csv_path = out_dir / f"{prefix}_estimates.csv"

    save_overlay(rgb, results, args.patch_size, args.peak_thres,
                 args.grad_thres, overlay_path)
    save_results_csv(results, csv_path)


if __name__ == '__main__':
    main()
