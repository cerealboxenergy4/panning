#!/usr/bin/env python3
"""Blind blur-kernel estimation from a single panning image.

This is an experimental alternative to kernel_estimation.py. It does not use a
registered sharp reference. Instead, it assumes the latent background spectrum is
locally smooth and estimates the 1D box-blur length from sinc-like notches in the
blurred patch spectrum.

Outputs follow the same kernel_map.npz schema used by trajectory_fitting.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import median_filter, rotate as nd_rotate
from scipy.optimize import minimize_scalar

try:
    from .kernel_estimation import estimate_blur_direction, load_gray_rgb, load_optional_binary_mask, patch_structure_metrics
except ImportError:  # pragma: no cover - supports `python pipeline/blind_kernel_estimation.py`
    from kernel_estimation import estimate_blur_direction, load_gray_rgb, load_optional_binary_mask, patch_structure_metrics


def robust_zscore(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad if mad > 1e-12 else float(np.std(vals)) + 1e-12
    return (vals - med) / scale


def whiten_log_power(log_power: np.ndarray, smooth_bins: int) -> np.ndarray:
    size = max(5, int(smooth_bins) | 1)
    envelope = median_filter(log_power, size=size, mode='nearest')
    whitened = median_filter(log_power - envelope, size=3, mode='nearest')
    return robust_zscore(whitened)


def blind_sinc2_fit(
    blur_patch: np.ndarray,
    blur_dir_deg: float,
    b_min: float,
    b_max: float,
    f_min: float = 0.006,
    f_max: float = 0.22,
    smooth_bins: int = 21,
) -> tuple[float, float]:
    """Estimate box-kernel length from spectral notch correlation.

    The latent sharp spectrum is unknown, so we remove a smooth spectral envelope
    from the blurred patch's marginal power spectrum and correlate the residual
    notch pattern against a whitened log-sinc^2 model over candidate b values.
    Returns (b_px, confidence_score).
    """
    rot = nd_rotate(blur_patch, -blur_dir_deg, reshape=False, mode='reflect')
    N = min(rot.shape)
    N -= N % 2
    if N < 32:
        raise ValueError('patch too small for blind spectral fit')

    cy, cx = np.array(rot.shape) // 2
    sl = slice(cy - N // 2, cy + N // 2), slice(cx - N // 2, cx + N // 2)
    patch = rot[sl].astype(np.float32)
    patch = patch - float(np.mean(patch))
    win2d = np.outer(np.hanning(N), np.hanning(N))
    power = np.abs(np.fft.fft2(patch * win2d)) ** 2

    freqs = np.fft.fftfreq(N)
    pos = freqs > 0
    f_pos = freqs[pos]
    marginal = power[:, pos].sum(axis=0)
    mask = (f_pos >= f_min) & (f_pos <= f_max)
    f = f_pos[mask]
    if len(f) < 12:
        raise ValueError('too few frequency bins for blind spectral fit')

    observed = whiten_log_power(np.log(marginal[mask] + 1e-30), smooth_bins=smooth_bins)
    weights = 1.0 / (f + 0.015)
    weights = weights / float(weights.sum())

    def score_for_b(b: float) -> float:
        model_log = np.log(np.sinc(float(b) * f) ** 2 + 1e-5)
        model = whiten_log_power(model_log, smooth_bins=smooth_bins)
        denom = np.sqrt(np.dot(weights, observed ** 2) * np.dot(weights, model ** 2)) + 1e-12
        return float(np.dot(weights, observed * model) / denom)

    b_min = max(2.0, float(b_min))
    b_max = max(b_min + 1.0, float(b_max))
    grid = np.linspace(b_min, b_max, 220)
    scores = np.array([score_for_b(b) for b in grid])
    best = int(np.argmax(scores))
    lo = grid[max(0, best - 2)]
    hi = grid[min(len(grid) - 1, best + 2)]
    try:
        opt = minimize_scalar(lambda b: -score_for_b(b), bounds=(lo, hi), method='bounded')
        b_est = float(opt.x)
        score = float(-opt.fun)
    except Exception:
        b_est = float(grid[best])
        score = float(scores[best])
    return b_est, score


def save_kernel_npz(records: list[dict], path: Path) -> None:
    np.savez(
        path,
        row=np.array([r['row'] for r in records]),
        col=np.array([r['col'] for r in records]),
        x0=np.array([r['x0'] for r in records]),
        y0=np.array([r['y0'] for r in records]),
        cx=np.array([r['cx'] for r in records]),
        cy=np.array([r['cy'] for r in records]),
        patch_size=np.array([r['patch_size'] for r in records]),
        b_px=np.array([r['b_px'] if r['b_px'] is not None else np.nan for r in records]),
        b_px_spec=np.array([r['b_px_spec'] if r['b_px_spec'] is not None else np.nan for r in records]),
        b_px_pixel=np.array([r['b_px_pixel'] if r['b_px_pixel'] is not None else np.nan for r in records]),
        phi_deg=np.array([r['phi_deg'] if r['phi_deg'] is not None else np.nan for r in records]),
        confidence=np.array([r['confidence'] if r['confidence'] is not None else 0.0 for r in records]),
        blind_score=np.array([r['blind_score'] if r['blind_score'] is not None else 0.0 for r in records]),
        grad_energy=np.array([r['grad_energy'] if r['grad_energy'] is not None else 0.0 for r in records]),
        grad_mag_var=np.array([r['grad_mag_var'] if r['grad_mag_var'] is not None else 0.0 for r in records]),
        grad_p95=np.array([r['grad_p95'] if r['grad_p95'] is not None else 0.0 for r in records]),
        harris_max=np.array([r['harris_max'] if r['harris_max'] is not None else 0.0 for r in records]),
        harris_mean=np.array([r['harris_mean'] if r['harris_mean'] is not None else 0.0 for r in records]),
        texture_score=np.array([r['texture_score'] if r['texture_score'] is not None else 0.0 for r in records]),
        car_frac=np.array([r['car_frac'] for r in records]),
        exclude_frac=np.array([r['exclude_frac'] for r in records]),
        status=np.array([r['status'] for r in records]),
        estimator=np.array(['blind_spectral_notch'] * len(records)),
    )


def save_csv(records: list[dict], path: Path) -> None:
    fields = [
        'row', 'col', 'x0', 'y0', 'cx', 'cy', 'patch_size', 'car_frac', 'exclude_frac',
        'grad_energy', 'grad_mag_var', 'grad_p95', 'harris_max', 'harris_mean',
        'texture_score', 'blind_score', 'b_px', 'b_px_spec', 'b_px_pixel',
        'phi_deg', 'confidence', 'status',
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fields})


def save_overlay(blur_rgb: np.ndarray, records: list[dict], valid: list[dict], out_path: Path, title: str) -> None:
    H, W = blur_rgb.shape[:2]
    fig_w = 14
    fig_h = max(6, fig_w * H / max(W, 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(blur_rgb)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    ax.set_title(title)

    b_valid = np.array([r['b_px_pixel'] for r in valid]) if valid else np.array([0.0, 1.0])
    norm = colors.Normalize(vmin=float(np.nanmin(b_valid)), vmax=float(np.nanmax(b_valid)))
    cmap = plt.get_cmap('turbo')
    P = int(records[0]['patch_size']) if records else 400
    arrow_len = P * 0.34
    status_color = {
        'ok': 'white',
        'skip_car': 'red',
        'skip_excluded_region': '#f2c94c',
        'skip_flat_blurry': '0.45',
        'skip_low_texture': '0.55',
        'skip_no_corner': 'cyan',
    }

    for rec in records:
        ec = status_color.get(rec['status'], 'orange')
        ax.add_patch(Rectangle((rec['x0'], rec['y0']), P, P, fill=False, edgecolor=ec, linewidth=0.7, alpha=0.5))
        if rec['status'] != 'ok':
            continue
        phi_r = np.radians(rec['phi_deg'])
        dx = np.cos(phi_r) * arrow_len / 2
        dy = np.sin(phi_r) * arrow_len / 2
        color = cmap(norm(rec['b_px_pixel']))
        ax.annotate('', xy=(rec['cx'] + dx, rec['cy'] + dy), xytext=(rec['cx'] - dx, rec['cy'] - dy),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=2.0, shrinkA=0, shrinkB=0))
        ax.text(rec['cx'], rec['cy'] + P * 0.22, f"{rec['b_px_pixel']:.0f}px", color='white', fontsize=7,
                ha='center', va='center', bbox=dict(facecolor='black', edgecolor='none', alpha=0.4, pad=1.5))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02).set_label('Blind blur length b (px)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_uniform_summary(valid: list[dict], global_phi: float, out_dir: Path) -> None:
    if not valid:
        return
    b_vals = np.array([r['b_px_pixel'] for r in valid], dtype=float)
    weights = np.array([max(float(r.get('confidence') or 0.0), 0.0) for r in valid], dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(b_vals)
    global_b = float(np.average(b_vals, weights=weights))
    B_x = global_b * np.cos(np.radians(global_phi))
    B_y = global_b * np.sin(np.radians(global_phi))
    payload = {
        'B_x_px': float(B_x),
        'B_y_px': float(B_y),
        'b_total_px': global_b,
        'phi_deg': float(global_phi),
        'n_patches_used': int(len(valid)),
        'method': 'blind_uniform_kernel_weighted_mean',
    }
    with (out_dir / 'uniform_traj.json').open('w') as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blurry', default='pan_1.jpg')
    p.add_argument('--car_mask', default=None, help='Car mask path; defaults to <out_dir>/car_mask.png')
    p.add_argument('--exclude_mask', default=None,
                   help='Optional binary mask for background regions to exclude, e.g. track_mask.png')
    p.add_argument('--exclude_overlap_thres', type=float, default=0.25,
                   help='Skip patch if >this fraction overlaps the exclusion mask')
    p.add_argument('--patch_size', type=int, default=400)
    p.add_argument('--grad_energy_thres', type=float, default=100.0,
                   help='Minimum Sobel energy in the blurred patch.')
    p.add_argument('--grad_var_thres', type=float, default=0.0,
                   help='Minimum gradient-magnitude variance; 0 disables this gate.')
    p.add_argument('--harris_thres', type=float, default=0.0,
                   help='Minimum max Harris response; 0 disables this gate.')
    p.add_argument('--harris_block_size', type=int, default=5)
    p.add_argument('--harris_k', type=float, default=0.04)
    p.add_argument('--car_overlap_thres', type=float, default=0.10)
    p.add_argument('--global_phi', type=float, default=None)
    p.add_argument('--b_min', type=float, default=120.0)
    p.add_argument('--b_max', type=float, default=None,
                   help='Maximum blind blur length. Defaults to 0.45 * patch_size.')
    p.add_argument('--blind_score_thres', type=float, default=0.03,
                   help='Minimum spectral-notch correlation score.')
    p.add_argument('--output_root', default='outputs')
    p.add_argument('--out_dir', default=None,
                   help='Explicit output directory; defaults to <output_root>/<blurry_stem>__blind')
    args = p.parse_args()

    blurry_path = Path(args.blurry)
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.output_root) / f'{blurry_path.stem}__blind'
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.car_mask is None:
        args.car_mask = str(out_dir / 'car_mask.png')

    blur_rgb, blur_gray = load_gray_rgb(args.blurry)
    car_mask = np.array(Image.open(args.car_mask).convert('L')) > 127
    exclude_mask, exclude_mask_path = load_optional_binary_mask(
        args.exclude_mask, blur_gray.shape, 'exclusion'
    )
    print(f'Exclusion mask: {exclude_mask_path or "none"}  fraction={float(exclude_mask.mean()):.3f}')
    H, W = blur_gray.shape
    P = int(args.patch_size)
    n_rows, n_cols = H // P, W // P
    print(f'Image: {W}x{H}  patch: {P}px  grid: {n_cols}x{n_rows}')

    if args.global_phi is not None:
        global_phi = float(args.global_phi)
        print(f'Using supplied global blur direction: {global_phi:.2f} deg')
    else:
        global_phi, peak_score = estimate_blur_direction(blur_gray, weight_mask=(~car_mask) & (~exclude_mask))
        if global_phi is None:
            raise ValueError('Could not estimate blur direction from the blurred image.')
        print(f'Global blur direction: {global_phi:.2f} deg  (peak_score={peak_score:.1f})')

    b_max = float(args.b_max) if args.b_max is not None else P * 0.45
    records: list[dict] = []
    for row in range(n_rows):
        for col in range(n_cols):
            y0, x0 = row * P, col * P
            blur_p = blur_gray[y0:y0 + P, x0:x0 + P]
            car_p = car_mask[y0:y0 + P, x0:x0 + P]
            exclude_p = exclude_mask[y0:y0 + P, x0:x0 + P]
            rec = {
                'row': row, 'col': col, 'x0': x0, 'y0': y0,
                'cx': x0 + P / 2.0, 'cy': y0 + P / 2.0, 'patch_size': P,
                'car_frac': float(car_p.mean()),
                'exclude_frac': float(exclude_p.mean()),
                'b_px': None, 'b_px_spec': None, 'b_px_pixel': None,
                'phi_deg': global_phi, 'confidence': None, 'blind_score': None,
                'grad_energy': None, 'grad_mag_var': None, 'grad_p95': None,
                'harris_max': None, 'harris_mean': None, 'texture_score': None,
                'status': 'pending',
            }
            if rec['car_frac'] > args.car_overlap_thres:
                rec['status'] = 'skip_car'
                records.append(rec)
                continue
            if rec['exclude_frac'] > args.exclude_overlap_thres:
                rec['status'] = 'skip_excluded_region'
                records.append(rec)
                continue

            metrics = patch_structure_metrics(blur_p, harris_block_size=args.harris_block_size, harris_k=args.harris_k)
            rec.update(metrics)
            if metrics['grad_energy'] < args.grad_energy_thres:
                rec['status'] = 'skip_flat_blurry'
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

            try:
                b_est, blind_score = blind_sinc2_fit(blur_p, global_phi, args.b_min, b_max)
                rec['blind_score'] = float(blind_score)
                if blind_score < args.blind_score_thres:
                    rec['status'] = 'skip_low_blind_score'
                    records.append(rec)
                    continue
                rec['b_px'] = b_est
                rec['b_px_spec'] = b_est
                rec['b_px_pixel'] = b_est
                rec['confidence'] = float(metrics['texture_score'] * max(blind_score, 1e-3) ** 2)
                rec['status'] = 'ok'
            except Exception as exc:
                rec['status'] = f'fit_failed:{exc}'
            records.append(rec)

    valid = [r for r in records if r['status'] == 'ok']
    print(f'Patches: {len(records)} total | {len(valid)} valid | {len(records) - len(valid)} skipped')
    if valid:
        b_arr = np.array([r['b_px_pixel'] for r in valid], dtype=float)
        score_arr = np.array([r['blind_score'] for r in valid], dtype=float)
        w_arr = np.array([r['confidence'] for r in valid], dtype=float)
        w_arr = w_arr / w_arr.sum() if float(w_arr.sum()) > 0 else np.full_like(w_arr, 1.0 / len(w_arr))
        print(f'blind b_px: mean={b_arr.mean():.1f} std={b_arr.std():.1f} w_mean={float(np.dot(w_arr, b_arr)):.1f}')
        print(f'blind_score: mean={score_arr.mean():.3f} min={score_arr.min():.3f} max={score_arr.max():.3f}')

    save_kernel_npz(records, out_dir / 'kernel_map.npz')
    print(f"Saved: {out_dir / 'kernel_map.npz'}")
    save_csv(records, out_dir / 'kernel_map.csv')
    print(f"Saved: {out_dir / 'kernel_map.csv'}")
    save_overlay(blur_rgb, records, valid, out_dir / 'kernel_map.png', f'Blind kernel estimates | patch={P}px')
    print(f"Saved: {out_dir / 'kernel_map.png'}")
    save_uniform_summary(valid, global_phi, out_dir)
    if valid:
        print(f"Saved: {out_dir / 'uniform_traj.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
