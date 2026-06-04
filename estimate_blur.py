"""
Stage 1: Estimate global pan direction from full image gradient histogram.
Stage 2: Estimate blur kernel length from each crop via anisotropic power
         spectrum ratio.

Physics: a horizontal box blur of width b multiplies the horizontal power
spectrum by sinc²(b·f).  If the original image is isotropic (P_x ≈ P_y),
then  P_x_blurred(f) / P_y_unblurred(f)  ≈  sinc²(b·f).
Fitting that ratio to a sinc² model gives b without needing isolated edges.
"""

import argparse

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import sobel, rotate as nd_rotate, gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Camera constants (EXIF) ───────────────────────────────────────────────────
FOCAL_MM    = 31.0
SENSOR_W_MM = 23.5
IMAGE_W_PX  = 6000   # raw sensor long axis (23.5 mm)
PIXEL_PITCH = SENSOR_W_MM / IMAGE_W_PX   # mm/px  ≈ 0.003917
EXPOSURE_S  = 1 / 125


# ── Image loader ──────────────────────────────────────────────────────────────

def load_gray(path):
    """Load image applying EXIF orientation, return float32 gray."""
    img = ImageOps.exif_transpose(Image.open(path)).convert('L')
    return np.array(img, dtype=np.float32)


# ── Stage 1: Pan direction ────────────────────────────────────────────────────

def estimate_pan_direction(img_gray, n_bins=360, downsample=4):
    """Returns dominant blur direction in degrees (0 = right, 90 = down)."""
    h = img_gray[::downsample, ::downsample]
    gx = sobel(h, axis=1)
    gy = sobel(h, axis=0)
    mag = np.hypot(gx, gy)
    angle = np.degrees(np.arctan2(gy, gx)) % 180
    hist, edges = np.histogram(angle, bins=n_bins, range=(0, 180), weights=mag)
    centers = (edges[:-1] + edges[1:]) / 2
    peaks, props = find_peaks(hist, height=hist.max() * 0.3)
    peak_idx = peaks[np.argmax(props['peak_heights'])] if len(peaks) else np.argmax(hist)
    # Gradient peak is perpendicular to blur; blur direction is +90° away.
    gradient_peak = float(centers[peak_idx])
    blur_direction = (gradient_peak + 90.0) % 180.0
    return blur_direction, hist, centers


# ── Stage 2: Blur length via anisotropic power spectrum ratio ────────────────

def rotate_patch(patch, angle_deg):
    """Rotate patch so blur is exactly horizontal."""
    return nd_rotate(patch, -angle_deg, reshape=False, mode='reflect')


def power_spectrum_ratio_fit(patch, pan_angle_deg, f_max=0.25, f_min=0.01):
    """
    Estimate blur kernel length b (px) from the anisotropic power spectrum.

    Steps:
    1. Rotate patch so blur is horizontal.
    2. Compute 2D power spectrum P(fx, fy).
    3. Marginals: P_x(f) = sum_fy P(f, fy),  P_y(f) = sum_fx P(fx, f).
    4. Ratio R(f) = P_x(f) / P_y(f)  ≈  sinc²(b·f).
    5. Fit sinc²(b·f) + c to R(f) → b.

    Returns b_px, freq array, measured ratio, fitted ratio.
    """
    rot = rotate_patch(patch, pan_angle_deg)
    N = min(rot.shape)
    N = N - (N % 2)   # ensure even
    cy, cx = np.array(rot.shape) // 2
    crop = rot[cy - N//2: cy + N//2, cx - N//2: cx + N//2]

    # Apodize
    win = np.outer(np.hanning(N), np.hanning(N))
    F = np.fft.fftshift(np.fft.fft2(crop * win))
    PS = np.abs(F) ** 2

    freqs = np.fft.fftshift(np.fft.fftfreq(N))  # cycles/px

    # Marginal spectra — sum only positive halves (symmetry)
    pos = freqs > 0
    f_pos = freqs[pos]
    P_x = PS[:, pos].sum(axis=0)   # sum over all fy, positive fx
    P_y = PS[pos, :].sum(axis=1)   # sum over all fx, positive fy

    # Restrict to fitting band [f_min, f_max]
    mask = (f_pos >= f_min) & (f_pos <= f_max)
    f_fit = f_pos[mask]
    R_fit = (P_x / (P_y + 1e-30))[mask]
    R_fit /= R_fit.max() + 1e-30   # normalise to [0, 1]

    # Sinc² model
    def sinc2(f, b, c):
        return np.sinc(b * f) ** 2 + c   # np.sinc(x) = sin(πx)/(πx)

    # Init guess from first local minimum of R
    valleys, _ = find_peaks(-R_fit)
    b_init = (1.0 / f_fit[valleys[0]]) if len(valleys) else 30.0

    try:
        popt, _ = curve_fit(sinc2, f_fit, R_fit,
                            p0=[b_init, R_fit.min()],
                            bounds=([1, -0.5], [N, 0.5]),
                            maxfev=10000)
        b_fit, c_fit = popt
    except Exception as e:
        print(f"    [warn] fit failed: {e}  (using init guess {b_init:.1f})")
        b_fit, c_fit = b_init, 0.0

    f_dense = np.linspace(f_min, f_max, 500)
    R_model = sinc2(f_dense, b_fit, c_fit)

    return float(b_fit), f_pos[mask], R_fit, f_dense, R_model


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_background_strips(img_raw, strip_size=2000):
    """
    Extract square background strips from the EXIF-oriented image.
    For the original full pan image, three bands roughly sample sky/upper/lower
    background. Smaller images use fractional row positions.

    Returns list of (label, patch_gray).
    """
    H, W = img_raw.shape
    strip_size = min(strip_size, H, W)
    if strip_size < 16:
        return []

    if H >= 3500:
        row_centres = [700, 1800, 2800]
    else:
        row_centres = [int(H * f) for f in (0.25, 0.5, 0.75)]

    strips = []
    for rc in row_centres:
        r0 = min(max(0, rc - strip_size // 2), H - strip_size)
        r1 = r0 + strip_size
        c0 = min(max(0, (W - strip_size) // 2), W - strip_size)
        c1 = c0 + strip_size
        label = f'full_row{rc}'
        strips.append((label, img_raw[r0:r1, c0:c1].copy()))
    return strips


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, default='crop_4.jpg',
                   help='Main image used for Stage 1 and large strip fitting')
    p.add_argument('--small_crop', '--small_crops', nargs='+', action='append',
                   default=[],
                   help='Optional small crop image(s) for Stage 2a diagnostics')
    p.add_argument('--strip_size', type=int, default=2000,
                   help='Square strip size for Stage 2b')
    args = p.parse_args()
    small_crop_paths = [path for group in args.small_crop for path in group]

    # ── Stage 1: pan direction ────────────────────────────────────────────────
    print("=== Stage 1: Pan Direction ===")
    img = load_gray(args.image)
    print(f"  Image: {args.image}  ({img.shape[1]}×{img.shape[0]} px, EXIF applied)")
    pan_angle, hist, centers = estimate_pan_direction(img)
    print(f"  Pan direction: {pan_angle:.2f}°  (0° = horizontal after EXIF orientation)")

    # ── Stage 2a: small crops (diagnostic) ───────────────────────────────────
    if small_crop_paths:
        print("\n=== Stage 2a: Small Crops (diagnostic — expect patch-size bias) ===")
        small_results = []
        for path in small_crop_paths:
            patch = load_gray(path)
            b_px, *_ = power_spectrum_ratio_fit(patch, pan_angle)
            small_results.append((path, patch.shape[1], b_px))
            print(f"  {path} ({patch.shape[1]}px):  b = {b_px:.1f} px   b/N = {b_px/patch.shape[1]:.3f}")
        print("  (b/N constant → bias; use full-image strips below for reliable estimate)")
    else:
        print("\n=== Stage 2a: Small Crops skipped (no --small_crop given) ===")

    # ── Stage 2b: large strips from full image ────────────────────────────────
    print("\n=== Stage 2b: Large Background Strips (reliable measurement) ===")
    strips = extract_background_strips(img, args.strip_size)
    if not strips:
        raise ValueError('Image is too small for Stage 2b strip analysis')

    n = len(strips)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    b_values, alpha_rads = [], []

    for idx, (label, patch) in enumerate(strips):
        H_p, W_p = patch.shape
        b_px, f_m, R_m, f_d, R_model = power_spectrum_ratio_fit(
            patch, pan_angle, f_min=0.005, f_max=0.2)
        b_values.append(b_px)
        alpha_rad = (b_px * PIXEL_PITCH) / FOCAL_MM
        alpha_rads.append(alpha_rad)

        print(f"\n  {label}  ({W_p}×{H_p} px):")
        print(f"    Blur length b:  {b_px:.1f} px   (b/N = {b_px/W_p:.3f})")
        print(f"    Angular disp α: {alpha_rad*1000:.2f} mrad  =  {np.degrees(alpha_rad):.4f}°")

        ax = axes[idx]
        ax.plot(f_m, R_m, '.', markersize=2, alpha=0.6, label='P_x / P_y  (measured)')
        ax.plot(f_d, R_model, 'r-', linewidth=2,
                label=f'sinc²(b·f),  b = {b_px:.1f} px')
        ax.set_xlabel('Spatial frequency (cycles/px)')
        ax.set_ylabel('Normalised ratio')
        ax.set_title(f'{label} — anisotropic power spectrum ratio')
        ax.legend()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    b_arr = np.array(b_values)
    a_arr = np.array(alpha_rads)
    print(f"  Blur lengths b  (px):   {b_arr.round(1).tolist()}")
    print(f"  α per strip (mrad): {(a_arr*1000).round(2).tolist()}")
    print(f"  Mean α: {a_arr.mean()*1000:.2f} mrad  ±  {a_arr.std()*1000:.2f} mrad")
    omega = a_arr.mean() / EXPOSURE_S
    print(f"  Camera ω: {np.degrees(omega):.1f} °/s  =  {omega:.3f} rad/s")

    cv = a_arr.std() / a_arr.mean()
    if cv > 0.10:
        print(f"  [!] α CV = {cv:.1%} > 10% → depth-dependent blur (parallax present)")
    else:
        print(f"  [✓] α consistent (CV = {cv:.1%}) — pure camera rotation.")

    plt.tight_layout()
    plt.savefig('blur_psratio_analysis.png', dpi=120)
    print("\nPlot saved: blur_psratio_analysis.png")

    # Pan direction histogram
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(centers, hist / hist.max())
    ax2.axvline((pan_angle + 90) % 180, color='gray', linestyle=':',
                label='gradient peak')
    ax2.axvline(pan_angle, color='r', linestyle='--',
                label=f'blur direction = {pan_angle:.1f}°')
    ax2.set_xlabel('Gradient orientation (°)');  ax2.set_ylabel('Normalised weight')
    ax2.set_title('Gradient orientation histogram (raw full image)')
    ax2.legend();  plt.tight_layout()
    plt.savefig('pan_direction.png', dpi=120)
    print("Plot saved: pan_direction.png")


if __name__ == '__main__':
    main()
