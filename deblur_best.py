"""
Best-in-class panning-shot deblurrer.

Method: DPIR-style Plug-and-Play (Half-Quadratic Splitting) with DRUNet
        and a geometrically decreasing noise-level schedule.

Compared to the fixed-sigma ADMM in deblur.py, the decreasing σ schedule
gives significantly sharper results in fewer iterations:
  - high σ at start → strong regularisation, suppresses ringing
  - low σ at end   → light touch, recovers fine texture

Pipeline
────────
1. Auto-estimate blur angle + length from the image (sinc² power-spectrum
   ratio method, reusing estimate_blur.py), unless --angle / --b are given.
2. Wiener deconvolution (fast baseline).
3. DPIR-HQS with DRUNet color prior (best quality).
4. 4-panel comparison: input | Wiener | DPIR | DPIR detail crop.

Usage examples
──────────────
  python deblur_best.py --image crop_7.jpg
  python deblur_best.py --image crop_4.jpg --b 75 --angle 0.25
  python deblur_best.py --image full_pan_2.jpg --crop 800 --crop_row 1500
  python deblur_best.py --image crop_4.jpg --dpir_iters 30 --sigma_start 49 --sigma_end 2
"""

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── reuse blur-estimation from estimate_blur.py ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from estimate_blur import estimate_pan_direction, power_spectrum_ratio_fit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── DRUNet (copy from deblur.py – same architecture) ─────────────────────────

class ResBlock(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(n, n, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n, n, 3, padding=1, bias=False),
        )
    def forward(self, x):
        return x + self.res(x)


class DRUNetColor(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_head  = nn.Conv2d(4, 64, 3, padding=1, bias=False)
        self.m_down1 = nn.Sequential(ResBlock(64), ResBlock(64), ResBlock(64), ResBlock(64),
                                     nn.Conv2d(64, 128, 2, stride=2, bias=False))
        self.m_down2 = nn.Sequential(ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128),
                                     nn.Conv2d(128, 256, 2, stride=2, bias=False))
        self.m_down3 = nn.Sequential(ResBlock(256), ResBlock(256), ResBlock(256), ResBlock(256),
                                     nn.Conv2d(256, 512, 2, stride=2, bias=False))
        self.m_body  = nn.Sequential(ResBlock(512), ResBlock(512), ResBlock(512), ResBlock(512))
        self.m_up3   = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False),
                                     ResBlock(256), ResBlock(256), ResBlock(256), ResBlock(256))
        self.m_up2   = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
                                     ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128))
        self.m_up1   = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False),
                                     ResBlock(64), ResBlock(64), ResBlock(64), ResBlock(64))
        self.m_tail  = nn.Conv2d(64, 3, 3, padding=1, bias=False)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x  = self.m_body(x4)
        x  = self.m_up3(x + x4)
        x  = self.m_up2(x + x3)
        x  = self.m_up1(x + x2)
        return self.m_tail(x + x1)


def load_drunet(path):
    m = DRUNetColor().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


@torch.no_grad()
def drunet_denoise(v, model, sigma):
    """
    Denoise a (3, H, W) float32 tensor in [0, 255] at noise level sigma/255.
    Pads to multiples of 8 for the encoder-decoder strides.
    """
    h, w = v.shape[-2], v.shape[-1]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    x  = v.unsqueeze(0).clamp(0, 255) / 255.0
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='replicate')
    noise_map = torch.full((1, 1, x.shape[-2], x.shape[-1]),
                           sigma / 255.0, device=DEVICE)
    out = model(torch.cat([x, noise_map], dim=1))
    return out.squeeze(0)[:, :h, :w].clamp(0, 1) * 255.0


# ── Blur kernel ───────────────────────────────────────────────────────────────

def make_kernel_fft(L, angle_deg, shape):
    """Sinc kernel in Fourier domain: K(fx,fy) = sinc(L·(fx·cosθ + fy·sinθ))."""
    H, W = shape
    fy = torch.fft.fftfreq(H, device=DEVICE).view(-1, 1)
    fx = torch.fft.fftfreq(W, device=DEVICE).view(1, -1)
    c  = math.cos(math.radians(angle_deg))
    s  = math.sin(math.radians(angle_deg))
    K  = torch.sinc(L * (fx * c + fy * s))
    return K, K ** 2  # (K, |K|²) — real since centred box is even


# ── Wiener deconvolution ──────────────────────────────────────────────────────

def wiener_deconvolve(y, K, K2, eps):
    """Direct Tikhonov deconvolution for a (C, H, W) tensor."""
    Y = torch.fft.fft2(y)
    return torch.fft.ifft2(K * Y / (K2 + eps)).real


# ── DPIR-HQS core ─────────────────────────────────────────────────────────────

def dpir_hqs(y, K, K2, model, sigma_schedule, sigma_n=10.0, mu=0.23,
             K2_clip=0.01, wiener_warmstart_eps=0.02):
    """
    DPIR Half-Quadratic Splitting (Zhang et al., 2022) with DRUNet.

    Two key improvements over vanilla DPIR for very strong blur (b≥50px):

    1. Frequency-adaptive ρ_f: where |K(f)|² < K2_clip (sinc nulls and
       near-nulls), ρ_f is boosted so the x-update falls back to z rather
       than amplifying noise.  At safe frequencies, ρ_f = ρ.

         ρ_f = ρ · K2_clip / max(|K|², K2_clip)

    2. Wiener warm-start for z: initialise z with a Tikhonov-regularised
       deconvolution rather than the raw blurred image.  This gives the
       first DRUNet call a head-start on the deblurred structure.

    y   : (3, H, W) float32 in [0, 255]
    K   : (H, W) blur kernel in Fourier domain (real sinc)
    K2  : |K|²
    Returns (3, H, W) float32 in [0, 255].
    """
    sn  = sigma_n / 255.0
    y01 = y / 255.0
    Y   = torch.fft.fft2(y01)

    # Warm-start z with a conservative Wiener estimate
    with torch.no_grad():
        K2_safe_ws = K2.clamp(min=K2_clip)
        z01_init   = torch.fft.ifft2(K * Y / (K2_safe_ws + wiener_warmstart_eps)).real
    z01 = z01_init.clone()

    # Precompute frequency-adaptive kernel clip: boosts ρ near sinc nulls
    K2_safe = K2.clamp(min=K2_clip)          # never let |K|² drop below clip
    freq_weight = K2_clip / K2_safe          # 1.0 at safe freqs, >1 near nulls

    for i, sigma_k in enumerate(sigma_schedule):
        sk  = sigma_k / 255.0
        rho = mu * (sn / sk) ** 2            # base DPIR penalty
        rho_f = rho * freq_weight            # freq-adaptive: large at nulls

        Z   = torch.fft.fft2(z01)
        x01 = torch.fft.ifft2((K * Y + rho_f * Z) / (K2 + rho_f)).real

        x_255 = (x01 * 255.0).clamp(0, 255)
        z_255 = drunet_denoise(x_255, model, sigma_k)
        z01   = z_255 / 255.0

        if (i + 1) % 5 == 0 or i == len(sigma_schedule) - 1:
            resid = (x01 - z01).norm().item()
            print(f'  iter {i+1:3d}/{len(sigma_schedule)}  '
                  f'σ={sigma_k:.2f}  ρ_base={rho:.4f}  resid={resid:.4f}',
                  flush=True)

    return (x01 * 255.0).clamp(0, 255)


def notch_cleanup(x_255, K2, model, b, angle_deg, sigma_clean=5.0, K2_null_thresh=0.02):
    """
    Post-process DPIR output to suppress sinc-null ringing.

    Two steps:
    1. Frequency-domain notch: attenuate bands near |K|²<thresh using the
       DRUNet estimate (which correctly fills those from image statistics).
    2. One final DRUNet pass at low sigma to smooth residual artifacts.
    """
    x01 = x_255 / 255.0

    # Identify null/near-null frequency mask
    null_mask = (K2 < K2_null_thresh).float()            # 1 at nulls, 0 safe
    safe_mask = 1.0 - null_mask

    # Fill null frequencies with DRUNet estimate
    z_255 = drunet_denoise(x_255, model, sigma_clean)
    z01   = z_255 / 255.0
    X     = torch.fft.fft2(x01)
    Z     = torch.fft.fft2(z01)

    # Blend: keep safe frequencies from DPIR; use DRUNet at nulls
    X_blend = safe_mask * X + null_mask * Z
    x_blend = torch.fft.ifft2(X_blend).real

    # Final light DRUNet pass to smooth edges between regions
    x_blend_255 = (x_blend * 255.0).clamp(0, 255)
    out = drunet_denoise(x_blend_255, model, sigma_clean)
    return out.clamp(0, 255)


# ── Auto blur estimation ──────────────────────────────────────────────────────

def auto_estimate_blur(img_np, strip_size=None):
    """
    Estimate (angle_deg, b_px) from a grayscale image using power-spectrum ratio.
    Returns (angle, b_px, confidence) where confidence = 1 - std/mean of b estimates.
    """
    gray = img_np.mean(axis=2).astype(np.float32) if img_np.ndim == 3 else img_np.astype(np.float32)
    H, W = gray.shape

    # Stage 1: pan direction
    angle, *_ = estimate_pan_direction(gray)
    print(f'  Auto-estimated angle: {angle:.2f}°', flush=True)

    # Stage 2: fit sinc² on up to 3 horizontal strips
    ss = min(strip_size or min(H, W, 2000), H, W)
    row_centres = [int(H * f) for f in (0.25, 0.5, 0.75)]
    b_vals = []
    for rc in row_centres:
        r0 = min(max(0, rc - ss // 2), H - ss)
        c0 = min(max(0, (W - ss) // 2), W - ss)
        patch = gray[r0:r0+ss, c0:c0+ss]
        try:
            b, *_ = power_spectrum_ratio_fit(patch, angle, f_min=0.005, f_max=0.2)
            b_vals.append(b)
        except Exception:
            pass

    if not b_vals:
        raise RuntimeError('Power-spectrum fit failed on all strips.')

    b_arr  = np.array(b_vals)
    b_mean = float(b_arr.mean())
    conf   = 1.0 - float(b_arr.std() / (b_mean + 1e-6))
    print(f'  Auto-estimated b: {b_mean:.1f} px  (strips: {b_arr.round(1).tolist()}  conf={conf:.2f})',
          flush=True)
    return angle, b_mean, conf


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_result(img_np, wiener_np, dpir_np, out_path, crop_box=None, title_suffix=''):
    """
    Save a 4-panel figure:
      [Input] [Wiener] [DPIR] [DPIR detail crop]
    """
    H, W = img_np.shape[:2]

    # pick a central crop for the detail panel
    if crop_box is None:
        sz  = min(256, H // 2, W // 2)
        cy, cx = H // 2, W // 2
        crop_box = (cy - sz // 2, cx - sz // 2, cy + sz // 2, cx + sz // 2)

    r0, c0, r1, c1 = crop_box
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(H, r1), min(W, c1)

    panels = [
        (img_np,          'Input (blurred)'),
        (wiener_np,       'Wiener (baseline)'),
        (dpir_np,         f'DPIR-HQS + DRUNet'),
        (dpir_np[r0:r1, c0:c1], f'DPIR detail crop\n[{c0}:{c1}, {r0}:{r1}]'),
    ]

    fig = plt.figure(figsize=(22, 7))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04)
    for i, (im, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(im)
        ax.set_title(lab, fontsize=12)
        ax.axis('off')
        if i < 3:
            ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0,
                                        fill=False, edgecolor='lime',
                                        linewidth=2))
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def save_side_by_side(img_np, dpir_np, out_path):
    """Simple 2-panel comparison for quick inspection."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, im, lab in zip(axes, [img_np, dpir_np], ['Input (blurred)', 'DPIR-HQS + DRUNet']):
        ax.imshow(im)
        ax.set_title(lab, fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Best-in-class panning deblurrer (DPIR-HQS + DRUNet)')
    p.add_argument('--image',        default='crop_7.jpg')
    p.add_argument('--drunet_path',  default='drunet_color.pth')
    p.add_argument('--out_dir',      default='outputs')

    # Kernel (auto-estimated if not provided)
    p.add_argument('--angle',  type=float, default=None,
                   help='Pan angle in degrees (auto-estimated if omitted)')
    p.add_argument('--b',      type=float, default=None,
                   help='Blur length in pixels (auto-estimated if omitted)')

    # Crop
    p.add_argument('--crop',     type=int, default=None,
                   help='Crop a square of this size from the image')
    p.add_argument('--crop_row', type=int, default=1500)
    p.add_argument('--crop_col', type=int, default=None)

    # DPIR schedule
    p.add_argument('--dpir_iters',   type=int,   default=24,
                   help='Number of HQS iterations (24 is usually enough)')
    p.add_argument('--sigma_start',  type=float, default=49.0,
                   help='Starting denoiser σ (pixel units)')
    p.add_argument('--sigma_end',    type=float, default=10.0,
                   help='Ending denoiser σ; should match sigma_n (DPIR convention)')
    p.add_argument('--sigma_n',      type=float, default=10.0,
                   help='Observation noise σ (pixel units; ~10 for JPEG panning shots)')
    p.add_argument('--K2_clip',      type=float, default=0.01,
                   help='|K|² clip for freq-adaptive ρ: controls null-freq fallback')

    # Wiener baseline
    p.add_argument('--wiener_eps',   type=float, default=0.01)

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Device: {DEVICE}', flush=True)

    # ── Load image ────────────────────────────────────────────────────────────
    img_pil = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    img_np  = np.array(img_pil, dtype=np.uint8)

    if args.crop:
        H, W = img_np.shape[:2]
        r0 = args.crop_row
        c0 = args.crop_col if args.crop_col is not None else (W - args.crop) // 2
        img_np = img_np[r0:r0 + args.crop, c0:c0 + args.crop]
        print(f'Crop: {img_np.shape[1]}×{img_np.shape[0]} @ row={r0} col={c0}')

    H, W = img_np.shape[:2]
    print(f'Image: {args.image}  {W}×{H} px', flush=True)

    # ── Auto blur estimation ──────────────────────────────────────────────────
    if args.angle is None or args.b is None:
        print('Auto-estimating blur kernel …', flush=True)
        est_angle, est_b, _ = auto_estimate_blur(img_np)
        angle = est_angle if args.angle is None else args.angle
        b     = est_b     if args.b     is None else args.b
    else:
        angle, b = args.angle, args.b

    print(f'Kernel: b={b:.1f} px  angle={angle:.2f}°', flush=True)

    # ── Build kernel ──────────────────────────────────────────────────────────
    y  = torch.from_numpy(np.moveaxis(img_np.astype(np.float32), -1, 0)).to(DEVICE)
    K, K2 = make_kernel_fft(b, angle, (H, W))

    # ── Wiener baseline ───────────────────────────────────────────────────────
    print('\nWiener deconvolution …', flush=True)
    t0 = time.time()
    with torch.no_grad():
        wiener_t = wiener_deconvolve(y, K, K2, args.wiener_eps)
    wiener_np = wiener_t.clamp(0, 255).cpu().numpy().astype(np.uint8)
    wiener_np = np.moveaxis(wiener_np, 0, -1)
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    # ── Load DRUNet ───────────────────────────────────────────────────────────
    print(f'\nLoading DRUNet from {args.drunet_path} …', flush=True)
    model = load_drunet(args.drunet_path)

    # ── DPIR-HQS ─────────────────────────────────────────────────────────────
    sigma_schedule = np.geomspace(args.sigma_start, args.sigma_end,
                                  args.dpir_iters).tolist()
    print(f'\nDPIR-HQS: {args.dpir_iters} iters  '
          f'σ {args.sigma_start:.1f}→{args.sigma_end:.2f}  σ_n={args.sigma_n}', flush=True)
    t0 = time.time()
    dpir_t   = dpir_hqs(y, K, K2, model, sigma_schedule, args.sigma_n,
                         K2_clip=args.K2_clip)

    # ── Notch cleanup (suppress sinc-null ringing) ────────────────────────
    print('\nNotch cleanup …', flush=True)
    dpir_clean_t = notch_cleanup(dpir_t, K2, model, b, angle,
                                  sigma_clean=args.sigma_end * 0.5,
                                  K2_null_thresh=args.K2_clip)
    dpir_np       = dpir_t.cpu().numpy().astype(np.uint8)
    dpir_np       = np.moveaxis(dpir_np, 0, -1)
    dpir_clean_np = dpir_clean_t.cpu().numpy().astype(np.uint8)
    dpir_clean_np = np.moveaxis(dpir_clean_np, 0, -1)
    print(f'  total in {time.time()-t0:.1f}s', flush=True)

    # ── Save outputs ──────────────────────────────────────────────────────────
    stem = Path(args.image).stem
    tag  = f'{stem}_b{b:.0f}_dpir{args.dpir_iters}'

    Image.fromarray(wiener_np).save(out_dir / f'{tag}_wiener.png')
    Image.fromarray(dpir_np).save(out_dir / f'{tag}_dpir.png')
    Image.fromarray(dpir_clean_np).save(out_dir / f'{tag}_dpir_clean.png')
    print(f'Saved: {out_dir}/{tag}_wiener.png')
    print(f'Saved: {out_dir}/{tag}_dpir.png')
    print(f'Saved: {out_dir}/{tag}_dpir_clean.png')

    save_side_by_side(img_np, dpir_clean_np,
                      out_dir / f'{tag}_comparison.png')

    # 5-panel: input | Wiener | DPIR | DPIR+cleanup | detail crop
    H2, W2 = img_np.shape[:2]
    sz = min(256, H2 // 2, W2 // 2)
    cy, cx = H2 // 2, W2 // 2
    r0, c0, r1, c1 = cy - sz//2, cx - sz//2, cy + sz//2, cx + sz//2

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(28, 7))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.04)
    panels5 = [
        (img_np,        'Input (blurred)'),
        (wiener_np,     'Wiener baseline'),
        (dpir_np,       f'DPIR-HQS (σ {args.sigma_start:.0f}→{args.sigma_end:.0f})'),
        (dpir_clean_np, 'DPIR + notch cleanup'),
        (dpir_clean_np[r0:r1, c0:c1], f'Detail crop\n[{c0}:{c1}, {r0}:{r1}]'),
    ]
    for i, (im, lab) in enumerate(panels5):
        ax = fig.add_subplot(gs[i])
        ax.imshow(im)
        ax.set_title(lab, fontsize=11)
        ax.axis('off')
        if i < 4:
            ax.add_patch(plt.Rectangle((c0, r0), c1-c0, r1-r0,
                                        fill=False, edgecolor='lime', linewidth=2))
    fig.suptitle(f'{args.image} | b={b:.0f}px angle={angle:.2f}° | '
                 f'K2_clip={args.K2_clip}', fontsize=10, y=1.01)
    plt.savefig(out_dir / f'{tag}_5panel.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_dir}/{tag}_5panel.png')


if __name__ == '__main__':
    main()
