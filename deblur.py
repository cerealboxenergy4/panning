"""
ADMM-based deconvolution of the panning shot.

Two modes
─────────
Non-blind (--mode nonblind):
  Fixed kernel of length --b.  Runs TV/DnCNN/DRUNet P&P ADMM, or direct
  Wiener filtering with --prior wiener.

Blind (--mode blind):
  Alternating optimisation over image x and scalar kernel length L:

    Initialise L in [L_lo, L_hi]
    for outer = 1 … T:
        build K_L in Fourier domain  (sinc(L · f_proj))
        run --inner PnP-ADMM iterations for x  (warm-started)
        1-D golden-section search:  L ← argmin_L ||y - K_L * x||²
        shrink bracket:  [L - δ, L + δ],  δ ← δ · shrink

Kernel in Fourier domain (continuous L, no pixel rounding):
    K_L(fx, fy) = sinc(L · (fx·cosθ + fy·sinθ))
which is the exact DFT of a centred 1-D box of length L along angle θ.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ── Defaults (from blur-estimation stage) ────────────────────────────────────
B_PX          = 77
PAN_ANGLE_DEG = 0.25

# ── DPS: path to the OpenAI guided-diffusion repo ────────────────────────────
_GUIDED_DIFFUSION_REPO = str(Path(__file__).resolve().parent.parent / 'guided-diffusion')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── DnCNN ─────────────────────────────────────────────────────────────────────
# dncnn_25.pth: 17 conv layers, bias=True, no BN, residual (predict noise).

class DnCNN(nn.Module):
    def __init__(self, n_layers=17, n_features=64):
        super().__init__()
        layers = [nn.Conv2d(1, n_features, 3, padding=1, bias=True),
                  nn.ReLU(inplace=True)]
        for _ in range(n_layers - 2):
            layers += [nn.Conv2d(n_features, n_features, 3, padding=1, bias=True),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(n_features, 1, 3, padding=1, bias=True)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.model(x)   # clean = noisy − predicted_noise


def load_dncnn(path, device):
    m = DnCNN().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


# ── DRUNet color ──────────────────────────────────────────────────────────────
# drunet_color.pth: RGB denoiser conditioned on a 1-channel noise-level map.

class ResBlock(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.res(x)


class DRUNetColor(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_head = nn.Conv2d(4, 64, 3, padding=1, bias=False)
        self.m_down1 = nn.Sequential(
            ResBlock(64), ResBlock(64), ResBlock(64), ResBlock(64),
            nn.Conv2d(64, 128, 2, stride=2, bias=False))
        self.m_down2 = nn.Sequential(
            ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128),
            nn.Conv2d(128, 256, 2, stride=2, bias=False))
        self.m_down3 = nn.Sequential(
            ResBlock(256), ResBlock(256), ResBlock(256), ResBlock(256),
            nn.Conv2d(256, 512, 2, stride=2, bias=False))
        self.m_body = nn.Sequential(
            ResBlock(512), ResBlock(512), ResBlock(512), ResBlock(512))
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False),
            ResBlock(256), ResBlock(256), ResBlock(256), ResBlock(256))
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128))
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False),
            ResBlock(64), ResBlock(64), ResBlock(64), ResBlock(64))
        self.m_tail = nn.Conv2d(64, 3, 3, padding=1, bias=False)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


def load_drunet_color(path, device):
    m = DRUNetColor().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


# ── DPS (Diffusion Posterior Sampling) ───────────────────────────────────────
# Reference: Chung et al., "Diffusion Posterior Sampling for General Noisy
#            Inverse Problems", ICLR 2023.  https://arxiv.org/abs/2209.14687
#
# Model: OpenAI 512×512 ImageNet ADM (class-conditional, learn_sigma=True).
# Checkpoint: 512x512_diffusion.pt
# URL: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt

# Config from guided-diffusion README (512×512 model flags)
_CFG_512 = dict(
    image_size=512, num_channels=256, num_res_blocks=2,
    num_heads=4, num_head_channels=64, num_heads_upsample=-1,
    attention_resolutions='32,16,8', channel_mult='', dropout=0.0,
    class_cond=True, use_checkpoint=False, use_scale_shift_norm=True,
    resblock_updown=True, use_fp16=False, use_new_attention_order=False,
    learn_sigma=True, diffusion_steps=1000, noise_schedule='linear',
    use_kl=False, predict_xstart=False, rescale_timesteps=False,
    rescale_learned_sigmas=False,
)


def _guided_diffusion_create():
    """Import create_model_and_diffusion, adding the repo to sys.path if needed."""
    if _GUIDED_DIFFUSION_REPO not in sys.path:
        sys.path.insert(0, _GUIDED_DIFFUSION_REPO)
    try:
        from guided_diffusion.script_util import create_model_and_diffusion
        return create_model_and_diffusion
    except ImportError:
        raise ImportError(
            'guided_diffusion not found.  Clone it:\n'
            f'  git -C {Path(_GUIDED_DIFFUSION_REPO).parent} clone '
            'https://github.com/openai/guided-diffusion.git\n'
            'Then download the checkpoint:\n'
            '  wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt'
        )


class _MeanClassEmbedding(nn.Module):
    """Replaces the label-embedding lookup with the mean over all 1000 classes.

    This marginalises out the class conditioning so the model behaves as a
    class-agnostic (effectively unconditional) natural-image prior.  The UNet
    still requires a `y` argument to satisfy its internal assert; we ignore it.
    """
    def __init__(self, label_emb: nn.Embedding):
        super().__init__()
        with torch.no_grad():
            mean = label_emb.weight.mean(0, keepdim=True)   # (1, D)
        self.register_buffer('mean_emb', mean)

    def forward(self, y):
        return self.mean_emb.expand(y.shape[0], -1)


def load_openai_512_diffusion(checkpoint_path, device, ddim_steps=100,
                               dps_class=-1):
    """
    Load OpenAI 512×512 ImageNet ADM with DDIM respacing.

    dps_class  : ImageNet label 0-999 to condition on, or -1 (default) to
                 marginalise over all classes by using the mean class embedding.
    ddim_steps : number of DDIM reverse steps (fewer = faster).
    """
    create_model_and_diffusion = _guided_diffusion_create()
    cfg = dict(_CFG_512, timestep_respacing=f'ddim{ddim_steps}')
    model, diffusion = create_model_and_diffusion(**cfg)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)

    if dps_class == -1:
        model.label_emb = _MeanClassEmbedding(model.label_emb)
        print('  DPS: using mean class embedding (marginalised over all 1000 classes)',
              flush=True)

    model.to(device).eval()
    return model, diffusion


def _dps_ddim_step(x_t, t_batch, model, diffusion,
                   y_tile_norm, L, angle_deg, model_kwargs, zeta):
    """
    One DDIM reverse step with DPS measurement-gradient correction.

    x_t          : (1, 3, H, W)  current noisy estimate in [-1, 1]
    y_tile_norm  : (1, 3, H, W)  blurred measurement in [-1, 1]
    Returns x_{t-1} as a detached tensor.
    """
    th_tile, tw_tile = x_t.shape[2], x_t.shape[3]

    x_t = x_t.detach().requires_grad_(True)

    # DDIM step — gradient graph flows through pred_xstart → x_t
    out    = diffusion.ddim_sample(model, x_t, t_batch,
                                   clip_denoised=True, model_kwargs=model_kwargs)
    x0_hat = out['pred_xstart']   # (1, 3, H, W) in [-1, 1]
    x_prev = out['sample']        # (1, 3, H, W) DDIM update

    # DPS likelihood: ||y − A(x̂₀)||² where A = blur kernel
    x0_255 = (x0_hat + 1.0) * 127.5
    y_255  = (y_tile_norm + 1.0) * 127.5
    K_t, _, _ = make_kernel_fft(L, angle_deg, (th_tile, tw_tile), x_t.device)
    Ax0 = torch.fft.ifft2(K_t * torch.fft.fft2(x0_255)).real

    loss = ((y_255 - Ax0) ** 2).sum()
    grad = torch.autograd.grad(loss, x_t)[0]

    # Normalise gradient so step size is predictable across noise levels
    x_corrected = x_prev.detach() - zeta * grad / (grad.norm() + 1e-8)
    return x_corrected.detach()


def dps_deblur_tiled(y_img, L, angle_deg, model, diffusion,
                     n_steps, zeta, dps_class,
                     tile=512, overlap=64, device=DEVICE):
    """
    DPS deblurring on a (3, H, W) float32 image in [0, 255].

    Runs full DDIM reverse diffusion (n_steps steps) with blur-likelihood
    gradient guidance on overlapping 512×512 tiles, then blends with a
    cosine weight mask.  Returns (3, H, W) float32 in [0, 255].

    The model is the OpenAI 512×512 ImageNet ADM (class-conditional).
    dps_class selects the ImageNet conditioning label (1000 classes; 0 = tench).
    """
    C, H, W = y_img.shape
    if C != 3:
        raise ValueError('dps prior requires a 3-channel RGB image; use --deblur_space rgb')

    y_norm = (y_img / 127.5) - 1.0   # (3, H, W) in [-1, 1]

    # When dps_class == -1, _MeanClassEmbedding ignores the label value;
    # we still pass y=0 to satisfy the UNet's internal assert.
    y_label = max(dps_class, 0)
    model_kwargs = {'y': torch.tensor([y_label], device=device)}

    # DDIM timestep indices (respaced, high→low)
    indices = list(range(diffusion.num_timesteps))[::-1]

    out_sum    = torch.zeros_like(y_img)            # (3, H, W)
    out_weight = torch.zeros(1, H, W, device=device)

    step = tile - 2 * overlap
    ys_list = sorted(set(
        list(range(0, max(H - tile + 1, 1), step)) + [max(0, H - tile)]))
    xs_list = sorted(set(
        list(range(0, max(W - tile + 1, 1), step)) + [max(0, W - tile)]))
    total_tiles = len(ys_list) * len(xs_list)

    # Cosine blend weight for a 1-D range of length n (peaks at centre)
    def _cos_weight_1d(n):
        w = torch.hann_window(n, periodic=False, device=device)
        return w.clamp(min=1e-3)

    print(f'  DPS: {total_tiles} tile(s) × {n_steps} DDIM steps  '
          f'(class={dps_class}, ζ={zeta})', flush=True)

    tile_idx = 0
    for y0 in ys_list:
        for x0 in xs_list:
            tile_idx += 1
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            th = y1 - y0
            tw = x1 - x0

            y_tile = y_norm[:, y0:y1, x0:x1].unsqueeze(0)   # (1, 3, th, tw)

            # Pad to 512×512 when tile is smaller (edge tiles)
            ph, pw = tile - th, tile - tw
            if ph or pw:
                y_tile = F.pad(y_tile, (0, pw, 0, ph), mode='replicate')

            # Start from pure noise
            x_t = torch.randn(1, 3, tile, tile, device=device)

            print(f'    tile {tile_idx}/{total_tiles}  '
                  f'({th}×{tw} @ row={y0} col={x0})', flush=True)

            for step_i, idx in enumerate(indices):
                t_batch = torch.tensor([idx], device=device)
                x_t = _dps_ddim_step(
                    x_t, t_batch, model, diffusion,
                    y_tile, L, angle_deg, model_kwargs, zeta)
                if (step_i + 1) % 10 == 0 or step_i == len(indices) - 1:
                    print(f'      step {step_i+1}/{len(indices)}', flush=True)

            # Unpad
            x_out = x_t.squeeze(0)[:, :th, :tw]              # (3, th, tw) in [-1, 1]
            x_out_255 = (x_out + 1.0) * 127.5

            # Cosine blend weights
            wy = _cos_weight_1d(th).view(1, th, 1)
            wx = _cos_weight_1d(tw).view(1, 1, tw)
            w  = wy * wx                                      # (1, th, tw)

            out_sum   [:, y0:y1, x0:x1] += x_out_255 * w
            out_weight[0, y0:y1, x0:x1] += w.squeeze(0)

    return (out_sum / out_weight.clamp(min=1e-6)).clamp(0, 255)


# ── Blur kernel ───────────────────────────────────────────────────────────────

def make_kernel_fft(L, angle_deg, shape, device):
    """
    Fourier-domain representation of a 1-D box of continuous length L
    along angle_deg from horizontal.

    K(fx, fy) = sinc(L · (fx·cosθ + fy·sinθ))

    This is the exact DFT of a centred box; supports non-integer L and
    is differentiable w.r.t. L (used by the blind cost function).

    Returns (K, K*, |K|²) — all real tensors since the centred box is even.
    """
    H, W = shape
    fy = torch.fft.fftfreq(H, device=device).view(-1, 1)   # (H, 1)
    fx = torch.fft.fftfreq(W, device=device).view(1, -1)   # (1, W)
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    f_proj = fx * cos_a + fy * sin_a                        # (H, W)
    K = torch.sinc(L * f_proj)   # normalised sinc; real since centred box is even
    Kn2 = K ** 2
    return K, K, Kn2             # (K, K_conj=K, |K|²) — conj = self for real K


# ── ADMM building blocks ──────────────────────────────────────────────────────

def x_step(y_fft, K, Kc, Kn2, z, u, rho):
    """Wiener / proximal data-fidelity step (Fourier domain)."""
    num = Kc * y_fft + rho * torch.fft.fft2(z - u)
    return torch.fft.ifft2(num / (Kn2 + rho)).real


def wiener_deconvolve_tensor(y, Kc, Kn2, eps):
    """Direct Tikhonov/Wiener inverse in Fourier domain."""
    y_fft = torch.fft.fft2(y)
    return torch.fft.ifft2(Kc * y_fft / (Kn2 + eps)).real


def image_to_tensor(img_np):
    if img_np.ndim == 2:
        return torch.from_numpy(img_np.astype(np.float32)).to(DEVICE), False
    return torch.from_numpy(np.moveaxis(img_np.astype(np.float32), -1, 0)).to(DEVICE), True


def tensor_to_image(x, has_channels):
    arr = x.clamp(0, 255).cpu().numpy().astype(np.uint8)
    return np.moveaxis(arr, 0, -1) if has_channels else arr


def wiener_deconvolve_np(img_np, Kc, Kn2, eps):
    y, has_channels = image_to_tensor(img_np)
    x = wiener_deconvolve_tensor(y, Kc, Kn2, eps)
    return tensor_to_image(x, has_channels)


def tv_prox(v, lam, n_iter=30, tv_power=0.7):
    """
    Isotropic TV proximal operator — Chambolle (2004) dual.
    argmin_z  lam · TV(z) + ½||z − v||²
    tv_power controls the Lp norm of the gradient (default 0.7 for sparsity).
    """
    p = torch.zeros(2, *v.shape, device=v.device)
    tau = 0.249

    def div_p(p):
        d = torch.zeros_like(v)
        d[:-1, :] += p[0, :-1, :];  d[1:,  :] -= p[0, :-1, :]
        d[:, :-1] += p[1, :, :-1];  d[:, 1:]  -= p[1, :, :-1]
        return d

    def grad_u(u):
        gx = torch.zeros_like(u);  gy = torch.zeros_like(u)
        gx[:-1, :] = u[1:, :] - u[:-1, :]
        gy[:, :-1] = u[:, 1:] - u[:, :-1]
        return torch.stack([gx, gy])

    for _ in range(n_iter):
        g = grad_u(div_p(p) - v / lam)
        norm_g = (g[0].abs() ** tv_power + g[1].abs() ** tv_power).pow(1.0 / tv_power).clamp(min=1e-8)
        p = (p + tau * g) / (1 + tau * norm_g.unsqueeze(0))

    return v - lam * div_p(p)


def dncnn_denoise_tiled(v, model, tile=512, overlap=32):
    """
    DnCNN inference on arbitrary-size image via overlapping tiles.
    v: (H, W) float tensor in [0, 255].
    """
    H, W = v.shape
    out = torch.zeros_like(v)
    weight = torch.zeros_like(v)
    step = tile - 2 * overlap
    ys = sorted(set(list(range(0, H - tile + 1, step)) + [max(0, H - tile)]))
    xs = sorted(set(list(range(0, W - tile + 1, step)) + [max(0, W - tile)]))
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                p = v[y0:y1, x0:x1].unsqueeze(0).unsqueeze(0) / 255.0
                q = model(p).squeeze().clamp(0, 1) * 255.0
                out[y0:y1, x0:x1] += q
                weight[y0:y1, x0:x1] += 1.0
    return out / weight.clamp(min=1)


def drunet_denoise_tiled(v, model, sigma=25.0, tile=512, overlap=32):
    """
    DRUNet color inference on a CHW RGB tensor in [0, 255].
    sigma is in pixel units, e.g. 25 means a 25/255 noise-level map.
    """
    C, H, W = v.shape
    if C != 3:
        raise ValueError('drunet_color expects an RGB tensor with 3 channels')

    out = torch.zeros_like(v)
    weight = torch.zeros_like(v)
    step = tile - 2 * overlap
    ys = sorted(set(list(range(0, H - tile + 1, step)) + [max(0, H - tile)]))
    xs = sorted(set(list(range(0, W - tile + 1, step)) + [max(0, W - tile)]))

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                patch = v[:, y0:y1, x0:x1].unsqueeze(0).clamp(0, 255) / 255.0
                h, w = patch.shape[-2:]
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8
                if pad_h or pad_w:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')
                noise = torch.full((1, 1, patch.shape[-2], patch.shape[-1]),
                                   sigma / 255.0, device=v.device)
                q = model(torch.cat([patch, noise], dim=1))
                q = q[:, :, :h, :w].squeeze(0).clamp(0, 1) * 255.0
                out[:, y0:y1, x0:x1] += q
                weight[:, y0:y1, x0:x1] += 1.0
    return out / weight.clamp(min=1)


def z_step(v, prior, lam, rho, tv_inner, model):
    if prior == 'tv':
        return tv_prox(v, lam / rho, n_iter=tv_inner)
    elif prior == 'dncnn':
        return dncnn_denoise_tiled(v.clamp(0, 255), model)
    raise ValueError(prior)


# ── Blind kernel-length optimisation ─────────────────────────────────────────

def l_data_cost(L_val, y_fft, x_fft, angle_deg):
    """
    ||y − K_L * x||²_F in Fourier domain.
    K_L built analytically — only sinc evaluations, no FFT.
    """
    K, _, _ = make_kernel_fft(float(L_val), angle_deg, y_fft.shape, y_fft.device)
    R = y_fft - K * x_fft
    return (R.real ** 2 + R.imag ** 2).sum().item()


def optimize_L(L_curr, L_lo, L_hi, y_fft, x_fft, angle_deg, shrink,
               L_reg_center=None, L_reg_mu=0.0):
    """
    Golden-section search for L in [L_lo, L_hi], then shrink the bracket.

    Optional Tikhonov regularisation on L:
        cost(L) = data_cost(L) + L_reg_mu * (L − L_reg_center)²

    This resists the DnCNN-smoothing bias that drives L to smaller values:
    as the denoiser progressively over-smooths x, data_cost alone would
    keep decreasing with smaller L; the quadratic anchor prevents drift.

    Returns (L_new, L_lo_new, L_hi_new, cost_before, cost_after).
    """
    def cost_fn(L):
        c = l_data_cost(L, y_fft, x_fft, angle_deg)
        if L_reg_mu > 0 and L_reg_center is not None:
            c += L_reg_mu * (L - L_reg_center) ** 2
        return c

    cost_before = cost_fn(L_curr)
    res = minimize_scalar(cost_fn, bounds=(L_lo, L_hi), method='bounded',
                          options={'xatol': 0.1})
    L_new    = float(res.x)
    cost_after = float(res.fun)

    half     = (L_hi - L_lo) * shrink / 2.0
    L_lo_new = max(1.0, L_new - half)
    L_hi_new = L_new + half

    return L_new, L_lo_new, L_hi_new, cost_before, cost_after


# ── Single-channel solvers ────────────────────────────────────────────────────

def admm_channel(y_ch, K, Kc, Kn2, prior, lam, rho, n_iter, tv_inner, model,
                 x_init=None, z_init=None, u_init=None):
    """
    Run n_iter ADMM steps for one channel.  Supports warm-start via *_init.
    Returns (x, z, u) tensors (on DEVICE) for warm-starting next call.
    """
    y = torch.from_numpy(y_ch).float().to(DEVICE) if isinstance(y_ch, np.ndarray) else y_ch
    y_fft = torch.fft.fft2(y)
    x = y.clone() if x_init is None else x_init
    z = y.clone() if z_init is None else z_init
    u = torch.zeros_like(y) if u_init is None else u_init

    for i in range(n_iter):
        x = x_step(y_fft, K, Kc, Kn2, z, u, rho)
        z = z_step(x + u, prior, lam, rho, tv_inner, model)
        u = u + x - z

    return x, z, u


def nonblind_channel(y_ch, K, Kc, Kn2, prior, lam, rho, n_iter, tv_inner, model,
                     log_every=10):
    x, z, u = None, None, None
    done = 0
    for _ in range(0, n_iter, log_every):
        steps = min(log_every, n_iter - done)
        x, z, u = admm_channel(y_ch, K, Kc, Kn2, prior, lam, rho,
                                steps, tv_inner, model, x, z, u)
        done += steps
        resid = (x - z).norm().item()
        print(f"    iter {done:4d}  resid={resid:.3f}", flush=True)
    return x.clamp(0, 255).cpu().numpy().astype(np.uint8)


def admm_image(y_img, K, Kc, Kn2, prior, lam, rho, n_iter, tv_inner, model,
               drunet_sigma=25.0, x_init=None, z_init=None, u_init=None):
    """
    Run ADMM on an RGB image tensor for color priors such as DRUNet.
    y_img is HWC numpy or CHW torch.
    """
    if isinstance(y_img, np.ndarray):
        y = torch.from_numpy(np.moveaxis(y_img.astype(np.float32), -1, 0)).to(DEVICE)
    else:
        y = y_img
    y_fft = torch.fft.fft2(y)
    x = y.clone() if x_init is None else x_init
    z = y.clone() if z_init is None else z_init
    u = torch.zeros_like(y) if u_init is None else u_init

    for _ in range(n_iter):
        x = x_step(y_fft, K, Kc, Kn2, z, u, rho)
        if prior == 'drunet_color':
            z = drunet_denoise_tiled(x + u, model, sigma=drunet_sigma)
        else:
            raise ValueError(prior)
        u = u + x - z

    return x, z, u


def nonblind_image(y_img, K, Kc, Kn2, prior, lam, rho, n_iter, tv_inner, model,
                   drunet_sigma=25.0, log_every=10):
    x, z, u = None, None, None
    done = 0
    for _ in range(0, n_iter, log_every):
        steps = min(log_every, n_iter - done)
        x, z, u = admm_image(y_img, K, Kc, Kn2, prior, lam, rho,
                             steps, tv_inner, model, drunet_sigma, x, z, u)
        done += steps
        resid = (x - z).norm().item()
        print(f"    iter {done:4d}  resid={resid:.3f}", flush=True)
    return np.moveaxis(x.clamp(0, 255).cpu().numpy().astype(np.uint8), 0, -1)




# ── Color wrappers ────────────────────────────────────────────────────────────

def run_nonblind(img_np, K, Kc, Kn2, prior, lam, rho, n_iter, tv_inner, model,
                 channel_names='RGB', drunet_sigma=25.0, wiener_eps=0.01,
                 dps_kwargs=None):
    if prior == 'wiener':
        print(f"  Wiener filtering only (eps={wiener_eps:g})", flush=True)
        return wiener_deconvolve_np(img_np, Kc, Kn2, wiener_eps)

    if prior == 'dps':
        # model is (dps_model, diffusion); dps_kwargs carries L, angle, etc.
        dps_model, diffusion = model
        kw = dps_kwargs or {}
        y = torch.from_numpy(
            np.moveaxis(img_np.astype(np.float32), -1, 0)).to(DEVICE)
        x = dps_deblur_tiled(
            y, kw['L'], kw['angle_deg'], dps_model, diffusion,
            kw['n_steps'], kw['zeta'], kw['dps_class'],
            tile=512, overlap=kw.get('overlap', 64), device=DEVICE)
        return np.moveaxis(x.cpu().numpy().astype(np.uint8), 0, -1)

    if prior == 'drunet_color':
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise ValueError('drunet_color requires RGB input; use --deblur_space rgb')
        print("  Channels RGB:", flush=True)
        return nonblind_image(img_np, K, Kc, Kn2, prior, lam, rho, n_iter,
                              tv_inner, model, drunet_sigma)

    out = np.zeros_like(img_np)
    n_channels = img_np.shape[2] if img_np.ndim == 3 else 1
    for c in range(n_channels):
        ch = channel_names[c] if c < len(channel_names) else str(c)
        print(f"  Channel {ch}:", flush=True)
        y_ch = img_np if img_np.ndim == 2 else img_np[..., c]
        x_ch = nonblind_channel(
            y_ch.astype(np.float32),
            K, Kc, Kn2, prior, lam, rho, n_iter, tv_inner, model)
        if img_np.ndim == 2:
            out = x_ch
        else:
            out[..., c] = x_ch
    return out


def run_blind_color(img_np, L_init, L_lo, L_hi, angle_deg, prior,
                    lam, rho, T, inner, tv_inner, model, shrink, warmup,
                    L_reg_mu=0.0, diag_plot=True,
                    diag_path='blind_cost_landscape.png',
                    drunet_sigma=25.0):
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError('drunet_color requires RGB input; use --deblur_space rgb')

    H, W = img_np.shape[:2]
    y = torch.from_numpy(np.moveaxis(img_np.astype(np.float32), -1, 0)).to(DEVICE)
    y_ffts = torch.fft.fft2(y)

    L = float(L_init)
    L_lo = float(L_lo);  L_hi = float(L_hi)
    state = (None, None, None)

    if warmup > 0:
        print(f"  Warm-up ({warmup} iters, L={L:.1f}) …", flush=True)
        K, Kc, Kn2 = make_kernel_fft(L, angle_deg, (H, W), DEVICE)
        state = admm_image(y, K, Kc, Kn2, prior, lam, rho, warmup,
                           tv_inner, model, drunet_sigma, *state)
        resid = (state[0] - state[1]).norm().item()
        print(f"  Warm-up done  resid={resid:.2f}", flush=True)

    if diag_plot:
        Ls_probe = np.linspace(max(1, L_lo * 0.5), L_hi * 1.3, 80)
        x_now = state[0] if state[0] is not None else y
        x_ffts_now = torch.fft.fft2(x_now)

        def joint_cost_np(Lv):
            return sum(l_data_cost(Lv, y_ffts[c], x_ffts_now[c], angle_deg)
                       for c in range(3))

        costs_probe = [joint_cost_np(Lv) for Lv in Ls_probe]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(Ls_probe, costs_probe)
        ax.axvline(L, color='r', linestyle='--', label=f'L_init={L:.0f}')
        ax.set_xlabel('L (px)');  ax.set_ylabel('Joint data cost')
        ax.set_title('Cost landscape after warm-up')
        ax.legend();  plt.tight_layout()
        plt.savefig(diag_path, dpi=100)
        plt.close()
        print(f"  Saved: {diag_path}", flush=True)

    L_hist = [L]

    for outer in range(T):
        K, Kc, Kn2 = make_kernel_fft(L, angle_deg, (H, W), DEVICE)
        state = admm_image(y, K, Kc, Kn2, prior, lam, rho, inner,
                           tv_inner, model, drunet_sigma, *state)

        x_ffts = torch.fft.fft2(state[0])

        def full_cost(Lv):
            c = sum(l_data_cost(Lv, y_ffts[ch], x_ffts[ch], angle_deg)
                    for ch in range(3))
            if L_reg_mu > 0:
                c += L_reg_mu * (Lv - L_init) ** 2
            return c

        c_before = full_cost(L)
        res = minimize_scalar(full_cost, bounds=(L_lo, L_hi),
                              method='bounded', options={'xatol': 0.1})
        L = float(res.x)
        c_after = float(res.fun)

        half = (L_hi - L_lo) * shrink / 2.0
        L_lo = max(1.0, L - half)
        L_hi = L + half
        L_hist.append(L)

        resid = (state[0] - state[1]).norm().item()
        print(f"  outer {outer+1:3d}/{T}  L={L:6.2f}  "
              f"[{L_lo:.1f},{L_hi:.1f}]  "
              f"cost {c_before:.3e}→{c_after:.3e}  "
              f"resid={resid:.2f}", flush=True)

    out = np.moveaxis(state[0].clamp(0, 255).cpu().numpy().astype(np.uint8), 0, -1)
    print(f"\n  L_final = {L:.2f} px")
    return out, L, [L_hist]


def run_blind_wiener(img_np, L_init, L_lo, L_hi, angle_deg, eps, T, shrink,
                     L_reg_mu=0.0, diag_plot=True,
                     diag_path='blind_cost_landscape.png'):
    """
    Blind alternation with no image prior: Wiener x-update, then L search.
    """
    H, W = img_np.shape[:2]
    n_channels = img_np.shape[2] if img_np.ndim == 3 else 1
    ys = []
    for c in range(n_channels):
        y_ch = img_np if img_np.ndim == 2 else img_np[..., c]
        ys.append(torch.from_numpy(y_ch.astype(np.float32)).to(DEVICE))
    y_ffts = [torch.fft.fft2(y) for y in ys]

    L = float(L_init)
    L_lo = float(L_lo);  L_hi = float(L_hi)

    if diag_plot:
        K, Kc, Kn2 = make_kernel_fft(L, angle_deg, (H, W), DEVICE)
        xs_now = [wiener_deconvolve_tensor(y, Kc, Kn2, eps) for y in ys]
        x_ffts_now = [torch.fft.fft2(x) for x in xs_now]
        Ls_probe = np.linspace(max(1, L_lo * 0.5), L_hi * 1.3, 80)

        def joint_cost_np(Lv):
            return sum(l_data_cost(Lv, y_ffts[c], x_ffts_now[c], angle_deg)
                       for c in range(n_channels))

        costs_probe = [joint_cost_np(Lv) for Lv in Ls_probe]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(Ls_probe, costs_probe)
        ax.axvline(L, color='r', linestyle='--', label=f'L_init={L:.0f}')
        ax.set_xlabel('L (px)');  ax.set_ylabel('Joint data cost')
        ax.set_title('Cost landscape after Wiener update')
        ax.legend();  plt.tight_layout()
        plt.savefig(diag_path, dpi=100)
        plt.close()
        print(f"  Saved: {diag_path}", flush=True)

    L_hist = [L]

    for outer in range(T):
        K, Kc, Kn2 = make_kernel_fft(L, angle_deg, (H, W), DEVICE)
        xs = [wiener_deconvolve_tensor(y, Kc, Kn2, eps) for y in ys]
        x_ffts = [torch.fft.fft2(x) for x in xs]

        def full_cost(Lv):
            c = sum(l_data_cost(Lv, y_ffts[ch], x_ffts[ch], angle_deg)
                    for ch in range(n_channels))
            if L_reg_mu > 0:
                c += L_reg_mu * (Lv - L_init) ** 2
            return c

        c_before = full_cost(L)
        res = minimize_scalar(full_cost, bounds=(L_lo, L_hi),
                              method='bounded', options={'xatol': 0.1})
        L = float(res.x)
        c_after = float(res.fun)

        half = (L_hi - L_lo) * shrink / 2.0
        L_lo = max(1.0, L - half)
        L_hi = L + half
        L_hist.append(L)

        print(f"  outer {outer+1:3d}/{T}  L={L:6.2f}  "
              f"[{L_lo:.1f},{L_hi:.1f}]  "
              f"cost {c_before:.3e}→{c_after:.3e}", flush=True)

    K, Kc, Kn2 = make_kernel_fft(L, angle_deg, (H, W), DEVICE)
    xs = [wiener_deconvolve_tensor(y, Kc, Kn2, eps) for y in ys]
    out = np.zeros_like(img_np)
    for c, x in enumerate(xs):
        x_ch = x.clamp(0, 255).cpu().numpy().astype(np.uint8)
        if img_np.ndim == 2:
            out = x_ch
        else:
            out[..., c] = x_ch

    print(f"\n  L_final = {L:.2f} px")
    return out, L, [L_hist]


def run_blind(img_np, L_init, L_lo, L_hi, angle_deg, prior,
              lam, rho, T, inner, tv_inner, model, shrink, warmup,
              L_reg_mu=0.0, diag_plot=True,
              diag_path='blind_cost_landscape.png',
              channel_names='RGB', drunet_sigma=25.0,
              wiener_eps=0.01):
    if prior == 'wiener':
        return run_blind_wiener(
            img_np, L_init, L_lo, L_hi, angle_deg, wiener_eps, T, shrink,
            L_reg_mu, diag_plot, diag_path)

    if prior == 'drunet_color':
        return run_blind_color(
            img_np, L_init, L_lo, L_hi, angle_deg, prior, lam, rho, T, inner,
            tv_inner, model, shrink, warmup, L_reg_mu, diag_plot, diag_path,
            drunet_sigma)

    """
    Alternating optimisation with a SHARED L across all colour channels.

    The joint cost  f(L) = Σ_c ||Y_c − K_L · X_c||²  pools all three
    channels, making the 1-D search more stable and ensuring a single
    physically meaningful blur length.
    """
    H, W = img_np.shape[:2]
    n_channels = img_np.shape[2] if img_np.ndim == 3 else 1
    ys = []
    for c in range(n_channels):
        y_ch = img_np if img_np.ndim == 2 else img_np[..., c]
        ys.append(torch.from_numpy(y_ch.astype(np.float32)).to(DEVICE))
    y_ffts = [torch.fft.fft2(y) for y in ys]

    L    = float(L_init)
    L_lo = float(L_lo);  L_hi = float(L_hi)
    states = [(None, None, None)] * n_channels   # (x, z, u) per channel

    # ── Warm-up: drive x toward the sharp image with fixed L ─────────────
    if warmup > 0:
        print(f"  Warm-up ({warmup} iters, L={L:.1f}) …", flush=True)
        K, Kc, Kn2 = make_kernel_fft(L, angle_deg, (H, W), DEVICE)
        for c in range(n_channels):
            ch = channel_names[c] if c < len(channel_names) else str(c)
            print(f"  Warm-up channel {ch}", flush=True)
            x, z, u = admm_channel(ys[c], K, Kc, Kn2, prior, lam, rho,
                                    warmup, tv_inner, model, *states[c])
            states[c] = (x, z, u)
        resid = sum((s[0] - s[1]).norm().item() for s in states) / n_channels
        print(f"  Warm-up done  avg_resid={resid:.2f}", flush=True)

    # ── Optional diagnostic: plot the joint cost curve after warm-up ──────
    if diag_plot:
        Ls_probe = np.linspace(max(1, L_lo * 0.5), L_hi * 1.3, 80)
        x_ffts_now = [torch.fft.fft2(s[0] if s[0] is not None else ys[c])
                      for c, s in enumerate(states)]

        def joint_cost_np(Lv):
            return sum(l_data_cost(Lv, y_ffts[c], x_ffts_now[c], angle_deg)
                       for c in range(n_channels))

        costs_probe = [joint_cost_np(Lv) for Lv in Ls_probe]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(Ls_probe, costs_probe)
        ax.axvline(L, color='r', linestyle='--', label=f'L_init={L:.0f}')
        ax.set_xlabel('L (px)');  ax.set_ylabel('Joint data cost')
        ax.set_title('Cost landscape after warm-up')
        ax.legend();  plt.tight_layout()
        plt.savefig(diag_path, dpi=100)
        plt.close()
        print(f"  Saved: {diag_path}", flush=True)

    # ── Alternating loop ──────────────────────────────────────────────────
    L_hist = [L]

    for outer in range(T):
        K, Kc, Kn2 = make_kernel_fft(L, angle_deg, (H, W), DEVICE)

        # x-update: inner ADMM steps, warm-started, same kernel for all channels
        for c in range(n_channels):
            x, z, u = admm_channel(ys[c], K, Kc, Kn2, prior, lam, rho,
                                    inner, tv_inner, model, *states[c])
            states[c] = (x, z, u)

        # L-update: 1-D golden-section, joint cost over all channels.
        # Optional L² regularisation anchors L near L_init, resisting the
        # DnCNN-smoothing bias that otherwise drives L toward smaller values.
        x_ffts = [torch.fft.fft2(s[0]) for s in states]

        def full_cost(Lv):
            c = sum(l_data_cost(Lv, y_ffts[ch], x_ffts[ch], angle_deg)
                    for ch in range(n_channels))
            if L_reg_mu > 0:
                c += L_reg_mu * (Lv - L_init) ** 2
            return c

        c_before = full_cost(L)
        res      = minimize_scalar(full_cost, bounds=(L_lo, L_hi),
                                   method='bounded', options={'xatol': 0.1})
        L        = float(res.x)
        c_after  = float(res.fun)

        # Shrink bracket around new L
        half  = (L_hi - L_lo) * shrink / 2.0
        L_lo  = max(1.0, L - half)
        L_hi  = L + half
        L_hist.append(L)

        resid = sum((s[0] - s[1]).norm().item() for s in states) / n_channels
        print(f"  outer {outer+1:3d}/{T}  L={L:6.2f}  "
              f"[{L_lo:.1f},{L_hi:.1f}]  "
              f"cost {c_before:.3e}→{c_after:.3e}  "
              f"resid={resid:.2f}", flush=True)

    out = np.zeros_like(img_np)
    for c in range(n_channels):
        x_ch = states[c][0].clamp(0, 255).cpu().numpy().astype(np.uint8)
        if img_np.ndim == 2:
            out = x_ch
        else:
            out[..., c] = x_ch

    print(f"\n  L_final = {L:.2f} px")
    return out, L, [L_hist]   # single L history shared across channels


# ── Visualisation ─────────────────────────────────────────────────────────────

def safe_name_part(text):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_'
                   for ch in text).strip('_')


def safe_value_part(value):
    return safe_name_part(f'{value:g}')


def make_deblur_input(img_rgb, deblur_space):
    if deblur_space == 'rgb':
        return img_rgb, None, 'RGB'
    if deblur_space == 'ycbcr_y':
        ycbcr_np = np.array(Image.fromarray(img_rgb).convert('YCbCr'),
                            dtype=np.uint8)
        return ycbcr_np[..., :1], ycbcr_np, 'Y'
    raise ValueError(deblur_space)


def restore_deblur_output(result_np, ycbcr_np, deblur_space):
    if deblur_space == 'rgb':
        return result_np
    if deblur_space == 'ycbcr_y':
        out_ycbcr = ycbcr_np.copy()
        out_ycbcr[..., 0] = result_np[..., 0] if result_np.ndim == 3 else result_np
        return np.array(Image.fromarray(out_ycbcr, mode='YCbCr').convert('RGB'),
                        dtype=np.uint8)
    raise ValueError(deblur_space)


def save_comparison(images, labels, path):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 10))
    if n == 1: axes = [axes]
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img, cmap=None)
        ax.set_title(label, fontsize=13)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def save_L_history(L_hists, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ['L'] if len(L_hists) == 1 else list('RGB')
    for hist, ch in zip(L_hists, labels):
        ax.plot(hist, label=ch, marker='o', markersize=4)
    ax.set_xlabel('Outer iteration')
    ax.set_ylabel('L (px)')
    ax.set_title('Blur kernel length optimisation')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()

    # Shared
    p.add_argument('--mode',     choices=['nonblind', 'blind'], default='nonblind')
    p.add_argument('--prior',    choices=['tv', 'dncnn', 'drunet_color', 'wiener', 'dps'],
                   default='dncnn')
    p.add_argument('--angle',    type=float, default=PAN_ANGLE_DEG)
    p.add_argument('--rho',      type=float, default=0.5)
    p.add_argument('--lam_tv',   type=float, default=0.05)
    p.add_argument('--lam_dn',   type=float, default=0.005)
    p.add_argument('--wiener_eps', type=float, default=0.01,
                   help='Fourier-domain Wiener/Tikhonov epsilon for --prior wiener')
    p.add_argument('--drunet_sigma', type=float, default=25.0)
    p.add_argument('--tv_inner', type=int,   default=30)
    p.add_argument('--image',    type=str,   default='full_pan_2.jpg')
    p.add_argument('--deblur_space', choices=['rgb', 'ycbcr_y'], default='rgb')
    p.add_argument('--crop',     type=int,   default=None)
    p.add_argument('--crop_row', type=int,   default=2000)
    p.add_argument('--crop_col', type=int,   default=None)
    p.add_argument('--out_tag',  type=str,   default='')

    # Non-blind only
    p.add_argument('--b',     type=float, default=B_PX)
    p.add_argument('--iters', type=int, default=80)

    # DPS only
    p.add_argument('--dps_ckpt',  type=str, default='512x512_diffusion.pt',
                   help='Path to OpenAI 512x512_diffusion.pt checkpoint')
    p.add_argument('--dps_steps', type=int, default=100,
                   help='Number of DDIM reverse-diffusion steps')
    p.add_argument('--dps_zeta',  type=float, default=1.0,
                   help='DPS likelihood-gradient step size')
    p.add_argument('--dps_class', type=int, default=-1,
                   help='ImageNet class label 0-999, or -1 (default) to '
                        'marginalise over all classes via mean embedding')

    # Blind only
    p.add_argument('--L_init',  type=float, default=float(B_PX))
    p.add_argument('--L_lo',    type=float, default=40.0)
    p.add_argument('--L_hi',    type=float, default=120.0)
    p.add_argument('--T',       type=int,   default=12,
                   help='Outer alternating iterations')
    p.add_argument('--inner',   type=int,   default=5,
                   help='ADMM steps per outer iteration')
    p.add_argument('--warmup',  type=int,   default=40,
                   help='ADMM steps with fixed L before alternating begins')
    p.add_argument('--shrink',   type=float, default=0.75,
                   help='Bracket half-width multiplier each outer iter')
    p.add_argument('--L_reg_mu', type=float, default=0.0,
                   help='Tikhonov weight anchoring L near L_init (0 = disabled)')

    args = p.parse_args()
    if args.prior == 'drunet_color' and args.deblur_space != 'rgb':
        p.error('--prior drunet_color requires --deblur_space rgb')
    if args.prior == 'dps' and args.deblur_space != 'rgb':
        p.error('--prior dps requires --deblur_space rgb')
    if args.prior == 'dps' and args.mode == 'blind':
        p.error('--prior dps is only supported with --mode nonblind')

    image_name = safe_name_part(Path(args.image).stem) or 'image'
    if args.prior == 'tv':
        lam = args.lam_tv
        reg_name = 'lam_tv'
        reg_value = lam
    elif args.prior == 'wiener':
        lam = 0.0
        reg_name = 'wiener_eps'
        reg_value = args.wiener_eps
    else:
        lam = args.lam_dn
        reg_name = 'lam_dn'
        reg_value = lam

    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    out_parts = []
    if args.out_tag:
        out_parts.append(safe_name_part(args.out_tag) or 'tag')
    out_parts.append(image_name)
    out_parts.append(args.deblur_space)
    out_parts.append(f'{reg_name}_{safe_value_part(reg_value)}')
    if args.prior == 'drunet_color':
        out_parts.append(f'drunet_sigma_{safe_value_part(args.drunet_sigma)}')
    if args.prior == 'dps':
        out_parts.append(f'steps_{args.dps_steps}')
        out_parts.append(f'zeta_{safe_value_part(args.dps_zeta)}')
    out_parts.append(f'b_{args.b}')
    out_parts.append(f'iters_{args.iters}')
    tag = '_'.join(out_parts) + '_'

    print(f"Device: {DEVICE}  |  mode={args.mode}  prior={args.prior}")

    # Load image
    img_pil = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    print(f"Input: {args.image}")
    img_np = np.array(img_pil, dtype=np.uint8)
    if args.crop:
        H, W = img_np.shape[:2]
        r0 = args.crop_row
        c0 = args.crop_col if args.crop_col is not None else (W - args.crop) // 2
        img_np = img_np[r0: min(H, r0 + args.crop), c0: min(W, c0 + args.crop)]
        print(f"Crop: {img_np.shape[1]}×{img_np.shape[0]} @ row={r0} col={c0}")
    print(f"Image: {img_np.shape[1]}×{img_np.shape[0]} px")
    print(f"Deblur space: {args.deblur_space}")

    deblur_np, ycbcr_np, channel_names = make_deblur_input(img_np, args.deblur_space)

    H, W = deblur_np.shape[:2]

    # Denoiser / diffusion prior (loaded once)
    model = None
    if args.prior == 'dncnn':
        model = load_dncnn('dncnn_25.pth', DEVICE)
    elif args.prior == 'drunet_color':
        model = load_drunet_color('drunet_color.pth', DEVICE)
    elif args.prior == 'dps':
        print(f'Loading DPS model from {args.dps_ckpt} '
              f'(DDIM {args.dps_steps} steps) …', flush=True)
        dps_model, dps_diffusion = load_openai_512_diffusion(
            args.dps_ckpt, DEVICE, ddim_steps=args.dps_steps,
            dps_class=args.dps_class)
        model = (dps_model, dps_diffusion)

    t0 = time.time()

    if args.mode == 'nonblind':
        if args.prior == 'wiener':
            print(f"\nb={args.b} px  wiener_eps={args.wiener_eps}")
        else:
            print(f"\nb={args.b} px  λ={lam}  ρ={args.rho}  {args.iters} iters")
        K, Kc, Kn2 = make_kernel_fft(args.b, args.angle, (H, W), DEVICE)
        dps_kw = dict(L=args.b, angle_deg=args.angle, n_steps=args.dps_steps,
                      zeta=args.dps_zeta, dps_class=args.dps_class) \
                 if args.prior == 'dps' else None
        result_work = run_nonblind(deblur_np, K, Kc, Kn2, args.prior,
                                   lam, args.rho, args.iters, args.tv_inner,
                                   model, channel_names, args.drunet_sigma,
                                   args.wiener_eps, dps_kwargs=dps_kw)
        result = restore_deblur_output(result_work, ycbcr_np, args.deblur_space)
        out_path = output_dir / f'{tag}deblur_{args.prior}.png'
        Image.fromarray(result).save(out_path)
        print(f"Saved: {out_path}")
        if args.prior == 'wiener':
            method_label = f'Wiener b={args.b}px'
        elif args.prior == 'dps':
            method_label = (f'DPS b={args.b}px '
                            f'steps={args.dps_steps} ζ={args.dps_zeta}')
        else:
            method_label = f'ADMM-{args.prior.upper()} b={args.b}px'
        save_comparison([img_np, result],
                        ['Blurred (input)', method_label],
                        output_dir / f'{tag}comparison_{args.prior}.png')

    elif args.mode == 'blind':
        print(f"\nL_init={args.L_init}  range=[{args.L_lo},{args.L_hi}]  "
              f"T={args.T}  inner={args.inner}  shrink={args.shrink}")
        if args.prior == 'wiener':
            print(f"wiener_eps={args.wiener_eps}")
        else:
            print(f"λ={lam}  ρ={args.rho}")
        result, L_final, L_hists = run_blind(
            deblur_np, args.L_init, args.L_lo, args.L_hi,
            args.angle, args.prior, lam, args.rho,
            args.T, args.inner, args.tv_inner, model, args.shrink, args.warmup,
            L_reg_mu=args.L_reg_mu,
            diag_path=output_dir / f'{tag}blind_cost_landscape.png',
            channel_names=channel_names,
            drunet_sigma=args.drunet_sigma,
            wiener_eps=args.wiener_eps)
        result = restore_deblur_output(result, ycbcr_np, args.deblur_space)
        out_path = output_dir / f'{tag}deblur_blind_{args.prior}.png'
        Image.fromarray(result).save(out_path)
        print(f"Saved: {out_path}  (L={L_final:.2f}px)")
        method_label = (f'Blind Wiener L={L_final:.1f}px'
                        if args.prior == 'wiener'
                        else f'Blind ADMM-{args.prior.upper()} L={L_final:.1f}px')
        save_comparison(
            [img_np, result],
            ['Blurred (input)', method_label],
            output_dir / f'{tag}comparison_blind_{args.prior}.png')
        save_L_history(L_hists, output_dir / f'{tag}L_history.png')

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
