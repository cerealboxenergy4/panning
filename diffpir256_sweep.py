#!/usr/bin/env python3
"""Run DiffPIR with the official 256 ImageNet unconditional checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from deblur import make_kernel_fft
from globant_patch_deblur import crop_rgb, edge_energy, mae, psnr


def save_grid(panels, ref, path: Path):
    ncols = 3
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (label, img) in zip(axes, panels):
        ax.imshow(img)
        if label not in {"blurred", "sharp_registered"}:
            ax.set_title(f"{label}\nPSNR {psnr(img, ref):.2f} dB  MAE {mae(img, ref):.1f}", fontsize=9)
        else:
            ax.set_title(label, fontsize=9)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def tile_starts(length: int, tile: int, overlap: int):
    step = tile - 2 * overlap
    if step <= 0:
        raise ValueError("--overlap must be less than half of --tile")
    return sorted(set(list(range(0, max(length - tile + 1, 1), step)) + [max(0, length - tile)]))


def hann_weight(h: int, w: int, device):
    wy = torch.hann_window(h, periodic=False, device=device).clamp(min=1e-3).view(1, h, 1)
    wx = torch.hann_window(w, periodic=False, device=device).clamp(min=1e-3).view(1, 1, w)
    return wy * wx


def load_diffpir256(repo: Path, ckpt: Path, steps: int, device):
    sys.path.insert(0, str(repo.resolve()))
    from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
    from utils import utils_model

    model_config = dict(
        model_path=str(ckpt),
        num_channels=256,
        num_res_blocks=2,
        attention_resolutions="8,16,32",
        timestep_respacing=f"ddim{steps}",
    )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, diffusion


def data_solution_fft(z_255, y_255, b: float, angle: float, rho: float):
    _, _, h, w = z_255.shape
    K, _, K2 = make_kernel_fft(b, angle, (h, w), z_255.device)
    Y = torch.fft.fft2(y_255)
    Z = torch.fft.fft2(z_255)
    X = (K * Y + rho * Z) / (K2 + rho)
    return torch.fft.ifft2(X).real.clamp(0, 255)


@torch.no_grad()
def init_from_measurement(y_norm, alpha: float):
    return math.sqrt(alpha) * y_norm + math.sqrt(max(1.0 - alpha, 0.0)) * torch.randn_like(y_norm)


def diffpir_tile(y_tile_255, b, angle, model, diffusion, lambda_, zeta, rho_floor, device):
    y_norm = (y_tile_255 / 127.5) - 1.0
    indices = list(range(diffusion.num_timesteps))[::-1]
    alphas = torch.as_tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)
    x_t = init_from_measurement(y_norm, float(alphas[indices[0]]))

    for step_i, idx in enumerate(indices):
        t_batch = torch.tensor([idx], device=device)
        out = diffusion.ddim_sample(model, x_t, t_batch, clip_denoised=True, model_kwargs={})
        x0_hat = out["pred_xstart"].detach()
        alpha_t = float(alphas[idx])
        sigma_k = math.sqrt(max((1.0 - alpha_t) / max(alpha_t, 1e-12), 1e-12))
        rho = max(float(lambda_) / (sigma_k * sigma_k + 1e-12), rho_floor)
        z_255 = (x0_hat + 1.0) * 127.5
        x0_p_255 = data_solution_fft(z_255, y_tile_255, b, angle, rho)
        x0_p = (x0_p_255 / 127.5) - 1.0
        if step_i == len(indices) - 1:
            x_t = x0_p.clamp(-1, 1).detach()
            continue
        next_idx = indices[step_i + 1]
        alpha_next = float(alphas[next_idx])
        eps = (x_t - math.sqrt(alpha_t) * x0_hat) / math.sqrt(max(1.0 - alpha_t, 1e-12))
        noise = torch.randn_like(x_t)
        deterministic = math.sqrt(max(1.0 - zeta, 0.0)) * math.sqrt(max(1.0 - alpha_next, 0.0)) * eps
        stochastic = math.sqrt(max(zeta, 0.0)) * math.sqrt(max(1.0 - alpha_next, 0.0)) * noise
        x_t = (math.sqrt(alpha_next) * x0_p + deterministic + stochastic).clamp(-2, 2).detach()
        if (step_i + 1) % 10 == 0 or step_i == len(indices) - 1:
            print(f"      step {step_i + 1}/{len(indices)} rho={rho:.3g}", flush=True)
    return ((x_t + 1.0) * 127.5).clamp(0, 255)


def run_tiled(img, b, angle, model, diffusion, steps, lambda_, zeta, rho_floor, tile, overlap, device):
    y_img = torch.from_numpy(np.moveaxis(img.astype(np.float32), -1, 0)).to(device)
    _, h, w = y_img.shape
    ys = tile_starts(h, tile, overlap)
    xs = tile_starts(w, tile, overlap)
    out_sum = torch.zeros_like(y_img)
    out_weight = torch.zeros(1, h, w, device=device)
    total = len(ys) * len(xs)
    print(f"  official DiffPIR256: {total} tile(s) x {steps} steps lambda={lambda_} zeta={zeta}", flush=True)
    idx = 0
    for y0 in ys:
        for x0 in xs:
            idx += 1
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            if y1 - y0 != tile or x1 - x0 != tile:
                raise ValueError("This quick runner expects crop dimensions compatible with full tiles.")
            print(f"    tile {idx}/{total} ({tile}x{tile} @ row={y0} col={x0})", flush=True)
            x_tile = diffpir_tile(y_img[:, y0:y1, x0:x1].unsqueeze(0), b, angle, model, diffusion, lambda_, zeta, rho_floor, device).squeeze(0)
            weight = hann_weight(tile, tile, device)
            out_sum[:, y0:y1, x0:x1] += x_tile * weight
            out_weight[:, y0:y1, x0:x1] += weight
    out = (out_sum / out_weight.clamp(min=1e-6)).clamp(0, 255)
    return np.moveaxis(out.cpu().numpy().astype(np.uint8), 0, -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blurry", default="pan_1.jpg")
    p.add_argument("--sharp_reg", default="outputs_tttnvs_probe/sharp_registered.png")
    p.add_argument("--out_dir", default="outputs_globant_deblur_512_diffpir256")
    p.add_argument("--repo", default="third_party/DiffPIR")
    p.add_argument("--ckpt", default="third_party/DiffPIR/model_zoo/256x256_diffusion_uncond.pt")
    p.add_argument("--x0", type=int, default=2050)
    p.add_argument("--y0", type=int, default=2500)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--b", type=float, default=128.14)
    p.add_argument("--angle", type=float, default=0.25)
    p.add_argument("--steps", type=int, nargs="+", default=[25])
    p.add_argument("--lambdas", type=float, nargs="+", default=[30.0, 100.0])
    p.add_argument("--zeta", type=float, default=0.1)
    p.add_argument("--rho_floor", type=float, default=0.001)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Crop: x={args.x0} y={args.y0} size={args.size}")
    blur = crop_rgb(args.blurry, args.x0, args.y0, args.size)
    ref = crop_rgb(args.sharp_reg, args.x0, args.y0, args.size)
    Image.fromarray(blur).save(out_dir / "globant_blurry.png")
    Image.fromarray(ref).save(out_dir / "globant_sharp_registered.png")

    metadata = {
        "method": "official_diffpir_256_uncond",
        "crop": {"x0": args.x0, "y0": args.y0, "size": args.size},
        "b": args.b,
        "angle_deg": args.angle,
        "tile": args.tile,
        "overlap": args.overlap,
        "zeta": args.zeta,
        "rho_floor": args.rho_floor,
        "seed": args.seed,
        "ckpt": args.ckpt,
        "outputs": {},
    }
    panels = [("blurred", blur), ("sharp_registered", ref)]

    for steps in args.steps:
        print(f"Loading official DiffPIR 256 prior for {steps} steps", flush=True)
        model, diffusion = load_diffpir256(Path(args.repo), Path(args.ckpt), steps, device)
        for lambda_ in args.lambdas:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
                torch.cuda.reset_peak_memory_stats()
            label = f"diffpir256_s{steps}_lam{lambda_:g}"
            print(f"\nRunning {label}", flush=True)
            t0 = time.time()
            try:
                out = run_tiled(blur, args.b, args.angle, model, diffusion, steps, lambda_, args.zeta, args.rho_floor, args.tile, args.overlap, device)
                elapsed = time.time() - t0
                path = out_dir / f"globant_diffpir256_b{args.b:.1f}_steps{steps}_lambda{lambda_:g}.png"
                Image.fromarray(out).save(path)
                peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
                metadata["outputs"][path.name] = {
                    "steps": steps,
                    "lambda": lambda_,
                    "psnr": psnr(out, ref),
                    "mae": mae(out, ref),
                    "edge_energy": edge_energy(out),
                    "runtime_seconds": elapsed,
                    "peak_memory_mb": peak,
                }
                panels.append((label, out))
                print(f"  saved {path}", flush=True)
                print(f"  PSNR {metadata['outputs'][path.name]['psnr']:.4f} dB, runtime {elapsed:.1f}s, peak {peak:.1f} MB", flush=True)
            except Exception as exc:
                metadata["outputs"][label] = {"error": repr(exc), "steps": steps, "lambda": lambda_}
                print(f"  failed: {exc}", flush=True)
            with (out_dir / "diffpir256_metrics.json").open("w") as f:
                json.dump(metadata, f, indent=2)
        del model, diffusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = out_dir / "diffpir256_comparison.png"
    save_grid(panels, ref, comparison)
    metadata["comparison"] = str(comparison)
    with (out_dir / "diffpir256_metrics.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved comparison: {comparison}")
    print(f"Saved metrics: {out_dir / 'diffpir256_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
