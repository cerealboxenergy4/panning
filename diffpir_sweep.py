#!/usr/bin/env python3
"""Run a DiffPIR-style plug-and-play deblur sweep on the Globant crop."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from deblur import load_openai_512_diffusion, make_kernel_fft
from globant_patch_deblur import crop_rgb, edge_energy, mae, psnr


def save_grid(panels: list[tuple[str, np.ndarray]], ref: np.ndarray, path: Path) -> None:
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


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    step = tile - 2 * overlap
    if step <= 0:
        raise ValueError("--overlap must be less than half of --tile")
    return sorted(set(list(range(0, max(length - tile + 1, 1), step)) + [max(0, length - tile)]))


def hann_weight(h: int, w: int, device: torch.device) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False, device=device).clamp(min=1e-3).view(1, h, 1)
    wx = torch.hann_window(w, periodic=False, device=device).clamp(min=1e-3).view(1, 1, w)
    return wy * wx


def data_solution_fft(
    z_255: torch.Tensor,
    y_255: torch.Tensor,
    b: float,
    angle: float,
    rho: float,
) -> torch.Tensor:
    _, _, h, w = z_255.shape
    K, _, K2 = make_kernel_fft(b, angle, (h, w), z_255.device)
    Y = torch.fft.fft2(y_255)
    Z = torch.fft.fft2(z_255)
    X = (K * Y + rho * Z) / (K2 + rho)
    return torch.fft.ifft2(X).real.clamp(0, 255)


@torch.no_grad()
def init_from_measurement(y_norm: torch.Tensor, alpha: float) -> torch.Tensor:
    noise = torch.randn_like(y_norm)
    return math.sqrt(alpha) * y_norm + math.sqrt(max(1.0 - alpha, 0.0)) * noise


def diffpir_tile(
    y_tile_255: torch.Tensor,
    b: float,
    angle: float,
    model: torch.nn.Module,
    diffusion,
    dps_class: int,
    lambda_: float,
    zeta: float,
    rho_floor: float,
    device: torch.device,
) -> torch.Tensor:
    y_norm = (y_tile_255 / 127.5) - 1.0
    indices = list(range(diffusion.num_timesteps))[::-1]
    alphas = torch.as_tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)
    y_label = max(dps_class, 0)
    model_kwargs = {"y": torch.tensor([y_label], device=device)}

    x_t = init_from_measurement(y_norm, float(alphas[indices[0]]))
    for step_i, idx in enumerate(indices):
        t_batch = torch.tensor([idx], device=device)
        out = diffusion.ddim_sample(
            model, x_t, t_batch, clip_denoised=True, model_kwargs=model_kwargs)
        x0_hat = out["pred_xstart"].detach()

        sigma_k = math.sqrt(max((1.0 - float(alphas[idx])) / max(float(alphas[idx]), 1e-12), 1e-12))
        rho = max(float(lambda_) / (sigma_k * sigma_k + 1e-12), rho_floor)
        z_255 = (x0_hat + 1.0) * 127.5
        x0_p_255 = data_solution_fft(z_255, y_tile_255, b, angle, rho)
        x0_p = (x0_p_255 / 127.5) - 1.0

        if step_i == len(indices) - 1:
            x_t = x0_p.clamp(-1, 1).detach()
            continue

        next_idx = indices[step_i + 1]
        alpha_t = float(alphas[idx])
        alpha_next = float(alphas[next_idx])
        eps = (x_t - math.sqrt(alpha_t) * x0_hat) / math.sqrt(max(1.0 - alpha_t, 1e-12))
        noise = torch.randn_like(x_t)
        stochastic = math.sqrt(max(zeta, 0.0)) * math.sqrt(max(1.0 - alpha_next, 0.0)) * noise
        deterministic = math.sqrt(max(1.0 - zeta, 0.0)) * math.sqrt(max(1.0 - alpha_next, 0.0)) * eps
        x_t = (math.sqrt(alpha_next) * x0_p + deterministic + stochastic).clamp(-2, 2).detach()

        if (step_i + 1) % 10 == 0 or step_i == len(indices) - 1:
            print(f"      step {step_i + 1}/{len(indices)} rho={rho:.3g}", flush=True)

    return ((x_t + 1.0) * 127.5).clamp(0, 255)


def run_diffpir_tiled(
    img: np.ndarray,
    b: float,
    angle: float,
    ckpt: str,
    steps: int,
    lambda_: float,
    zeta: float,
    rho_floor: float,
    dps_class: int,
    tile: int,
    overlap: int,
    device: torch.device,
) -> np.ndarray:
    model, diffusion = load_openai_512_diffusion(
        ckpt, str(device), ddim_steps=steps, dps_class=dps_class)
    y_img = torch.from_numpy(np.moveaxis(img.astype(np.float32), -1, 0)).to(device)
    _, h, w = y_img.shape
    ys = tile_starts(h, tile, overlap)
    xs = tile_starts(w, tile, overlap)
    out_sum = torch.zeros_like(y_img)
    out_weight = torch.zeros(1, h, w, device=device)
    total = len(ys) * len(xs)
    tile_idx = 0
    print(f"  DiffPIR-style: {total} tile(s) x {steps} steps lambda={lambda_} zeta={zeta}", flush=True)
    for y0 in ys:
        for x0 in xs:
            tile_idx += 1
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            th, tw = y1 - y0, x1 - x0
            y_tile = y_img[:, y0:y1, x0:x1].unsqueeze(0)
            if th != tile or tw != tile:
                raise ValueError("Edge tiles smaller than tile are not supported in this quick runner.")
            print(f"    tile {tile_idx}/{total} ({th}x{tw} @ row={y0} col={x0})", flush=True)
            x_tile = diffpir_tile(
                y_tile, b, angle, model, diffusion, dps_class,
                lambda_, zeta, rho_floor, device,
            ).squeeze(0)
            weight = hann_weight(th, tw, device)
            out_sum[:, y0:y1, x0:x1] += x_tile * weight
            out_weight[:, y0:y1, x0:x1] += weight

    del model, diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = (out_sum / out_weight.clamp(min=1e-6)).clamp(0, 255)
    return np.moveaxis(out.cpu().numpy().astype(np.uint8), 0, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blurry", default="pan_1.jpg")
    parser.add_argument("--sharp_reg", default="outputs_tttnvs_probe/sharp_registered.png")
    parser.add_argument("--out_dir", default="outputs_globant_deblur_1024_diffpir")
    parser.add_argument("--x0", type=int, default=2050)
    parser.add_argument("--y0", type=int, default=2500)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--b", type=float, default=128.14)
    parser.add_argument("--angle", type=float, default=0.25)
    parser.add_argument("--steps", type=int, nargs="+", default=[25])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--zeta", type=float, default=0.1)
    parser.add_argument("--rho_floor", type=float, default=0.001)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--ckpt", default="512x512_diffusion.pt")
    parser.add_argument("--dps_class", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

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
        "method": "diffpir_style_openai_512",
        "crop": {"x0": args.x0, "y0": args.y0, "size": args.size},
        "b": args.b,
        "angle_deg": args.angle,
        "tile": args.tile,
        "overlap": args.overlap,
        "zeta": args.zeta,
        "rho_floor": args.rho_floor,
        "dps_class": args.dps_class,
        "seed": args.seed,
        "outputs": {},
    }
    panels = [("blurred", blur), ("sharp_registered", ref)]

    for steps in args.steps:
        for lambda_ in args.lambdas:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
                torch.cuda.reset_peak_memory_stats()
            label = f"diffpir_s{steps}_lam{lambda_:g}"
            print(f"\nRunning {label}")
            t0 = time.time()
            try:
                out = run_diffpir_tiled(
                    blur, args.b, args.angle, args.ckpt,
                    steps, lambda_, args.zeta, args.rho_floor,
                    args.dps_class, args.tile, args.overlap, device,
                )
                elapsed = time.time() - t0
                path = out_dir / f"globant_diffpir_b{args.b:.1f}_steps{steps}_lambda{lambda_:g}.png"
                Image.fromarray(out).save(path)
                peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
                metadata["outputs"][path.name] = {
                    "steps": steps,
                    "lambda": lambda_,
                    "psnr": psnr(out, ref),
                    "mae": mae(out, ref),
                    "edge_energy": edge_energy(out),
                    "runtime_seconds": elapsed,
                    "peak_memory_mb": peak_mb,
                }
                panels.append((label, out))
                print(f"  saved {path}")
                print(f"  PSNR {metadata['outputs'][path.name]['psnr']:.4f} dB, runtime {elapsed:.1f}s, peak {peak_mb:.1f} MB")
            except Exception as exc:
                metadata["outputs"][label] = {"error": repr(exc), "steps": steps, "lambda": lambda_}
                print(f"  failed: {exc}")
            with (out_dir / "diffpir_metrics.json").open("w") as f:
                json.dump(metadata, f, indent=2)

    comparison_path = out_dir / "diffpir_comparison.png"
    save_grid(panels, ref, comparison_path)
    metadata["comparison"] = str(comparison_path)
    with (out_dir / "diffpir_metrics.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved comparison: {comparison_path}")
    print(f"Saved metrics: {out_dir / 'diffpir_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
