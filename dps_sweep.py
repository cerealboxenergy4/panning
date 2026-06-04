#!/usr/bin/env python3
"""Run a focused DPS sweep for the Globant crop."""

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

from deblur import dps_deblur_tiled, load_openai_512_diffusion
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


def run_dps_once(
    img: np.ndarray,
    b: float,
    angle: float,
    ckpt: str,
    steps: int,
    zeta: float,
    dps_class: int,
    tile: int,
    overlap: int,
    device: str,
) -> np.ndarray:
    model, diffusion = load_openai_512_diffusion(
        ckpt, device, ddim_steps=steps, dps_class=dps_class)
    y = torch.from_numpy(np.moveaxis(img.astype(np.float32), -1, 0)).to(device)
    out = dps_deblur_tiled(
        y, b, angle, model, diffusion,
        n_steps=steps, zeta=zeta, dps_class=dps_class,
        tile=tile, overlap=overlap, device=device,
    )
    del model, diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.moveaxis(out.cpu().numpy().astype(np.uint8), 0, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blurry", default="pan_1.jpg")
    parser.add_argument("--sharp_reg", default="outputs_tttnvs_probe/sharp_registered.png")
    parser.add_argument("--out_dir", default="outputs_globant_deblur_1024_dps_sweep")
    parser.add_argument("--x0", type=int, default=2050)
    parser.add_argument("--y0", type=int, default=2500)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--b", type=float, default=128.14)
    parser.add_argument("--angle", type=float, default=0.25)
    parser.add_argument("--steps", type=int, nargs="+", default=[25, 50])
    parser.add_argument("--zetas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--dps_ckpt", default="512x512_diffusion.pt")
    parser.add_argument("--dps_class", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Crop: x={args.x0} y={args.y0} size={args.size}")
    print(f"DPS sweep: b={args.b} angle={args.angle} steps={args.steps} zetas={args.zetas} tile={args.tile} overlap={args.overlap}")

    blur = crop_rgb(args.blurry, args.x0, args.y0, args.size)
    ref = crop_rgb(args.sharp_reg, args.x0, args.y0, args.size)
    Image.fromarray(blur).save(out_dir / "globant_blurry.png")
    Image.fromarray(ref).save(out_dir / "globant_sharp_registered.png")

    metadata = {
        "crop": {"x0": args.x0, "y0": args.y0, "size": args.size},
        "angle_deg": args.angle,
        "b": args.b,
        "tile": args.tile,
        "overlap": args.overlap,
        "dps_class": args.dps_class,
        "seed": args.seed,
        "outputs": {},
    }
    panels = [("blurred", blur), ("sharp_registered", ref)]

    for steps in args.steps:
        for zeta in args.zetas:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
                torch.cuda.reset_peak_memory_stats()
            label = f"dps_s{steps}_z{zeta:g}"
            print(f"\nRunning {label}")
            t0 = time.time()
            try:
                img = run_dps_once(
                    blur, args.b, args.angle, args.dps_ckpt,
                    steps, zeta, args.dps_class, args.tile, args.overlap, device,
                )
                elapsed = time.time() - t0
                path = out_dir / f"globant_dps_b{args.b:.1f}_steps{steps}_zeta{zeta:g}.png"
                Image.fromarray(img).save(path)
                peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
                metadata["outputs"][path.name] = {
                    "method": "dps",
                    "b": args.b,
                    "steps": steps,
                    "zeta": zeta,
                    "psnr": psnr(img, ref),
                    "mae": mae(img, ref),
                    "edge_energy": edge_energy(img),
                    "runtime_seconds": elapsed,
                    "peak_memory_mb": peak_mb,
                }
                panels.append((label, img))
                print(f"  saved {path}")
                print(f"  PSNR {metadata['outputs'][path.name]['psnr']:.4f} dB, runtime {elapsed:.1f}s, peak {peak_mb:.1f} MB")
            except Exception as exc:
                metadata["outputs"][label] = {"error": repr(exc), "steps": steps, "zeta": zeta}
                print(f"  failed: {exc}")

            metrics_path = out_dir / "dps_sweep_metrics.json"
            with metrics_path.open("w") as f:
                json.dump(metadata, f, indent=2)

    comparison_path = out_dir / "dps_sweep_comparison.png"
    save_grid(panels, ref, comparison_path)
    metadata["comparison"] = str(comparison_path)
    with (out_dir / "dps_sweep_metrics.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved comparison: {comparison_path}")
    print(f"Saved metrics: {out_dir / 'dps_sweep_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
