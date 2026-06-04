#!/usr/bin/env python3
"""Deblur a 512x512 Globant-fence patch from the panning image."""

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
from PIL import Image, ImageOps
import torch

from deblur_best import (
    dpir_hqs,
    load_drunet,
    make_kernel_fft as make_kernel_fft_dpir,
    notch_cleanup,
    wiener_deconvolve,
)
from deblur import dps_deblur_tiled, load_openai_512_diffusion


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), dtype=np.uint8)


def crop_rgb(path: str | Path, x0: int, y0: int, size: int) -> np.ndarray:
    img = load_rgb(path)
    return img[y0:y0 + size, x0:x0 + size].copy()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def edge_energy(img: np.ndarray) -> float:
    gray = img.astype(np.float32).mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.mean(gx * gx) + np.mean(gy * gy))


def overlap_area(ax0: float, ay0: float, ax1: float, ay1: float,
                 bx0: float, by0: float, bx1: float, by1: float) -> float:
    w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    h = max(0.0, min(ay1, by1) - max(ay0, by0))
    return w * h


def local_kernel_length(kernel_map: Path, x0: int, y0: int, size: int) -> dict[str, float]:
    data = np.load(kernel_map, allow_pickle=True)
    statuses = np.array([str(x) for x in data["status"]])
    ok = statuses == "ok"
    patch_sizes = np.full_like(data["x0"].astype(float), 400.0)
    if "patch_size" in data.files:
        patch_sizes = data["patch_size"].astype(float)

    weights = []
    b_pixels = []
    b_specs = []
    b_diag = []
    for i in np.where(ok)[0]:
        px0 = float(data["x0"][i])
        py0 = float(data["y0"][i])
        p = float(patch_sizes[i])
        area = overlap_area(x0, y0, x0 + size, y0 + size, px0, py0, px0 + p, py0 + p)
        if area <= 0:
            continue
        conf = float(data["confidence"][i])
        weights.append(area * max(conf, 1e-6))
        b_pixels.append(float(data["b_px_pixel"][i]))
        b_specs.append(float(data["b_px_spec"][i]))
        b_diag.append(float(data["b_px"][i]))

    if not weights:
        return {}

    w = np.array(weights, dtype=np.float64)
    return {
        "b_px_pixel_weighted": float(np.average(np.array(b_pixels), weights=w)),
        "b_px_spec_weighted": float(np.average(np.array(b_specs), weights=w)),
        "b_px_diag_weighted": float(np.average(np.array(b_diag), weights=w)),
        "n_overlapping_kernel_patches": int(len(weights)),
    }


def run_wiener(img: np.ndarray, b: float, angle: float, eps: float, device: str) -> np.ndarray:
    y = torch.from_numpy(np.moveaxis(img.astype(np.float32), -1, 0)).to(device)
    K, K2 = make_kernel_fft_dpir(b, angle, img.shape[:2])
    with torch.no_grad():
        out = wiener_deconvolve(y, K, K2, eps)
    arr = out.clamp(0, 255).cpu().numpy().astype(np.uint8)
    return np.moveaxis(arr, 0, -1)


def reblur_rgb(img: np.ndarray, b: float, angle: float, device: str) -> np.ndarray:
    y = torch.from_numpy(np.moveaxis(img.astype(np.float32), -1, 0)).to(device)
    K, _ = make_kernel_fft_dpir(b, angle, img.shape[:2])
    with torch.no_grad():
        out = torch.fft.ifft2(K * torch.fft.fft2(y)).real
    arr = out.clamp(0, 255).cpu().numpy()
    return np.moveaxis(arr, 0, -1)


def reference_guided_restore(blur: np.ndarray, ref: np.ndarray, b: float,
                             angle: float, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Use the registered sharp crop as a deblurred reference with photometric matching."""
    ref_reblur = reblur_rgb(ref, b, angle, device)
    restored = np.empty_like(ref, dtype=np.float32)

    for c in range(3):
        x = ref_reblur[..., c].reshape(-1).astype(np.float32)
        y = blur[..., c].reshape(-1).astype(np.float32)
        # Robust-ish trim: ignore strongest residual outliers from registration/parallax.
        lo, hi = np.percentile(x, [2, 98])
        mask = (x >= lo) & (x <= hi)
        A = np.stack([x[mask], np.ones(mask.sum(), dtype=np.float32)], axis=1)
        scale, bias = np.linalg.lstsq(A, y[mask], rcond=None)[0]
        restored[..., c] = ref[..., c].astype(np.float32) * scale + bias

    return np.clip(restored, 0, 255).astype(np.uint8), np.clip(ref_reblur, 0, 255).astype(np.uint8)


def run_dpir(img: np.ndarray, b: float, angle: float, drunet_path: str,
             iters: int, sigma_start: float, sigma_end: float,
             sigma_n: float, k2_clip: float, device: str) -> tuple[np.ndarray, np.ndarray]:
    y = torch.from_numpy(np.moveaxis(img.astype(np.float32), -1, 0)).to(device)
    K, K2 = make_kernel_fft_dpir(b, angle, img.shape[:2])
    model = load_drunet(drunet_path)
    schedule = np.geomspace(sigma_start, sigma_end, iters).tolist()
    raw = dpir_hqs(y, K, K2, model, schedule, sigma_n=sigma_n, K2_clip=k2_clip)
    clean = notch_cleanup(
        raw, K2, model, b, angle,
        sigma_clean=max(1.0, sigma_end * 0.5),
        K2_null_thresh=k2_clip,
    )
    raw_np = np.moveaxis(raw.cpu().numpy().astype(np.uint8), 0, -1)
    clean_np = np.moveaxis(clean.cpu().numpy().astype(np.uint8), 0, -1)
    return raw_np, clean_np


def run_dps(img: np.ndarray, b: float, angle: float, ckpt: str,
            steps: int, zeta: float, dps_class: int, device: str) -> np.ndarray:
    model, diffusion = load_openai_512_diffusion(
        ckpt, device, ddim_steps=steps, dps_class=dps_class)
    y = torch.from_numpy(np.moveaxis(img.astype(np.float32), -1, 0)).to(device)
    out = dps_deblur_tiled(
        y, b, angle, model, diffusion,
        n_steps=steps, zeta=zeta, dps_class=dps_class,
        tile=512, overlap=0, device=device,
    )
    return np.moveaxis(out.cpu().numpy().astype(np.uint8), 0, -1)


def save_comparison(panels: list[tuple[str, np.ndarray]], ref: np.ndarray, path: Path) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
    if n == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, panels):
        ax.imshow(img)
        if label != "sharp_registered":
            title = f"{label}\nPSNR {psnr(img, ref):.2f} dB  MAE {mae(img, ref):.1f}"
        else:
            title = label
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--blurry", default="pan_1.jpg")
    p.add_argument("--sharp_reg", default="outputs_tttnvs_probe/sharp_registered.png")
    p.add_argument("--kernel_map", default="outputs_tttnvs_probe/kernel_map.npz")
    p.add_argument("--out_dir", default="outputs_globant_deblur")
    p.add_argument("--x0", type=int, default=2350)
    p.add_argument("--y0", type=int, default=2760)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--angle", type=float, default=0.25)
    p.add_argument("--b", type=float, default=None)
    p.add_argument("--extra_b", type=float, nargs="*", default=[139.7])
    p.add_argument("--wiener_eps", type=float, default=0.02)
    p.add_argument("--drunet_path", default="drunet_color.pth")
    p.add_argument("--dpir_iters", type=int, default=18)
    p.add_argument("--sigma_start", type=float, default=49.0)
    p.add_argument("--sigma_end", type=float, default=4.0)
    p.add_argument("--sigma_n", type=float, default=8.0)
    p.add_argument("--k2_clip", type=float, default=0.015)
    p.add_argument("--run_dps", action="store_true")
    p.add_argument("--dps_ckpt", default="512x512_diffusion.pt")
    p.add_argument("--dps_steps", type=int, default=25)
    p.add_argument("--dps_zeta", type=float, default=0.7)
    p.add_argument("--dps_class", type=int, default=-1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Crop: x={args.x0} y={args.y0} size={args.size}")

    blur = crop_rgb(args.blurry, args.x0, args.y0, args.size)
    ref = crop_rgb(args.sharp_reg, args.x0, args.y0, args.size)
    Image.fromarray(blur).save(out_dir / "globant_blurry.png")
    Image.fromarray(ref).save(out_dir / "globant_sharp_registered.png")

    local = local_kernel_length(Path(args.kernel_map), args.x0, args.y0, args.size)
    local_b = local.get("b_px_pixel_weighted", 126.9)
    b_main = float(args.b if args.b is not None else local_b)
    candidates = []
    for b in [b_main] + list(args.extra_b):
        if all(abs(b - prev) > 0.5 for prev in candidates):
            candidates.append(float(b))

    metadata = {
        "crop": {"x0": args.x0, "y0": args.y0, "size": args.size},
        "angle_deg": args.angle,
        "local_kernel": local,
        "b_candidates": candidates,
        "outputs": {},
    }

    panels = [("blurred", blur), ("sharp_registered", ref)]

    print(f"Local kernel: {json.dumps(local, indent=2)}")
    ref_guided, ref_reblur = reference_guided_restore(blur, ref, b_main, args.angle, device)
    for name, img, method in [
        ("globant_reference_reblur.png", ref_reblur, "reference_reblur"),
        ("globant_reference_guided.png", ref_guided, "reference_guided"),
    ]:
        path = out_dir / name
        Image.fromarray(img).save(path)
        metadata["outputs"][path.name] = {
            "method": method, "b": b_main,
            "psnr": psnr(img, ref), "mae": mae(img, ref),
            "edge_energy": edge_energy(img),
        }
    panels.append((f"reference guided b={b_main:.1f}", ref_guided))

    for b in candidates:
        print(f"\nWiener b={b:.2f}")
        wiener = run_wiener(blur, b, args.angle, args.wiener_eps, device)
        w_path = out_dir / f"globant_wiener_b{b:.1f}.png"
        Image.fromarray(wiener).save(w_path)
        metadata["outputs"][w_path.name] = {
            "method": "wiener", "b": b,
            "psnr": psnr(wiener, ref), "mae": mae(wiener, ref),
            "edge_energy": edge_energy(wiener),
        }
        panels.append((f"wiener b={b:.1f}", wiener))

        print(f"DPIR b={b:.2f}")
        t0 = time.time()
        dpir, dpir_clean = run_dpir(
            blur, b, args.angle, args.drunet_path,
            args.dpir_iters, args.sigma_start, args.sigma_end,
            args.sigma_n, args.k2_clip, device,
        )
        print(f"  DPIR done in {time.time() - t0:.1f}s")
        for suffix, img in [("dpir", dpir), ("dpir_clean", dpir_clean)]:
            path = out_dir / f"globant_{suffix}_b{b:.1f}.png"
            Image.fromarray(img).save(path)
            metadata["outputs"][path.name] = {
                "method": suffix, "b": b,
                "psnr": psnr(img, ref), "mae": mae(img, ref),
                "edge_energy": edge_energy(img),
            }
        panels.append((f"dpir clean b={b:.1f}", dpir_clean))

    if args.run_dps:
        b = b_main
        print(f"\nDPS b={b:.2f} steps={args.dps_steps} zeta={args.dps_zeta}")
        t0 = time.time()
        try:
            dps = run_dps(
                blur, b, args.angle, args.dps_ckpt,
                args.dps_steps, args.dps_zeta, args.dps_class, device,
            )
            print(f"  DPS done in {time.time() - t0:.1f}s")
            path = out_dir / f"globant_dps_b{b:.1f}_steps{args.dps_steps}.png"
            Image.fromarray(dps).save(path)
            metadata["outputs"][path.name] = {
                "method": "dps", "b": b, "steps": args.dps_steps,
                "zeta": args.dps_zeta,
                "psnr": psnr(dps, ref), "mae": mae(dps, ref),
                "edge_energy": edge_energy(dps),
            }
            panels.append((f"dps b={b:.1f}", dps))
        except Exception as exc:
            metadata["dps_error"] = repr(exc)
            print(f"  DPS failed: {exc}")

    comparison_path = out_dir / "globant_deblur_comparison.png"
    save_comparison(panels, ref, comparison_path)
    metadata["comparison"] = str(comparison_path)

    metrics_path = out_dir / "globant_deblur_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved comparison: {comparison_path}")
    print(f"Saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
