#!/usr/bin/env python3
"""Uniform Restormer wrapper for panning deblur candidates."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F


def load_restormer_class(repo: Path):
    arch_path = repo / "basicsr" / "models" / "archs" / "restormer_arch.py"
    spec = importlib.util.spec_from_file_location("panning_restormer_arch", arch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Restormer architecture from {arch_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Restormer


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), dtype=np.uint8)


def save_rgb(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def build_model(repo: Path, weights: Path, device: torch.device) -> torch.nn.Module:
    Restormer = load_restormer_class(repo)
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )
    checkpoint = torch.load(weights, map_location="cpu")
    state = checkpoint.get("params", checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def raised_cosine_window(h: int, w: int, overlap: int, device: torch.device) -> torch.Tensor:
    if overlap <= 0:
        return torch.ones((1, 1, h, w), device=device)
    y = torch.ones(h, device=device)
    x = torch.ones(w, device=device)
    ramp_y = min(overlap, h // 2)
    ramp_x = min(overlap, w // 2)
    if ramp_y > 0:
        ramp = torch.linspace(0.0, 1.0, ramp_y + 2, device=device)[1:-1]
        y[:ramp_y] = ramp
        y[-ramp_y:] = torch.flip(ramp, dims=[0])
    if ramp_x > 0:
        ramp = torch.linspace(0.0, 1.0, ramp_x + 2, device=device)[1:-1]
        x[:ramp_x] = ramp
        x[-ramp_x:] = torch.flip(ramp, dims=[0])
    return (y[:, None] * x[None, :]).clamp_min(1e-3)[None, None]


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def infer_tile(model: torch.nn.Module, tile: torch.Tensor, factor: int) -> torch.Tensor:
    h, w = tile.shape[-2:]
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")
    restored = model(tile)
    return restored[..., :h, :w].clamp(0, 1)


@torch.no_grad()
def infer_tiled(
    model: torch.nn.Module,
    img: np.ndarray,
    device: torch.device,
    tile: int,
    overlap: int,
    factor: int,
) -> np.ndarray:
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    _, _, h, w = tensor.shape
    if tile <= 0 or (h <= tile and w <= tile):
        out = infer_tile(model, tensor, factor)
        return (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

    tile = min(tile, max(h, w))
    stride = max(1, tile - overlap)
    ys = tile_starts(h, min(tile, h), stride)
    xs = tile_starts(w, min(tile, w), stride)
    accum = torch.zeros_like(tensor)
    weight_sum = torch.zeros((1, 1, h, w), device=device)

    total = len(ys) * len(xs)
    done = 0
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            patch = tensor[..., y0:y1, x0:x1]
            restored = infer_tile(model, patch, factor)
            window = raised_cosine_window(y1 - y0, x1 - x0, overlap, device)
            accum[..., y0:y1, x0:x1] += restored * window
            weight_sum[..., y0:y1, x0:x1] += window
            done += 1
            print(f"tile {done}/{total} y={y0}:{y1} x={x0}:{x1}", flush=True)

    out = (accum / weight_sum.clamp_min(1e-6)).clamp(0, 1)
    return (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--repo", default="third_party/Restormer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--factor", type=int, default=8)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    t0 = time.time()
    print(f"Restormer wrapper device={device} tile={args.tile} overlap={args.overlap}", flush=True)
    model = build_model(Path(args.repo), Path(args.weights), device)
    img = load_rgb(args.input)
    print(f"input shape={img.shape[1]}x{img.shape[0]}", flush=True)
    restored = infer_tiled(model, img, device, args.tile, args.overlap, args.factor)
    save_rgb(args.output, restored)
    elapsed = time.time() - t0
    print(f"saved {args.output}", flush=True)
    print(f"runtime_seconds={elapsed:.2f}", flush=True)
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        print(f"peak_memory_mb={peak_mb:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
