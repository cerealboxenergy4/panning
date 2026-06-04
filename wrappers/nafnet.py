#!/usr/bin/env python3
"""Uniform NAFNet wrapper for panning deblur candidates."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        n, c, h, w = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        _, c, _, _ = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, c, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True))
        self.sg = SimpleGate()
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + self.dropout1(x) * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.dropout2(x) * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, inp):
        _, _, h, w = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)
        x = self.ending(x) + inp
        return x[:, :, :h, :w]


def load_rgb(path):
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), dtype=np.uint8)


def save_rgb(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def load_model(weights, device):
    model = NAFNet().to(device).eval()
    checkpoint = torch.load(weights, map_location="cpu")
    state = checkpoint.get("params", checkpoint.get("params_ema", checkpoint))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def window(h, w, overlap, device):
    if overlap <= 0:
        return torch.ones((1, 1, h, w), device=device)
    y = torch.ones(h, device=device)
    x = torch.ones(w, device=device)
    ry = min(overlap, h // 2)
    rx = min(overlap, w // 2)
    if ry:
        ramp = torch.linspace(0.0, 1.0, ry + 2, device=device)[1:-1]
        y[:ry] = ramp
        y[-ry:] = torch.flip(ramp, [0])
    if rx:
        ramp = torch.linspace(0.0, 1.0, rx + 2, device=device)[1:-1]
        x[:rx] = ramp
        x[-rx:] = torch.flip(ramp, [0])
    return (y[:, None] * x[None, :]).clamp_min(1e-3)[None, None]


@torch.no_grad()
def infer_tiled(model, img, device, tile, overlap):
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    _, _, h, w = tensor.shape
    if h <= tile and w <= tile:
        out = model(tensor).clamp(0, 1)
        return (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    stride = max(1, tile - overlap)
    ys = tile_starts(h, min(tile, h), stride)
    xs = tile_starts(w, min(tile, w), stride)
    accum = torch.zeros_like(tensor)
    weights = torch.zeros((1, 1, h, w), device=device)
    total = len(ys) * len(xs)
    done = 0
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            restored = model(tensor[..., y0:y1, x0:x1]).clamp(0, 1)
            win = window(y1 - y0, x1 - x0, overlap, device)
            accum[..., y0:y1, x0:x1] += restored * win
            weights[..., y0:y1, x0:x1] += win
            done += 1
            print(f"tile {done}/{total} y={y0}:{y1} x={x0}:{x1}", flush=True)
    out = (accum / weights.clamp_min(1e-6)).clamp(0, 1)
    return (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=96)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    t0 = time.time()
    print(f"NAFNet wrapper device={device} tile={args.tile} overlap={args.overlap}", flush=True)
    model = load_model(args.weights, device)
    img = load_rgb(args.input)
    print(f"input shape={img.shape[1]}x{img.shape[0]}", flush=True)
    restored = infer_tiled(model, img, device, args.tile, args.overlap)
    save_rgb(args.output, restored)
    elapsed = time.time() - t0
    print(f"saved {args.output}", flush=True)
    print(f"runtime_seconds={elapsed:.2f}", flush=True)
    if device.type == "cuda":
        print(f"peak_memory_mb={torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0):.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
