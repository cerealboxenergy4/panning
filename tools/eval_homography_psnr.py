#!/usr/bin/env python3
"""Evaluate restored image quality after homography alignment.

The project registration convention maps the sharp reference into the blurry or
restored image frame.  This evaluator follows that convention, then computes
PSNR only on pixels covered by the warped sharp reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), dtype=np.uint8)


def load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L")) > 0
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def estimate_homography(
    restored: np.ndarray,
    sharp_ref: np.ndarray,
    max_features: int,
    ratio: float,
    ransac_thresh: float,
    min_matches: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Estimate H mapping sharp_ref -> restored using local feature matches."""
    restored_gray = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
    sharp_gray = cv2.cvtColor(sharp_ref, cv2.COLOR_RGB2GRAY)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
        feature_name = "SIFT"
    else:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING
        feature_name = "ORB"

    kp_restored, desc_restored = detector.detectAndCompute(restored_gray, None)
    kp_sharp, desc_sharp = detector.detectAndCompute(sharp_gray, None)
    if desc_restored is None or desc_sharp is None:
        raise RuntimeError(f"{feature_name} found no descriptors in one of the images.")

    matcher = cv2.BFMatcher(norm)
    raw = matcher.knnMatch(desc_sharp, desc_restored, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_matches:
        raise RuntimeError(
            f"Only {len(good)} good {feature_name} matches; need at least {min_matches}."
        )

    pts_sharp = np.float32([kp_sharp[m.queryIdx].pt for m in good])
    pts_restored = np.float32([kp_restored[m.trainIdx].pt for m in good])
    H, inlier_mask = cv2.findHomography(pts_sharp, pts_restored, cv2.RANSAC, ransac_thresh)
    if H is None or inlier_mask is None:
        raise RuntimeError("cv2.findHomography failed.")

    inliers = int(inlier_mask.sum())
    if inliers < min_matches:
        raise RuntimeError(f"Only {inliers} RANSAC inliers; homography is unreliable.")

    stats: dict[str, float | int | str] = {
        "feature_type": feature_name,
        "restored_keypoints": len(kp_restored),
        "sharp_ref_keypoints": len(kp_sharp),
        "raw_match_count": len(raw),
        "good_match_count": len(good),
        "homography_inlier_count": inliers,
        "homography_inlier_ratio": float(inliers / max(len(good), 1)),
        "ransac_thresh": float(ransac_thresh),
        "match_ratio": float(ratio),
    }
    return H, stats


def warp_sharp(
    sharp_ref: np.ndarray,
    H: np.ndarray,
    out_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = out_shape
    warped = cv2.warpPerspective(
        sharp_ref,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ones = np.ones(sharp_ref.shape[:2], dtype=np.uint8) * 255
    valid = cv2.warpPerspective(
        ones,
        H,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    return warped, valid


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    k = 2 * pixels + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0


def metric_values(restored: np.ndarray, warped_sharp: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        raise RuntimeError("Overlap mask is empty; cannot compute PSNR.")
    diff = restored.astype(np.float32) - warped_sharp.astype(np.float32)
    valid_diff = diff[mask]
    mse = float(np.mean(valid_diff * valid_diff))
    mae = float(np.mean(np.abs(valid_diff)))
    psnr = float("inf") if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "homography_overlap_psnr": psnr,
        "overlap_mse": mse,
        "overlap_mae": mae,
        "overlap_pixel_count": count,
        "overlap_ratio": float(count / (mask.shape[0] * mask.shape[1])),
    }


def save_visual(
    restored: np.ndarray,
    warped_sharp: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    h, w = restored.shape[:2]
    thumb_w = 420
    scale = thumb_w / max(w, 1)
    thumb_h = max(1, int(round(h * scale)))

    def resize(img: np.ndarray) -> Image.Image:
        return Image.fromarray(img).resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)

    residual = np.zeros_like(restored)
    absdiff = np.abs(restored.astype(np.int16) - warped_sharp.astype(np.int16)).astype(np.uint8)
    residual[mask] = absdiff[mask]
    mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)

    panels = [
        ("restored", resize(restored)),
        ("warped sharp_ref", resize(warped_sharp)),
        ("overlap mask", resize(mask_rgb)),
        ("absolute residual", resize(residual)),
    ]
    label_h = 34
    canvas = Image.new("RGB", (thumb_w * len(panels), thumb_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, panel) in enumerate(panels):
        x = i * thumb_w
        canvas.paste(panel, (x, label_h))
        draw.text((x + 8, 9), label, fill=(0, 0, 0))
    draw.text((8, thumb_h + label_h - 20), title, fill=(0, 0, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restored", required=True, help="Restored/deblurred image in target frame.")
    parser.add_argument("--sharp_ref", required=True, help="Sharp reference image.")
    parser.add_argument("--out_json", required=True, help="Where to write metric JSON.")
    parser.add_argument("--out_vis", required=True, help="Where to write visual diagnostics.")
    parser.add_argument("--homography", default=None, help="Optional .npy homography mapping sharp_ref -> restored.")
    parser.add_argument("--valid_mask", default=None, help="Optional precomputed warped-reference valid mask.")
    parser.add_argument("--restored_mask", default=None, help="Optional valid mask for restored image.")
    parser.add_argument("--max_features", type=int, default=8000)
    parser.add_argument("--match_ratio", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=8.0)
    parser.add_argument("--min_matches", type=int, default=12)
    parser.add_argument("--mask_erode", type=int, default=3, help="Erode overlap mask to avoid border interpolation.")
    args = parser.parse_args()

    restored = load_rgb(args.restored)
    sharp_ref = load_rgb(args.sharp_ref)

    if args.homography:
        H = np.load(args.homography)
        match_stats: dict[str, float | int | str] = {
            "homography_source": str(args.homography),
            "feature_type": "provided",
            "homography_inlier_count": -1,
            "homography_inlier_ratio": -1.0,
        }
    else:
        H, match_stats = estimate_homography(
            restored,
            sharp_ref,
            args.max_features,
            args.match_ratio,
            args.ransac_thresh,
            args.min_matches,
        )
        match_stats["homography_source"] = "estimated"

    warped_sharp, warp_valid = warp_sharp(sharp_ref, H, restored.shape[:2])
    if args.valid_mask:
        warp_valid = warp_valid & load_mask(args.valid_mask, restored.shape[:2])
    if args.restored_mask:
        warp_valid = warp_valid & load_mask(args.restored_mask, restored.shape[:2])
    overlap = erode_mask(warp_valid, args.mask_erode)

    metrics = metric_values(restored, warped_sharp, overlap)
    result = {
        "restored": str(args.restored),
        "sharp_ref": str(args.sharp_ref),
        "restored_shape_hw": list(restored.shape[:2]),
        "sharp_ref_shape_hw": list(sharp_ref.shape[:2]),
        "homography_sharp_ref_to_restored": H.tolist(),
        **match_stats,
        **metrics,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(result, f, indent=2)

    psnr = result["homography_overlap_psnr"]
    psnr_text = "inf" if math.isinf(float(psnr)) else f"{float(psnr):.2f} dB"
    title = f"PSNR {psnr_text}, overlap {100.0 * float(result['overlap_ratio']):.1f}%"
    save_visual(restored, warped_sharp, overlap, args.out_vis, title)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
