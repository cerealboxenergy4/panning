"""
Stage 1 - Register sharp_1.jpg to pan_1.jpg.

The default registration path is LoFTR feature matching followed by a RANSAC
homography. SIFT/ORB remain available as explicit local-feature alternatives.

Optionally, this script can also run a depth-based sidecar warp: estimate depth
and cameras for the blurry/sharp pair, unproject sharp pixels, and reproject
those 3D points into the blurry camera. This writes comparison artifacts without
replacing the homography output used by the rest of the pipeline.

Outputs:
  outputs/<blurry_stem>/sharp_registered.png   - homography-warped sharp reference
  outputs/<blurry_stem>/homography.npy         - 3x3 homography matrix H (maps sharp -> blurry)
  outputs/<blurry_stem>/registration_debug.png - inlier match visualization
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb_gray(path, long_side=840):
    """
    Load image, apply EXIF orientation, resize so long side = long_side.
    Resized dimensions are snapped to multiples of 8 for LoFTR compatibility.
    """
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    rgb = np.array(img, dtype=np.uint8)
    height, width = rgb.shape[:2]

    scale = long_side / max(height, width)
    new_h = max(8, int(round(height * scale / 8)) * 8)
    new_w = max(8, int(round(width * scale / 8)) * 8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return rgb, gray, gray_small, (height, width), (new_h, new_w)


def gray_tensor(gray_small):
    import torch

    return torch.from_numpy(gray_small).float()[None, None] / 255.0


def create_local_feature_detector(matcher_name, max_features):
    if matcher_name == 'sift':
        if not hasattr(cv2, 'SIFT_create'):
            raise RuntimeError('OpenCV was built without SIFT support.')
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2, 'SIFT'

    if matcher_name == 'orb':
        return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING, 'ORB'

    raise ValueError(f'Unsupported local matcher: {matcher_name}')


def run_local_features(gray_blur, gray_sharp, car_mask_small, matcher_name, max_features, match_ratio):
    detector, norm, feature_name = create_local_feature_detector(matcher_name, max_features)
    background_mask = (~car_mask_small).astype(np.uint8) * 255

    kp_blur, desc_blur = detector.detectAndCompute(gray_blur, background_mask)
    kp_sharp, desc_sharp = detector.detectAndCompute(gray_sharp, None)
    if desc_blur is None or desc_sharp is None:
        raise RuntimeError(f'{feature_name} found no descriptors in one of the images.')

    matcher = cv2.BFMatcher(norm)
    raw = matcher.knnMatch(desc_sharp, desc_blur, k=2)

    good = []
    scores = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < match_ratio * n.distance:
            good.append(m)
            scores.append(1.0 - float(m.distance / max(n.distance, 1e-12)))

    if good:
        kpts_sharp = np.float32([kp_sharp[m.queryIdx].pt for m in good])
        kpts_blur = np.float32([kp_blur[m.trainIdx].pt for m in good])
        scores = np.float32(scores)
    else:
        kpts_sharp = np.empty((0, 2), dtype=np.float32)
        kpts_blur = np.empty((0, 2), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)

    stats = {
        'feature_name': feature_name,
        'blur_keypoints': len(kp_blur),
        'sharp_keypoints': len(kp_sharp),
        'raw_matches': len(raw),
        'good_matches': len(good),
    }
    return kpts_blur, kpts_sharp, scores, stats


def run_loftr(tensor_blur, tensor_sharp, device):
    import torch
    from kornia.feature import LoFTR

    matcher = LoFTR(pretrained='outdoor').to(device).eval()
    data = {
        'image0': tensor_blur.to(device),
        'image1': tensor_sharp.to(device),
    }
    with torch.no_grad():
        result = matcher(data)

    kpts0 = result['keypoints0'].cpu().numpy()
    kpts1 = result['keypoints1'].cpu().numpy()
    conf = result['confidence'].cpu().numpy()
    return kpts0, kpts1, conf


def as_homogeneous44(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape == (4, 4):
        return extrinsic
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = extrinsic
        return out
    raise ValueError(f'Expected extrinsic shape (3,4) or (4,4), got {extrinsic.shape}')


def normalize_rgb_u8(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.max(initial=0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def zbuffer_splat(flat_idx, z, colors, valid, out_flat, zbuf):
    if not np.any(valid):
        return 0
    idx = flat_idx[valid]
    z_valid = z[valid]
    colors_valid = colors[valid]
    np.minimum.at(zbuf, idx, z_valid)
    keep = z_valid <= zbuf[idx] + 1e-6
    out_flat[idx[keep]] = colors_valid[keep]
    return int(np.count_nonzero(keep))


def reproject_sharp_to_blurry(
    sharp_rgb,
    sharp_depth,
    sharp_K,
    sharp_w2c,
    blur_K,
    blur_w2c,
    out_hw,
    conf=None,
    conf_percentile=0.0,
    splat_radius=1,
):
    src_h, src_w = sharp_depth.shape
    out_h, out_w = out_hw

    ys, xs = np.meshgrid(np.arange(src_h), np.arange(src_w), indexing='ij')
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3).astype(np.float64)
    depth = sharp_depth.reshape(-1).astype(np.float64)

    valid = np.isfinite(depth) & (depth > 1e-8)
    if conf is not None:
        conf_flat = conf.reshape(-1)
        conf_valid = np.isfinite(conf_flat)
        if conf_percentile > 0:
            thr = np.percentile(conf_flat[conf_valid], conf_percentile) if np.any(conf_valid) else np.inf
            valid &= conf_valid & (conf_flat >= thr)
        else:
            valid &= conf_valid

    if not np.any(valid):
        raise RuntimeError('Depth reprojection found no valid sharp pixels.')

    K_inv = np.linalg.inv(np.asarray(sharp_K, dtype=np.float64))
    sharp_c2w = np.linalg.inv(as_homogeneous44(sharp_w2c))
    blur_w2c = as_homogeneous44(blur_w2c)
    blur_K = np.asarray(blur_K, dtype=np.float64)

    vidx = np.flatnonzero(valid)
    rays = K_inv @ pix[vidx].T
    Xc_sharp = rays * depth[vidx][None, :]
    Xc_sharp_h = np.vstack([Xc_sharp, np.ones((1, Xc_sharp.shape[1]), dtype=np.float64)])
    Xw = sharp_c2w @ Xc_sharp_h
    Xc_blur = blur_w2c @ Xw

    z = Xc_blur[2]
    front = np.isfinite(z) & (z > 1e-8)
    Xc_blur = Xc_blur[:3, front]
    z = z[front]
    vidx = vidx[front]

    proj = blur_K @ Xc_blur
    u = proj[0] / proj[2]
    v = proj[1] / proj[2]
    ui0 = np.rint(u).astype(np.int64)
    vi0 = np.rint(v).astype(np.int64)
    colors = normalize_rgb_u8(sharp_rgb).reshape(-1, 3)[vidx]

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out_flat = out.reshape(-1, 3)
    zbuf = np.full(out_h * out_w, np.inf, dtype=np.float64)
    assigned = 0

    radius = max(0, int(splat_radius))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ui = ui0 + dx
            vi = vi0 + dy
            in_bounds = (ui >= 0) & (ui < out_w) & (vi >= 0) & (vi < out_h)
            flat_idx = vi * out_w + ui
            assigned += zbuffer_splat(flat_idx, z, colors, in_bounds, out_flat, zbuf)

    valid_mask = np.isfinite(zbuf).reshape(out_h, out_w)
    return out, valid_mask, {
        'source_pixels': int(src_h * src_w),
        'valid_depth_pixels': int(len(vidx)),
        'assigned_splats': int(assigned),
        'coverage_fraction': float(valid_mask.mean()),
    }


def visualize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros(depth.shape + (3,), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def labeled_panel(label, image, size=None):
    image = normalize_rgb_u8(image)
    if size is not None and (image.shape[1], image.shape[0]) != size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 520), 34), (0, 0, 0), -1)
    cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def save_comparison_grid(panels, path):
    h = max(panel.shape[0] for panel in panels)
    w = max(panel.shape[1] for panel in panels)
    normalized = []
    for panel in panels:
        if panel.shape[:2] != (h, w):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[:panel.shape[0], :panel.shape[1]] = panel
            panel = canvas
        normalized.append(panel)
    top = np.concatenate(normalized[:2], axis=1)
    bottom = np.concatenate(normalized[2:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    Image.fromarray(grid).save(path)



def save_depth_provider_inputs(args, out_dir):
    paths = []
    for idx, image_path in enumerate([args.blurry, args.sharp]):
        stem = Path(image_path).stem
        out_path = out_dir / f'depth_input_{idx}_{stem}_exif.png'
        rgb = np.array(ImageOps.exif_transpose(Image.open(image_path)).convert('RGB'), dtype=np.uint8)
        Image.fromarray(rgb).save(out_path)
        paths.append(str(out_path))
    return paths

def load_da3_prediction(args, image_paths, device):
    repo_src = Path(args.da3_repo) / 'src'
    sys.path.insert(0, str(repo_src))
    from depth_anything_3.api import DepthAnything3

    print(f'Loading Depth-Anything-3 from {args.da3_model} ...')
    model = DepthAnything3.from_pretrained(args.da3_model).to(device).eval()
    print(f'Running DA3 inference at process_res={args.depth_process_res} ...')
    return model.inference(
        image_paths,
        process_res=args.depth_process_res,
        process_res_method=args.depth_process_res_method,
        export_dir=None,
        export_format='mini_npz',
        use_ray_pose=args.depth_use_ray_pose,
        ref_view_strategy=args.depth_ref_view_strategy,
    )


def load_vggt_prediction(args, image_paths, device):
    import torch

    sys.path.insert(0, args.vggt_repo)
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    print(f'Loading VGGT-Omega from {args.vggt_checkpoint} ...')
    images = load_and_preprocess_images(image_paths, image_resolution=args.vggt_image_resolution)
    model = VGGTOmega().eval()
    state_dict = torch.load(args.vggt_checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    images = images.to(device)

    print(f'Running VGGT-Omega inference at {tuple(images.shape[-2:])} ...')
    with torch.inference_mode():
        pred = model(images)

    image_shape_hw = tuple(images.shape[-2:])
    extrinsic, intrinsic = encoding_to_camera(pred['pose_enc'], image_shape_hw, build_intrinsics=True)
    depth = pred['depth'][0].detach().cpu().numpy()
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    conf = pred.get('depth_conf')
    conf_np = None if conf is None else conf[0].detach().cpu().numpy()
    processed = (pred['images'][0].detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    class PredictionLike:
        pass

    out = PredictionLike()
    out.depth = depth
    out.conf = conf_np
    out.intrinsics = intrinsic[0].detach().cpu().numpy()
    out.extrinsics = extrinsic[0].detach().cpu().numpy()
    out.processed_images = processed
    out.is_metric = 0
    return out


def run_depth_warp(args, sharp_warped_homography, out_dir, device):
    provider = args.depth_provider
    image_paths = save_depth_provider_inputs(args, out_dir)
    if provider == 'da3':
        prediction = load_da3_prediction(args, image_paths, device)
    elif provider == 'vggt':
        prediction = load_vggt_prediction(args, image_paths, device)
    else:
        raise ValueError(f'Unsupported depth provider: {provider}')

    required = ['depth', 'intrinsics', 'extrinsics', 'processed_images']
    missing = [name for name in required if getattr(prediction, name, None) is None]
    if missing:
        raise RuntimeError(f'{provider} prediction missing required fields: {missing}')

    depth = np.asarray(prediction.depth)
    intrinsics = np.asarray(prediction.intrinsics)
    extrinsics = np.asarray(prediction.extrinsics)
    processed = normalize_rgb_u8(prediction.processed_images)
    conf = None if getattr(prediction, 'conf', None) is None else np.asarray(prediction.conf)

    if depth.shape[0] < 2 or intrinsics.shape[0] < 2 or extrinsics.shape[0] < 2:
        raise RuntimeError(f'{provider} returned fewer than two views.')

    blur_img = processed[0]
    sharp_img = processed[1]
    out_h, out_w = depth[0].shape
    sharp_conf = None if conf is None else conf[1]

    depth_warp, depth_valid, stats = reproject_sharp_to_blurry(
        sharp_img,
        depth[1],
        intrinsics[1],
        extrinsics[1],
        intrinsics[0],
        extrinsics[0],
        (out_h, out_w),
        conf=sharp_conf,
        conf_percentile=args.depth_conf_percentile,
        splat_radius=args.depth_splat_radius,
    )

    prefix = f'depth_{provider}'
    warp_path = out_dir / f'sharp_registered_{prefix}.png'
    valid_path = out_dir / f'sharp_registered_{prefix}_valid.png'
    compare_path = out_dir / f'registration_compare_homography_vs_{prefix}.png'
    meta_path = out_dir / f'registration_compare_homography_vs_{prefix}.json'
    depth0_path = out_dir / f'{prefix}_blurry_depth.png'
    depth1_path = out_dir / f'{prefix}_sharp_depth.png'

    Image.fromarray(depth_warp).save(warp_path)
    Image.fromarray((depth_valid.astype(np.uint8) * 255)).save(valid_path)
    Image.fromarray(visualize_depth(depth[0])).save(depth0_path)
    Image.fromarray(visualize_depth(depth[1])).save(depth1_path)

    panel_size = (out_w, out_h)
    homography_small = cv2.resize(sharp_warped_homography, panel_size, interpolation=cv2.INTER_AREA)
    panels = [
        labeled_panel('blurry input (provider res)', blur_img, panel_size),
        labeled_panel('sharp input (provider res)', sharp_img, panel_size),
        labeled_panel('LoFTR homography warp', homography_small, panel_size),
        labeled_panel(f'{provider} depth reprojection', depth_warp, panel_size),
    ]
    save_comparison_grid(panels, compare_path)

    metadata = {
        'provider': provider,
        'image_order': ['blurry', 'sharp'],
        'process_shape_hw': [int(out_h), int(out_w)],
        'depth_is_metric': int(getattr(prediction, 'is_metric', 0)),
        'depth_conf_percentile': float(args.depth_conf_percentile),
        'depth_splat_radius': int(args.depth_splat_radius),
        'warp_stats': stats,
        'outputs': {
            'depth_warp': str(warp_path),
            'depth_valid_mask': str(valid_path),
            'comparison': str(compare_path),
            'blurry_depth_vis': str(depth0_path),
            'sharp_depth_vis': str(depth1_path),
        },
    }
    with meta_path.open('w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Saved: {warp_path}')
    print(f'Saved: {valid_path}  ({100.0 * stats["coverage_fraction"]:.1f}% coverage)')
    print(f'Saved: {compare_path}')
    print(f'Saved: {meta_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--blurry', default='pan_1.jpg')
    p.add_argument('--sharp', default='sharp_1.jpg')
    p.add_argument('--car_mask', default=None,
                   help='Car mask path; defaults to <out_dir>/car_mask.png')
    p.add_argument('--output_root', default='outputs',
                   help='Parent directory for per-image output subdirectories')
    p.add_argument('--out_dir', default=None,
                   help='Explicit output directory; defaults to <output_root>/<blurry_stem>')
    p.add_argument(
        '--matcher',
        choices=['loftr', 'sift', 'orb'],
        default='loftr',
        help='Feature matcher for homography registration; default is LoFTR.',
    )
    p.add_argument('--long_side', type=int, default=840,
                   help='Long side (px) to resize images before feature matching.')
    p.add_argument('--conf_thres', type=float, default=0.5,
                   help='Minimum LoFTR match confidence to keep.')
    p.add_argument('--max_features', type=int, default=8000,
                   help='Maximum local features for SIFT/ORB matchers.')
    p.add_argument('--match_ratio', type=float, default=0.75,
                   help='Lowe ratio threshold for SIFT/ORB matching.')
    p.add_argument('--ransac_thresh', type=float, default=8.0,
                   help='RANSAC reprojection threshold in original-image pixels.')
    p.add_argument('--min_inliers', type=int, default=12)

    p.add_argument('--depth_warp', action='store_true',
                   help='Also save a depth unprojection/reprojection warp for comparison.')
    p.add_argument('--depth_provider', choices=['da3', 'vggt'], default='da3')
    p.add_argument('--depth_process_res', type=int, default=504)
    p.add_argument('--depth_process_res_method', default='upper_bound_resize')
    p.add_argument('--depth_ref_view_strategy', default='first')
    p.add_argument('--depth_use_ray_pose', action='store_true')
    p.add_argument('--depth_conf_percentile', type=float, default=0.0)
    p.add_argument('--depth_splat_radius', type=int, default=1)
    p.add_argument('--da3_repo', default=os.environ.get('DA3_REPO', 'third_party/Depth-Anything-3'))
    p.add_argument('--da3_model', default=os.environ.get('DA3_MODEL', 'third_party/Depth-Anything-3/da3_streaming/weights'))
    p.add_argument('--vggt_repo', default=os.environ.get('VGGT_REPO', 'third_party/vggt-omega'))
    p.add_argument('--vggt_checkpoint', default=os.environ.get('VGGT_CHECKPOINT', 'checkpoints/vggt_omega_1b_512.pt'))
    p.add_argument('--vggt_image_resolution', type=int, default=512)
    args = p.parse_args()

    needs_torch = args.matcher == 'loftr' or args.depth_warp
    if needs_torch:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    blurry_path = Path(args.blurry)
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.output_root) / blurry_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.car_mask is None:
        args.car_mask = str(out_dir / 'car_mask.png')

    # Load images.
    blur_rgb, _, blur_small, (H_b, W_b), (h_b, w_b) = load_rgb_gray(args.blurry, args.long_side)
    sharp_rgb, _, sharp_small, (H_s, W_s), (h_s, w_s) = load_rgb_gray(args.sharp, args.long_side)

    print(f'Matcher: {args.matcher}')
    print(f'Blurry:  {args.blurry}  ({W_b}x{H_b})  ->  matcher input {w_b}x{h_b}')
    print(f'Sharp:   {args.sharp}  ({W_s}x{H_s})  ->  matcher input {w_s}x{h_s}')

    # Car mask for blurry image, resized to matcher scale.
    car_mask_orig = np.array(Image.open(args.car_mask).convert('L')) > 127
    car_mask_small = cv2.resize(
        car_mask_orig.astype(np.uint8) * 255,
        (w_b, h_b),
        interpolation=cv2.INTER_NEAREST,
    ) > 127

    # Feature matching for homography baseline.
    if args.matcher == 'loftr':
        print('Running LoFTR ...')
        blur_t = gray_tensor(blur_small)
        sharp_t = gray_tensor(sharp_small)
        kpts_blur, kpts_sharp, match_scores = run_loftr(blur_t, sharp_t, device)
        print(f'Raw LoFTR matches: {len(match_scores)}')

        conf_ok = match_scores >= args.conf_thres
        kpts_blur = kpts_blur[conf_ok]
        kpts_sharp = kpts_sharp[conf_ok]
        match_scores = match_scores[conf_ok]
        print(f'After conf >= {args.conf_thres}: {len(match_scores)} matches')
    else:
        print('Running OpenCV local feature matcher ...')
        kpts_blur, kpts_sharp, match_scores, match_stats = run_local_features(
            blur_small,
            sharp_small,
            car_mask_small,
            args.matcher,
            args.max_features,
            args.match_ratio,
        )
        print(
            f"{match_stats['feature_name']} keypoints: "
            f"blurry={match_stats['blur_keypoints']} sharp={match_stats['sharp_keypoints']}"
        )
        print(
            f"Raw descriptor matches: {match_stats['raw_matches']} | "
            f"after ratio < {args.match_ratio}: {match_stats['good_matches']}"
        )

    # Filter out matches inside car region of blurry image.
    kx = np.clip(kpts_blur[:, 0].astype(int), 0, w_b - 1)
    ky = np.clip(kpts_blur[:, 1].astype(int), 0, h_b - 1)
    not_car = ~car_mask_small[ky, kx]
    kpts_blur = kpts_blur[not_car]
    kpts_sharp = kpts_sharp[not_car]
    match_scores = match_scores[not_car]
    print(f'After car-mask exclusion: {len(match_scores)} matches')

    if len(match_scores) < args.min_inliers:
        raise RuntimeError(
            f'Only {len(match_scores)} matches remain after filtering - '
            'try relaxing --conf_thres/--match_ratio or checking the car mask.'
        )

    # Scale keypoints back to original image coordinates.
    sx_b, sy_b = W_b / w_b, H_b / h_b
    sx_s, sy_s = W_s / w_s, H_s / h_s
    pts_blur = kpts_blur * np.array([sx_b, sy_b])
    pts_sharp = kpts_sharp * np.array([sx_s, sy_s])

    # RANSAC homography.
    H_mat, inlier_mask = cv2.findHomography(
        pts_sharp.astype(np.float32),
        pts_blur.astype(np.float32),
        cv2.RANSAC,
        args.ransac_thresh,
    )

    if H_mat is None:
        raise RuntimeError('findHomography returned None.')

    n_inliers = int(inlier_mask.sum())
    print(f'RANSAC inliers: {n_inliers} / {len(match_scores)}')
    if n_inliers < args.min_inliers:
        raise RuntimeError(
            f'Only {n_inliers} RANSAC inliers - homography unreliable. '
            'Try --ransac_thresh 12 or a different --matcher.'
        )

    print(f'Homography (sharp -> blurry):\n{H_mat}')

    # Warp sharp reference into blurry frame.
    sharp_warped = cv2.warpPerspective(
        sharp_rgb,
        H_mat,
        (W_b, H_b),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Validity mask: warp an all-ones canvas to find which pixels have real data.
    ones = np.ones((H_s, W_s), dtype=np.uint8) * 255
    valid_mask = cv2.warpPerspective(
        ones,
        H_mat,
        (W_b, H_b),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    reg_path = out_dir / 'sharp_registered.png'
    Image.fromarray(sharp_warped).save(reg_path)
    print(f'Saved: {reg_path}')

    valid_path = out_dir / 'sharp_registered_valid.png'
    Image.fromarray(valid_mask).save(valid_path)
    n_valid = int((valid_mask > 0).sum())
    print(f'Saved: {valid_path}  ({100.0 * n_valid / (H_b * W_b):.1f}% of image covered)')

    np.save(out_dir / 'homography.npy', H_mat)
    print(f"Saved: {out_dir / 'homography.npy'}")

    # Debug visualization.
    vis_scale = min(1.0, 800.0 / max(H_b, W_b))
    blur_vis = cv2.resize(
        cv2.cvtColor(blur_rgb, cv2.COLOR_RGB2BGR),
        (int(W_b * vis_scale), int(H_b * vis_scale)),
    )
    sharp_vis = cv2.resize(
        cv2.cvtColor(sharp_rgb, cv2.COLOR_RGB2BGR),
        (int(W_s * vis_scale), int(H_s * vis_scale)),
    )

    inlier_idx = np.where(inlier_mask.ravel())[0][:80]
    kp_b = [
        cv2.KeyPoint(float(pts_blur[i, 0] * vis_scale), float(pts_blur[i, 1] * vis_scale), 4.0)
        for i in inlier_idx
    ]
    kp_s = [
        cv2.KeyPoint(float(pts_sharp[i, 0] * vis_scale), float(pts_sharp[i, 1] * vis_scale), 4.0)
        for i in inlier_idx
    ]
    matches_vis = [cv2.DMatch(j, j, 0) for j in range(len(inlier_idx))]

    debug = cv2.drawMatches(
        blur_vis,
        kp_b,
        sharp_vis,
        kp_s,
        matches_vis,
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    debug_path = out_dir / 'registration_debug.png'
    Image.fromarray(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)).save(debug_path)
    print(f'Saved: {debug_path}')

    if args.depth_warp:
        run_depth_warp(args, sharp_warped, out_dir, device)


if __name__ == '__main__':
    main()
