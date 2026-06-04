#!/usr/bin/env python3
"""Unified runner for the panning-shot analysis pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    command: list[str]
    outputs: list[Path]


STAGE_KEYS = ("segment", "register", "kernel", "trajectory", "speed")


def default_run_dir(output_root: str | Path, blurry: Path, sharp: Path) -> Path:
    return Path(output_root) / f"{blurry.stem}__{sharp.stem}"


def existing(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def check_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def run_stage(stage: Stage, env: dict[str, str], force: bool, dry_run: bool) -> str:
    outputs_exist = existing(stage.outputs)
    if outputs_exist and not force and not dry_run:
        print(f"\n== {stage.label} ==", flush=True)
        print("Skipping; expected outputs already exist.", flush=True)
        return "skipped"

    print(f"\n== {stage.label} ==", flush=True)
    print("+ " + " ".join(stage.command), flush=True)
    if outputs_exist and not force:
        print("# outputs exist; normal run would skip this stage", flush=True)
    if dry_run:
        return "dry-run"

    subprocess.run(stage.command, check=True, env=env)
    missing = [path for path in stage.outputs if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"{stage.label} finished but did not create: {missing_str}")
    return "ran"


def selected_stages(stages: list[Stage], start_at: str, stop_after: str) -> list[Stage]:
    start = STAGE_KEYS.index(start_at)
    stop = STAGE_KEYS.index(stop_after)
    if stop < start:
        raise ValueError("--stop-after must be the same as or later than --start-at")
    selected_keys = set(STAGE_KEYS[start : stop + 1])
    return [stage for stage in stages if stage.key in selected_keys]


def print_summary(out_dir: Path) -> None:
    traj_path = out_dir / "trajectory.json"
    if not traj_path.exists():
        return

    with traj_path.open() as f:
        traj = json.load(f)

    print("\n== Result Summary ==", flush=True)
    print(f"trajectory: {traj_path}", flush=True)
    print(f"b_total_px: {traj.get('b_total_px', float('nan')):.2f}", flush=True)
    print(f"phi_deg: {traj.get('phi_deg', float('nan')):.2f}", flush=True)
    print(f"omega_deg_s: {traj.get('omega_deg_s', float('nan')):.2f}", flush=True)
    print(f"weighted_rmse_px: {traj.get('weighted_rmse_px', float('nan')):.2f}", flush=True)
    print(
        "patches: "
        f"{traj.get('n_patches_inlier', '?')} inlier / "
        f"{traj.get('n_patches_total', '?')} total",
        flush=True,
    )

    speed_path = out_dir / "car_speed.json"
    if speed_path.exists():
        with speed_path.open() as f:
            speed = json.load(f)
        print(f"car_speed_km_h: {speed.get('speed_km_h', float('nan')):.1f}", flush=True)
        print(f"car_depth_m: {speed.get('depth_m', float('nan')):.2f}", flush=True)
        print(f"wheelbase_px: {speed.get('wheelbase_px', float('nan')):.1f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run segmentation, registration, kernel estimation, trajectory fitting, and car-speed estimation."
    )
    parser.add_argument("--blurry", default="pan_1.jpg", help="Panning/blurry image.")
    parser.add_argument("--sharp", default="sharp_1.jpg", help="Sharp reference image.")
    parser.add_argument(
        "--output_root",
        default="outputs",
        help="Parent directory for per-pair output subdirectories.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Explicit output directory override; defaults to <output_root>/<blurry_stem>__<sharp_stem>.",
    )
    parser.add_argument("--gpu", default="4", help="Physical GPU id exposed to child processes.")
    parser.add_argument("--cpu", action="store_true", help="Run child processes without CUDA.")
    parser.add_argument("--force", action="store_true", help="Rerun stages even if outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--start-at", choices=STAGE_KEYS, default="segment")
    parser.add_argument("--stop-after", choices=STAGE_KEYS, default="speed")

    parser.add_argument(
        "--matcher",
        choices=["loftr", "sift", "orb"],
        default="loftr",
        help="Registration matcher; default is LoFTR.",
    )
    parser.add_argument("--long_side", type=int, default=840, help="Registration matcher resize long side.")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="LoFTR confidence threshold.")
    parser.add_argument("--max_features", type=int, default=8000, help="Maximum SIFT/ORB features.")
    parser.add_argument("--match_ratio", type=float, default=0.75, help="SIFT/ORB Lowe ratio threshold.")
    parser.add_argument("--depth_warp", action="store_true", help="Also save a depth reprojection registration comparison.")
    parser.add_argument("--depth_provider", choices=["da3", "vggt"], default="da3")
    parser.add_argument("--depth_process_res", type=int, default=504)
    parser.add_argument("--depth_process_res_method", default="upper_bound_resize")
    parser.add_argument("--depth_ref_view_strategy", default="first")
    parser.add_argument("--depth_use_ray_pose", action="store_true")
    parser.add_argument("--depth_conf_percentile", type=float, default=0.0)
    parser.add_argument("--depth_splat_radius", type=int, default=1)
    parser.add_argument("--da3_repo", default=os.environ.get("DA3_REPO", "third_party/Depth-Anything-3"))
    parser.add_argument("--da3_model", default=os.environ.get("DA3_MODEL", "third_party/Depth-Anything-3/da3_streaming/weights"))
    parser.add_argument("--vggt_repo", default=os.environ.get("VGGT_REPO", "third_party/vggt-omega"))
    parser.add_argument("--vggt_checkpoint", default=os.environ.get("VGGT_CHECKPOINT", "checkpoints/vggt_omega_1b_512.pt"))
    parser.add_argument("--vggt_image_resolution", type=int, default=512)
    parser.add_argument("--ransac_thresh", type=float, default=8.0, help="RANSAC threshold in px.")
    parser.add_argument("--patch_size", type=int, default=400, help="Kernel-estimation patch size.")
    parser.add_argument(
        "--grad_energy_thres",
        type=float,
        default=100.0,
        help="Minimum sharp-patch Sobel energy for kernel estimation.",
    )
    parser.add_argument(
        "--grad_var_thres",
        type=float,
        default=0.0,
        help="Minimum variance of sharp-patch gradient magnitudes; 0 disables this gate.",
    )
    parser.add_argument(
        "--harris_thres",
        type=float,
        default=0.0,
        help="Minimum max Harris response per patch; 0 disables this gate.",
    )
    parser.add_argument("--harris_block_size", type=int, default=5)
    parser.add_argument("--harris_k", type=float, default=0.04)
    parser.add_argument("--global_phi", type=float, default=None, help="Override blur direction.")
    parser.add_argument(
        "--weight_field",
        default="confidence",
        help="Kernel-map field used as trajectory fitting weights.",
    )
    parser.add_argument(
        "--disable_ransac",
        action="store_true",
        help="Use sigma-clipped WLS instead of weighted RANSAC for trajectory fitting.",
    )
    parser.add_argument("--ransac_iters", type=int, default=256)
    parser.add_argument("--ransac_min_samples", type=int, default=3)
    parser.add_argument("--ransac_residual_thres", type=float, default=15.0)
    parser.add_argument("--ransac_seed", type=int, default=0)
    parser.add_argument("--outlier_sigma", type=float, default=2.5, help="Fallback WLS sigma cutoff.")
    parser.add_argument("--wheelbase_mm", type=float, default=3400.0, help="Known car wheelbase in millimeters.")
    parser.add_argument("--wheelbase_px", type=float, default=None, help="Manual wheel-center distance in pixels.")
    parser.add_argument("--left_wheel", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--right_wheel", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--wheel_text_prompt", default="wheel . tire . car wheel . racing wheel .")
    parser.add_argument("--wheel_box_threshold", type=float, default=0.18)
    parser.add_argument("--wheel_text_threshold", type=float, default=0.15)
    parser.add_argument("--wheelbase_bbox_ratio_prior", type=float, default=0.50)
    parser.add_argument("--speed_focal_px", type=float, default=None, help="Manual focal length in pixels for speed estimation.")
    parser.add_argument("--speed_focal_mm", type=float, default=None, help="Manual focal length in mm for speed estimation.")
    parser.add_argument("--speed_sensor_width_mm", type=float, default=None, help="Sensor width in mm for speed estimation.")
    args = parser.parse_args()

    blurry = Path(args.blurry)
    sharp = Path(args.sharp)
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_run_dir(args.output_root, blurry, sharp)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    check_inputs([blurry, sharp])

    car_mask = out_dir / "car_mask.png"
    car_detection_json = out_dir / "car_detection.json"
    sharp_reg = out_dir / "sharp_registered.png"
    valid_mask = out_dir / "sharp_registered_valid.png"
    kernel_map = out_dir / "kernel_map.npz"
    trajectory_json = out_dir / "trajectory.json"

    py = sys.executable
    script_dir = Path(__file__).resolve().parent
    stages = [
        Stage(
            key="segment",
            label="Stage 1: car segmentation",
            command=[
                py,
                str(script_dir / "segment_car.py"),
                "--image",
                str(blurry),
                "--out_dir",
                str(out_dir),
            ],
            outputs=[car_mask, out_dir / "car_detection.png", car_detection_json],
        ),
        Stage(
            key="register",
            label="Stage 2: reference registration",
            command=[
                py,
                str(script_dir / "register_reference.py"),
                "--blurry",
                str(blurry),
                "--sharp",
                str(sharp),
                "--car_mask",
                str(car_mask),
                "--out_dir",
                str(out_dir),
                "--matcher",
                str(args.matcher),
                "--long_side",
                str(args.long_side),
                "--conf_thres",
                str(args.conf_thres),
                "--max_features",
                str(args.max_features),
                "--match_ratio",
                str(args.match_ratio),
                "--ransac_thresh",
                str(args.ransac_thresh),
            ]
            + (
                [
                    "--depth_warp",
                    "--depth_provider",
                    str(args.depth_provider),
                    "--depth_process_res",
                    str(args.depth_process_res),
                    "--depth_process_res_method",
                    str(args.depth_process_res_method),
                    "--depth_ref_view_strategy",
                    str(args.depth_ref_view_strategy),
                    "--depth_conf_percentile",
                    str(args.depth_conf_percentile),
                    "--depth_splat_radius",
                    str(args.depth_splat_radius),
                    "--da3_repo",
                    str(args.da3_repo),
                    "--da3_model",
                    str(args.da3_model),
                    "--vggt_repo",
                    str(args.vggt_repo),
                    "--vggt_checkpoint",
                    str(args.vggt_checkpoint),
                    "--vggt_image_resolution",
                    str(args.vggt_image_resolution),
                ]
                + (["--depth_use_ray_pose"] if args.depth_use_ray_pose else [])
                if args.depth_warp
                else []
            ),
            outputs=[
                sharp_reg,
                valid_mask,
                out_dir / "homography.npy",
                out_dir / "registration_debug.png",
            ],
        ),
        Stage(
            key="kernel",
            label="Stage 3: kernel estimation",
            command=[
                py,
                str(script_dir / "kernel_estimation.py"),
                "--blurry",
                str(blurry),
                "--sharp_reg",
                str(sharp_reg),
                "--car_mask",
                str(car_mask),
                "--valid_mask",
                str(valid_mask),
                "--out_dir",
                str(out_dir),
                "--patch_size",
                str(args.patch_size),
                "--grad_energy_thres",
                str(args.grad_energy_thres),
                "--grad_var_thres",
                str(args.grad_var_thres),
                "--harris_thres",
                str(args.harris_thres),
                "--harris_block_size",
                str(args.harris_block_size),
                "--harris_k",
                str(args.harris_k),
            ]
            + ([] if args.global_phi is None else ["--global_phi", str(args.global_phi)]),
            outputs=[
                kernel_map,
                out_dir / "kernel_map.csv",
                out_dir / "kernel_map.png",
                out_dir / "uniform_traj.json",
                out_dir / "uniform_trajectory_residual.png",
            ],
        ),
        Stage(
            key="trajectory",
            label="Stage 4: trajectory fitting",
            command=[
                py,
                str(script_dir / "trajectory_fitting.py"),
                "--kernel_map",
                str(kernel_map),
                "--sharp_reg",
                str(sharp_reg),
                "--blurry",
                str(blurry),
                "--out_dir",
                str(out_dir),
                "--outlier_sigma",
                str(args.outlier_sigma),
                "--weight_field",
                str(args.weight_field),
                "--ransac_iters",
                str(args.ransac_iters),
                "--ransac_min_samples",
                str(args.ransac_min_samples),
                "--ransac_residual_thres",
                str(args.ransac_residual_thres),
                "--ransac_seed",
                str(args.ransac_seed),
            ]
            + (["--disable_ransac"] if args.disable_ransac else []),
            outputs=[
                trajectory_json,
                out_dir / "trajectory_residual.png",
                out_dir / "trajectory_scatter.png",
            ],
        ),
        Stage(
            key="speed",
            label="Stage 5: car speed estimation",
            command=[
                py,
                str(script_dir / "estimate_car_speed.py"),
                "--image",
                str(blurry),
                "--car_mask",
                str(car_mask),
                "--car_detection_json",
                str(car_detection_json),
                "--trajectory",
                str(trajectory_json),
                "--out_dir",
                str(out_dir),
                "--wheelbase_mm",
                str(args.wheelbase_mm),
                "--wheel_text_prompt",
                str(args.wheel_text_prompt),
                "--wheel_box_threshold",
                str(args.wheel_box_threshold),
                "--wheel_text_threshold",
                str(args.wheel_text_threshold),
                "--wheelbase_bbox_ratio_prior",
                str(args.wheelbase_bbox_ratio_prior),
            ]
            + ([] if args.wheelbase_px is None else ["--wheelbase_px", str(args.wheelbase_px)])
            + ([] if args.left_wheel is None else ["--left_wheel", *(str(v) for v in args.left_wheel)])
            + ([] if args.right_wheel is None else ["--right_wheel", *(str(v) for v in args.right_wheel)])
            + ([] if args.speed_focal_px is None else ["--focal_px", str(args.speed_focal_px)])
            + ([] if args.speed_focal_mm is None else ["--focal_mm", str(args.speed_focal_mm)])
            + ([] if args.speed_sensor_width_mm is None else ["--sensor_width_mm", str(args.speed_sensor_width_mm)]),
            outputs=[
                out_dir / "car_speed.json",
                out_dir / "car_speed_wheels.png",
                out_dir / "final_result.png",
            ],
        ),
    ]

    env = os.environ.copy()
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        device_msg = "CPU"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device_msg = f"GPU {args.gpu}"

    chosen = selected_stages(stages, args.start_at, args.stop_after)
    print(f"Inputs: blurry={blurry} sharp={sharp} out_dir={out_dir}", flush=True)
    print(f"Device selection for child processes: {device_msg}", flush=True)

    status_by_stage: dict[str, str] = {}
    for stage in chosen:
        status_by_stage[stage.key] = run_stage(stage, env, args.force, args.dry_run)

    print("\n== Pipeline Status ==", flush=True)
    for stage in chosen:
        print(f"{stage.key}: {status_by_stage[stage.key]}", flush=True)

    if not args.dry_run and any(stage.key in {"trajectory", "speed"} for stage in chosen):
        print_summary(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
