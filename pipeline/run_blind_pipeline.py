#!/usr/bin/env python3
"""Unified runner for the experimental blind panning pipeline."""

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


STAGE_KEYS = ('segment', 'kernel', 'trajectory', 'speed')


def default_run_dir(output_root: str | Path, blurry: Path) -> Path:
    return Path(output_root) / f'{blurry.stem}__blind'


def existing(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def check_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing required input(s): ' + ', '.join(missing))


def run_stage(stage: Stage, env: dict[str, str], force: bool, dry_run: bool) -> str:
    outputs_exist = existing(stage.outputs)
    if outputs_exist and not force and not dry_run:
        print(f'\n== {stage.label} ==', flush=True)
        print('Skipping; expected outputs already exist.', flush=True)
        return 'skipped'

    print(f'\n== {stage.label} ==', flush=True)
    print('+ ' + ' '.join(stage.command), flush=True)
    if outputs_exist and not force:
        print('# outputs exist; normal run would skip this stage', flush=True)
    if dry_run:
        return 'dry-run'

    subprocess.run(stage.command, check=True, env=env)
    missing = [path for path in stage.outputs if not path.exists()]
    if missing:
        missing_str = ', '.join(str(path) for path in missing)
        raise RuntimeError(f'{stage.label} finished but did not create: {missing_str}')
    return 'ran'


def selected_stages(stages: list[Stage], start_at: str, stop_after: str) -> list[Stage]:
    start = STAGE_KEYS.index(start_at)
    stop = STAGE_KEYS.index(stop_after)
    if stop < start:
        raise ValueError('--stop-after must be the same as or later than --start-at')
    selected_keys = set(STAGE_KEYS[start:stop + 1])
    return [stage for stage in stages if stage.key in selected_keys]


def print_summary(out_dir: Path) -> None:
    traj_path = out_dir / 'trajectory.json'
    speed_path = out_dir / 'car_speed.json'
    if not traj_path.exists():
        return
    with traj_path.open() as f:
        traj = json.load(f)
    print('\n== Blind Result Summary ==', flush=True)
    print(f'trajectory: {traj_path}', flush=True)
    print(f"b_total_px: {traj.get('b_total_px', float('nan')):.2f}", flush=True)
    print(f"phi_deg: {traj.get('phi_deg', float('nan')):.2f}", flush=True)
    print(f"omega_deg_s: {traj.get('omega_deg_s', float('nan')):.2f}", flush=True)
    print(f"weighted_rmse_px: {traj.get('weighted_rmse_px', float('nan')):.2f}", flush=True)
    print(f"patches: {traj.get('n_patches_inlier', '?')} inlier / {traj.get('n_patches_total', '?')} total", flush=True)
    if speed_path.exists():
        with speed_path.open() as f:
            speed = json.load(f)
        print(f"car_speed_km_h: {speed.get('speed_km_h', float('nan')):.1f}", flush=True)
        print(f"car_depth_m: {speed.get('depth_m', float('nan')):.2f}", flush=True)
        print(f"wheelbase_px: {speed.get('wheelbase_px', float('nan')):.1f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run segmentation, blind kernel estimation, trajectory fitting, and car-speed estimation.'
    )
    parser.add_argument('--blurry', default='pan_1.jpg', help='Panning/blurry image.')
    parser.add_argument('--output_root', default='outputs', help='Parent directory for blind output subdirectories.')
    parser.add_argument('--out_dir', default=None, help='Explicit output directory; defaults to <output_root>/<blurry_stem>__blind.')
    parser.add_argument('--gpu', default='4', help='Physical GPU id exposed to child processes.')
    parser.add_argument('--cpu', action='store_true', help='Run child processes without CUDA.')
    parser.add_argument('--force', action='store_true', help='Rerun stages even if outputs exist.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running them.')
    parser.add_argument('--start-at', choices=STAGE_KEYS, default='segment')
    parser.add_argument('--stop-after', choices=STAGE_KEYS, default='speed')

    parser.add_argument('--patch_size', type=int, default=400)
    parser.add_argument('--grad_energy_thres', type=float, default=100.0)
    parser.add_argument('--grad_var_thres', type=float, default=0.0)
    parser.add_argument('--harris_thres', type=float, default=0.0)
    parser.add_argument('--global_phi', type=float, default=None)
    parser.add_argument('--b_min', type=float, default=120.0)
    parser.add_argument('--b_max', type=float, default=None)
    parser.add_argument('--blind_score_thres', type=float, default=0.03)
    parser.add_argument('--weight_field', default='confidence')
    parser.add_argument('--disable_ransac', action='store_true')
    parser.add_argument('--manual_blur_px', type=float, default=None, help='Optional human-estimated blur length annotated on the contribution map.')
    parser.add_argument('--ransac_iters', type=int, default=256)
    parser.add_argument('--ransac_min_samples', type=int, default=3)
    parser.add_argument('--ransac_residual_thres', type=float, default=15.0)
    parser.add_argument('--ransac_seed', type=int, default=0)
    parser.add_argument('--outlier_sigma', type=float, default=2.5)
    parser.add_argument('--wheelbase_mm', type=float, default=3400.0)
    parser.add_argument('--wheelbase_px', type=float, default=None)
    parser.add_argument('--left_wheel', nargs=2, type=float, default=None, metavar=('X', 'Y'))
    parser.add_argument('--right_wheel', nargs=2, type=float, default=None, metavar=('X', 'Y'))
    parser.add_argument('--wheel_text_prompt', default='wheel . tire . car wheel . racing wheel .')
    parser.add_argument('--wheel_box_threshold', type=float, default=0.18)
    parser.add_argument('--wheel_text_threshold', type=float, default=0.15)
    parser.add_argument('--wheelbase_bbox_ratio_prior', type=float, default=0.50)
    parser.add_argument('--pan_focal_px', type=float, default=None, help='Manual panning-image focal length in pixels for trajectory fitting.')
    parser.add_argument('--pan_focal_mm', type=float, default=None, help='Manual panning-image focal length in mm for trajectory fitting.')
    parser.add_argument('--pan_sensor_width_mm', type=float, default=None, help='Sensor width in mm for trajectory fitting.')
    parser.add_argument('--pan_exposure_s', type=float, default=None, help='Manual panning-image exposure time in seconds.')
    parser.add_argument('--speed_focal_px', type=float, default=None)
    parser.add_argument('--speed_focal_mm', type=float, default=None)
    parser.add_argument('--speed_sensor_width_mm', type=float, default=None)
    args = parser.parse_args()

    blurry = Path(args.blurry)
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_run_dir(args.output_root, blurry)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    check_inputs([blurry])

    speed_focal_px = args.speed_focal_px if args.speed_focal_px is not None else args.pan_focal_px
    speed_focal_mm = args.speed_focal_mm if args.speed_focal_mm is not None else args.pan_focal_mm
    speed_sensor_width_mm = (
        args.speed_sensor_width_mm
        if args.speed_sensor_width_mm is not None
        else args.pan_sensor_width_mm
    )

    car_mask = out_dir / 'car_mask.png'
    car_detection_json = out_dir / 'car_detection.json'
    kernel_map = out_dir / 'kernel_map.npz'
    trajectory_json = out_dir / 'trajectory.json'

    py = sys.executable
    script_dir = Path(__file__).resolve().parent
    stages = [
        Stage(
            key='segment',
            label='Stage 1: car segmentation',
            command=[py, str(script_dir / 'segment_car.py'), '--image', str(blurry), '--out_dir', str(out_dir)],
            outputs=[car_mask, out_dir / 'car_detection.png', car_detection_json],
        ),
        Stage(
            key='kernel',
            label='Stage 2: blind kernel estimation',
            command=[
                py, str(script_dir / 'blind_kernel_estimation.py'),
                '--blurry', str(blurry),
                '--car_mask', str(car_mask),
                '--out_dir', str(out_dir),
                '--patch_size', str(args.patch_size),
                '--grad_energy_thres', str(args.grad_energy_thres),
                '--grad_var_thres', str(args.grad_var_thres),
                '--harris_thres', str(args.harris_thres),
                '--b_min', str(args.b_min),
                '--blind_score_thres', str(args.blind_score_thres),
            ]
            + ([] if args.global_phi is None else ['--global_phi', str(args.global_phi)])
            + ([] if args.b_max is None else ['--b_max', str(args.b_max)]),
            outputs=[kernel_map, out_dir / 'kernel_map.csv', out_dir / 'kernel_map.png', out_dir / 'uniform_traj.json'],
        ),
        Stage(
            key='trajectory',
            label='Stage 3: trajectory fitting',
            command=[
                py, str(script_dir / 'trajectory_fitting.py'),
                '--kernel_map', str(kernel_map),
                '--blurry', str(blurry),
                '--out_dir', str(out_dir),
                '--skip_residual_image',
                '--outlier_sigma', str(args.outlier_sigma),
                '--weight_field', str(args.weight_field),
                '--ransac_iters', str(args.ransac_iters),
                '--ransac_min_samples', str(args.ransac_min_samples),
                '--ransac_residual_thres', str(args.ransac_residual_thres),
                '--ransac_seed', str(args.ransac_seed),
            ]
            + ([] if args.manual_blur_px is None else ['--manual_blur_px', str(args.manual_blur_px)])
            + ([] if args.pan_focal_px is None else ['--focal_px', str(args.pan_focal_px)])
            + ([] if args.pan_focal_mm is None else ['--focal_mm', str(args.pan_focal_mm)])
            + ([] if args.pan_sensor_width_mm is None else ['--sensor_width_mm', str(args.pan_sensor_width_mm)])
            + ([] if args.pan_exposure_s is None else ['--exposure_s', str(args.pan_exposure_s)])
            + (['--disable_ransac'] if args.disable_ransac else []),
            outputs=[trajectory_json, out_dir / 'trajectory_scatter.png', out_dir / 'kernel_contribution_map.png'],
        ),
        Stage(
            key='speed',
            label='Stage 4: car speed estimation',
            command=[
                py, str(script_dir / 'estimate_car_speed.py'),
                '--image', str(blurry),
                '--car_mask', str(car_mask),
                '--car_detection_json', str(car_detection_json),
                '--trajectory', str(trajectory_json),
                '--out_dir', str(out_dir),
                '--wheelbase_mm', str(args.wheelbase_mm),
                '--wheel_text_prompt', str(args.wheel_text_prompt),
                '--wheel_box_threshold', str(args.wheel_box_threshold),
                '--wheel_text_threshold', str(args.wheel_text_threshold),
                '--wheelbase_bbox_ratio_prior', str(args.wheelbase_bbox_ratio_prior),
            ]
            + ([] if args.wheelbase_px is None else ['--wheelbase_px', str(args.wheelbase_px)])
            + ([] if args.left_wheel is None else ['--left_wheel', *(str(v) for v in args.left_wheel)])
            + ([] if args.right_wheel is None else ['--right_wheel', *(str(v) for v in args.right_wheel)])
            + ([] if speed_focal_px is None else ['--focal_px', str(speed_focal_px)])
            + ([] if speed_focal_mm is None else ['--focal_mm', str(speed_focal_mm)])
            + ([] if speed_sensor_width_mm is None else ['--sensor_width_mm', str(speed_sensor_width_mm)]),
            outputs=[out_dir / 'car_speed.json', out_dir / 'car_speed_wheels.png', out_dir / 'final_result.png'],
        ),
    ]

    env = os.environ.copy()
    if args.cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
        device_msg = 'CPU'
    else:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device_msg = f'GPU {args.gpu}'

    chosen = selected_stages(stages, args.start_at, args.stop_after)
    print(f'Input: blurry={blurry} out_dir={out_dir}', flush=True)
    print(f'Device selection for child processes: {device_msg}', flush=True)

    status_by_stage: dict[str, str] = {}
    for stage in chosen:
        status_by_stage[stage.key] = run_stage(stage, env, args.force, args.dry_run)

    print('\n== Blind Pipeline Status ==', flush=True)
    for stage in chosen:
        print(f'{stage.key}: {status_by_stage[stage.key]}', flush=True)

    if not args.dry_run and any(stage.key in {'trajectory', 'speed'} for stage in chosen):
        print_summary(out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
