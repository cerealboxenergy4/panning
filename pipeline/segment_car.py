"""
Segment the F1 car and track/asphalt exclusion regions using Grounded SAM 2:
  1. Grounding DINO  — open-vocabulary detection with text prompt → bounding boxes
  2. SAM 2           — box-prompted segmentation → instance masks

Both models are loaded from HuggingFace (transformers >= 4.49).

Outputs:
  outputs/<image_stem>/car_mask.png        — binary mask (255=car, 0=background)
  outputs/<image_stem>/track_mask.png      — binary mask (255=track/asphalt, 0=other)
  outputs/<image_stem>/car_detection.png   — car visualization overlay with boxes + mask
  outputs/<image_stem>/track_detection.png — track/asphalt visualization overlay with boxes + mask
  outputs/<image_stem>/car_detection.json  — GroundingDINO boxes and selected car crop bbox
  outputs/<image_stem>/track_detection.json — GroundingDINO boxes for track/asphalt exclusion
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch


def run_grounding_dino(image_pil, text_prompt, box_threshold, text_threshold, device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"Loading Grounding DINO ({model_id}) ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    H, W = image_pil.height, image_pil.width
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(H, W)],
    )
    boxes = results[0]["boxes"].cpu().numpy()   # (N, 4) in xyxy, pixel coords
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0].get("text_labels", results[0].get("labels", []))
    print(f"Grounding DINO detections: {len(boxes)}")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        print(f"  [{i}] {label}  score={score:.3f}  box={box.round(1).tolist()}")
    return boxes, scores, labels


def run_sam2(image_pil, boxes_xyxy, device):
    from transformers import Sam2Processor, Sam2Model

    model_id = "facebook/sam2-hiera-small"
    print(f"\nLoading SAM 2 ({model_id}) ...")
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).to(device)
    model.eval()

    # Sam2Processor expects input_boxes as list[list[list[float]]]
    # Shape: [batch, num_boxes, 4]
    input_boxes = [boxes_xyxy.tolist()]

    inputs = processor(
        images=image_pil,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # post_process_masks returns list of tensors, one per batch item
    masks_list = processor.post_process_masks(
        outputs.pred_masks,
        original_sizes=inputs["original_sizes"],
    )
    # masks_list[0]: (num_boxes, 1, H, W) bool tensor
    masks = masks_list[0].squeeze(1).cpu().numpy()  # (N, H, W)
    print(f"SAM 2 masks: {masks.shape}  (N, H, W)")
    return masks


def mask_bbox(mask):
    if mask is None or not np.any(mask > 0):
        return None
    ys, xs = np.where(mask > 0)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def save_detection_json(path, image_path, image_size, text_prompt, box_threshold, text_threshold,
                        boxes, scores, labels, object_mask=None, object_name="car"):
    boxes_list = boxes.tolist() if len(boxes) else []
    detections = []
    for i, box in enumerate(boxes_list):
        detections.append({
            "index": i,
            "label": str(labels[i]) if i < len(labels) else "",
            "score": float(scores[i]) if i < len(scores) else None,
            "box_xyxy": [float(v) for v in box],
        })

    if len(boxes_list):
        arr = np.asarray(boxes_list, dtype=float)
        object_bbox = [
            float(arr[:, 0].min()),
            float(arr[:, 1].min()),
            float(arr[:, 2].max()),
            float(arr[:, 3].max()),
        ]
        primary = int(np.argmax(scores)) if len(scores) else 0
        source = "grounding_dino_union"
    else:
        object_bbox = mask_bbox(object_mask)
        primary = None
        source = "mask_fallback" if object_bbox else None

    meta = {
        "image": str(image_path),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "text_prompt": text_prompt,
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        f"{object_name}_bbox_source": source,
        f"{object_name}_bbox_xyxy": object_bbox,
        "mask_bbox_xyxy": mask_bbox(object_mask),
        "primary_detection_index": primary,
        "detections": detections,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {path}")


def save_detection_vis(image_pil, mask, boxes, out_path, color):
    img_np = np.array(image_pil, dtype=np.float32)
    overlay = img_np.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.4 + np.array(color, dtype=np.float32) * 0.6
    img_vis = np.clip(overlay, 0, 255).astype(np.uint8)
    H, W = mask.shape
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box
        x1 = int(np.clip(x1, 0, W - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        y2 = int(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        for t in range(3):
            img_vis[max(0, y1 + t), x1:x2] = color
            img_vis[min(H - 1, y2 - t), x1:x2] = color
            img_vis[y1:y2, max(0, x1 + t)] = color
            img_vis[y1:y2, min(W - 1, x2 - t)] = color
    Image.fromarray(img_vis).save(out_path)
    print(f"Saved: {out_path}")


def segment_prompt(image_pil, text_prompt, box_threshold, text_threshold, device, empty_msg):
    H, W = image_pil.height, image_pil.width
    boxes, scores, labels = run_grounding_dino(
        image_pil, text_prompt, box_threshold, text_threshold, device)
    if len(boxes) == 0:
        print(empty_msg)
        return np.zeros((H, W), dtype=np.uint8), boxes, scores, labels

    masks = run_sam2(image_pil, boxes, device)
    mask = masks.any(axis=0).astype(np.uint8)
    return mask, boxes, scores, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="pan_1.jpg")
    p.add_argument("--text_prompt", default="formula 1 racing car . f1 car . race car .",
                   help="Grounding DINO text prompt; use period-separated concepts")
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--track_text_prompt",
                   default="asphalt . road surface . race track surface . track . pavement .",
                   help="Prompt used to segment road/track/asphalt patches excluded from kernel estimation.")
    p.add_argument("--track_box_threshold", type=float, default=0.15)
    p.add_argument("--track_text_threshold", type=float, default=0.15)
    p.add_argument("--skip_track_segmentation", action="store_true",
                   help="Write an empty track mask instead of running road/track segmentation.")
    p.add_argument("--output_root", default="outputs",
                   help="Parent directory for per-image output subdirectories")
    p.add_argument("--out_dir", default=None,
                   help="Explicit output directory; defaults to <output_root>/<image_stem>")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    image_path = Path(args.image)
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.output_root) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image_pil = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    H, W = image_pil.height, image_pil.width
    print(f"Image: {args.image}  ({W}x{H})")

    # Step 1: car Grounding DINO -> SAM 2
    car_mask, car_boxes, car_scores, car_labels = segment_prompt(
        image_pil,
        args.text_prompt,
        args.box_threshold,
        args.text_threshold,
        device,
        "[warn] No car detections. Try lowering --box_threshold or adjusting --text_prompt. Saving empty car mask.",
    )
    n_car_px = int(car_mask.sum())
    print(f"\nCar mask: {n_car_px} px  ({100.0 * n_car_px / (H * W):.1f}% of image)")

    mask_path = out_dir / "car_mask.png"
    Image.fromarray(car_mask * 255).save(mask_path)
    print(f"Saved: {mask_path}")

    save_detection_json(
        out_dir / "car_detection.json",
        args.image,
        (W, H),
        args.text_prompt,
        args.box_threshold,
        args.text_threshold,
        car_boxes,
        car_scores,
        car_labels,
        car_mask,
        object_name="car",
    )
    save_detection_vis(image_pil, car_mask, car_boxes, out_dir / "car_detection.png", [60, 180, 255])

    # Step 2: track/asphalt Grounding DINO -> SAM 2
    if args.skip_track_segmentation:
        track_mask = np.zeros((H, W), dtype=np.uint8)
        track_boxes = np.zeros((0, 4), dtype=np.float32)
        track_scores = np.zeros((0,), dtype=np.float32)
        track_labels = []
        print("\nTrack mask: skipped; writing empty mask.")
    else:
        print("\nSegmenting track/asphalt exclusion mask ...")
        track_mask, track_boxes, track_scores, track_labels = segment_prompt(
            image_pil,
            args.track_text_prompt,
            args.track_box_threshold,
            args.track_text_threshold,
            device,
            "[warn] No track/asphalt detections. Saving empty track mask.",
        )
        # The car itself is handled by car_mask; do not let it inflate track overlap.
        track_mask[car_mask > 0] = 0

    n_track_px = int(track_mask.sum())
    print(f"Track/asphalt mask: {n_track_px} px  ({100.0 * n_track_px / (H * W):.1f}% of image)")
    track_mask_path = out_dir / "track_mask.png"
    Image.fromarray(track_mask * 255).save(track_mask_path)
    print(f"Saved: {track_mask_path}")

    save_detection_json(
        out_dir / "track_detection.json",
        args.image,
        (W, H),
        args.track_text_prompt,
        args.track_box_threshold,
        args.track_text_threshold,
        track_boxes,
        track_scores,
        track_labels,
        track_mask,
        object_name="track",
    )
    save_detection_vis(image_pil, track_mask, track_boxes, out_dir / "track_detection.png", [245, 190, 55])


if __name__ == "__main__":
    main()
