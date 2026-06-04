"""
Segment the F1 car using Grounded SAM 2:
  1. Grounding DINO  — open-vocabulary detection with text prompt → bounding boxes
  2. SAM 2           — box-prompted segmentation → instance masks

Both models are loaded from HuggingFace (transformers ≥ 4.49).

Outputs:
  outputs/car_mask.png       — binary mask (255=car, 0=background)
  outputs/car_detection.png  — visualization overlay with boxes + mask
"""

import argparse
import numpy as np
from pathlib import Path
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
    labels = results[0]["labels"]
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="pan_1.jpg")
    p.add_argument("--text_prompt", default="formula 1 racing car . f1 car . race car .",
                   help="Grounding DINO text prompt; use period-separated concepts")
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    image_pil = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    H, W = image_pil.height, image_pil.width
    print(f"Image: {args.image}  ({W}×{H})")

    # ── Step 1: Grounding DINO → boxes ───────────────────────────────────────
    boxes, _, _ = run_grounding_dino(
        image_pil, args.text_prompt,
        args.box_threshold, args.text_threshold, device)

    if len(boxes) == 0:
        print("[warn] No detections. Try lowering --box_threshold or adjusting --text_prompt.")
        print("       Saving empty mask.")
        Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(out_dir / "car_mask.png")
        image_pil.save(out_dir / "car_detection.png")
        return

    # ── Step 2: SAM 2 → masks ────────────────────────────────────────────────
    masks = run_sam2(image_pil, boxes, device)   # (N, H, W) bool

    # Union of all instance masks
    car_mask = masks.any(axis=0).astype(np.uint8)  # (H, W)
    n_car_px = int(car_mask.sum())
    print(f"\nCar mask: {n_car_px} px  ({100.0 * n_car_px / (H * W):.1f}% of image)")

    mask_path = out_dir / "car_mask.png"
    Image.fromarray(car_mask * 255).save(mask_path)
    print(f"Saved: {mask_path}")

    # ── Visualization ─────────────────────────────────────────────────────────
    img_np = np.array(image_pil, dtype=np.float32)
    overlay = img_np.copy()
    overlay[car_mask > 0] = overlay[car_mask > 0] * 0.4 + np.array([60, 180, 255]) * 0.6

    # Draw bounding boxes
    img_vis = np.clip(overlay, 0, 255).astype(np.uint8)
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box
        for t in range(3):
            img_vis[max(0, y1 + t), x1:x2] = [60, 180, 255]
            img_vis[min(H - 1, y2 - t), x1:x2] = [60, 180, 255]
            img_vis[y1:y2, max(0, x1 + t)] = [60, 180, 255]
            img_vis[y1:y2, min(W - 1, x2 - t)] = [60, 180, 255]

    vis_path = out_dir / "car_detection.png"
    Image.fromarray(img_vis).save(vis_path)
    print(f"Saved: {vis_path}")


if __name__ == "__main__":
    main()
