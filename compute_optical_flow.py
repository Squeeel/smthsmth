"""
Compute TVL1 optical flow for consecutive frame pairs in a dataset.

Input structure:
  frames/{split}/{class_folder (optional)}/{video_folder}/frame_000.jpg ...

Outputs mirror the input structure under a separate output root
(default: optical_flow/), preserving {split}/{class}/{video}/ layout:
  flow_x_000.jpg, flow_y_000.jpg  (000->001)
  flow_x_001.jpg, flow_y_001.jpg  (001->002)
  flow_x_002.jpg, flow_y_002.jpg  (002->003)
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


CLIP_RANGE = 20.0  # TVL1 values clipped to [-20, 20] before [0,255] normalization


def flow_to_img(component: np.ndarray) -> np.ndarray:
    """Normalize a single flow component to uint8 [0, 255]."""
    clipped = np.clip(component, -CLIP_RANGE, CLIP_RANGE)
    return ((clipped + CLIP_RANGE) / (2 * CLIP_RANGE) * 255).astype(np.uint8)


def _make_tvl1():
    # opencv-contrib-python exposes cv2.optflow; plain opencv-python exposes cv2.createOptFlow_DualTVL1
    if hasattr(cv2, "optflow"):
        return cv2.optflow.DualTVL1OpticalFlow_create()
    if hasattr(cv2, "createOptFlow_DualTVL1"):
        return cv2.createOptFlow_DualTVL1()
    raise RuntimeError(
        "TVL1 optical flow not found. Install opencv-contrib-python:\n"
        "  pip uninstall opencv-python && pip install opencv-contrib-python"
    )


def compute_tvl1(img1_gray: np.ndarray, img2_gray: np.ndarray) -> np.ndarray:
    return _make_tvl1().calc(img1_gray, img2_gray, None)


def process_video_folder(task: tuple) -> str | None:
    video_dir, output_dir = task

    frames = sorted(video_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        return f"SKIP (< 2 frames): {video_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(frames) - 1):
        img1 = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frames[i + 1]), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return f"ERROR reading frames in {video_dir}"

        flow = compute_tvl1(img1, img2)

        flow_x = flow_to_img(flow[..., 0])
        flow_y = flow_to_img(flow[..., 1])

        cv2.imwrite(str(output_dir / f"flow_x_{i:03d}.jpg"), flow_x)
        cv2.imwrite(str(output_dir / f"flow_y_{i:03d}.jpg"), flow_y)

    return None


def main():
    parser = argparse.ArgumentParser(description="Compute TVL1 optical flow over a frames dataset.")
    parser.add_argument("--input", default="data/frames", help="Root directory of input frames")
    parser.add_argument("--output", default="data/optical_flow", help="Root directory for flow outputs (mirrors input layout)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    video_dirs = [d for d in sorted(input_root.rglob("*")) if d.is_dir() and any(d.glob("frame_*.jpg"))]

    if not video_dirs:
        raise RuntimeError(f"No video folders found under {input_root}")

    print(f"Found {len(video_dirs)} video folder(s) — using {args.workers} worker(s)")
    print(f"Writing flows to: {output_root}")

    tasks = [(d, output_root / d.relative_to(input_root)) for d in video_dirs]

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(executor.map(process_video_folder, tasks), total=len(tasks), unit="video"):
            if result is not None:
                errors.append(result)

    if errors:
        print(f"\n{len(errors)} issue(s):")
        for e in errors:
            print(f"  {e}")
    else:
        print("Done — all flows computed successfully.")


if __name__ == "__main__":
    main()
