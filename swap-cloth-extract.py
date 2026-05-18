import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


LABEL_MAP = {
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
}


def _get_label_ids(category: str) -> list[int]:
    normalized = category.strip().lower()
    if normalized == "upper_body":
        return [LABEL_MAP["upper_clothes"], LABEL_MAP["dress"]]
    if normalized == "lower_body":
        return [LABEL_MAP["skirt"], LABEL_MAP["pants"], LABEL_MAP["dress"]]
    if normalized == "dresses":
        return [
            LABEL_MAP["upper_clothes"],
            LABEL_MAP["skirt"],
            LABEL_MAP["pants"],
            LABEL_MAP["dress"],
        ]
    raise RuntimeError(f"Unsupported category: {category}")


def _fit_on_canvas(image: np.ndarray, mask: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("No clothing region detected in cloth reference image")

    left = max(int(xs.min()) - 10, 0)
    top = max(int(ys.min()) - 10, 0)
    right = min(int(xs.max()) + 11, image.shape[1])
    bottom = min(int(ys.max()) + 11, image.shape[0])

    cropped_image = image[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]
    if cropped_image.size == 0:
        raise RuntimeError("Detected clothing region is empty after crop")

    target_width, target_height = canvas_size
    scale = min(target_width / cropped_image.shape[1], target_height / cropped_image.shape[0])
    resized_width = max(1, int(round(cropped_image.shape[1] * scale)))
    resized_height = max(1, int(round(cropped_image.shape[0] * scale)))

    resized_image = cv2.resize(cropped_image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    resized_mask = cv2.resize(cropped_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2

    region = canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width]
    region[resized_mask > 0] = resized_image[resized_mask > 0]
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract clothing region from a worn-cloth reference image")
    parser.add_argument("--oot-root", required=True, help="OOTDiffusion repository root")
    parser.add_argument("--input", required=True, help="Input cloth image path")
    parser.add_argument("--output", required=True, help="Output extracted cloth image path")
    parser.add_argument("--category", required=True, help="OOTDiffusion category: upper_body/lower_body/dresses")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID for parsing model")
    args = parser.parse_args()

    oot_root = Path(args.oot_root).resolve()
    sys.path.insert(0, str(oot_root))

    from preprocess.humanparsing.run_parsing import Parsing

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise RuntimeError(f"Cloth reference image not found: {input_path}")

    source_image = Image.open(input_path).convert("RGB")
    parsing = Parsing(args.gpu_id)
    parsed_image, _face_mask = parsing(source_image.resize((384, 512)))
    parsed_array = np.array(parsed_image.resize(source_image.size, Image.NEAREST), dtype=np.uint8)

    label_ids = _get_label_ids(args.category)
    mask = np.isin(parsed_array, label_ids).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    image_bgr = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
    canvas_bgr = _fit_on_canvas(image_bgr, mask, source_image.size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(output_rgb).save(output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())