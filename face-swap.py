import argparse
from pathlib import Path
import os
import urllib.request


FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

CORE_FEATURE_INDICES = sorted(
    {
        6, 8, 9, 55, 65, 66, 70, 71, 105, 107,
        122, 129, 142, 168, 193, 197, 198, 217,
        285, 295, 296, 300, 301, 334, 336,
        351, 358, 371, 399, 420, 437,
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173,
        362, 382, 381, 380, 374, 373, 390, 249,
        466, 388, 387, 386, 385, 384, 398,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        17, 18, 200, 199, 175,
        1, 2, 4, 5, 45, 51, 48, 115, 98, 327, 330, 278, 275,
    }
)


def _read_kv_config(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    if not path.exists():
        return config

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key:
            config[key] = value

    return config


def _resolve_path(base_dir: Path, configured_path: str | None, default_path: str | None = None) -> Path:
    raw = (configured_path or "").strip() or (default_path or "")
    if not raw:
        raise RuntimeError("Missing required path in config")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FaceSwap/1.0", "Accept": "*/*"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_path)


def _ensure_model(path: Path, url: str) -> Path:
    if not path.exists():
        print(f"Downloading model: {path.name}")
        _download(url, path)
    return path


def _import_deps():
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        import numpy as np  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Install: pip install mediapipe opencv-contrib-python numpy"
        ) from exc
    return cv2, mp, np, vision, BaseOptions


def _landmarks_to_points(landmarks, width: int, height: int):
    points = []
    for landmark in landmarks:
        x = min(max(int(round(landmark.x * width)), 0), width - 1)
        y = min(max(int(round(landmark.y * height)), 0), height - 1)
        points.append((x, y))
    return points


def _bbox_area(points) -> int:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _create_face_landmarker(mp, vision, BaseOptions, model_path: Path, max_num_faces: int = 5):
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max_num_faces,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _detect_largest_face_points(mp, detector, image_path: Path):
    image = mp.Image.create_from_file(str(image_path))
    result = detector.detect(image)
    if not result.face_landmarks:
        raise RuntimeError(f"No face detected: {image_path}")

    image_h, image_w = image.numpy_view().shape[:2]
    candidates = [
        _landmarks_to_points(face_landmarks, image_w, image_h)
        for face_landmarks in result.face_landmarks
    ]
    return max(candidates, key=_bbox_area)


def _closest_point_index(points, point) -> int:
    best_index = -1
    best_dist = None
    px, py = point
    for index, (x, y) in enumerate(points):
        dist = (x - px) * (x - px) + (y - py) * (y - py)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_index = index
    return best_index


def _triangle_indices_from_points(cv2, hull_points, rect):
    subdiv = cv2.Subdiv2D(rect)
    for point in hull_points:
        subdiv.insert((int(point[0]), int(point[1])))

    triangle_list = subdiv.getTriangleList()
    if triangle_list is None:
        return []

    indices = []
    seen = set()
    x, y, w, h = rect
    for triangle in triangle_list:
        pts = [
            (int(round(triangle[0])), int(round(triangle[1]))),
            (int(round(triangle[2])), int(round(triangle[3]))),
            (int(round(triangle[4])), int(round(triangle[5]))),
        ]
        if not all(x <= px < x + w and y <= py < y + h for px, py in pts):
            continue

        tri_idx = tuple(_closest_point_index(hull_points, point) for point in pts)
        if len(set(tri_idx)) != 3:
            continue
        if tri_idx in seen:
            continue
        seen.add(tri_idx)
        indices.append(tri_idx)
    return indices


def _polygon_bounds(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return x0, y0, x1, y1


def _refine_mask(np, cv2, mask, polygon_points):
    x0, y0, x1, y1 = _polygon_bounds(polygon_points)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    kernel_size = max(3, ((min(width, height) // 18) | 1))
    blur_size = max(7, ((min(width, height) // 12) | 1))

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    refined = cv2.erode(mask, kernel, iterations=1)
    refined = cv2.GaussianBlur(refined, (blur_size, blur_size), 0)
    return refined


def _scaled_polygon(np, points, scale: float):
    pts = np.array(points, dtype=np.float32)
    center = pts.mean(axis=0)
    scaled = center + (pts - center) * float(scale)
    return [(int(round(x)), int(round(y))) for x, y in scaled]


def _apply_mask(np, image, mask):
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    return image.astype(np.float32) * mask_f


def _color_correct_face(np, cv2, warped_face, source_img, mask):
    warped_f = warped_face.astype(np.float32)
    source_f = source_img.astype(np.float32)
    active = mask > 0
    if not active.any():
        return warped_face

    corrected = warped_f.copy()
    for channel in range(3):
        src_vals = source_f[..., channel][active]
        warp_vals = warped_f[..., channel][active]
        src_mean, src_std = float(src_vals.mean()), float(src_vals.std())
        warp_mean, warp_std = float(warp_vals.mean()), float(warp_vals.std())
        if warp_std < 1e-6:
            corrected[..., channel][active] = src_mean
            continue
        corrected[..., channel] = (
            (corrected[..., channel] - warp_mean) * (src_std / warp_std)
        ) + src_mean

    corrected = np.clip(corrected, 0, 255).astype(warped_face.dtype)
    return corrected


def _warp_triangle(np, cv2, src_img, dst_img, src_tri, dst_tri):
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))

    src_rect_points = []
    dst_rect_points = []
    for i in range(3):
        src_rect_points.append(
            (src_tri[i][0] - src_rect[0], src_tri[i][1] - src_rect[1])
        )
        dst_rect_points.append(
            (dst_tri[i][0] - dst_rect[0], dst_tri[i][1] - dst_rect[1])
        )

    src_crop = src_img[
        src_rect[1] : src_rect[1] + src_rect[3],
        src_rect[0] : src_rect[0] + src_rect[2],
    ]

    warp_mat = cv2.getAffineTransform(
        np.float32(src_rect_points), np.float32(dst_rect_points)
    )
    warped = cv2.warpAffine(
        src_crop,
        warp_mat,
        (dst_rect[2], dst_rect[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_rect_points), (1.0, 1.0, 1.0), 16, 0)

    target_roi = dst_img[
        dst_rect[1] : dst_rect[1] + dst_rect[3],
        dst_rect[0] : dst_rect[0] + dst_rect[2],
    ]
    target_roi *= 1.0 - mask
    target_roi += warped * mask


def _swap_face(np, cv2, source_img, target_img, source_points, target_points):
    source_outline = [source_points[index] for index in FACE_OVAL_INDICES]
    source_core = [source_points[index] for index in CORE_FEATURE_INDICES]

    rect = cv2.boundingRect(np.float32([source_outline]))
    triangle_indices = _triangle_indices_from_points(cv2, source_points, rect)
    if not triangle_indices:
        raise RuntimeError("Failed to build face triangulation")

    warped_face = np.zeros_like(source_img, dtype=np.float32)
    for tri in triangle_indices:
        src_tri = [target_points[tri[0]], target_points[tri[1]], target_points[tri[2]]]
        dst_tri = [source_points[tri[0]], source_points[tri[1]], source_points[tri[2]]]
        _warp_triangle(np, cv2, target_img, warped_face, src_tri, dst_tri)

    mask = np.zeros(source_img.shape[:2], dtype=np.uint8)
    face_hull = cv2.convexHull(np.array(source_outline, dtype=np.int32))
    face_polygon = [tuple(point[0]) for point in face_hull]
    face_polygon = _scaled_polygon(np, face_polygon, 0.96)
    core_hull = cv2.convexHull(np.array(source_core, dtype=np.int32))
    core_polygon = [tuple(point[0]) for point in core_hull]
    core_polygon = _scaled_polygon(np, core_polygon, 1.04)
    cv2.fillPoly(mask, [np.int32(face_polygon)], 255)
    cv2.fillPoly(mask, [np.int32(core_polygon)], 255)
    clone_mask = _refine_mask(np, cv2, mask, face_polygon)

    warped_face_uint8 = np.clip(warped_face, 0, 255).astype(source_img.dtype)
    corrected_face = _color_correct_face(np, cv2, warped_face_uint8, source_img, clone_mask)
    corrected_face = _apply_mask(np, corrected_face, clone_mask).astype(source_img.dtype)

    x, y, w, h = cv2.boundingRect(np.float32([face_polygon]))
    center = (x + w // 2, y + h // 2)
    output = cv2.seamlessClone(
        corrected_face,
        source_img,
        clone_mask,
        center,
        cv2.MIXED_CLONE,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Swap target face onto source image")
    parser.add_argument("--config", default="face-swap.config", help="Config file path")
    parser.add_argument("--source", default="", help="Base image path")
    parser.add_argument("--target", default="", help="Reference face image path")
    parser.add_argument("--output", default="", help="Output image path")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config = _read_kv_config(_resolve_path(base_dir, args.config))

    source_path = _resolve_path(base_dir, args.source or config.get("Source"))
    target_path = _resolve_path(base_dir, args.target or config.get("Target"))

    default_output = source_path.with_name(source_path.stem + "-swap" + source_path.suffix)
    output_path = _resolve_path(base_dir, args.output or config.get("Output"), str(default_output))
    output_path = _unique_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2, mp, np, vision, BaseOptions = _import_deps()

    models_dir = (base_dir / "models").resolve()
    landmarker_path = _ensure_model(
        models_dir / "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    )
    detector = _create_face_landmarker(mp, vision, BaseOptions, landmarker_path)

    source_img = cv2.imread(str(source_path))
    target_img = cv2.imread(str(target_path))
    if source_img is None:
        raise RuntimeError(f"Failed to read source image: {source_path}")
    if target_img is None:
        raise RuntimeError(f"Failed to read target image: {target_path}")

    source_points = _detect_largest_face_points(mp, detector, source_path)
    target_points = _detect_largest_face_points(mp, detector, target_path)

    output = _swap_face(np, cv2, source_img, target_img, source_points, target_points)
    ok = cv2.imwrite(str(output_path), output)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
