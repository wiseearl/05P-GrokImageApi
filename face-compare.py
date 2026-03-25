import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path


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


def _resolve_path(base_dir: Path, configured_path: str | None) -> Path:
    if not configured_path:
        raise RuntimeError("Missing path in config")
    p = Path(configured_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FaceCompare/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_path)


def _ensure_model(path: Path, url: str, sha256: str | None = None) -> Path:
    if not path.exists():
        print(f"Downloading model: {path.name}")
        _download(url, path)

    if sha256:
        got = _sha256_file(path)
        if got.lower() != sha256.lower():
            raise RuntimeError(
                f"Model checksum mismatch for {path} (got {got}, expected {sha256})"
            )

    return path


def _import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        print(
            "Missing dependency: opencv-contrib-python\n"
            "Install: pip install opencv-contrib-python\n",
            file=sys.stderr,
        )
        raise


def _largest_face(faces) -> object:
    # faces: Nx15 (x,y,w,h,score,lmks...)
    best = None
    best_area = -1.0
    for face in faces:
        x, y, w, h = face[0], face[1], face[2], face[3]
        area = float(w) * float(h)
        if area > best_area:
            best_area = area
            best = face
    return best


def _extract_feature(cv2, detector, recognizer, image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _ret, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        raise RuntimeError(f"No face detected in: {image_path}")

    face = _largest_face(faces)
    aligned = recognizer.alignCrop(img, face)
    feat = recognizer.feature(aligned)
    return feat


def _cosine_to_score(sim: float) -> float:
    # OpenCV SFace cosine similarity is typically in [0, 1]. Clamp and scale to 0~100.
    if sim != sim:  # NaN
        return 0.0
    sim = max(0.0, min(1.0, float(sim)))
    return sim * 100.0


def _calibrated_score(sim: float, mid: float, k: float) -> float:
    """Map cosine similarity to a more 'confidence-like' 0~100 score.

    This is a heuristic calibration curve (sigmoid). It is NOT comparable across
    different models/vendors unless calibrated on your own validation set.
    """
    if sim != sim:  # NaN
        return 0.0
    x = max(0.0, min(1.0, float(sim)))
    # score = 100 / (1 + exp(-k * (x - mid)))
    try:
        z = -float(k) * (x - float(mid))
        # clamp z to avoid overflow in exp
        z = max(-60.0, min(60.0, z))
        return 100.0 / (1.0 + math.exp(z))
    except Exception:
        return 0.0


def _float_from_config(config: dict[str, str], key: str, default: float) -> float:
    raw = (config.get(key) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two faces and output similarity score (0~100)"
    )
    parser.add_argument(
        "--config",
        default="compare.config",
        help="Config file path (default: compare.config)",
    )
    parser.add_argument("--source", default="", help="Override Source image path")
    parser.add_argument("--target", default="", help="Override Target image path")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store/download models (default: models)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON to stdout",
    )
    parser.add_argument(
        "--score-mode",
        choices=["raw", "calibrated", "both"],
        default="both",
        help=(
            "Which score to print. raw = cosine*100; calibrated = sigmoid-mapped 0~100; "
            "both = print both (default)."
        ),
    )
    parser.add_argument(
        "--calib-mid",
        type=float,
        default=None,
        help="Calibration midpoint for calibrated score (default: from config or 0.60)",
    )
    parser.add_argument(
        "--calib-k",
        type=float,
        default=None,
        help="Calibration steepness for calibrated score (default: from config or 20)",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = _resolve_path(base_dir, args.config)
    config = _read_kv_config(config_path)

    source_path = _resolve_path(base_dir, args.source or config.get("Source"))
    target_path = _resolve_path(base_dir, args.target or config.get("Target"))

    cv2 = _import_cv2()

    models_dir = _resolve_path(base_dir, args.models_dir)

    # Models from OpenCV Zoo (Apache-2.0 / permissive). Download if missing.
    yunet_path = models_dir / "face_detection_yunet_2023mar.onnx"
    sface_path = models_dir / "face_recognition_sface_2021dec.onnx"

    yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

    _ensure_model(yunet_path, yunet_url)
    _ensure_model(sface_path, sface_url)

    detector = cv2.FaceDetectorYN.create(
        str(yunet_path),
        "",
        (320, 320),
        0.9,
        0.3,
        5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")

    feat1 = _extract_feature(cv2, detector, recognizer, source_path)
    feat2 = _extract_feature(cv2, detector, recognizer, target_path)

    sim = float(recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))
    raw_score = _cosine_to_score(sim)

    calib_mid = args.calib_mid
    if calib_mid is None:
        calib_mid = _float_from_config(config, "CalibMid", 0.60)
    calib_k = args.calib_k
    if calib_k is None:
        calib_k = _float_from_config(config, "CalibK", 20.0)
    calibrated = _calibrated_score(sim, calib_mid, calib_k)

    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(source_path),
        "target": str(target_path),
        "cosine_similarity": sim,
        "score_raw_0_100": raw_score,
        "score_calibrated_0_100": calibrated,
        "calibration": {
            "mode": "sigmoid",
            "mid": calib_mid,
            "k": calib_k,
        },
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if args.score_mode == "raw":
            print(f"Score(raw): {raw_score:.2f}")
        elif args.score_mode == "calibrated":
            print(f"Score(calibrated): {calibrated:.2f}")
        else:
            print(f"Score(calibrated): {calibrated:.2f}")
            print(f"Score(raw): {raw_score:.2f}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
