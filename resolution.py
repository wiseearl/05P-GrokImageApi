import argparse
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

REALESRGAN_ZIP_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
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


def _resolve_path(base_dir: Path, configured_path: str | None) -> Path:
    if not configured_path:
        raise RuntimeError("Missing path in config")
    p = Path(configured_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _ensure_realesrgan(tools_dir: Path) -> Path:
    exe_path = tools_dir / "realesrgan-ncnn-vulkan.exe"
    models_dir = tools_dir / "models"
    if exe_path.exists() and models_dir.exists():
        return exe_path

    print("Downloading Real-ESRGAN-ncnn-vulkan (first run only)...")
    tools_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tools_dir.parent / "realesrgan-ncnn-vulkan.zip.tmp"
    req = urllib.request.Request(
        REALESRGAN_ZIP_URL,
        headers={"User-Agent": "Resolution/1.0"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    zip_path.write_bytes(data)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tools_dir)
    finally:
        zip_path.unlink(missing_ok=True)

    if not exe_path.exists():
        raise RuntimeError(f"Extraction did not produce expected exe: {exe_path}")

    return exe_path


def _default_output_path(source_path: Path, rate: str, model: str) -> Path:
    suffix = f"_x{rate}"
    if "anime" in model:
        suffix += "-anime"
    return source_path.with_name(f"{source_path.stem}{suffix}{source_path.suffix}")


def _pick_exe_scale(models_dir: Path, model: str, rate: str) -> str:
    """Pick the scale to hand to the exe.

    Models that ship a dedicated per-scale file (e.g. realesr-animevideov3-x2)
    upscale natively at that rate. Models that only ship an x4 file (e.g.
    realesrgan-x4plus) go through the exe's internal non-native-scale resize
    path when asked for -s 2/3, which is broken in this build and produces
    a tiled/quilted image. For those, always run at native x4 and downscale
    in Python afterwards.
    """
    if (models_dir / f"{model}-x{rate}.param").exists():
        return rate
    return "4"


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upscale an image's resolution using Real-ESRGAN (ncnn-vulkan)"
    )
    parser.add_argument(
        "--config",
        default="config-resolution.txt",
        help="Config file path (default: config-resolution.txt)",
    )
    parser.add_argument("--file", default="", help="Override source image path")
    parser.add_argument("--rate", default="", help="Override upscale rate (2, 3, or 4)")
    parser.add_argument("--output", default="", help="Output image path")
    parser.add_argument(
        "--model",
        default="realesrgan-x4plus",
        help=(
            "Model name (default: realesrgan-x4plus). Other options: "
            "realesrgan-x4plus-anime, realesr-animevideov3"
        ),
    )
    parser.add_argument(
        "--tools-dir",
        default="tools/realesrgan-ncnn-vulkan",
        help="Directory holding (or to receive) the realesrgan-ncnn-vulkan tool",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = _resolve_path(base_dir, args.config)
    config = _read_kv_config(config_path)

    source_path = _resolve_path(base_dir, args.file or config.get("file"))
    rate = args.rate or config.get("rate") or "2"
    if rate not in ("2", "3", "4"):
        raise RuntimeError(f"Unsupported rate '{rate}': must be 2, 3, or 4")

    if not source_path.exists():
        raise RuntimeError(f"Source image not found: {source_path}")

    output_path = (
        _resolve_path(base_dir, args.output)
        if args.output
        else _default_output_path(source_path, rate, args.model)
    )

    tools_dir = _resolve_path(base_dir, args.tools_dir)
    exe_path = _ensure_realesrgan(tools_dir)
    models_dir = tools_dir / "models"
    exe_scale = _pick_exe_scale(models_dir, args.model, rate)

    print(f"Source: {source_path}")
    print(f"Rate:   x{rate}")
    print(f"Model:  {args.model}")
    print(f"Output: {output_path}")
    if exe_scale != rate:
        print(f"(no native x{rate} model for {args.model}; rendering at x{exe_scale} then downscaling)")

    start = time.time()
    render_path = output_path if exe_scale == rate else Path(
        tempfile.mktemp(suffix=source_path.suffix, dir=str(output_path.parent))
    )
    try:
        result = subprocess.run(
            [
                str(exe_path),
                "-i", str(source_path),
                "-o", str(render_path),
                "-s", exe_scale,
                "-n", args.model,
                "-m", str(models_dir),
            ],
            cwd=str(tools_dir),
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            raise RuntimeError(f"realesrgan-ncnn-vulkan exited with code {result.returncode}")

        if exe_scale != rate:
            cv2 = _import_cv2()
            img = cv2.imread(str(render_path))
            if img is None:
                raise RuntimeError(f"Failed to read rendered image: {render_path}")
            src_h, src_w = cv2.imread(str(source_path)).shape[:2]
            target_w, target_h = src_w * int(rate), src_h * int(rate)
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(output_path), resized)
    finally:
        if render_path != output_path:
            render_path.unlink(missing_ok=True)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s -> {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
