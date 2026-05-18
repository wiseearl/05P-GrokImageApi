import base64
import argparse
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


XAI_API_BASE = "https://api.x.ai/v1"


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
    raw = (configured_path or "").strip()
    if not raw:
        raise RuntimeError("Missing required path in config")
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_optional_path(base_dir: Path, configured_path: str | None) -> Path | None:
    raw = (configured_path or "").strip()
    if not raw:
        return None
    return _resolve_path(base_dir, raw)


def _resolve_existing_input_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()

    parent = path.parent
    stem = path.stem
    if parent.exists():
        matches = sorted(item for item in parent.glob(f"{stem}.*") if item.is_file())
        if len(matches) == 1:
            return matches[0].resolve()

    return path.resolve()


def _load_role_presets(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Missing role preset file: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("roles") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise RuntimeError(f"Invalid role preset format: {path}")

    presets: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        preset: dict[str, str] = {}
        for key, value in item.items():
            if isinstance(value, str):
                preset[str(key)] = value.strip()
        if preset.get("Name"):
            presets.append(preset)

    return presets


def _select_role_source(base_dir: Path, role_data_path: Path, source_name: str) -> Path:
    role_name = source_name.strip()
    if not role_name:
        raise RuntimeError("source is required in config-swap-cloth.config")

    presets = _load_role_presets(role_data_path)
    selected = next(
        (item for item in presets if item.get("Name", "").lower() == role_name.lower()),
        None,
    )
    if selected is None:
        raise RuntimeError(f"Source role not found in {role_data_path.name}: {role_name}")

    source_path = _resolve_path(base_dir, selected.get("Source"))
    source_path = _resolve_existing_input_path(source_path)
    if not source_path.exists():
        raise RuntimeError(f"Source image not found for role {role_name}: {source_path}")
    return source_path


def _build_output_path(source_path: Path, cloth_path: Path) -> Path:
    return source_path.parent / cloth_path.name


def _find_ootdiffusion_root(base_dir: Path, config: dict[str, str], cli_value: str | None) -> Path:
    candidates: list[Path] = []

    for raw in [
        cli_value,
        os.environ.get("OOTDIFFUSION_DIR", "").strip(),
        config.get("OOTDiffusionDir", "").strip(),
    ]:
        if raw:
            candidates.append(_resolve_path(base_dir, raw))

    candidates.extend(
        [
            (base_dir / "OOTDiffusion").resolve(),
            (base_dir.parent / "OOTDiffusion").resolve(),
            (base_dir.parent / "OOTDiffusion-main").resolve(),
        ]
    )

    for candidate in candidates:
        run_script = candidate / "run" / "run_ootd.py"
        if run_script.exists():
            return candidate

    raise RuntimeError(
        "Cannot locate OOTDiffusion. Set OOTDIFFUSION_DIR, add OOTDiffusionDir to config-swap-cloth.config, "
        "or place the repository at ../OOTDiffusion."
    )


def _find_ootdiffusion_python(oot_root: Path, config: dict[str, str], cli_value: str | None) -> str:
    for value in [
        cli_value,
        os.environ.get("OOTDIFFUSION_PYTHON", "").strip(),
        config.get("OOTDiffusionPython", "").strip(),
    ]:
        if value:
            python_path = Path(value).expanduser()
            if python_path.exists():
                return str(python_path.resolve())
            return value

    candidate_paths = [
        oot_root / ".venv" / "Scripts" / "python.exe",
        oot_root / "venv" / "Scripts" / "python.exe",
        oot_root / ".venv" / "bin" / "python",
        oot_root / "venv" / "bin" / "python",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate.resolve())

    return sys.executable


def _validate_ootdiffusion_assets(oot_root: Path) -> None:
    required_paths = [
        oot_root / "checkpoints" / "ootd",
        oot_root / "checkpoints" / "humanparsing" / "parsing_atr.onnx",
        oot_root / "checkpoints" / "humanparsing" / "parsing_lip.onnx",
        oot_root / "checkpoints" / "openpose" / "ckpts" / "body_pose_model.pth",
        oot_root / "checkpoints" / "clip-vit-large-patch14",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise RuntimeError(
            "OOTDiffusion checkpoints are missing. Download ootd, humanparsing, openpose, and clip-vit-large-patch14 into the OOTDiffusion/checkpoints folder before running.\n"
            f"Missing paths:\n{joined}"
        )


def _get_int(config: dict[str, str], key: str, default: int) -> int:
    raw = (config.get(key) or "").strip()
    if not raw:
        return default
    return int(raw)


def _get_float(config: dict[str, str], key: str, default: float) -> float:
    raw = (config.get(key) or "").strip()
    if not raw:
        return default
    return float(raw)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    raw = (value or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _get_category_name(config: dict[str, str]) -> str:
    model_type = (config.get("ModelType") or config.get("OOTDiffusionModelType") or "hd").strip().lower()
    category = _get_int(config, "Category", 0)

    if model_type == "hd":
        return "upper_body"

    category_map = {
        0: "upper_body",
        1: "lower_body",
        2: "dresses",
    }
    if category not in category_map:
        raise RuntimeError(f"Unsupported OOTDiffusion category: {category}")
    return category_map[category]


def _read_api_key(api_key_file: Path) -> str:
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if api_key:
        return api_key

    if not api_key_file.exists():
        raise FileNotFoundError(f"API key file not found: {api_key_file}")

    api_key = api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(f"API key file is empty: {api_key_file}")
    return api_key


def _file_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "application/octet-stream"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _http_post_json(url: str, api_key: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SwapCloth/1.0 (+https://api.x.ai)",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body_text[:2000]}") from None


def _extract_message_text(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Unexpected response: {json.dumps(payload, ensure_ascii=False)[:2000]}")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Unexpected response: {json.dumps(payload, ensure_ascii=False)[:2000]}")

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        text = "\n".join(part.strip() for part in text_parts if part.strip()).strip()
        if text:
            return text

    raise RuntimeError(f"Unexpected response: {json.dumps(payload, ensure_ascii=False)[:2000]}")


def _get_swap_mode(config: dict[str, str]) -> str:
    raw = (config.get("SwapMode") or "vton").strip().lower()
    if raw not in {"vton", "image_edit"}:
        raise RuntimeError(f"Unsupported SwapMode: {raw}")
    return raw


def _resolve_cloth_description(base_dir: Path, cloth_path: Path, config: dict[str, str]) -> str:
    cloth_description = (config.get("ClothDescription") or "").strip()
    auto_describe = _parse_bool(config.get("AutoClothDescription"), default=not cloth_description)
    if cloth_description and not auto_describe:
        return cloth_description

    key_path = _resolve_optional_path(base_dir, config.get("Key"))
    if key_path is None:
        key_path = (base_dir / "image2image2026.txt").resolve()
    api_key = _read_api_key(key_path)
    model = (config.get("ClothDescriptionModel") or "grok-4-fast-non-reasoning").strip()
    image_url = _file_to_data_url(cloth_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Describe only the main outfit visible in the image. "
                    "Return one concise English clothing description only, no markdown, no bullets, no explanation."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the outfit worn by the main person in this image as a concise clothing description for image editing. "
                            "Mention garment type, sleeve style, neckline, fit, length, and overall style. Ignore face, hair, body, background, shoes, and accessories."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
        "temperature": 0.2,
    }
    result = _http_post_json(f"{XAI_API_BASE}/chat/completions", api_key, payload)
    generated = _extract_message_text(result).strip()
    if not generated:
        raise RuntimeError("Auto cloth description returned empty text")
    return generated


def _build_worn_reference_prompt(cloth_description: str) -> str:
    return (
        "Keep the same woman, same face, same hairstyle, same body shape, same pose, same hands, "
        "same beach background, same camera angle, and same lighting. "
        "Replace only the clothing with a realistic outfit matching this description: "
        f"{cloth_description}. "
        "Preserve the original identity and anatomy. "
        "Use natural fabric structure, clean seams, coherent sleeves, and believable garment edges. "
        "Do not change the background, hair color, or facial features. "
        "Do not expose extra skin beyond what the described outfit would naturally cover."
    )


def _run_image_edit_mode(
    base_dir: Path,
    source_path: Path,
    cloth_path: Path,
    output_path: Path,
    config: dict[str, str],
) -> Path:
    image_edit_script = (base_dir / "image-edit.py").resolve()
    if not image_edit_script.exists():
        raise RuntimeError(f"Missing script: {image_edit_script}")

    key_path = _resolve_optional_path(base_dir, config.get("Key"))
    cloth_description = _resolve_cloth_description(base_dir, cloth_path, config)
    prompt = _build_worn_reference_prompt(cloth_description)

    command = [
        sys.executable,
        str(image_edit_script),
        "--config",
        "config-image-edit.config",
        "--input",
        str(source_path),
        "--out",
        str(output_path),
        "--prompt",
        prompt,
        "--resolution",
        (config.get("ImageEditResolution") or "2k").strip(),
        "--quality",
        (config.get("ImageEditQuality") or "high").strip(),
    ]
    if key_path is not None:
        command.extend(["--api-key-file", str(key_path)])

    completed = subprocess.run(
        command,
        cwd=str(base_dir),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        parts = [
            "[swap-cloth] image-edit mode failed",
            f"Exit code: {completed.returncode}",
            "Command:",
            _quote_command(command),
        ]
        stdout_tail = _tail_text(completed.stdout or "")
        stderr_tail = _tail_text(completed.stderr or "")
        if stdout_tail:
            parts.append("--- stdout (tail) ---")
            parts.append(stdout_tail)
        if stderr_tail:
            parts.append("--- stderr (tail) ---")
            parts.append(stderr_tail)
        raise RuntimeError("\n".join(parts).strip())

    if not output_path.exists():
        raise RuntimeError(f"image-edit mode completed without output: {output_path}")
    print(f"Cloth description: {cloth_description}")
    return output_path


def _detect_output_candidates(images_output_dir: Path) -> set[Path]:
    if not images_output_dir.exists():
        return set()
    return {item.resolve() for item in images_output_dir.glob("out_*.*") if item.is_file()}


def _pick_generated_output(images_output_dir: Path, before: set[Path], started_at: float) -> Path:
    candidates = [item.resolve() for item in images_output_dir.glob("out_*.*") if item.is_file()]
    recent = [item for item in candidates if item.stat().st_mtime >= started_at - 1 and item.resolve() not in before]
    if recent:
        return max(recent, key=lambda item: item.stat().st_mtime)
    if candidates:
        raise RuntimeError(
            "OOTDiffusion finished without producing a new output image. Existing out_* files are stale. "
            f"Output folder: {images_output_dir}"
        )
    raise RuntimeError(f"OOTDiffusion finished without creating output under {images_output_dir}")


def _clear_previous_outputs(images_output_dir: Path) -> None:
    for item in images_output_dir.glob("out_*.*"):
        if item.is_file():
            item.unlink()


def _extract_worn_cloth_reference(
    base_dir: Path,
    oot_root: Path,
    oot_python: str,
    cloth_path: Path,
    config: dict[str, str],
    gpu_id: int,
) -> Path:
    helper_script = (base_dir / "swap-cloth-extract.py").resolve()
    category_name = _get_category_name(config)

    with tempfile.NamedTemporaryFile(
        suffix=cloth_path.suffix or ".png",
        delete=False,
        dir=str(base_dir),
    ) as temp_file:
        extracted_path = Path(temp_file.name)

    command = [
        oot_python,
        str(helper_script),
        "--oot-root",
        str(oot_root),
        "--input",
        str(cloth_path),
        "--output",
        str(extracted_path),
        "--category",
        category_name,
        "--gpu-id",
        str(gpu_id),
    ]
    completed = subprocess.run(
        command,
        cwd=str(base_dir),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr_tail = _tail_text(completed.stderr or completed.stdout or "")
        raise RuntimeError(
            "Failed to extract clothing region from worn cloth reference image.\n"
            f"Command: {_quote_command(command)}\n{stderr_tail}"
        )
    if not extracted_path.exists():
        raise RuntimeError(f"Cloth extraction did not create output: {extracted_path}")
    return extracted_path


def _copy_or_convert_image(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == output_path.suffix.lower():
        shutil.copy2(source_path, output_path)
        return

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to convert OOTDiffusion output into the requested extension."
        ) from exc

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read generated image: {source_path}")

    params: list[int] = []
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    if not cv2.imwrite(str(output_path), image, params):
        raise RuntimeError(f"Failed to write output image: {output_path}")


def _quote_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _tail_text(text: str, max_lines: int = 120, max_chars: int = 12000) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _build_ootdiffusion_command(
    python_executable: str,
    run_script: Path,
    source_path: Path,
    cloth_path: Path,
    config: dict[str, str],
    gpu_id: int,
) -> list[str]:
    model_type = (config.get("ModelType") or config.get("OOTDiffusionModelType") or "hd").strip().lower()
    category = _get_int(config, "Category", 0)
    scale = _get_float(config, "Scale", 2.0)
    step = _get_int(config, "Step", 20)
    sample = _get_int(config, "Sample", 1)
    seed = _get_int(config, "Seed", 42)

    command = [
        python_executable,
        str(run_script),
        "--gpu_id",
        str(gpu_id),
        "--model_path",
        str(source_path),
        "--cloth_path",
        str(cloth_path),
        "--model_type",
        model_type,
        "--category",
        str(category),
        "--scale",
        str(scale),
        "--step",
        str(step),
        "--sample",
        str(sample),
        "--seed",
        str(seed),
    ]

    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Swap clothing with OOTDiffusion based on project config files")
    parser.add_argument("--config", default="config-swap-cloth.config", help="Swap-cloth config path")
    parser.add_argument(
        "--role-data",
        default="config-image-edit-swap.json",
        help="Role preset JSON path",
    )
    parser.add_argument("--ootdiffusion-dir", default="", help="Path to OOTDiffusion repository root")
    parser.add_argument("--ootdiffusion-python", default="", help="Python executable used to run OOTDiffusion")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID passed to OOTDiffusion")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = _resolve_path(base_dir, args.config)
    role_data_path = _resolve_path(base_dir, args.role_data)
    config = _read_kv_config(config_path)
    if not config:
        raise RuntimeError(f"Config is empty or missing: {config_path}")

    source_path = _select_role_source(base_dir, role_data_path, config.get("source") or config.get("Source") or "")
    cloth_path = _resolve_path(base_dir, config.get("cloth") or config.get("Cloth"))
    cloth_path = _resolve_existing_input_path(cloth_path)
    if not cloth_path.exists():
        raise RuntimeError(f"Cloth image not found: {cloth_path}")

    output_path = _build_output_path(source_path, cloth_path)

    swap_mode = _get_swap_mode(config)
    if swap_mode == "image_edit":
        generated_output = _run_image_edit_mode(base_dir, source_path, cloth_path, output_path, config)
        print(f"Source role: {config.get('source') or config.get('Source')}")
        print(f"Source image: {source_path}")
        print(f"Cloth image: {cloth_path}")
        print("Swap mode: image_edit")
        print(f"Generated image: {generated_output}")
        print(f"Output: {output_path}")
        return 0

    oot_root = _find_ootdiffusion_root(base_dir, config, args.ootdiffusion_dir)
    oot_python = _find_ootdiffusion_python(oot_root, config, args.ootdiffusion_python)
    _validate_ootdiffusion_assets(oot_root)
    run_script = oot_root / "run" / "run_ootd.py"
    images_output_dir = oot_root / "run" / "images_output"
    images_output_dir.mkdir(parents=True, exist_ok=True)

    effective_cloth_path = cloth_path
    extracted_cloth_path: Path | None = None
    if _parse_bool(config.get("ParseClothReference"), default=False):
        extracted_cloth_path = _extract_worn_cloth_reference(
            base_dir,
            oot_root,
            oot_python,
            cloth_path,
            config,
            args.gpu_id,
        )
        effective_cloth_path = extracted_cloth_path

    _clear_previous_outputs(images_output_dir)
    before_outputs = _detect_output_candidates(images_output_dir)
    command = _build_ootdiffusion_command(
        oot_python,
        run_script,
        source_path,
        effective_cloth_path,
        config,
        args.gpu_id,
    )

    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=str(run_script.parent),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        parts = [
            "[swap-cloth] OOTDiffusion failed",
            f"Exit code: {completed.returncode}",
            f"CWD: {run_script.parent}",
            "Command:",
            _quote_command(command),
        ]
        stdout_tail = _tail_text(completed.stdout or "")
        stderr_tail = _tail_text(completed.stderr or "")
        if stdout_tail:
            parts.append("--- stdout (tail) ---")
            parts.append(stdout_tail)
        if stderr_tail:
            parts.append("--- stderr (tail) ---")
            parts.append(stderr_tail)
        raise RuntimeError("\n".join(parts).strip())

    try:
        generated_output = _pick_generated_output(images_output_dir, before_outputs, started_at)
        _copy_or_convert_image(generated_output, output_path)
    finally:
        if extracted_cloth_path and extracted_cloth_path.exists():
            try:
                extracted_cloth_path.unlink()
            except OSError:
                pass

    print(f"Source role: {config.get('source') or config.get('Source')}")
    print(f"Source image: {source_path}")
    print(f"Cloth image: {cloth_path}")
    if extracted_cloth_path:
        print("Cloth preprocessing: enabled")
    print(f"Generated image: {generated_output}")
    print(f"Output: {output_path}")
    print(f"OOTDiffusion: {oot_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)