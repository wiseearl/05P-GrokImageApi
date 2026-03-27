import argparse
import json
import mimetypes
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


STABILITY_API_BASE = "https://api.stability.ai"


def _read_kv_config(path: str) -> dict[str, str]:
    config: dict[str, str] = {}
    if not os.path.exists(path):
        return config

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
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


def _render_prompt_template(template_text: str, variables: dict[str, str]) -> str:
    rendered = template_text
    for key, value in variables.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _resolve_config_path(
    base_dir: Path, configured_path: str | None, default_relative_path: str
) -> Path:
    raw = (configured_path or "").strip()
    if not raw:
        raw = default_relative_path

    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _resolve_existing_input_path(configured_path: str) -> Path:
    candidate = Path(configured_path).resolve()
    if candidate.exists():
        return candidate

    parent = candidate.parent
    stem = candidate.stem
    if parent.exists():
        matches = sorted(path for path in parent.glob(f"{stem}.*") if path.is_file())
        if len(matches) == 1:
            return matches[0].resolve()

    return candidate


def _load_prompt_from_config(base_dir: Path, config: dict[str, str]) -> tuple[str, Path]:
    template_path = _resolve_config_path(
        base_dir, config.get("Prompt"), "prompt/c1.txt"
    )

    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read().strip()

    return _render_prompt_template(template_text, config), template_path


def _merge_prompt_affixes(
    prompt: str,
    prefix: str | None,
    suffix: str | None,
) -> str:
    parts: list[str] = []
    if prefix and prefix.strip():
        parts.append(prefix.strip())
    parts.append(prompt.strip())
    if suffix and suffix.strip():
        parts.append(suffix.strip())
    return ". ".join(part for part in parts if part)


def _pick_unique_output_path(out_dir: Path, stem: str, ext: str) -> Path:
    candidate = out_dir / f"{stem}.{ext}"
    if not candidate.exists():
        return candidate

    i = 1
    while True:
        candidate = out_dir / f"{stem}-{i}.{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def _read_api_key(api_key_file: str | None) -> str:
    api_key = os.environ.get("STABILITY_API_KEY", "").strip()
    if api_key:
        return api_key

    if not api_key_file:
        raise RuntimeError(
            "Missing API key. Set env var STABILITY_API_KEY, or pass --api-key-file."
        )

    with open(api_key_file, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    if not api_key:
        raise RuntimeError(
            f"API key file '{api_key_file}' is empty. Set STABILITY_API_KEY instead."
        )

    return api_key


def _parse_optional_int(raw: str | None) -> int | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as e:
        raise RuntimeError(f"Invalid integer value: {text}") from e


def _parse_optional_float(raw: str | None) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as e:
        raise RuntimeError(f"Invalid float value: {text}") from e


def _stringify_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


def _encode_multipart_formdata(
    fields: list[tuple[str, str]],
    files: list[tuple[str, Path]],
) -> tuple[str, bytes]:
    boundary = f"----StabilityBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    for name, path in files:
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "application/octet-stream"
        with open(path, "rb") as f:
            content = f.read()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", bytes(body)


def _decode_error_body(content_type: str, body: bytes) -> str:
    if not body:
        return ""

    text = body.decode("utf-8", errors="replace")
    if "application/json" not in (content_type or ""):
        return text[:2000]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:2000]

    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(str(item) for item in errors)[:2000]
    if isinstance(errors, str) and errors:
        return errors[:2000]

    message = payload.get("message") or payload.get("name")
    if message:
        return str(message)[:2000]

    return text[:2000]


def _http_post_multipart(
    url: str,
    api_key: str,
    fields: list[tuple[str, str]],
    files: list[tuple[str, Path]],
) -> tuple[bytes, str]:
    content_type, body = _encode_multipart_formdata(fields, files)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "image/*",
            "Content-Type": content_type,
            "User-Agent": "StabilityImageEdit/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read(), resp.headers.get_content_type()
    except urllib.error.HTTPError as e:
        err_body = e.read()
        err_type = e.headers.get_content_type() if e.headers else ""
        message = _decode_error_body(err_type, err_body)
        raise RuntimeError(f"HTTP {e.code}: {message}") from None


def _pick_output_ext(output_format: str, content_type: str | None = None) -> str:
    if output_format == "jpeg":
        return "jpg"
    if output_format in {"png", "webp"}:
        return output_format

    if content_type == "image/jpeg":
        return "jpg"
    if content_type == "image/webp":
        return "webp"
    return "png"


def _redact_event_value(value: str | None, max_len: int = 500) -> str | None:
    if value is None:
        return None
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."


def _append_run_log(base_dir: Path, event: dict) -> None:
    log_dir = base_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d')}.json"

    records: list[dict]
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                records = existing
            else:
                records = [{"_legacy": existing}]
        except Exception:
            records = []
    else:
        records = []

    records.append(event)

    tmp_path = log_path.with_name(log_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stability image generate/edit")
    parser.add_argument(
        "--config",
        default="config-image-sd.config",
        help="Config file path (default: config-image-sd.config)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input image path. Default: read from the selected config file (Source=...). "
            "If omitted, performs text-to-image."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Prompt text. Default: read prompt template from the selected config file (Prompt=...) "
            "and substitute variables from that config."
        ),
    )
    parser.add_argument(
        "--prompt-prefix",
        default=None,
        help="Prompt prefix appended before the rendered prompt.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default=None,
        help="Prompt suffix appended after the rendered prompt.",
    )
    parser.add_argument(
        "--out",
        default="",
        help=(
            "Output path. Default: same folder as input image when using --input, otherwise current project folder; "
            "filename is based on prompt template and auto-increments if needed."
        ),
    )
    parser.add_argument(
        "--edit-mode",
        choices=["auto", "generate", "search-and-replace"],
        default="auto",
        help=(
            "Edit mode. auto uses search-and-replace when SearchPrompt exists, otherwise generate."
        ),
    )
    parser.add_argument(
        "--search-prompt",
        default=None,
        help="Short description of the object/region to replace for search-and-replace mode.",
    )
    parser.add_argument(
        "--grow-mask",
        type=int,
        default=None,
        help="Search-and-replace grow_mask value from 0 to 20.",
    )
    parser.add_argument(
        "--service",
        choices=["auto", "core", "sd3"],
        default="auto",
        help=(
            "Stability service. auto = sd3 for image-to-image, core for text-to-image (default: auto)."
        ),
    )
    parser.add_argument(
        "--model",
        default="",
        help="SD3 model name, e.g. sd3.5-large, sd3.5-large-turbo, sd3.5-medium, sd3.5-flash.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Image-to-image strength from 0 to 1 (default: config Strength or 0.30).",
    )
    parser.add_argument(
        "--aspect-ratio",
        default=None,
        help="Aspect ratio such as 1:1, 2:3, 3:2, 16:9, 9:16.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Omit or use 0 for random.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="SD3 cfg_scale from 1 to 10.",
    )
    parser.add_argument(
        "--style-preset",
        default=None,
        help="Style preset such as photographic, anime, cinematic.",
    )
    parser.add_argument(
        "--output-format",
        choices=["jpeg", "png", "webp"],
        default=None,
        help="Output format (default: config OutputFormat or png).",
    )
    parser.add_argument(
        "--api-key-file",
        default=None,
        help=(
            "Fallback API key file. Default: read from the selected config file (Key=...). "
            "Prefer STABILITY_API_KEY env var."
        ),
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = _resolve_config_path(base_dir, args.config, "config-image-sd.config")
    config = _read_kv_config(str(config_path))

    if args.input is None:
        configured_source = (config.get("Source") or "").strip()
        if configured_source:
            args.input = str(_resolve_config_path(base_dir, configured_source, configured_source))

    input_path: Path | None = None
    if args.input:
        input_path = _resolve_existing_input_path(args.input)
        args.input = str(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

    if not args.prompt:
        args.prompt, prompt_template_path = _load_prompt_from_config(base_dir, config)
    else:
        prompt_template_path = _resolve_config_path(base_dir, config.get("Prompt"), "prompt/c1.txt")

    prompt_prefix = (
        args.prompt_prefix if args.prompt_prefix is not None else config.get("PromptPrefix")
    )
    prompt_suffix = (
        args.prompt_suffix if args.prompt_suffix is not None else config.get("PromptSuffix")
    )
    args.prompt = _merge_prompt_affixes(args.prompt, prompt_prefix, prompt_suffix)

    if not args.api_key_file:
        args.api_key_file = str(
            _resolve_config_path(base_dir, config.get("Key"), "key-sd-202603.txt")
        )

    search_prompt = (
        args.search_prompt if args.search_prompt is not None else config.get("SearchPrompt")
    )
    if search_prompt is not None:
        search_prompt = search_prompt.strip() or None

    grow_mask = args.grow_mask if args.grow_mask is not None else _parse_optional_int(config.get("GrowMask"))

    edit_mode = args.edit_mode
    if edit_mode == "auto":
        edit_mode = "search-and-replace" if input_path and search_prompt else "generate"

    if edit_mode == "search-and-replace" and not input_path:
        raise RuntimeError("search-and-replace requires an input image.")
    if edit_mode == "search-and-replace" and not search_prompt:
        raise RuntimeError("search-and-replace requires SearchPrompt or --search-prompt.")
    if grow_mask is not None and not 0 <= grow_mask <= 20:
        raise RuntimeError("grow_mask must be between 0 and 20.")

    service = args.service
    if service == "auto":
        service = "sd3" if input_path else "core"

    if service == "core" and input_path:
        raise RuntimeError(
            "Stable Image Core does not support image-to-image. Use --service sd3 or omit --input."
        )

    model = (args.model or config.get("Model") or "").strip()
    if service == "sd3" and not model:
        model = "sd3.5-large"

    strength = args.strength
    if strength is None:
        strength = _parse_optional_float(config.get("Strength"))
    if strength is None and input_path:
        strength = 0.30

    aspect_ratio = (args.aspect_ratio or config.get("AspectRatio") or "").strip() or None
    negative_prompt = (
        args.negative_prompt if args.negative_prompt is not None else config.get("NegativePrompt")
    )
    if negative_prompt is not None:
        negative_prompt = negative_prompt.strip() or None

    style_preset = (
        args.style_preset if args.style_preset is not None else config.get("StylePreset")
    )
    if style_preset is not None:
        style_preset = style_preset.strip() or None

    output_format = (
        args.output_format or (config.get("OutputFormat") or "").strip() or "png"
    )

    seed = args.seed if args.seed is not None else _parse_optional_int(config.get("Seed"))
    cfg_scale = (
        args.cfg_scale if args.cfg_scale is not None else _parse_optional_float(config.get("CfgScale"))
    )
    if cfg_scale is None and service == "sd3" and input_path:
        cfg_scale = 5.0

    if edit_mode == "generate" and input_path and strength is None:
        raise RuntimeError("Missing strength for image-to-image request.")

    if strength is not None and not 0 <= strength <= 1:
        raise RuntimeError("Strength must be between 0 and 1.")

    if cfg_scale is not None and not 1 <= cfg_scale <= 10:
        raise RuntimeError("cfg_scale must be between 1 and 10.")

    api_key = _read_api_key(args.api_key_file)

    fields: list[tuple[str, str]] = [("prompt", args.prompt), ("output_format", output_format)]
    files: list[tuple[str, Path]] = []

    if aspect_ratio:
        fields.append(("aspect_ratio", aspect_ratio))
    if negative_prompt:
        fields.append(("negative_prompt", negative_prompt))
    if style_preset:
        fields.append(("style_preset", style_preset))
    if seed is not None and seed > 0:
        fields.append(("seed", str(seed)))

    if edit_mode == "search-and-replace":
        url = f"{STABILITY_API_BASE}/v2beta/stable-image/edit/search-and-replace"
        assert input_path is not None
        assert search_prompt is not None
        files.append(("image", input_path))
        fields.append(("search_prompt", search_prompt))
        if grow_mask is not None:
            fields.append(("grow_mask", str(grow_mask)))
    elif service == "core":
        url = f"{STABILITY_API_BASE}/v2beta/stable-image/generate/core"
    else:
        url = f"{STABILITY_API_BASE}/v2beta/stable-image/generate/sd3"
        fields.append(("model", model))
        if cfg_scale is not None:
            fields.append(("cfg_scale", _stringify_number(cfg_scale)))
        if input_path:
            assert strength is not None
            fields.append(("mode", "image-to-image"))
            fields.append(("strength", _stringify_number(strength)))
            files.append(("image", input_path))

    out_dir = input_path.parent if input_path else base_dir
    if not args.out:
        stem = prompt_template_path.stem if prompt_template_path else "output"
        ext = _pick_output_ext(output_format)
        args.out = str(_pick_unique_output_path(out_dir, stem, ext))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    try:
        image_bytes, response_content_type = _http_post_multipart(url, api_key, fields, files)
    except Exception as e:
        _append_run_log(
            base_dir,
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ok": False,
                "service": service,
                "edit_mode": edit_mode,
                "url": url,
                "input": str(input_path) if input_path else None,
                "error": str(e),
                "model": model or None,
            },
        )
        raise

    with open(args.out, "wb") as f:
        f.write(image_bytes)

    _append_run_log(
        base_dir,
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ok": True,
            "service": service,
            "edit_mode": edit_mode,
            "url": url,
            "input": str(input_path) if input_path else None,
            "out": args.out,
            "model": model or None,
            "strength": strength,
            "aspect_ratio": aspect_ratio,
            "negative_prompt": _redact_event_value(negative_prompt),
            "style_preset": style_preset,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "search_prompt": search_prompt,
            "grow_mask": grow_mask,
            "output_format": output_format,
            "response_content_type": response_content_type,
            "prompt": _redact_event_value(args.prompt),
            "prompt_template": str(prompt_template_path) if prompt_template_path else None,
        },
    )

    print(args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)