import argparse
import json
import mimetypes
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


PIXVERSE_API_BASE = "https://app-api.pixverse.ai/openapi/v2"
PIXVERSE_UPLOAD_MAX_BYTES = 20 * 1024 * 1024
PIXVERSE_ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
PIXVERSE_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


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


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _load_prompt_from_config(base_dir: Path, config: dict[str, str]) -> tuple[str, Path]:
    template_path = _resolve_config_path(
        base_dir, config.get("Prompt"), "prompt/c1.txt"
    )

    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read().strip()

    return _render_prompt_template(template_text, config), template_path


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
    api_key = os.environ.get("PIXVERSE_API_KEY", "").strip()
    if api_key:
        return api_key

    if not api_key_file:
        raise RuntimeError(
            "Missing API key. Set env var PIXVERSE_API_KEY, or pass --api-key-file."
        )

    with open(api_key_file, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    if not api_key:
        raise RuntimeError(
            f"API key file '{api_key_file}' is empty. Set PIXVERSE_API_KEY instead."
        )

    return api_key


def _http_post_json(url: str, api_key: str, trace_id: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "API-KEY": api_key,
            "Ai-trace-id": trace_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "PixVerseImageEdit/1.0",
        },
    )
    return _urlopen_json(req)


def _http_get_json(url: str, api_key: str, trace_id: str) -> dict:
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "API-KEY": api_key,
            "Ai-trace-Id": trace_id,
            "Accept": "application/json",
            "User-Agent": "PixVerseImageEdit/1.0",
        },
    )
    return _urlopen_json(req)


def _urlopen_json(req: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp_bytes = resp.read()
            return json.loads(resp_bytes.decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTP {e.code}: {err_body[:2000]}") from None


def _http_post_multipart_image(url: str, api_key: str, trace_id: str, image_path: str) -> dict:
    boundary = f"----PixVerseBoundary{uuid.uuid4().hex}"
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    filename = os.path.basename(image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(image_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={
            "API-KEY": api_key,
            "Ai-trace-id": trace_id,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            "User-Agent": "PixVerseImageEdit/1.0",
        },
    )
    return _urlopen_json(req)


def _http_post_image_url(url: str, api_key: str, trace_id: str, image_url: str) -> dict:
    boundary = f"----PixVerseBoundary{uuid.uuid4().hex}"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="image_url"\r\n\r\n')
    body.extend(image_url.encode("utf-8"))
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={
            "API-KEY": api_key,
            "Ai-trace-id": trace_id,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            "User-Agent": "PixVerseImageEdit/1.0",
        },
    )
    return _urlopen_json(req)


def _download_to_file(url: str, out_path: str) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PixVerseImageEdit/1.0",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        content = resp.read()
    with open(out_path, "wb") as f:
        f.write(content)


def _normalize_api_response(result: dict) -> dict:
    if not isinstance(result, dict):
        return {"ErrCode": -1, "ErrMsg": str(result), "Resp": {}}
    return result


def _ensure_ok(result: dict, action: str) -> dict:
    result = _normalize_api_response(result)
    err_code = result.get("ErrCode", -1)
    if err_code != 0:
        err_msg = result.get("ErrMsg", "Unknown error")
        raise RuntimeError(f"{action} failed: ErrCode={err_code}, ErrMsg={err_msg}")
    resp = result.get("Resp")
    if not isinstance(resp, dict):
        raise RuntimeError(f"{action} failed: missing Resp in {json.dumps(result)[:2000]}")
    return resp


def _guess_output_ext(result_url: str | None, fallback_path: Path) -> str:
    if result_url:
        parsed = urllib.parse.urlparse(result_url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return "jpg" if suffix == ".jpeg" else suffix.lstrip(".")

    fallback_suffix = fallback_path.suffix.lower()
    if fallback_suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "jpg" if fallback_suffix == ".jpeg" else fallback_suffix.lstrip(".")
    return "png"


def _guess_output_ext_from_input(input_value: str, result_url: str | None) -> str:
    if result_url:
        parsed = urllib.parse.urlparse(result_url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix in PIXVERSE_ALLOWED_SUFFIXES:
            return "jpg" if suffix == ".jpeg" else suffix.lstrip(".")

    if _is_http_url(input_value):
        parsed = urllib.parse.urlparse(input_value)
        suffix = Path(parsed.path).suffix.lower()
        if suffix in PIXVERSE_ALLOWED_SUFFIXES:
            return "jpg" if suffix == ".jpeg" else suffix.lstrip(".")
        return "png"

    return _guess_output_ext(result_url, Path(input_value))


def _validate_local_upload_image(image_path: str) -> None:
    suffix = Path(image_path).suffix.lower()
    if suffix not in PIXVERSE_ALLOWED_SUFFIXES:
        raise RuntimeError(
            "Unsupported input image format. PixVerse upload supports jpg, jpeg, png, webp."
        )

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in PIXVERSE_ALLOWED_MIME_TYPES:
        raise RuntimeError(
            "Unsupported input image MIME type. PixVerse upload supports image/jpeg, image/jpg, image/png, image/webp."
        )

    file_size = os.path.getsize(image_path)
    if file_size >= PIXVERSE_UPLOAD_MAX_BYTES:
        raise RuntimeError("Input image is too large. PixVerse upload requires file size less than 20MB.")


def _resolve_output_dir(base_dir: Path, input_value: str) -> Path:
    if _is_http_url(input_value):
        return base_dir
    return Path(input_value).resolve().parent


def _redact_api_response_for_log(result: dict) -> dict:
    try:
        text = json.dumps(result, ensure_ascii=False)
    except TypeError:
        return {"_raw": str(result)}
    if len(text) <= 4000:
        return result
    return {"_truncated": text[:4000]}


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


def _parse_int(raw: str | None, default: int | None = None) -> int | None:
    text = (raw or "").strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError as e:
        raise RuntimeError(f"Invalid integer value: {text}") from e


def _build_generate_payload(
    prompt: str,
    img_id: int,
    template_id: int | None,
    translate: int,
    original_umodel: int | None,
    prompt_generate_endpoint: str | None,
) -> tuple[str, dict, str]:
    if template_id is not None:
        return (
            f"{PIXVERSE_API_BASE}/image/template/generate",
            {
                "img_ids": [img_id],
                "template_id": template_id,
            },
            "template",
        )

    if not prompt_generate_endpoint:
        raise RuntimeError(
            "Prompt image generation endpoint is not documented in the public PixVerse API pages we validated. "
            "Set GenerateEndpoint=... in config or pass --generate-endpoint if you have the correct endpoint, "
            "or use TemplateId for the documented template generation flow."
        )

    payload: dict[str, int | str] = {
        "img_id": img_id,
        "prompt": prompt,
        "translate": translate,
    }
    if original_umodel is not None:
        payload["OriginalUmodel"] = original_umodel

    return (
        prompt_generate_endpoint,
        payload,
        "prompt",
    )


def _poll_image_result(
    image_id: int,
    api_key: str,
    poll_interval: float,
    timeout_seconds: float,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_resp: dict | None = None
    pending_statuses = {0, 2, 3, 4, 5, 6}
    failed_statuses = {7, 8}

    while True:
        trace_id = str(uuid.uuid4())
        result = _http_get_json(
            f"{PIXVERSE_API_BASE}/image/result/{image_id}",
            api_key,
            trace_id,
        )
        resp = _ensure_ok(result, "Get image result")
        last_resp = resp

        if resp.get("image_id") != image_id:
            raise RuntimeError(
                "Get image result returned mismatched image_id: "
                f"expected={image_id}, actual={resp.get('image_id')}"
            )

        status = resp.get("status")
        out_url = resp.get("url")
        if status == 1:
            if not isinstance(out_url, str) or not out_url:
                raise RuntimeError(
                    "Image generation completed but result url is missing: "
                    f"response={json.dumps(resp)[:2000]}"
                )
            return resp
        if status in failed_statuses:
            raise RuntimeError(
                f"Image generation failed: status={status}, response={json.dumps(resp)[:2000]}"
            )
        if not isinstance(status, int):
            raise RuntimeError(
                "Get image result returned invalid status: "
                f"status={status!r}, response={json.dumps(resp)[:2000]}"
            )
        if status not in pending_statuses:
            raise RuntimeError(
                "Get image result returned unknown status: "
                f"status={status}, response={json.dumps(resp)[:2000]}"
            )

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for PixVerse image generation result: "
                f"last_response={json.dumps(last_resp)[:2000]}"
            )

        time.sleep(poll_interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="PixVerse image edit / generation")
    parser.add_argument(
        "--config",
        default="config-image-pixverse.config",
        help="Config file path (default: config-image-pixverse.config)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input image path. Default: read from config Source=...",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. Default: load from config Prompt=... and substitute config variables.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output path. Default: next to input image, named by prompt file stem.",
    )
    parser.add_argument(
        "--template-id",
        type=int,
        default=None,
        help="PixVerse image template_id. If set, uses the official template generation endpoint.",
    )
    parser.add_argument(
        "--generate-endpoint",
        default="",
        help=(
            "Prompt mode generation endpoint override. "
            "Default: read GenerateEndpoint=... from config. "
            "Required for prompt mode because PixVerse public docs do not expose that endpoint clearly."
        ),
    )
    parser.add_argument(
        "--translate",
        type=int,
        default=-1,
        help="PixVerse translate flag for prompt generation mode. Default: read Translate=... from config, else 0.",
    )
    parser.add_argument(
        "--original-umodel",
        type=int,
        default=-1,
        help="Optional PixVerse OriginalUmodel for prompt generation mode.",
    )
    parser.add_argument(
        "--api-key-file",
        default=None,
        help="Fallback API key file. Default: read from config Key=... Prefer PIXVERSE_API_KEY env var.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=3.0,
        help="Result polling interval in seconds (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Result polling timeout in seconds (default: 300)",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = _resolve_config_path(base_dir, args.config, "config-image-pixverse.config")
    config = _read_kv_config(str(config_path))

    if not args.input:
        configured_source = (config.get("Source") or "").strip()
        if configured_source and _is_http_url(configured_source):
            args.input = configured_source
        else:
            args.input = str(
                _resolve_config_path(base_dir, config.get("Source"), "images/pic-mr-Kamiki.jpg")
            )

    if _is_http_url(args.input):
        input_mode = "image_url"
    else:
        args.input = str(_resolve_existing_input_path(args.input))
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input image not found: {args.input}")
        _validate_local_upload_image(args.input)
        input_mode = "image"

    prompt_template_path: Path | None = None
    if args.prompt is None:
        args.prompt, prompt_template_path = _load_prompt_from_config(base_dir, config)

    if args.api_key_file is None:
        args.api_key_file = str(
            _resolve_config_path(base_dir, config.get("Key"), "key-pixverse-cyu202603.txt")
        )

    if args.template_id is None:
        args.template_id = _parse_int(config.get("TemplateId"), None)

    if args.translate < 0:
        args.translate = _parse_int(config.get("Translate"), 0) or 0

    if args.original_umodel < 0:
        args.original_umodel = _parse_int(config.get("OriginalUmodel"), None)

    if not args.generate_endpoint:
        args.generate_endpoint = (config.get("GenerateEndpoint") or "").strip()

    if args.template_id is None and not args.prompt:
        raise RuntimeError(
            "Missing prompt. Provide --prompt, or set Prompt=... in config, or use --template-id / TemplateId=."
        )

    api_key = _read_api_key(args.api_key_file)
    upload_trace_id = str(uuid.uuid4())
    if input_mode == "image_url":
        upload_result = _http_post_image_url(
            f"{PIXVERSE_API_BASE}/image/upload",
            api_key,
            upload_trace_id,
            args.input,
        )
    else:
        upload_result = _http_post_multipart_image(
            f"{PIXVERSE_API_BASE}/image/upload",
            api_key,
            upload_trace_id,
            args.input,
        )
    upload_resp = _ensure_ok(upload_result, "Upload image")

    img_id = upload_resp.get("img_id")
    if not isinstance(img_id, int):
        raise RuntimeError(
            f"Upload image failed: missing img_id in {json.dumps(upload_resp)[:2000]}"
        )

    generate_url, payload, mode = _build_generate_payload(
        args.prompt or "",
        img_id,
        args.template_id,
        args.translate,
        args.original_umodel,
        args.generate_endpoint,
    )
    generate_trace_id = str(uuid.uuid4())
    generate_result = _http_post_json(generate_url, api_key, generate_trace_id, payload)
    generate_resp = _ensure_ok(generate_result, "Generate image")

    image_id = generate_resp.get("image_id")
    if not isinstance(image_id, int):
        raise RuntimeError(
            f"Generate image failed: missing image_id in {json.dumps(generate_resp)[:2000]}"
        )

    result_resp = _poll_image_result(
        image_id=image_id,
        api_key=api_key,
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout,
    )

    out_url = result_resp.get("url")
    if not isinstance(out_url, str) or not out_url:
        raise RuntimeError(
            f"Image generation succeeded but result url is missing: {json.dumps(result_resp)[:2000]}"
        )

    source_dir = _resolve_output_dir(base_dir, args.input)
    if not args.out:
        if prompt_template_path is not None:
            stem = prompt_template_path.stem
        elif args.template_id is not None:
            stem = f"template-{args.template_id}"
        else:
            stem = "pixverse-output"
        ext = _guess_output_ext_from_input(args.input, out_url)
        args.out = str(_pick_unique_output_path(source_dir, stem, ext))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    _download_to_file(out_url, args.out)

    _append_run_log(
        base_dir,
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ok": True,
            "provider": "pixverse",
            "mode": mode,
            "input_mode": input_mode,
            "input": args.input,
            "prompt": args.prompt,
            "template_id": args.template_id,
            "translate": args.translate,
            "original_umodel": args.original_umodel,
            "generate_endpoint": args.generate_endpoint,
            "uploaded_img_id": img_id,
            "image_id": image_id,
            "out": args.out,
            "upload_response": _redact_api_response_for_log(upload_result),
            "generate_response": _redact_api_response_for_log(generate_result),
            "result_response": _redact_api_response_for_log({"Resp": result_resp, "ErrCode": 0, "ErrMsg": "Success"}),
        },
    )

    print(args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        base_dir = Path(__file__).resolve().parent
        try:
            _append_run_log(
                base_dir,
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "ok": False,
                    "provider": "pixverse",
                    "error": str(e),
                },
            )
        except Exception:
            pass
        raise