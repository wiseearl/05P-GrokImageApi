import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


XAI_API_BASE = "https://api.x.ai/v1"


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


def _pick_output_ext(item: dict) -> str:
    ext = "png"
    mime_type = item.get("mime_type")
    if mime_type == "image/jpeg":
        ext = "jpg"
    elif mime_type == "image/webp":
        ext = "webp"
    return ext


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
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if api_key:
        return api_key

    if not api_key_file:
        raise RuntimeError(
            "Missing API key. Set env var XAI_API_KEY, or pass --api-key-file."
        )

    with open(api_key_file, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    if not api_key:
        raise RuntimeError(
            f"API key file '{api_key_file}' is empty. Set XAI_API_KEY instead."
        )

    return api_key


def _file_to_data_url(path: str) -> tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "application/octet-stream"

    with open(path, "rb") as f:
        raw = f.read()

    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64}"
    return data_url, mime_type


def _http_post_json(url: str, api_key: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "GrokImageEdit/1.0 (+https://api.x.ai)",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp_bytes = resp.read()
            return json.loads(resp_bytes.decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Avoid printing secrets; show a compact server error.
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTP {e.code}: {err_body[:2000]}") from None


def _download_to_file(url: str, out_path: str) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "GrokImageEdit/1.0 (+https://api.x.ai)",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        content = resp.read()
    with open(out_path, "wb") as f:
        f.write(content)


def _redact_api_response_for_log(result: dict) -> dict:
    if not isinstance(result, dict):
        return {"_raw": str(result)}

    redacted: dict = {k: v for k, v in result.items() if k != "data"}
    data = result.get("data")
    if isinstance(data, list):
        new_data = []
        for item in data:
            if isinstance(item, dict):
                item2 = dict(item)
                if "b64_json" in item2:
                    b64_json = item2.get("b64_json") or ""
                    item2["b64_json"] = f"<omitted {len(b64_json)} chars>"
                new_data.append(item2)
            else:
                new_data.append(item)
        redacted["data"] = new_data
    else:
        redacted["data"] = data

    return redacted


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
    parser = argparse.ArgumentParser(description="xAI Grok image-to-image edit")
    parser.add_argument(
        "--config",
        default="config-image-edit.config",
        help="Config file path (default: config-image-edit.config)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input image path. Default: read from the selected config file (Source=...)."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Edit prompt. Default: read prompt template from the selected config file (Prompt=...) "
            "and substitute variables from that config (e.g. {SwitchColor})."
        ),
    )
    parser.add_argument(
        "--out",
        default="",
        help=(
            "Output path. Default: same folder as input image, with filename based on "
            "prompt template (e.g. Prompt=c1.txt -> c1.jpg). If exists, uses -1, -2..."
        ),
    )
    parser.add_argument(
        "--model",
        default="grok-imagine-image",
        help="Image model (default: grok-imagine-image)",
    )
    parser.add_argument(
        "--response-format",
        choices=["url", "b64_json"],
        default="b64_json",
        help="Return url or b64_json (default: b64_json)",
    )
    parser.add_argument(
        "--resolution",
        choices=["1k", "2k"],
        default="1k",
        help="Resolution (default: 1k)",
    )
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="high",
        help="Quality (default: high)",
    )
    parser.add_argument(
        "--api-key-file",
        default=None,
        help=(
            "Fallback API key file. Default: read from the selected config file (Key=...). "
            "Prefer XAI_API_KEY env var."
        ),
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = _resolve_config_path(base_dir, args.config, "config-image-edit.config")
    config = _read_kv_config(str(config_path))

    if not args.input:
        args.input = str(
            _resolve_config_path(base_dir, config.get("Source"), "images/pic-mr-Kamiki.jpg")
        )

    args.input = str(_resolve_existing_input_path(args.input))

    if not args.prompt:
        args.prompt, prompt_template_path = _load_prompt_from_config(base_dir, config)
    else:
        prompt_template_path = _resolve_config_path(base_dir, config.get("Prompt"), "prompt/c1.txt")

    if not args.api_key_file:
        args.api_key_file = str(
            _resolve_config_path(base_dir, config.get("Key"), "image2image2026.txt")
        )

    source_dir = Path(args.input).resolve().parent

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    api_key = _read_api_key(args.api_key_file)

    image_url, _mime = _file_to_data_url(args.input)
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "image": {"url": image_url, "type": "image_url"},
        "response_format": args.response_format,
        "resolution": args.resolution,
        "quality": args.quality,
    }

    url = f"{XAI_API_BASE}/images/edits"
    try:
        result = _http_post_json(url, api_key, payload)
    except Exception as e:
        _append_run_log(
            base_dir,
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ok": False,
                "error": str(e),
                "model": args.model,
                "response_format": args.response_format,
                "resolution": args.resolution,
                "quality": args.quality,
                "input": args.input,
            },
        )
        raise

    data = result.get("data")
    if not data:
        raise RuntimeError(f"Unexpected response (missing data): {json.dumps(result)[:2000]}")

    item = data[0]

    if not args.out:
        ext = _pick_output_ext(item)
        stem = prompt_template_path.stem if prompt_template_path else "output"
        args.out = str(_pick_unique_output_path(source_dir, stem, ext))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.response_format == "b64_json":
        b64_json = item.get("b64_json")
        if not b64_json:
            raise RuntimeError(
                f"Expected b64_json in response, got: {json.dumps(item)[:2000]}"
            )
        img_bytes = base64.b64decode(b64_json)
        with open(args.out, "wb") as f:
            f.write(img_bytes)
    else:
        out_url = item.get("url")
        if not out_url:
            raise RuntimeError(
                f"Expected url in response, got: {json.dumps(item)[:2000]}"
            )
        _download_to_file(out_url, args.out)

    _append_run_log(
        base_dir,
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ok": True,
            "model": args.model,
            "response_format": args.response_format,
            "resolution": args.resolution,
            "quality": args.quality,
            "input": args.input,
            "out": args.out,
            "api_response": _redact_api_response_for_log(result),
        },
    )

    print(args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
