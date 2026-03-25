import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request


XAI_API_BASE = "https://api.x.ai/v1"


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


def main() -> int:
    parser = argparse.ArgumentParser(description="xAI Grok image-to-image edit")
    parser.add_argument(
        "--input",
        default=os.path.join("images", "pic-mr-Kamiki.jpg"),
        help="Input image path",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "keep face features and hair color. change cloth to an Elegant "
            "Japanese-inspired long gown, flowing silk fabric, delicate sakura floral "
            "patterns, soft pastel tones, wide sleeves, obi-style waist detail, "
            "graceful posture, cinematic lighting, ultra-detailed fabric texture, 8k, "
            "high fashion photography"
        ),
        help="Edit prompt",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output path. Default: images/output-<timestamp>.<ext>",
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
        default="image2image2026.txt",
        help="Fallback API key file (default: image2image2026.txt). Prefer XAI_API_KEY env var.",
    )

    args = parser.parse_args()

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
    result = _http_post_json(url, api_key, payload)

    data = result.get("data")
    if not data:
        raise RuntimeError(f"Unexpected response (missing data): {json.dumps(result)[:2000]}")

    item = data[0]

    if not args.out:
        ts = time.strftime("%Y%m%d-%H%M%S")
        ext = "png"
        mime_type = item.get("mime_type")
        if mime_type == "image/jpeg":
            ext = "jpg"
        elif mime_type == "image/webp":
            ext = "webp"
        args.out = os.path.join("images", f"output-{ts}.{ext}")

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

    print(args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
