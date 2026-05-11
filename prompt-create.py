import argparse
import json
import os
from pathlib import Path
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


def _resolve_path(base_dir: Path, configured_path: str | None, default_path: str) -> Path:
    raw = (configured_path or "").strip() or default_path
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


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
            "User-Agent": "GrokPromptCreate/1.0 (+https://api.x.ai)",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body_text}") from None


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


def _extract_structured_prompts(payload: dict, prompt_number: int) -> list[str]:
    message_text = _extract_message_text(payload)

    try:
        data = json.loads(message_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Structured output is not valid JSON: {message_text[:2000]}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Structured output must be a JSON object: {message_text[:2000]}")

    prompts = data.get("prompts")
    if not isinstance(prompts, list):
        raise RuntimeError(f"Structured output missing prompts array: {message_text[:2000]}")

    cleaned_prompts = [item.strip() for item in prompts if isinstance(item, str) and item.strip()]
    if len(cleaned_prompts) != prompt_number:
        raise RuntimeError(
            f"Expected {prompt_number} prompts from structured output, got {len(cleaned_prompts)}"
        )

    return cleaned_prompts


def _build_user_prompt(prompt_hint: str, prompt_number: int) -> str:
    return (
        f"Generate {prompt_number} different image creation prompts in English based on this hint. "
        "Each prompt must be detailed, vivid, and suitable for an image model. "
        "Keep each prompt as plain prompt text only, with no numbering, no title, no explanation, no quotes, and no markdown. "
        "Make the prompts meaningfully different from each other while staying faithful to the hint. "
        f"Hint: {prompt_hint}"
    )


def _compose_file_content(prompt_prefix: str, generated_prompt: str) -> str:
    prefix = prompt_prefix.strip()
    body = generated_prompt.strip()
    if not prefix:
        return body
    if not body:
        return prefix
    if prefix.endswith((" ", "\n", ",")):
        return f"{prefix}{body}"
    return f"{prefix} {body}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate prompt text files with xAI Grok")
    parser.add_argument(
        "--config",
        default="config-prompt-create.txt",
        help="Config file path (default: config-prompt-create.txt)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional Grok text model. Default: use Model from config or grok-3-mini.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = _resolve_path(base_dir, args.config, "config-prompt-create.txt")
    config = _read_kv_config(config_path)
    if not config:
        raise RuntimeError(f"Config is empty or missing: {config_path}")

    prompt_hint = (config.get("prompt-hint") or "").strip()
    prompt_prefix = config.get("prompt-prefix") or ""
    prompt_file_head = (config.get("prompt-file-head") or "").strip()
    prompt_dir = _resolve_path(base_dir, config.get("prompt-dir"), "prompt")
    api_key_path = _resolve_path(base_dir, config.get("Key"), "image2image2026.txt")
    model = (args.model or config.get("Model") or "grok-3-mini").strip()

    if not prompt_hint:
        raise RuntimeError("Missing required config: prompt-hint")
    if not prompt_file_head:
        raise RuntimeError("Missing required config: prompt-file-head")

    try:
        prompt_number = int((config.get("prompt-number") or "").strip())
    except ValueError as exc:
        raise RuntimeError("Invalid prompt-number. Expected integer.") from exc

    if prompt_number <= 0:
        raise RuntimeError("prompt-number must be greater than 0")

    api_key = _read_api_key(api_key_path)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write clean image-generation prompts. "
                    "Return valid JSON only, matching the requested schema."
                ),
            },
            {
                "role": "user",
                "content": _build_user_prompt(prompt_hint, prompt_number),
            },
        ],
        "temperature": 0.9,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "prompt_batch",
                "schema": {
                    "type": "object",
                    "properties": {
                        "prompts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": prompt_number,
                            "maxItems": prompt_number,
                        }
                    },
                    "required": ["prompts"],
                    "additionalProperties": False,
                },
            },
        },
    }

    result = _http_post_json(f"{XAI_API_BASE}/chat/completions", api_key, payload)
    generated_prompts = _extract_structured_prompts(result, prompt_number)

    for index, generated_prompt in enumerate(generated_prompts, start=1):
        file_content = _compose_file_content(prompt_prefix, generated_prompt)

        out_path = prompt_dir / f"{prompt_file_head}{index}.txt"
        out_path.write_text(file_content + "\n", encoding="utf-8")
        print(out_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)