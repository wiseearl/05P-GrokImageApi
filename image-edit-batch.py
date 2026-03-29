import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


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
        raise RuntimeError("Missing required path")
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _split_numbered_prompt_path(path: Path) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.*?)(\d+)$", path.stem)
    if not match:
        raise RuntimeError(
            "Prompt filename must end with a number, for example c1.txt or s1.txt"
        )

    prefix = match.group(1)
    start_index = int(match.group(2))
    return prefix, start_index, path.suffix


def _write_kv_config(path: Path, config: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in sorted(config.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-run image-edit.py using the numbered Prompt file from config-image-edit.config"
    )
    parser.add_argument(
        "--config",
        default="config-image-edit.config",
        help="Base config file path (default: config-image-edit.config)",
    )
    parser.add_argument(
        "--file-numbers",
        type=int,
        default=0,
        help=(
            "How many prompt files to run from the configured Prompt seed. Default: read from config (FileNumbers=...)."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        help="Sleep seconds after each successful output (default: 5)",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    script_path = (base_dir / "image-edit.py").resolve()
    if not script_path.exists():
        raise RuntimeError(f"Missing script: {script_path}")

    base_config_path = _resolve_path(base_dir, args.config)
    base_config = _read_kv_config(base_config_path)
    if not base_config:
        raise RuntimeError(f"Config is empty or missing: {base_config_path}")

    configured_file_numbers = 0
    try:
        configured_file_numbers = int((base_config.get("FileNumbers") or "").strip() or "0")
    except ValueError:
        configured_file_numbers = 0

    file_numbers = args.file_numbers if args.file_numbers > 0 else configured_file_numbers
    if file_numbers <= 0:
        raise RuntimeError(
            "Missing FileNumbers. Set FileNumbers=<N> in config-image-edit.config, or pass --file-numbers N."
        )

    prompt_seed_path = _resolve_path(base_dir, base_config.get("Prompt"))
    prompt_prefix, start_index, prompt_suffix = _split_numbered_prompt_path(prompt_seed_path)

    failures: list[tuple[int, str]] = []

    for offset in range(file_numbers):
        prompt_index = start_index + offset
        prompt_file = prompt_seed_path.with_name(f"{prompt_prefix}{prompt_index}{prompt_suffix}")
        if not prompt_file.exists():
            message = f"Missing prompt template: {prompt_file}"
            print(f"[{prompt_index}] ERROR {message}")
            failures.append((prompt_index, message))
            continue

        rel_prompt = os.path.relpath(prompt_file, base_dir).replace("\\", "/")
        run_config = dict(base_config)
        run_config["Prompt"] = rel_prompt
        run_config.pop("FileNumbers", None)

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=f".{prompt_prefix}{prompt_index}.config",
                delete=False,
                dir=str(base_dir),
            ) as tf:
                tmp_path = Path(tf.name)
            _write_kv_config(tmp_path, run_config)

            print(f"[{prompt_index}] Running prompt: {prompt_file.name}")
            command = [
                sys.executable,
                str(script_path),
                "--config",
                str(tmp_path),
            ]
            completed = subprocess.run(
                command,
                cwd=str(base_dir),
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                message = (
                    completed.stderr
                    or completed.stdout
                    or f"Exit code {completed.returncode}"
                ).strip()
                if completed.returncode == 4:
                    print(f"[{prompt_index}] MODERATION {message}")
                else:
                    print(f"[{prompt_index}] ERROR {message}")
                failures.append((prompt_index, message))
                continue

            out = (completed.stdout or "").strip()
            if out:
                print(out)

            if args.sleep > 0:
                time.sleep(args.sleep)
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    success_count = file_numbers - len(failures)
    print(f"Completed: {success_count} succeeded, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
