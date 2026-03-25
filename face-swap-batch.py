import argparse
from pathlib import Path
import re
import subprocess
import sys


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
        return path
    return (base_dir / path).resolve()


def _split_target_pattern(target_path: Path) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.*?)(\d+)$", target_path.stem)
    if not match:
        raise RuntimeError(
            f"Target filename must end with a number, for example c1.jpg: {target_path.name}"
        )
    prefix = match.group(1)
    start_index = int(match.group(2))
    return prefix, start_index, target_path.suffix


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run face-swap.py for sequential image files")
    parser.add_argument("--config", default="config-swap.config", help="Config file path")
    parser.add_argument("--start", type=int, default=None, help="Starting index, default comes from config target")
    parser.add_argument("--end", type=int, default=None, help="Ending index inclusive, default is start + 9")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if one image fails")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = _resolve_path(base_dir, args.config)
    config = _read_kv_config(config_path)

    source_path = _resolve_path(base_dir, config.get("Source"))
    target_seed_path = _resolve_path(base_dir, config.get("Target"))
    prefix, default_start, suffix = _split_target_pattern(target_seed_path)
    start = args.start if args.start is not None else default_start
    end = args.end if args.end is not None else start + 9
    if end < start:
        raise RuntimeError(f"Invalid range: start={start}, end={end}")

    script_path = (base_dir / "face-swap.py").resolve()
    if not script_path.exists():
        raise RuntimeError(f"Missing script: {script_path}")

    failures: list[tuple[int, str]] = []
    for index in range(start, end + 1):
        target_path = target_seed_path.with_name(f"{prefix}{index}{suffix}")
        output_path = target_path.with_name(f"{target_path.stem}-swap{target_path.suffix}")
        if not target_path.exists():
            message = f"Missing target image: {target_path}"
            print(f"[{index}] ERROR {message}")
            failures.append((index, message))
            if args.stop_on_error:
                break
            continue

        command = [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--source",
            str(source_path),
            "--target",
            str(target_path),
            "--output",
            str(output_path),
        ]
        print(f"[{index}] Running: {target_path.name} -> {output_path.name}")
        completed = subprocess.run(command, cwd=str(base_dir), capture_output=True, text=True)
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or f"Exit code {completed.returncode}").strip()
            print(f"[{index}] ERROR {message}")
            failures.append((index, message))
            if args.stop_on_error:
                break
            continue

        print(f"[{index}] OK {output_path}")

    success_count = (end - start + 1) - len(failures)
    print(f"Completed: {success_count} succeeded, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)