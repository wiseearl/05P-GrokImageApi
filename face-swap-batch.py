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


def _split_numbered_filename(path: Path) -> tuple[str, int, str, str]:
    """Split a filename into (prefix, index, stem_suffix, extension).

    Supports patterns like:
    - bed1.jpg        -> ("bed", 1, "", ".jpg")
    - bed1-swap.jpg   -> ("bed", 1, "-swap", ".jpg")

    The index is taken from the last digit group in the stem.
    """

    match = re.fullmatch(r"(.*)(\d+)([^0-9]*)", path.stem)
    if not match:
        raise RuntimeError(
            f"Filename must contain a number, for example bed1.jpg or bed1-swap.jpg: {path.name}"
        )

    prefix = match.group(1)
    index = int(match.group(2))
    stem_suffix = match.group(3)
    return prefix, index, stem_suffix, path.suffix


def _get_int(config: dict[str, str], key: str) -> int | None:
    raw = (config.get(key) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Invalid integer for {key}: {raw}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run face-swap.py for sequential image files")
    parser.add_argument("--config", default="config-swap-batch.config", help="Batch config file path")
    parser.add_argument("--swap-config", default="config-swap.config", help="face-swap.py config file path")
    parser.add_argument("--target", default="", help="Override seed Target image path (must end with a number, e.g. s2.jpg)")
    parser.add_argument("--start", type=int, default=None, help="Starting index, default comes from config target")
    parser.add_argument("--end", type=int, default=None, help="Ending index inclusive, default is start + 9")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if one image fails")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    batch_config_path = _resolve_path(base_dir, args.config)
    batch_config = _read_kv_config(batch_config_path)
    swap_config_path = _resolve_path(base_dir, args.swap_config)

    script_path = (base_dir / "face-swap.py").resolve()
    if not script_path.exists():
        raise RuntimeError(f"Missing script: {script_path}")

    # New batch config format:
    # Source=...
    # TargetStart=.../bed1.jpg
    # FileNumbers=20
    # OutputStart=.../bed1-swap.jpg
    if "TargetStart" in batch_config or "FileNumbers" in batch_config or "OutputStart" in batch_config:
        source_path = _resolve_path(base_dir, batch_config.get("Source"))
        target_seed_path = _resolve_path(base_dir, args.target or batch_config.get("TargetStart"))
        output_seed_raw = batch_config.get("OutputStart") or batch_config.get("Output")
        output_seed_path = _resolve_path(base_dir, output_seed_raw) if output_seed_raw else None

        target_prefix, default_start, target_stem_suffix, target_ext = _split_numbered_filename(
            target_seed_path
        )

        file_numbers = _get_int(batch_config, "FileNumbers")
        if file_numbers is None:
            # Backward compatible fallback (if someone used TargetEnd before).
            target_end_raw = batch_config.get("TargetEnd")
            if target_end_raw:
                target_end_path = _resolve_path(base_dir, target_end_raw)
                _p2, end_index, _stem2, _ext2 = _split_numbered_filename(target_end_path)
                file_numbers = end_index - default_start + 1
            else:
                raise RuntimeError("Missing FileNumbers in batch config")
        if file_numbers <= 0:
            raise RuntimeError(f"Invalid FileNumbers: {file_numbers}")

        start = args.start if args.start is not None else default_start
        end = start + file_numbers - 1

        if output_seed_path is None:
            output_seed_path = target_seed_path.with_name(
                f"{target_seed_path.stem}-swap{target_seed_path.suffix}"
            )
        output_prefix, output_start_index, output_stem_suffix, output_ext = _split_numbered_filename(
            output_seed_path
        )
        output_offset = output_start_index - default_start

        if not source_path.exists():
            raise RuntimeError(f"Missing source image: {source_path}")

        failures: list[tuple[int, str]] = []
        for index in range(start, end + 1):
            target_path = target_seed_path.with_name(
                f"{target_prefix}{index}{target_stem_suffix}{target_ext}"
            )
            output_index = index + output_offset
            output_path = output_seed_path.with_name(
                f"{output_prefix}{output_index}{output_stem_suffix}{output_ext}"
            )

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
                str(swap_config_path),
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
                message = (
                    completed.stderr
                    or completed.stdout
                    or f"Exit code {completed.returncode}"
                ).strip()
                print(f"[{index}] ERROR {message}")
                failures.append((index, message))
                if args.stop_on_error:
                    break
                continue

            print(f"[{index}] OK {output_path}")

        success_count = (end - start + 1) - len(failures)
        print(f"Completed: {success_count} succeeded, {len(failures)} failed")
        return 1 if failures else 0

    # Old config format fallback (Source/Target) for compatibility.
    source_path = _resolve_path(base_dir, batch_config.get("Source"))
    target_seed_path = _resolve_path(base_dir, args.target or batch_config.get("Target"))
    prefix, default_start, stem_suffix, ext = _split_numbered_filename(target_seed_path)
    start = args.start if args.start is not None else default_start
    end = args.end if args.end is not None else start + 9
    if end < start:
        raise RuntimeError(f"Invalid range: start={start}, end={end}")

    failures: list[tuple[int, str]] = []
    for index in range(start, end + 1):
        target_path = target_seed_path.with_name(f"{prefix}{index}{stem_suffix}{ext}")
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
            str(swap_config_path),
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