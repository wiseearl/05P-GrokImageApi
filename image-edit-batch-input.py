import argparse
from pathlib import Path
import subprocess
import sys
import time


def _resolve_path(base_dir: Path, configured_path: str) -> Path:
    p = Path(configured_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-run image-edit.py for s1.jpg ~ s20.jpg; sleep 5s after each success"
    )
    parser.add_argument(
        "--config",
        default="config-image-edit.config",
        help="Config file path passed to image-edit.py (default: config-image-edit.config)",
    )
    parser.add_argument(
        "--dir",
        default="images/Kamiki",
        help="Directory containing sN.jpg files (default: images/Kamiki)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Start index (default: 1)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=20,
        help="End index inclusive (default: 20)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        help="Sleep seconds after each successful output (default: 5)",
    )

    args = parser.parse_args()

    if args.end < args.start:
        raise RuntimeError(f"Invalid range: start={args.start}, end={args.end}")

    base_dir = Path(__file__).resolve().parent
    script_path = (base_dir / "image-edit.py").resolve()
    if not script_path.exists():
        raise RuntimeError(f"Missing script: {script_path}")

    config_path = _resolve_path(base_dir, args.config)
    if not config_path.exists():
        raise RuntimeError(f"Missing config: {config_path}")

    images_dir = _resolve_path(base_dir, args.dir)
    if not images_dir.exists():
        raise RuntimeError(f"Missing images directory: {images_dir}")

    failures: list[tuple[int, str]] = []

    for i in range(args.start, args.end + 1):
        input_path = images_dir / f"s{i}.jpg"
        if not input_path.exists():
            message = f"Missing input image: {input_path}"
            print(f"[{i}] ERROR {message}")
            failures.append((i, message))
            continue

        command = [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--input",
            str(input_path),
        ]

        print(f"[{i}] Running: {input_path.name}")
        completed = subprocess.run(command, cwd=str(base_dir), capture_output=True, text=True)
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or f"Exit code {completed.returncode}").strip()
            print(f"[{i}] ERROR {message}")
            failures.append((i, message))
            continue

        out = (completed.stdout or "").strip()
        if out:
            print(out)

        if args.sleep > 0:
            time.sleep(args.sleep)

    total = args.end - args.start + 1
    success_count = total - len(failures)
    print(f"Completed: {success_count} succeeded, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
