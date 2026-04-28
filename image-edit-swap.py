import argparse
from pathlib import Path
import subprocess
import sys
from typing import Iterable


def _iter_existing_files_from_output_lines(base_dir: Path, lines: Iterable[str]) -> list[Path]:
    outputs: list[Path] = []
    for raw in lines:
        text = raw.strip().strip('"').strip("'")
        if not text:
            continue
        try:
            p = Path(text)
        except Exception:
            continue

        if not p.is_absolute():
            p = (base_dir / p).resolve()

        if p.exists() and p.is_file():
            outputs.append(p)

    return outputs


def _run_and_stream(command: list[str], cwd: Path) -> tuple[int, list[str]]:
    collected: list[str] = []

    with subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            collected.append(line)
            print(line, end="")

        returncode = proc.wait()

    return returncode, collected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run image-edit-batch.py then face-swap-batch.py using one config file"
    )
    parser.add_argument(
        "--config",
        default="config-image-edit-swap.config",
        help="Config file path (default: config-image-edit-swap.config)",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = (base_dir / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)

    image_edit_batch = (base_dir / "image-edit-batch.py").resolve()
    face_swap_batch = (base_dir / "face-swap-batch.py").resolve()

    if not image_edit_batch.exists():
        raise RuntimeError(f"Missing script: {image_edit_batch}")
    if not face_swap_batch.exists():
        raise RuntimeError(f"Missing script: {face_swap_batch}")

    print("Step 1/2: image-edit-batch.py")
    step1_cmd = [sys.executable, str(image_edit_batch), "--config", str(config_path)]
    rc1, out_lines = _run_and_stream(step1_cmd, cwd=base_dir)
    if rc1 != 0:
        print(f"Step 1 failed (exit={rc1}). Stop.")
        return rc1

    outputs = _iter_existing_files_from_output_lines(base_dir, out_lines)
    if not outputs:
        raise RuntimeError(
            "Step 1 succeeded but could not detect any output image paths from stdout. "
            "Ensure image-edit.py prints the output path and the file exists."
        )

    target_seed = outputs[0]
    print("Step 2/2: face-swap-batch.py")
    step2_cmd = [
        sys.executable,
        str(face_swap_batch),
        "--config",
        str(config_path),
        "--target",
        str(target_seed),
    ]
    rc2, _out2 = _run_and_stream(step2_cmd, cwd=base_dir)
    return rc2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
