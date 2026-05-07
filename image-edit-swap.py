import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable


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


def _write_kv_config(path: Path, config: dict[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in config.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_role_data_path(base_dir: Path, config_path: Path) -> Path:
    candidates = [
        config_path.with_suffix(".json"),
        base_dir / "config-imge-edit-swap.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_role_presets(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Missing role preset file: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("roles")
    else:
        items = payload

    if not isinstance(items, list):
        raise RuntimeError(f"Invalid role preset format: {path}")

    presets: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        preset: dict[str, str] = {}
        for key, value in item.items():
            if isinstance(value, str):
                preset[str(key)] = value.strip()
        if preset.get("Name"):
            presets.append(preset)

    return presets


def _rewrite_output_for_role(configured_output: str, role_name: str) -> str:
    raw = configured_output.strip()
    if not raw:
        return raw

    parts = raw.replace("\\", "/").split("/")
    offset = 1 if parts and parts[0] == "." else 0
    if len(parts) > offset + 2 and parts[offset] == "images":
        parts[offset + 1] = role_name
        return "/".join(parts)
    return raw


def _build_model_config(base_dir: Path, config_path: Path) -> dict[str, str]:
    config = _read_kv_config(config_path)
    if not config:
        raise RuntimeError(f"Config is empty or missing: {config_path}")

    if not (_parse_bool(config.get("Model")) or _parse_bool(config.get("isModel"))):
        return config

    role_name = (config.get("RoleName") or "").strip()
    if not role_name:
        raise RuntimeError("RoleName is required when Model=true")

    role_data_path = _resolve_role_data_path(base_dir, config_path)
    presets = _load_role_presets(role_data_path)
    selected = next(
        (item for item in presets if item.get("Name", "").lower() == role_name.lower()),
        None,
    )
    if selected is None:
        raise RuntimeError(f"RoleName not found in {role_data_path.name}: {role_name}")

    merged = dict(config)
    for key in ("SwitchColor", "SwitchCountry", "SwitchProfessional", "Source"):
        value = (selected.get(key) or "").strip()
        if value:
            merged[key] = value

    preset_name = (selected.get("Name") or role_name).strip()
    if preset_name and (merged.get("Output") or "").strip():
        merged["Output"] = _rewrite_output_for_role(merged["Output"], preset_name)

    merged["RoleName"] = preset_name
    return merged


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
    run_config = _build_model_config(base_dir, config_path)

    image_edit_batch = (base_dir / "image-edit-batch.py").resolve()
    face_swap_batch = (base_dir / "face-swap-batch.py").resolve()

    if not image_edit_batch.exists():
        raise RuntimeError(f"Missing script: {image_edit_batch}")
    if not face_swap_batch.exists():
        raise RuntimeError(f"Missing script: {face_swap_batch}")

    temp_config_path: Path | None = None
    effective_config_path = config_path
    if run_config != _read_kv_config(config_path):
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".model.config",
            delete=False,
            dir=str(base_dir),
        ) as tf:
            temp_config_path = Path(tf.name)
        _write_kv_config(temp_config_path, run_config)
        effective_config_path = temp_config_path

    try:
        print("Step 1/2: image-edit-batch.py")
        step1_cmd = [sys.executable, str(image_edit_batch), "--config", str(effective_config_path)]
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
            str(effective_config_path),
            "--target",
            str(target_seed),
        ]
        rc2, _out2 = _run_and_stream(step2_cmd, cwd=base_dir)
        return rc2
    finally:
        if temp_config_path and temp_config_path.exists():
            try:
                temp_config_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
