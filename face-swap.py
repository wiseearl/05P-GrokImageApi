import argparse
import os
from pathlib import Path
import shutil
import shlex
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


def _resolve_path(base_dir: Path, configured_path: str | None, default_path: str | None = None) -> Path:
    raw = (configured_path or "").strip() or (default_path or "")
    if not raw:
        raise RuntimeError("Missing required path in config")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_optional_path(base_dir: Path, configured_path: str | None) -> Path | None:
    raw = (configured_path or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _normalize_output_path(output_path: Path, target_path: Path) -> Path:
    if output_path.suffix.lower() == target_path.suffix.lower():
        return output_path
    return output_path.with_suffix(target_path.suffix)


def _parse_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    return shlex.split(value, posix=os.name != "nt")


def _get_config_value(config: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = config.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _find_facefusion_script(base_dir: Path, config: dict[str, str], cli_value: str | None) -> Path:
    if cli_value:
        script_path = _resolve_path(base_dir, cli_value)
        if not script_path.exists():
            raise RuntimeError(f"FaceFusion script not found: {script_path}")
        return script_path

    env_script = os.environ.get("FACEFUSION_SCRIPT", "").strip()
    if env_script:
        script_path = Path(env_script).expanduser().resolve()
        if not script_path.exists():
            raise RuntimeError(f"FaceFusion script not found: {script_path}")
        return script_path

    configured_script = _get_config_value(config, "FaceFusionScript", "facefusion_script")
    if configured_script:
        script_path = _resolve_path(base_dir, configured_script)
        if not script_path.exists():
            raise RuntimeError(f"FaceFusion script not found: {script_path}")
        return script_path

    env_dir = os.environ.get("FACEFUSION_DIR", "").strip()
    configured_dir = _get_config_value(config, "FaceFusionDir", "facefusion_dir")
    candidate_dirs = []
    if env_dir:
        candidate_dirs.append(Path(env_dir).expanduser().resolve())
    if configured_dir:
        candidate_dirs.append(_resolve_path(base_dir, configured_dir))
    candidate_dirs.extend(
        [
            (base_dir / "facefusion").resolve(),
            (base_dir.parent / "facefusion").resolve(),
        ]
    )

    for candidate_dir in candidate_dirs:
        script_path = candidate_dir / "facefusion.py"
        if script_path.exists():
            return script_path

    raise RuntimeError(
        "Cannot locate FaceFusion. Set FACEFUSION_DIR or FACEFUSION_SCRIPT, or add FaceFusionDir/FaceFusionScript to face-swap.config."
    )


def _find_facefusion_python(config: dict[str, str], cli_value: str | None) -> str:
    for value in [
        cli_value,
        os.environ.get("FACEFUSION_PYTHON", "").strip(),
        _get_config_value(config, "FaceFusionPython", "facefusion_python"),
    ]:
        if value:
            python_path = Path(value).expanduser()
            if python_path.exists():
                return str(python_path.resolve())
            return value
    return sys.executable


def _find_ffmpeg_bin_dir(base_dir: Path, config: dict[str, str]) -> Path | None:
    configured_dir = _get_config_value(config, "FFmpegBinDir", "ffmpeg_bin_dir")
    if configured_dir:
        path = _resolve_path(base_dir, configured_dir)
        if (path / "ffmpeg.exe").exists():
            return path
        raise RuntimeError(f"FFmpeg binary directory not found: {path}")

    detected = shutil.which("ffmpeg")
    if detected:
        return Path(detected).resolve().parent

    candidate_dirs = [
        Path.home() / "AppData/Local/Microsoft/WinGet/Links",
        Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin",
        Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-7.0.2-full_build/bin",
        Path("C:/Program Files/ffmpeg/bin"),
    ]
    for candidate_dir in candidate_dirs:
        if (candidate_dir / "ffmpeg.exe").exists():
            return candidate_dir.resolve()

    return None


def _build_subprocess_env(base_dir: Path, config: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    ffmpeg_bin_dir = _find_ffmpeg_bin_dir(base_dir, config)
    if ffmpeg_bin_dir:
        path_sep = os.pathsep
        current_path = env.get("PATH", "")
        env["PATH"] = str(ffmpeg_bin_dir) if not current_path else str(ffmpeg_bin_dir) + path_sep + current_path
    return env


def _append_option(command: list[str], flag: str, value: str | None) -> None:
    if value is None:
        return
    value = value.strip()
    if not value:
        return
    command.extend([flag, value])


def _append_multi_option(command: list[str], flag: str, tokens: list[str]) -> None:
    if tokens:
        command.append(flag)
        command.extend(tokens)


def _build_facefusion_command(
    base_dir: Path,
    script_path: Path,
    python_executable: str,
    config: dict[str, str],
    source_path: Path,
    target_path: Path,
    output_path: Path,
    jobs_path: Path,
    temp_path: Path,
) -> list[str]:
    processors = _parse_tokens(_get_config_value(config, "Processors", "processors") or "face_swapper")
    if not processors:
        processors = ["face_swapper"]

    command = [
        python_executable,
        str(script_path),
        "headless-run",
        "--jobs-path",
        str(jobs_path),
        "--temp-path",
        str(temp_path),
        "--source-paths",
        str(source_path),
        "--target-path",
        str(target_path),
        "--output-path",
        str(output_path),
        "--processors",
        *processors,
    ]

    option_map = [
        ("--face-swapper-model", _get_config_value(config, "FaceSwapperModel", "face_swapper_model")),
        ("--face-swapper-pixel-boost", _get_config_value(config, "FaceSwapperPixelBoost", "face_swapper_pixel_boost")),
        ("--face-swapper-weight", _get_config_value(config, "FaceSwapperWeight", "face_swapper_weight")),
        ("--face-selector-mode", _get_config_value(config, "FaceSelectorMode", "face_selector_mode")),
        ("--face-selector-order", _get_config_value(config, "FaceSelectorOrder", "face_selector_order")),
        ("--reference-face-distance", _get_config_value(config, "ReferenceFaceDistance", "reference_face_distance")),
        ("--reference-face-position", _get_config_value(config, "ReferenceFacePosition", "reference_face_position")),
        ("--reference-frame-number", _get_config_value(config, "ReferenceFrameNumber", "reference_frame_number")),
        ("--face-mask-blur", _get_config_value(config, "FaceMaskBlur", "face_mask_blur")),
        ("--output-image-quality", _get_config_value(config, "OutputImageQuality", "output_image_quality")),
        ("--output-image-scale", _get_config_value(config, "OutputImageScale", "output_image_scale")),
    ]
    for flag, value in option_map:
        _append_option(command, flag, value)

    _append_multi_option(
        command,
        "--face-mask-types",
        _parse_tokens(_get_config_value(config, "FaceMaskTypes", "face_mask_types")),
    )
    _append_multi_option(
        command,
        "--face-mask-padding",
        _parse_tokens(_get_config_value(config, "FaceMaskPadding", "face_mask_padding")),
    )
    command.extend(_parse_tokens(_get_config_value(config, "ExtraArgs", "extra_args")))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Swap face using FaceFusion headless CLI")
    parser.add_argument("--config", default="face-swap.config", help="Config file path")
    parser.add_argument("--source", default="", help="Base image path")
    parser.add_argument("--target", default="", help="Reference face image path")
    parser.add_argument("--output", default="", help="Output image path")
    parser.add_argument("--facefusion-script", default="", help="Path to FaceFusion facefusion.py")
    parser.add_argument("--facefusion-python", default="", help="Python executable used to run FaceFusion")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config = _read_kv_config(_resolve_path(base_dir, args.config))

    source_path = _resolve_path(base_dir, args.source or config.get("Source"))
    target_path = _resolve_path(base_dir, args.target or config.get("Target"))

    default_output = target_path.with_name(target_path.stem + "-swap" + target_path.suffix)
    output_path = _resolve_path(base_dir, args.output or config.get("Output"), str(default_output))
    output_path = _normalize_output_path(output_path, target_path)
    output_path = _unique_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise RuntimeError(f"Source image not found: {source_path}")
    if not target_path.exists():
        raise RuntimeError(f"Target image not found: {target_path}")

    jobs_path = _resolve_optional_path(base_dir, _get_config_value(config, "JobsPath", "jobs_path"))
    if jobs_path is None:
        jobs_path = (base_dir / ".facefusion" / "jobs").resolve()
    temp_path = _resolve_optional_path(base_dir, _get_config_value(config, "TempPath", "temp_path"))
    if temp_path is None:
        temp_path = (base_dir / ".facefusion" / "temp").resolve()
    jobs_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)

    script_path = _find_facefusion_script(base_dir, config, args.facefusion_script)
    python_executable = _find_facefusion_python(config, args.facefusion_python)
    command = _build_facefusion_command(
        base_dir,
        script_path,
        python_executable,
        config,
        source_path,
        target_path,
        output_path,
        jobs_path,
        temp_path,
    )

    completed = subprocess.run(
        command,
        cwd=str(script_path.parent),
        capture_output=True,
        env=_build_subprocess_env(base_dir, config),
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        if not details:
            details = f"FaceFusion exited with code {completed.returncode}"
        raise RuntimeError(details)

    if not output_path.exists():
        raise RuntimeError(f"FaceFusion finished without creating output: {output_path}")

    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Output: {output_path}")
    print(f"FaceFusion: {script_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
