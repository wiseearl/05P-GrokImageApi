import argparse
import os
from pathlib import Path
import shutil
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
        return path.resolve()
    return (base_dir / path).resolve()


def _get_config_value(config: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = config.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


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
        current_path = env.get("PATH", "")
        env["PATH"] = (
            str(ffmpeg_bin_dir)
            if not current_path
            else str(ffmpeg_bin_dir) + os.pathsep + current_path
        )
    return env


def _build_output_path(source_path: Path, configured_output: str | None) -> Path:
    if configured_output and configured_output.strip():
        output_path = Path(configured_output.strip())
        if output_path.is_absolute():
            return output_path.resolve()
        return output_path
    return source_path.with_name(f"{source_path.stem}-mute.mp4")


def _make_absolute_output_path(base_dir: Path, output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path.resolve()
    return (base_dir / output_path).resolve()


def _build_ffmpeg_copy_command(source_path: Path, output_path: Path, overwrite: bool) -> list[str]:
    return [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "copy",
        str(output_path),
    ]


def _build_ffmpeg_transcode_command(source_path: Path, output_path: Path, overwrite: bool) -> list[str]:
    return [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _run_ffmpeg(command: list[str], base_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(base_dir),
        env=env,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a muted mp4 video from the Source defined in config-video-mute.config"
    )
    parser.add_argument(
        "--config",
        default="config-video-mute.config",
        help="Config file path (default: config-video-mute.config)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target file if it already exists",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = _resolve_path(base_dir, args.config)
    config = _read_kv_config(config_path)
    if not config:
        raise RuntimeError(f"Config is empty or missing: {config_path}")

    source_path = _resolve_path(base_dir, _get_config_value(config, "Source", "source"))
    if not source_path.exists():
        raise RuntimeError(f"Source video not found: {source_path}")

    output_path = _build_output_path(source_path, _get_config_value(config, "Output", "output"))
    output_path = _make_absolute_output_path(base_dir, output_path).with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = _build_subprocess_env(base_dir, config)
    if not shutil.which("ffmpeg", path=env.get("PATH")):
        raise RuntimeError(
            "Cannot locate ffmpeg. Install FFmpeg or set FFmpegBinDir in config-video-mute.config."
        )

    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    completed = _run_ffmpeg(
        _build_ffmpeg_copy_command(source_path, output_path, args.overwrite),
        base_dir,
        env,
    )
    if completed.returncode != 0:
        if output_path.exists():
            output_path.unlink()
        completed = _run_ffmpeg(
            _build_ffmpeg_transcode_command(source_path, output_path, args.overwrite),
            base_dir,
            env,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
            raise RuntimeError(message)

    print("Created muted video successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)