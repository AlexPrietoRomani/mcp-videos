"""Tests de simulacion de usuario de la Fase 0: importacion, CLI doctor y `.env`."""

import json
import os
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def uv_run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--no-sync", "python", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        check=False,
    )


def clean_env(**overrides: str) -> dict[str, str]:
    prefix = (
        "MCP_VIDEOS_",
        "LOCAL_INFERENCE_",
        "WHISPER_",
        "LOCAL_LLM_",
        "ANTHROPIC_",
        "GEMINI_",
        "OPENAI_",
        "ELEVENLABS_",
        "PREVIEW_PORT",
        "DEFAULT_VIDEO_FPS",
        "FFMPEG_CRF",
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith(prefix)}
    env.update(overrides)
    return env


def test_import_installed_package_from_clean_terminal() -> None:
    result = uv_run("-c", "import mcp_videos; print(mcp_videos.__version__)")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"


def test_cli_help_lists_doctor_command() -> None:
    result = uv_run("-m", "mcp_videos.cli", "--help")

    assert result.returncode == 0, result.stderr
    assert "doctor" in result.stdout


def test_doctor_runs_from_terminal_and_reports_ok() -> None:
    result = uv_run("-m", "mcp_videos.cli", "doctor", "--json")

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["healthy"] is True
    assert payload["ffmpeg"]
    assert payload["ffprobe"]
    assert payload["chromium"]


def test_uv_lock_is_up_to_date() -> None:
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_temporary_env_file_bootstraps_data_directories(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    db_path = tmp_path / "data" / "mcp_videos.sqlite"
    workspaces_dir = tmp_path / "workspaces"
    voices_dir = tmp_path / "data" / "voices"
    env_file.write_text(
        "\n".join(
            [
                f"MCP_VIDEOS_DB_PATH={db_path}",
                f"MCP_VIDEOS_WORKSPACES_DIR={workspaces_dir}",
                f"MCP_VIDEOS_VOICES_DIR={voices_dir}",
                "LOCAL_INFERENCE_DEVICE=cpu",
                "PREVIEW_PORT=8181",
                "DEFAULT_VIDEO_FPS=30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    code = textwrap.dedent(
        f"""
        import json
        from mcp_videos.config import AppSettings

        settings = AppSettings(_env_file={str(env_file)!r})
        settings.ensure_base_dirs()
        print(json.dumps({{
            "db": str(settings.mcp_videos_db_path),
            "workspaces": str(settings.mcp_videos_workspaces_dir),
            "voices": str(settings.mcp_videos_voices_dir),
            "port": settings.preview_port,
            "fps": settings.default_video_fps,
            "device": settings.resolved_device,
        }}))
        """
    )

    result = uv_run("-c", code, env=clean_env())

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["db"] == str(db_path)
    assert payload["workspaces"] == str(workspaces_dir)
    assert payload["voices"] == str(voices_dir)
    assert payload["port"] == 8181
    assert payload["fps"] == 30
    assert payload["device"] == "cpu"
    assert db_path.parent.is_dir()
    assert workspaces_dir.is_dir()
    assert voices_dir.is_dir()
