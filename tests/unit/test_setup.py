"""Tests unitarios de la Fase 0: manifiesto, configuracion y diagnostico del sistema."""

import json
import re
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mcp_videos
from mcp_videos import cli, config
from mcp_videos.config import AppSettings, resolve_device

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO_ROOT / ".env.example"
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"

REQUIRED_DEPENDENCIES = {
    "mcp",
    "typer",
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic-settings",
    "httpx",
    "playwright",
    "faster-whisper",
    "kokoro-onnx",
}

SETTINGS_ENV_VARS = (
    "MCP_VIDEOS_DB_PATH",
    "MCP_VIDEOS_WORKSPACES_DIR",
    "MCP_VIDEOS_VOICES_DIR",
    "LOCAL_INFERENCE_DEVICE",
    "WHISPER_MODEL_SIZE",
    "WHISPER_COMPUTE_TYPE",
    "LOCAL_LLM_BASE_URL",
    "LOCAL_LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ELEVENLABS_API_KEY",
    "PREVIEW_PORT",
    "DEFAULT_VIDEO_FPS",
    "FFMPEG_CRF",
)


def load_pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def dependency_names(dependencies: list[str]) -> set[str]:
    return {re.split(r"[<>=!~\[; ]", dependency, maxsplit=1)[0] for dependency in dependencies}


def read_env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in SETTINGS_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def test_pyproject_declares_project_metadata() -> None:
    data = load_pyproject()
    project = data["project"]

    assert project["name"] == "mcp-videos"
    assert project["version"] == mcp_videos.__version__
    assert project["requires-python"] == ">=3.11,<3.13"
    assert data["build-system"]["build-backend"] == "hatchling.build"


def test_pyproject_declares_required_dependencies_and_script() -> None:
    data = load_pyproject()
    project = data["project"]

    dependencies = dependency_names(project["dependencies"])
    dev_dependencies = dependency_names(data["dependency-groups"]["dev"])

    assert dependencies >= REQUIRED_DEPENDENCIES
    assert project["scripts"]["mcp-videos"] == "mcp_videos.cli:app"
    assert dev_dependencies >= {"pytest", "pytest-asyncio", "ruff"}


def test_pyproject_configures_ruff_and_pytest() -> None:
    data = load_pyproject()

    assert data["tool"]["ruff"]["target-version"] == "py311"
    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_uv_lock_exists_and_pins_project() -> None:
    assert UV_LOCK.is_file()
    content = UV_LOCK.read_text(encoding="utf-8")
    assert 'name = "mcp-videos"' in content


def test_env_example_matches_settings_fields() -> None:
    values = read_env_example()
    assert set(values) == {name.upper() for name in AppSettings.model_fields}


def test_settings_load_every_env_example_variable() -> None:
    values = read_env_example()
    settings = AppSettings(_env_file=None, **{key.lower(): value for key, value in values.items()})

    assert settings.mcp_videos_db_path == Path("data/mcp_videos.sqlite")
    assert settings.mcp_videos_workspaces_dir == Path("workspaces")
    assert settings.mcp_videos_voices_dir == Path("data/voices")
    assert settings.local_inference_device == "auto"
    assert settings.preview_port == 8080
    assert settings.default_video_fps == 60
    assert settings.ffmpeg_crf == 18


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    settings = AppSettings(_env_file=None)

    assert settings.mcp_videos_db_path == Path("data/mcp_videos.sqlite")
    assert settings.mcp_videos_workspaces_dir == Path("workspaces")
    assert settings.mcp_videos_voices_dir == Path("data/voices")
    assert settings.local_inference_device == "auto"
    assert settings.whisper_model_size == "small"
    assert settings.whisper_compute_type == "int8"
    assert settings.local_llm_base_url == "http://localhost:11434/v1"
    assert settings.local_llm_model == "qwen2.5-coder:14b"
    assert settings.anthropic_api_key is None
    assert settings.preview_port == 8080
    assert settings.default_video_fps == 60
    assert settings.ffmpeg_crf == 18


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("MCP_VIDEOS_DB_PATH", str(tmp_path / "db" / "custom.sqlite"))
    monkeypatch.setenv("MCP_VIDEOS_WORKSPACES_DIR", str(tmp_path / "spaces"))
    monkeypatch.setenv("MCP_VIDEOS_VOICES_DIR", str(tmp_path / "voices"))
    monkeypatch.setenv("LOCAL_INFERENCE_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "medium")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float32")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2:3b")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PREVIEW_PORT", "9090")
    monkeypatch.setenv("DEFAULT_VIDEO_FPS", "30")
    monkeypatch.setenv("FFMPEG_CRF", "23")

    settings = AppSettings(_env_file=None)

    assert settings.mcp_videos_db_path == tmp_path / "db" / "custom.sqlite"
    assert settings.mcp_videos_workspaces_dir == tmp_path / "spaces"
    assert settings.mcp_videos_voices_dir == tmp_path / "voices"
    assert settings.local_inference_device == "cpu"
    assert settings.whisper_model_size == "medium"
    assert settings.whisper_compute_type == "float32"
    assert settings.local_llm_model == "llama3.2:3b"
    assert settings.openai_api_key == "sk-test"
    assert settings.preview_port == 9090
    assert settings.default_video_fps == 30
    assert settings.ffmpeg_crf == 23


def test_settings_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("PREVIEW_PORT", "70000")

    with pytest.raises(ValueError):
        AppSettings(_env_file=None)


def test_ensure_base_dirs_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_settings_env(monkeypatch)
    settings = AppSettings(
        _env_file=None,
        mcp_videos_db_path=tmp_path / "data" / "mcp_videos.sqlite",
        mcp_videos_workspaces_dir=tmp_path / "workspaces",
        mcp_videos_voices_dir=tmp_path / "data" / "voices",
    )

    settings.ensure_base_dirs()
    settings.ensure_base_dirs()

    assert settings.mcp_videos_db_path.parent.is_dir()
    assert settings.mcp_videos_workspaces_dir.is_dir()
    assert settings.mcp_videos_voices_dir.is_dir()


def test_resolve_device_explicit_values() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("CPU") == "cpu"


def test_resolve_device_auto_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.shutil, "which", lambda _name: None)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_auto_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    assert resolve_device("auto") == "cuda"


def test_resolve_device_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_INFERENCE_DEVICE", "cpu")
    assert resolve_device() == "cpu"


def test_resolve_device_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_settings_resolved_device_property(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    settings = AppSettings(_env_file=None, local_inference_device="cpu")
    assert settings.resolved_device == "cpu"


def test_check_system_dependencies_all_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli, "_find_playwright_chromium", lambda cache_dir=None: None)

    result = cli.check_system_dependencies()

    assert result.ffmpeg_path == "/usr/bin/ffmpeg"
    assert result.ffprobe_path == "/usr/bin/ffprobe"
    assert result.chromium_path == "/usr/bin/chromium"
    assert result.is_healthy


def test_check_system_dependencies_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {
        "ffmpeg": None,
        "ffprobe": "/usr/bin/ffprobe",
        "chromium": "/usr/bin/chromium",
    }
    monkeypatch.setattr(cli.shutil, "which", lambda name: paths.get(name))
    monkeypatch.setattr(cli, "_find_playwright_chromium", lambda cache_dir=None: None)

    result = cli.check_system_dependencies()

    assert result.ffmpeg_path is None
    assert not result.is_healthy


def test_check_system_dependencies_uses_playwright_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        cli,
        "_find_playwright_chromium",
        lambda cache_dir=None: "/home/user/.cache/ms-playwright/chromium-1/chrome",
    )

    result = cli.check_system_dependencies()

    assert result.chromium_path == "/home/user/.cache/ms-playwright/chromium-1/chrome"
    assert not result.is_healthy


def test_find_playwright_chromium_locates_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "chromium-1234" / "chrome-linux" / "chrome"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(cli.os, "access", lambda _path, _mode: True)

    assert cli._find_playwright_chromium(tmp_path) == str(binary)


def test_find_playwright_chromium_returns_none_without_cache(tmp_path: Path) -> None:
    assert cli._find_playwright_chromium(tmp_path) is None


def test_doctor_command_reports_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    healthy = cli.SystemCheckResult(
        ffmpeg_path="/usr/bin/ffmpeg",
        ffprobe_path="/usr/bin/ffprobe",
        chromium_path="/usr/bin/chromium",
    )
    monkeypatch.setattr(cli, "check_system_dependencies", lambda: healthy)
    monkeypatch.setenv("LOCAL_INFERENCE_DEVICE", "cpu")

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "OK" in result.stdout
    assert "status: OK" in result.stdout


def test_doctor_command_reports_json(monkeypatch: pytest.MonkeyPatch) -> None:
    healthy = cli.SystemCheckResult(
        ffmpeg_path="/usr/bin/ffmpeg",
        ffprobe_path="/usr/bin/ffprobe",
        chromium_path="/usr/bin/chromium",
    )
    monkeypatch.setattr(cli, "check_system_dependencies", lambda: healthy)
    monkeypatch.setenv("LOCAL_INFERENCE_DEVICE", "cpu")

    result = CliRunner().invoke(cli.app, ["doctor", "--json"])

    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload["status"] == "ok"
    assert payload["healthy"] is True
    assert payload["ffmpeg"] == "/usr/bin/ffmpeg"


def test_doctor_command_fails_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    unhealthy = cli.SystemCheckResult(
        ffmpeg_path=None,
        ffprobe_path="/usr/bin/ffprobe",
        chromium_path="/usr/bin/chromium",
    )
    monkeypatch.setattr(cli, "check_system_dependencies", lambda: unhealthy)
    monkeypatch.setenv("LOCAL_INFERENCE_DEVICE", "cpu")

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "MISSING" in result.stdout
    assert "status: ERROR" in result.stdout
