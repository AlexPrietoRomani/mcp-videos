"""Unit tests for dynamic ASS subtitles and render engine preconditions."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mcp_videos.cli import _detect_hyperframes_cli, _detect_playwright_chromium, app
from mcp_videos.config import AppSettings
from mcp_videos.core.models import (
    Project,
    RenderStatus,
    Scene,
    TimelineCoverageError,
    WordTimestamp,
)
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import ProjectRepository, RenderJobRepository, SceneRepository
from mcp_videos.render.engine import (
    DeterministicRenderEngine,
    RenderEngineError,
    SceneNotValidatedError,
    _escape_filter_path,
    _ffmpeg_supports_ass_filter,
)
from mcp_videos.render.subtitles import (
    ASS_HIGHLIGHT_COLOR,
    build_ass_document,
    format_ass_time,
    generate_ass_subtitles,
    group_words_into_blocks,
)

PROJECT_ID = "short-render-unit"
TOTAL_DURATION_MS = 1000


def make_project(**overrides: object) -> Project:
    data: dict[str, object] = {
        "id": PROJECT_ID,
        "title": "Short de prueba de render",
        "width": 360,
        "height": 640,
        "fps": 60,
        "total_duration_ms": TOTAL_DURATION_MS,
        "master_audio_path": "audio/dialogue_master.wav",
        "status": "scenes_ready",
        "created_at": datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return Project.model_validate(data)


def make_scene(scene_index: int = 0, **overrides: object) -> Scene:
    data: dict[str, object] = {
        "project_id": PROJECT_ID,
        "scene_index": scene_index,
        "title": f"Escena {scene_index}",
        "start_ms": scene_index * 500,
        "end_ms": (scene_index + 1) * 500,
        "html_path": f"scenes/scene_{scene_index:02d}.html",
        "validation_status": "valid",
        "validation_errors": [],
        "updated_at": datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return Scene.model_validate(data)


def make_word(word_index: int, word: str, start_ms: int, end_ms: int) -> WordTimestamp:
    return WordTimestamp(
        word_index=word_index,
        word=word,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=0.99,
    )


def write_silence_wav(path: Path, duration_ms: int = TOTAL_DURATION_MS) -> Path:
    sample_rate = 24000
    frames = round(sample_rate * duration_ms / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(tmp_path / "data" / "mcp_videos.sqlite")
    manager.apply_migrations()
    return manager


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(
        workspaces_root=tmp_path / "workspaces",
        voices_root=tmp_path / "data" / "voices",
    )


@pytest.fixture
def engine(db: DatabaseManager, workspace: WorkspaceManager) -> DeterministicRenderEngine:
    return DeterministicRenderEngine(
        db,
        workspace,
        AppSettings(_env_file=None),
        fallback_chromium_path=None,
    )


class TestFormatAssTime:
    @pytest.mark.parametrize(
        ("milliseconds", "expected"),
        [
            (0, "0:00:00.00"),
            (10, "0:00:00.01"),
            (999, "0:00:01.00"),
            (1000, "0:00:01.00"),
            (65430, "0:01:05.43"),
            (3_600_000, "1:00:00.00"),
            (-250, "0:00:00.00"),
        ],
    )
    def test_formats_milliseconds(self, milliseconds: int, expected: str) -> None:
        assert format_ass_time(milliseconds) == expected

    @pytest.mark.parametrize("value", [True, 1.5, "100"])
    def test_rejects_invalid_values(self, value: object) -> None:
        with pytest.raises(ValueError):
            format_ass_time(value)  # type: ignore[arg-type]


class TestWordBlocks:
    def test_groups_consecutive_words_in_blocks_of_four(self) -> None:
        words = [
            make_word(index, f"w{index}", index * 200, index * 200 + 180) for index in range(9)
        ]

        blocks = group_words_into_blocks(words)

        assert [len(block) for block in blocks] == [4, 4, 1]
        assert [word.word for block in blocks for word in block] == [f"w{i}" for i in range(9)]

    def test_respects_custom_block_size_and_line_boundaries(self) -> None:
        words = [
            WordTimestamp(line_id=1, word_index=0, word="uno", start_ms=0, end_ms=100),
            WordTimestamp(line_id=1, word_index=1, word="dos", start_ms=100, end_ms=200),
            WordTimestamp(line_id=1, word_index=2, word="tres", start_ms=200, end_ms=300),
            WordTimestamp(line_id=2, word_index=0, word="cuatro", start_ms=400, end_ms=500),
            WordTimestamp(line_id=2, word_index=1, word="cinco", start_ms=500, end_ms=600),
        ]

        blocks = group_words_into_blocks(words, max_words_per_block=4)

        assert [[word.word for word in block] for block in blocks] == [
            ["uno", "dos", "tres"],
            ["cuatro", "cinco"],
        ]


class TestAssSubtitles:
    def test_document_contains_header_style_and_all_tokens(self) -> None:
        words = [
            make_word(0, "LA", 0, 280),
            make_word(1, "ANTENA", 300, 580),
            make_word(2, "MÁS", 600, 880),
            make_word(3, "RARA", 900, 1180),
            make_word(4, "DE", 1200, 1380),
        ]

        document = build_ass_document(words, 1080, 1920)

        assert "[Script Info]" in document
        assert "PlayResX: 1080" in document
        assert "PlayResY: 1920" in document
        assert "[V4+ Styles]" in document
        assert "Style: Shorts," in document
        assert "[Events]" in document
        assert ASS_HIGHLIGHT_COLOR in document
        for word in words:
            assert word.word in document
        assert document.count("Dialogue:") >= len(words)

    def test_generate_writes_utf8_file(self, tmp_path: Path) -> None:
        words = [make_word(0, "AÑO", 0, 500), make_word(1, "NUEVO", 500, 1000)]
        target = tmp_path / "output" / "subtitles.ass"

        result = generate_ass_subtitles(words, 1080, 1920, target)

        assert result == target
        content = target.read_text(encoding="utf-8")
        assert "AÑO" in content
        assert "NUEVO" in content

    def test_empty_word_list_still_produces_valid_header(self, tmp_path: Path) -> None:
        target = generate_ass_subtitles([], 360, 640, tmp_path / "subtitles.ass")

        content = target.read_text(encoding="utf-8")
        assert "[Events]" in content
        assert "Dialogue:" not in content

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None or not _ffmpeg_supports_ass_filter("ffmpeg"),
        reason="ffmpeg con soporte libass (filtro ass) no está disponible",
    )
    def test_ffmpeg_accepts_generated_ass(self, tmp_path: Path) -> None:
        words = [
            make_word(0, "SUBTITULO", 0, 400),
            make_word(1, "DINAMICO", 400, 800),
        ]
        target = generate_ass_subtitles(words, 360, 640, tmp_path / "subtitles.ass")
        null_out = "-"

        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=360x640:r=60:d=1",
                "-vf",
                f"ass={_escape_filter_path(target)}",
                "-f",
                "null",
                null_out,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert "Error" not in completed.stderr


class TestFfmpegCommand:
    def test_builds_h264_aac_image2pipe_command_with_subtitles(
        self,
        engine: DeterministicRenderEngine,
        tmp_path: Path,
    ) -> None:
        command = engine.build_ffmpeg_command(
            make_project(),
            tmp_path / "dialogue_master.wav",
            tmp_path / "final.mp4",
            burn_subtitles=True,
            subtitles_path=tmp_path / "subtitles.ass",
            total_frames=60,
        )

        assert command[0] == "ffmpeg"
        assert command[command.index("-f") + 1] == "image2pipe"
        assert command[command.index("-framerate") + 1] == "60"
        assert command[command.index("-i") + 1] == "-"
        assert "-c:v" in command
        assert command[command.index("-c:v") + 1] == "libx264"
        assert command[command.index("-pix_fmt") + 1] == "yuv420p"
        assert command[command.index("-c:a") + 1] == "aac"
        assert command[command.index("-b:a") + 1] == "192k"
        assert command[command.index("-frames:v") + 1] == "60"
        assert command[command.index("-movflags") + 1] == "+faststart"
        assert any(argument.startswith("ass=") for argument in command)

    def test_omits_subtitle_filter_when_burning_is_disabled(
        self,
        engine: DeterministicRenderEngine,
        tmp_path: Path,
    ) -> None:
        command = engine.build_ffmpeg_command(
            make_project(),
            tmp_path / "dialogue_master.wav",
            tmp_path / "final.mp4",
            burn_subtitles=False,
            total_frames=60,
        )

        assert not any(argument.startswith("ass=") for argument in command)
        assert "-vf" not in command


class TestRenderPreconditions:
    def test_pending_scene_aborts_before_creating_render_job(
        self,
        engine: DeterministicRenderEngine,
        db: DatabaseManager,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        scenes = [make_scene(0, validation_status="pending"), make_scene(1)]

        with pytest.raises(SceneNotValidatedError) as error:
            engine.render_project(project, scenes)

        assert error.value.scene_indexes == (0,)
        assert RenderJobRepository(db).list_by_project(project.id) == []

    def test_invalid_scene_aborts_before_creating_render_job(
        self,
        engine: DeterministicRenderEngine,
        db: DatabaseManager,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        scenes = [make_scene(0), make_scene(1, validation_status="invalid")]

        with pytest.raises(SceneNotValidatedError):
            engine.render_project(project, scenes)

        assert RenderJobRepository(db).list_by_project(project.id) == []

    def test_timeline_gap_is_rejected(
        self,
        engine: DeterministicRenderEngine,
        db: DatabaseManager,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        scenes = [
            make_scene(0, start_ms=0, end_ms=400),
            make_scene(1, start_ms=600, end_ms=1000),
        ]

        with pytest.raises(TimelineCoverageError) as error:
            engine.render_project(project, scenes)

        assert error.value.gap_start_ms == 400
        assert error.value.gap_end_ms == 600
        assert RenderJobRepository(db).list_by_project(project.id) == []

    def test_missing_scene_html_is_rejected(
        self,
        engine: DeterministicRenderEngine,
        db: DatabaseManager,
    ) -> None:
        project = ProjectRepository(db).create(make_project())

        with pytest.raises(RenderEngineError, match="no existe el HTML"):
            engine.render_project(project, [make_scene(0), make_scene(1)])

    def test_scene_from_another_project_is_rejected(
        self,
        engine: DeterministicRenderEngine,
        db: DatabaseManager,
        workspace: WorkspaceManager,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        workspace.write_scene_html(project.id, 0, "<html></html>")
        workspace.write_scene_html(project.id, 1, "<html></html>")
        scenes = [make_scene(0, project_id="otro-proyecto"), make_scene(1)]

        with pytest.raises(RenderEngineError, match="pertenece"):
            engine.render_project(project, scenes)


class TestRenderFailures:
    def test_runtime_failure_marks_job_and_project_as_failed(
        self,
        engine: DeterministicRenderEngine,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        write_silence_wav(workspace.dialogue_master_path(project.id))
        workspace.write_scene_html(project.id, 0, "<html></html>")
        workspace.write_scene_html(project.id, 1, "<html></html>")
        scenes = [make_scene(0), make_scene(1)]
        for scene in scenes:
            SceneRepository(db).upsert(scene)

        class DummyProcess:
            def __init__(self) -> None:
                self.stdin = io.BytesIO()
                self.stderr = io.BytesIO(b"")
                self.returncode = 0

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

        monkeypatch.setattr(engine, "_start_ffmpeg", lambda cmd: DummyProcess())

        def raise_capture(*args: object, **kwargs: object) -> None:
            raise RenderEngineError("fallo simulado de captura")

        monkeypatch.setattr(engine, "_capture_frames", raise_capture)

        with pytest.raises(RenderEngineError, match="fallo simulado"):
            engine.render_project(project, scenes, burn_subtitles=False)

        jobs = RenderJobRepository(db).list_by_project(project.id)
        assert len(jobs) == 1
        assert jobs[0].status is RenderStatus.FAILED
        assert jobs[0].error_message == "fallo simulado de captura"
        assert ProjectRepository(db).get_or_raise(project.id).status.value == "failed"
        assert not workspace.final_mp4_path(project.id).exists()


class TestDetectPlaywrightChromium:
    def test_detect_via_custom_env_var(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        custom_chrome = tmp_path / "custom_chrome.exe"
        custom_chrome.touch()
        monkeypatch.setenv("MCP_VIDEOS_CHROMIUM_PATH", str(custom_chrome))

        detected = _detect_playwright_chromium()
        assert detected == str(custom_chrome)

    def test_detect_windows_ms_playwright_tree(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MCP_VIDEOS_CHROMIUM_PATH", raising=False)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        monkeypatch.setattr("mcp_videos.cli.WINDOWS_CHROME_SYSTEM_PATHS", ())
        monkeypatch.setattr("mcp_videos.cli.MAC_CHROME_SYSTEM_PATHS", ())

        fake_local_app_data = tmp_path / "AppData" / "Local"
        chrome_exe = (
            fake_local_app_data
            / "ms-playwright"
            / "chromium-1155"
            / "chrome-win64"
            / "chrome.exe"
        )
        chrome_exe.parent.mkdir(parents=True, exist_ok=True)
        chrome_exe.touch()

        monkeypatch.setenv("LOCALAPPDATA", str(fake_local_app_data))
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "")

        detected = _detect_playwright_chromium()
        assert detected == str(chrome_exe)

    def test_detect_linux_ms_playwright_tree(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MCP_VIDEOS_CHROMIUM_PATH", raising=False)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        monkeypatch.setattr("mcp_videos.cli.WINDOWS_CHROME_SYSTEM_PATHS", ())
        monkeypatch.setattr("mcp_videos.cli.MAC_CHROME_SYSTEM_PATHS", ())

        fake_home = tmp_path / "home"
        chrome_linux = (
            fake_home / ".cache" / "ms-playwright" / "chromium-1140" / "chrome-linux" / "chrome"
        )
        chrome_linux.parent.mkdir(parents=True, exist_ok=True)
        chrome_linux.touch()

        monkeypatch.setattr(Path, "home", lambda: fake_home)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr(os, "access", lambda path, mode: True)

        detected = _detect_playwright_chromium()
        assert detected == str(chrome_linux)

    def test_detect_returns_none_when_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MCP_VIDEOS_CHROMIUM_PATH", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        monkeypatch.setattr("mcp_videos.cli.WINDOWS_CHROME_SYSTEM_PATHS", ())
        monkeypatch.setattr("mcp_videos.cli.MAC_CHROME_SYSTEM_PATHS", ())
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty_home")

        detected = _detect_playwright_chromium()
        assert detected is None


class TestDetectHyperframesCli:
    def test_detect_via_env_dir_binary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bin_dir = tmp_path / "hyperframes_pkg" / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        cli_cmd = bin_dir / "hyperframes.cmd"
        cli_cmd.touch()

        monkeypatch.setenv("MCP_VIDEOS_HYPERFRAMES_DIR", str(tmp_path / "hyperframes_pkg"))
        assert _detect_hyperframes_cli() == str(cli_cmd)

    def test_detect_via_shutil_which(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MCP_VIDEOS_HYPERFRAMES_DIR", raising=False)

        def mock_which(cmd: str) -> str | None:
            if cmd == "hyperframes":
                return "C:\\Program Files\\nodejs\\hyperframes.cmd"
            return None

        monkeypatch.setattr(shutil, "which", mock_which)
        assert _detect_hyperframes_cli() == "C:\\Program Files\\nodejs\\hyperframes.cmd"

    def test_detect_via_npx_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MCP_VIDEOS_HYPERFRAMES_DIR", raising=False)
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        def mock_which(cmd: str) -> str | None:
            if cmd == "npx":
                return "/usr/bin/npx"
            return None

        monkeypatch.setattr(shutil, "which", mock_which)
        assert _detect_hyperframes_cli() == "/usr/bin/npx"

    def test_detect_returns_none_when_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MCP_VIDEOS_HYPERFRAMES_DIR", raising=False)
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        assert _detect_hyperframes_cli() is None


class TestDoctorCommand:
    def test_doctor_json_contains_hyperframes_and_binaries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["doctor", "--json"])
        data = json.loads(result.stdout)
        assert "hyperframes" in data
        assert "binaries" in data
        assert "hyperframes" in data["binaries"]
        assert "chromium" in data["binaries"]
        assert "ffmpeg" in data["binaries"]

    def test_doctor_human_contains_hyperframes_entry(
        self,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["doctor"])
        assert "hyperframes:" in result.stdout


class TestFfmpegAssFilter:
    def test_ffmpeg_supports_ass_filter_when_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ffmpeg_supports_ass_filter.cache_clear()

        fake_stdout = (
            "Filters:\n"
            " ... aresample         A->A       Resample audio data.\n"
            " ... ass               V->V       Render ASS subtitles onto input video.\n"
            " ... scale             V->V       Scale the input video size.\n"
        )

        def mock_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["ffmpeg", "-hide_banner", "-filters"],
                returncode=0,
                stdout=fake_stdout,
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        assert _ffmpeg_supports_ass_filter("mock_ffmpeg_ass") is True

    def test_ffmpeg_supports_ass_filter_when_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ffmpeg_supports_ass_filter.cache_clear()

        fake_stdout = (
            "Filters:\n"
            " ... aresample         A->A       Resample audio data.\n"
            " ... scale             V->V       Scale the input video size.\n"
        )

        def mock_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["ffmpeg", "-hide_banner", "-filters"],
                returncode=0,
                stdout=fake_stdout,
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        assert _ffmpeg_supports_ass_filter("mock_ffmpeg_no_ass") is False

    def test_ffmpeg_supports_ass_filter_on_os_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ffmpeg_supports_ass_filter.cache_clear()

        def mock_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError("ffmpeg no encontrado")

        monkeypatch.setattr(subprocess, "run", mock_run)
        assert _ffmpeg_supports_ass_filter("nonexistent_ffmpeg") is False


class TestRenderWithoutLibass:
    def test_render_burn_subtitles_skips_filter_when_no_libass(
        self,
        engine: DeterministicRenderEngine,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        write_silence_wav(workspace.dialogue_master_path(project.id))
        workspace.write_scene_html(project.id, 0, "<html></html>")
        scenes = [make_scene(0, start_ms=0, end_ms=1000)]
        for scene in scenes:
            SceneRepository(db).upsert(scene)

        # Simular que FFmpeg NO tiene soporte para ass
        monkeypatch.setattr(
            "mcp_videos.render.engine._ffmpeg_supports_ass_filter",
            lambda bin_path: False,
        )

        captured_commands: list[list[str]] = []

        class DummyProcess:
            def __init__(self, cmd: list[str]) -> None:
                self.stdin = io.BytesIO()
                self.stderr = io.BytesIO(b"")
                self.returncode = 0
                captured_commands.append(cmd)

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                # Simular creación de archivo temporal de salida
                for arg in reversed(captured_commands[-1]):
                    if arg.endswith(".mp4"):
                        p = Path(arg)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(b"dummy mp4")
                        break
                return 0

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

        monkeypatch.setattr(engine, "_start_ffmpeg", lambda cmd: DummyProcess(cmd))
        monkeypatch.setattr(engine, "_capture_frames", lambda *args, **kwargs: None)

        result = engine.render_project(project, scenes, burn_subtitles=True)
        assert result.output_path.exists()
        assert len(captured_commands) == 1
        cmd = captured_commands[0]
        # Debe haber omitido el filtro -vf ass
        assert not any("ass=" in arg for arg in cmd)


class TestRenderBrokenPipeStderr:
    def test_broken_pipe_includes_process_stderr_in_exception(
        self,
        engine: DeterministicRenderEngine,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        write_silence_wav(workspace.dialogue_master_path(project.id))
        workspace.write_scene_html(project.id, 0, "<html></html>")
        scenes = [make_scene(0, start_ms=0, end_ms=1000)]
        for scene in scenes:
            SceneRepository(db).upsert(scene)

        ffmpeg_stderr_message = (
            b"[Parsed_ass_0 @ 0x55] No such filter: 'ass'\n"
            b"Error initializing filtergraph: Filter not found\n"
        )

        class BrokenPipeProcess:
            def __init__(self) -> None:
                self.stdin = io.BytesIO()
                self.stderr = io.BytesIO(ffmpeg_stderr_message)
                self.returncode = 1

            def poll(self) -> int:
                return 1

            def wait(self, timeout: float | None = None) -> int:
                return 1

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

        broken_proc = BrokenPipeProcess()
        monkeypatch.setattr(engine, "_start_ffmpeg", lambda cmd: broken_proc)

        def raise_broken_pipe(*args: object, **kwargs: object) -> None:
            raise BrokenPipeError("pipe cerrado por el subproceso")

        monkeypatch.setattr(engine, "_capture_frames", raise_broken_pipe)

        with pytest.raises(RenderEngineError) as exc_info:
            engine.render_project(project, scenes, burn_subtitles=False)

        error_text = str(exc_info.value)
        assert "No such filter: 'ass'" in error_text

        # Comprobar que en BD el job guardó el mensaje detallado de stderr
        jobs = RenderJobRepository(db).list_by_project(project.id)
        assert len(jobs) == 1
        assert jobs[0].status is RenderStatus.FAILED
        assert "No such filter: 'ass'" in jobs[0].error_message
