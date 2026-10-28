"""Unit tests for dynamic ASS subtitles and render engine preconditions."""

from __future__ import annotations

import shutil
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

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

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no está disponible")
    def test_ffmpeg_accepts_generated_ass(self, tmp_path: Path) -> None:
        words = [
            make_word(0, "SUBTITULO", 0, 400),
            make_word(1, "DINAMICO", 400, 800),
        ]
        target = generate_ass_subtitles(words, 360, 640, tmp_path / "subtitles.ass")

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
                f"ass={target}",
                "-f",
                "null",
                "-",
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
