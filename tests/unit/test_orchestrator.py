"""Unit tests for the high-level ProjectOrchestrator application service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_videos.audio.aligner import WordAligner
from mcp_videos.config import AppSettings
from mcp_videos.core.models import (
    DialogueLine,
    Project,
    ProjectStatus,
    RenderJob,
    RenderStatus,
    Scene,
    ValidationStatus,
)
from mcp_videos.core.orchestrator import (
    OrchestratorError,
    ProjectNotReadyError,
    ProjectOrchestrator,
)
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectRepository,
    SceneRepository,
)
from mcp_videos.scenes.generator import SceneGenerator

PROJECT_ID = "short-orchestrator-unit"
TOTAL_DURATION_MS = 2400
TURNS = [
    {
        "voice_id": "preset-narrador-es",
        "speaker_label": "Narrador",
        "text_content": "En dos mil seis la NASA lanzó una antena evolutiva.",
        "pause_after_ms": 200,
    },
    {
        "voice_id": "preset-ingeniero-es",
        "speaker_label": "Ingeniero",
        "text_content": "Nadie la diseñó: un algoritmo encontró la forma.",
        "pause_after_ms": 100,
    },
]


class StubSynthesizer:
    """Persist deterministic dialogue lines without running any TTS engine."""

    def __init__(self, db: DatabaseManager) -> None:
        self.projects = ProjectRepository(db)
        self.dialogues = DialogueRepository(db)

    def synthesize_project(self, project_id: str, turns: list[dict[str, Any]]) -> object:
        project = self.projects.get_or_raise(project_id)
        self.projects.save(
            project.model_copy(
                update={
                    "total_duration_ms": TOTAL_DURATION_MS,
                    "master_audio_path": "audio/dialogue_master.wav",
                }
            )
        )
        lines: list[DialogueLine] = []
        span = TOTAL_DURATION_MS // len(turns)
        for index, turn in enumerate(turns):
            start_ms = index * span
            lines.append(
                self.dialogues.insert_line(
                    DialogueLine(
                        project_id=project_id,
                        line_index=index,
                        voice_id=str(turn["voice_id"]),
                        speaker_label=str(turn["speaker_label"]),
                        text_content=str(turn["text_content"]),
                        pause_after_ms=int(turn.get("pause_after_ms", 0)),
                        start_ms=start_ms,
                        end_ms=start_ms + span - 100,
                        audio_path=f"audio/line_{index:02d}.wav",
                    )
                )
            )
        return SimpleNamespace(
            lines=tuple(lines),
            total_duration_ms=TOTAL_DURATION_MS,
            master_audio_path="audio/dialogue_master.wav",
        )


class StubSceneValidator:
    """Persist scene rows without launching Chromium."""

    def __init__(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        *,
        status: ValidationStatus = ValidationStatus.VALID,
        errors: list[str] | None = None,
    ) -> None:
        self.scenes = SceneRepository(db)
        self.workspace = workspace
        self.status = status
        self.errors = list(errors or [])
        self.calls: list[tuple[str, int, int, int]] = []

    def validate_and_save(
        self,
        project_id: str,
        scene_index: int,
        html_content: str,
        start_ms: int,
        end_ms: int,
        *,
        title: str | None = None,
        prompt_used: str = "",
        width: int | None = None,
        height: int | None = None,
        workspace: WorkspaceManager | None = None,
    ) -> Scene:
        resolved = workspace or self.workspace
        path = resolved.write_scene_html(project_id, scene_index, html_content)
        self.calls.append((project_id, scene_index, start_ms, end_ms))
        return self.scenes.upsert(
            Scene(
                project_id=project_id,
                scene_index=scene_index,
                title=title or f"Escena {scene_index:02d}",
                start_ms=start_ms,
                end_ms=end_ms,
                html_path=resolved.relative_project_path(project_id, path),
                prompt_used=prompt_used,
                validation_status=self.status,
                validation_errors=self.errors,
            )
        )


class StubSceneGenerator:
    def __init__(self, html: str) -> None:
        self.html = html
        self.calls: list[dict[str, Any]] = []

    def generate_scene_html(
        self,
        title: str,
        words: object,
        start_ms: int,
        end_ms: int,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "title": title,
                "words": words,
                "start_ms": start_ms,
                "end_ms": end_ms,
                **kwargs,
            }
        )
        return self.html


class StubRenderEngine:
    def __init__(self, db: DatabaseManager, workspace: WorkspaceManager) -> None:
        self.projects = ProjectRepository(db)
        self.workspace = workspace
        self.calls: list[dict[str, Any]] = []

    def render_project(
        self,
        project: Project,
        scenes: list[Scene],
        *,
        burn_subtitles: bool = True,
    ) -> object:
        self.calls.append(
            {
                "project_id": project.id,
                "scene_count": len(scenes),
                "burn_subtitles": burn_subtitles,
            }
        )
        output_path = self.workspace.final_mp4_path(project.id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp4")
        subtitles_path = self.workspace.project_dir(project.id) / "output" / "subtitles.ass"
        self.projects.update_status(project.id, ProjectStatus.RENDERED)
        job = RenderJob(
            project_id=project.id,
            status=RenderStatus.COMPLETED,
            total_frames=120,
            rendered_frames=120,
            burn_subtitles=burn_subtitles,
            output_mp4_path="output/final.mp4",
        )
        return SimpleNamespace(
            job=job,
            output_path=output_path,
            subtitles_path=subtitles_path,
            total_frames=120,
            rendered_frames=120,
        )


@pytest.fixture
def orchestrator(tmp_path: Path) -> ProjectOrchestrator:
    settings = AppSettings(
        _env_file=None,
        mcp_videos_db_path=tmp_path / "data" / "mcp_videos.sqlite",
        mcp_videos_workspaces_dir=tmp_path / "workspaces",
        mcp_videos_voices_dir=tmp_path / "data" / "voices",
    )
    service = ProjectOrchestrator(settings)
    service.dialogue_synthesizer = StubSynthesizer(service.db)
    service.aligner = WordAligner(
        service.db,
        service.workspace,
        settings,
        enable_whisper=False,
    )
    service.scene_validator = StubSceneValidator(service.db, service.workspace)
    service.scene_generator = SceneGenerator(settings, provider="template")
    service.render_engine = StubRenderEngine(service.db, service.workspace)
    return service


def prepare_audio(service: ProjectOrchestrator, project_id: str = PROJECT_ID) -> dict[str, Any]:
    service.voice_registry.seed_defaults()
    service.create_project(project_id, "Antena evolutiva")
    return service.synthesize_and_align(project_id, TURNS)


class TestVoiceOrchestration:
    def test_register_and_list_preset_voices(self, orchestrator: ProjectOrchestrator) -> None:
        payload = orchestrator.register_voice(
            "voz-narrador",
            "Narrador cósmico",
            engine="kokoro",
            language="es",
        )

        assert payload["status"] == "ready"
        assert payload["voice_id"] == "voz-narrador"
        assert payload["sample_rate"] == 24000
        voices = orchestrator.list_voices("es")
        assert [voice["voice_id"] for voice in voices] == ["voz-narrador"]
        assert orchestrator.list_voices("en") == []

    def test_clone_requires_a_sample(self, orchestrator: ProjectOrchestrator) -> None:
        with pytest.raises(OrchestratorError, match="sample_wav_path"):
            orchestrator.register_voice("voz-clon", "Clon", mode="clone")

    def test_default_voice_catalog_is_seeded_for_demo(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        orchestrator.build_demo_short("short-demo-seed", render=False)

        voice_ids = {voice["voice_id"] for voice in orchestrator.list_voices()}
        assert "preset-narrador-es" in voice_ids
        assert "preset-ingeniero-es" in voice_ids


class TestProjectAndDialogueOrchestration:
    def test_create_project_is_idempotent_and_updates_metadata(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        first = orchestrator.create_project(PROJECT_ID, "Título inicial", width=360, height=640)
        second = orchestrator.create_project(
            PROJECT_ID,
            "Título actualizado",
            width=720,
            height=1280,
            fps=24,
        )

        assert first["project_id"] == second["project_id"] == PROJECT_ID
        assert second["title"] == "Título actualizado"
        assert (second["width"], second["height"], second["fps"]) == (720, 1280, 24)
        assert Path(second["workspace_dir"]).is_dir()

    def test_synthesize_and_align_persists_words_and_manifest(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        payload = prepare_audio(orchestrator)

        assert payload["status"] == ProjectStatus.AUDIO_READY.value
        assert payload["total_duration_ms"] == TOTAL_DURATION_MS
        assert payload["lines_count"] == len(TURNS)
        assert payload["words_count"] > 0
        manifest = orchestrator.get_project_manifest(PROJECT_ID)
        assert manifest["counts"]["lines"] == len(TURNS)
        assert manifest["counts"]["words"] == payload["words_count"]
        assert manifest["cues"] == manifest["words"]
        assert manifest["project"]["status"] == ProjectStatus.AUDIO_READY.value

    def test_synthesis_requires_at_least_one_turn(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        orchestrator.create_project(PROJECT_ID, "Sin diálogo")

        with pytest.raises(OrchestratorError, match="al menos un turno"):
            orchestrator.synthesize_and_align(PROJECT_ID, [])


class TestSceneOrchestration:
    def test_save_and_validate_scene_marks_project_scenes_ready(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        prepare_audio(orchestrator)

        payload = orchestrator.save_and_validate_scene(
            PROJECT_ID,
            0,
            "Hook",
            0,
            TOTAL_DURATION_MS,
            html_content="<html><body>Hook</body></html>",
        )

        assert payload["validation_status"] == ValidationStatus.VALID.value
        assert payload["scenes_ready"] is True
        assert payload["project_status"] == ProjectStatus.SCENES_READY.value
        assert orchestrator.get_project(PROJECT_ID).status is ProjectStatus.SCENES_READY

    def test_invalid_scene_keeps_project_in_audio_ready(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        prepare_audio(orchestrator)
        orchestrator.scene_validator = StubSceneValidator(
            orchestrator.db,
            orchestrator.workspace,
            status=ValidationStatus.INVALID,
            errors=["window.__setFrameTime no está definido"],
        )

        payload = orchestrator.save_and_validate_scene(
            PROJECT_ID,
            0,
            "Hook",
            0,
            TOTAL_DURATION_MS,
            html_content="<html><body>Hook</body></html>",
        )

        assert payload["scenes_ready"] is False
        assert payload["console_errors"] == ["window.__setFrameTime no está definido"]
        assert orchestrator.get_project(PROJECT_ID).status is ProjectStatus.AUDIO_READY

    def test_scene_requires_audio_first(self, orchestrator: ProjectOrchestrator) -> None:
        orchestrator.create_project(PROJECT_ID, "Sin audio")

        with pytest.raises(ProjectNotReadyError, match="synthesize_and_align"):
            orchestrator.save_and_validate_scene(
                PROJECT_ID,
                0,
                "Hook",
                0,
                1000,
                html_content="<html><body>Hook</body></html>",
            )

    def test_scene_cannot_exceed_project_duration(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        prepare_audio(orchestrator)

        with pytest.raises(OrchestratorError, match="el proyecto dura"):
            orchestrator.save_and_validate_scene(
                PROJECT_ID,
                0,
                "Hook",
                0,
                TOTAL_DURATION_MS + 1,
                html_content="<html><body>Hook</body></html>",
            )

    def test_generate_scene_with_ai_uses_generator_and_validator(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        prepare_audio(orchestrator)
        generator = StubSceneGenerator("<html><body>Generada</body></html>")
        orchestrator.scene_generator = generator

        payload = orchestrator.generate_scene_with_ai(
            PROJECT_ID,
            0,
            "Generada",
            0,
            TOTAL_DURATION_MS,
            prompt="antena desplegable",
            provider="vllm",
        )

        assert payload["validation_status"] == ValidationStatus.VALID.value
        assert generator.calls[0]["provider"] == "vllm"
        assert generator.calls[0]["instructions"] == "antena desplegable"
        assert generator.calls[0]["words_are_absolute"] is True


class TestRenderAndDemoOrchestration:
    def test_render_project_maps_render_result(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        prepare_audio(orchestrator)
        orchestrator.save_and_validate_scene(
            PROJECT_ID,
            0,
            "Hook",
            0,
            TOTAL_DURATION_MS,
            html_content="<html><body>Hook</body></html>",
        )

        payload = orchestrator.render_project(PROJECT_ID, burn_subtitles=True)

        assert payload["status"] == RenderStatus.COMPLETED.value
        assert payload["project_status"] == ProjectStatus.RENDERED.value
        assert payload["total_frames"] == payload["rendered_frames"] == 120
        assert payload["output_mp4_relative_path"] == "output/final.mp4"
        assert payload["subtitles_path"].endswith("subtitles.ass")
        engine = orchestrator.render_engine
        assert isinstance(engine, StubRenderEngine)
        assert engine.calls[0]["burn_subtitles"] is True

    def test_build_demo_short_without_render_creates_valid_scenes(
        self,
        orchestrator: ProjectOrchestrator,
    ) -> None:
        payload = orchestrator.build_demo_short(
            "short-demo-unit",
            width=360,
            height=640,
            fps=24,
            scene_count=2,
            render=False,
        )

        assert len(payload["scenes"]) == 2
        assert all(scene["validation_status"] == "valid" for scene in payload["scenes"])
        assert payload["render"] is None
        assert payload["final_video_path"].endswith("output/final.mp4")
        assert payload["project"]["status"] == ProjectStatus.SCENES_READY.value
        assert payload["synthesis"]["words_count"] > 0

    def test_scene_boundaries_cover_the_timeline(self) -> None:
        assert ProjectOrchestrator._scene_boundaries(2400, 2) == [(0, 1200), (1200, 2400)]
        assert ProjectOrchestrator._scene_boundaries(1000, 3) == [
            (0, 333),
            (333, 667),
            (667, 1000),
        ]
        with pytest.raises(OrchestratorError, match="entero > 0"):
            ProjectOrchestrator._scene_boundaries(2400, 0)
        with pytest.raises(OrchestratorError, match="demasiado corta"):
            ProjectOrchestrator._scene_boundaries(1, 2)
