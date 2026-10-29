"""High-level application service coordinating every mcp-videos pipeline flow."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from mcp_videos.config import AppSettings
from mcp_videos.core.models import (
    LanguageCode,
    Project,
    ProjectStatus,
    Scene,
    ValidationStatus,
    VoiceEngine,
    VoiceMode,
    validate_scenes_timeline_coverage,
)
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectRepository,
    SceneRepository,
    VoiceRepository,
)

if TYPE_CHECKING:
    from mcp_videos.audio.aligner import WordAligner
    from mcp_videos.audio.dialogue_synth import DialogueSynthesizer
    from mcp_videos.audio.voice_registry import VoiceRegistryService
    from mcp_videos.render.engine import DeterministicRenderEngine
    from mcp_videos.scenes.generator import SceneGenerator
    from mcp_videos.scenes.validator import SceneValidator

DEMO_PROJECT_ID = "short-antena-st5"
DEMO_TITLE = "La antena más rara que fue al espacio no la diseñó nadie"
DEMO_DIALOGUE: tuple[dict[str, object], ...] = (
    {
        "voice_id": "preset-narrador-es",
        "speaker_label": "Narrador",
        "text_content": (
            "En 2006, la NASA necesitaba una antena imposible: ligera, "
            "flexible y capaz de desplegarse sola en órbita."
        ),
        "pause_after_ms": 300,
    },
    {
        "voice_id": "preset-ingeniero-es",
        "speaker_label": "Ingeniero",
        "text_content": (
            "Nadie la diseñó. Un algoritmo evolutivo probó miles de formas "
            "hasta encontrar la más rara de todas."
        ),
        "pause_after_ms": 200,
    },
)
DEMO_SCENES: tuple[dict[str, str], ...] = (
    {
        "title": "LA ANTENA IMPOSIBLE",
        "subtitle": "Ligera, flexible y desplegable en órbita.",
        "eyebrow": "MISIÓN NASA 2006",
        "accent": "#67e8f9",
    },
    {
        "title": "DISEÑADA POR EVOLUCIÓN",
        "subtitle": "Nadie la dibujó: el algoritmo encontró la forma.",
        "eyebrow": "ALGORITMO GENÉTICO",
        "accent": "#f5c842",
    },
)


class OrchestratorError(RuntimeError):
    """Raised when the application service cannot complete a pipeline step."""


class ProjectNotReadyError(OrchestratorError):
    """Raised when a step requires an audio timeline that does not exist yet."""


class SceneGenerationUnavailableError(OrchestratorError):
    """Raised when neither the LLM provider nor the local template can build a scene."""


def _dump(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


class ProjectOrchestrator:
    """Coordinate voices, dialogue, scenes and rendering behind one clean API."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        db: DatabaseManager | None = None,
        workspace: WorkspaceManager | None = None,
        voice_registry: VoiceRegistryService | None = None,
        dialogue_synthesizer: DialogueSynthesizer | None = None,
        aligner: WordAligner | None = None,
        scene_validator: SceneValidator | None = None,
        scene_generator: SceneGenerator | None = None,
        render_engine: DeterministicRenderEngine | None = None,
    ) -> None:
        from mcp_videos.audio.aligner import WordAligner as DefaultWordAligner
        from mcp_videos.audio.dialogue_synth import DialogueSynthesizer as DefaultSynthesizer
        from mcp_videos.audio.voice_registry import VoiceRegistryService as DefaultVoiceRegistry
        from mcp_videos.render.engine import DeterministicRenderEngine as DefaultRenderEngine
        from mcp_videos.scenes.generator import SceneGenerator as DefaultSceneGenerator
        from mcp_videos.scenes.validator import SceneValidator as DefaultSceneValidator

        self.settings = settings or AppSettings()
        self.settings.ensure_base_dirs()
        self.db = db or DatabaseManager(self.settings.mcp_videos_db_path)
        self.db.apply_migrations()
        self.workspace = workspace or WorkspaceManager(
            self.settings.mcp_videos_workspaces_dir,
            self.settings.mcp_videos_voices_dir,
        )
        self.projects = ProjectRepository(self.db)
        self.dialogues = DialogueRepository(self.db)
        self.scenes = SceneRepository(self.db)
        self.voice_catalog = VoiceRepository(self.db)

        self.voice_registry = voice_registry or DefaultVoiceRegistry(
            self.voice_catalog,
            self.workspace,
        )
        self.dialogue_synthesizer = dialogue_synthesizer or DefaultSynthesizer(
            self.db,
            self.workspace,
            self.settings,
        )
        self.aligner = aligner or DefaultWordAligner(
            self.db,
            self.workspace,
            self.settings,
        )
        self.scene_validator = scene_validator or DefaultSceneValidator(
            self.scenes,
            self.workspace,
        )
        self.scene_generator = scene_generator or DefaultSceneGenerator(
            self.settings,
            validator=self.scene_validator,
        )
        self.render_engine = render_engine or DefaultRenderEngine(
            self.db,
            self.workspace,
            self.settings,
        )

    def register_voice(
        self,
        voice_id: str,
        name: str | None = None,
        *,
        engine: str | VoiceEngine = VoiceEngine.KOKORO,
        mode: str | VoiceMode = VoiceMode.PRESET,
        sample_wav_path: str | None = None,
        language: str | LanguageCode = LanguageCode.ES,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register or update one preset or cloned voice in the catalog."""

        from mcp_videos.audio.ffmpeg_utils import SAMPLE_RATE_HZ

        display_name = (name or voice_id.replace("-", " ").title()).strip()
        resolved_mode = VoiceMode(mode)
        if resolved_mode is VoiceMode.CLONE:
            if not sample_wav_path:
                raise OrchestratorError(
                    "una voz clonada requiere sample_wav_path con un WAV de 3-30 s"
                )
            voice = self.voice_registry.register_clone(
                voice_id,
                display_name,
                sample_wav_path,
                engine=engine,
                language=language,
                config=config,
            )
        else:
            voice = self.voice_registry.register_preset(
                voice_id,
                display_name,
                engine=engine,
                language=language,
                config=config,
            )
        payload = _dump(voice)
        payload.update(
            {
                "voice_id": voice.id,
                "status": "ready",
                "sample_rate": SAMPLE_RATE_HZ,
            }
        )
        return payload

    def list_voices(self, language: str | LanguageCode | None = None) -> list[dict[str, Any]]:
        """List registered voices, optionally filtered by language."""

        voices = self.voice_registry.list_voices()
        if language is not None:
            resolved = LanguageCode(language)
            voices = [voice for voice in voices if voice.language is resolved]
        payloads: list[dict[str, Any]] = []
        for voice in voices:
            payload = _dump(voice)
            payload["voice_id"] = voice.id
            payloads.append(payload)
        return payloads

    def create_project(
        self,
        project_id: str,
        title: str,
        *,
        width: int = 1080,
        height: int = 1920,
        fps: int = 60,
        description: str = "",
    ) -> dict[str, Any]:
        """Create (or refresh) a project and its workspace directories."""

        project = Project(
            id=project_id,
            title=title,
            description=description,
            width=width,
            height=height,
            fps=fps,
        )
        existing = self.projects.get(project.id)
        if existing is None:
            project = self.projects.create(project)
        else:
            project = self.projects.save(
                existing.model_copy(
                    update={
                        "title": project.title,
                        "description": project.description,
                        "width": project.width,
                        "height": project.height,
                        "fps": project.fps,
                        "updated_at": datetime.now(UTC),
                    }
                )
            )
        workspace_dir = self.workspace.ensure_project_dirs(project.id)
        return {
            "project_id": project.id,
            "title": project.title,
            "workspace_dir": str(workspace_dir),
            "status": project.status.value,
            "width": project.width,
            "height": project.height,
            "fps": project.fps,
        }

    def get_project(self, project_id: str) -> Project:
        """Return a project or raise :class:`ProjectNotFoundError`."""

        return self.projects.get_or_raise(project_id)

    def get_project_manifest(self, project_id: str) -> dict[str, Any]:
        """Return the full JSON-friendly manifest used by MCP, CLI and preview."""

        project = self.projects.get_or_raise(project_id)
        lines = self.dialogues.list_by_project(project_id)
        line_payloads: list[dict[str, Any]] = []
        flat_words: list[dict[str, Any]] = []
        for line in lines:
            words = self.dialogues.list_word_timestamps(line.id or 0)
            word_payloads = [_dump(word) for word in words]
            line_payload = _dump(line)
            line_payload["words"] = word_payloads
            line_payloads.append(line_payload)
            flat_words.extend(
                {
                    **_dump(word),
                    "line_index": line.line_index,
                    "speaker_label": line.speaker_label,
                }
                for word in words
            )
        scene_payloads = [_dump(scene) for scene in self.scenes.list_by_project(project_id)]
        return {
            "project": _dump(project),
            "lines": line_payloads,
            "words": flat_words,
            "cues": flat_words,
            "scenes": scene_payloads,
            "counts": {
                "lines": len(line_payloads),
                "words": len(flat_words),
                "scenes": len(scene_payloads),
            },
            "workspace_dir": str(self.workspace.project_dir(project_id)),
            "final_video_path": str(self.workspace.final_mp4_path(project_id)),
        }

    def synthesize_and_align(
        self,
        project_id: str,
        turns: Sequence[object],
    ) -> dict[str, Any]:
        """Synthesize the multi-voice dialogue and align every word timestamp."""

        self.projects.get_or_raise(project_id)
        if not turns:
            raise OrchestratorError("la síntesis requiere al menos un turno de diálogo")
        synthesis = self.dialogue_synthesizer.synthesize_project(project_id, turns)
        alignment = self.aligner.align_project(project_id)
        self.projects.update_status(project_id, ProjectStatus.AUDIO_READY)
        lines = self.dialogues.list_by_project(project_id)
        words = list(alignment.words)
        return {
            "project_id": project_id,
            "status": ProjectStatus.AUDIO_READY.value,
            "total_duration_ms": synthesis.total_duration_ms,
            "master_audio_path": synthesis.master_audio_path,
            "lines_count": len(synthesis.lines),
            "words_count": len(alignment),
            "lines": [_dump(line) for line in lines],
            "words": [_dump(word) for word in words],
        }

    def save_and_validate_scene(
        self,
        project_id: str,
        scene_index: int,
        title: str,
        start_ms: int,
        end_ms: int,
        *,
        html_content: str | None = None,
        prompt: str = "",
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Persist or generate one scene and run the headless validation gate."""

        project = self.projects.get_or_raise(project_id)
        if project.total_duration_ms <= 0:
            raise ProjectNotReadyError(
                "el proyecto no tiene audio: ejecute synthesize_and_align antes de crear escenas"
            )
        if isinstance(scene_index, bool) or scene_index < 0:
            raise OrchestratorError("scene_index debe ser un entero >= 0")
        if isinstance(start_ms, bool) or start_ms < 0:
            raise OrchestratorError("start_ms debe ser un entero >= 0")
        if isinstance(end_ms, bool) or end_ms <= start_ms:
            raise OrchestratorError("end_ms debe ser mayor que start_ms")
        if end_ms > project.total_duration_ms:
            raise OrchestratorError(
                f"la escena termina en {end_ms} ms pero el proyecto dura "
                f"{project.total_duration_ms} ms"
            )

        if html_content is None:
            html_content = self._generate_scene_html(
                project,
                scene_index,
                title,
                start_ms,
                end_ms,
                prompt=prompt,
                provider=provider,
            )
        if not isinstance(html_content, str) or not html_content.strip():
            raise OrchestratorError("html_content no puede estar vacío")

        scene = self.scene_validator.validate_and_save(
            project_id,
            scene_index,
            html_content,
            start_ms,
            end_ms,
            title=title,
            prompt_used=prompt,
            width=project.width,
            height=project.height,
            workspace=self.workspace,
        )
        scenes_ready = self._refresh_scene_status(project_id)
        return self._scene_payload(scene, scenes_ready=scenes_ready)

    def generate_scene_with_ai(
        self,
        project_id: str,
        scene_index: int,
        title: str,
        start_ms: int,
        end_ms: int,
        *,
        prompt: str = "",
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Generate one scene with the configured LLM (local template fallback)."""

        project = self.projects.get_or_raise(project_id)
        if project.total_duration_ms <= 0:
            raise ProjectNotReadyError(
                "el proyecto no tiene audio: ejecute synthesize_and_align antes de crear escenas"
            )
        html_content = self._generate_scene_html(
            project,
            scene_index,
            title,
            start_ms,
            end_ms,
            prompt=prompt,
            provider=provider,
        )
        return self.save_and_validate_scene(
            project_id,
            scene_index,
            title,
            start_ms,
            end_ms,
            html_content=html_content,
            prompt=prompt,
        )

    def render_project(
        self,
        project_id: str,
        *,
        burn_subtitles: bool = True,
        fps: int | None = None,
    ) -> dict[str, Any]:
        """Render every validated scene into the project ``final.mp4``."""

        project = self.projects.get_or_raise(project_id)
        if fps is not None:
            project = Project.model_validate(
                {
                    **project.model_dump(),
                    "fps": fps,
                    "updated_at": datetime.now(UTC),
                }
            )
            project = self.projects.save(project)
        scenes = self.scenes.list_by_project(project_id)
        result = self.render_engine.render_project(
            project,
            scenes,
            burn_subtitles=burn_subtitles,
        )
        updated = self.projects.get_or_raise(project_id)
        relative_output = self.workspace.relative_project_path(
            project_id,
            result.output_path,
        )
        return {
            "job_id": result.job.id,
            "project_id": project_id,
            "status": result.job.status.value,
            "project_status": updated.status.value,
            "output_mp4_path": str(result.output_path),
            "output_mp4_relative_path": relative_output,
            "fps": project.fps,
            "total_frames": result.total_frames,
            "rendered_frames": result.rendered_frames,
            "duration_ms": project.total_duration_ms,
            "subtitles_path": (
                str(result.subtitles_path) if result.subtitles_path is not None else None
            ),
        }

    def build_demo_short(
        self,
        project_id: str = DEMO_PROJECT_ID,
        *,
        title: str = DEMO_TITLE,
        width: int = 1080,
        height: int = 1920,
        fps: int = 60,
        scene_count: int = 2,
        burn_subtitles: bool = True,
        render: bool = True,
        dialogue: Sequence[object] | None = None,
    ) -> dict[str, Any]:
        """Build the antenna demo short end-to-end through every application flow."""

        from mcp_videos.scenes.generator import build_technical_explainer_scene_html

        self.voice_registry.seed_defaults()
        created = self.create_project(
            project_id,
            title,
            width=width,
            height=height,
            fps=fps,
            description="Short demostrativo construido por ProjectOrchestrator.",
        )
        turns = [dict(turn) for turn in (dialogue or DEMO_DIALOGUE)]
        synthesis = self.synthesize_and_align(project_id, turns)
        total_duration_ms = int(synthesis["total_duration_ms"])
        boundaries = self._scene_boundaries(total_duration_ms, scene_count)

        saved_scenes: list[dict[str, Any]] = []
        for index, (start_ms, end_ms) in enumerate(boundaries):
            spec = DEMO_SCENES[index % len(DEMO_SCENES)]
            words = self.dialogues.word_timestamps_in_window(
                start_ms,
                end_ms,
                project_id=project_id,
            )
            html = build_technical_explainer_scene_html(
                spec["title"],
                words=words,
                start_ms=start_ms,
                end_ms=end_ms,
                subtitle=spec["subtitle"],
                eyebrow=spec["eyebrow"],
                width=width,
                height=height,
                accent=spec["accent"],
                words_are_absolute=True,
            )
            saved = self.save_and_validate_scene(
                project_id,
                index,
                spec["title"],
                start_ms,
                end_ms,
                html_content=html,
                prompt="demo-template",
            )
            if saved["validation_status"] != ValidationStatus.VALID.value:
                errors = "; ".join(saved["validation_errors"]) or "sin diagnóstico"
                raise OrchestratorError(
                    f"la escena {index} no superó la validación headless: {errors}"
                )
            saved_scenes.append(saved)

        render_summary = (
            self.render_project(project_id, burn_subtitles=burn_subtitles) if render else None
        )
        manifest = self.get_project_manifest(project_id)
        project = self.get_project(project_id)
        final_scenes = self.scenes.list_by_project(project_id)
        scenes_ready = self._scenes_are_ready(project, final_scenes)
        saved_scenes = [
            self._scene_payload(scene, scenes_ready=scenes_ready) for scene in final_scenes
        ]
        return {
            "project_id": project_id,
            "title": title,
            "project": manifest["project"],
            "workspace_dir": created["workspace_dir"],
            "voices": self.list_voices(),
            "synthesis": synthesis,
            "scenes": saved_scenes,
            "render": render_summary,
            "final_video_path": (
                render_summary["output_mp4_path"]
                if render_summary is not None
                else str(self.workspace.final_mp4_path(project_id))
            ),
        }

    def _generate_scene_html(
        self,
        project: Project,
        scene_index: int,
        title: str,
        start_ms: int,
        end_ms: int,
        *,
        prompt: str = "",
        provider: str | None = None,
    ) -> str:
        from mcp_videos.scenes.generator import (
            LLMProviderError,
            SceneGenerationError,
            build_technical_explainer_scene_html,
        )

        words = self.dialogues.word_timestamps_in_window(
            start_ms,
            end_ms,
            project_id=project.id,
        )
        resolved_provider = provider or self._default_scene_provider()
        try:
            return self.scene_generator.generate_scene_html(
                title,
                words,
                start_ms,
                end_ms,
                width=project.width,
                height=project.height,
                provider=resolved_provider,
                instructions=prompt,
                words_are_absolute=True,
            )
        except (LLMProviderError, SceneGenerationError) as exc:
            if resolved_provider == "template":
                raise SceneGenerationUnavailableError(
                    f"no se pudo generar la escena {scene_index}: {exc}"
                ) from exc
            spec = DEMO_SCENES[scene_index % len(DEMO_SCENES)]
            return build_technical_explainer_scene_html(
                title,
                words=words,
                start_ms=start_ms,
                end_ms=end_ms,
                subtitle=spec["subtitle"],
                eyebrow=spec["eyebrow"],
                width=project.width,
                height=project.height,
                accent=spec["accent"],
                words_are_absolute=True,
            )

    @staticmethod
    def _default_scene_provider() -> str:
        return (
            os.environ.get("MCP_VIDEOS_LLM_PROVIDER")
            or os.environ.get("LLM_PROVIDER")
            or "template"
        )

    @staticmethod
    def _scene_boundaries(total_duration_ms: int, scene_count: int) -> list[tuple[int, int]]:
        if isinstance(scene_count, bool) or not isinstance(scene_count, int) or scene_count <= 0:
            raise OrchestratorError("scene_count debe ser un entero > 0")
        if total_duration_ms < scene_count:
            raise OrchestratorError(
                "la duración del audio es demasiado corta para el número de escenas"
            )
        cuts = [round(total_duration_ms * index / scene_count) for index in range(scene_count + 1)]
        cuts[0] = 0
        cuts[-1] = total_duration_ms
        for index in range(1, len(cuts)):
            if cuts[index] <= cuts[index - 1]:
                cuts[index] = cuts[index - 1] + 1
        return list(pairwise(cuts))

    def _refresh_scene_status(self, project_id: str) -> bool:
        project = self.projects.get_or_raise(project_id)
        scenes = self.scenes.list_by_project(project_id)
        scenes_ready = self._scenes_are_ready(project, scenes)
        target = ProjectStatus.SCENES_READY if scenes_ready else ProjectStatus.AUDIO_READY
        if project.status is not target:
            self.projects.update_status(project_id, target)
        return scenes_ready

    @staticmethod
    def _scenes_are_ready(project: Project, scenes: Sequence[Scene]) -> bool:
        if not scenes or project.total_duration_ms <= 0:
            return False
        try:
            validate_scenes_timeline_coverage(scenes, project.total_duration_ms)
        except ValueError:
            return False
        return all(scene.validation_status is ValidationStatus.VALID for scene in scenes)

    def _scene_payload(self, scene: Scene, *, scenes_ready: bool) -> dict[str, Any]:
        project = self.projects.get_or_raise(scene.project_id)
        payload = _dump(scene)
        payload.update(
            {
                "scene_id": scene.id,
                "project_id": scene.project_id,
                "console_errors": list(scene.validation_errors),
                "scenes_ready": scenes_ready,
                "project_status": project.status.value,
            }
        )
        return payload


__all__ = [
    "DEMO_DIALOGUE",
    "DEMO_PROJECT_ID",
    "DEMO_SCENES",
    "DEMO_TITLE",
    "OrchestratorError",
    "ProjectNotReadyError",
    "ProjectOrchestrator",
    "SceneGenerationUnavailableError",
]
