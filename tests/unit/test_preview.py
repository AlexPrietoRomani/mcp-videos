"""Unit tests for the FastAPI Live Studio Preview endpoints."""

from __future__ import annotations

import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mcp_videos.config import AppSettings
from mcp_videos.core.models import (
    DialogueLine,
    LanguageCode,
    Project,
    RenderJob,
    RenderStatus,
    Scene,
    VoiceEngine,
    VoiceMode,
    VoiceProfile,
    WordTimestamp,
)
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectRepository,
    SceneRepository,
    VoiceRepository,
)
from mcp_videos.preview.server import create_preview_app
from mcp_videos.render.engine import RenderEngineError, SceneNotValidatedError
from mcp_videos.scenes.validator import HARNESS_MARKER

PROJECT_ID = "short-preview-unit"
VOICE_ID = "preset-preview-es"
TOTAL_DURATION_MS = 4000

SCENE_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <style>body { margin: 0; background: #07111f; color: #eaf6ff; }</style>
</head>
<body>
  <div id="stage">Escena de preview</div>
  <script>
    window.onFrameTime = function (timeMs) {
      document.getElementById("stage").dataset.time = String(timeMs);
    };
  </script>
</body>
</html>
"""


@dataclass
class PreviewEnvironment:
    db: DatabaseManager
    workspace: WorkspaceManager
    project_repo: ProjectRepository
    scene_repo: SceneRepository
    dialogue_repo: DialogueRepository
    project: Project
    line: DialogueLine
    words: list[WordTimestamp]
    scenes: list[Scene]


@dataclass
class FakeRenderEngine:
    calls: list[tuple[str, bool]] = field(default_factory=list)

    def render_project(
        self,
        project: Project,
        *,
        burn_subtitles: bool = True,
    ) -> SimpleNamespace:
        self.calls.append((project.id, burn_subtitles))
        job = RenderJob(
            id="11111111-1111-4111-8111-111111111111",
            project_id=project.id,
            status=RenderStatus.COMPLETED,
            total_frames=240,
            rendered_frames=240,
            burn_subtitles=burn_subtitles,
            output_mp4_path="output/final.mp4",
        )
        return SimpleNamespace(
            job=job,
            output_path=Path("output/final.mp4"),
            total_frames=240,
            rendered_frames=240,
        )


class FakeOrchestrator:
    def __init__(self, response: dict[str, object] | None = None) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.response = response or {
            "job_id": "22222222-2222-4222-8222-222222222222",
            "status": "running",
            "output_mp4_path": "output/final.mp4",
        }

    def render_project_video(
        self,
        project_id: str,
        burn_subtitles: bool = True,
    ) -> dict[str, object]:
        self.calls.append((project_id, burn_subtitles))
        return dict(self.response)


class FailingRenderEngine:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def render_project(
        self,
        project: Project,
        *,
        burn_subtitles: bool = True,
    ) -> object:
        raise self.error


def write_silence_wav(path: Path, duration_ms: int = TOTAL_DURATION_MS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 24000
    frames = round(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


@pytest.fixture
def environment(tmp_path: Path) -> PreviewEnvironment:
    db = DatabaseManager(tmp_path / "data" / "mcp_videos.sqlite")
    db.initialize()
    workspace = WorkspaceManager(
        workspaces_root=tmp_path / "workspaces",
        voices_root=tmp_path / "data" / "voices",
    )
    workspace.ensure_project_dirs(PROJECT_ID)

    project_repo = ProjectRepository(db)
    scene_repo = SceneRepository(db)
    dialogue_repo = DialogueRepository(db)
    voice_repo = VoiceRepository(db)

    voice_repo.upsert(
        VoiceProfile(
            id=VOICE_ID,
            name="Narrador de preview",
            engine=VoiceEngine.KOKORO,
            mode=VoiceMode.PRESET,
            language=LanguageCode.ES,
        )
    )
    project = project_repo.create(
        Project(
            id=PROJECT_ID,
            title="La antena más rara de la NASA",
            description="Short de prueba para el estudio de preview.",
            width=1080,
            height=1920,
            fps=60,
            total_duration_ms=TOTAL_DURATION_MS,
            master_audio_path="audio/dialogue_master.wav",
            status="scenes_ready",
        )
    )

    write_silence_wav(workspace.dialogue_master_path(PROJECT_ID))
    write_silence_wav(workspace.line_audio_path(PROJECT_ID, 0))
    line = dialogue_repo.insert_line(
        DialogueLine(
            project_id=PROJECT_ID,
            line_index=0,
            voice_id=VOICE_ID,
            speaker_label="NARRADOR",
            text_content="En 2026 la NASA lanzó una antena evolutiva.",
            pause_after_ms=100,
            start_ms=0,
            end_ms=TOTAL_DURATION_MS,
            audio_path="audio/line_00.wav",
        )
    )
    assert line.id is not None
    words = dialogue_repo.insert_word_timestamps(
        line.id,
        [
            WordTimestamp(word_index=0, word="En", start_ms=0, end_ms=300, confidence=0.99),
            WordTimestamp(
                word_index=1,
                word="NASA",
                start_ms=1000,
                end_ms=1400,
                confidence=0.97,
            ),
            WordTimestamp(
                word_index=2,
                word="antena",
                start_ms=2200,
                end_ms=2700,
                confidence=0.96,
            ),
            WordTimestamp(
                word_index=3,
                word="evolutiva",
                start_ms=2800,
                end_ms=3600,
                confidence=0.95,
            ),
        ],
    )

    workspace.write_scene_html(PROJECT_ID, 0, SCENE_HTML)
    workspace.write_scene_html(PROJECT_ID, 1, SCENE_HTML.replace("preview", "segunda"))
    scenes = [
        scene_repo.upsert(
            Scene(
                project_id=PROJECT_ID,
                scene_index=0,
                title="Hook antena evolutiva",
                start_ms=0,
                end_ms=2000,
                html_path="scenes/scene_00.html",
                validation_status="valid",
            )
        ),
        scene_repo.upsert(
            Scene(
                project_id=PROJECT_ID,
                scene_index=1,
                title="Telemetría NASA",
                start_ms=2000,
                end_ms=TOTAL_DURATION_MS,
                html_path="scenes/scene_01.html",
                validation_status="valid",
            )
        ),
    ]

    return PreviewEnvironment(
        db=db,
        workspace=workspace,
        project_repo=project_repo,
        scene_repo=scene_repo,
        dialogue_repo=dialogue_repo,
        project=project,
        line=line,
        words=words,
        scenes=scenes,
    )


def build_client(
    environment: PreviewEnvironment,
    *,
    render_engine: object | None = None,
    orchestrator: object | None = None,
) -> TestClient:
    app = create_preview_app(
        environment.db,
        environment.workspace,
        orchestrator,
        settings=AppSettings(_env_file=None),
        render_engine=render_engine,
    )
    return TestClient(app)


@pytest.fixture
def client(environment: PreviewEnvironment) -> Iterator[TestClient]:
    with build_client(environment) as test_client:
        yield test_client


def test_index_serves_interactive_studio(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Live Studio Preview" in response.text
    assert 'id="stage-iframe"' in response.text
    assert 'id="timeline"' in response.text
    assert "word-cue" in response.text


def test_list_projects_returns_serialized_catalog(client: TestClient) -> None:
    response = client.get("/api/projects")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 1
    project = payload[0]
    assert project["id"] == PROJECT_ID
    assert project["title"] == "La antena más rara de la NASA"
    assert project["scene_count"] == 2
    assert project["word_count"] == 4
    assert project["has_audio"] is True
    assert project["master_audio_url"] == f"/preview/{PROJECT_ID}/audio"
    assert project["status"] == "scenes_ready"


def test_project_manifest_includes_lines_words_and_scenes(client: TestClient) -> None:
    response = client.get(f"/api/projects/{PROJECT_ID}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["project"]["id"] == PROJECT_ID
    assert payload["total_duration_ms"] == TOTAL_DURATION_MS
    assert len(payload["lines"]) == 1
    assert payload["lines"][0]["speaker_label"] == "NARRADOR"

    words = payload["words"]
    assert [word["word"] for word in words] == ["En", "NASA", "antena", "evolutiva"]
    assert [word["start_ms"] for word in words] == [0, 1000, 2200, 2800]
    assert words[1]["speaker_label"] == "NARRADOR"
    assert words[1]["line_index"] == 0

    scenes = payload["scenes"]
    assert [scene["scene_index"] for scene in scenes] == [0, 1]
    assert scenes[0]["html_url"] == f"/preview/{PROJECT_ID}/scene/0"
    assert scenes[0]["duration_ms"] == 2000
    assert payload["master_audio_url"] == f"/preview/{PROJECT_ID}/audio"
    assert payload["has_audio"] is True


def test_manifest_unknown_project_returns_404(client: TestClient) -> None:
    response = client.get("/api/projects/no-existe")

    assert response.status_code == 404
    assert "no-existe" in response.json()["detail"]


def test_scene_endpoint_serves_html_with_injected_harness(client: TestClient) -> None:
    response = client.get(f"/preview/{PROJECT_ID}/scene/0")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert HARNESS_MARKER in response.text
    assert "window.__setFrameTime" in response.text
    assert "Escena de preview" in response.text
    assert response.text.count(HARNESS_MARKER) == 1


def test_scene_endpoint_returns_404_for_unknown_scene(client: TestClient) -> None:
    response = client.get(f"/preview/{PROJECT_ID}/scene/99")

    assert response.status_code == 404
    assert "99" in response.json()["detail"]


def test_scene_endpoint_returns_404_when_html_is_missing(
    environment: PreviewEnvironment,
    client: TestClient,
) -> None:
    environment.workspace.scene_html_path(PROJECT_ID, 1).unlink()

    response = client.get(f"/preview/{PROJECT_ID}/scene/1")

    assert response.status_code == 404
    assert "scene_01.html" in response.json()["detail"]


def test_audio_endpoint_serves_master_wav(
    environment: PreviewEnvironment,
    client: TestClient,
) -> None:
    expected = environment.workspace.dialogue_master_path(PROJECT_ID).read_bytes()

    response = client.get(f"/preview/{PROJECT_ID}/audio")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == expected


def test_audio_endpoint_returns_404_without_master(
    environment: PreviewEnvironment,
    client: TestClient,
) -> None:
    environment.workspace.dialogue_master_path(PROJECT_ID).unlink()

    response = client.get(f"/preview/{PROJECT_ID}/audio")

    assert response.status_code == 404
    assert PROJECT_ID in response.json()["detail"]


def test_render_endpoint_invokes_injected_engine(
    environment: PreviewEnvironment,
) -> None:
    engine = FakeRenderEngine()
    with build_client(environment, render_engine=engine) as client:
        response = client.post(
            f"/api/projects/{PROJECT_ID}/render",
            json={"scene_index": 0, "burn_subtitles": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "11111111-1111-4111-8111-111111111111"
    assert payload["status"] == "completed"
    assert payload["output_mp4_path"] == "output/final.mp4"
    assert payload["total_frames"] == 240
    assert engine.calls == [(PROJECT_ID, False)]


def test_render_endpoint_uses_orchestrator_when_injected(
    environment: PreviewEnvironment,
) -> None:
    orchestrator = FakeOrchestrator()
    with build_client(environment, orchestrator=orchestrator) as client:
        response = client.post(f"/api/projects/{PROJECT_ID}/render", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "22222222-2222-4222-8222-222222222222"
    assert payload["status"] == "running"
    assert payload["project_id"] == PROJECT_ID
    assert orchestrator.calls == [(PROJECT_ID, True)]


def test_render_endpoint_accepts_empty_body(
    environment: PreviewEnvironment,
) -> None:
    engine = FakeRenderEngine()
    with build_client(environment, render_engine=engine) as client:
        response = client.post(f"/api/projects/{PROJECT_ID}/render")

    assert response.status_code == 200
    assert engine.calls == [(PROJECT_ID, True)]


def test_render_endpoint_returns_404_for_unknown_project(
    environment: PreviewEnvironment,
) -> None:
    with build_client(environment, render_engine=FakeRenderEngine()) as client:
        response = client.post("/api/projects/no-existe/render", json={})

    assert response.status_code == 404


def test_render_endpoint_returns_404_for_unknown_scene(
    environment: PreviewEnvironment,
) -> None:
    with build_client(environment, render_engine=FakeRenderEngine()) as client:
        response = client.post(
            f"/api/projects/{PROJECT_ID}/render",
            json={"scene_index": 42},
        )

    assert response.status_code == 404
    assert "42" in response.json()["detail"]


def test_render_endpoint_maps_invalid_scenes_to_409(
    environment: PreviewEnvironment,
) -> None:
    error = SceneNotValidatedError(
        "las escenas deben estar validadas antes de renderizar",
        scene_indexes=[0],
    )
    with build_client(environment, render_engine=FailingRenderEngine(error)) as client:
        response = client.post(f"/api/projects/{PROJECT_ID}/render", json={})

    assert response.status_code == 409
    assert "validadas" in response.json()["detail"]


def test_render_endpoint_maps_engine_errors_to_409(
    environment: PreviewEnvironment,
) -> None:
    error = RenderEngineError("FFmpeg no está disponible")
    with build_client(environment, render_engine=FailingRenderEngine(error)) as client:
        response = client.post(f"/api/projects/{PROJECT_ID}/render", json={})

    assert response.status_code == 409
    assert "FFmpeg" in response.json()["detail"]


def test_app_accepts_settings_as_third_positional_argument(
    environment: PreviewEnvironment,
) -> None:
    app = create_preview_app(
        environment.db,
        environment.workspace,
        AppSettings(_env_file=None),
        render_engine=FakeRenderEngine(),
    )
    with TestClient(app) as client:
        response = client.get("/api/projects")

    assert response.status_code == 200
    assert response.json()[0]["id"] == PROJECT_ID
