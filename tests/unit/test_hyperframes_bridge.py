"""Unit tests for HyperFramesBridge: composition export, visual check and hybrid rendering."""

from __future__ import annotations

import io
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mcp_videos.config import AppSettings
from mcp_videos.core.models import Project, RenderStatus, Scene, WordTimestamp
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import ProjectRepository, RenderJobRepository, SceneRepository
from mcp_videos.render.engine import DeterministicRenderEngine, RenderEngineError
from mcp_videos.render.hyperframes_bridge import HyperFramesBridge

PROJECT_ID = "hf-bridge-unit"
TOTAL_DURATION_MS = 2000


def make_project(**overrides: object) -> Project:
    data: dict[str, object] = {
        "id": PROJECT_ID,
        "title": "Short HyperFrames Test",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "total_duration_ms": TOTAL_DURATION_MS,
        "master_audio_path": "audio/dialogue_master.wav",
        "status": "scenes_ready",
        "created_at": datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return Project.model_validate(data)


def make_scene(scene_index: int = 0, **overrides: object) -> Scene:
    data: dict[str, object] = {
        "project_id": PROJECT_ID,
        "scene_index": scene_index,
        "title": f"Escena {scene_index}",
        "start_ms": scene_index * 1000,
        "end_ms": (scene_index + 1) * 1000,
        "html_path": f"scenes/scene_{scene_index:02d}.html",
        "validation_status": "valid",
        "validation_errors": [],
        "updated_at": datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return Scene.model_validate(data)


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
def bridge(
    db: DatabaseManager,
    workspace: WorkspaceManager,
    tmp_path: Path,
) -> HyperFramesBridge:
    settings = AppSettings(_env_file=None)
    render_engine = DeterministicRenderEngine(
        db,
        workspace,
        settings,
        fallback_chromium_path=None,
    )
    return HyperFramesBridge(
        settings=settings,
        db_path=tmp_path / "data" / "mcp_videos.sqlite",
        render_engine=render_engine,
        workspace=workspace,
    )


class TestExportComposition:
    def test_exports_hyperframes_index_html_with_audio_and_clips(
        self,
        bridge: HyperFramesBridge,
        db: DatabaseManager,
        workspace: WorkspaceManager,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        ws_root = workspace.project_dir(project.id)

        # Crear audio maestro y escenas
        audio_path = ws_root / "audio" / "dialogue_master.wav"
        write_silence_wav(audio_path)

        scene_0_html = "<html><head><title>S0</title></head><body>Scene 0</body></html>"
        scene_1_html = "<html><head><title>S1</title></head><body>Scene 1</body></html>"
        workspace.write_scene_html(project.id, 0, scene_0_html)
        workspace.write_scene_html(project.id, 1, scene_1_html)

        scenes = [make_scene(0), make_scene(1)]
        cues = [
            WordTimestamp(word_index=0, word="HOLA", start_ms=0, end_ms=500),
            WordTimestamp(word_index=1, word="MUNDO", start_ms=500, end_ms=1000),
        ]

        index_path = bridge.export_composition(
            project,
            scenes,
            cues_data=cues,
            workspace_root=ws_root,
            burn_subtitles=True,
        )

        assert index_path.is_file()
        hf_dir = index_path.parent
        assert (hf_dir / "dialogue_master.wav").is_file()
        assert (hf_dir / "scene_00.html").is_file()
        assert (hf_dir / "scene_01.html").is_file()

        content = index_path.read_text(encoding="utf-8")
        assert 'data-composition-id="main"' in content
        assert 'id="dialogue-master"' in content
        assert 'id="scene-clip-0"' in content
        assert 'id="scene-clip-1"' in content
        assert 'data-track-kind="captions"' in content
        assert '__timelines["main"]' in content
        assert "HOLA" in content
        assert "MUNDO" in content


class TestCheckProject:
    def test_check_project_with_hyperframes_cli_json_mock(
        self,
        bridge: HyperFramesBridge,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        ws_root = workspace.project_dir(project.id)
        workspace.write_scene_html(project.id, 0, "<html><body>S0</body></html>")
        scenes = [make_scene(0, start_ms=0, end_ms=2000)]

        monkeypatch.setattr(bridge, "is_available", lambda: True)
        monkeypatch.setattr(bridge, "_resolve_command_prefix", lambda: ["fake_hf"])

        fake_stdout = (
            "[hyperframes] check: analyzing composition...\n"
            '{"ok": true, "duration": 2.0, "fps": 30, "warnings": []}\n'
        )

        def mock_run(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["fake_hf", "check"],
                returncode=0,
                stdout=fake_stdout,
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)

        res = bridge.check_project(project, scenes, workspace_root=ws_root, backend="auto")
        assert res["ok"] is True
        assert res["backend_used"] == "hyperframes"
        assert res["report"]["fps"] == 30

    def test_check_project_hyperframes_explicit_fails_when_unavailable(
        self,
        bridge: HyperFramesBridge,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        scenes = [make_scene(0)]
        monkeypatch.setattr(bridge, "is_available", lambda: False)

        with pytest.raises(RenderEngineError, match="no está disponible"):
            bridge.check_project(
                project,
                scenes,
                workspace_root=workspace.project_dir(project.id),
                backend="hyperframes",
            )

    def test_check_project_fallback_playwright(
        self,
        bridge: HyperFramesBridge,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        workspace.write_scene_html(project.id, 0, "<html><body>S0</body></html>")
        scenes = [make_scene(0, start_ms=0, end_ms=2000)]
        monkeypatch.setattr(bridge, "is_available", lambda: False)

        res = bridge.check_project(
            project,
            scenes,
            workspace_root=workspace.project_dir(project.id),
            backend="playwright",
        )
        assert res["ok"] is True
        assert res["backend_used"] == "playwright"


class TestRenderProject:
    def test_render_project_hyperframes_success(
        self,
        bridge: HyperFramesBridge,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        ws_root = workspace.project_dir(project.id)
        workspace.write_scene_html(project.id, 0, "<html><body>S0</body></html>")
        scenes = [make_scene(0, start_ms=0, end_ms=2000)]

        monkeypatch.setattr(bridge, "is_available", lambda: True)
        monkeypatch.setattr(bridge, "_resolve_command_prefix", lambda: ["fake_hf"])

        out_mp4 = ws_root / "output" / "final.mp4"

        def mock_run(
            cmd: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            out_idx = cmd.index("-o") + 1
            target = Path(cmd[out_idx])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"dummy mp4 output")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", mock_run)

        res = bridge.render_project(
            project,
            scenes,
            workspace_root=ws_root,
            output_mp4_path=out_mp4,
            backend="hyperframes",
        )

        assert res["status"] == "completed"
        assert res["backend_used"] == "hyperframes"
        assert Path(res["output_mp4_path"]).is_file()

        jobs = RenderJobRepository(db).list_by_project(project.id)
        assert len(jobs) == 1
        assert jobs[0].status == RenderStatus.COMPLETED
        assert ProjectRepository(db).get_or_raise(project.id).status.value == "rendered"

    def test_render_project_hyperframes_explicit_fails_when_unavailable(
        self,
        bridge: HyperFramesBridge,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        scenes = [make_scene(0)]
        monkeypatch.setattr(bridge, "is_available", lambda: False)

        with pytest.raises(RenderEngineError, match="no está disponible"):
            bridge.render_project(
                project,
                scenes,
                workspace_root=workspace.project_dir(project.id),
                backend="hyperframes",
            )

    def test_render_project_auto_falls_back_when_unavailable(
        self,
        bridge: HyperFramesBridge,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project = ProjectRepository(db).create(make_project())
        ws_root = workspace.project_dir(project.id)
        write_silence_wav(ws_root / "audio" / "dialogue_master.wav")
        workspace.write_scene_html(project.id, 0, "<html><body>S0</body></html>")
        scenes = [make_scene(0, start_ms=0, end_ms=2000)]
        for s in scenes:
            SceneRepository(db).upsert(s)

        monkeypatch.setattr(bridge, "is_available", lambda: False)

        class DummyProcess:
            def __init__(self, cmd: list[str]) -> None:
                self.stdin = io.BytesIO()
                self.stderr = io.BytesIO(b"")
                self.returncode = 0
                for arg in reversed(cmd):
                    if arg.endswith(".mp4"):
                        p = Path(arg)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(b"dummy playwright mp4")
                        break

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

        monkeypatch.setattr(
            bridge.render_engine,
            "_start_ffmpeg",
            lambda cmd: DummyProcess(cmd),
        )
        monkeypatch.setattr(bridge.render_engine, "_capture_frames", lambda *args, **kwargs: None)

        res = bridge.render_project(
            project,
            scenes,
            workspace_root=ws_root,
            backend="auto",
            burn_subtitles=False,
        )

        assert res["backend_used"] == "playwright"
        assert res["status"] == "completed"
