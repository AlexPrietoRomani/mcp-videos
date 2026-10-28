"""User-simulation test for the full deterministic render pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mcp_videos.config import AppSettings
from mcp_videos.core.models import (
    DialogueLine,
    Project,
    ProjectStatus,
    RenderStatus,
    Scene,
    VoiceProfile,
    WordTimestamp,
)
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectRepository,
    RenderJobRepository,
    SceneRepository,
    VoiceRepository,
)
from mcp_videos.render.engine import DeterministicRenderEngine

PROJECT_ID = "short-render-sim"
VOICE_ID = "preset-narrador-render"
WIDTH = 360
HEIGHT = 640
FPS = 60
TOTAL_DURATION_MS = 1000
WORDS = ("LA", "ANTENA", "MÁS", "RARA", "DE", "LA", "NASA")


def scene_html(title: str, accent: str) -> str:
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<style>
  html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; }}
  body {{ background: #07101f; font-family: sans-serif; }}
  #panel {{ position: absolute; left: 0; top: 38%; width: 100%; text-align: center;
            color: #ffffff; font-size: 42px; font-weight: 700; }}
  #bar {{ position: absolute; left: 0; bottom: 0; height: 10px; background: {accent}; }}
</style>
</head>
<body>
  <div id="panel">{title}</div>
  <div id="bar"></div>
  <script>
    window.__setFrameTime = function (timeMs) {{
      var panel = document.getElementById('panel');
      var bar = document.getElementById('bar');
      panel.style.transform = 'translateY(' + (timeMs / 40.0) + 'px)';
      panel.style.opacity = String(Math.max(0.25, 1.0 - timeMs / 4000.0));
      bar.style.width = Math.min(100, timeMs / 5.0) + '%';
    }};
    window.__setFrameTime(0);
  </script>
</body>
</html>
"""


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


def probe_mp4(path: Path) -> tuple[dict[str, dict[str, object]], float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,nb_frames,r_frame_rate,pix_fmt",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = {stream["codec_type"]: stream for stream in payload["streams"]}
    return streams, float(payload["format"]["duration"])


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
def project(db: DatabaseManager) -> Project:
    repository = ProjectRepository(db)
    VoiceRepository(db).upsert(
        VoiceProfile(
            id=VOICE_ID,
            name="Narrador",
            engine="kokoro",
            mode="preset",
            language="es",
        )
    )
    return repository.create(
        Project(
            id=PROJECT_ID,
            title="La antena más rara de la NASA",
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            total_duration_ms=TOTAL_DURATION_MS,
            master_audio_path="audio/dialogue_master.wav",
            status=ProjectStatus.SCENES_READY,
            created_at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
            updated_at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
        )
    )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg y ffprobe son obligatorios para la simulación",
)
def test_user_renders_final_mp4_with_audio_and_dynamic_subtitles(
    db: DatabaseManager,
    workspace: WorkspaceManager,
    project: Project,
) -> None:
    dialogues = DialogueRepository(db)
    line = dialogues.insert_line(
        DialogueLine(
            project_id=project.id,
            line_index=0,
            voice_id=VOICE_ID,
            speaker_label="Narrador",
            text_content="La antena más rara de la NASA",
            pause_after_ms=0,
            start_ms=0,
            end_ms=TOTAL_DURATION_MS,
            audio_path="audio/line_00.wav",
        )
    )
    assert line.id is not None
    words = [
        WordTimestamp(
            line_id=line.id,
            word_index=index,
            word=token,
            start_ms=50 + index * 130,
            end_ms=150 + index * 130,
            confidence=0.98,
        )
        for index, token in enumerate(WORDS)
    ]
    dialogues.replace_word_timestamps(line.id, words)

    write_silence_wav(workspace.dialogue_master_path(project.id))
    workspace.write_scene_html(project.id, 0, scene_html("ANTENA EVOLUTIVA", "#ffd400"))
    workspace.write_scene_html(project.id, 1, scene_html("SEÑAL ESPACIAL", "#00d4ff"))
    scenes = [
        Scene(
            project_id=project.id,
            scene_index=0,
            title="Hook",
            start_ms=0,
            end_ms=500,
            html_path="scenes/scene_00.html",
            validation_status="valid",
        ),
        Scene(
            project_id=project.id,
            scene_index=1,
            title="Cierre",
            start_ms=500,
            end_ms=TOTAL_DURATION_MS,
            html_path="scenes/scene_01.html",
            validation_status="valid",
        ),
    ]
    for scene in scenes:
        SceneRepository(db).upsert(scene)

    engine = DeterministicRenderEngine(
        db,
        workspace,
        AppSettings(_env_file=None, ffmpeg_crf=28),
    )

    result = engine.render_project(
        project,
        SceneRepository(db).list_by_project(project.id),
        burn_subtitles=True,
    )

    output_path = workspace.final_mp4_path(project.id)
    assert result.output_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 1000
    assert result.total_frames == 60
    assert result.rendered_frames == 60
    assert result.job.status is RenderStatus.COMPLETED
    assert result.job.rendered_frames == 60
    assert result.job.output_mp4_path == "output/final.mp4"
    assert result.subtitles_path is not None
    assert result.subtitles_path.is_file()

    subtitle_content = result.subtitles_path.read_text(encoding="utf-8")
    for token in WORDS:
        assert token in subtitle_content
    assert "{\\c&H00D4FF&}" in subtitle_content

    streams, duration_s = probe_mp4(output_path)
    video = streams["video"]
    audio = streams["audio"]
    assert video["codec_name"] == "h264"
    assert video["width"] == WIDTH
    assert video["height"] == HEIGHT
    assert video["pix_fmt"] == "yuv420p"
    assert video["nb_frames"] == str(60)
    assert video["r_frame_rate"] == "60/1"
    assert audio["codec_name"] == "aac"
    assert duration_s == pytest.approx(TOTAL_DURATION_MS / 1000, abs=0.1)

    persisted_job = RenderJobRepository(db).get_or_raise(result.job.id)
    assert persisted_job.status is RenderStatus.COMPLETED
    assert persisted_job.rendered_frames == persisted_job.total_frames == 60
    assert ProjectRepository(db).get_or_raise(project.id).status is ProjectStatus.RENDERED
