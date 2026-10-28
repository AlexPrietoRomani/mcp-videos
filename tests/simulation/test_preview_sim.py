"""User-simulation tests for the Live Studio Preview with a real browser."""

from __future__ import annotations

import socket
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

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
from mcp_videos.scenes.validator import find_chromium_executable

PROJECT_ID = "short-preview-sim"
VOICE_ID = "preset-preview-sim-es"
TOTAL_DURATION_MS = 4000
JOB_ID = "33333333-3333-4333-8333-333333333333"

SCENE_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <style>
    html, body { margin: 0; height: 100%; background: #07111f; color: #eaf6ff; }
    #frame-time { font-size: 96px; font-family: monospace; padding: 48px; }
  </style>
</head>
<body data-scene="__SCENE_INDEX__">
  <div id="frame-time">0</div>
  <script>
    window.onFrameTime = function (timeMs) {
      document.getElementById("frame-time").textContent = String(Math.round(timeMs));
      document.body.dataset.frameTime = String(Math.round(timeMs));
    };
  </script>
</body>
</html>
"""


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
            id=JOB_ID,
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


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_chromium(playwright: object) -> object:
    try:
        return playwright.chromium.launch(headless=True)
    except PlaywrightError as bundled_error:
        executable = find_chromium_executable()
        if executable is None:
            pytest.skip(f"Chromium no disponible: {bundled_error}")
        return playwright.chromium.launch(headless=True, executable_path=executable)


def prepare_environment(tmp_path: Path) -> tuple[DatabaseManager, WorkspaceManager]:
    db = DatabaseManager(tmp_path / "data" / "mcp_videos.sqlite")
    db.initialize()
    workspace = WorkspaceManager(
        workspaces_root=tmp_path / "workspaces",
        voices_root=tmp_path / "data" / "voices",
    )
    workspace.ensure_project_dirs(PROJECT_ID)

    voices = VoiceRepository(db)
    projects = ProjectRepository(db)
    dialogues = DialogueRepository(db)
    scenes = SceneRepository(db)

    voices.upsert(
        VoiceProfile(
            id=VOICE_ID,
            name="Narrador de simulación",
            engine=VoiceEngine.KOKORO,
            mode=VoiceMode.PRESET,
            language=LanguageCode.ES,
        )
    )
    projects.create(
        Project(
            id=PROJECT_ID,
            title="La antena más rara de la NASA",
            description="Short de simulación para el estudio de preview.",
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
    line = dialogues.insert_line(
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
    dialogues.insert_word_timestamps(
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

    for index, (start_ms, end_ms) in enumerate(((0, 2000), (2000, TOTAL_DURATION_MS))):
        workspace.write_scene_html(
            PROJECT_ID,
            index,
            SCENE_TEMPLATE.replace("__SCENE_INDEX__", str(index)),
        )
        scenes.upsert(
            Scene(
                project_id=PROJECT_ID,
                scene_index=index,
                title=f"Escena {index}",
                start_ms=start_ms,
                end_ms=end_ms,
                html_path=f"scenes/scene_{index:02d}.html",
                validation_status="valid",
            )
        )

    return db, workspace


@dataclass
class RunningPreview:
    base_url: str
    engine: FakeRenderEngine


@pytest.fixture
def running_preview(tmp_path: Path) -> Iterator[RunningPreview]:
    db, workspace = prepare_environment(tmp_path)
    engine = FakeRenderEngine()
    app = create_preview_app(
        db,
        workspace,
        settings=AppSettings(_env_file=None),
        render_engine=engine,
    )

    port = free_tcp_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15.0
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("uvicorn terminó antes de arrancar el estudio")
        if time.monotonic() > deadline:
            raise RuntimeError("timeout esperando al servidor de preview")
        time.sleep(0.02)

    try:
        yield RunningPreview(base_url=f"http://127.0.0.1:{port}", engine=engine)
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


def iframe_clock(page: object) -> float:
    return page.evaluate(
        "() => document.getElementById('stage-iframe').contentWindow.__currentTimeMs"
    )


def iframe_label(page: object) -> str:
    return page.evaluate(
        "() => document.getElementById('stage-iframe')"
        ".contentDocument.getElementById('frame-time').textContent"
    )


def test_scrubber_and_word_cue_click_update_stage_and_render(
    running_preview: RunningPreview,
) -> None:
    with sync_playwright() as playwright:
        browser = launch_chromium(playwright)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(running_preview.base_url, wait_until="domcontentloaded")
            page.wait_for_selector(".word-cue", state="attached")
            page.wait_for_function(
                "() => window.__previewState && window.__previewState.manifest !== null"
            )
            page.wait_for_function(
                "() => {"
                "  const frame = document.getElementById('stage-iframe');"
                "  return Boolean("
                "    frame && frame.contentWindow"
                "    && typeof frame.contentWindow.__setFrameTime === 'function'"
                "  );"
                "}"
            )

            assert page.locator(".word-cue").count() == 4
            assert page.locator(".scene-chip").count() == 2

            nasa = page.locator('.word-cue[data-start-ms="1000"]')
            assert nasa.inner_text() == "NASA"
            nasa.click()
            page.wait_for_function(
                "() => Math.abs("
                "document.getElementById('stage-iframe')"
                ".contentWindow.__currentTimeMs - 1000) < 1"
            )

            assert iframe_clock(page) == pytest.approx(1000)
            assert iframe_label(page) == "1000"
            assert nasa.evaluate("(element) => element.classList.contains('active')")
            assert page.evaluate("() => window.__previewState.currentTimeMs") == 1000

            page.evaluate(
                "() => {"
                "  const slider = document.getElementById('timeline');"
                "  slider.value = '1500';"
                "  slider.dispatchEvent(new Event('input', { bubbles: true }));"
                "}"
            )
            page.wait_for_function(
                "() => Math.abs("
                "document.getElementById('stage-iframe')"
                ".contentWindow.__currentTimeMs - 1500) < 1"
            )
            assert iframe_clock(page) == pytest.approx(1500)
            assert iframe_label(page) == "1500"
            assert page.evaluate(
                "() => document.getElementById('time-label').textContent"
            ).startswith("00:01.500")

            page.evaluate(
                "() => {"
                "  const slider = document.getElementById('timeline');"
                "  slider.value = '2500';"
                "  slider.dispatchEvent(new Event('input', { bubbles: true }));"
                "}"
            )
            page.wait_for_function("() => window.__previewState.activeSceneIndex === 1")
            page.wait_for_function(
                "() => Math.abs("
                "document.getElementById('stage-iframe')"
                ".contentWindow.__currentTimeMs - 500) < 1"
            )
            assert iframe_clock(page) == pytest.approx(500)
            assert iframe_label(page) == "500"
            assert (
                page.evaluate("() => document.getElementById('stage-iframe').getAttribute('src')")
                == f"/preview/{PROJECT_ID}/scene/1"
            )

            evolutiva = page.locator('.word-cue[data-start-ms="2800"]')
            evolutiva.click()
            page.wait_for_function(
                "() => Math.abs("
                "document.getElementById('stage-iframe')"
                ".contentWindow.__currentTimeMs - 800) < 1"
            )
            assert iframe_clock(page) == pytest.approx(800)
            assert iframe_label(page) == "800"
            assert evolutiva.evaluate("(element) => element.classList.contains('active')")

            page.click("#render-button")
            page.wait_for_function("() => window.__previewState.lastRenderResponse !== null")
            payload = page.evaluate("() => window.__previewState.lastRenderResponse")
            assert payload["job_id"] == JOB_ID
            assert payload["status"] == "completed"
            assert payload["output_mp4_path"] == "output/final.mp4"
            assert running_preview.engine.calls == [(PROJECT_ID, True)]
            assert "completed" in page.locator("#render-status").inner_text()
        finally:
            browser.close()
