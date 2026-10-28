"""User-simulation tests for headless scene validation and generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from mcp_videos.core.models import Project
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import ProjectRepository, SceneRepository
from mcp_videos.scenes.generator import SceneGenerator
from mcp_videos.scenes.validator import (
    HARNESS_MARKER,
    SceneValidator,
    find_chromium_executable,
    inject_harness,
)

PROJECT_ID = "short-antena-st5"
TOTAL_DURATION_MS = 4500
GOLDEN_TITLE = "La antena más rara de la NASA"
GOLDEN_WORDS = [
    {"word": "En", "start_ms": 0, "end_ms": 250, "confidence": 0.99},
    {"word": "dos", "start_ms": 250, "end_ms": 520, "confidence": 0.98},
    {"word": "mil", "start_ms": 520, "end_ms": 860, "confidence": 0.97},
    {"word": "seis", "start_ms": 860, "end_ms": 1200, "confidence": 0.96},
    {"word": "la", "start_ms": 1200, "end_ms": 1380, "confidence": 0.99},
    {"word": "NASA", "start_ms": 1380, "end_ms": 1800, "confidence": 0.95},
    {"word": "lanzó", "start_ms": 1800, "end_ms": 2200, "confidence": 0.94},
    {"word": "una", "start_ms": 2200, "end_ms": 2400, "confidence": 0.98},
    {"word": "antena", "start_ms": 2400, "end_ms": 2900, "confidence": 0.97},
    {"word": "evolutiva", "start_ms": 2900, "end_ms": 3600, "confidence": 0.93},
]

VALID_SCENE_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <style>
    html, body { margin: 0; background: #05070d; color: #eaf6ff; }
    #probe { display: block; width: 240px; height: 160px; }
  </style>
</head>
<body>
  <svg id="probe" viewBox="0 0 240 160">
    <circle id="dot" cx="120" cy="80" r="20" fill="#67e8f9"></circle>
  </svg>
  <div id="frame-label">0</div>
  <script>
    window.onFrameTime = function (timeMs) {
      var progress = window.MotionKit.progressInRange(timeMs, 0, 1200);
      var probe = document.getElementById("probe");
      probe.setAttribute("data-progress", progress.toFixed(4));
      probe.style.opacity = String(progress);
      document.getElementById("dot").setAttribute(
        "r",
        (20 + progress * 30).toFixed(2)
      );
      document.getElementById("frame-label").textContent = String(Math.round(timeMs));
    };
  </script>
</body>
</html>
"""

MOCK_LLM_SCENE = """<!DOCTYPE html>
<html lang="es">
<head><style>body { margin: 0; background: #07111f; color: #eaf6ff; }</style></head>
<body>
  <main id="scene">Escena LLM</main>
  <script>
    window.onFrameTime = function (timeMs) {
      document.getElementById("scene").dataset.time = String(timeMs);
    };
  </script>
</body>
</html>
"""


@dataclass
class SceneEnvironment:
    db: DatabaseManager
    project_repo: ProjectRepository
    scene_repo: SceneRepository
    workspace: WorkspaceManager
    validator: SceneValidator


@pytest.fixture
def environment(tmp_path: Path) -> SceneEnvironment:
    db = DatabaseManager(tmp_path / "mcp_videos.sqlite")
    db.initialize()
    project_repo = ProjectRepository(db)
    scene_repo = SceneRepository(db)
    workspace = WorkspaceManager(
        workspaces_root=tmp_path / "workspaces",
        voices_root=tmp_path / "data" / "voices",
    )
    project_repo.create(
        Project(
            id=PROJECT_ID,
            title=GOLDEN_TITLE,
            description="Antena evolucionada por algoritmo genético.",
            width=1080,
            height=1920,
            fps=60,
            total_duration_ms=TOTAL_DURATION_MS,
            master_audio_path="audio/dialogue_master.wav",
            status="audio_ready",
        )
    )
    validator = SceneValidator(
        scene_repo,
        workspace=workspace,
        width=1080,
        height=1920,
        timeout_ms=20_000,
    )
    return SceneEnvironment(
        db=db,
        project_repo=project_repo,
        scene_repo=scene_repo,
        workspace=workspace,
        validator=validator,
    )


def launch_chromium(playwright: object) -> object:
    try:
        return playwright.chromium.launch(headless=True)
    except PlaywrightError as bundled_error:
        executable = find_chromium_executable()
        if executable is None:
            pytest.skip(f"Chromium no disponible: {bundled_error}")
        return playwright.chromium.launch(headless=True, executable_path=executable)


def test_harness_mutates_dom_and_motion_kit_math(tmp_path: Path) -> None:
    scene_path = tmp_path / "probe.html"
    scene_path.write_text(inject_harness(VALID_SCENE_HTML), encoding="utf-8")

    with sync_playwright() as playwright:
        browser = launch_chromium(playwright)
        try:
            page = browser.new_page(viewport={"width": 1080, "height": 1920})
            page.goto(scene_path.as_uri(), wait_until="load")

            assert page.evaluate("window.MotionKit.progress(500, 0, 1000)") == pytest.approx(0.5)
            assert page.evaluate("window.MotionKit.clamp(12, 0, 10)") == 10
            assert page.evaluate("window.MotionKit.lerp(0, 100, 0.25)") == pytest.approx(25)
            assert page.evaluate("window.MotionKit.easeOutCubic(0.5)") == pytest.approx(0.875)
            assert page.evaluate("window.MotionKit.spring(0)") == pytest.approx(0)
            assert page.evaluate("window.MotionKit.spring(4)") > 0.9
            assert page.evaluate(
                "window.MotionKit.activeWordAt(1, [{start_ms: 0, end_ms: 2, word: 'x'}]).word"
            ) == "x"

            page.evaluate(
                "window.__probeAnimation = document.getElementById('probe')"
                ".animate([{ opacity: 0 }, { opacity: 1 }], { duration: 1000 })"
            )
            page.evaluate("window.__setFrameTime(300)")
            assert page.evaluate("window.__probeAnimation.currentTime") == pytest.approx(300)
            assert page.evaluate("window.__probeAnimation.playState") == "paused"
            assert page.evaluate("window.__currentTimeMs") == pytest.approx(300)
            assert page.evaluate(
                "document.getElementById('probe').getAttribute('data-progress')"
            ) == "0.2500"
            assert page.evaluate(
                "document.getElementById('dot').getAttribute('r')"
            ) == "27.50"
            assert page.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--time-ms').trim()"
            ) == "300"

            page.evaluate("window.__setFrameTime(1200)")
            assert page.evaluate(
                "document.getElementById('probe').getAttribute('data-progress')"
            ) == "1.0000"
        finally:
            browser.close()


def test_valid_scene_is_persisted_as_valid(environment: SceneEnvironment) -> None:
    scene = environment.validator.validate_and_save(
        PROJECT_ID,
        0,
        VALID_SCENE_HTML,
        0,
        TOTAL_DURATION_MS,
        title="Hook Antena Evolucionada",
        prompt_used="golden-valid",
    )

    assert scene.validation_status == "valid"
    assert scene.validation_errors == []
    assert scene.html_path == "scenes/scene_00.html"

    stored = environment.scene_repo.get(PROJECT_ID, 0)
    assert stored is not None
    assert stored.validation_status == "valid"
    assert stored.validation_errors_json == "[]"

    scene_file = environment.workspace.project_dir(PROJECT_ID) / "scenes" / "scene_00.html"
    content = scene_file.read_text(encoding="utf-8")
    assert HARNESS_MARKER in content
    assert "window.onFrameTime" in content


def test_runtime_error_marks_scene_invalid_with_exact_message(
    environment: SceneEnvironment,
) -> None:
    broken = """<!DOCTYPE html>
    <html><head></head><body>
      <div id="stage">Escena rota</div>
      <script>
        window.onFrameTime = function (timeMs) {
          if (timeMs > 0) { nonExistentFunction(timeMs); }
        };
      </script>
    </body></html>"""

    scene = environment.validator.validate_and_save(
        PROJECT_ID,
        0,
        broken,
        0,
        TOTAL_DURATION_MS,
        title="Escena rota",
    )

    assert scene.validation_status == "invalid"
    joined = "\n".join(scene.validation_errors)
    assert "nonExistentFunction" in joined
    assert "ReferenceError" in joined

    stored = environment.scene_repo.get(PROJECT_ID, 0)
    assert stored is not None
    assert stored.validation_status == "invalid"
    persisted_errors = json.loads(stored.validation_errors_json)
    assert any("nonExistentFunction" in message for message in persisted_errors)


def test_console_error_marks_scene_invalid(environment: SceneEnvironment) -> None:
    noisy = """<!DOCTYPE html>
    <html><head></head><body>
      <div id="stage">Escena con ruido</div>
      <script>
        window.onFrameTime = function () {
          console.error("telemetria rota");
        };
      </script>
    </body></html>"""

    scene = environment.validator.validate_and_save(
        PROJECT_ID,
        0,
        noisy,
        0,
        TOTAL_DURATION_MS,
        title="Escena con ruido",
    )

    assert scene.validation_status == "invalid"
    assert any("telemetria rota" in message for message in scene.validation_errors)


def test_missing_clock_contract_marks_scene_invalid(
    environment: SceneEnvironment,
) -> None:
    no_clock = """<!DOCTYPE html>
    <html><head></head><body>
      <div id="stage">Sin reloj</div>
      <script>window.__setFrameTime = null;</script>
    </body></html>"""

    scene = environment.validator.validate_and_save(
        PROJECT_ID,
        0,
        no_clock,
        0,
        TOTAL_DURATION_MS,
        title="Sin reloj",
    )

    assert scene.validation_status == "invalid"
    assert any(
        "__setFrameTime" in message for message in scene.validation_errors
    )


def test_failed_resource_request_marks_scene_invalid(
    environment: SceneEnvironment,
) -> None:
    missing_asset = """<!DOCTYPE html>
    <html><head></head><body>
      <div id="stage">Recurso ausente</div>
      <script src="missing-dependency.js"></script>
      <script>
        window.onFrameTime = function (timeMs) {
          document.getElementById("stage").dataset.time = String(timeMs);
        };
      </script>
    </body></html>"""

    scene = environment.validator.validate_and_save(
        PROJECT_ID,
        0,
        missing_asset,
        0,
        TOTAL_DURATION_MS,
        title="Recurso ausente",
    )

    assert scene.validation_status == "invalid"
    assert any("requestfailed" in message for message in scene.validation_errors)


def test_generated_technical_scene_validates_in_chromium(
    environment: SceneEnvironment,
) -> None:
    generator = SceneGenerator(provider="template")
    html = generator.generate_scene_html(
        GOLDEN_TITLE,
        GOLDEN_WORDS,
        0,
        TOTAL_DURATION_MS,
        subtitle="Un algoritmo evolutivo probó miles de formas.",
        eyebrow="TELEMETRÍA NASA",
    )

    scene = environment.validator.validate_and_save(
        PROJECT_ID,
        1,
        html,
        0,
        TOTAL_DURATION_MS,
        title="Diagrama Antena Evolutiva",
        prompt_used="template-technical-explainer",
    )

    assert scene.validation_status == "valid"
    assert scene.validation_errors == []

    scene_file = environment.workspace.project_dir(PROJECT_ID) / "scenes" / "scene_01.html"
    content = scene_file.read_text(encoding="utf-8")
    assert GOLDEN_TITLE in content
    assert '"word":"NASA"' in content
    assert "mcp-word-cues" in content
    assert "technical-diagram" in content


def test_llm_generated_scene_validates_in_chromium(
    environment: SceneEnvironment,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": MOCK_LLM_SCENE}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        generator = SceneGenerator(
            provider="vllm",
            base_url="http://vllm.local/v1",
            http_client=client,
            validator=environment.validator,
        )
        scene = generator.generate_scene(
            PROJECT_ID,
            2,
            "Escena LLM local",
            GOLDEN_WORDS,
            0,
            TOTAL_DURATION_MS,
            workspace=environment.workspace,
            subtitle="Generada por un modelo local compatible con OpenAI.",
            duration_ms=TOTAL_DURATION_MS,
            provider="vllm",
        )
    finally:
        client.close()

    assert scene.validation_status == "valid"
    assert scene.validation_errors == []
    scene_file = environment.workspace.project_dir(PROJECT_ID) / "scenes" / "scene_02.html"
    content = scene_file.read_text(encoding="utf-8")
    assert HARNESS_MARKER in content
    assert "mcp-word-cues" in content
    assert '"word":"NASA"' in content
