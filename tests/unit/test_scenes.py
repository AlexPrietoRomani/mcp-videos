"""Unit tests for the scene harness, validator instrumentation and generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from mcp_videos.config import AppSettings
from mcp_videos.core.models import WordTimestamp
from mcp_videos.scenes.generator import (
    LLMProviderError,
    SceneGenerationError,
    SceneGenerator,
    build_scene_prompt,
    build_system_prompt,
    build_technical_explainer_scene_html,
    canonical_provider,
    embed_word_cues,
    extract_html_document,
    normalize_word_cues,
)
from mcp_videos.scenes.templates import (
    AVAILABLE_TEMPLATES,
    TemplateNotFoundError,
    list_templates,
    load_template,
    unresolved_placeholders,
)
from mcp_videos.scenes.validator import (
    HARNESS_MARKER,
    SceneValidator,
    ValidationResult,
    find_chromium_executable,
    inject_harness,
    is_hyperframes_composition,
    load_harness_script,
    needs_harness,
    prepare_scene_html,
)

MOCK_SCENE_HTML = """<!DOCTYPE html>
<html lang="es">
<head><style>#scene { color: #67e8f9; }</style></head>
<body>
  <div id="scene">Telemetría</div>
  <script>
    window.onFrameTime = function (timeMs) {
      document.getElementById("scene").dataset.time = String(timeMs);
    };
  </script>
</body>
</html>
"""


class TestHarnessInjection:
    def test_harness_exposes_clock_and_motion_kit(self) -> None:
        script = load_harness_script()

        assert "window.__setFrameTime" in script
        assert "__currentTimeMs" in script
        assert "--time-ms" in script
        assert "window.MotionKit" in script
        assert "__timelines" in script
        assert "_isMcpBridge" in script
        assert "__registerSceneRender" in script
        for helper in (
            "clamp",
            "lerp",
            "easeOutCubic",
            "easeInOutQuad",
            "spring",
            "progress",
            "activeWordAt",
            "progressInRange",
        ):
            assert f"{helper}:" in script

    def test_inject_harness_inside_head_without_altering_body(self) -> None:
        html = (
            "<!DOCTYPE html><html><head><title>x</title></head>"
            "<body><main id='root'>Hola</main></body></html>"
        )

        instrumented = inject_harness(html)

        assert instrumented.count(HARNESS_MARKER) == 1
        assert instrumented.index(HARNESS_MARKER) > instrumented.index("<head>")
        assert instrumented.index(HARNESS_MARKER) < instrumented.index("<title>")
        assert "<main id='root'>Hola</main>" in instrumented
        assert not needs_harness(instrumented)

    def test_injection_is_idempotent(self) -> None:
        once = inject_harness("<html><head></head><body>x</body></html>")
        twice = inject_harness(once)

        assert twice == once

    def test_existing_clock_is_respected(self) -> None:
        html = (
            "<html><head><script>"
            "window.__setFrameTime = function (t) { window.__currentTimeMs = t; };"
            "</script></head><body>x</body></html>"
        )

        assert needs_harness(html) is False
        assert inject_harness(html) == html

    def test_inject_harness_creates_head_when_absent(self) -> None:
        instrumented = inject_harness("<html><body>Hola</body></html>")

        assert "<head>" in instrumented
        assert HARNESS_MARKER in instrumented
        assert instrumented.index(HARNESS_MARKER) < instrumented.index("<body>")

    def test_prepare_scene_html_alias(self) -> None:
        assert prepare_scene_html("<html><body>x</body></html>") == inject_harness(
            "<html><body>x</body></html>"
        )


def _launch_chromium(playwright: Any) -> Any:
    try:
        return playwright.chromium.launch(headless=True)
    except PlaywrightError:
        executable = find_chromium_executable()
        if executable is None:
            pytest.skip("Chromium no disponible en el entorno")
        return playwright.chromium.launch(headless=True, executable_path=executable)


class TestDualTimelineBridge:
    def test_is_hyperframes_composition(self) -> None:
        hf_html = '<div id="root" data-composition-id="main"><h1>Composición</h1></div>'
        standard_html = '<div id="scene">Escena Estándar</div>'
        assert is_hyperframes_composition(hf_html) is True
        assert is_hyperframes_composition(standard_html) is False

    def test_harness_dual_timeline_bridge_seek_updates_frame_time(self, tmp_path: Path) -> None:
        html = """<!DOCTYPE html>
        <html>
        <head></head>
        <body>
          <div id="stage">Test Dual Bridge</div>
          <script>
            window.__sceneRenderCalledWith = null;
            window.__registerSceneRender(function (timeMs, motionKit) {
              window.__sceneRenderCalledWith = timeMs;
            });
          </script>
        </body>
        </html>
        """
        prepared = inject_harness(html)
        scene_file = tmp_path / "test_bridge.html"
        scene_file.write_text(prepared, encoding="utf-8")

        with sync_playwright() as playwright:
            browser = _launch_chromium(playwright)
            try:
                page = browser.new_page()
                page.goto(scene_file.as_uri())

                bridge_info = page.evaluate(
                    """
                    () => ({
                      hasTimelines: typeof window.__timelines === "object",
                      isMcpBridge: Boolean(
                        window.__timelines &&
                        window.__timelines["main"] &&
                        window.__timelines["main"]._isMcpBridge
                      ),
                      defaultTime: window.__timelines["main"].time(),
                      duration: window.__timelines["main"].duration(),
                    })
                    """
                )
                assert bridge_info["hasTimelines"] is True
                assert bridge_info["isMcpBridge"] is True
                assert bridge_info["defaultTime"] == 0
                assert bridge_info["duration"] == 10.0

                result = page.evaluate(
                    """
                    () => {
                      const tl = window.__timelines["main"];
                      tl.seek(1.25);
                      return {
                        currentTimeMs: window.__currentTimeMs,
                        sceneRenderCalledWith: window.__sceneRenderCalledWith,
                        timelineTimeSec: tl.time(),
                      };
                    }
                    """
                )
                assert result["currentTimeMs"] == 1250
                assert result["sceneRenderCalledWith"] == 1250
                assert result["timelineTimeSec"] == 1.25
            finally:
                browser.close()

    def test_harness_forwards_set_frame_time_to_custom_gsap_timeline(self, tmp_path: Path) -> None:
        html = """<!DOCTYPE html>
        <html>
        <head>
          <script>
            window.__customSeekSec = null;
            window.__timelines = window.__timelines || {};
            window.__timelines["main"] = {
              seek: function (sec) {
                window.__customSeekSec = sec;
                return this;
              },
              time: function (sec) {
                if (sec === undefined) return window.__customSeekSec || 0;
                return this.seek(sec);
              },
              duration: function () { return 5.0; },
              pause: function () { return this; },
              play: function () { return this; },
            };
          </script>
        </head>
        <body>
          <div id="stage">Timeline Personalizado</div>
        </body>
        </html>
        """
        validator = SceneValidator(width=1080, height=1920)
        validation = validator.validate_html(html, duration_ms=5000)

        assert validation.valid is True
        assert validation.status.value == "valid"

        prepared = inject_harness(html)
        scene_file = tmp_path / "custom_timeline.html"
        scene_file.write_text(prepared, encoding="utf-8")

        with sync_playwright() as playwright:
            browser = _launch_chromium(playwright)
            try:
                page = browser.new_page()
                page.goto(scene_file.as_uri())

                eval_result = page.evaluate(
                    """
                    () => {
                      window.__setFrameTime(2500);
                      return {
                        currentTimeMs: window.__currentTimeMs,
                        customSeekSec: window.__customSeekSec,
                      };
                    }
                    """
                )
                assert eval_result["currentTimeMs"] == 2500
                assert eval_result["customSeekSec"] == 2.5
            finally:
                browser.close()

    def test_validator_accepts_hyperframes_composition(self) -> None:
        html = """<!DOCTYPE html>
        <html lang="es">
        <head>
          <meta charset="utf-8">
          <style>body { margin: 0; background: #000; color: #fff; }</style>
        </head>
        <body>
          <div id="root"
               data-composition-id="main"
               data-width="1080"
               data-height="1920"
               data-fps="60"
               data-duration="4.5">
            <h1 id="title">Composición HyperFrames</h1>
          </div>
          <script>
            window.__timelines = window.__timelines || {};
            window.__timelines["main"] = {
              _t: 0,
              seek: function(sec) {
                this._t = Number(sec) || 0;
                document.getElementById("title").textContent = "Time: " + this._t;
                return this;
              },
              time: function(sec) {
                if (sec === undefined) return this._t;
                return this.seek(sec);
              },
              duration: function() { return 4.5; },
              pause: function() { return this; },
              play: function() { return this; },
            };
          </script>
        </body>
        </html>
        """
        assert is_hyperframes_composition(html) is True
        validator = SceneValidator(width=1080, height=1920)
        res = validator.validate_html(html, duration_ms=4500)

        assert res.valid is True
        assert res.status.value == "valid"
        assert res.errors == []


class TestValidationResult:
    def test_valid_result_serializes(self) -> None:
        result = ValidationResult(
            duration_ms=1000,
            checked_times_ms=[0.0, 500.0, 1000.0],
        )

        assert result.valid is True
        assert result.status.value == "valid"
        assert result.as_dict()["validation_status"] == "valid"
        assert result.as_dict()["validation_errors"] == []

    def test_invalid_result_keeps_errors(self) -> None:
        result = ValidationResult(errors=["pageerror: boom"])

        assert result.valid is False
        assert result.validation_status.value == "invalid"
        assert result.console_errors == ["pageerror: boom"]


class TestExtractHtmlDocument:
    def test_extracts_fenced_document(self) -> None:
        raw = (
            "Aquí tienes la escena:\n```html\n"
            '<!DOCTYPE html>\n<html><body><h1>Hola</h1></body></html>\n```\n'
        )

        html = extract_html_document(raw)

        assert html.startswith("<!DOCTYPE html>")
        assert html.endswith("</html>")
        assert "Hola" in html

    def test_extracts_document_after_prose(self) -> None:
        raw = "Explicación\n<html lang='es'><body>x</body></html>\nFin"

        assert extract_html_document(raw) == "<html lang='es'><body>x</body></html>"

    @pytest.mark.parametrize(
        "raw",
        ["", "no hay documento", "<html><body>sin cierre</body>"],
    )
    def test_rejects_invalid_payloads(self, raw: str) -> None:
        with pytest.raises(ValueError):
            extract_html_document(raw)

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError):
            extract_html_document(b"<html></html>")  # type: ignore[arg-type]


class TestWordCueNormalization:
    def test_converts_absolute_cues_to_scene_local_time(self) -> None:
        cues = normalize_word_cues(
            [
                {"word": "dos", "start_ms": 1500, "end_ms": 1800, "confidence": 0.9},
                {"word": "mil", "start_ms": 1800, "end_ms": 2300},
            ],
            1000,
            3000,
        )

        assert cues[0] == {
            "word": "dos",
            "start_ms": 500,
            "end_ms": 800,
            "absolute_start_ms": 1500,
            "absolute_end_ms": 1800,
            "confidence": 0.9,
        }
        assert cues[1]["start_ms"] == 800
        assert cues[1]["end_ms"] == 1300

    def test_keeps_local_cues_when_explicitly_requested(self) -> None:
        cues = normalize_word_cues(
            [{"word": "local", "start_ms": 100, "end_ms": 400}],
            5000,
            6000,
            absolute=False,
        )

        assert cues[0]["start_ms"] == 100
        assert cues[0]["end_ms"] == 400

    def test_accepts_domain_models_and_skips_missing_timestamps(self) -> None:
        cues = normalize_word_cues(
            [
                WordTimestamp(
                    line_id=3,
                    word_index=0,
                    word="NASA",
                    start_ms=0,
                    end_ms=250,
                ),
                {"word": "sin tiempos"},
            ],
            0,
            1000,
        )

        assert len(cues) == 1
        assert cues[0]["word"] == "NASA"
        assert cues[0]["line_id"] == 3

    def test_clamps_reversed_timestamps(self) -> None:
        cues = normalize_word_cues(
            [{"word": "eco", "start_ms": 900, "end_ms": 900}],
            0,
            1000,
        )

        assert cues[0]["start_ms"] == 900
        assert cues[0]["end_ms"] == 901


class TestPromptConstruction:
    def test_scene_prompt_contains_word_timestamps_and_contract(self) -> None:
        prompt = build_scene_prompt(
            "La antena más rara de la NASA",
            [
                {"word": "antena", "start_ms": 0, "end_ms": 400},
                {"word": "NASA", "start_ms": 400, "end_ms": 900},
            ],
            0,
            1200,
            subtitle="Un algoritmo evolutivo",
        )

        assert "La antena más rara de la NASA" in prompt
        assert "antena" in prompt
        assert "NASA" in prompt
        assert "0.400s" in prompt
        assert "MotionKit" in prompt
        assert "mcp-word-cues" in prompt
        assert "window.onFrameTime" in prompt
        assert "1200 ms" in prompt

    def test_system_prompt_mentions_resolution_and_determinism(self) -> None:
        prompt = build_system_prompt(1080, 1920, "dark")

        assert "1080x1920" in prompt
        assert "requestAnimationFrame" in prompt
        assert "window.onFrameTime" in prompt

    def test_empty_prompt_has_explicit_no_cue_marker(self) -> None:
        prompt = build_scene_prompt("Escena", [], 0, 500)

        assert "Sin cues de palabra" in prompt


class TestTechnicalTemplate:
    def test_catalog_lists_bundled_templates(self) -> None:
        assert list_templates() == list(AVAILABLE_TEMPLATES)
        assert "technical_explainer.html" in AVAILABLE_TEMPLATES
        with pytest.raises(TemplateNotFoundError):
            load_template("../secret.html")

    def test_builds_complete_technical_scene(self) -> None:
        html = build_technical_explainer_scene_html(
            "La antena más rara de la NASA",
            [
                {"word": "En", "start_ms": 0, "end_ms": 250},
                {"word": "dos", "start_ms": 250, "end_ms": 550},
                {"word": "mil", "start_ms": 550, "end_ms": 900},
            ],
            0,
            1200,
            subtitle="Un algoritmo evolutivo diseñó una antena imposible.",
        )

        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")
        assert unresolved_placeholders(html) == []
        assert HARNESS_MARKER in html
        assert "window.__setFrameTime" in html
        assert "window.onFrameTime" in html
        assert "MotionKit" in html
        assert "La antena más rara de la NASA" in html
        assert "mcp-word-cues" in html
        assert '"word":"En"' in html
        assert "mcp-scene-metrics" in html
        assert "GANANCIA" in html
        assert 'data-duration-ms="1200"' in html
        assert "technical-diagram" in html

    def test_light_theme_and_custom_accent(self) -> None:
        html = build_technical_explainer_scene_html(
            "Escena clara",
            [],
            0,
            1000,
            theme="claro",
            accent="#f59e0b",
        )

        assert 'class="theme-light"' in html
        assert "--accent: #f59e0b" in html

    def test_metric_aliases_are_normalized(self) -> None:
        html = build_technical_explainer_scene_html(
            "Métricas",
            [{"word": "hola", "start_ms": 0, "end_ms": 500}],
            0,
            1000,
            metrics={"Ruido": 12.5, "Ganancia": {"value": 33, "unit": "dBi"}},
        )

        assert '"label":"Ruido"' in html
        assert '"label":"Ganancia"' in html
        assert '"unit":"dBi"' in html

    def test_embed_word_cues_adds_json_once(self) -> None:
        html = "<html><body><main>x</main></body></html>"
        cues = normalize_word_cues(
            [{"word": "hola", "start_ms": 0, "end_ms": 100}],
            0,
            500,
        )

        once = embed_word_cues(html, cues)
        twice = embed_word_cues(once, cues)

        assert twice == once
        assert "mcp-word-cues" in once
        assert json.loads(
            once.split('id="mcp-word-cues" data-mcp-word-cues="1">', 1)[1].split("</script>", 1)[0]
        )[0]["word"] == "hola"


class TestSceneGeneratorProviders:
    @staticmethod
    def _client(handler: object) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_ollama_native_chat_uses_local_api(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            payload = json.loads(request.content)
            assert payload["model"] == "qwen-local"
            assert payload["stream"] is False
            assert payload["messages"][0]["role"] == "system"
            return httpx.Response(200, json={"message": {"content": "OK ollama"}})

        generator = SceneGenerator(
            provider="ollama",
            model="qwen-local",
            base_url="http://localhost:11434/v1",
            http_client=self._client(handler),
        )

        assert generator.complete("hola", system="sistema") == "OK ollama"
        assert str(requests[0].url) == "http://localhost:11434/api/chat"

    def test_vllm_openai_compatible_chat(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://vllm.local/v1/chat/completions"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "OK vllm"}}]},
            )

        generator = SceneGenerator(
            provider="vllm",
            model="qwen",
            base_url="http://vllm.local/v1",
            http_client=self._client(handler),
        )

        assert generator.complete("hola") == "OK vllm"

    def test_anthropic_messages_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://anthropic.local/v1/messages"
            assert request.headers["x-api-key"] == "secret"
            assert request.headers["anthropic-version"] == "2023-06-01"
            payload = json.loads(request.content)
            assert payload["system"] == "sistema"
            assert payload["messages"] == [{"role": "user", "content": "hola"}]
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "OK claude"}]},
            )

        generator = SceneGenerator(
            provider="anthropic",
            model="claude-test",
            base_url="http://anthropic.local",
            api_key="secret",
            http_client=self._client(handler),
        )

        assert generator.complete("hola", system="sistema") == "OK claude"

    def test_gemini_generate_content_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1beta/models/gemini-test:generateContent"
            assert request.url.params["key"] == "google-key"
            payload = json.loads(request.content)
            assert payload["system_instruction"]["parts"][0]["text"] == "sistema"
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "OK gemini"}]}}
                    ]
                },
            )

        generator = SceneGenerator(
            provider="gemini",
            model="gemini-test",
            base_url="http://gemini.local",
            api_key="google-key",
            http_client=self._client(handler),
        )

        assert generator.complete("hola", system="sistema") == "OK gemini"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = AppSettings(openai_api_key=None)
        generator = SceneGenerator(settings=settings, provider="openai")

        with pytest.raises(SceneGenerationError):
            generator.complete("hola")

    def test_http_error_is_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="fallo interno")

        generator = SceneGenerator(
            provider="vllm",
            base_url="http://vllm.local/v1",
            http_client=self._client(handler),
        )

        with pytest.raises(LLMProviderError):
            generator.complete("hola")

    def test_generated_llm_scene_is_instrumented_with_cues(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": MOCK_SCENE_HTML}}]},
            )

        generator = SceneGenerator(
            provider="vllm",
            base_url="http://vllm.local/v1",
            http_client=self._client(handler),
        )
        words = [
            {"word": "antena", "start_ms": 0, "end_ms": 400},
            {"word": "NASA", "start_ms": 400, "end_ms": 900},
        ]

        html = generator.generate_scene_html(
            "Antena NASA",
            words,
            0,
            1000,
        )

        assert HARNESS_MARKER in html
        assert "mcp-word-cues" in html
        assert '"word":"antena"' in html

    def test_template_provider_never_calls_http(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("No debe usar HTTP")

        generator = SceneGenerator(
            provider="template",
            http_client=self._client(handler),
        )

        html = generator.generate_scene_html(
            "Escena local",
            [{"word": "hola", "start_ms": 0, "end_ms": 300}],
            0,
            800,
        )

        assert "window.onFrameTime" in html
        assert "mcp-word-cues" in html

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(SceneGenerationError):
            canonical_provider("desconocido")

    def test_template_complete_is_rejected(self) -> None:
        generator = SceneGenerator(provider="template")

        with pytest.raises(SceneGenerationError):
            generator.complete("hola")


def test_generator_module_exposes_template_path() -> None:
    template = Path("src/mcp_videos/scenes/templates/technical_explainer.html")

    assert template.is_file()
