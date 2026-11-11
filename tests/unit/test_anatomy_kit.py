"""Unit and simulation tests for Anatomy IsoKit packaging, kinematic solvers and templates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from mcp_videos.core.models import Project, ValidationStatus
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import ProjectRepository, SceneRepository
from mcp_videos.scenes.anatomy_kit import (
    ANATOMY_KIT_DIR,
    ISO_KIT_EXPORTS,
    load_iso_css,
    load_iso_kit_script,
    load_line_ends_script,
)
from mcp_videos.scenes.generator import (
    SceneGenerator,
    build_system_prompt,
    render_deterministic_scene,
)
from mcp_videos.scenes.templates import (
    AVAILABLE_TEMPLATES,
    ISOMETRIC_INSTRUMENT_TEMPLATE,
    list_templates,
    load_template,
    render_template,
    unresolved_placeholders,
)
from mcp_videos.scenes.validator import (
    HARNESS_MARKER,
    SceneValidator,
    find_chromium_executable,
    inject_harness,
    load_harness_script,
)


def _launch_chromium(playwright: Any) -> Any:
    try:
        return playwright.chromium.launch(headless=True)
    except PlaywrightError:
        executable = find_chromium_executable()
        if executable is None:
            pytest.skip("Chromium no disponible en el entorno")
        return playwright.chromium.launch(headless=True, executable_path=executable)


class TestAnatomyKitPackaging:
    """Verifica el empaquetado y utilidades de inlining para Anatomy IsoKit (T7.1.1)."""

    def test_anatomy_kit_directory_and_files_exist(self) -> None:
        assert ANATOMY_KIT_DIR.is_dir()
        for filename in ("iso.css", "iso-kit.mjs", "line-ends.mjs", "lathe.mjs", "tube.mjs"):
            assert (ANATOMY_KIT_DIR / filename).is_file(), f"Falta archivo: {filename}"

    def test_load_iso_css(self) -> None:
        css = load_iso_css()
        assert isinstance(css, str)
        assert len(css) > 500
        assert ".iso" in css
        assert "--anatomy-page" in css
        assert "--anatomy-weight" in css
        assert ".iso-svg" in css

    def test_load_iso_kit_script(self) -> None:
        script = load_iso_kit_script()
        assert isinstance(script, str)
        assert len(script) > 10_000
        assert "window.IsoKit" in script or "global.IsoKit" in script
        assert "ensureStyles" in script
        for export_name in ISO_KIT_EXPORTS:
            assert export_name in script, f"Falta export: {export_name}"

    def test_load_line_ends_script(self) -> None:
        script = load_line_ends_script()
        assert isinstance(script, str)
        assert "lineEnds" in script
        assert "window.lineEnds = lineEnds" in script

    def test_iso_kit_in_headless_browser(self) -> None:
        script = load_iso_kit_script()
        with sync_playwright() as playwright:
            browser = _launch_chromium(playwright)
            try:
                page = browser.new_page()
                page.set_content("<!DOCTYPE html><html><head></head><body></body></html>")
                page.evaluate(script)

                evaluated = page.evaluate(
                    """
                    () => {
                      const k = window.IsoKit;
                      const P = k.fitProjection([[-50, -50, 0], [50, 50, 20]], 400, 300);
                      const slab = k.slabOf({ x: -20, y: -20, w: 40, d: 40, r: 2 }, 0, 10, P);
                      const cyl = k.cylinder(0, 0, 8, 10, 15, P);
                      const svg = k.figureSvg({
                        width: 400,
                        height: 300,
                        body: [k.solidSvg(slab), k.solidSvg(cyl)],
                      });
                      k.ensureStyles(document);
                      const hasStyle = Boolean(document.querySelector("style[data-mcp-iso-css]"));
                      return {
                        hasIsoKit: typeof k === "object",
                        hasFitProjection: typeof k.fitProjection === "function",
                        hasSlab: Boolean(slab && slab.top),
                        hasCylinder: Boolean(cyl && cyl.top),
                        hasSvg: Boolean(svg && svg.includes("<svg")),
                        hasStyle: hasStyle,
                      };
                    }
                    """
                )
                assert evaluated["hasIsoKit"] is True
                assert evaluated["hasFitProjection"] is True
                assert evaluated["hasSlab"] is True
                assert evaluated["hasCylinder"] is True
                assert evaluated["hasSvg"] is True
                assert evaluated["hasStyle"] is True
            finally:
                browser.close()


class TestHarnessDeterminismAndIsoKit:
    """Verifica la integración de IsoKit y cinemática determinista en harness.js (T7.1.2)."""

    def test_harness_bundles_isokit_and_kinematics(self) -> None:
        script = load_harness_script()
        assert "window.IsoKit" in script
        assert "stepDampedAt" in script
        assert "stepSpringAt" in script
        assert "ensureStyles" in script

    def test_step_damped_at_mathematical_determinism(self) -> None:
        script = load_harness_script()
        with sync_playwright() as playwright:
            browser = _launch_chromium(playwright)
            try:
                page = browser.new_page()
                page.set_content("<!DOCTYPE html><html><head></head><body></body></html>")
                page.evaluate(script)

                keyframes = [[0, 0], [1000, 100], [3000, 50]]

                # Evaluar t=0
                r0 = page.evaluate(
                    "(args) => window.MotionKit.stepDampedAt(args[0], args[1], args[2])",
                    [0, keyframes, 0.28],
                )
                assert r0["value"] == 0.0
                assert r0["target"] == 0.0
                assert r0["index"] == 0

                # Evaluar t=1280 (1 tau = 280ms tras escalón en 1000ms -> aprox 63.2% de subida)
                r_tau = page.evaluate(
                    "(args) => window.MotionKit.stepDampedAt(args[0], args[1], args[2])",
                    [1280, keyframes, 0.28],
                )
                expected_tau = 100.0 * (1.0 - math.exp(-1.0))
                assert math.isclose(r_tau["value"], expected_tau, rel_tol=1e-3)
                assert r_tau["target"] == 100.0
                assert r_tau["index"] == 1

                # Determinismo estricto: scrub hacia adelante y atrás
                val_t2000_first = page.evaluate(
                    "(args) => window.MotionKit.stepDampedAt(args[0], args[1], args[2]).value",
                    [2000, keyframes, 0.28],
                )
                val_t500 = page.evaluate(
                    "(args) => window.MotionKit.stepDampedAt(args[0], args[1], args[2]).value",
                    [500, keyframes, 0.28],
                )
                val_t2000_second = page.evaluate(
                    "(args) => window.MotionKit.stepDampedAt(args[0], args[1], args[2]).value",
                    [2000, keyframes, 0.28],
                )

                assert val_t500 == 0.0
                assert val_t2000_first == val_t2000_second
            finally:
                browser.close()

    def test_step_spring_at_mathematical_determinism(self) -> None:
        script = load_harness_script()
        with sync_playwright() as playwright:
            browser = _launch_chromium(playwright)
            try:
                page = browser.new_page()
                page.set_content("<!DOCTYPE html><html><head></head><body></body></html>")
                page.evaluate(script)

                keyframes = [[0, 0], [200, 10]]

                # A t=0 debe valer 0
                s0 = page.evaluate(
                    "(args) => window.MotionKit.stepSpringAt(args[0], args[1])",
                    [0, keyframes],
                )
                assert s0["value"] == 0.0
                assert s0["velocity"] == 0.0

                # A t=1500 el resorte debe haberse asentado en 10
                s_settled = page.evaluate(
                    "(args) => window.MotionKit.stepSpringAt(args[0], args[1])",
                    [1500, keyframes],
                )
                assert math.isclose(s_settled["value"], 10.0, abs_tol=1e-2)

                # Probar con función targetFn(tMs)
                s_fn = page.evaluate(
                    """
                    () => {
                      const fn = (tMs) => (tMs >= 300 ? 25 : 5);
                      return window.MotionKit.stepSpringAt(
                        1500, fn, { stiffness: 420, damping: 26 }
                      );
                    }
                    """
                )
                assert math.isclose(s_fn["value"], 25.0, abs_tol=1e-2)

                # Prueba de determinismo idéntico ante scrubbing desordenado
                run1 = page.evaluate(
                    "() => window.MotionKit.stepSpringAt(600, [[0, 0], [100, 15]]).value"
                )
                page.evaluate("() => window.MotionKit.stepSpringAt(2000, [[0, 0], [100, 15]])")
                run2 = page.evaluate(
                    "() => window.MotionKit.stepSpringAt(600, [[0, 0], [100, 15]]).value"
                )
                assert run1 == run2
            finally:
                browser.close()

    def test_harness_auto_injects_iso_styles_in_dom(self) -> None:
        html = (
            "<!DOCTYPE html><html><head></head>"
            '<body><div class="iso-plate">Plate</div></body></html>'
        )
        prepared = inject_harness(html)

        with sync_playwright() as playwright:
            browser = _launch_chromium(playwright)
            try:
                page = browser.new_page()
                page.set_content(prepared)

                # Antes de setFrameTime no hay estilo aún
                before = page.evaluate('Boolean(document.querySelector("style[data-mcp-iso-css]"))')
                assert before is False

                # Al disparar setFrameTime, detecta .iso-plate e inyecta
                page.evaluate("window.__setFrameTime(0)")
                after = page.evaluate('Boolean(document.querySelector("style[data-mcp-iso-css]"))')
                assert after is True
            finally:
                browser.close()


class TestIsometricInstrumentTemplate:
    """Verifica la plantilla isometric_instrument.html y su cinemática (T7.2.1)."""

    def test_template_is_registered(self) -> None:
        assert ISOMETRIC_INSTRUMENT_TEMPLATE in AVAILABLE_TEMPLATES
        assert ISOMETRIC_INSTRUMENT_TEMPLATE in list_templates()
        raw = load_template(ISOMETRIC_INSTRUMENT_TEMPLATE)
        assert len(raw) > 5000
        assert "diagram-card" in raw
        assert "iso-plate" in raw

    def test_render_isometric_instrument_template(self) -> None:
        context = {
            "TITLE": "Instrumento Isométrico de Precisión",
            "SUBTITLE": "Comparador micrométrico con bloques patrón calibrados.",
            "EYEBROW": "METROLOGÍA DE ALTA RESOLUCIÓN",
            "THEME": "dark",
            "ACCENT": "#67e8f9",
            "WIDTH": 1080,
            "HEIGHT": 1920,
            "START_MS": 0,
            "DURATION_MS": 4500,
            "END_SECONDS": "00:04.500",
            "METRICS_HTML": "[]",
            "WORDS_HTML": '<span class="word" data-start-ms="0" data-end-ms="500">Medición</span>',
            "WORD_CUES_JSON": '[{"word": "Medición", "start_ms": 0, "end_ms": 500}]',
        }
        rendered = render_template(ISOMETRIC_INSTRUMENT_TEMPLATE, context)
        assert unresolved_placeholders(rendered) == []
        assert "Instrumento Isométrico de Precisión" in rendered
        assert 'data-part="carriage"' in rendered
        assert 'data-part="plunger"' in rendered
        assert 'data-part="needle"' in rendered
        assert 'data-free="both"' in rendered
        assert "data-readout" in rendered

    def test_isometric_instrument_validates_in_scene_validator(self) -> None:
        context = {
            "TITLE": "Validación Headless de Instrumento",
            "SUBTITLE": "Comprobación en 3 puntos temporales con SceneValidator.",
            "EYEBROW": "AUDITORÍA TÉCNICA",
            "THEME": "dark",
            "ACCENT": "#67e8f9",
            "WIDTH": 1080,
            "HEIGHT": 1920,
            "START_MS": 0,
            "DURATION_MS": 3000,
            "END_SECONDS": "00:03.000",
            "METRICS_HTML": "[]",
            "WORDS_HTML": "<span>calibración</span>",
            "WORD_CUES_JSON": "[]",
        }
        rendered = render_template(ISOMETRIC_INSTRUMENT_TEMPLATE, context)
        validator = SceneValidator(width=1080, height=1920)
        result = validator.validate_html(rendered, duration_ms=3000)

        assert result.valid is True
        assert result.status.value == "valid"
        assert result.errors == []

    def test_isometric_instrument_animation_scrub_in_playwright(self) -> None:
        context = {
            "TITLE": "Scrubbing Animado de Instrumento",
            "SUBTITLE": "Verificación del movimiento del carro y rotación de aguja.",
            "EYEBROW": "CINEMÁTICA EN VIVO",
            "THEME": "dark",
            "ACCENT": "#67e8f9",
            "WIDTH": 1080,
            "HEIGHT": 1920,
            "START_MS": 0,
            "DURATION_MS": 4000,
            "END_SECONDS": "00:04.000",
            "METRICS_HTML": "[]",
            "WORDS_HTML": "<span>test</span>",
            "WORD_CUES_JSON": "[]",
        }
        rendered = inject_harness(render_template(ISOMETRIC_INSTRUMENT_TEMPLATE, context))

        with sync_playwright() as playwright:
            browser = _launch_chromium(playwright)
            try:
                page = browser.new_page(viewport={"width": 1080, "height": 1920})
                page.set_content(rendered)

                # En t=0 (primer bloque)
                page.evaluate("window.__setFrameTime(0)")
                s0 = page.evaluate(
                    """
                    () => {
                      const carriage = document.querySelector('[data-part="carriage"]');
                      const needle = document.querySelector('[data-part="needle"]');
                      const readout = document.querySelector('[data-readout]');
                      return {
                        carriageTransform: carriage.getAttribute("transform"),
                        needleD: needle.getAttribute("d"),
                        readout: readout.textContent,
                      };
                    }
                    """
                )
                assert "translate" in s0["carriageTransform"]
                assert s0["needleD"].startswith("M")
                assert "BLOQUE" in s0["readout"] or "mm" in s0["readout"]

                # En t=3500 (desplazado a bloque posterior)
                page.evaluate("window.__setFrameTime(3500)")
                s_later = page.evaluate(
                    """
                    () => {
                      const carriage = document.querySelector('[data-part="carriage"]');
                      const needle = document.querySelector('[data-part="needle"]');
                      const readout = document.querySelector('[data-readout]');
                      return {
                        carriageTransform: carriage.getAttribute("transform"),
                        needleD: needle.getAttribute("d"),
                        readout: readout.textContent,
                      };
                    }
                    """
                )
                # El carro y aguja deben haberse movido respecto a t=0
                assert s_later["carriageTransform"] != s0["carriageTransform"]
                assert s_later["needleD"] != s0["needleD"]
            finally:
                browser.close()


class TestValidatorIsoLineEnds:
    """Verifica la auditoría de puntas de línea (lineEnds) en SceneValidator (T7.2.2)."""

    def test_clean_plate_passes_iso_line_audit(self) -> None:
        html = """<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"></head>
        <body>
          <div class="iso iso-plate">
            <svg width="400" height="400" viewBox="0 0 400 400">
              <path class="iso-line" data-free="both"
                    d="M 50 50 L 350 350" stroke="#fff" stroke-width="2"></path>
            </svg>
          </div>
        </body>
        </html>
        """
        validator = SceneValidator(width=1080, height=1920)
        res = validator.validate_html(html, duration_ms=1000, check_iso_lines=True)
        assert res.valid is True
        assert res.iso_line_problems == []
        assert res.errors == []

    def test_unanchored_line_fails_when_check_iso_lines_is_true(self) -> None:
        html = """<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"></head>
        <body>
          <div class="iso iso-plate">
            <svg width="400" height="400" viewBox="0 0 400 400">
              <path class="iso-line" d="M 50 50 L 350 350" stroke="#fff" stroke-width="2"></path>
            </svg>
          </div>
        </body>
        </html>
        """
        validator = SceneValidator(width=1080, height=1920)
        res = validator.validate_html(html, duration_ms=1000, check_iso_lines=True)
        assert res.valid is False
        assert len(res.iso_line_problems) > 0
        assert any("iso-line flotante" in err for err in res.errors)

    def test_unanchored_line_with_data_audit_lines_attribute(self) -> None:
        html = """<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"></head>
        <body>
          <div class="iso iso-plate" data-audit-lines="1">
            <svg width="400" height="400" viewBox="0 0 400 400">
              <path class="iso-line" d="M 50 50 L 350 350" stroke="#fff" stroke-width="2"></path>
            </svg>
          </div>
        </body>
        </html>
        """
        validator = SceneValidator(width=1080, height=1920)
        res = validator.validate_html(html, duration_ms=1000)
        assert res.valid is False
        assert len(res.iso_line_problems) > 0

    def test_isometric_instrument_template_has_zero_line_problems(self) -> None:
        context = {
            "TITLE": "Instrumento Isométrico sin Problemas de Líneas",
            "SUBTITLE": "Comprobación estricta de puntas de línea.",
            "EYEBROW": "AUDITORÍA GEOMÉTRICA",
            "THEME": "dark",
            "ACCENT": "#67e8f9",
            "WIDTH": 1080,
            "HEIGHT": 1920,
            "START_MS": 0,
            "DURATION_MS": 3000,
            "END_SECONDS": "00:03.000",
            "METRICS_HTML": "[]",
            "WORDS_HTML": "<span>calibración</span>",
            "WORD_CUES_JSON": "[]",
        }
        rendered = render_template(ISOMETRIC_INSTRUMENT_TEMPLATE, context)
        validator = SceneValidator(width=1080, height=1920)
        res = validator.validate_html(rendered, duration_ms=3000, check_iso_lines=True)
        assert res.valid is True
        assert res.iso_line_problems == []
        assert res.errors == []


class TestGeneratorIsometric:
    """Verifica el soporte de IsoKit e isometric_instrument.html en SceneGenerator (T7.3.1)."""

    def test_build_system_prompt_documents_isokit_and_kinematics(self) -> None:
        prompt = build_system_prompt(1080, 1920, "dark")
        assert "window.IsoKit" in prompt
        assert "IsoKit.fitProjection" in prompt
        assert "MotionKit.stepSpringAt" in prompt
        assert "MotionKit.stepDampedAt" in prompt
        assert "wheresryan22/anatomy" in prompt

    def test_render_deterministic_scene_explicit_isometric_template(self) -> None:
        html = render_deterministic_scene(
            title="Calibrador de Precisión",
            template_name="isometric_instrument.html",
            duration_ms=3500,
            subtitle="Banco micrométrico",
        )
        assert html.startswith("<!DOCTYPE html>")
        assert "60 FPS · ISOKIT" in html
        assert "Calibrador de Precisión" in html
        assert "Banco micrométrico" in html
        assert HARNESS_MARKER in html
        assert unresolved_placeholders(html) == []

    @pytest.mark.parametrize(
        "title",
        [
            "Calibrador micrométrico",
            "Anatomy machine",
            "Instrumento de precisión",
            "Dial de medición",
            "Máquina isométrica",
            "Esquema Isométrico",
        ],
    )
    def test_render_deterministic_scene_auto_detects_isometric_keywords(self, title: str) -> None:
        html = render_deterministic_scene(title=title, duration_ms=2000)
        assert "60 FPS · ISOKIT" in html

    def test_render_deterministic_scene_fallback_to_technical_explainer(self) -> None:
        html = render_deterministic_scene(title="Diagrama de red neuronal", duration_ms=2000)
        assert "technical-diagram" in html
        assert "60 FPS · ISOKIT" not in html

    def test_scene_generator_generate_scene_with_isometric_template(
        self, tmp_path: Path
    ) -> None:
        db = DatabaseManager(tmp_path / "data" / "test.sqlite")
        db.apply_migrations()
        projects = ProjectRepository(db)
        scenes = SceneRepository(db)
        workspace = WorkspaceManager(tmp_path / "workspaces", tmp_path / "voices")

        project = projects.upsert(
            Project(
                id="proj-iso-1",
                title="Proyecto Isométrico",
                total_duration_ms=4000,
                width=1080,
                height=1920,
            )
        )
        validator = SceneValidator(repository=scenes, workspace=workspace)
        generator = SceneGenerator(provider="template", validator=validator)

        scene = generator.generate_scene(
            project_id=project.id,
            scene_index=0,
            title="Instrumento Calibrador",
            words=[{"word": "medición", "start_ms": 0, "end_ms": 500}],
            start_ms=0,
            end_ms=3000,
            workspace=workspace,
            prompt_used="calibrador isométrico",
        )

        assert scene.validation_status is ValidationStatus.VALID
        assert scene.validation_errors == []
        file_path = workspace.resolve_project_path(project.id, scene.html_path)
        assert file_path.is_file()
        content = file_path.read_text(encoding="utf-8")
        assert "60 FPS · ISOKIT" in content
