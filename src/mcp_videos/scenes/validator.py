"""Deterministic virtual-clock harness injection and Playwright scene validation."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from mcp_videos.core.models import Project, Scene, ValidationStatus
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.repositories import SceneRepository

HARNESS_MARKER = "data-mcp-videos-harness"
HARNESS_GLOBAL = "__MCP_VIDEOS_HARNESS__"
HARNESS_PATH = Path(__file__).with_name("harness.js")
CHROMIUM_ENV_VARS = (
    "MCP_VIDEOS_CHROMIUM_PATH",
    "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH",
)
CHROMIUM_BINARIES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "microsoft-edge",
)
DEFAULT_VIEWPORT_WIDTH = 1080
DEFAULT_VIEWPORT_HEIGHT = 1920
DEFAULT_TIMEOUT_MS = 15_000.0
DEFAULT_LAUNCH_ARGS = ("--disable-dev-shm-usage", "--hide-scrollbars")
_HEAD_OPEN_RE = re.compile(r"<head\b[^>]*>", flags=re.IGNORECASE)
_HTML_OPEN_RE = re.compile(r"<html\b[^>]*>", flags=re.IGNORECASE)
_DOCTYPE_RE = re.compile(r"^\s*<!doctype\b[^>]*>", flags=re.IGNORECASE)
_CLOCK_RE = re.compile(r"__setFrameTime\s*(?:=|:)", flags=re.IGNORECASE)
_HARNESS_CACHE: str | None = None


class SceneValidationError(RuntimeError):
    """Raised when the validator cannot prepare or run a scene."""


@dataclass(slots=True)
class ValidationResult:
    """Structured outcome of a headless DOM/JS validation pass."""

    duration_ms: int = 0
    checked_times_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def status(self) -> ValidationStatus:
        return ValidationStatus.VALID if self.valid else ValidationStatus.INVALID

    @property
    def validation_status(self) -> ValidationStatus:
        return self.status

    @property
    def console_errors(self) -> list[str]:
        return list(self.errors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "validation_status": self.status.value,
            "valid": self.valid,
            "duration_ms": self.duration_ms,
            "checked_times_ms": list(self.checked_times_ms),
            "validation_errors": list(self.errors),
        }


def load_harness_script() -> str:
    """Read the JavaScript harness bundled inside the scenes package."""

    global _HARNESS_CACHE
    if _HARNESS_CACHE is None:
        try:
            _HARNESS_CACHE = HARNESS_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise SceneValidationError(f"No se pudo leer harness.js: {exc}") from exc
    return _HARNESS_CACHE


def _is_instrumented(html: str) -> bool:
    lower = html.lower()
    return HARNESS_MARKER in lower or HARNESS_GLOBAL.lower() in lower


def html_declares_clock(html: str) -> bool:
    """Whether the document already assigns its own ``__setFrameTime``."""

    return _CLOCK_RE.search(html) is not None


def needs_harness(html: str) -> bool:
    """Return ``True`` when neither the harness nor a clock contract is present."""

    return not _is_instrumented(html) and not html_declares_clock(html)


def _harness_script_tag(harness_js: str | None = None) -> str:
    script = load_harness_script() if harness_js is None else harness_js
    safe_script = re.sub(r"</script", r"<\\/script", script, flags=re.IGNORECASE)
    return f'<script {HARNESS_MARKER}="1">\n{safe_script}\n</script>'


def inject_harness(
    html: str,
    harness_js: str | None = None,
    *,
    force: bool = False,
) -> str:
    """Inject ``harness.js`` into ``<head>`` without touching the rest of the document."""

    if not isinstance(html, str):
        raise TypeError("html debe ser una cadena de texto")

    if _is_instrumented(html):
        return html
    if not force and html_declares_clock(html):
        return html

    tag = _harness_script_tag(harness_js)
    head_match = _HEAD_OPEN_RE.search(html)
    if head_match is not None:
        return html[: head_match.end()] + "\n" + tag + html[head_match.end() :]

    html_match = _HTML_OPEN_RE.search(html)
    if html_match is not None:
        return html[: html_match.end()] + f"\n<head>{tag}</head>" + html[html_match.end() :]

    doctype_match = _DOCTYPE_RE.match(html)
    if doctype_match is not None:
        return html[: doctype_match.end()] + f"\n<head>{tag}</head>" + html[doctype_match.end() :]

    return f"{tag}\n{html}"


def prepare_scene_html(html: str, harness_js: str | None = None) -> str:
    """Alias used by generators and orchestrators to prepare a scene document."""

    return inject_harness(html, harness_js)


def inject_harness_into_html(
    html: str,
    harness_js: str | None = None,
    *,
    force: bool = False,
) -> str:
    """Explicit alias for harness injection."""

    return inject_harness(html, harness_js, force=force)


def find_chromium_executable() -> str | None:
    """Resolve an explicit or system Chromium binary when Playwright's bundle is absent."""

    for variable in CHROMIUM_ENV_VARS:
        value = os.environ.get(variable)
        if value:
            candidate = Path(value).expanduser()
            if candidate.is_file():
                return str(candidate)
    for binary in CHROMIUM_BINARIES:
        resolved = shutil.which(binary)
        if resolved:
            return resolved
    return None


def _normalize_message(error: object) -> str:
    message = str(error).strip()
    return message or error.__class__.__name__


def _deduplicate(errors: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for error in errors:
        if error not in seen:
            seen.add(error)
            unique.append(error)
    return unique


class SceneValidator:
    """Validate scene HTML inside headless Chromium and persist the quality gate."""

    def __init__(
        self,
        repository: SceneRepository | None = None,
        workspace: WorkspaceManager | None = None,
        *,
        width: int = DEFAULT_VIEWPORT_WIDTH,
        height: int = DEFAULT_VIEWPORT_HEIGHT,
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
        headless: bool = True,
        executable_path: str | Path | None = None,
        launch_args: Sequence[str] | None = None,
    ) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width debe ser un entero > 0")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("height debe ser un entero > 0")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms debe ser > 0")
        if executable_path is not None:
            executable_path = Path(executable_path).expanduser()

        self.repository = repository
        self.workspace = workspace
        self.width = width
        self.height = height
        self.timeout_ms = float(timeout_ms)
        self.headless = headless
        self.executable_path = executable_path
        self.launch_args = tuple(DEFAULT_LAUNCH_ARGS if launch_args is None else launch_args)

    def _launch_browser(self, playwright: Any) -> Any:
        kwargs: dict[str, Any] = {
            "headless": self.headless,
            "args": list(self.launch_args),
        }
        if self.executable_path is not None:
            return playwright.chromium.launch(
                executable_path=str(self.executable_path),
                **kwargs,
            )
        try:
            return playwright.chromium.launch(**kwargs)
        except PlaywrightError as bundled_error:
            fallback = find_chromium_executable()
            if fallback is None:
                raise SceneValidationError(
                    "No hay Chromium disponible para Playwright ni en el sistema"
                ) from bundled_error
            return playwright.chromium.launch(executable_path=fallback, **kwargs)

    def validate_html(
        self,
        html: str,
        *,
        duration_ms: int,
        html_path: Path | str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> ValidationResult:
        """Instrument ``html`` and validate it at ``t=0``, ``t=dur/2`` and ``t=dur``."""

        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
            raise ValueError("duration_ms debe ser un entero")
        if duration_ms <= 0:
            raise ValueError("duration_ms debe ser > 0")

        prepared = inject_harness(html)
        viewport_width = self.width if width is None else width
        viewport_height = self.height if height is None else height

        if html_path is None:
            with tempfile.TemporaryDirectory(prefix="mcp-videos-scene-") as temp_dir:
                target = Path(temp_dir) / "scene.html"
                target.write_text(prepared, encoding="utf-8")
                return self._validate_file(
                    target,
                    duration_ms=duration_ms,
                    width=viewport_width,
                    height=viewport_height,
                )

        target = Path(html_path).expanduser()
        if not target.is_absolute():
            target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(prepared, encoding="utf-8")
        return self._validate_file(
            target,
            duration_ms=duration_ms,
            width=viewport_width,
            height=viewport_height,
        )

    def validate_file(
        self,
        html_path: Path | str,
        *,
        duration_ms: int,
        width: int | None = None,
        height: int | None = None,
    ) -> ValidationResult:
        """Read, instrument and validate an existing scene file."""

        target = Path(html_path).expanduser()
        if not target.is_file():
            raise SceneValidationError(f"No existe la escena: {target}")
        html = target.read_text(encoding="utf-8")
        return self.validate_html(
            html,
            duration_ms=duration_ms,
            html_path=target,
            width=width,
            height=height,
        )

    def _validate_file(
        self,
        scene_path: Path,
        *,
        duration_ms: int,
        width: int,
        height: int,
    ) -> ValidationResult:
        times = [0.0, duration_ms / 2.0, float(duration_ms)]
        result = ValidationResult(duration_ms=duration_ms, checked_times_ms=list(times))

        with sync_playwright() as playwright:
            browser = self._launch_browser(playwright)
            try:
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                )
                page = context.new_page()
                page.set_default_timeout(self.timeout_ms)

                page.on("pageerror", self._make_pageerror_handler(result))
                page.on("console", self._make_console_handler(result))
                page.on("requestfailed", self._make_requestfailed_handler(result))
                page.on("response", self._make_response_handler(result))

                try:
                    page.goto(
                        scene_path.resolve().as_uri(),
                        wait_until="load",
                        timeout=self.timeout_ms,
                    )
                except PlaywrightError as exc:
                    result.errors.append(f"navigation: {_normalize_message(exc)}")
                else:
                    self._inspect_and_seek(page, times, result)

                with contextlib.suppress(PlaywrightError):
                    context.close()
            finally:
                browser.close()

        result.errors = _deduplicate(result.errors)
        return result

    @staticmethod
    def _make_pageerror_handler(result: ValidationResult) -> Any:
        def handle_pageerror(error: object) -> None:
            result.errors.append(f"pageerror: {_normalize_message(error)}")

        return handle_pageerror

    @staticmethod
    def _make_console_handler(result: ValidationResult) -> Any:
        def handle_console(message: Any) -> None:
            try:
                if message.type == "error":
                    result.errors.append(f"console.error: {message.text}")
            except Exception:
                return

        return handle_console

    @staticmethod
    def _make_requestfailed_handler(result: ValidationResult) -> Any:
        def handle_requestfailed(request: Any) -> None:
            failure = getattr(request, "failure", None)
            reason = failure if isinstance(failure, str) and failure else "fallo de carga"
            result.errors.append(f"requestfailed: {request.url} ({reason})")

        return handle_requestfailed

    @staticmethod
    def _make_response_handler(result: ValidationResult) -> Any:
        def handle_response(response: Any) -> None:
            try:
                status = int(response.status)
                if status >= 400:
                    result.errors.append(f"http {status}: {response.url}")
            except Exception:
                return

        return handle_response

    def _inspect_and_seek(
        self,
        page: Any,
        times: Sequence[float],
        result: ValidationResult,
    ) -> None:
        try:
            has_clock = bool(
                page.evaluate("typeof window.__setFrameTime === 'function'")
            )
        except PlaywrightError as exc:
            result.errors.append(f"clock: {_normalize_message(exc)}")
            return

        if not has_clock:
            result.errors.append("window.__setFrameTime no está definido")
            return

        try:
            has_content = bool(
                page.evaluate(
                    """
                    () => Boolean(
                      document.body &&
                      (document.body.childElementCount > 0 ||
                       document.body.textContent.trim().length > 0)
                    )
                    """
                )
            )
        except PlaywrightError as exc:
            result.errors.append(f"dom: {_normalize_message(exc)}")
            has_content = False

        if not has_content:
            result.errors.append("el DOM de la escena está vacío")

        for time_ms in times:
            try:
                page.evaluate(
                    """
                    (timeMs) => {
                      const setter = window.__setFrameTime;
                      if (typeof setter !== 'function') {
                        throw new Error('window.__setFrameTime no es una función');
                      }
                      return setter(timeMs);
                    }
                    """,
                    time_ms,
                )
            except PlaywrightError as exc:
                result.errors.append(
                    f"seek t={time_ms:g}ms: {_normalize_message(exc)}"
                )

    def validate_scene(
        self,
        scene: Scene,
        *,
        project: Project | None = None,
        workspace: WorkspaceManager | None = None,
        html_content: str | None = None,
    ) -> Scene:
        """Validate an already persisted scene and update its repository gate."""

        resolved_workspace = workspace or self.workspace
        if resolved_workspace is None:
            raise SceneValidationError("Se requiere un WorkspaceManager para validar escenas")

        candidate = Path(scene.html_path)
        if candidate.is_absolute():
            scene_path = candidate.resolve()
        else:
            scene_path = resolved_workspace.resolve_project_path(scene.project_id, candidate)

        if html_content is None:
            if not scene_path.is_file():
                raise SceneValidationError(f"No existe la escena: {scene_path}")
            html_content = scene_path.read_text(encoding="utf-8")

        prepared = inject_harness(html_content)
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        scene_path.write_text(prepared, encoding="utf-8")

        width = project.width if project is not None else self.width
        height = project.height if project is not None else self.height
        result = self.validate_html(
            prepared,
            duration_ms=scene.end_ms - scene.start_ms,
            html_path=scene_path,
            width=width,
            height=height,
        )

        if self.repository is None:
            return scene.model_copy(
                update={
                    "validation_status": result.status,
                    "validation_errors": list(result.errors),
                }
            )
        return self.repository.update_validation(
            scene.project_id,
            scene.scene_index,
            result.status,
            result.errors,
        )

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
        """Write a scene into its workspace, validate it and persist the result."""

        if self.repository is None:
            raise SceneValidationError("Se requiere un SceneRepository para persistir escenas")

        resolved_workspace = workspace or self.workspace
        if resolved_workspace is None:
            raise SceneValidationError("Se requiere un WorkspaceManager para persistir escenas")

        if isinstance(start_ms, bool) or not isinstance(start_ms, int) or start_ms < 0:
            raise ValueError("start_ms debe ser un entero >= 0")
        if isinstance(end_ms, bool) or not isinstance(end_ms, int) or end_ms <= start_ms:
            raise ValueError("end_ms debe ser mayor que start_ms")

        resolved_workspace.ensure_project_dirs(project_id)
        prepared = inject_harness(html_content)
        scene_path = resolved_workspace.write_scene_html(project_id, scene_index, prepared)
        relative_path = resolved_workspace.relative_project_path(project_id, scene_path)

        scene = Scene(
            project_id=project_id,
            scene_index=scene_index,
            title=title or f"Escena {scene_index:02d}",
            start_ms=start_ms,
            end_ms=end_ms,
            html_path=relative_path,
            prompt_used=prompt_used,
            validation_status=ValidationStatus.PENDING,
        )
        self.repository.upsert(scene)

        try:
            result = self.validate_html(
                prepared,
                duration_ms=end_ms - start_ms,
                html_path=scene_path,
                width=width,
                height=height,
            )
        except Exception as exc:
            result = ValidationResult(
                duration_ms=end_ms - start_ms,
                checked_times_ms=[0.0, (end_ms - start_ms) / 2.0, float(end_ms - start_ms)],
                errors=[f"validation exception: {_normalize_message(exc)}"],
            )

        return self.repository.update_validation(
            project_id,
            scene_index,
            result.status,
            result.errors,
        )


__all__ = [
    "CHROMIUM_BINARIES",
    "CHROMIUM_ENV_VARS",
    "DEFAULT_TIMEOUT_MS",
    "DEFAULT_VIEWPORT_HEIGHT",
    "DEFAULT_VIEWPORT_WIDTH",
    "HARNESS_GLOBAL",
    "HARNESS_MARKER",
    "HARNESS_PATH",
    "SceneValidationError",
    "SceneValidator",
    "ValidationResult",
    "find_chromium_executable",
    "html_declares_clock",
    "inject_harness",
    "inject_harness_into_html",
    "load_harness_script",
    "needs_harness",
    "prepare_scene_html",
]
