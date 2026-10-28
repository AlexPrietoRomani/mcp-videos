"""Deterministic frame-by-frame rendering from validated HTML scenes to MP4."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from playwright.sync_api import Browser, Playwright, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from mcp_videos.config import AppSettings
from mcp_videos.core.models import (
    Project,
    ProjectStatus,
    RenderJob,
    RenderStatus,
    Scene,
    ValidationStatus,
    WordTimestamp,
    frame_to_time_ms,
    total_frames_for_duration,
    validate_scenes_timeline_coverage,
)
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectRepository,
    RenderJobRepository,
    SceneRepository,
)
from mcp_videos.render.subtitles import generate_ass_subtitles

SUBTITLES_FILENAME = "subtitles.ass"
DEFAULT_FALLBACK_CHROMIUM_PATH = Path("/usr/bin/chromium")
FALLBACK_CHROMIUM_BINARIES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
)
DEFAULT_AUDIO_BITRATE = "192k"
DEFAULT_PAGE_LOAD_TIMEOUT_MS = 30_000
DEFAULT_FFMPEG_TIMEOUT_S = 600.0
ProgressCallback = Callable[[int], None]


class RenderEngineError(RuntimeError):
    """Raised when the deterministic render pipeline cannot complete."""


class SceneNotValidatedError(RenderEngineError):
    """Raised when one or more scenes did not pass the validation gate."""

    def __init__(self, message: str, *, scene_indexes: Sequence[int] = ()) -> None:
        super().__init__(message)
        self.scene_indexes = tuple(scene_indexes)


@dataclass(frozen=True)
class RenderResult:
    """Summary of a completed deterministic render."""

    job: RenderJob
    output_path: Path
    subtitles_path: Path | None
    total_frames: int
    rendered_frames: int


def _escape_filter_path(path: Path) -> str:
    value = str(path).replace("\\", "\\\\")
    for character in (":", "'", ",", "[", "]"):
        value = value.replace(character, f"\\{character}")
    return value


class DeterministicRenderEngine:
    """Execute Flow 6: capture validated scenes at a constant frame rate and mux MP4."""

    def __init__(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        settings: AppSettings | None = None,
        *,
        ffmpeg_binary: str | None = None,
        chromium_executable_path: str | Path | None = None,
        fallback_chromium_path: str | Path | None = DEFAULT_FALLBACK_CHROMIUM_PATH,
        screenshot_format: str = "png",
        job_repository: RenderJobRepository | None = None,
    ) -> None:
        if screenshot_format not in {"png", "jpeg"}:
            raise ValueError("screenshot_format debe ser 'png' o 'jpeg'")
        self.db = db
        self.workspace = workspace
        self.settings = settings or AppSettings()
        self.ffmpeg_binary = ffmpeg_binary or os.environ.get("FFMPEG_BINARY", "ffmpeg")
        self.chromium_executable_path = (
            str(chromium_executable_path) if chromium_executable_path is not None else None
        )
        self.fallback_chromium_path = (
            str(fallback_chromium_path) if fallback_chromium_path is not None else None
        )
        self.screenshot_format = screenshot_format
        self.projects = ProjectRepository(db)
        self.scenes = SceneRepository(db)
        self.dialogues = DialogueRepository(db)
        self.jobs = job_repository or RenderJobRepository(db)

    def subtitles_path(self, project_id: str) -> Path:
        """Return the canonical ASS subtitle path inside the project workspace."""

        self.workspace.project_dir(project_id)
        return self.workspace.output_dir(project_id) / SUBTITLES_FILENAME

    def generate_subtitles(
        self,
        project: Project | str,
        words: Sequence[WordTimestamp] | None = None,
    ) -> Path:
        """Generate the project ``subtitles.ass`` from persisted word timestamps."""

        resolved = self._resolve_project(project)
        if resolved.total_duration_ms <= 0:
            raise RenderEngineError("el proyecto no tiene una duración de audio válida")
        if words is None:
            words = self.dialogues.word_timestamps_in_window(
                0,
                resolved.total_duration_ms,
                project_id=resolved.id,
            )
        return generate_ass_subtitles(
            list(words),
            resolved.width,
            resolved.height,
            self.subtitles_path(resolved.id),
        )

    def validate_scene_preconditions(
        self,
        project: Project,
        scenes: Sequence[Scene],
    ) -> tuple[Scene, ...]:
        """Ensure every scene is valid and continuously covers the project timeline."""

        ordered = tuple(sorted(scenes, key=lambda scene: scene.scene_index))
        invalid = [
            scene.scene_index
            for scene in ordered
            if scene.validation_status != ValidationStatus.VALID
        ]
        if invalid:
            raise SceneNotValidatedError(
                "las escenas deben estar validadas antes de renderizar: "
                f"índices no válidos {invalid}",
                scene_indexes=invalid,
            )

        for scene in ordered:
            if scene.project_id != project.id:
                raise RenderEngineError(
                    f"la escena {scene.scene_index} pertenece a {scene.project_id!r}, "
                    f"no a {project.id!r}"
                )

        validate_scenes_timeline_coverage(ordered, project.total_duration_ms)

        for scene in ordered:
            html_path = self._resolve_scene_html(project, scene)
            if not html_path.is_file():
                raise RenderEngineError(
                    f"no existe el HTML de la escena {scene.scene_index}: {html_path}"
                )
        return ordered

    def build_ffmpeg_command(
        self,
        project: Project,
        master_audio_path: Path | str,
        output_path: Path | str,
        *,
        burn_subtitles: bool = True,
        subtitles_path: Path | str | None = None,
        total_frames: int | None = None,
    ) -> list[str]:
        """Build the FFmpeg invocation used to mux captured frames and master audio."""

        if total_frames is None:
            total_frames = total_frames_for_duration(project.total_duration_ms, project.fps)
        if total_frames <= 0:
            raise ValueError("total_frames debe ser mayor que 0")

        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "image2pipe",
            "-framerate",
            str(project.fps),
            "-i",
            "-",
            "-i",
            str(master_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(self.settings.ffmpeg_crf),
            "-pix_fmt",
            "yuv420p",
        ]

        if burn_subtitles:
            if subtitles_path is None:
                raise ValueError("subtitles_path es obligatorio cuando burn_subtitles=True")
            command.extend(["-vf", f"ass={_escape_filter_path(Path(subtitles_path))}"])

        command.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                DEFAULT_AUDIO_BITRATE,
                "-frames:v",
                str(total_frames),
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        return command

    def render_project(
        self,
        project: Project | str,
        scenes: Sequence[Scene] | None = None,
        master_audio_path: Path | str | None = None,
        *,
        burn_subtitles: bool | None = None,
        words: Sequence[WordTimestamp] | None = None,
    ) -> RenderResult:
        """Render every scene into ``output/final.mp4`` and persist the render job."""

        resolved = self._resolve_project(project)
        project_scenes = self.validate_scene_preconditions(
            resolved,
            scenes if scenes is not None else self.scenes.list_by_project(resolved.id),
        )
        self._validate_frame_dimensions(resolved)
        total_frames = total_frames_for_duration(resolved.total_duration_ms, resolved.fps)
        master_path = self._resolve_master_audio(resolved, master_audio_path)
        should_burn_subtitles = True if burn_subtitles is None else bool(burn_subtitles)

        job = RenderJob(
            project_id=resolved.id,
            status=RenderStatus.QUEUED,
            total_frames=total_frames,
            burn_subtitles=should_burn_subtitles,
        )
        job = self.jobs.create(job)
        output_path = self.workspace.final_mp4_path(resolved.id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_name(f".{output_path.name}.{job.id}.tmp.mp4")
        subtitles_path: Path | None = None
        process: subprocess.Popen[bytes] | None = None

        try:
            self.jobs.update_progress(job.id, 0, status=RenderStatus.RUNNING)

            if should_burn_subtitles:
                subtitles_path = self.generate_subtitles(resolved, words)

            command = self.build_ffmpeg_command(
                resolved,
                master_path,
                temporary_output,
                burn_subtitles=should_burn_subtitles,
                subtitles_path=subtitles_path,
                total_frames=total_frames,
            )
            process = self._start_ffmpeg(command)
            assert process.stdin is not None

            def report_progress(rendered_frames: int) -> None:
                self.jobs.update_progress(job.id, rendered_frames)

            self._capture_frames(
                resolved,
                project_scenes,
                process.stdin,
                total_frames,
                report_progress,
            )
            process.stdin.close()
            stderr = self._read_process_stderr(process)
            returncode = process.wait(timeout=DEFAULT_FFMPEG_TIMEOUT_S)
            if returncode != 0:
                raise RenderEngineError(
                    f"FFmpeg falló con código {returncode}: {stderr.strip()[-1200:]}"
                )

            os.replace(temporary_output, output_path)
            relative_output = self.workspace.relative_project_path(
                resolved.id,
                output_path,
            )
            completed_job = self.jobs.complete(job.id, relative_output)
            self.projects.update_status(resolved.id, ProjectStatus.RENDERED)
            return RenderResult(
                job=completed_job,
                output_path=output_path,
                subtitles_path=subtitles_path,
                total_frames=total_frames,
                rendered_frames=total_frames,
            )
        except Exception as exc:
            if process is not None:
                self._abort_process(process)
            with contextlib.suppress(FileNotFoundError):
                temporary_output.unlink()
            with contextlib.suppress(Exception):
                self.jobs.fail(job.id, str(exc))
            with contextlib.suppress(Exception):
                self.projects.update_status(resolved.id, ProjectStatus.FAILED)
            if isinstance(exc, (RenderEngineError, ValueError)):
                raise
            raise RenderEngineError(f"falló el render determinista: {exc}") from exc

    def render(
        self,
        project: Project,
        scenes: Sequence[Scene],
        master_audio_path: Path | str | None = None,
        *,
        burn_subtitles: bool | None = None,
        words: Sequence[WordTimestamp] | None = None,
    ) -> RenderResult:
        """Alias for :meth:`render_project` accepting an explicit project and scenes."""

        return self.render_project(
            project,
            scenes,
            master_audio_path,
            burn_subtitles=burn_subtitles,
            words=words,
        )

    def _resolve_project(self, project: Project | str) -> Project:
        if isinstance(project, Project):
            return project
        return self.projects.get_or_raise(str(project))

    def _resolve_scene_html(self, project: Project, scene: Scene) -> Path:
        candidate = Path(scene.html_path)
        if candidate.is_absolute():
            return candidate
        return self.workspace.resolve_project_path(project.id, candidate)

    def _resolve_master_audio(
        self,
        project: Project,
        master_audio_path: Path | str | None,
    ) -> Path:
        if master_audio_path is None:
            if project.master_audio_path:
                candidate = Path(project.master_audio_path)
                path = (
                    candidate
                    if candidate.is_absolute()
                    else self.workspace.resolve_project_path(project.id, candidate)
                )
            else:
                path = self.workspace.dialogue_master_path(project.id)
        else:
            candidate = Path(master_audio_path)
            path = (
                candidate
                if candidate.is_absolute()
                else self.workspace.resolve_project_path(project.id, candidate)
            )
        if not path.is_file():
            raise RenderEngineError(f"no existe el audio maestro: {path}")
        return path

    def _validate_frame_dimensions(self, project: Project) -> None:
        if project.width % 2 or project.height % 2:
            raise RenderEngineError(
                "width y height deben ser pares para codificar con yuv420p: "
                f"{project.width}x{project.height}"
            )

    def _start_ffmpeg(self, command: Sequence[str]) -> subprocess.Popen[bytes]:
        try:
            return subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RenderEngineError(f"no se pudo iniciar FFmpeg: {exc}") from exc

    @staticmethod
    def _read_process_stderr(process: subprocess.Popen[bytes]) -> str:
        if process.stderr is None:
            return ""
        try:
            detail = process.stderr.read().decode("utf-8", errors="replace")
        except (OSError, ValueError):
            detail = ""
        with contextlib.suppress(OSError, ValueError):
            process.stderr.close()
        return detail

    @staticmethod
    def _abort_process(process: subprocess.Popen[bytes]) -> None:
        with contextlib.suppress(OSError, ValueError):
            if process.stdin is not None:
                process.stdin.close()
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                process.wait(timeout=10.0)
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                process.wait(timeout=10.0)
        if process.stderr is not None:
            with contextlib.suppress(OSError, ValueError):
                process.stderr.close()

    def _launch_browser(self, playwright: Playwright) -> Browser:
        launch_options: dict[str, object] = {
            "headless": True,
            "args": ["--hide-scrollbars", "--mute-audio"],
        }
        if self.chromium_executable_path is not None:
            launch_options["executable_path"] = self.chromium_executable_path
            try:
                return playwright.chromium.launch(**launch_options)
            except PlaywrightError as exc:
                raise RenderEngineError(
                    f"no se pudo abrir Chromium en {self.chromium_executable_path}: {exc}"
                ) from exc

        try:
            return playwright.chromium.launch(**launch_options)
        except PlaywrightError as bundled_error:
            fallback = self._fallback_chromium()
            if fallback is None:
                raise RenderEngineError(
                    "no se pudo abrir Chromium y no hay un binario de respaldo disponible: "
                    f"{bundled_error}"
                ) from bundled_error
            try:
                return playwright.chromium.launch(
                    **launch_options,
                    executable_path=str(fallback),
                )
            except PlaywrightError as fallback_error:
                raise RenderEngineError(
                    "no se pudo abrir Chromium ni con el binario empaquetado ni con "
                    f"{fallback}: {fallback_error}"
                ) from fallback_error

    def _fallback_chromium(self) -> str | None:
        if self.fallback_chromium_path is not None:
            candidate = Path(self.fallback_chromium_path)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        for binary in FALLBACK_CHROMIUM_BINARIES:
            located = shutil.which(binary)
            if located is not None:
                return located
        return None

    def _capture_frames(
        self,
        project: Project,
        scenes: Sequence[Scene],
        stdin: BinaryIO,
        total_frames: int,
        progress: ProgressCallback,
    ) -> None:
        if not scenes:
            raise RenderEngineError("no hay escenas para capturar")

        with sync_playwright() as playwright:
            browser = self._launch_browser(playwright)
            try:
                context = browser.new_context(
                    viewport={"width": project.width, "height": project.height},
                    device_scale_factor=1,
                )
                page = context.new_page()
                page.set_default_timeout(DEFAULT_PAGE_LOAD_TIMEOUT_MS)
                loaded_scene_index: int | None = None
                scene_position = 0

                for frame_index in range(total_frames):
                    time_ms = frame_to_time_ms(frame_index, project.fps)
                    while (
                        scene_position + 1 < len(scenes)
                        and time_ms >= scenes[scene_position].end_ms
                    ):
                        scene_position += 1
                    scene = scenes[scene_position]
                    if not scene.start_ms <= time_ms < scene.end_ms:
                        raise RenderEngineError(
                            f"el frame {frame_index} ({time_ms} ms) queda fuera de la "
                            f"escena {scene.scene_index}"
                        )

                    if loaded_scene_index != scene.scene_index:
                        html_path = self._resolve_scene_html(project, scene)
                        try:
                            page.goto(
                                html_path.as_uri(),
                                wait_until="load",
                                timeout=DEFAULT_PAGE_LOAD_TIMEOUT_MS,
                            )
                            page.wait_for_function(
                                "typeof window.__setFrameTime === 'function'",
                                timeout=DEFAULT_PAGE_LOAD_TIMEOUT_MS,
                            )
                        except PlaywrightError as exc:
                            raise RenderEngineError(
                                f"no se pudo cargar la escena {scene.scene_index}: {exc}"
                            ) from exc
                        loaded_scene_index = scene.scene_index

                    local_time_ms = max(0.0, time_ms - scene.start_ms)
                    try:
                        page.evaluate(
                            "(timeMs) => window.__setFrameTime(timeMs)",
                            local_time_ms,
                        )
                        image = page.screenshot(type=self.screenshot_format)
                    except PlaywrightError as exc:
                        raise RenderEngineError(
                            f"falló la captura de la escena {scene.scene_index} "
                            f"en el frame {frame_index}: {exc}"
                        ) from exc

                    try:
                        stdin.write(image)
                        stdin.flush()
                    except (BrokenPipeError, OSError) as exc:
                        raise RenderEngineError(
                            "FFmpeg cerró su stdin antes de terminar la captura"
                        ) from exc

                    progress(frame_index + 1)
            finally:
                browser.close()


__all__ = [
    "DEFAULT_AUDIO_BITRATE",
    "DEFAULT_FALLBACK_CHROMIUM_PATH",
    "SUBTITLES_FILENAME",
    "DeterministicRenderEngine",
    "RenderEngineError",
    "RenderResult",
    "SceneNotValidatedError",
]
