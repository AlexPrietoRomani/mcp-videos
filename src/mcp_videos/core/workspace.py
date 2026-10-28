"""Safe, idempotent filesystem management for project workspaces."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from mcp_videos.core.models import SLUG_RE

PROJECT_SUBDIRS: tuple[str, ...] = ("audio", "scenes", "output")
DIALOGUE_MASTER_FILENAME = "dialogue_master.wav"
FINAL_VIDEO_FILENAME = "final.mp4"


class WorkspaceError(ValueError):
    """Base error for invalid workspace operations."""


class InvalidSlugError(WorkspaceError):
    """Raised when a project or voice identifier is not a canonical slug."""


class InvalidArtifactIndexError(WorkspaceError):
    """Raised when a line or scene index is not a non-negative integer."""


class PathTraversalError(WorkspaceError):
    """Raised when a path would escape its allowed workspace root."""


class WorkspaceManager:
    """Resolve and create project artifacts without leaving their roots."""

    def __init__(
        self,
        workspaces_root: Path | str,
        voices_root: Path | str,
    ) -> None:
        self.workspaces_root = Path(workspaces_root).expanduser().resolve()
        self.voices_root = Path(voices_root).expanduser().resolve()

    @staticmethod
    def _validate_slug(value: str, kind: str) -> str:
        if not isinstance(value, str) or SLUG_RE.fullmatch(value) is None:
            raise InvalidSlugError(f"{kind} inválido: {value!r}")
        return value

    @staticmethod
    def _validate_index(value: int, kind: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidArtifactIndexError(f"{kind} inválido: {value!r}")
        return value

    def _resolve_within(self, root: Path, relative_path: Path | str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise PathTraversalError("las rutas absolutas no están permitidas")
        if not candidate.parts or any(part in {"", ".."} for part in candidate.parts):
            raise PathTraversalError(f"ruta relativa no permitida: {relative_path!r}")
        if any("\x00" in part for part in candidate.parts):
            raise PathTraversalError("la ruta contiene un byte nulo")

        target = (root / candidate).resolve()
        if not target.is_relative_to(root):
            raise PathTraversalError(f"acceso fuera de {root} denegado: {relative_path!r}")
        return target

    def project_dir(self, project_id: str) -> Path:
        """Return the canonical project directory without creating it."""

        slug = self._validate_slug(project_id, "project_id")
        target = (self.workspaces_root / slug).resolve()
        if not target.is_relative_to(self.workspaces_root):
            raise PathTraversalError("acceso fuera de workspaces_root denegado")
        return target

    def audio_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "audio"

    def scenes_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "scenes"

    def output_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "output"

    def ensure_project_dirs(self, project_id: str) -> Path:
        """Create the complete idempotent project tree and return its root."""

        project = self.project_dir(project_id)
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
        self.voices_root.mkdir(parents=True, exist_ok=True)
        project.mkdir(parents=True, exist_ok=True)
        for subdir in PROJECT_SUBDIRS:
            (project / subdir).mkdir(parents=True, exist_ok=True)
        return project

    def ensure_voices_dir(self) -> Path:
        """Create the canonical voice samples directory if needed."""

        self.voices_root.mkdir(parents=True, exist_ok=True)
        return self.voices_root

    def scene_html_path(self, project_id: str, scene_index: int) -> Path:
        index = self._validate_index(scene_index, "scene_index")
        self.project_dir(project_id)
        return self.scenes_dir(project_id) / f"scene_{index:02d}.html"

    def line_audio_path(self, project_id: str, line_index: int) -> Path:
        index = self._validate_index(line_index, "line_index")
        self.project_dir(project_id)
        return self.audio_dir(project_id) / f"line_{index:02d}.wav"

    def dialogue_master_path(self, project_id: str) -> Path:
        self.project_dir(project_id)
        return self.audio_dir(project_id) / DIALOGUE_MASTER_FILENAME

    def final_mp4_path(self, project_id: str) -> Path:
        self.project_dir(project_id)
        return self.output_dir(project_id) / FINAL_VIDEO_FILENAME

    def voice_sample_path(self, voice_id: str) -> Path:
        slug = self._validate_slug(voice_id, "voice_id")
        target = (self.voices_root / f"{slug}.wav").resolve()
        if not target.is_relative_to(self.voices_root):
            raise PathTraversalError("acceso fuera de voices_root denegado")
        return target

    def resolve_project_path(
        self,
        project_id: str,
        relative_path: Path | str,
    ) -> Path:
        """Resolve an artifact path relative to the project root safely."""

        project = self.project_dir(project_id)
        return self._resolve_within(project, relative_path)

    def relative_project_path(
        self,
        project_id: str,
        path: Path | str,
    ) -> str:
        """Return a POSIX path suitable for storing in SQLite."""

        project = self.project_dir(project_id)
        candidate = Path(path)
        target = (
            self.resolve_project_path(project_id, candidate)
            if not candidate.is_absolute()
            else candidate.expanduser().resolve()
        )
        if target == project or not target.is_relative_to(project):
            raise PathTraversalError(f"la ruta no pertenece al proyecto {project_id!r}: {path!r}")
        return target.relative_to(project).as_posix()

    def write_project_file(
        self,
        project_id: str,
        relative_path: Path | str,
        content: bytes | str,
    ) -> Path:
        """Atomically write a project artifact and return its absolute path."""

        target = self.resolve_project_path(project_id, relative_path)
        self._atomic_write(target, content)
        return target

    def write_scene_html(
        self,
        project_id: str,
        scene_index: int,
        html_content: str,
    ) -> Path:
        """Atomically write one scene HTML file into the project workspace."""

        self.ensure_project_dirs(project_id)
        target = self.scene_html_path(project_id, scene_index)
        self._atomic_write(target, html_content)
        return target

    @staticmethod
    def _atomic_write(target: Path, content: bytes | str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode("utf-8") if isinstance(content, str) else content

        file_descriptor, temp_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name)
            raise


__all__ = [
    "DIALOGUE_MASTER_FILENAME",
    "FINAL_VIDEO_FILENAME",
    "PROJECT_SUBDIRS",
    "InvalidArtifactIndexError",
    "InvalidSlugError",
    "PathTraversalError",
    "WorkspaceError",
    "WorkspaceManager",
]
