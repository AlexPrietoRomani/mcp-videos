"""Unit tests for safe workspace path management."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_videos.core.workspace import (
    InvalidArtifactIndexError,
    InvalidSlugError,
    PathTraversalError,
    WorkspaceManager,
)


@pytest.fixture
def manager(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(
        workspaces_root=tmp_path / "workspaces",
        voices_root=tmp_path / "data" / "voices",
    )


class TestRootsAndDirectories:
    def test_roots_are_resolved(self, tmp_path: Path) -> None:
        manager = WorkspaceManager(
            workspaces_root=tmp_path / "workspaces",
            voices_root=tmp_path / "voices",
        )

        assert manager.workspaces_root.is_absolute()
        assert manager.voices_root.is_absolute()

    def test_ensure_project_dirs_creates_full_tree(
        self,
        manager: WorkspaceManager,
    ) -> None:
        project = manager.ensure_project_dirs("short-antena-st5")

        assert project == manager.workspaces_root / "short-antena-st5"
        assert (project / "audio").is_dir()
        assert (project / "scenes").is_dir()
        assert (project / "output").is_dir()

    def test_ensure_project_dirs_is_idempotent(
        self,
        manager: WorkspaceManager,
    ) -> None:
        first = manager.ensure_project_dirs("proyecto-uno")
        (first / "audio" / "marker.txt").write_text("ok", encoding="utf-8")

        second = manager.ensure_project_dirs("proyecto-uno")

        assert second == first
        assert (second / "audio" / "marker.txt").read_text(encoding="utf-8") == "ok"

    def test_ensure_voices_dir(self, manager: WorkspaceManager) -> None:
        voices = manager.ensure_voices_dir()

        assert voices == manager.voices_root
        assert voices.is_dir()

    def test_helper_directories(self, manager: WorkspaceManager) -> None:
        manager.ensure_project_dirs("proyecto-uno")

        assert manager.audio_dir("proyecto-uno").is_dir()
        assert manager.scenes_dir("proyecto-uno").is_dir()
        assert manager.output_dir("proyecto-uno").is_dir()

    @pytest.mark.parametrize(
        "slug",
        [
            "",
            "Mayusculas",
            "con_guion_bajo",
            "-inicio",
            "fin-",
            "doble--guion",
            "con espacio",
            "../escape",
            "con/punto",
            "..",
            ".",
        ],
    )
    def test_invalid_project_slugs_are_rejected(
        self,
        manager: WorkspaceManager,
        slug: str,
    ) -> None:
        with pytest.raises(InvalidSlugError):
            manager.project_dir(slug)

    def test_invalid_voice_slug_is_rejected(self, manager: WorkspaceManager) -> None:
        with pytest.raises(InvalidSlugError):
            manager.voice_sample_path("../../etc/passwd")


class TestCanonicalPaths:
    def test_scene_html_path_is_zero_padded(
        self,
        manager: WorkspaceManager,
    ) -> None:
        path = manager.scene_html_path("proyecto-uno", 0)

        assert path == manager.workspaces_root / "proyecto-uno" / "scenes" / "scene_00.html"
        assert manager.scene_html_path("proyecto-uno", 12).name == "scene_12.html"

    def test_line_audio_path_is_zero_padded(
        self,
        manager: WorkspaceManager,
    ) -> None:
        path = manager.line_audio_path("proyecto-uno", 0)

        assert path == manager.workspaces_root / "proyecto-uno" / "audio" / "line_00.wav"
        assert manager.line_audio_path("proyecto-uno", 12).name == "line_12.wav"

    def test_master_and_output_paths(self, manager: WorkspaceManager) -> None:
        assert manager.dialogue_master_path("proyecto-uno").name == "dialogue_master.wav"
        assert manager.final_mp4_path("proyecto-uno").name == "final.mp4"

    def test_voice_sample_path(self, manager: WorkspaceManager) -> None:
        assert manager.voice_sample_path("clon-alex") == manager.voices_root / "clon-alex.wav"

    @pytest.mark.parametrize("index", [-1, True])
    def test_invalid_scene_index_is_rejected(
        self,
        manager: WorkspaceManager,
        index: int,
    ) -> None:
        with pytest.raises(InvalidArtifactIndexError):
            manager.scene_html_path("proyecto-uno", index)

    @pytest.mark.parametrize("index", [-1, True])
    def test_invalid_line_index_is_rejected(
        self,
        manager: WorkspaceManager,
        index: int,
    ) -> None:
        with pytest.raises(InvalidArtifactIndexError):
            manager.line_audio_path("proyecto-uno", index)


class TestPathTraversalProtection:
    @pytest.mark.parametrize(
        "relative_path",
        [
            "../secreto.txt",
            "audio/../../secreto.txt",
            "scenes/../../../etc/passwd",
            "",
            ".",
        ],
    )
    def test_relative_traversal_is_rejected(
        self,
        manager: WorkspaceManager,
        relative_path: str,
    ) -> None:
        manager.ensure_project_dirs("proyecto-uno")

        with pytest.raises(PathTraversalError):
            manager.resolve_project_path("proyecto-uno", relative_path)

    @pytest.mark.parametrize(
        "absolute_path",
        ["/etc/passwd", "/tmp/secreto.txt"],
    )
    def test_absolute_paths_are_rejected(
        self,
        manager: WorkspaceManager,
        absolute_path: str,
    ) -> None:
        manager.ensure_project_dirs("proyecto-uno")

        with pytest.raises(PathTraversalError):
            manager.resolve_project_path("proyecto-uno", absolute_path)

    def test_null_byte_is_rejected(self, manager: WorkspaceManager) -> None:
        manager.ensure_project_dirs("proyecto-uno")

        with pytest.raises(PathTraversalError, match="nulo"):
            manager.resolve_project_path("proyecto-uno", "audio/\x00evil.wav")

    def test_symlink_escape_is_rejected(
        self,
        manager: WorkspaceManager,
        tmp_path: Path,
    ) -> None:
        project = manager.ensure_project_dirs("proyecto-uno")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secreto.txt").write_text("no leer", encoding="utf-8")
        (project / "link").symlink_to(outside, target_is_directory=True)

        with pytest.raises(PathTraversalError):
            manager.resolve_project_path("proyecto-uno", "link/secreto.txt")

    def test_relative_path_inside_project_is_allowed(
        self,
        manager: WorkspaceManager,
    ) -> None:
        manager.ensure_project_dirs("proyecto-uno")

        target = manager.resolve_project_path(
            "proyecto-uno",
            "scenes/scene_00.html",
        )

        assert target == manager.scenes_dir("proyecto-uno") / "scene_00.html"

    def test_voice_sample_symlink_escape_is_rejected(
        self,
        manager: WorkspaceManager,
        tmp_path: Path,
    ) -> None:
        manager.ensure_voices_dir()
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"audio")
        (manager.voices_root / "voz-mala.wav").symlink_to(outside)

        with pytest.raises(PathTraversalError):
            manager.voice_sample_path("voz-mala")


class TestAtomicWritesAndRelativePaths:
    def test_write_scene_html_creates_dirs_and_file(
        self,
        manager: WorkspaceManager,
    ) -> None:
        path = manager.write_scene_html(
            "short-antena-st5",
            0,
            "<html><body>Escena</body></html>",
        )

        assert path == manager.scene_html_path("short-antena-st5", 0)
        assert "Escena" in path.read_text(encoding="utf-8")
        assert not list(path.parent.glob(".*.tmp"))

    def test_write_scene_html_overwrites_atomically(
        self,
        manager: WorkspaceManager,
    ) -> None:
        manager.write_scene_html("proyecto-uno", 0, "primera")
        path = manager.write_scene_html("proyecto-uno", 0, "segunda")

        assert path.read_text(encoding="utf-8") == "segunda"
        assert not list(path.parent.glob(".*.tmp"))

    def test_write_project_file_accepts_bytes(
        self,
        manager: WorkspaceManager,
    ) -> None:
        path = manager.write_project_file(
            "proyecto-uno",
            "audio/line_00.wav",
            b"RIFF....WAVE",
        )

        assert path.read_bytes() == b"RIFF....WAVE"

    def test_write_project_file_rejects_traversal(
        self,
        manager: WorkspaceManager,
    ) -> None:
        with pytest.raises(PathTraversalError):
            manager.write_project_file("proyecto-uno", "../evil.txt", "x")

    def test_relative_project_path_is_posix(
        self,
        manager: WorkspaceManager,
    ) -> None:
        manager.ensure_project_dirs("proyecto-uno")
        path = manager.scene_html_path("proyecto-uno", 0)

        assert manager.relative_project_path("proyecto-uno", path) == ("scenes/scene_00.html")
        assert (
            manager.relative_project_path(
                "proyecto-uno",
                "audio/line_00.wav",
            )
            == "audio/line_00.wav"
        )

    def test_relative_project_path_rejects_outside_path(
        self,
        manager: WorkspaceManager,
        tmp_path: Path,
    ) -> None:
        manager.ensure_project_dirs("proyecto-uno")
        outside = tmp_path / "outside.txt"

        with pytest.raises(PathTraversalError):
            manager.relative_project_path("proyecto-uno", outside)

    def test_relative_project_path_rejects_project_root(
        self,
        manager: WorkspaceManager,
    ) -> None:
        project = manager.ensure_project_dirs("proyecto-uno")

        with pytest.raises(PathTraversalError):
            manager.relative_project_path("proyecto-uno", project)


class TestWorkspaceIsolation:
    def test_projects_do_not_share_artifacts(
        self,
        manager: WorkspaceManager,
    ) -> None:
        first = manager.ensure_project_dirs("proyecto-uno")
        second = manager.ensure_project_dirs("proyecto-dos")

        (first / "output" / "final.mp4").write_bytes(b"uno")

        assert not (second / "output" / "final.mp4").exists()
        assert first != second
