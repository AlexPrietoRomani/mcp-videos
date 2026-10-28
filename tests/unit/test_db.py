"""Unit tests for the SQLite schema, migrations and repositories."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mcp_videos.core.models import (
    DialogueLine,
    Project,
    ProjectStatus,
    RenderJob,
    RenderStatus,
    Scene,
    ValidationStatus,
    VoiceProfile,
    WordTimestamp,
)
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueLineNotFoundError,
    DialogueRepository,
    ProjectNotFoundError,
    ProjectRepository,
    RenderJobNotFoundError,
    RenderJobRepository,
    SceneNotFoundError,
    SceneRepository,
    VoiceNotFoundError,
    VoiceRepository,
    replace_project_dialogue,
)

PROJECT_ID = "short-antena-st5"
VOICE_ID = "preset-narrador-es"
CLONE_VOICE_ID = "clon-ingeniero-es"
TOTAL_DURATION_MS = 4500

EXPECTED_COLUMNS = {
    "voices": [
        "id",
        "name",
        "engine",
        "mode",
        "language",
        "sample_wav_path",
        "config_json",
        "created_at",
    ],
    "projects": [
        "id",
        "title",
        "description",
        "width",
        "height",
        "fps",
        "total_duration_ms",
        "master_audio_path",
        "status",
        "created_at",
        "updated_at",
    ],
    "dialogue_lines": [
        "id",
        "project_id",
        "line_index",
        "voice_id",
        "speaker_label",
        "text_content",
        "pause_after_ms",
        "start_ms",
        "end_ms",
        "audio_path",
    ],
    "word_timestamps": [
        "id",
        "line_id",
        "word_index",
        "word",
        "start_ms",
        "end_ms",
        "confidence",
    ],
    "scenes": [
        "id",
        "project_id",
        "scene_index",
        "title",
        "start_ms",
        "end_ms",
        "html_path",
        "prompt_used",
        "validation_status",
        "validation_errors_json",
        "rendered_clip_path",
        "updated_at",
    ],
    "render_jobs": [
        "id",
        "project_id",
        "status",
        "total_frames",
        "rendered_frames",
        "burn_subtitles",
        "output_mp4_path",
        "error_message",
        "started_at",
        "completed_at",
    ],
}


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(tmp_path / "data" / "mcp_videos.sqlite")
    manager.apply_migrations()
    return manager


def make_voice(voice_id: str = VOICE_ID, **overrides: object) -> VoiceProfile:
    data: dict[str, object] = {
        "id": voice_id,
        "name": "Narrador",
        "engine": "kokoro",
        "mode": "preset",
        "language": "es",
        "config": {"voice_code": "ef_dora", "speed": 1.0},
        "created_at": datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return VoiceProfile.model_validate(data)


def make_project(project_id: str = PROJECT_ID, **overrides: object) -> Project:
    data: dict[str, object] = {
        "id": project_id,
        "title": "La antena mas rara de la NASA",
        "description": "Antena evolucionada por algoritmo genetico.",
        "width": 1080,
        "height": 1920,
        "fps": 60,
        "total_duration_ms": 0,
        "status": "draft",
        "created_at": datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return Project.model_validate(data)


def make_line(line_index: int = 0, **overrides: object) -> DialogueLine:
    data: dict[str, object] = {
        "project_id": PROJECT_ID,
        "line_index": line_index,
        "voice_id": VOICE_ID,
        "speaker_label": "Narrador",
        "text_content": f"Linea de dialogo {line_index}",
        "pause_after_ms": 150,
        "start_ms": line_index * 1000,
        "end_ms": line_index * 1000 + 900,
        "audio_path": f"audio/line_{line_index:02d}.wav",
    }
    data.update(overrides)
    return DialogueLine.model_validate(data)


def make_word(word_index: int = 0, **overrides: object) -> WordTimestamp:
    data: dict[str, object] = {
        "word_index": word_index,
        "word": f"palabra{word_index}",
        "start_ms": word_index * 500,
        "end_ms": word_index * 500 + 400,
        "confidence": 0.95,
    }
    data.update(overrides)
    return WordTimestamp.model_validate(data)


def make_scene(scene_index: int = 0, **overrides: object) -> Scene:
    data: dict[str, object] = {
        "project_id": PROJECT_ID,
        "scene_index": scene_index,
        "title": "Hook Antena Evolucionada",
        "start_ms": 0,
        "end_ms": TOTAL_DURATION_MS,
        "html_path": f"scenes/scene_{scene_index:02d}.html",
        "prompt_used": "",
        "validation_status": "pending",
        "validation_errors": [],
        "rendered_clip_path": None,
        "updated_at": datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return Scene.model_validate(data)


def make_job(total_frames: int = 270, **overrides: object) -> RenderJob:
    data: dict[str, object] = {
        "project_id": PROJECT_ID,
        "total_frames": total_frames,
        "burn_subtitles": True,
        "started_at": datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return RenderJob.model_validate(data)


def seed_project(db: DatabaseManager) -> Project:
    ProjectRepository(db).create(make_project())
    VoiceRepository(db).upsert(make_voice())
    return ProjectRepository(db).get_or_raise(PROJECT_ID)


def unique_column_sets(conn: sqlite3.Connection, table: str) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()
    for index in conn.execute(f"PRAGMA index_list({table})").fetchall():
        if index["unique"]:
            columns = tuple(
                row["name"] for row in conn.execute(f"PRAGMA index_info({index['name']})")
            )
            result.add(columns)
    return result


class TestSchema:
    def test_tables_and_migration_control_exist(self, db: DatabaseManager) -> None:
        with db.connection() as conn:
            tables = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        assert set(EXPECTED_COLUMNS) <= tables
        assert "_schema_migrations" in tables

    @pytest.mark.parametrize(("table", "columns"), EXPECTED_COLUMNS.items())
    def test_columns_match_design(
        self,
        db: DatabaseManager,
        table: str,
        columns: list[str],
    ) -> None:
        with db.connection() as conn:
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        assert [row["name"] for row in info] == columns

    def test_column_types_match_design(self, db: DatabaseManager) -> None:
        expected = {
            "voices": {"id": "TEXT", "created_at": "TEXT"},
            "projects": {"id": "TEXT", "width": "INTEGER", "fps": "INTEGER"},
            "dialogue_lines": {"id": "INTEGER", "start_ms": "INTEGER", "text_content": "TEXT"},
            "word_timestamps": {"confidence": "REAL", "word": "TEXT"},
            "scenes": {"validation_errors_json": "TEXT", "scene_index": "INTEGER"},
            "render_jobs": {"total_frames": "INTEGER", "burn_subtitles": "INTEGER"},
        }
        with db.connection() as conn:
            for table, columns in expected.items():
                info = {
                    row["name"]: row["type"] for row in conn.execute(f"PRAGMA table_info({table})")
                }
                for column, column_type in columns.items():
                    assert info[column] == column_type

    def test_foreign_keys_and_actions(self, db: DatabaseManager) -> None:
        with db.connection() as conn:
            line_fks = {
                row["table"]: row["on_delete"]
                for row in conn.execute("PRAGMA foreign_key_list(dialogue_lines)")
            }
            word_fks = {
                row["table"]: row["on_delete"]
                for row in conn.execute("PRAGMA foreign_key_list(word_timestamps)")
            }
            scene_fks = {
                row["table"]: row["on_delete"]
                for row in conn.execute("PRAGMA foreign_key_list(scenes)")
            }
            job_fks = {
                row["table"]: row["on_delete"]
                for row in conn.execute("PRAGMA foreign_key_list(render_jobs)")
            }
        assert line_fks == {"projects": "CASCADE", "voices": "RESTRICT"}
        assert word_fks == {"dialogue_lines": "CASCADE"}
        assert scene_fks == {"projects": "CASCADE"}
        assert job_fks == {"projects": "CASCADE"}

    def test_unique_natural_keys(self, db: DatabaseManager) -> None:
        with db.connection() as conn:
            assert ("project_id", "line_index") in unique_column_sets(conn, "dialogue_lines")
            assert ("line_id", "word_index") in unique_column_sets(conn, "word_timestamps")
            assert ("project_id", "scene_index") in unique_column_sets(conn, "scenes")

    def test_indexes_exist(self, db: DatabaseManager) -> None:
        expected = {
            "idx_dialogue_lines_project_start",
            "idx_word_timestamps_line_start",
            "idx_word_timestamps_window",
            "idx_scenes_project_status",
            "idx_render_jobs_project_status",
        }
        with db.connection() as conn:
            names = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
            }
        assert expected <= names

    def test_connection_pragmas(self, db: DatabaseManager) -> None:
        with db.connection() as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


class TestMigrations:
    def test_apply_migrations_is_idempotent(self, db: DatabaseManager) -> None:
        assert db.apply_migrations() == []
        assert db.migrate() == []
        with db.connection() as conn:
            versions = [
                row["version"] for row in conn.execute("SELECT version FROM _schema_migrations")
            ]
        assert versions == ["0001_initial_schema.sql"]

    def test_fresh_manager_records_version(self, tmp_path: Path) -> None:
        manager = DatabaseManager(tmp_path / "fresh.sqlite")
        assert manager.initialize() == ["0001_initial_schema.sql"]
        assert manager.apply_migrations() == []

    def test_missing_migrations_dir_is_reported(self, tmp_path: Path) -> None:
        manager = DatabaseManager(tmp_path / "db.sqlite", migrations_dir=tmp_path / "missing")

        with pytest.raises(FileNotFoundError):
            manager.apply_migrations()


class TestTransactions:
    def test_transaction_rolls_back_on_exception(self, db: DatabaseManager) -> None:
        with pytest.raises(RuntimeError), db.transaction() as conn:
            conn.execute(
                """
                    INSERT INTO voices (id, name, engine, mode, language)
                    VALUES ('voz-rollback', 'Rollback', 'kokoro', 'preset', 'es')
                    """
            )
            raise RuntimeError("fallo simulado")

        assert VoiceRepository(db).get("voz-rollback") is None

    def test_transaction_commits_on_success(self, db: DatabaseManager) -> None:
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO voices (id, name, engine, mode, language)
                VALUES ('voz-commit', 'Commit', 'kokoro', 'preset', 'es')
                """
            )

        assert VoiceRepository(db).get("voz-commit") is not None


class TestSchemaConstraints:
    @pytest.mark.parametrize(
        "statement",
        [
            pytest.param(
                """
                INSERT INTO voices (id, name, engine, mode, language)
                VALUES ('voz-mala', 'Mala', 'magic', 'preset', 'es')
                """,
                id="voice-engine",
            ),
            pytest.param(
                """
                INSERT INTO voices (id, name, engine, mode, language)
                VALUES ('voz-clon', 'Clon', 'f5', 'clone', 'es')
                """,
                id="clone-without-sample",
            ),
            pytest.param(
                """
                INSERT INTO projects (id, title, fps, created_at, updated_at)
                VALUES ('proyecto-fps', 'FPS', 45, '2026-10-28T12:00:00Z', '2026-10-28T12:00:00Z')
                """,
                id="fps",
            ),
            pytest.param(
                """
                INSERT INTO projects (id, title, width, created_at, updated_at)
                VALUES ('proyecto-ancho', 'Ancho', 0,
                        '2026-10-28T12:00:00Z', '2026-10-28T12:00:00Z')
                """,
                id="width",
            ),
            pytest.param(
                """
                INSERT INTO projects (id, title, status, created_at, updated_at)
                VALUES ('proyecto-estado', 'Estado', 'bogus',
                        '2026-10-28T12:00:00Z', '2026-10-28T12:00:00Z')
                """,
                id="project-status",
            ),
        ],
    )
    def test_voice_and_project_checks(self, db: DatabaseManager, statement: str) -> None:
        with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
            conn.execute(statement)

    def test_dialogue_checks_and_foreign_keys(self, db: DatabaseManager) -> None:
        seed_project(db)
        invalid = [
            """
            INSERT INTO dialogue_lines (
                project_id, line_index, voice_id, speaker_label, text_content,
                pause_after_ms, start_ms, end_ms, audio_path
            ) VALUES ('short-antena-st5', 0, 'preset-narrador-es', 'Narrador',
                      'Texto', 3000, 0, 1000, 'audio/line_00.wav')
            """,
            """
            INSERT INTO dialogue_lines (
                project_id, line_index, voice_id, speaker_label, text_content,
                pause_after_ms, start_ms, end_ms, audio_path
            ) VALUES ('short-antena-st5', 0, 'preset-narrador-es', 'Narrador',
                      'Texto', 150, 1000, 1000, 'audio/line_00.wav')
            """,
            """
            INSERT INTO dialogue_lines (
                project_id, line_index, voice_id, speaker_label, text_content,
                pause_after_ms, start_ms, end_ms, audio_path
            ) VALUES ('short-antena-st5', 0, 'preset-narrador-es', 'Narrador',
                      '   ', 150, 0, 1000, 'audio/line_00.wav')
            """,
            """
            INSERT INTO dialogue_lines (
                project_id, line_index, voice_id, speaker_label, text_content,
                pause_after_ms, start_ms, end_ms, audio_path
            ) VALUES ('proyecto-fantasma', 0, 'preset-narrador-es', 'Narrador',
                      'Texto', 150, 0, 1000, 'audio/line_00.wav')
            """,
            """
            INSERT INTO dialogue_lines (
                project_id, line_index, voice_id, speaker_label, text_content,
                pause_after_ms, start_ms, end_ms, audio_path
            ) VALUES ('short-antena-st5', 0, 'voz-fantasma', 'Narrador',
                      'Texto', 150, 0, 1000, 'audio/line_00.wav')
            """,
        ]
        for statement in invalid:
            with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
                conn.execute(statement)

    def test_duplicate_line_index_is_rejected(self, db: DatabaseManager) -> None:
        seed_project(db)
        DialogueRepository(db).insert_line(make_line(0))

        with pytest.raises(sqlite3.IntegrityError):
            DialogueRepository(db).insert_line(make_line(0))

    def test_word_and_scene_checks(self, db: DatabaseManager) -> None:
        seed_project(db)
        line = DialogueRepository(db).insert_line(make_line(0))
        assert line.id is not None
        DialogueRepository(db).insert_word_timestamps(line.id, [make_word(0)])
        SceneRepository(db).upsert(make_scene(0))

        invalid = [
            (
                """
                INSERT INTO word_timestamps (
                    line_id, word_index, word, start_ms, end_ms, confidence
                )
                VALUES (?, 1, 'palabra', 500, 900, 1.5)
                """,
                [line.id],
            ),
            (
                """
                INSERT INTO word_timestamps (
                    line_id, word_index, word, start_ms, end_ms, confidence
                )
                VALUES (?, 0, 'palabra', 900, 1200, 0.9)
                """,
                [line.id],
            ),
            (
                """
                INSERT INTO word_timestamps (
                    line_id, word_index, word, start_ms, end_ms, confidence
                )
                VALUES (?, 1, 'palabra', 900, 900, 0.9)
                """,
                [line.id],
            ),
            (
                """
                INSERT INTO scenes (
                    project_id, scene_index, title, start_ms, end_ms, html_path, updated_at
                ) VALUES ('short-antena-st5', 0, 'Duplicada', 0, 1000, 'scenes/x.html',
                          '2026-10-28T12:00:00Z')
                """,
                [],
            ),
            (
                """
                INSERT INTO scenes (
                    project_id, scene_index, title, start_ms, end_ms, html_path, updated_at
                ) VALUES ('short-antena-st5', 1, 'Invalida', 1000, 1000, 'scenes/x.html',
                          '2026-10-28T12:00:00Z')
                """,
                [],
            ),
            (
                """
                INSERT INTO scenes (
                    project_id, scene_index, title, start_ms, end_ms, html_path,
                    validation_status, updated_at
                ) VALUES ('short-antena-st5', 1, 'Invalida', 0, 1000, 'scenes/x.html',
                          'bogus', '2026-10-28T12:00:00Z')
                """,
                [],
            ),
            (
                """
                INSERT INTO word_timestamps (line_id, word_index, word, start_ms, end_ms)
                VALUES (99999, 0, 'palabra', 0, 500)
                """,
                [],
            ),
        ]
        for statement, params in invalid:
            with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
                conn.execute(statement, params)

    def test_render_job_checks(self, db: DatabaseManager) -> None:
        seed_project(db)
        invalid = [
            """
            INSERT INTO render_jobs (id, project_id, status, total_frames, started_at)
            VALUES ('job-1', 'short-antena-st5', 'queued', 0, '2026-10-28T12:00:00Z')
            """,
            """
            INSERT INTO render_jobs (
                id, project_id, status, total_frames, burn_subtitles, started_at
            ) VALUES ('job-2', 'short-antena-st5', 'queued', 270, 2, '2026-10-28T12:00:00Z')
            """,
            """
            INSERT INTO render_jobs (id, project_id, status, total_frames, started_at)
            VALUES ('job-3', 'short-antena-st5', 'bogus', 270, '2026-10-28T12:00:00Z')
            """,
        ]
        for statement in invalid:
            with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
                conn.execute(statement)


class TestVoiceRepository:
    def test_upsert_get_list_and_delete(self, db: DatabaseManager) -> None:
        repo = VoiceRepository(db)
        stored = repo.upsert(make_voice())
        assert stored.id == VOICE_ID
        assert stored.config == {"voice_code": "ef_dora", "speed": 1.0}

        updated = repo.upsert(
            make_voice(
                name="Narrador principal",
                config={"voice_code": "ef_dora", "speed": 1.2},
            )
        )
        assert updated.name == "Narrador principal"
        assert updated.config["speed"] == 1.2
        assert updated.created_at == stored.created_at

        repo.upsert(
            make_voice(
                CLONE_VOICE_ID,
                name="Ingeniero",
                engine="f5",
                mode="clone",
                sample_wav_path="data/voices/clon-ingeniero-es.wav",
            )
        )
        assert [voice.id for voice in repo.list_all()] == [CLONE_VOICE_ID, VOICE_ID]

        assert repo.get(VOICE_ID) == updated
        assert repo.get_or_raise(VOICE_ID) == updated
        with pytest.raises(VoiceNotFoundError):
            repo.get_or_raise("voz-inexistente")
        assert repo.delete(VOICE_ID) is True
        assert repo.get(VOICE_ID) is None
        assert repo.delete(VOICE_ID) is False


class TestProjectRepository:
    def test_crud_status_and_save(self, db: DatabaseManager) -> None:
        repo = ProjectRepository(db)
        project = make_project()
        created = repo.create(project)
        assert created == project
        assert repo.get(PROJECT_ID) == project
        assert repo.get_or_raise(PROJECT_ID) == project
        assert [item.id for item in repo.list_all()] == [PROJECT_ID]

        running = repo.update_status(PROJECT_ID, ProjectStatus.AUDIO_READY)
        assert running.status is ProjectStatus.AUDIO_READY
        assert running.updated_at >= project.updated_at

        changed = running.model_copy(
            update={"title": "Titulo actualizado", "total_duration_ms": TOTAL_DURATION_MS}
        )
        saved = repo.save(changed)
        assert saved.title == "Titulo actualizado"
        assert saved.total_duration_ms == TOTAL_DURATION_MS

        with pytest.raises(ProjectNotFoundError):
            repo.update_status("proyecto-fantasma", ProjectStatus.DRAFT)
        with pytest.raises(ProjectNotFoundError):
            repo.save(make_project("proyecto-fantasma"))

        assert repo.delete(PROJECT_ID) is True
        assert repo.get(PROJECT_ID) is None
        assert repo.delete(PROJECT_ID) is False


class TestDialogueRepository:
    def test_lines_are_ordered_and_deleted_by_project(self, db: DatabaseManager) -> None:
        seed_project(db)
        repo = DialogueRepository(db)
        second = repo.insert_line(make_line(1))
        first = repo.insert_line(make_line(0))

        assert first.id is not None
        assert second.id is not None
        assert [line.line_index for line in repo.list_by_project(PROJECT_ID)] == [0, 1]
        assert repo.get_line(first.id) == first

        repo.insert_word_timestamps(first.id, [make_word(0), make_word(1)])
        assert len(repo.list_word_timestamps(first.id)) == 2

        assert repo.delete_by_project(PROJECT_ID) == 2
        assert repo.list_by_project(PROJECT_ID) == []
        assert repo.list_word_timestamps(first.id) == []

    def test_word_timestamps_window_and_replacement(self, db: DatabaseManager) -> None:
        seed_project(db)
        repo = DialogueRepository(db)
        line = repo.insert_line(make_line(0))
        assert line.id is not None
        repo.insert_word_timestamps(line.id, [make_word(0), make_word(1)])

        overlapping = repo.word_timestamps_in_window(300, 900, project_id=PROJECT_ID)
        assert [word.word_index for word in overlapping] == [0, 1]
        boundary = repo.word_timestamps_in_window(400, 900, project_id=PROJECT_ID)
        assert [word.word_index for word in boundary] == [1]
        assert repo.word_timestamps_in_window(900, 1000, project_id=PROJECT_ID) == []
        assert repo.word_timestamps_in_window(0, 10000, project_id="otro-proyecto") == []

        replaced = repo.replace_word_timestamps(line.id, [make_word(0, word="Hola")])
        assert [word.word for word in replaced] == ["Hola"]
        assert [word.word for word in repo.list_word_timestamps(line.id)] == ["Hola"]

        with pytest.raises(ValueError):
            repo.word_timestamps_in_window(100, 100)
        with pytest.raises(DialogueLineNotFoundError):
            repo.insert_word_timestamps(99999, [make_word(0)])

    def test_deleting_project_cascades_lines_and_words(self, db: DatabaseManager) -> None:
        seed_project(db)
        line = DialogueRepository(db).insert_line(make_line(0))
        assert line.id is not None
        DialogueRepository(db).insert_word_timestamps(line.id, [make_word(0)])

        ProjectRepository(db).delete(PROJECT_ID)

        with db.connection() as conn:
            lines = conn.execute("SELECT COUNT(*) FROM dialogue_lines").fetchone()[0]
            words = conn.execute("SELECT COUNT(*) FROM word_timestamps").fetchone()[0]
        assert lines == 0
        assert words == 0


class TestSceneRepository:
    def test_upsert_updates_existing_scene(self, db: DatabaseManager) -> None:
        seed_project(db)
        repo = SceneRepository(db)
        first = repo.upsert(make_scene(0, validation_status="valid"))
        second = repo.upsert(
            make_scene(
                0,
                title="Hook actualizado",
                validation_status="invalid",
                validation_errors=["ReferenceError: x is not defined"],
                rendered_clip_path="output/scene_00.mp4",
            )
        )

        assert second.id == first.id
        assert second.title == "Hook actualizado"
        assert second.validation_errors == ["ReferenceError: x is not defined"]
        assert repo.get(PROJECT_ID, 0) == second
        assert repo.get_or_raise(PROJECT_ID, 0) == second
        assert [scene.scene_index for scene in repo.list_by_project(PROJECT_ID)] == [0]
        with pytest.raises(SceneNotFoundError):
            repo.get_or_raise(PROJECT_ID, 9)

    def test_validation_updates_and_invalidation(self, db: DatabaseManager) -> None:
        seed_project(db)
        repo = SceneRepository(db)
        repo.upsert(make_scene(0, rendered_clip_path="output/scene_00.mp4"))

        valid = repo.update_validation(PROJECT_ID, 0, ValidationStatus.VALID)
        assert valid.validation_status is ValidationStatus.VALID
        invalid = repo.update_validation(
            PROJECT_ID,
            0,
            ValidationStatus.INVALID,
            ["SyntaxError"],
        )
        assert invalid.validation_errors == ["SyntaxError"]
        with pytest.raises(SceneNotFoundError):
            repo.update_validation(PROJECT_ID, 9, ValidationStatus.VALID)

        assert repo.invalidate_project_scenes(PROJECT_ID) == 1
        invalidated = repo.get_or_raise(PROJECT_ID, 0)
        assert invalidated.validation_status is ValidationStatus.PENDING
        assert invalidated.rendered_clip_path is None
        assert repo.delete(PROJECT_ID, 0) is True
        assert repo.get(PROJECT_ID, 0) is None


class TestRenderJobRepository:
    def test_lifecycle_progress_completion_and_failure(self, db: DatabaseManager) -> None:
        seed_project(db)
        repo = RenderJobRepository(db)
        job = make_job()
        created = repo.create(job)
        assert created.status is RenderStatus.QUEUED
        assert created.rendered_frames == 0
        assert repo.get(job.id) == created
        assert repo.get_or_raise(job.id) == created
        assert [item.id for item in repo.list_by_project(PROJECT_ID)] == [job.id]

        running = repo.update_progress(job.id, 100, status=RenderStatus.RUNNING)
        assert running.rendered_frames == 100
        assert running.status is RenderStatus.RUNNING
        with pytest.raises(ValueError):
            repo.update_progress(job.id, 271)

        completed = repo.complete(job.id, "output/final.mp4")
        assert completed.status is RenderStatus.COMPLETED
        assert completed.rendered_frames == completed.total_frames
        assert completed.output_mp4_path == "output/final.mp4"
        assert completed.completed_at is not None

        failed = repo.fail(job.id, "FFmpeg exit code 1")
        assert failed.status is RenderStatus.FAILED
        assert failed.error_message == "FFmpeg exit code 1"

        with pytest.raises(RenderJobNotFoundError):
            repo.update_progress("00000000-0000-4000-8000-000000000000", 1)
        with pytest.raises(RenderJobNotFoundError):
            repo.complete("00000000-0000-4000-8000-000000000000", "x.mp4")
        with pytest.raises(RenderJobNotFoundError):
            repo.fail("00000000-0000-4000-8000-000000000000", "x")

    def test_deleting_project_cascades_render_jobs(self, db: DatabaseManager) -> None:
        seed_project(db)
        job = RenderJobRepository(db).create(make_job())
        ProjectRepository(db).delete(PROJECT_ID)

        assert RenderJobRepository(db).get(job.id) is None


def seed_dialogue_state(db: DatabaseManager) -> None:
    ProjectRepository(db).create(make_project(status="scenes_ready"))
    VoiceRepository(db).upsert(make_voice())
    VoiceRepository(db).upsert(
        make_voice(
            CLONE_VOICE_ID,
            name="Ingeniero",
            engine="f5",
            mode="clone",
            sample_wav_path="data/voices/clon-ingeniero-es.wav",
        )
    )
    line = DialogueRepository(db).insert_line(make_line(0))
    assert line.id is not None
    DialogueRepository(db).insert_line(
        make_line(1, voice_id=CLONE_VOICE_ID, start_ms=1000, end_ms=2000)
    )
    DialogueRepository(db).insert_word_timestamps(line.id, [make_word(0), make_word(1)])
    SceneRepository(db).upsert(
        make_scene(
            0,
            validation_status="valid",
            rendered_clip_path="output/scene_00.mp4",
        )
    )


class TestReplaceProjectDialogue:
    def test_replaces_lines_updates_duration_and_invalidates_scenes(
        self,
        db: DatabaseManager,
    ) -> None:
        seed_dialogue_state(db)
        old_lines = DialogueRepository(db).list_by_project(PROJECT_ID)
        old_line_ids = {line.id for line in old_lines}
        old_line_with_words = old_lines[0]
        assert old_line_with_words.id is not None

        new_lines = [
            make_line(0, text_content="Nuevo inicio", end_ms=1800),
            make_line(
                1,
                voice_id=CLONE_VOICE_ID,
                text_content="Nuevo cierre",
                start_ms=2000,
                end_ms=4100,
            ),
        ]
        inserted = replace_project_dialogue(db, PROJECT_ID, new_lines, 4150)

        assert [line.line_index for line in inserted] == [0, 1]
        assert all(line.id is not None and line.id not in old_line_ids for line in inserted)
        project = ProjectRepository(db).get_or_raise(PROJECT_ID)
        assert project.total_duration_ms == 4150
        assert project.status is ProjectStatus.AUDIO_READY

        scene = SceneRepository(db).get_or_raise(PROJECT_ID, 0)
        assert scene.validation_status is ValidationStatus.PENDING
        assert scene.rendered_clip_path is None

        assert DialogueRepository(db).list_word_timestamps(old_line_with_words.id) == []
        assert (
            DialogueRepository(db).word_timestamps_in_window(0, 5000, project_id=PROJECT_ID) == []
        )

    def test_master_audio_path_is_preserved_when_not_provided(
        self,
        db: DatabaseManager,
    ) -> None:
        ProjectRepository(db).create(make_project(master_audio_path="audio/dialogue_master.wav"))
        VoiceRepository(db).upsert(make_voice())

        replace_project_dialogue(db, PROJECT_ID, [make_line(0)], 2000)
        assert ProjectRepository(db).get_or_raise(PROJECT_ID).master_audio_path == (
            "audio/dialogue_master.wav"
        )

        replace_project_dialogue(
            db,
            PROJECT_ID,
            [make_line(0)],
            2000,
            master_audio_path="audio/new_master.wav",
        )
        assert ProjectRepository(db).get_or_raise(PROJECT_ID).master_audio_path == (
            "audio/new_master.wav"
        )

    def test_invalid_replacement_rolls_back_everything(self, db: DatabaseManager) -> None:
        seed_dialogue_state(db)
        before_lines = DialogueRepository(db).list_by_project(PROJECT_ID)
        before_scene = SceneRepository(db).get_or_raise(PROJECT_ID, 0)

        bad_lines = [
            make_line(0, text_content="Primera valida"),
            make_line(1, voice_id="voz-inexistente", start_ms=1000, end_ms=2000),
        ]
        with pytest.raises(sqlite3.IntegrityError):
            replace_project_dialogue(db, PROJECT_ID, bad_lines, 2500)

        assert DialogueRepository(db).list_by_project(PROJECT_ID) == before_lines
        project = ProjectRepository(db).get_or_raise(PROJECT_ID)
        assert project.total_duration_ms == 0
        assert project.status is ProjectStatus.SCENES_READY
        assert SceneRepository(db).get_or_raise(PROJECT_ID, 0) == before_scene

    def test_validation_errors_before_opening_transaction(self, db: DatabaseManager) -> None:
        seed_project(db)

        with pytest.raises(ValueError):
            replace_project_dialogue(db, PROJECT_ID, [make_line(0)], -1)
        with pytest.raises(ValueError):
            replace_project_dialogue(
                db,
                PROJECT_ID,
                [make_line(0), make_line(0)],
                2000,
            )
        with pytest.raises(ValueError):
            replace_project_dialogue(
                db,
                PROJECT_ID,
                [make_line(0, project_id="otro-proyecto")],
                2000,
            )
        with pytest.raises(ValueError):
            replace_project_dialogue(
                db,
                PROJECT_ID,
                [make_line(0, audio_path=None)],
                2000,
            )
        with pytest.raises(ProjectNotFoundError):
            replace_project_dialogue(
                db,
                "proyecto-fantasma",
                [make_line(0, project_id="proyecto-fantasma")],
                2000,
            )
