"""Typed SQLite repositories and transactional project operations."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime

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


class RepositoryError(RuntimeError):
    """Base error for persistence operations."""


class ProjectNotFoundError(RepositoryError):
    """Raised when a project id does not exist."""


class VoiceNotFoundError(RepositoryError):
    """Raised when a voice id does not exist."""


class DialogueLineNotFoundError(RepositoryError):
    """Raised when a dialogue line id does not exist."""


class SceneNotFoundError(RepositoryError):
    """Raised when a scene coordinate does not exist."""


class RenderJobNotFoundError(RepositoryError):
    """Raised when a render job id does not exist."""


def _utc_iso(value: datetime | None = None) -> str:
    moment = value if value is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _voice_from_row(row: sqlite3.Row) -> VoiceProfile:
    return VoiceProfile.model_validate(dict(row))


def _project_from_row(row: sqlite3.Row) -> Project:
    return Project.model_validate(dict(row))


def _line_from_row(row: sqlite3.Row) -> DialogueLine:
    return DialogueLine.model_validate(dict(row))


def _word_from_row(row: sqlite3.Row) -> WordTimestamp:
    return WordTimestamp.model_validate(dict(row))


def _scene_from_row(row: sqlite3.Row) -> Scene:
    return Scene.model_validate(dict(row))


def _render_job_from_row(row: sqlite3.Row) -> RenderJob:
    return RenderJob.model_validate(dict(row))


def _insert_line(conn: sqlite3.Connection, line: DialogueLine) -> DialogueLine:
    if line.audio_path is None or not line.audio_path.strip():
        raise ValueError("audio_path es obligatorio para persistir una linea de dialogo")
    cursor = conn.execute(
        """
        INSERT INTO dialogue_lines (
            project_id, line_index, voice_id, speaker_label, text_content,
            pause_after_ms, start_ms, end_ms, audio_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            line.project_id,
            line.line_index,
            line.voice_id,
            line.speaker_label,
            line.text_content,
            line.pause_after_ms,
            line.start_ms,
            line.end_ms,
            line.audio_path,
        ),
    )
    return line.model_copy(update={"id": int(cursor.lastrowid)})


def _insert_words(
    conn: sqlite3.Connection,
    line_id: int,
    timestamps: Sequence[WordTimestamp],
) -> list[WordTimestamp]:
    inserted: list[WordTimestamp] = []
    for timestamp in timestamps:
        if timestamp.line_id is not None and timestamp.line_id != line_id:
            raise ValueError("line_id de WordTimestamp no coincide con la linea destino")
        cursor = conn.execute(
            """
            INSERT INTO word_timestamps (
                line_id, word_index, word, start_ms, end_ms, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                line_id,
                timestamp.word_index,
                timestamp.word,
                timestamp.start_ms,
                timestamp.end_ms,
                timestamp.confidence,
            ),
        )
        inserted.append(
            timestamp.model_copy(update={"id": int(cursor.lastrowid), "line_id": line_id})
        )
    return inserted


class VoiceRepository:
    """CRUD access for the ``voices`` catalog."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def upsert(self, voice: VoiceProfile) -> VoiceProfile:
        """Insert a voice or replace its mutable fields when the id exists."""

        created_at = _utc_iso(voice.created_at)
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO voices (
                    id, name, engine, mode, language, sample_wav_path,
                    config_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    name = excluded.name,
                    engine = excluded.engine,
                    mode = excluded.mode,
                    language = excluded.language,
                    sample_wav_path = excluded.sample_wav_path,
                    config_json = excluded.config_json
                """,
                (
                    voice.id,
                    voice.name,
                    voice.engine.value,
                    voice.mode.value,
                    voice.language.value,
                    voice.sample_wav_path,
                    json.dumps(voice.config, ensure_ascii=False, sort_keys=True),
                    created_at,
                ),
            )
            row = conn.execute("SELECT * FROM voices WHERE id = ?", (voice.id,)).fetchone()
        assert row is not None
        return _voice_from_row(row)

    def get(self, voice_id: str) -> VoiceProfile | None:
        """Return one voice or ``None`` when it is not registered."""

        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM voices WHERE id = ?", (voice_id,)).fetchone()
        return _voice_from_row(row) if row is not None else None

    def get_or_raise(self, voice_id: str) -> VoiceProfile:
        """Return one voice, raising :class:`VoiceNotFoundError` when missing."""

        voice = self.get(voice_id)
        if voice is None:
            raise VoiceNotFoundError(f"Voz no registrada: {voice_id}")
        return voice

    def list_all(self) -> list[VoiceProfile]:
        """List every voice ordered by its canonical id."""

        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM voices ORDER BY id").fetchall()
        return [_voice_from_row(row) for row in rows]

    def delete(self, voice_id: str) -> bool:
        """Delete a voice; referenced voices are protected by ``ON DELETE RESTRICT``."""

        with self.db.transaction() as conn:
            cursor = conn.execute("DELETE FROM voices WHERE id = ?", (voice_id,))
        return cursor.rowcount > 0


class ProjectRepository:
    """CRUD access for ``projects`` and their lifecycle metadata."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def create(self, project: Project) -> Project:
        """Insert a new project."""

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, title, description, width, height, fps,
                    total_duration_ms, master_audio_path, status,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.title,
                    project.description,
                    project.width,
                    project.height,
                    project.fps,
                    project.total_duration_ms,
                    project.master_audio_path,
                    project.status.value,
                    _utc_iso(project.created_at),
                    _utc_iso(project.updated_at),
                ),
            )
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project.id,)).fetchone()
        assert row is not None
        return _project_from_row(row)

    def upsert(self, project: Project) -> Project:
        """Insert or update a project, preserving its original ``created_at``."""

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, title, description, width, height, fps,
                    total_duration_ms, master_audio_path, status,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    width = excluded.width,
                    height = excluded.height,
                    fps = excluded.fps,
                    total_duration_ms = excluded.total_duration_ms,
                    master_audio_path = excluded.master_audio_path,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    project.id,
                    project.title,
                    project.description,
                    project.width,
                    project.height,
                    project.fps,
                    project.total_duration_ms,
                    project.master_audio_path,
                    project.status.value,
                    _utc_iso(project.created_at),
                    _utc_iso(project.updated_at),
                ),
            )
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project.id,)).fetchone()
        assert row is not None
        return _project_from_row(row)

    def get(self, project_id: str) -> Project | None:
        """Return one project or ``None`` when it does not exist."""

        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return _project_from_row(row) if row is not None else None

    def get_or_raise(self, project_id: str) -> Project:
        """Return one project, raising :class:`ProjectNotFoundError` when missing."""

        project = self.get(project_id)
        if project is None:
            raise ProjectNotFoundError(f"Proyecto no registrado: {project_id}")
        return project

    def list_all(self) -> list[Project]:
        """List every project ordered by creation time and id."""

        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at, id").fetchall()
        return [_project_from_row(row) for row in rows]

    def update_status(self, project_id: str, status: ProjectStatus) -> Project:
        """Update the lifecycle status and touch ``updated_at``."""

        with self.db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                (ProjectStatus(status).value, _utc_iso(), project_id),
            )
            if cursor.rowcount == 0:
                raise ProjectNotFoundError(f"Proyecto no registrado: {project_id}")
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        assert row is not None
        return _project_from_row(row)

    def save(self, project: Project) -> Project:
        """Persist every mutable project field."""

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE projects
                   SET title = ?,
                       description = ?,
                       width = ?,
                       height = ?,
                       fps = ?,
                       total_duration_ms = ?,
                       master_audio_path = ?,
                       status = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (
                    project.title,
                    project.description,
                    project.width,
                    project.height,
                    project.fps,
                    project.total_duration_ms,
                    project.master_audio_path,
                    project.status.value,
                    _utc_iso(project.updated_at),
                    project.id,
                ),
            )
            if cursor.rowcount == 0:
                raise ProjectNotFoundError(f"Proyecto no registrado: {project.id}")
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project.id,)).fetchone()
        assert row is not None
        return _project_from_row(row)

    def delete(self, project_id: str) -> bool:
        """Delete a project and cascade its lines, words, scenes and jobs."""

        with self.db.transaction() as conn:
            cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0


class DialogueRepository:
    """CRUD access for dialogue lines and their word timestamps."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def insert_line(self, line: DialogueLine) -> DialogueLine:
        """Insert one dialogue line and return it with its assigned id."""

        with self.db.transaction() as conn:
            return _insert_line(conn, line)

    def replace_lines(self, project_id: str, lines: Sequence[DialogueLine]) -> list[DialogueLine]:
        """Replace every line of a project without touching project metadata."""

        with self.db.transaction() as conn:
            conn.execute("DELETE FROM dialogue_lines WHERE project_id = ?", (project_id,))
            return [_insert_line(conn, line) for line in lines if line.project_id == project_id]

    def get_line(self, line_id: int) -> DialogueLine | None:
        """Return one dialogue line by its surrogate id."""

        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM dialogue_lines WHERE id = ?", (line_id,)).fetchone()
        return _line_from_row(row) if row is not None else None

    def list_by_project(self, project_id: str) -> list[DialogueLine]:
        """List the dialogue of a project ordered by ``line_index``."""

        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM dialogue_lines WHERE project_id = ? ORDER BY line_index",
                (project_id,),
            ).fetchall()
        return [_line_from_row(row) for row in rows]

    def delete_by_project(self, project_id: str) -> int:
        """Delete every line of a project and cascade their word timestamps."""

        with self.db.transaction() as conn:
            cursor = conn.execute("DELETE FROM dialogue_lines WHERE project_id = ?", (project_id,))
        return cursor.rowcount

    def insert_word_timestamps(
        self,
        line_id: int,
        timestamps: Sequence[WordTimestamp],
    ) -> list[WordTimestamp]:
        """Append word timestamps to an existing dialogue line."""

        with self.db.transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM dialogue_lines WHERE id = ?",
                (line_id,),
            ).fetchone()
            if exists is None:
                raise DialogueLineNotFoundError(f"Linea de dialogo no registrada: {line_id}")
            return _insert_words(conn, line_id, timestamps)

    def replace_word_timestamps(
        self,
        line_id: int,
        timestamps: Sequence[WordTimestamp],
    ) -> list[WordTimestamp]:
        """Delete and reinsert every word timestamp of one line atomically."""

        with self.db.transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM dialogue_lines WHERE id = ?",
                (line_id,),
            ).fetchone()
            if exists is None:
                raise DialogueLineNotFoundError(f"Linea de dialogo no registrada: {line_id}")
            conn.execute("DELETE FROM word_timestamps WHERE line_id = ?", (line_id,))
            return _insert_words(conn, line_id, timestamps)

    def list_word_timestamps(self, line_id: int) -> list[WordTimestamp]:
        """List the timestamps of one line ordered by ``word_index``."""

        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM word_timestamps WHERE line_id = ? ORDER BY word_index",
                (line_id,),
            ).fetchall()
        return [_word_from_row(row) for row in rows]

    def word_timestamps_in_window(
        self,
        start_ms: int,
        end_ms: int,
        *,
        project_id: str | None = None,
    ) -> list[WordTimestamp]:
        """List words whose pronunciation interval intersects ``[start_ms, end_ms)``."""

        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("la ventana debe cumplir 0 <= start_ms < end_ms")
        query = """
            SELECT wt.*
              FROM word_timestamps AS wt
              JOIN dialogue_lines AS dl ON dl.id = wt.line_id
             WHERE wt.end_ms > ? AND wt.start_ms < ?
        """
        params: list[object] = [start_ms, end_ms]
        if project_id is not None:
            query += " AND dl.project_id = ?"
            params.append(project_id)
        query += " ORDER BY wt.start_ms, wt.line_id, wt.word_index"
        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_word_from_row(row) for row in rows]


class SceneRepository:
    """CRUD access for scenes and their validation state."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def upsert(self, scene: Scene) -> Scene:
        """Insert or update a scene identified by ``(project_id, scene_index)``."""

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO scenes (
                    project_id, scene_index, title, start_ms, end_ms, html_path,
                    prompt_used, validation_status, validation_errors_json,
                    rendered_clip_path, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (project_id, scene_index) DO UPDATE SET
                    title = excluded.title,
                    start_ms = excluded.start_ms,
                    end_ms = excluded.end_ms,
                    html_path = excluded.html_path,
                    prompt_used = excluded.prompt_used,
                    validation_status = excluded.validation_status,
                    validation_errors_json = excluded.validation_errors_json,
                    rendered_clip_path = excluded.rendered_clip_path,
                    updated_at = excluded.updated_at
                """,
                (
                    scene.project_id,
                    scene.scene_index,
                    scene.title,
                    scene.start_ms,
                    scene.end_ms,
                    scene.html_path,
                    scene.prompt_used,
                    ValidationStatus(scene.validation_status).value,
                    scene.validation_errors_json,
                    scene.rendered_clip_path,
                    _utc_iso(scene.updated_at),
                ),
            )
            row = conn.execute(
                "SELECT * FROM scenes WHERE project_id = ? AND scene_index = ?",
                (scene.project_id, scene.scene_index),
            ).fetchone()
        assert row is not None
        return _scene_from_row(row)

    def get(self, project_id: str, scene_index: int) -> Scene | None:
        """Return one scene by its natural key."""

        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM scenes WHERE project_id = ? AND scene_index = ?",
                (project_id, scene_index),
            ).fetchone()
        return _scene_from_row(row) if row is not None else None

    def get_or_raise(self, project_id: str, scene_index: int) -> Scene:
        """Return one scene, raising :class:`SceneNotFoundError` when missing."""

        scene = self.get(project_id, scene_index)
        if scene is None:
            raise SceneNotFoundError(
                f"Escena no registrada: project_id={project_id}, scene_index={scene_index}"
            )
        return scene

    def list_by_project(self, project_id: str) -> list[Scene]:
        """List the scenes of a project ordered by ``scene_index``."""

        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scenes WHERE project_id = ? ORDER BY scene_index",
                (project_id,),
            ).fetchall()
        return [_scene_from_row(row) for row in rows]

    def update_validation(
        self,
        project_id: str,
        scene_index: int,
        status: ValidationStatus,
        errors: Sequence[str] | None = None,
    ) -> Scene:
        """Update the validation gate result of one scene."""

        payload = "[]" if errors is None else json.dumps(list(errors), ensure_ascii=False)
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE scenes
                   SET validation_status = ?,
                       validation_errors_json = ?,
                       updated_at = ?
                 WHERE project_id = ? AND scene_index = ?
                """,
                (
                    ValidationStatus(status).value,
                    payload,
                    _utc_iso(),
                    project_id,
                    scene_index,
                ),
            )
            if cursor.rowcount == 0:
                raise SceneNotFoundError(
                    f"Escena no registrada: project_id={project_id}, scene_index={scene_index}"
                )
            row = conn.execute(
                "SELECT * FROM scenes WHERE project_id = ? AND scene_index = ?",
                (project_id, scene_index),
            ).fetchone()
        assert row is not None
        return _scene_from_row(row)

    def invalidate_project_scenes(self, project_id: str) -> int:
        """Mark every scene of a project as pending and drop rendered clips."""

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE scenes
                   SET validation_status = ?,
                       rendered_clip_path = NULL,
                       updated_at = ?
                 WHERE project_id = ?
                """,
                (ValidationStatus.PENDING.value, _utc_iso(), project_id),
            )
        return cursor.rowcount

    def delete(self, project_id: str, scene_index: int) -> bool:
        """Delete one scene by its natural key."""

        with self.db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM scenes WHERE project_id = ? AND scene_index = ?",
                (project_id, scene_index),
            )
        return cursor.rowcount > 0


class RenderJobRepository:
    """CRUD access for render jobs and their progress."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def create(self, job: RenderJob) -> RenderJob:
        """Insert a render job."""

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO render_jobs (
                    id, project_id, status, total_frames, rendered_frames,
                    burn_subtitles, output_mp4_path, error_message,
                    started_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.project_id,
                    RenderStatus(job.status).value,
                    job.total_frames,
                    job.rendered_frames,
                    int(job.burn_subtitles),
                    job.output_mp4_path,
                    job.error_message,
                    _utc_iso(job.started_at),
                    _utc_iso(job.completed_at) if job.completed_at is not None else None,
                ),
            )
            row = conn.execute("SELECT * FROM render_jobs WHERE id = ?", (job.id,)).fetchone()
        assert row is not None
        return _render_job_from_row(row)

    def get(self, job_id: str) -> RenderJob | None:
        """Return one render job by id."""

        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM render_jobs WHERE id = ?", (job_id,)).fetchone()
        return _render_job_from_row(row) if row is not None else None

    def get_or_raise(self, job_id: str) -> RenderJob:
        """Return one render job, raising :class:`RenderJobNotFoundError` when missing."""

        job = self.get(job_id)
        if job is None:
            raise RenderJobNotFoundError(f"Job de render no registrado: {job_id}")
        return job

    def list_by_project(self, project_id: str) -> list[RenderJob]:
        """List render jobs of a project ordered by start time."""

        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM render_jobs WHERE project_id = ? ORDER BY started_at, id",
                (project_id,),
            ).fetchall()
        return [_render_job_from_row(row) for row in rows]

    def update_progress(
        self,
        job_id: str,
        rendered_frames: int,
        *,
        status: RenderStatus | None = None,
    ) -> RenderJob:
        """Update the number of rendered frames and optionally the status."""

        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM render_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise RenderJobNotFoundError(f"Job de render no registrado: {job_id}")
            job = _render_job_from_row(row)
            if rendered_frames < 0 or rendered_frames > job.total_frames:
                raise ValueError("rendered_frames debe estar entre 0 y total_frames")
            resolved_status = RenderStatus(status).value if status is not None else job.status.value
            conn.execute(
                "UPDATE render_jobs SET rendered_frames = ?, status = ? WHERE id = ?",
                (rendered_frames, resolved_status, job_id),
            )
            updated = conn.execute("SELECT * FROM render_jobs WHERE id = ?", (job_id,)).fetchone()
        assert updated is not None
        return _render_job_from_row(updated)

    def complete(self, job_id: str, output_mp4_path: str) -> RenderJob:
        """Mark one job as completed and record its final artifact."""

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE render_jobs
                   SET status = ?,
                       rendered_frames = total_frames,
                       output_mp4_path = ?,
                       error_message = NULL,
                       completed_at = ?
                 WHERE id = ?
                """,
                (RenderStatus.COMPLETED.value, output_mp4_path, _utc_iso(), job_id),
            )
            if cursor.rowcount == 0:
                raise RenderJobNotFoundError(f"Job de render no registrado: {job_id}")
            row = conn.execute("SELECT * FROM render_jobs WHERE id = ?", (job_id,)).fetchone()
        assert row is not None
        return _render_job_from_row(row)

    def fail(self, job_id: str, error_message: str) -> RenderJob:
        """Mark one job as failed and store its diagnostic message."""

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE render_jobs
                   SET status = ?,
                       error_message = ?,
                       completed_at = ?
                 WHERE id = ?
                """,
                (RenderStatus.FAILED.value, error_message, _utc_iso(), job_id),
            )
            if cursor.rowcount == 0:
                raise RenderJobNotFoundError(f"Job de render no registrado: {job_id}")
            row = conn.execute("SELECT * FROM render_jobs WHERE id = ?", (job_id,)).fetchone()
        assert row is not None
        return _render_job_from_row(row)


def replace_project_dialogue(
    db: DatabaseManager,
    project_id: str,
    lines: Sequence[DialogueLine],
    total_duration_ms: int,
    *,
    master_audio_path: str | None = None,
    status: ProjectStatus = ProjectStatus.AUDIO_READY,
) -> list[DialogueLine]:
    """Atomically replace a project's dialogue and invalidate its scenes.

    The operation deletes previous lines (cascading their word timestamps),
    inserts the new ones, updates the project duration and status, and marks
    every scene as pending while dropping cached rendered clips. Any failure
    rolls the whole operation back.
    """

    if (
        isinstance(total_duration_ms, bool)
        or not isinstance(total_duration_ms, int)
        or total_duration_ms < 0
    ):
        raise ValueError("total_duration_ms debe ser un entero >= 0")

    prepared = list(lines)
    seen_indexes: set[int] = set()
    for line in prepared:
        if line.project_id != project_id:
            raise ValueError("todas las lineas deben pertenecer al project_id indicado")
        if line.audio_path is None or not line.audio_path.strip():
            raise ValueError("audio_path es obligatorio para persistir una linea de dialogo")
        if line.line_index in seen_indexes:
            raise ValueError(f"line_index duplicado: {line.line_index}")
        seen_indexes.add(line.line_index)

    resolved_status = ProjectStatus(status)
    timestamp = _utc_iso()

    with db.transaction() as conn:
        project_row = conn.execute(
            "SELECT 1 FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if project_row is None:
            raise ProjectNotFoundError(f"Proyecto no registrado: {project_id}")

        conn.execute("DELETE FROM dialogue_lines WHERE project_id = ?", (project_id,))
        inserted = [_insert_line(conn, line) for line in prepared]

        conn.execute(
            """
            UPDATE projects
               SET total_duration_ms = ?,
                   status = ?,
                   master_audio_path = COALESCE(?, master_audio_path),
                   updated_at = ?
             WHERE id = ?
            """,
            (total_duration_ms, resolved_status.value, master_audio_path, timestamp, project_id),
        )
        conn.execute(
            """
            UPDATE scenes
               SET validation_status = ?,
                   rendered_clip_path = NULL,
                   updated_at = ?
             WHERE project_id = ?
            """,
            (ValidationStatus.PENDING.value, timestamp, project_id),
        )

    return inserted


__all__ = [
    "DialogueLineNotFoundError",
    "DialogueRepository",
    "ProjectNotFoundError",
    "ProjectRepository",
    "RenderJobNotFoundError",
    "RenderJobRepository",
    "RepositoryError",
    "SceneNotFoundError",
    "SceneRepository",
    "VoiceNotFoundError",
    "VoiceRepository",
    "replace_project_dialogue",
]
