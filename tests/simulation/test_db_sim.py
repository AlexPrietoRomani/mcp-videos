"""User-simulation tests for the Golden project persistence cycle."""

from __future__ import annotations

import sqlite3
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
    compute_dialogue_timeline,
    total_frames_for_duration,
)
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectRepository,
    RenderJobRepository,
    SceneRepository,
    VoiceRepository,
    replace_project_dialogue,
)

PROJECT_ID = "short-antena-st5"
NARRATOR_ID = "preset-narrador-es"
ENGINEER_ID = "clon-ingeniero-es"
TOTAL_DURATION_MS = 4500


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "mcp_videos.sqlite"


@pytest.fixture
def db(db_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path)
    manager.apply_migrations()
    return manager


def golden_voices() -> list[VoiceProfile]:
    return [
        VoiceProfile(
            id=NARRATOR_ID,
            name="Narrador",
            engine="kokoro",
            mode="preset",
            language="es",
            config={"voice_code": "ef_dora", "speed": 1.0},
        ),
        VoiceProfile(
            id=ENGINEER_ID,
            name="Ingeniero clonado",
            engine="f5",
            mode="clone",
            language="es",
            sample_wav_path=f"data/voices/{ENGINEER_ID}.wav",
            config={"ref_text": "Muestra de voz del ingeniero", "speed": 1.0},
        ),
    ]


def golden_project() -> Project:
    return Project(
        id=PROJECT_ID,
        title="La antena mas rara de la NASA",
        description="Antena evolucionada por algoritmo genetico.",
        width=1080,
        height=1920,
        fps=60,
        status="draft",
    )


def golden_lines() -> list[DialogueLine]:
    timeline = compute_dialogue_timeline(
        durations_ms=[2000, 2200],
        pauses_after_ms=[200, 100],
    )
    assert timeline.total_duration_ms == TOTAL_DURATION_MS
    texts = [
        "En dos mil seis la NASA lanzó una antena que ningún humano diseñó.",
        "Un algoritmo evolutivo probó miles de formas hasta hallar la óptima.",
    ]
    return [
        DialogueLine(
            project_id=PROJECT_ID,
            line_index=index,
            voice_id=voice_id,
            speaker_label=label,
            text_content=text,
            pause_after_ms=pause,
            start_ms=timing.start_ms,
            end_ms=timing.end_ms,
            audio_path=f"audio/line_{index:02d}.wav",
        )
        for index, (timing, text, voice_id, label, pause) in enumerate(
            zip(
                timeline.lines,
                texts,
                [NARRATOR_ID, ENGINEER_ID],
                ["Narrador", "Ingeniero"],
                [200, 100],
                strict=True,
            )
        )
    ]


def fifteen_words(line_zero_id: int, line_one_id: int) -> list[WordTimestamp]:
    first = [
        WordTimestamp(
            line_id=line_zero_id,
            word_index=index,
            word=f"palabra{index}",
            start_ms=index * 240,
            end_ms=index * 240 + 200,
            confidence=0.97,
        )
        for index in range(8)
    ]
    second = [
        WordTimestamp(
            line_id=line_one_id,
            word_index=index,
            word=f"termino{index}",
            start_ms=2200 + index * 300,
            end_ms=2200 + index * 300 + 250,
            confidence=0.93,
        )
        for index in range(7)
    ]
    return first + second


def golden_scene() -> Scene:
    return Scene(
        project_id=PROJECT_ID,
        scene_index=0,
        title="Hook Antena Evolucionada",
        start_ms=0,
        end_ms=TOTAL_DURATION_MS,
        html_path="scenes/scene_00.html",
        prompt_used="Tarjeta SVG animada con progreso timeMs / 4500",
        validation_status="valid",
        validation_errors=[],
        rendered_clip_path="output/scene_00.mp4",
    )


def test_golden_project_full_persistence_cycle(db: DatabaseManager, db_path: Path) -> None:
    voices = VoiceRepository(db)
    projects = ProjectRepository(db)
    dialogues = DialogueRepository(db)
    scenes = SceneRepository(db)
    jobs = RenderJobRepository(db)

    projects.create(golden_project())
    for voice in golden_voices():
        voices.upsert(voice)

    persisted_lines = replace_project_dialogue(
        db,
        PROJECT_ID,
        golden_lines(),
        TOTAL_DURATION_MS,
        master_audio_path="audio/dialogue_master.wav",
    )
    assert len(persisted_lines) == 2
    assert all(line.id is not None for line in persisted_lines)

    line_zero_id = persisted_lines[0].id
    line_one_id = persisted_lines[1].id
    assert line_zero_id is not None
    assert line_one_id is not None
    dialogues.insert_word_timestamps(line_zero_id, fifteen_words(line_zero_id, line_one_id)[:8])
    dialogues.insert_word_timestamps(line_one_id, fifteen_words(line_zero_id, line_one_id)[8:])
    assert len(dialogues.list_word_timestamps(line_zero_id)) == 8
    assert len(dialogues.list_word_timestamps(line_one_id)) == 7

    scenes.upsert(golden_scene())
    projects.update_status(PROJECT_ID, ProjectStatus.SCENES_READY)
    total_frames = total_frames_for_duration(
        projects.get_or_raise(PROJECT_ID).total_duration_ms,
        60,
    )
    assert total_frames == 270
    job = jobs.create(
        RenderJob(
            project_id=PROJECT_ID,
            total_frames=total_frames,
            output_mp4_path="output/final.mp4",
        )
    )

    reopened = DatabaseManager(db_path)
    assert reopened.apply_migrations() == []

    persisted_project = ProjectRepository(reopened).get_or_raise(PROJECT_ID)
    assert persisted_project.total_duration_ms == TOTAL_DURATION_MS
    assert persisted_project.master_audio_path == "audio/dialogue_master.wav"
    assert persisted_project.status is ProjectStatus.SCENES_READY
    assert [voice.id for voice in VoiceRepository(reopened).list_all()] == [
        ENGINEER_ID,
        NARRATOR_ID,
    ]
    assert [
        line.line_index for line in DialogueRepository(reopened).list_by_project(PROJECT_ID)
    ] == [
        0,
        1,
    ]
    all_words = DialogueRepository(reopened).word_timestamps_in_window(
        0,
        TOTAL_DURATION_MS,
        project_id=PROJECT_ID,
    )
    assert len(all_words) == 15
    persisted_scene = SceneRepository(reopened).get_or_raise(PROJECT_ID, 0)
    assert persisted_scene.validation_status is ValidationStatus.VALID
    assert RenderJobRepository(reopened).get_or_raise(job.id).total_frames == 270

    edited_lines = [
        DialogueLine(
            project_id=PROJECT_ID,
            line_index=0,
            voice_id=NARRATOR_ID,
            speaker_label="Narrador",
            text_content="La NASA necesitaba una antena imposible de diseñar a mano.",
            pause_after_ms=150,
            start_ms=0,
            end_ms=1800,
            audio_path="audio/line_00.wav",
        ),
        DialogueLine(
            project_id=PROJECT_ID,
            line_index=1,
            voice_id=ENGINEER_ID,
            speaker_label="Ingeniero",
            text_content="Entonces dejó que la evolución hiciera el trabajo.",
            pause_after_ms=100,
            start_ms=1950,
            end_ms=4050,
            audio_path="audio/line_01.wav",
        ),
    ]
    edited_total = 4150
    replaced = replace_project_dialogue(
        db,
        PROJECT_ID,
        edited_lines,
        edited_total,
    )
    assert [line.text_content for line in replaced] == [
        edited_lines[0].text_content,
        edited_lines[1].text_content,
    ]
    assert DialogueRepository(db).list_word_timestamps(line_zero_id) == []
    assert DialogueRepository(db).list_word_timestamps(line_one_id) == []
    assert DialogueRepository(db).word_timestamps_in_window(0, 5000, project_id=PROJECT_ID) == []

    invalidated_scene = SceneRepository(db).get_or_raise(PROJECT_ID, 0)
    assert invalidated_scene.validation_status is ValidationStatus.PENDING
    assert invalidated_scene.rendered_clip_path is None
    updated_project = ProjectRepository(db).get_or_raise(PROJECT_ID)
    assert updated_project.total_duration_ms == edited_total
    assert updated_project.status is ProjectStatus.AUDIO_READY

    new_line_id = replaced[0].id
    assert new_line_id is not None
    DialogueRepository(db).insert_word_timestamps(
        new_line_id,
        [
            WordTimestamp(
                line_id=new_line_id,
                word_index=0,
                word="La",
                start_ms=0,
                end_ms=180,
                confidence=1.0,
            )
        ],
    )
    window = DialogueRepository(db).word_timestamps_in_window(0, 500, project_id=PROJECT_ID)
    assert [word.word for word in window] == ["La"]

    projects.update_status(PROJECT_ID, ProjectStatus.SCENES_READY)
    runtime = jobs.update_progress(job.id, 100, status=RenderStatus.RUNNING)
    assert runtime.status is RenderStatus.RUNNING
    completed = jobs.complete(job.id, "output/final.mp4")
    assert completed.status is RenderStatus.COMPLETED
    assert completed.rendered_frames == 270

    assert ProjectRepository(db).delete(PROJECT_ID) is True
    with db.connection() as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("dialogue_lines", "word_timestamps", "scenes", "render_jobs")
        }
        remaining_voices = conn.execute("SELECT COUNT(*) FROM voices").fetchone()[0]
    assert counts == {
        "dialogue_lines": 0,
        "word_timestamps": 0,
        "scenes": 0,
        "render_jobs": 0,
    }
    assert remaining_voices == 2


def test_invalid_user_payloads_raise_integrity_errors(db: DatabaseManager) -> None:
    ProjectRepository(db).create(golden_project())
    VoiceRepository(db).upsert(golden_voices()[0])

    invalid_payloads = [
        (
            "fps no soportado",
            """
            INSERT INTO projects (id, title, fps, created_at, updated_at)
            VALUES ('proyecto-fps', 'FPS', 45, '2026-10-28T12:00:00Z', '2026-10-28T12:00:00Z')
            """,
            (),
        ),
        (
            "fin anterior al inicio",
            """
            INSERT INTO dialogue_lines (
                project_id, line_index, voice_id, speaker_label, text_content,
                pause_after_ms, start_ms, end_ms, audio_path
            ) VALUES ('short-antena-st5', 0, 'preset-narrador-es', 'Narrador',
                      'Texto', 150, 1000, 500, 'audio/line_00.wav')
            """,
            (),
        ),
        (
            "proyecto inexistente",
            """
            INSERT INTO dialogue_lines (
                project_id, line_index, voice_id, speaker_label, text_content,
                pause_after_ms, start_ms, end_ms, audio_path
            ) VALUES ('proyecto-fantasma', 0, 'preset-narrador-es', 'Narrador',
                      'Texto', 150, 0, 1000, 'audio/line_00.wav')
            """,
            (),
        ),
        (
            "voz clonada sin muestra",
            """
            INSERT INTO voices (id, name, engine, mode, language)
            VALUES ('clon-sin-muestra', 'Clon', 'f5', 'clone', 'es')
            """,
            (),
        ),
    ]

    for label, statement, params in invalid_payloads:
        with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
            conn.execute(statement, params)
        assert label


def test_atomic_dialogue_replacement_keeps_previous_state_on_failure(
    db: DatabaseManager,
) -> None:
    ProjectRepository(db).create(golden_project())
    for voice in golden_voices():
        VoiceRepository(db).upsert(voice)
    original_lines = replace_project_dialogue(db, PROJECT_ID, golden_lines(), TOTAL_DURATION_MS)
    SceneRepository(db).upsert(golden_scene())

    invalid_line = DialogueLine(
        project_id=PROJECT_ID,
        line_index=1,
        voice_id="voz-eliminada",
        speaker_label="Fantasma",
        text_content="Esta linea no debe persistirse.",
        pause_after_ms=150,
        start_ms=1000,
        end_ms=2000,
        audio_path="audio/line_01.wav",
    )

    with pytest.raises(sqlite3.IntegrityError):
        replace_project_dialogue(db, PROJECT_ID, [golden_lines()[0], invalid_line], 3000)

    assert DialogueRepository(db).list_by_project(PROJECT_ID) == original_lines
    project = ProjectRepository(db).get_or_raise(PROJECT_ID)
    assert project.total_duration_ms == TOTAL_DURATION_MS
    assert project.status is ProjectStatus.AUDIO_READY
    scene = SceneRepository(db).get_or_raise(PROJECT_ID, 0)
    assert scene.validation_status is ValidationStatus.VALID
    assert scene.rendered_clip_path == "output/scene_00.mp4"
