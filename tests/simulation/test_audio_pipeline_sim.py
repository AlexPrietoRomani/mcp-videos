"""User-simulation tests for the Flow 1, Flow 2 and Flow 3 audio pipeline."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from mcp_videos.audio.aligner import WordAligner, tokenize_words, validate_alignment
from mcp_videos.audio.dialogue_synth import DialogueSynthesizer
from mcp_videos.audio.ffmpeg_utils import get_wav_duration_ms
from mcp_videos.audio.voice_registry import VoiceRegistryService
from mcp_videos.config import AppSettings
from mcp_videos.core.models import Project, ProjectStatus
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectRepository,
    VoiceRepository,
)

PROJECT_ID = "short-antena-st5"
CLONE_ID = "clon-ingeniero-es"
SAMPLE_RATE = 24000


def render_reference_voice(path: Path, duration_ms: int = 6000) -> Path:
    frames = round(SAMPLE_RATE * duration_ms / 1000)
    time_axis = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    signal = np.zeros(frames, dtype=np.float64)
    for harmonic in range(1, 8):
        signal += (0.26 / harmonic) * np.sin(2.0 * np.pi * 158.0 * harmonic * time_axis)
    signal *= 0.55 + 0.45 * np.sin(2.0 * np.pi * 1.7 * time_axis)
    envelope = np.ones(frames, dtype=np.float64)
    ramp = round(SAMPLE_RATE * 0.05)
    envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
    envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(
            np.round(np.clip(signal * envelope, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        )
    return path


def wav_format(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getframerate(), handle.getnchannels(), handle.getsampwidth()


def probe_wav(path: Path) -> tuple[int, int, float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    duration_ms = float(payload["format"]["duration"]) * 1000
    return int(stream["sample_rate"]), int(stream["channels"]), duration_ms


@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(tmp_path / "data" / "mcp_videos.sqlite")
    manager.apply_migrations()
    return manager


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(
        workspaces_root=tmp_path / "workspaces",
        voices_root=tmp_path / "data" / "voices",
    )


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(_env_file=None)


def test_user_runs_full_voice_dialogue_and_alignment_pipeline(
    db: DatabaseManager,
    workspace: WorkspaceManager,
    settings: AppSettings,
) -> None:
    registry = VoiceRegistryService(VoiceRepository(db), workspace)
    seeded = registry.seed_defaults()

    assert {voice.language.value for voice in seeded} >= {"es", "en"}
    raw_sample = render_reference_voice(workspace.voices_root / "raw-ingreso.wav")
    clone = registry.register_clone(
        CLONE_ID,
        "Ingeniero clonado",
        raw_sample,
        engine="f5",
        language="es",
        config={"ref_text": "Muestra de voz del ingeniero", "speed": 1.0},
    )

    canonical_sample = workspace.voice_sample_path(CLONE_ID)
    assert clone.sample_wav_path == str(canonical_sample)
    assert wav_format(canonical_sample) == (SAMPLE_RATE, 1, 2)
    assert 5900 <= get_wav_duration_ms(canonical_sample) <= 6000

    ProjectRepository(db).create(
        Project(
            id=PROJECT_ID,
            title="La antena más rara de la NASA",
            description="Antena evolucionada por algoritmo genético.",
        )
    )
    turns = [
        {
            "voice_id": "preset-narrador-es",
            "speaker_label": "Narrador",
            "text": "En dos mil seis la NASA lanzó una antena que ningún humano diseñó.",
            "pause_after_ms": 250,
        },
        {
            "voice_id": CLONE_ID,
            "speaker_label": "Ingeniero",
            "text": "Un algoritmo evolutivo encontró una forma que nadie había imaginado.",
            "pause_after_ms": 180,
        },
        {
            "voice_id": "preset-narrador-es",
            "speaker_label": "Narrador",
            "text": "La antena sigue transmitiendo desde el espacio.",
            "pause_after_ms": 0,
        },
    ]
    synthesizer = DialogueSynthesizer(db, workspace, settings)

    result = synthesizer.synthesize_project(PROJECT_ID, turns)

    master = workspace.dialogue_master_path(PROJECT_ID)
    assert result.master_path == master
    assert result.master_audio_path == "audio/dialogue_master.wav"
    assert master.is_file()
    assert abs(get_wav_duration_ms(master) - result.total_duration_ms) <= 5
    assert all(line.end_ms <= result.total_duration_ms for line in result.lines)
    sample_rate, channels, probe_duration_ms = probe_wav(master)
    assert sample_rate == SAMPLE_RATE
    assert channels == 1
    assert abs(probe_duration_ms - result.total_duration_ms) <= 5
    assert wav_format(master) == (SAMPLE_RATE, 1, 2)

    audio_dir = workspace.audio_dir(PROJECT_ID)
    assert [line.audio_path for line in result.lines] == [
        "audio/line_00.wav",
        "audio/line_01.wav",
        "audio/line_02.wav",
    ]
    for line in result.lines:
        line_path = audio_dir / f"line_{line.line_index:02d}.wav"
        assert line_path.is_file()
        assert get_wav_duration_ms(line_path) == line.end_ms - line.start_ms
        assert wav_format(line_path) == (SAMPLE_RATE, 1, 2)

    project = ProjectRepository(db).get_or_raise(PROJECT_ID)
    assert project.status is ProjectStatus.AUDIO_READY
    assert project.total_duration_ms == result.total_duration_ms
    assert project.master_audio_path == "audio/dialogue_master.wav"
    persisted_lines = DialogueRepository(db).list_by_project(PROJECT_ID)
    assert [line.id for line in persisted_lines] == [line.id for line in result.lines]

    aligner = WordAligner(db, workspace, settings, enable_whisper=False)
    alignment = aligner.align_project(PROJECT_ID)

    expected_tokens = sum(len(tokenize_words(line.text_content)) for line in result.lines)
    assert len(alignment) == expected_tokens
    assert alignment.cues_path.is_file()
    words_by_line = DialogueRepository(db)
    for line in persisted_lines:
        words = words_by_line.list_word_timestamps(line.id or 0)
        assert [word.word for word in words] == tokenize_words(line.text_content)
        validate_alignment(words, line)

    window_words = words_by_line.word_timestamps_in_window(
        0,
        result.total_duration_ms,
        project_id=PROJECT_ID,
    )
    assert len(window_words) == expected_tokens
    assert all(
        result.lines[0].start_ms <= word.start_ms < word.end_ms <= result.total_duration_ms
        for word in window_words
    )
    cues = json.loads(alignment.cues_path.read_text(encoding="utf-8"))
    assert cues["project_id"] == PROJECT_ID
    assert cues["total_duration_ms"] == result.total_duration_ms
    assert len(cues["lines"]) == 3
    assert len(cues["words"]) == expected_tokens
    assert cues["cues"] == cues["words"]
    assert cues["lines"][1]["voice_id"] == CLONE_ID
