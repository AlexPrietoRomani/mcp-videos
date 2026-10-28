"""Unit tests for voice registration, dialogue synthesis and word alignment."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from mcp_videos.audio.aligner import (
    AlignmentError,
    TranscribedWord,
    WordAligner,
    interpolate_words,
    tokenize_words,
    validate_alignment,
)
from mcp_videos.audio.dialogue_synth import (
    DialogueSynthesisError,
    DialogueSynthesizer,
    DialogueTurnInput,
)
from mcp_videos.audio.ffmpeg_utils import AudioProcessingError, get_wav_duration_ms
from mcp_videos.audio.voice_registry import VoiceRegistryError, VoiceRegistryService
from mcp_videos.config import AppSettings
from mcp_videos.core.models import DialogueLine, Project, ProjectStatus
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectRepository,
    VoiceRepository,
)

SAMPLE_RATE = 24000
PROJECT_ID = "short-audio-unit"
CLONE_ID = "clon-ingeniero-es"


def write_tone(
    path: Path,
    duration_ms: int,
    *,
    frequency_hz: float = 180.0,
    amplitude: float = 0.3,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = round(SAMPLE_RATE * duration_ms / 1000)
    time_axis = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    signal = np.zeros(frames, dtype=np.float64)
    for harmonic in range(1, 6):
        signal += (amplitude / harmonic) * np.sin(2.0 * np.pi * frequency_hz * harmonic * time_axis)
    payload = np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(payload)
    return path


def wav_params(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getframerate(), handle.getnchannels(), handle.getsampwidth()


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
def registry(db: DatabaseManager, workspace: WorkspaceManager) -> VoiceRegistryService:
    return VoiceRegistryService(VoiceRepository(db), workspace)


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(_env_file=None)


def prepare_dialogue_project(
    db: DatabaseManager,
    registry: VoiceRegistryService,
    tmp_path: Path,
) -> None:
    ProjectRepository(db).create(Project(id=PROJECT_ID, title="Audio pipeline unit"))
    registry.seed_defaults()
    sample = write_tone(tmp_path / "reference.wav", 5000, frequency_hz=155.0)
    registry.register_clone(
        CLONE_ID,
        "Ingeniero clonado",
        sample,
        engine="f5",
        config={"ref_text": "Muestra del ingeniero"},
    )


class FixedDurationProvider:
    """Deterministic TTS provider used to test offsets without model downloads."""

    def __init__(self, durations_ms: dict[str, int]) -> None:
        self.durations_ms = durations_ms
        self.calls: list[tuple[str, str]] = []

    def synthesize_to_wav(self, text: str, voice: object, output_path: Path | str) -> Path:
        target = Path(output_path)
        duration_ms = self.durations_ms[text]
        frequency = 205.0 if getattr(voice, "mode", None) == "preset" else 150.0
        write_tone(target, duration_ms, frequency_hz=frequency, amplitude=0.08)
        self.calls.append((str(getattr(voice, "id", "")), text))
        return target


class TestVoiceRegistry:
    def test_clone_sample_shorter_than_three_seconds_is_rejected(
        self,
        registry: VoiceRegistryService,
        workspace: WorkspaceManager,
        tmp_path: Path,
    ) -> None:
        source = write_tone(tmp_path / "short.wav", 1000)

        with pytest.raises(AudioProcessingError, match="fuera de rango"):
            registry.register_clone("clon-corto", "Corto", source)

        assert registry.get_voice("clon-corto") is None
        assert not workspace.voice_sample_path("clon-corto").exists()

    def test_clone_sample_longer_than_thirty_seconds_is_rejected(
        self,
        registry: VoiceRegistryService,
        tmp_path: Path,
    ) -> None:
        source = write_tone(tmp_path / "long.wav", 35000)

        with pytest.raises(AudioProcessingError, match="fuera de rango"):
            registry.register_clone("clon-largo", "Largo", source)

        assert registry.get_voice("clon-largo") is None

    def test_five_second_clone_is_canonicalized_and_persisted(
        self,
        registry: VoiceRegistryService,
        workspace: WorkspaceManager,
        tmp_path: Path,
    ) -> None:
        source = write_tone(tmp_path / "reference.wav", 5000, frequency_hz=160.0)

        voice = registry.register_clone(
            CLONE_ID,
            "Ingeniero",
            source,
            engine="xtts",
            language="es",
            config={"ref_text": "Muestra de referencia"},
        )

        canonical = workspace.voice_sample_path(CLONE_ID)
        assert voice.mode.value == "clone"
        assert voice.sample_wav_path == str(canonical)
        assert canonical.is_file()
        assert wav_params(canonical) == (SAMPLE_RATE, 1, 2)
        assert 4950 <= get_wav_duration_ms(canonical) <= 5000
        assert registry.get_voice(CLONE_ID) == voice

    def test_preset_upsert_updates_existing_voice_without_duplicates(
        self,
        registry: VoiceRegistryService,
        db: DatabaseManager,
    ) -> None:
        first = registry.register_preset("preset-prueba-es", "Prueba")
        second = registry.register_preset(
            "preset-prueba-es",
            "Prueba actualizada",
            engine="piper",
            config={"speed": 1.2},
        )

        assert first.id == second.id
        assert second.name == "Prueba actualizada"
        assert second.sample_wav_path is None
        assert [voice.id for voice in VoiceRepository(db).list_all()] == ["preset-prueba-es"]

    def test_engine_and_mode_mismatch_is_rejected(
        self,
        registry: VoiceRegistryService,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(VoiceRegistryError, match="preset"):
            registry.register_preset("preset-f5", "F5", engine="f5")

        sample = write_tone(tmp_path / "reference.wav", 5000)
        with pytest.raises(VoiceRegistryError, match="clone"):
            registry.register_clone("clon-kokoro", "Kokoro", sample, engine="kokoro")

    def test_seed_defaults_cover_spanish_and_english_idempotently(
        self,
        registry: VoiceRegistryService,
        db: DatabaseManager,
    ) -> None:
        seeded = registry.seed_defaults()

        assert {voice.language.value for voice in seeded} >= {"es", "en"}
        assert len(seeded) == 4
        assert len(VoiceRepository(db).list_all()) == 4
        assert all(voice.mode.value == "preset" for voice in seeded)

        again = registry.seed_default_voices()
        assert [voice.id for voice in again] == [voice.id for voice in seeded]
        assert len(VoiceRepository(db).list_all()) == 4


class TestDialogueSynthesizer:
    def test_offsets_master_duration_and_audio_ready_state(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        registry: VoiceRegistryService,
        settings: AppSettings,
        tmp_path: Path,
    ) -> None:
        prepare_dialogue_project(db, registry, tmp_path)
        provider = FixedDurationProvider({"Hola desde la antena.": 800, "Fin del diálogo.": 1200})
        synthesizer = DialogueSynthesizer(
            db,
            workspace,
            settings,
            provider_factory=lambda voice, app_settings: provider,
        )

        result = synthesizer.synthesize_project(
            PROJECT_ID,
            [
                {
                    "voice_id": "preset-narrador-es",
                    "speaker_label": "Narrador",
                    "text": "Hola desde la antena.",
                    "pause_ms": 250,
                },
                DialogueTurnInput(
                    voice_id=CLONE_ID,
                    text_content="Fin del diálogo.",
                    pause_after_ms=100,
                ),
            ],
        )

        assert len(result) == 2
        assert result.lines[0].start_ms == 0
        assert result.lines[0].end_ms - result.lines[0].start_ms == 800
        assert result.lines[1].start_ms == 1050
        assert result.lines[1].end_ms - result.lines[1].start_ms == 1200
        assert abs(result.total_duration_ms - 2350) <= 5
        assert abs(get_wav_duration_ms(result.master_path) - result.total_duration_ms) <= 5
        assert all(line.end_ms <= result.total_duration_ms for line in result.lines)
        assert result.master_audio_path == "audio/dialogue_master.wav"
        assert result.master_path == workspace.dialogue_master_path(PROJECT_ID)
        assert wav_params(result.master_path) == (SAMPLE_RATE, 1, 2)
        assert provider.calls == [
            ("preset-narrador-es", "Hola desde la antena."),
            (CLONE_ID, "Fin del diálogo."),
        ]

        persisted = DialogueRepository(db).list_by_project(PROJECT_ID)
        assert [line.id for line in persisted] == [line.id for line in result.lines]
        assert [line.audio_path for line in persisted] == [
            "audio/line_00.wav",
            "audio/line_01.wav",
        ]
        assert all(
            workspace.resolve_project_path(PROJECT_ID, line.audio_path or "").is_file()
            for line in persisted
        )
        project = ProjectRepository(db).get_or_raise(PROJECT_ID)
        assert project.status is ProjectStatus.AUDIO_READY
        assert project.total_duration_ms == result.total_duration_ms
        assert project.master_audio_path == "audio/dialogue_master.wav"

    def test_empty_script_is_rejected(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        settings: AppSettings,
    ) -> None:
        synthesizer = DialogueSynthesizer(db, workspace, settings)

        with pytest.raises(DialogueSynthesisError, match="al menos un turno"):
            synthesizer.synthesize_project(PROJECT_ID, [])

    def test_negative_pause_and_unknown_turn_are_rejected(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        settings: AppSettings,
    ) -> None:
        synthesizer = DialogueSynthesizer(db, workspace, settings)

        with pytest.raises(DialogueSynthesisError, match="turno 0"):
            synthesizer.synthesize_project(
                PROJECT_ID,
                [
                    {
                        "voice_id": "preset-narrador-es",
                        "text": "Pausa inválida.",
                        "pause_after_ms": -100,
                    }
                ],
            )

        with pytest.raises(DialogueSynthesisError, match="turno 0"):
            synthesizer.synthesize_project(PROJECT_ID, [object()])

    def test_unknown_voice_is_reported(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        settings: AppSettings,
    ) -> None:
        ProjectRepository(db).create(Project(id=PROJECT_ID, title="Sin voz"))
        synthesizer = DialogueSynthesizer(db, workspace, settings)

        with pytest.raises(Exception, match="Voz no registrada"):
            synthesizer.synthesize_project(
                PROJECT_ID,
                [
                    {
                        "voice_id": "voz-inexistente",
                        "text": "Hola.",
                    }
                ],
            )


class TestWordAligner:
    def test_interpolation_covers_numbers_punctuation_and_short_words(self) -> None:
        text = "En 2006, la NASA-2 probó 3.14 y a; b."
        tokens = tokenize_words(text)

        words = interpolate_words(text, 100, 2100)

        assert [word.word for word in words] == tokens
        assert len(words) == len(tokens)
        assert all(word.confidence == 0.5 for word in words)
        line = DialogueLine(
            id=1,
            project_id=PROJECT_ID,
            line_index=0,
            voice_id="preset-narrador-es",
            speaker_label="Narrador",
            text_content=text,
            start_ms=100,
            end_ms=2100,
        )
        validate_alignment(words, line)

    def test_interpolation_rejects_a_span_too_short_for_every_word(self) -> None:
        with pytest.raises(AlignmentError, match="demasiado corto"):
            interpolate_words("una dos tres cuatro", 0, 3)

    def test_align_project_persists_full_coverage_and_cues(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        registry: VoiceRegistryService,
        settings: AppSettings,
        tmp_path: Path,
    ) -> None:
        prepare_dialogue_project(db, registry, tmp_path)
        provider = FixedDurationProvider({"Hola desde la antena.": 800, "Fin del diálogo.": 1200})
        synthesizer = DialogueSynthesizer(
            db,
            workspace,
            settings,
            provider_factory=lambda voice, app_settings: provider,
        )
        synthesized = synthesizer.synthesize_project(
            PROJECT_ID,
            [
                {"voice_id": "preset-narrador-es", "text": "Hola desde la antena."},
                {"voice_id": CLONE_ID, "text": "Fin del diálogo.", "pause_after_ms": 100},
            ],
        )
        aligner = WordAligner(db, workspace, settings, enable_whisper=False)

        result = aligner.align_project(PROJECT_ID)

        expected_tokens = sum(len(tokenize_words(line.text_content)) for line in synthesized.lines)
        assert len(result) == expected_tokens
        assert result.cues_path == workspace.project_dir(PROJECT_ID) / "cues.json"
        assert result.cues_path.is_file()
        repository = DialogueRepository(db)
        for line in synthesized.lines:
            persisted = repository.list_word_timestamps(line.id or 0)
            assert len(persisted) == len(tokenize_words(line.text_content))
            validate_alignment(persisted, line)
        payload = json.loads(result.cues_path.read_text(encoding="utf-8"))
        assert payload["project_id"] == PROJECT_ID
        assert payload["total_duration_ms"] == synthesized.total_duration_ms
        assert len(payload["lines"]) == len(synthesized.lines)
        assert len(payload["words"]) == expected_tokens
        assert payload["cues"] == payload["words"]

    def test_partial_asr_anchors_keep_confidence_and_fill_missing_words(self) -> None:
        text = "Hola mundo desde la antena"
        line = DialogueLine(
            id=1,
            project_id=PROJECT_ID,
            line_index=0,
            voice_id="preset-narrador-es",
            speaker_label="Narrador",
            text_content=text,
            start_ms=0,
            end_ms=2000,
        )
        transcribed = [
            TranscribedWord("Hola", 0, 150, 0.95),
            TranscribedWord("antena", 1000, 1300, 0.8),
        ]
        aligner = WordAligner(
            DatabaseManager(":memory:"),
            WorkspaceManager("/tmp/mcp-videos-unit-ws", "/tmp/mcp-videos-unit-voices"),
            enable_whisper=False,
        )

        words = aligner.align_line(line, transcribed=transcribed, persist=False)

        assert [word.word for word in words] == tokenize_words(text)
        assert words[0].confidence == 0.95
        assert words[-1].confidence == 0.8
        assert all(word.confidence == 0.5 for word in words[1:-1])
        validate_alignment(words, line)

    def test_align_project_without_dialogue_is_rejected(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        settings: AppSettings,
    ) -> None:
        ProjectRepository(db).create(Project(id=PROJECT_ID, title="Vacío"))
        aligner = WordAligner(db, workspace, settings, enable_whisper=False)

        with pytest.raises(AlignmentError, match="no tiene líneas"):
            aligner.align_project(PROJECT_ID)

    def test_repair_keeps_out_of_order_asr_inside_line_bounds(self) -> None:
        text = "uno dos tres"
        line = DialogueLine(
            id=1,
            project_id=PROJECT_ID,
            line_index=0,
            voice_id="preset-narrador-es",
            speaker_label="Narrador",
            text_content=text,
            start_ms=500,
            end_ms=1500,
        )
        transcribed = [
            TranscribedWord("uno", 2000, 2400, 0.9),
            TranscribedWord("dos", -50, 0, 0.9),
            TranscribedWord("tres", 400, 500, 0.9),
        ]
        aligner = WordAligner(
            DatabaseManager(":memory:"),
            WorkspaceManager("/tmp/mcp-videos-unit-ws", "/tmp/mcp-videos-unit-voices"),
            enable_whisper=False,
        )

        words = aligner.align_line(line, transcribed=transcribed, persist=False)

        validate_alignment(words, line)
        assert all(500 <= word.start_ms < word.end_ms <= 1500 for word in words)
