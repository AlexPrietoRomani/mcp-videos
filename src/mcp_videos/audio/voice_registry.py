"""Voice catalog registration, canonicalization and default seeding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp_videos.audio.ffmpeg_utils import (
    MAX_VOICE_SAMPLE_MS,
    MIN_VOICE_SAMPLE_MS,
    canonicalize_voice_sample,
)
from mcp_videos.core.models import LanguageCode, VoiceEngine, VoiceMode, VoiceProfile
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.repositories import VoiceRepository

PRESET_ENGINES = frozenset(
    {
        VoiceEngine.KOKORO,
        VoiceEngine.PIPER,
        VoiceEngine.ELEVENLABS,
        VoiceEngine.OPENAI,
    }
)
CLONE_ENGINES = frozenset({VoiceEngine.XTTS, VoiceEngine.F5})


class VoiceRegistryError(ValueError):
    """Raised when a voice cannot be registered in the local catalog."""


@dataclass(frozen=True)
class SeedVoice:
    """Definition of a default preset voice seeded into an empty catalog."""

    id: str
    name: str
    engine: VoiceEngine
    language: LanguageCode
    config: Mapping[str, Any] = field(default_factory=dict)


DEFAULT_SEED_VOICES: tuple[SeedVoice, ...] = (
    SeedVoice(
        id="preset-narrador-es",
        name="Narrador",
        engine=VoiceEngine.KOKORO,
        language=LanguageCode.ES,
        config={"voice_code": "ef_dora", "gender": "female", "speed": 1.0},
    ),
    SeedVoice(
        id="preset-ingeniero-es",
        name="Ingeniero",
        engine=VoiceEngine.KOKORO,
        language=LanguageCode.ES,
        config={"voice_code": "em_alex", "gender": "male", "speed": 1.0},
    ),
    SeedVoice(
        id="preset-narrator-en",
        name="Narrator",
        engine=VoiceEngine.KOKORO,
        language=LanguageCode.EN,
        config={"voice_code": "af_heart", "gender": "female", "speed": 1.0},
    ),
    SeedVoice(
        id="preset-host-en",
        name="Host",
        engine=VoiceEngine.KOKORO,
        language=LanguageCode.EN,
        config={"voice_code": "am_adam", "gender": "male", "speed": 1.0},
    ),
)


def _coerce_engine(value: str | VoiceEngine) -> VoiceEngine:
    try:
        return VoiceEngine(value)
    except ValueError as exc:
        raise VoiceRegistryError(f"motor TTS no soportado: {value!r}") from exc


def _coerce_language(value: str | LanguageCode) -> LanguageCode:
    try:
        return LanguageCode(value)
    except ValueError as exc:
        raise VoiceRegistryError(f"idioma no soportado: {value!r}") from exc


def _coerce_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise VoiceRegistryError("config debe ser un mapeo de parámetros del motor")
    return dict(value)


class VoiceRegistryService:
    """Register preset and cloned voices while enforcing Flow 1 invariants."""

    def __init__(self, repository: VoiceRepository, workspace: WorkspaceManager) -> None:
        self.repository = repository
        self.workspace = workspace

    def register_voice(
        self,
        voice: VoiceProfile,
        sample_path: Path | str | None = None,
    ) -> VoiceProfile:
        """Persist ``voice`` after validating its mode and canonicalizing clone samples."""

        engine = _coerce_engine(voice.engine)
        mode = VoiceMode(voice.mode)
        if mode is VoiceMode.PRESET:
            if engine not in PRESET_ENGINES:
                raise VoiceRegistryError(f"el motor {engine.value!r} no admite el modo preset")
            if sample_path is not None:
                raise VoiceRegistryError("una voz preset no requiere muestra de audio")
            return self.repository.upsert(voice.model_copy(update={"sample_wav_path": None}))

        if engine not in CLONE_ENGINES:
            raise VoiceRegistryError(f"el motor {engine.value!r} no admite el modo clone")
        source = sample_path if sample_path is not None else voice.sample_wav_path
        if source is None or not str(source).strip():
            raise VoiceRegistryError("una voz clonada requiere un archivo de muestra")
        self.workspace.ensure_voices_dir()
        target = self.workspace.voice_sample_path(voice.id)
        canonicalize_voice_sample(
            source,
            target,
            min_duration_ms=MIN_VOICE_SAMPLE_MS,
            max_duration_ms=MAX_VOICE_SAMPLE_MS,
        )
        return self.repository.upsert(voice.model_copy(update={"sample_wav_path": str(target)}))

    def register_preset(
        self,
        voice_id: str,
        name: str,
        *,
        engine: str | VoiceEngine = VoiceEngine.KOKORO,
        language: str | LanguageCode = LanguageCode.ES,
        config: Mapping[str, Any] | None = None,
    ) -> VoiceProfile:
        """Register or update a preset voice backed by a local or cloud engine."""

        voice = VoiceProfile(
            id=voice_id,
            name=name,
            engine=_coerce_engine(engine),
            mode=VoiceMode.PRESET,
            language=_coerce_language(language),
            config=_coerce_config(config),
        )
        return self.register_voice(voice)

    def register_clone(
        self,
        voice_id: str,
        name: str,
        sample_path: Path | str,
        *,
        engine: str | VoiceEngine = VoiceEngine.F5,
        language: str | LanguageCode = LanguageCode.ES,
        config: Mapping[str, Any] | None = None,
    ) -> VoiceProfile:
        """Register or update a cloned voice from a 3-30 second audio sample."""

        voice = VoiceProfile(
            id=voice_id,
            name=name,
            engine=_coerce_engine(engine),
            mode=VoiceMode.CLONE,
            language=_coerce_language(language),
            sample_wav_path=str(sample_path),
            config=_coerce_config(config),
        )
        return self.register_voice(voice, sample_path=sample_path)

    def seed_defaults(self, *, overwrite: bool = False) -> list[VoiceProfile]:
        """Seed the default Spanish and English preset catalog idempotently."""

        seeded: list[VoiceProfile] = []
        for seed in DEFAULT_SEED_VOICES:
            existing = self.repository.get(seed.id)
            if existing is not None and not overwrite:
                seeded.append(existing)
                continue
            seeded.append(
                self.register_preset(
                    seed.id,
                    seed.name,
                    engine=seed.engine,
                    language=seed.language,
                    config=seed.config,
                )
            )
        return seeded

    def seed_default_voices(self, *, overwrite: bool = False) -> list[VoiceProfile]:
        """Alias for :meth:`seed_defaults` used by higher-level services."""

        return self.seed_defaults(overwrite=overwrite)

    def ensure_default_voices(self, *, overwrite: bool = False) -> list[VoiceProfile]:
        """Alias for :meth:`seed_defaults` used by orchestrators."""

        return self.seed_defaults(overwrite=overwrite)

    def get_voice(self, voice_id: str) -> VoiceProfile | None:
        """Return one registered voice or ``None`` when it is missing."""

        return self.repository.get(voice_id)

    def list_voices(self) -> list[VoiceProfile]:
        """Return the canonical voice catalog ordered by id."""

        return self.repository.list_all()

    def voice_sample_path(self, voice_id: str) -> Path:
        """Return the canonical sample path for a cloned voice."""

        return self.workspace.voice_sample_path(voice_id)


__all__ = [
    "CLONE_ENGINES",
    "DEFAULT_SEED_VOICES",
    "PRESET_ENGINES",
    "SeedVoice",
    "VoiceRegistryError",
    "VoiceRegistryService",
]
