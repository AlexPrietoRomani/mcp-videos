"""Multi-voice dialogue synthesis, loudness normalization and master assembly."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from mcp_videos.audio.ffmpeg_utils import (
    TARGET_LUFS,
    concat_audio_segments_with_pauses,
    get_wav_duration_ms,
    normalize_audio_lufs,
)
from mcp_videos.audio.providers import TTSProvider, get_tts_provider
from mcp_videos.config import AppSettings
from mcp_videos.core.models import (
    PAUSE_AFTER_MAX_MS,
    SLUG_RE,
    DialogueLine,
    ProjectStatus,
    VoiceProfile,
)
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import VoiceRepository, replace_project_dialogue

ProviderFactory = Callable[[VoiceProfile, AppSettings], TTSProvider]


class DialogueSynthesisError(ValueError):
    """Raised when a dialogue script cannot be synthesized into a master track."""


class DialogueTurnInput(BaseModel):
    """A dialogue turn provided by an MCP tool, CLI or orchestrator."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    voice_id: str = Field(pattern=SLUG_RE.pattern)
    speaker_label: str | None = None
    text_content: str = Field(
        min_length=1,
        validation_alias=AliasChoices("text_content", "text"),
    )
    pause_after_ms: int = Field(
        default=150,
        ge=0,
        le=PAUSE_AFTER_MAX_MS,
        validation_alias=AliasChoices("pause_after_ms", "pause_ms"),
    )

    @field_validator("speaker_label")
    @classmethod
    def _validate_speaker_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        label = value.strip()
        if not label:
            raise ValueError("speaker_label no puede estar vacío")
        return label


@dataclass(frozen=True)
class DialogueSynthesisResult(Sequence[DialogueLine]):
    """Summary of a synthesized project dialogue and its master audio file."""

    project_id: str
    lines: tuple[DialogueLine, ...]
    total_duration_ms: int
    master_audio_relative_path: str
    master_path: Path

    @property
    def master_audio_path(self) -> str:
        """Relative master audio path as persisted in ``projects.master_audio_path``."""

        return self.master_audio_relative_path

    def __getitem__(self, index: int | slice) -> DialogueLine | tuple[DialogueLine, ...]:
        return self.lines[index]

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self) -> Iterator[DialogueLine]:
        return iter(self.lines)


def _coerce_turn(raw: object, index: int) -> DialogueTurnInput:
    if isinstance(raw, DialogueTurnInput):
        return raw
    if isinstance(raw, DialogueLine):
        return DialogueTurnInput(
            voice_id=raw.voice_id,
            speaker_label=raw.speaker_label,
            text_content=raw.text_content,
            pause_after_ms=max(0, raw.pause_after_ms),
        )
    if isinstance(raw, Mapping):
        try:
            return DialogueTurnInput.model_validate(dict(raw))
        except ValueError as exc:
            raise DialogueSynthesisError(f"turno {index} inválido: {exc}") from exc
    raise DialogueSynthesisError(
        f"turno {index} inválido: se esperaba un mapeo o DialogueTurnInput"
    )


class DialogueSynthesizer:
    """Execute Flow 2: per-line TTS, normalization and master track assembly."""

    def __init__(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        settings: AppSettings | None = None,
        *,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self.db = db
        self.workspace = workspace
        self.settings = settings or AppSettings()
        self.voices = VoiceRepository(db)
        self.provider_factory = provider_factory or get_tts_provider

    def synthesize_project(
        self,
        project_id: str,
        turns: Sequence[object],
    ) -> DialogueSynthesisResult:
        """Synthesize every turn, normalize to -16 LUFS and persist the dialogue."""

        prepared = [_coerce_turn(turn, index) for index, turn in enumerate(turns)]
        if not prepared:
            raise DialogueSynthesisError("el guion debe contener al menos un turno")
        self.workspace.ensure_project_dirs(project_id)

        lines: list[DialogueLine] = []
        normalized_paths: list[Path] = []
        cursor_ms = 0

        with tempfile.TemporaryDirectory(prefix="mcp-videos-dialogue-") as directory:
            raw_dir = Path(directory)
            for index, turn in enumerate(prepared):
                voice = self.voices.get_or_raise(turn.voice_id)
                provider = self.provider_factory(voice, self.settings)
                raw_path = raw_dir / f"line_{index:02d}_raw.wav"
                generated = provider.synthesize_to_wav(turn.text_content, voice, raw_path)
                normalized_path = self.workspace.line_audio_path(project_id, index)
                normalize_audio_lufs(generated, normalized_path, target_lufs=TARGET_LUFS)
                duration_ms = get_wav_duration_ms(normalized_path)
                if duration_ms <= 0:
                    raise DialogueSynthesisError(
                        f"el turno {index} generó un audio de duración inválida"
                    )
                start_ms = cursor_ms
                end_ms = start_ms + duration_ms
                lines.append(
                    DialogueLine(
                        project_id=project_id,
                        line_index=index,
                        voice_id=voice.id,
                        speaker_label=turn.speaker_label or voice.name,
                        text_content=turn.text_content,
                        pause_after_ms=turn.pause_after_ms,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        audio_path=self.workspace.relative_project_path(
                            project_id,
                            normalized_path,
                        ),
                    )
                )
                normalized_paths.append(normalized_path)
                cursor_ms = end_ms + turn.pause_after_ms

        master_path = self.workspace.dialogue_master_path(project_id)
        concat_audio_segments_with_pauses(
            normalized_paths,
            master_path,
            [line.pause_after_ms for line in lines],
        )
        master_duration_ms = get_wav_duration_ms(master_path)
        total_duration_ms = max(master_duration_ms, cursor_ms)
        master_relative_path = self.workspace.relative_project_path(project_id, master_path)
        persisted = replace_project_dialogue(
            self.db,
            project_id,
            lines,
            total_duration_ms,
            master_audio_path=master_relative_path,
            status=ProjectStatus.AUDIO_READY,
        )
        return DialogueSynthesisResult(
            project_id=project_id,
            lines=tuple(persisted),
            total_duration_ms=total_duration_ms,
            master_audio_relative_path=master_relative_path,
            master_path=master_path,
        )

    def synthesize_project_lines(
        self,
        project_id: str,
        lines: Sequence[object],
    ) -> DialogueSynthesisResult:
        """Alias matching the Flow 2 sequence naming used by the orchestrator."""

        return self.synthesize_project(project_id, lines)

    def synthesize(self, project_id: str, turns: Sequence[object]) -> DialogueSynthesisResult:
        """Alias for :meth:`synthesize_project`."""

        return self.synthesize_project(project_id, turns)


def synthesize_dialogue(
    db: DatabaseManager,
    workspace: WorkspaceManager,
    project_id: str,
    turns: Sequence[object],
    settings: AppSettings | None = None,
    *,
    provider_factory: ProviderFactory | None = None,
) -> DialogueSynthesisResult:
    """Convenience wrapper around :class:`DialogueSynthesizer`."""

    synthesizer = DialogueSynthesizer(
        db,
        workspace,
        settings,
        provider_factory=provider_factory,
    )
    return synthesizer.synthesize_project(project_id, turns)


__all__ = [
    "DialogueSynthesisError",
    "DialogueSynthesisResult",
    "DialogueSynthesizer",
    "DialogueTurnInput",
    "ProviderFactory",
    "synthesize_dialogue",
]
