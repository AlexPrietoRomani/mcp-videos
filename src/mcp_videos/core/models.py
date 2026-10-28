"""Domain entities, invariants and temporal calculations for mcp-videos."""

from __future__ import annotations

import itertools
import json
import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SUPPORTED_FPS: tuple[int, ...] = (24, 30, 60)
SupportedFps = Literal[24, 30, 60]

PAUSE_AFTER_MIN_MS = -250
PAUSE_AFTER_MAX_MS = 2000


class VoiceEngine(StrEnum):
    """Supported text-to-speech engines."""

    KOKORO = "kokoro"
    PIPER = "piper"
    XTTS = "xtts"
    F5 = "f5"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"


class VoiceMode(StrEnum):
    """How a voice profile is obtained."""

    PRESET = "preset"
    CLONE = "clone"


class LanguageCode(StrEnum):
    """Languages supported by the synthesis pipeline."""

    ES = "es"
    EN = "en"


class ProjectStatus(StrEnum):
    """Lifecycle states of a video project."""

    DRAFT = "draft"
    AUDIO_READY = "audio_ready"
    SCENES_READY = "scenes_ready"
    RENDERED = "rendered"
    FAILED = "failed"


class ValidationStatus(StrEnum):
    """Result of the browser-based scene validation gate."""

    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"


class RenderStatus(StrEnum):
    """Lifecycle states of a render job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DomainModel(BaseModel):
    """Base model that rejects unknown fields and validates assignments."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class VoiceProfile(DomainModel):
    """A preset or cloned voice registered in the local catalog."""

    id: str = Field(pattern=SLUG_RE.pattern)
    name: str = Field(min_length=1)
    engine: VoiceEngine
    mode: VoiceMode
    language: LanguageCode = LanguageCode.ES
    sample_wav_path: str | None = None
    config: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("config", "config_json"),
    )
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("config", mode="before")
    @classmethod
    def _parse_config_json(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("config_json debe contener JSON válido") from exc
            if not isinstance(parsed, dict):
                raise ValueError("config_json debe contener un objeto JSON")
            return parsed
        return value

    @model_validator(mode="after")
    def _validate_clone_sample(self) -> VoiceProfile:
        if self.mode is VoiceMode.CLONE and not self.sample_wav_path:
            raise ValueError("sample_wav_path es obligatorio cuando mode='clone'")
        return self


class Project(DomainModel):
    """A video project and its global timeline metadata."""

    id: str = Field(pattern=SLUG_RE.pattern)
    title: str = Field(min_length=1)
    description: str = ""
    width: int = Field(default=1080, gt=0)
    height: int = Field(default=1920, gt=0)
    fps: SupportedFps = 60
    total_duration_ms: int = Field(default=0, ge=0)
    master_audio_path: str | None = None
    status: ProjectStatus = ProjectStatus.DRAFT
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class DialogueLine(DomainModel):
    """A single dialogue turn with its absolute position in the master audio."""

    id: int | None = Field(default=None, ge=1)
    project_id: str = Field(pattern=SLUG_RE.pattern)
    line_index: int = Field(ge=0)
    voice_id: str = Field(pattern=SLUG_RE.pattern)
    speaker_label: str = Field(min_length=1)
    text_content: str = Field(min_length=1)
    pause_after_ms: int = Field(
        default=150,
        ge=PAUSE_AFTER_MIN_MS,
        le=PAUSE_AFTER_MAX_MS,
    )
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    audio_path: str | None = None

    @field_validator("text_content")
    @classmethod
    def _validate_text_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text_content no puede estar vacío")
        return value

    @model_validator(mode="after")
    def _validate_interval(self) -> DialogueLine:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms debe ser mayor que start_ms")
        return self


class WordTimestamp(DomainModel):
    """Word-level timing aligned to the project master audio."""

    id: int | None = Field(default=None, ge=1)
    line_id: int | None = Field(default=None, ge=1)
    word_index: int = Field(ge=0)
    word: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("word")
    @classmethod
    def _validate_word(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("word no puede estar vacía")
        return value

    @model_validator(mode="after")
    def _validate_interval(self) -> WordTimestamp:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms debe ser mayor que start_ms")
        return self


class Scene(DomainModel):
    """A self-contained HTML scene covering a global timeline interval."""

    id: int | None = Field(default=None, ge=1)
    project_id: str = Field(pattern=SLUG_RE.pattern)
    scene_index: int = Field(ge=0)
    title: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    html_path: str = Field(min_length=1)
    prompt_used: str = ""
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_errors: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("validation_errors", "validation_errors_json"),
    )
    rendered_clip_path: str | None = None
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("validation_errors", mode="before")
    @classmethod
    def _parse_validation_errors_json(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("validation_errors_json debe contener JSON válido") from exc
            if not isinstance(parsed, list):
                raise ValueError("validation_errors_json debe contener un array JSON")
            return parsed
        return value

    @model_validator(mode="after")
    def _validate_interval(self) -> Scene:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms debe ser mayor que start_ms")
        return self

    @property
    def validation_errors_json(self) -> str:
        """Serialize validation errors using the SQLite column representation."""

        return json.dumps(self.validation_errors, ensure_ascii=False)


class RenderJob(DomainModel):
    """A deterministic render task and its progress."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = Field(pattern=SLUG_RE.pattern)
    status: RenderStatus = RenderStatus.QUEUED
    total_frames: int = Field(gt=0)
    rendered_frames: int = Field(default=0, ge=0)
    burn_subtitles: bool = True
    output_mp4_path: str | None = None
    error_message: str | None = None
    started_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None

    @field_validator("id")
    @classmethod
    def _validate_uuid(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except ValueError as exc:
            raise ValueError("id debe ser un UUID válido") from exc
        return value

    @model_validator(mode="after")
    def _validate_progress(self) -> RenderJob:
        if self.rendered_frames > self.total_frames:
            raise ValueError("rendered_frames no puede superar total_frames")
        return self


class LineTiming(DomainModel):
    """Calculated absolute interval for one dialogue turn."""

    line_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_interval(self) -> LineTiming:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms debe ser mayor que start_ms")
        return self


class DialogueTimeline(DomainModel):
    """Accumulated dialogue intervals and total project duration."""

    lines: list[LineTiming] = Field(default_factory=list)
    total_duration_ms: int = Field(ge=0)


class TimelineCoverageError(ValueError):
    """Raised when scenes do not continuously cover the project timeline."""

    def __init__(
        self,
        message: str,
        *,
        gap_start_ms: int | None = None,
        gap_end_ms: int | None = None,
        overlap_start_ms: int | None = None,
        overlap_end_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.gap_start_ms = gap_start_ms
        self.gap_end_ms = gap_end_ms
        self.overlap_start_ms = overlap_start_ms
        self.overlap_end_ms = overlap_end_ms


def _validate_fps(fps: int) -> None:
    if isinstance(fps, bool) or not isinstance(fps, int) or fps not in SUPPORTED_FPS:
        raise ValueError(f"fps no soportado: {fps}; use uno de {SUPPORTED_FPS}")


def _validate_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} debe ser un entero")
    return value


def compute_dialogue_timeline(
    durations_ms: Sequence[int],
    pauses_after_ms: Sequence[int] | None = None,
) -> DialogueTimeline:
    """Accumulate line intervals from synthesized audio durations and pauses.

    Each line starts after the previous line end plus its pause after value.
    The final pause is included in ``total_duration_ms`` so the master audio and
    the scene coverage share the exact same end boundary.
    """

    durations = list(durations_ms)
    if not durations:
        raise ValueError("durations_ms debe contener al menos una línea")

    pauses = list(pauses_after_ms) if pauses_after_ms is not None else [0] * len(durations)
    if len(pauses) != len(durations):
        raise ValueError("durations_ms y pauses_after_ms deben tener la misma longitud")

    timings: list[LineTiming] = []
    cursor_ms = 0
    for index, (duration_ms, pause_ms) in enumerate(zip(durations, pauses, strict=True)):
        duration_ms = _validate_int(duration_ms, f"durations_ms[{index}]")
        pause_ms = _validate_int(pause_ms, f"pauses_after_ms[{index}]")
        if duration_ms <= 0:
            raise ValueError(f"la duración de la línea {index} debe ser mayor que 0 ms")
        if not PAUSE_AFTER_MIN_MS <= pause_ms <= PAUSE_AFTER_MAX_MS:
            raise ValueError(
                f"la pausa de la línea {index} debe estar entre "
                f"{PAUSE_AFTER_MIN_MS} y {PAUSE_AFTER_MAX_MS} ms"
            )

        start_ms = cursor_ms
        end_ms = start_ms + duration_ms
        if start_ms < 0:
            raise ValueError(f"la línea {index} inicia antes de 0 ms")

        timings.append(LineTiming(line_index=index, start_ms=start_ms, end_ms=end_ms))
        cursor_ms = end_ms + pause_ms

    if cursor_ms < 0:
        raise ValueError("el timeline total no puede ser negativo")

    return DialogueTimeline(lines=timings, total_duration_ms=cursor_ms)


def total_frames_for_duration(duration_ms: int, fps: int = 60) -> int:
    """Return the exact frame count for a duration using ceiling arithmetic."""

    _validate_fps(fps)
    duration_ms = _validate_int(duration_ms, "duration_ms")
    if duration_ms <= 0:
        raise ValueError("duration_ms debe ser > 0")
    return (duration_ms * fps + 999) // 1000


def frame_to_time_ms(frame_index: int, fps: int = 60) -> float:
    """Convert a discrete frame index to its virtual clock instant in ms."""

    _validate_fps(fps)
    frame_index = _validate_int(frame_index, "frame_index")
    if frame_index < 0:
        raise ValueError("frame_index debe ser >= 0")
    return (frame_index * 1000.0) / fps


def validate_scenes_timeline_coverage(
    scenes: Sequence[Scene],
    total_duration_ms: int,
) -> None:
    """Ensure scenes cover ``[0, total_duration_ms]`` without gaps or overlaps."""

    total_duration_ms = _validate_int(total_duration_ms, "total_duration_ms")
    if total_duration_ms <= 0:
        raise ValueError("total_duration_ms debe ser > 0")

    ordered = sorted(scenes, key=lambda scene: scene.scene_index)
    if [scene.scene_index for scene in ordered] != list(range(len(ordered))):
        raise ValueError("scene_index debe ser contiguo y comenzar en 0")

    if not ordered:
        raise TimelineCoverageError(
            f"hueco temporal detectado: [0, {total_duration_ms}]",
            gap_start_ms=0,
            gap_end_ms=total_duration_ms,
        )

    first = ordered[0]
    if first.start_ms > 0:
        raise TimelineCoverageError(
            f"hueco temporal detectado: [0, {first.start_ms}]",
            gap_start_ms=0,
            gap_end_ms=first.start_ms,
        )

    for previous, current in itertools.pairwise(ordered):
        if current.start_ms > previous.end_ms:
            raise TimelineCoverageError(
                f"hueco temporal detectado: [{previous.end_ms}, {current.start_ms}]",
                gap_start_ms=previous.end_ms,
                gap_end_ms=current.start_ms,
            )
        if current.start_ms < previous.end_ms:
            raise TimelineCoverageError(
                f"solapamiento temporal detectado: [{current.start_ms}, {previous.end_ms}]",
                overlap_start_ms=current.start_ms,
                overlap_end_ms=previous.end_ms,
            )

    last = ordered[-1]
    if last.end_ms < total_duration_ms:
        raise TimelineCoverageError(
            f"hueco temporal detectado: [{last.end_ms}, {total_duration_ms}]",
            gap_start_ms=last.end_ms,
            gap_end_ms=total_duration_ms,
        )
    if last.end_ms > total_duration_ms:
        raise TimelineCoverageError(
            f"solapamiento temporal detectado: [{total_duration_ms}, {last.end_ms}]",
            overlap_start_ms=total_duration_ms,
            overlap_end_ms=last.end_ms,
        )


__all__ = [
    "PAUSE_AFTER_MAX_MS",
    "PAUSE_AFTER_MIN_MS",
    "SLUG_RE",
    "SUPPORTED_FPS",
    "DialogueLine",
    "DialogueTimeline",
    "DomainModel",
    "LanguageCode",
    "LineTiming",
    "Project",
    "ProjectStatus",
    "RenderJob",
    "RenderStatus",
    "Scene",
    "SupportedFps",
    "TimelineCoverageError",
    "ValidationStatus",
    "VoiceEngine",
    "VoiceMode",
    "VoiceProfile",
    "WordTimestamp",
    "compute_dialogue_timeline",
    "frame_to_time_ms",
    "total_frames_for_duration",
    "validate_scenes_timeline_coverage",
]
