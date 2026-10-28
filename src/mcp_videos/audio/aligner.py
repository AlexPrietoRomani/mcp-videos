"""Word-level alignment with faster-whisper and deterministic interpolation fallback."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from mcp_videos.config import AppSettings
from mcp_videos.core.models import DialogueLine, VoiceProfile, WordTimestamp
from mcp_videos.core.workspace import PathTraversalError, WorkspaceError, WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectRepository,
    VoiceRepository,
)

WORD_PATTERN = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", flags=re.UNICODE)
VOWEL_PATTERN = re.compile(r"[aeiouáéíóúü]+", flags=re.IGNORECASE)
INTERPOLATED_CONFIDENCE = 0.5
CUES_FILENAME = "cues.json"


class AlignmentError(ValueError):
    """Raised when dialogue lines cannot be aligned into valid word timestamps."""


@dataclass(frozen=True)
class TranscribedWord:
    """One word returned by an ASR backend, relative to the line audio."""

    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0


Transcriber = Callable[[Path, str | None], Sequence[TranscribedWord] | None]
ModelFactory = Callable[[], Any]


@dataclass(frozen=True)
class AlignmentResult(Sequence[WordTimestamp]):
    """Summary of a project alignment and its exported ``cues.json``."""

    project_id: str
    lines: tuple[DialogueLine, ...]
    words_by_line: Mapping[int, tuple[WordTimestamp, ...]]
    cues_path: Path
    total_duration_ms: int

    @property
    def words(self) -> list[WordTimestamp]:
        """Flatten every word timestamp ordered by line and word index."""

        return [word for line in self.lines for word in self.words_by_line.get(line.id or 0, ())]

    def __getitem__(self, index: int | slice) -> WordTimestamp | list[WordTimestamp]:
        return self.words[index]

    def __len__(self) -> int:
        return sum(len(words) for words in self.words_by_line.values())

    def __iter__(self) -> Iterator[WordTimestamp]:
        return iter(self.words)


def tokenize_words(text: str) -> list[str]:
    """Split a dialogue line into pronounceable words without punctuation tokens."""

    if not isinstance(text, str):
        raise TypeError("text debe ser una cadena")
    tokens = WORD_PATTERN.findall(text)
    if tokens:
        return tokens
    stripped = text.strip()
    return [stripped] if stripped else []


def _normalize_word(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return "".join(char for char in without_accents if char.isalnum())


def _word_weight(word: str) -> float:
    normalized = _normalize_word(word)
    if not normalized:
        return 1.0
    syllables = max(1, len(VOWEL_PATTERN.findall(normalized)))
    return max(1.0, 0.75 * len(normalized) + 0.5 * syllables)


def _validate_span(start_ms: int, end_ms: int, count: int) -> tuple[int, int]:
    if isinstance(start_ms, bool) or not isinstance(start_ms, int):
        raise AlignmentError("start_ms debe ser un entero")
    if isinstance(end_ms, bool) or not isinstance(end_ms, int):
        raise AlignmentError("end_ms debe ser un entero")
    if start_ms < 0 or end_ms <= start_ms:
        raise AlignmentError("el intervalo debe cumplir 0 <= start_ms < end_ms")
    if count > end_ms - start_ms:
        raise AlignmentError(
            "el intervalo es demasiado corto para garantizar un timestamp por palabra"
        )
    return start_ms, end_ms


def _allocate_durations(weights: Sequence[float], span_ms: int) -> list[int]:
    count = len(weights)
    if count == 0:
        return []
    if span_ms < count:
        raise AlignmentError(
            "el intervalo es demasiado corto para garantizar un timestamp por palabra"
        )
    total_weight = sum(weights)
    remaining = span_ms - count
    exact = [remaining * (weight / total_weight) for weight in weights]
    floors = [math.floor(value) for value in exact]
    leftover = remaining - sum(floors)
    order = sorted(
        range(count),
        key=lambda index: (exact[index] - floors[index], -index),
        reverse=True,
    )
    for index in order[:leftover]:
        floors[index] += 1
    return [duration + 1 for duration in floors]


def interpolate_words(
    text: str | Sequence[str],
    start_ms: int,
    end_ms: int,
) -> list[WordTimestamp]:
    """Distribute words proportionally by length and syllables inside a line span."""

    tokens = tokenize_words(text) if isinstance(text, str) else list(text)
    if not tokens:
        return []
    start_ms, end_ms = _validate_span(start_ms, end_ms, len(tokens))
    durations = _allocate_durations([_word_weight(token) for token in tokens], end_ms - start_ms)
    timestamps: list[WordTimestamp] = []
    cursor = start_ms
    for index, (token, duration) in enumerate(zip(tokens, durations, strict=True)):
        word_end = cursor + duration
        timestamps.append(
            WordTimestamp(
                word_index=index,
                word=token,
                start_ms=cursor,
                end_ms=word_end,
                confidence=INTERPOLATED_CONFIDENCE,
            )
        )
        cursor = word_end
    return timestamps


def _clamp_interval(start_ms: int, end_ms: int, line: DialogueLine) -> tuple[int, int]:
    clamped_start = min(max(start_ms, line.start_ms), line.end_ms - 1)
    clamped_end = min(max(end_ms, clamped_start + 1), line.end_ms)
    return clamped_start, clamped_end


def _match_anchors(
    tokens: Sequence[str],
    transcribed: Sequence[TranscribedWord],
) -> dict[int, TranscribedWord]:
    script_words = [_normalize_word(token) for token in tokens]
    asr_words = [_normalize_word(word.word) for word in transcribed]
    matcher = SequenceMatcher(a=script_words, b=asr_words, autojunk=False)
    anchors: dict[int, TranscribedWord] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            anchors[block.a + offset] = transcribed[block.b + offset]
    return anchors


def _candidate_intervals(
    tokens: Sequence[str],
    transcribed: Sequence[TranscribedWord],
    line: DialogueLine,
) -> list[tuple[int, int, float] | None]:
    candidates: list[tuple[int, int, float] | None] = [None] * len(tokens)
    for index, word in _match_anchors(tokens, transcribed).items():
        absolute_start = line.start_ms + word.start_ms
        absolute_end = line.start_ms + word.end_ms
        start_ms, end_ms = _clamp_interval(absolute_start, absolute_end, line)
        candidates[index] = (start_ms, end_ms, min(1.0, max(0.0, word.confidence)))

    index = 0
    while index < len(candidates):
        if candidates[index] is not None:
            index += 1
            continue
        run_end = index
        while run_end < len(candidates) and candidates[run_end] is None:
            run_end += 1
        previous = candidates[index - 1] if index > 0 else None
        following = candidates[run_end] if run_end < len(candidates) else None
        low = previous[1] if previous is not None else line.start_ms
        high = following[0] if following is not None else line.end_ms
        count = run_end - index
        if high - low < count:
            high = min(line.end_ms, low + count)
        if high - low < count:
            low = max(line.start_ms, high - count)
        weights = [_word_weight(token) for token in tokens[index:run_end]]
        durations = _allocate_durations(weights, high - low)
        cursor = low
        for offset, duration in enumerate(durations):
            candidates[index + offset] = (cursor, cursor + duration, INTERPOLATED_CONFIDENCE)
            cursor += duration
        index = run_end
    return candidates


def _repair_intervals(
    candidates: Sequence[tuple[int, int, float] | None],
    tokens: Sequence[str],
    line: DialogueLine,
) -> list[WordTimestamp]:
    count = len(candidates)
    timestamps: list[WordTimestamp] = []
    cursor = line.start_ms
    for index, (token, candidate) in enumerate(zip(tokens, candidates, strict=True)):
        remaining = count - 1 - index
        max_end = line.end_ms - remaining
        if candidate is None:
            start_ms = cursor
            end_ms = start_ms + 1
            confidence = INTERPOLATED_CONFIDENCE
        else:
            start_ms = max(cursor, min(candidate[0], max_end - 1))
            end_ms = max(start_ms + 1, min(candidate[1], max_end))
            confidence = candidate[2]
        timestamps.append(
            WordTimestamp(
                word_index=index,
                word=token,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=confidence,
            )
        )
        cursor = end_ms
    return timestamps


def validate_alignment(words: Sequence[WordTimestamp], line: DialogueLine) -> None:
    """Assert full coverage and strict monotonicity inside a dialogue line."""

    if not words:
        raise AlignmentError(f"la línea {line.line_index} no produjo palabras alineadas")
    previous_end = line.start_ms
    for index, word in enumerate(words):
        if word.word_index != index:
            raise AlignmentError("word_index debe ser contiguo y comenzar en 0")
        if not line.start_ms <= word.start_ms < word.end_ms <= line.end_ms:
            raise AlignmentError(
                f"la palabra {word.word!r} queda fuera de [{line.start_ms}, {line.end_ms}]"
            )
        if word.start_ms < previous_end:
            raise AlignmentError(f"monotonía rota en la palabra {word.word!r}")
        previous_end = word.end_ms


class WordAligner:
    """Execute Flow 3: transcribe words and guarantee timestamps for every script token."""

    def __init__(
        self,
        db: DatabaseManager,
        workspace: WorkspaceManager,
        settings: AppSettings | None = None,
        *,
        enable_whisper: bool = True,
        allow_model_download: bool = False,
        transcriber: Transcriber | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.db = db
        self.workspace = workspace
        self.settings = settings or AppSettings()
        self.enable_whisper = enable_whisper
        self.allow_model_download = allow_model_download
        self.transcriber = transcriber
        self.model_factory = model_factory
        self.dialogues = DialogueRepository(db)
        self.projects = ProjectRepository(db)
        self.voices = VoiceRepository(db)
        self._whisper_model: Any | None = None
        self._whisper_unavailable = False

    def cues_path(self, project_id: str) -> Path:
        """Return the exported cues path inside the project workspace."""

        return self.workspace.project_dir(project_id) / CUES_FILENAME

    def align_project(self, project_id: str) -> AlignmentResult:
        """Align every dialogue line, persist word timestamps and export ``cues.json``."""

        lines = self.dialogues.list_by_project(project_id)
        if not lines:
            raise AlignmentError(f"el proyecto {project_id!r} no tiene líneas de diálogo")
        self.workspace.ensure_project_dirs(project_id)

        words_by_line: dict[int, tuple[WordTimestamp, ...]] = {}
        for line in lines:
            if line.id is None:
                raise AlignmentError(f"la línea {line.line_index} no tiene id persistido")
            aligned = self.align_line(line, persist=True)
            words_by_line[line.id] = tuple(aligned)

        project = self.projects.get(project_id)
        total_duration_ms = (
            project.total_duration_ms
            if project is not None
            else max((line.end_ms for line in lines), default=0)
        )
        cues_path = self._write_cues(project_id, lines, words_by_line, total_duration_ms)
        return AlignmentResult(
            project_id=project_id,
            lines=tuple(lines),
            words_by_line=words_by_line,
            cues_path=cues_path,
            total_duration_ms=total_duration_ms,
        )

    def align_line(
        self,
        line: DialogueLine,
        *,
        transcribed: Sequence[TranscribedWord] | None = None,
        persist: bool = True,
    ) -> list[WordTimestamp]:
        """Align one line and optionally replace its persisted word timestamps."""

        tokens = tokenize_words(line.text_content)
        if transcribed is None:
            audio_path = self._line_audio_path(line)
            if audio_path is not None:
                transcribed = self._transcribe_line(audio_path, line)
        words = self._build_words(line, tokens, transcribed)
        validate_alignment(words, line)
        if persist:
            if line.id is None:
                raise AlignmentError(f"la línea {line.line_index} no tiene id persistido")
            return self.dialogues.replace_word_timestamps(line.id, words)
        return words

    def _build_words(
        self,
        line: DialogueLine,
        tokens: Sequence[str],
        transcribed: Sequence[TranscribedWord] | None,
    ) -> list[WordTimestamp]:
        if not tokens:
            return []
        if not transcribed:
            interpolated = interpolate_words(tokens, line.start_ms, line.end_ms)
            return [word.model_copy(update={"line_id": line.id}) for word in interpolated]
        candidates = _candidate_intervals(tokens, transcribed, line)
        repaired = _repair_intervals(candidates, tokens, line)
        return [word.model_copy(update={"line_id": line.id}) for word in repaired]

    def _line_audio_path(self, line: DialogueLine) -> Path | None:
        if not line.audio_path or not line.audio_path.strip():
            return None
        candidate = Path(line.audio_path)
        if candidate.is_absolute():
            return candidate.expanduser()
        try:
            return self.workspace.resolve_project_path(line.project_id, candidate)
        except (WorkspaceError, PathTraversalError):
            return None

    def _voice_language(self, voice_id: str) -> str | None:
        voice: VoiceProfile | None = self.voices.get(voice_id)
        return voice.language.value if voice is not None else None

    def _load_whisper_model(self) -> Any | None:
        if self._whisper_model is not None:
            return self._whisper_model
        if self._whisper_unavailable or not self.enable_whisper:
            return None
        try:
            model = (
                self.model_factory()
                if self.model_factory is not None
                else self._build_whisper_model()
            )
        except Exception:
            model = None
        if model is None:
            self._whisper_unavailable = True
            return None
        self._whisper_model = model
        return model

    def _build_whisper_model(self) -> Any | None:
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.settings.whisper_model_size,
            device=self.settings.resolved_device,
            compute_type=self.settings.whisper_compute_type,
            local_files_only=not self.allow_model_download,
        )

    def _transcribe_line(
        self,
        audio_path: Path,
        line: DialogueLine,
    ) -> Sequence[TranscribedWord] | None:
        if not audio_path.is_file():
            return None
        language = self._voice_language(line.voice_id)
        if self.transcriber is not None:
            try:
                return self.transcriber(audio_path, language)
            except Exception:
                return None
        model = self._load_whisper_model()
        if model is None:
            return None
        try:
            segments, _ = model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=True,
                vad_filter=False,
            )
            words: list[TranscribedWord] = []
            for segment in segments:
                for word in getattr(segment, "words", None) or ():
                    words.append(
                        TranscribedWord(
                            word=str(word.word).strip(),
                            start_ms=round(float(word.start) * 1000),
                            end_ms=round(float(word.end) * 1000),
                            confidence=min(1.0, max(0.0, float(word.probability))),
                        )
                    )
        except Exception:
            return None
        return words or None

    def _write_cues(
        self,
        project_id: str,
        lines: Sequence[DialogueLine],
        words_by_line: Mapping[int, Sequence[WordTimestamp]],
        total_duration_ms: int,
    ) -> Path:
        line_payloads: list[dict[str, object]] = []
        flat_words: list[dict[str, object]] = []
        for line in lines:
            line_words = list(words_by_line.get(line.id or 0, ()))
            word_payloads = [
                {
                    "word_index": word.word_index,
                    "word": word.word,
                    "start_ms": word.start_ms,
                    "end_ms": word.end_ms,
                    "confidence": word.confidence,
                }
                for word in line_words
            ]
            line_payloads.append(
                {
                    "line_id": line.id,
                    "line_index": line.line_index,
                    "voice_id": line.voice_id,
                    "speaker_label": line.speaker_label,
                    "text_content": line.text_content,
                    "start_ms": line.start_ms,
                    "end_ms": line.end_ms,
                    "audio_path": line.audio_path,
                    "words": word_payloads,
                }
            )
            flat_words.extend(
                {
                    "line_id": line.id,
                    "line_index": line.line_index,
                    **payload,
                }
                for payload in word_payloads
            )
        payload = {
            "project_id": project_id,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "total_duration_ms": total_duration_ms,
            "lines": line_payloads,
            "words": flat_words,
            "cues": flat_words,
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        return self.workspace.write_project_file(project_id, CUES_FILENAME, content)


def align_project(
    db: DatabaseManager,
    workspace: WorkspaceManager,
    project_id: str,
    settings: AppSettings | None = None,
    **kwargs: object,
) -> AlignmentResult:
    """Convenience wrapper around :class:`WordAligner`."""

    aligner = WordAligner(db, workspace, settings, **kwargs)
    return aligner.align_project(project_id)


__all__ = [
    "CUES_FILENAME",
    "INTERPOLATED_CONFIDENCE",
    "AlignmentError",
    "AlignmentResult",
    "TranscribedWord",
    "Transcriber",
    "WordAligner",
    "align_project",
    "interpolate_words",
    "tokenize_words",
    "validate_alignment",
]
