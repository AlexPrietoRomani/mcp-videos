"""Dynamic ASS subtitle generation from project word timestamps."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mcp_videos.core.models import WordTimestamp

ASS_HIGHLIGHT_COLOR = "&H00D4FF&"
ASS_BASE_COLOR = "&H00FFFFFF&"
DEFAULT_MAX_WORDS_PER_BLOCK = 4
DEFAULT_FONT_NAME = "Arial"

_BASE_COLOR_TAG = f"{{\\c{ASS_BASE_COLOR}}}"


@dataclass(frozen=True)
class AssEvent:
    """One rendered ASS dialogue event expressed in centiseconds."""

    start_cs: int
    end_cs: int
    text: str


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} debe ser un entero mayor que 0")
    return value


def _format_centiseconds(total_cs: int) -> str:
    cs = total_cs % 100
    total_seconds = total_cs // 100
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def format_ass_time(ms: int) -> str:
    """Convert milliseconds to the ASS ``H:MM:SS.cc`` timestamp format."""

    if isinstance(ms, bool) or not isinstance(ms, int):
        raise ValueError("ms debe ser un entero en milisegundos")
    return _format_centiseconds(max(0, round(ms / 10.0)))


def _milliseconds_to_centiseconds(ms: int) -> int:
    return round(ms / 10.0)


def _escape_ass_text(text: str) -> str:
    normalized = " ".join(text.splitlines())
    return normalized.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def group_words_into_blocks(
    words: Sequence[WordTimestamp],
    max_words_per_block: int = DEFAULT_MAX_WORDS_PER_BLOCK,
) -> list[list[WordTimestamp]]:
    """Group timestamped words into short visual blocks of at most four words."""

    max_words = _validate_positive_int(max_words_per_block, "max_words_per_block")
    ordered = sorted(
        words,
        key=lambda word: (word.start_ms, word.line_id or 0, word.word_index),
    )

    blocks: list[list[WordTimestamp]] = []
    current: list[WordTimestamp] = []
    current_line_id: int | None = None

    for word in ordered:
        line_changed = (
            current
            and word.line_id is not None
            and current_line_id is not None
            and word.line_id != current_line_id
        )
        if len(current) >= max_words or line_changed:
            blocks.append(current)
            current = []
            current_line_id = None
        if not current:
            current_line_id = word.line_id
        current.append(word)

    if current:
        blocks.append(current)
    return blocks


def _block_text(block: Sequence[WordTimestamp]) -> str:
    return " ".join(_escape_ass_text(word.word) for word in block)


def _highlighted_text(
    block: Sequence[WordTimestamp],
    active_index: int,
    highlight_color: str,
) -> str:
    highlight_tag = f"{{\\c{highlight_color}}}"
    parts: list[str] = []
    for index, word in enumerate(block):
        token = _escape_ass_text(word.word)
        if index == active_index:
            parts.append(f"{highlight_tag}{token}{_BASE_COLOR_TAG}")
        else:
            parts.append(token)
    return " ".join(parts)


def _build_events(
    blocks: Sequence[Sequence[WordTimestamp]],
    highlight_color: str,
) -> list[AssEvent]:
    events: list[AssEvent] = []
    global_cursor_cs = 0

    for block in blocks:
        if not block:
            continue
        plain_text = _block_text(block)
        cursor_cs = max(_milliseconds_to_centiseconds(block[0].start_ms), global_cursor_cs)

        for index, word in enumerate(block):
            start_cs = max(_milliseconds_to_centiseconds(word.start_ms), cursor_cs)
            if index > 0 and start_cs > cursor_cs:
                events.append(AssEvent(cursor_cs, start_cs, plain_text))
            end_cs = max(_milliseconds_to_centiseconds(word.end_ms), start_cs + 1)
            events.append(
                AssEvent(
                    start_cs,
                    end_cs,
                    _highlighted_text(block, index, highlight_color),
                )
            )
            cursor_cs = end_cs

        global_cursor_cs = cursor_cs

    return events


def build_ass_document(
    words: Sequence[WordTimestamp],
    width: int,
    height: int,
    *,
    max_words_per_block: int = DEFAULT_MAX_WORDS_PER_BLOCK,
    font_name: str = DEFAULT_FONT_NAME,
    highlight_color: str = ASS_HIGHLIGHT_COLOR,
) -> str:
    """Render a complete 9:16-friendly ASS document with per-word highlighting."""

    width = _validate_positive_int(width, "width")
    height = _validate_positive_int(height, "height")
    max_words = _validate_positive_int(max_words_per_block, "max_words_per_block")
    if not font_name or any(char in font_name for char in ",\r\n"):
        raise ValueError("font_name no puede estar vacío ni contener comas o saltos de línea")
    if not highlight_color.startswith("&H") or not highlight_color.endswith("&"):
        raise ValueError("highlight_color debe usar el formato ASS '&HBBGGRR&'")

    font_size = max(28, round(min(width, height) * 0.07))
    outline = max(2, round(font_size * 0.06))
    shadow = max(1, round(outline / 2))
    margin_lr = max(24, round(width * 0.05))
    margin_v = max(24, round(height * 0.16))

    header = [
        "[Script Info]",
        "; Generado por mcp-videos",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
            "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Shorts,{font_name},{font_size},&H00FFFFFF,&H0000D4FF,"
            f"&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},"
            f"2,{margin_lr},{margin_lr},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    blocks = group_words_into_blocks(words, max_words)
    events = _build_events(blocks, highlight_color)
    dialogue_lines = [
        (
            f"Dialogue: 0,{_format_centiseconds(event.start_cs)},"
            f"{_format_centiseconds(event.end_cs)},Shorts,,0,0,0,,{event.text}"
        )
        for event in events
    ]

    return "\n".join([*header, *dialogue_lines, ""])


def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def generate_ass_subtitles(
    words: Sequence[WordTimestamp],
    width: int,
    height: int,
    output_path: Path | str,
    *,
    max_words_per_block: int = DEFAULT_MAX_WORDS_PER_BLOCK,
    font_name: str = DEFAULT_FONT_NAME,
    highlight_color: str = ASS_HIGHLIGHT_COLOR,
) -> Path:
    """Write a dynamic ASS subtitle file and return its absolute destination path."""

    target = Path(output_path).expanduser()
    document = build_ass_document(
        words,
        width,
        height,
        max_words_per_block=max_words_per_block,
        font_name=font_name,
        highlight_color=highlight_color,
    )
    _atomic_write(target, document)
    return target


__all__ = [
    "ASS_BASE_COLOR",
    "ASS_HIGHLIGHT_COLOR",
    "DEFAULT_FONT_NAME",
    "DEFAULT_MAX_WORDS_PER_BLOCK",
    "AssEvent",
    "build_ass_document",
    "format_ass_time",
    "generate_ass_subtitles",
    "group_words_into_blocks",
]
