"""Deterministic rendering: dynamic ASS subtitles and 60 FPS frame capture."""

from mcp_videos.render.engine import (
    DEFAULT_AUDIO_BITRATE,
    DEFAULT_FALLBACK_CHROMIUM_PATH,
    SUBTITLES_FILENAME,
    DeterministicRenderEngine,
    RenderEngineError,
    RenderResult,
    SceneNotValidatedError,
)
from mcp_videos.render.subtitles import (
    ASS_BASE_COLOR,
    ASS_HIGHLIGHT_COLOR,
    DEFAULT_FONT_NAME,
    DEFAULT_MAX_WORDS_PER_BLOCK,
    AssEvent,
    build_ass_document,
    format_ass_time,
    generate_ass_subtitles,
    group_words_into_blocks,
)

__all__ = [
    "ASS_BASE_COLOR",
    "ASS_HIGHLIGHT_COLOR",
    "DEFAULT_AUDIO_BITRATE",
    "DEFAULT_FALLBACK_CHROMIUM_PATH",
    "DEFAULT_FONT_NAME",
    "DEFAULT_MAX_WORDS_PER_BLOCK",
    "SUBTITLES_FILENAME",
    "AssEvent",
    "DeterministicRenderEngine",
    "RenderEngineError",
    "RenderResult",
    "SceneNotValidatedError",
    "build_ass_document",
    "format_ass_time",
    "generate_ass_subtitles",
    "group_words_into_blocks",
]
