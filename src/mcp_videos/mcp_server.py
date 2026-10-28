"""FastMCP server exposing the mcp-videos application service to AI agents."""

from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: no cover - mcp >= 2 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP

from mcp_videos.core.orchestrator import ProjectOrchestrator

mcp = FastMCP("mcp-videos")

_orchestrator: ProjectOrchestrator | None = None


def get_orchestrator() -> ProjectOrchestrator:
    """Return the process-wide orchestrator, creating it lazily."""

    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ProjectOrchestrator()
    return _orchestrator


def set_orchestrator(orchestrator: ProjectOrchestrator | None) -> None:
    """Inject a custom orchestrator (used by tests and embedders)."""

    global _orchestrator
    _orchestrator = orchestrator


def _error_payload(error: Exception) -> dict[str, Any]:
    message = str(error).strip() or error.__class__.__name__
    return {
        "status": "error",
        "error": message,
        "error_type": error.__class__.__name__,
    }


@mcp.tool()
def register_voice(
    voice_id: str,
    name: str | None = None,
    engine: str = "kokoro",
    mode: str = "preset",
    sample_wav_path: str | None = None,
    language: str = "es",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a preset voice or clone one from a 3-30 second WAV sample."""

    try:
        return get_orchestrator().register_voice(
            voice_id,
            name,
            engine=engine,
            mode=mode,
            sample_wav_path=sample_wav_path,
            language=language,
            config=config,
        )
    except Exception as error:
        return _error_payload(error)


@mcp.tool()
def list_voices(language: str | None = None) -> list[dict[str, Any]]:
    """List the available voices, optionally filtered by language (es/en)."""

    try:
        return get_orchestrator().list_voices(language)
    except Exception as error:
        return [_error_payload(error)]


@mcp.tool()
def create_project(
    project_id: str,
    title: str,
    width: int = 1080,
    height: int = 1920,
    fps: int = 60,
    description: str = "",
) -> dict[str, Any]:
    """Create a video project workspace (vertical 9:16 by default, 60 FPS)."""

    try:
        return get_orchestrator().create_project(
            project_id,
            title,
            width=width,
            height=height,
            fps=fps,
            description=description,
        )
    except Exception as error:
        return _error_payload(error)


@mcp.tool()
def synthesize_dialogue(
    project_id: str,
    lines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Synthesize multi-voice dialogue, build dialogue_master.wav and align words."""

    try:
        return get_orchestrator().synthesize_and_align(project_id, lines)
    except Exception as error:
        return _error_payload(error)


@mcp.tool()
def save_and_validate_scene(
    project_id: str,
    scene_index: int,
    title: str,
    start_ms: int,
    end_ms: int,
    html_content: str | None = None,
    prompt: str = "",
    provider: str | None = None,
) -> dict[str, Any]:
    """Save or generate one HTML scene and validate it in headless Chromium."""

    try:
        return get_orchestrator().save_and_validate_scene(
            project_id,
            scene_index,
            title,
            start_ms,
            end_ms,
            html_content=html_content,
            prompt=prompt,
            provider=provider,
        )
    except Exception as error:
        return _error_payload(error)


@mcp.tool()
def render_project_video(
    project_id: str,
    burn_subtitles: bool = True,
) -> dict[str, Any]:
    """Render validated scenes frame-by-frame at the project FPS into final.mp4."""

    try:
        return get_orchestrator().render_project(
            project_id,
            burn_subtitles=burn_subtitles,
        )
    except Exception as error:
        return _error_payload(error)


def main() -> None:
    """Run the MCP server over stdio (used by IDE and agent integrations)."""

    mcp.run()


if __name__ == "__main__":
    main()


__all__ = [
    "FastMCP",
    "create_project",
    "get_orchestrator",
    "list_voices",
    "main",
    "mcp",
    "register_voice",
    "render_project_video",
    "save_and_validate_scene",
    "set_orchestrator",
    "synthesize_dialogue",
]
