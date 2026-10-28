"""Interfaz de linea de comandos de mcp-videos."""

import json
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

import typer

from mcp_videos import __version__
from mcp_videos.config import AppSettings

CHROMIUM_BINARIES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
)

PLAYWRIGHT_CACHE_DIR = Path.home() / ".cache" / "ms-playwright"

PLAYWRIGHT_CHROMIUM_PATTERNS = (
    "chromium-*/chrome-linux/chrome",
    "chromium-*/chrome-linux64/chrome",
    "chromium_headless_shell-*/chrome-linux/headless_shell",
    "chromium_headless_shell-*/chrome-linux64/headless_shell",
)

app = typer.Typer(
    name="mcp-videos",
    help="Servidor MCP, CLI y motor determinista de videos cortos.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Servidor MCP, CLI y motor determinista de videos cortos."""


@dataclass(frozen=True)
class SystemCheckResult:
    """Resultado del diagnostico de dependencias externas del sistema."""

    ffmpeg_path: str | None
    ffprobe_path: str | None
    chromium_path: str | None

    @property
    def is_healthy(self) -> bool:
        """Indica si todas las dependencias externas requeridas estan disponibles."""
        return bool(self.ffmpeg_path and self.ffprobe_path and self.chromium_path)


def _find_playwright_chromium(cache_dir: Path | None = None) -> str | None:
    """Busca un Chromium descargado por Playwright en su cache local."""
    base = cache_dir or PLAYWRIGHT_CACHE_DIR
    if not base.is_dir():
        return None
    for pattern in PLAYWRIGHT_CHROMIUM_PATTERNS:
        for candidate in sorted(base.glob(pattern), reverse=True):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def find_chromium() -> str | None:
    """Localiza Chromium en PATH o en la cache de navegadores de Playwright."""
    for binary in CHROMIUM_BINARIES:
        path = shutil.which(binary)
        if path:
            return path
    return _find_playwright_chromium()


def check_system_dependencies() -> SystemCheckResult:
    """Comprueba la disponibilidad de `ffmpeg`, `ffprobe` y Chromium."""
    return SystemCheckResult(
        ffmpeg_path=shutil.which("ffmpeg"),
        ffprobe_path=shutil.which("ffprobe"),
        chromium_path=find_chromium(),
    )


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Emite el diagnostico en formato JSON."),
) -> None:
    """Diagnostica ffmpeg, ffprobe, Chromium y la configuracion activa."""
    result = check_system_dependencies()
    settings = AppSettings()
    status = "ok" if result.is_healthy else "error"
    payload = {
        "status": status,
        "healthy": result.is_healthy,
        "version": __version__,
        "python": platform.python_version(),
        "ffmpeg": result.ffmpeg_path,
        "ffprobe": result.ffprobe_path,
        "chromium": result.chromium_path,
        "settings": {
            "db_path": str(settings.mcp_videos_db_path),
            "workspaces_dir": str(settings.mcp_videos_workspaces_dir),
            "voices_dir": str(settings.mcp_videos_voices_dir),
            "device": settings.resolved_device,
            "preview_port": settings.preview_port,
            "default_video_fps": settings.default_video_fps,
        },
    }

    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"mcp-videos {__version__} — doctor")
        typer.echo(f"Python: {payload['python']}")
        for label, key in (("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe"), ("chromium", "chromium")):
            value = payload[key]
            state = "OK" if value else "MISSING"
            typer.echo(f"{label}: {state} ({value or 'no encontrado'})")
        typer.echo(f"device: {settings.resolved_device}")
        typer.echo(f"preview_port: {settings.preview_port}")
        typer.echo(f"status: {status.upper()}")

    if not result.is_healthy:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
