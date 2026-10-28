"""Interfaz de linea de comandos de mcp-videos."""

import json
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from mcp_videos import __version__
from mcp_videos.config import AppSettings
from mcp_videos.core.orchestrator import DEMO_PROJECT_ID, DEMO_TITLE, ProjectOrchestrator

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


def get_orchestrator() -> ProjectOrchestrator:
    """Construye el orquestador de aplicacion usado por los comandos de datos."""
    return ProjectOrchestrator()


def _abort(error: Exception) -> None:
    typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _emit(payload: object, as_json: bool, human_lines: list[str]) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return
    for line in human_lines:
        typer.echo(line)


def _parse_config(raw: str | None) -> dict[str, Any] | None:
    if raw is None or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise typer.BadParameter(f"--config debe contener JSON válido: {error}") from error
    if not isinstance(payload, dict):
        raise typer.BadParameter("--config debe contener un objeto JSON")
    return payload


def _load_script_turns(script_file: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(script_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"no se pudo leer --script-file: {error}") from error
    if isinstance(payload, dict):
        payload = payload.get("lines", payload.get("turns"))
    if not isinstance(payload, list) or not payload:
        raise typer.BadParameter(
            "--script-file debe contener una lista de turnos o {'lines': [...]}"
        )
    turns: list[dict[str, Any]] = []
    for index, turn in enumerate(payload):
        if not isinstance(turn, dict):
            raise typer.BadParameter(f"el turno {index} no es un objeto JSON")
        turns.append(dict(turn))
    return turns


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


@app.command()
def voices(
    language: str | None = typer.Option(None, "--language", help="Filtra por idioma (es/en)."),
    as_json: bool = typer.Option(False, "--json", help="Emite la lista en formato JSON."),
) -> None:
    """Lista el catalogo de voces registradas."""
    try:
        payload = get_orchestrator().list_voices(language)
    except Exception as error:
        _abort(error)
        return
    human = [
        f"{voice.get('voice_id')}  [{voice.get('engine')}/{voice.get('mode')}]  "
        f"{voice.get('name')}  ({voice.get('language')})"
        for voice in payload
    ]
    _emit(payload, as_json, human or ["No hay voces registradas."])


@app.command("register-voice")
def register_voice(
    voice_id: str = typer.Option(..., "--voice-id", help="Slug canonico de la voz."),
    name: str = typer.Option(..., "--name", help="Nombre visible del personaje."),
    engine: str = typer.Option("kokoro", "--engine", help="Motor TTS (kokoro, piper, f5...)."),
    mode: str = typer.Option("preset", "--mode", help="Modo preset o clone."),
    sample: Annotated[
        Path | None,
        typer.Option(
            "--sample",
            help="WAV de 3-30 s para voces clonadas.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    language: str = typer.Option("es", "--language", help="Idioma de la voz."),
    config: str | None = typer.Option(None, "--config", help="Config del motor como JSON."),
    as_json: bool = typer.Option(False, "--json", help="Emite la respuesta en formato JSON."),
) -> None:
    """Registra una voz predefinida o clona una voz desde un archivo WAV."""
    try:
        payload = get_orchestrator().register_voice(
            voice_id,
            name,
            engine=engine,
            mode=mode,
            sample_wav_path=str(sample) if sample is not None else None,
            language=language,
            config=_parse_config(config),
        )
    except Exception as error:
        _abort(error)
        return
    _emit(
        payload,
        as_json,
        [
            f"Voz {payload['voice_id']} registrada ({payload['engine']}/{payload['mode']}).",
            f"sample_rate: {payload['sample_rate']} Hz",
        ],
    )


@app.command("create-project")
def create_project(
    project_id: str = typer.Option(..., "--project-id", help="Slug canonico del proyecto."),
    title: str = typer.Option(..., "--title", help="Titulo del short."),
    width: int = typer.Option(1080, "--width", help="Ancho del viewport."),
    height: int = typer.Option(1920, "--height", help="Alto del viewport."),
    fps: int = typer.Option(60, "--fps", help="FPS deterministas (24/30/60)."),
    description: str = typer.Option("", "--description", help="Descripcion opcional."),
    as_json: bool = typer.Option(False, "--json", help="Emite la respuesta en formato JSON."),
) -> None:
    """Crea un proyecto de video y su workspace."""
    try:
        payload = get_orchestrator().create_project(
            project_id,
            title,
            width=width,
            height=height,
            fps=fps,
            description=description,
        )
    except Exception as error:
        _abort(error)
        return
    _emit(
        payload,
        as_json,
        [
            f"Proyecto {payload['project_id']} creado ({payload['width']}x{payload['height']} "
            f"@ {payload['fps']} FPS).",
            f"workspace: {payload['workspace_dir']}",
        ],
    )


@app.command()
def synthesize(
    script_file: Annotated[
        Path,
        typer.Option(
            "--script-file",
            help="JSON con la lista de turnos de dialogo.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    project_id: str = typer.Option(..., "--project-id", help="Slug del proyecto."),
    as_json: bool = typer.Option(False, "--json", help="Emite la respuesta en formato JSON."),
) -> None:
    """Sintetiza el dialogo multivoz y alinea los word timestamps."""
    turns = _load_script_turns(script_file)
    try:
        payload = get_orchestrator().synthesize_and_align(project_id, turns)
    except Exception as error:
        _abort(error)
        return
    _emit(
        payload,
        as_json,
        [
            f"Dialogo sintetizado: {payload['lines_count']} lineas, "
            f"{payload['words_count']} palabras.",
            f"duracion: {payload['total_duration_ms']} ms",
            f"audio maestro: {payload['master_audio_path']}",
        ],
    )


@app.command("save-scene")
def save_scene(
    project_id: str = typer.Option(..., "--project-id", help="Slug del proyecto."),
    scene_index: int = typer.Option(..., "--scene-index", help="Indice contiguo de la escena."),
    start_ms: int = typer.Option(..., "--start-ms", help="Inicio absoluto en ms."),
    end_ms: int = typer.Option(..., "--end-ms", help="Fin absoluto en ms."),
    title: str = typer.Option("", "--title", help="Titulo de la escena."),
    html_file: Annotated[
        Path | None,
        typer.Option(
            "--html-file",
            help="HTML autocontenido a guardar y validar.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    prompt: str = typer.Option("", "--prompt", help="Prompt o nota de generacion."),
    provider: str | None = typer.Option(None, "--provider", help="Proveedor LLM de escenas."),
    as_json: bool = typer.Option(False, "--json", help="Emite la respuesta en formato JSON."),
) -> None:
    """Guarda o genera una escena HTML y la valida en Chromium Headless."""
    html_content = html_file.read_text(encoding="utf-8") if html_file is not None else None
    scene_title = title or f"Escena {scene_index:02d}"
    try:
        payload = get_orchestrator().save_and_validate_scene(
            project_id,
            scene_index,
            scene_title,
            start_ms,
            end_ms,
            html_content=html_content,
            prompt=prompt,
            provider=provider,
        )
    except Exception as error:
        _abort(error)
        return
    errors = payload.get("console_errors") or []
    _emit(
        payload,
        as_json,
        [
            f"Escena {payload['scene_index']} guardada: {payload['validation_status']}.",
            f"proyecto: {payload['project_status']}",
            *[f"error: {message}" for message in errors],
        ],
    )


@app.command()
def render(
    project_id: str = typer.Option(..., "--project-id", help="Slug del proyecto."),
    burn_subtitles: bool = typer.Option(
        True,
        "--burn-subtitles/--no-burn-subtitles",
        help="Quema los subtitulos ASS en el MP4.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emite la respuesta en formato JSON."),
) -> None:
    """Renderiza el video determinista a FPS constantes y subtitulos dinamicos."""
    try:
        payload = get_orchestrator().render_project(
            project_id,
            burn_subtitles=burn_subtitles,
        )
    except Exception as error:
        _abort(error)
        return
    _emit(
        payload,
        as_json,
        [
            f"Render {payload['status']}: {payload['rendered_frames']}/"
            f"{payload['total_frames']} frames.",
            f"video: {payload['output_mp4_path']}",
            f"subtitulos: {payload['subtitles_path'] or 'desactivados'}",
        ],
    )


@app.command()
def preview(
    host: str = typer.Option("127.0.0.1", "--host", help="Host del Estudio Web."),
    port: int | None = typer.Option(None, "--port", help="Puerto HTTP del Estudio Web."),
) -> None:
    """Levanta el Estudio Web de Previsualizacion Interactiva."""
    import uvicorn

    from mcp_videos.preview.server import create_preview_app

    settings = AppSettings()
    uvicorn.run(
        create_preview_app(),
        host=host,
        port=port or settings.preview_port,
    )


@app.command("build-demo")
def build_demo(
    project_id: str = typer.Option(DEMO_PROJECT_ID, "--project-id", help="Slug del proyecto demo."),
    title: str = typer.Option(DEMO_TITLE, "--title", help="Titulo del short demo."),
    width: int = typer.Option(1080, "--width", help="Ancho del viewport."),
    height: int = typer.Option(1920, "--height", help="Alto del viewport."),
    fps: int = typer.Option(60, "--fps", help="FPS deterministas (24/30/60)."),
    scene_count: int = typer.Option(2, "--scene-count", help="Numero de escenas HTML."),
    render_video: bool = typer.Option(
        True,
        "--render/--no-render",
        help="Renderiza el MP4 final al terminar.",
    ),
    burn_subtitles: bool = typer.Option(
        True,
        "--burn-subtitles/--no-burn-subtitles",
        help="Quema los subtitulos ASS en el MP4.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emite la respuesta en formato JSON."),
) -> None:
    """Construye y renderiza el short demostrativo completo de la antena."""
    try:
        payload = get_orchestrator().build_demo_short(
            project_id,
            title=title,
            width=width,
            height=height,
            fps=fps,
            scene_count=scene_count,
            burn_subtitles=burn_subtitles,
            render=render_video,
        )
    except Exception as error:
        _abort(error)
        return
    render_summary = payload.get("render") or {}
    _emit(
        payload,
        as_json,
        [
            f"Demo {payload['project_id']} lista: {len(payload['scenes'])} escenas.",
            f"audio: {payload['synthesis']['total_duration_ms']} ms",
            f"video: {render_summary.get('output_mp4_path', payload['final_video_path'])}",
        ],
    )


if __name__ == "__main__":
    app()
