"""FastAPI Live Studio Preview server for inspecting projects in the browser."""

from __future__ import annotations

import contextlib
import inspect
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from mcp_videos.config import AppSettings
from mcp_videos.core.models import Project, Scene
from mcp_videos.core.workspace import (
    PathTraversalError,
    WorkspaceError,
    WorkspaceManager,
)
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import (
    DialogueRepository,
    ProjectNotFoundError,
    ProjectRepository,
    RenderJobNotFoundError,
    RepositoryError,
    SceneNotFoundError,
    SceneRepository,
)
from mcp_videos.render.engine import (
    DeterministicRenderEngine,
    RenderEngineError,
    SceneNotValidatedError,
)
from mcp_videos.scenes.validator import SceneValidationError, inject_harness

PREVIEW_STATIC_DIR = Path(__file__).with_name("static")
INDEX_FILENAME = "index.html"
NO_STORE_HEADERS = {"Cache-Control": "no-store"}
DEFAULT_PREVIEW_TITLE = "mcp-videos Live Studio Preview"


class RenderRequest(BaseModel):
    """Optional render options sent by the studio render button."""

    scene_index: int | None = Field(default=None, ge=0)
    burn_subtitles: bool = True


def _json_model(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _status_value(value: object, default: str = "completed") -> str:
    if value is None:
        return default
    return str(getattr(value, "value", value))


def _resolve_scene_html(
    workspace: WorkspaceManager,
    project: Project,
    scene: Scene,
) -> Path:
    candidate = Path(scene.html_path)
    if candidate.is_absolute():
        return candidate
    return workspace.resolve_project_path(project.id, candidate)


def _master_audio_candidates(
    workspace: WorkspaceManager,
    project: Project,
) -> list[Path]:
    candidates: list[Path] = []
    if project.master_audio_path:
        stored = Path(project.master_audio_path)
        if stored.is_absolute():
            candidates.append(stored)
        else:
            with contextlib.suppress(PathTraversalError, WorkspaceError):
                candidates.append(workspace.resolve_project_path(project.id, stored))
    candidates.append(workspace.dialogue_master_path(project.id))
    return candidates


def _find_master_audio(
    workspace: WorkspaceManager,
    project: Project,
) -> Path | None:
    for candidate in _master_audio_candidates(workspace, project):
        if candidate.is_file():
            return candidate
    return None


def _render_job_payload(
    result: object,
    project: Project,
) -> dict[str, Any]:
    job = getattr(result, "job", None)
    if job is not None:
        job_id = getattr(job, "id", None)
        status = _status_value(getattr(job, "status", None))
        output_path = getattr(job, "output_mp4_path", None)
    else:
        job_id = getattr(result, "job_id", None)
        status = _status_value(getattr(result, "status", None))
        output_path = getattr(result, "output_mp4_path", None)

    output_path = output_path or getattr(result, "output_path", None)
    total_frames = getattr(result, "total_frames", None)
    rendered_frames = getattr(result, "rendered_frames", None)

    payload: dict[str, Any] = {
        "job_id": str(job_id) if job_id else str(uuid.uuid4()),
        "status": status,
        "project_id": project.id,
    }
    if output_path is not None:
        payload["output_mp4_path"] = str(output_path)
    if total_frames is not None:
        payload["total_frames"] = int(total_frames)
    if rendered_frames is not None:
        payload["rendered_frames"] = int(rendered_frames)
    return payload


def _mapping_payload(
    response: Mapping[str, Any],
    project: Project,
) -> dict[str, Any]:
    payload = dict(response)
    job = payload.get("job")
    if job is not None:
        payload.setdefault("job_id", getattr(job, "id", None))
        payload.setdefault("status", _status_value(getattr(job, "status", None)))
    payload.setdefault("project_id", project.id)
    payload["status"] = _status_value(payload.get("status"))
    if "job_id" in payload and payload["job_id"] is not None:
        payload["job_id"] = str(payload["job_id"])
    else:
        payload["job_id"] = str(uuid.uuid4())
    return payload


def _invoke_render_engine(
    render_method: Any,
    project: Project,
    burn_subtitles: bool,
) -> object:
    try:
        signature = inspect.signature(render_method)
    except (TypeError, ValueError):
        return render_method(project, burn_subtitles=burn_subtitles)

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if "project" in parameters or accepts_kwargs:
        kwargs["project"] = project
    elif "project_id" in parameters:
        kwargs["project_id"] = project.id
    if "burn_subtitles" in parameters or accepts_kwargs:
        kwargs["burn_subtitles"] = burn_subtitles
    if kwargs:
        return render_method(**kwargs)
    return render_method(project)


def _invoke_orchestrator(
    orchestrator: object,
    project: Project,
    burn_subtitles: bool,
) -> object:
    method = getattr(orchestrator, "render_project_video", None)
    if method is None:
        method = getattr(orchestrator, "render_project", None)
    if method is None:
        raise RenderEngineError("el orquestador inyectado no implementa render_project_video")

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(project.id, burn_subtitles)

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if "project_id" in parameters or accepts_kwargs:
        kwargs["project_id"] = project.id
    elif "project" in parameters:
        kwargs["project"] = project
    if "burn_subtitles" in parameters or accepts_kwargs:
        kwargs["burn_subtitles"] = burn_subtitles
    if kwargs:
        return method(**kwargs)
    return method(project.id)


def create_preview_app(
    db: DatabaseManager | None = None,
    workspace: WorkspaceManager | None = None,
    orchestrator: object | None = None,
    settings: AppSettings | None = None,
    *,
    render_engine: object | None = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    """Build the FastAPI application serving the interactive preview studio."""

    if isinstance(orchestrator, AppSettings) and settings is None:
        settings = orchestrator
        orchestrator = None

    resolved_settings = settings or AppSettings()
    if db is None:
        db = DatabaseManager(resolved_settings.mcp_videos_db_path)
    if workspace is None:
        workspace = WorkspaceManager(
            resolved_settings.mcp_videos_workspaces_dir,
            resolved_settings.mcp_videos_voices_dir,
        )

    db.initialize()

    projects_repo = ProjectRepository(db)
    dialogues_repo = DialogueRepository(db)
    scenes_repo = SceneRepository(db)
    resolved_static_dir = Path(static_dir) if static_dir is not None else PREVIEW_STATIC_DIR
    engine = render_engine

    app = FastAPI(title=DEFAULT_PREVIEW_TITLE, version="0.1.0")
    app.state.db = db
    app.state.workspace = workspace
    app.state.settings = resolved_settings
    app.state.orchestrator = orchestrator
    app.state.render_engine = engine
    app.state.static_dir = resolved_static_dir
    app.state.projects_repo = projects_repo
    app.state.dialogues_repo = dialogues_repo
    app.state.scenes_repo = scenes_repo

    def get_project_or_404(project_id: str) -> Project:
        project = projects_repo.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Proyecto no registrado: {project_id}")
        return project

    def audio_url_for(project_id: str) -> str:
        return f"/preview/{project_id}/audio"

    def project_audio_available(project: Project) -> bool:
        return _find_master_audio(workspace, project) is not None

    def scene_manifest(project: Project, scene: Scene) -> dict[str, Any]:
        payload = _json_model(scene)
        payload["html_url"] = f"/preview/{project.id}/scene/{scene.scene_index}"
        payload["scene_url"] = payload["html_url"]
        payload["duration_ms"] = scene.end_ms - scene.start_ms
        return payload

    def project_manifest(project: Project) -> dict[str, Any]:
        lines = dialogues_repo.list_by_project(project.id)

        words: list[dict[str, Any]] = []
        for line in lines:
            if line.id is None:
                continue
            for word in dialogues_repo.list_word_timestamps(line.id):
                payload = _json_model(word)
                payload["speaker_label"] = line.speaker_label
                payload["line_index"] = line.line_index
                words.append(payload)
        words.sort(key=lambda item: (item["start_ms"], item["line_id"] or 0, item["word_index"]))

        project_scenes = scenes_repo.list_by_project(project.id)
        has_audio = project_audio_available(project)
        audio_url = audio_url_for(project.id) if has_audio else None
        return {
            "project": _json_model(project),
            "lines": [_json_model(line) for line in lines],
            "words": words,
            "scenes": [scene_manifest(project, scene) for scene in project_scenes],
            "total_duration_ms": project.total_duration_ms,
            "master_audio_url": audio_url,
            "audio_url": audio_url,
            "has_audio": has_audio,
        }

    def dispatch_render(project: Project, burn_subtitles: bool) -> dict[str, Any]:
        if orchestrator is not None:
            try:
                response = _invoke_orchestrator(orchestrator, project, burn_subtitles)
            except (RenderEngineError, SceneNotValidatedError):
                raise
            except RepositoryError:
                raise
            except (TypeError, ValueError) as exc:
                raise RenderEngineError(f"el orquestador no pudo renderizar: {exc}") from exc
            if isinstance(response, Mapping):
                return _mapping_payload(response, project)
            return _render_job_payload(response, project)

        active_engine = engine or DeterministicRenderEngine(
            db,
            workspace,
            resolved_settings,
        )
        render_method = getattr(active_engine, "render_project", None)
        if render_method is None:
            render_method = getattr(active_engine, "render", None)
        if render_method is None and callable(active_engine):
            render_method = active_engine
        if render_method is None:
            raise RenderEngineError("el motor de render inyectado no es compatible")
        result = _invoke_render_engine(render_method, project, burn_subtitles)
        if isinstance(result, Mapping):
            return _mapping_payload(result, project)
        return _render_job_payload(result, project)

    @app.get("/", response_class=HTMLResponse)
    def studio_index() -> FileResponse:
        index_path = resolved_static_dir / INDEX_FILENAME
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="No existe la interfaz del estudio")
        return FileResponse(
            index_path,
            media_type="text/html",
            headers=NO_STORE_HEADERS,
        )

    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for project in projects_repo.list_all():
            project_scenes = scenes_repo.list_by_project(project.id)
            has_audio = project_audio_available(project)
            item = _json_model(project)
            item["scene_count"] = len(project_scenes)
            item["word_count"] = sum(
                len(dialogues_repo.list_word_timestamps(line.id))
                for line in dialogues_repo.list_by_project(project.id)
                if line.id is not None
            )
            item["has_audio"] = has_audio
            item["master_audio_url"] = audio_url_for(project.id) if has_audio else None
            payload.append(item)
        return payload

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        return project_manifest(get_project_or_404(project_id))

    @app.get("/preview/{project_id}/scene/{scene_index}", response_class=HTMLResponse)
    def get_scene(project_id: str, scene_index: int) -> HTMLResponse:
        project = get_project_or_404(project_id)
        scene = scenes_repo.get(project.id, scene_index)
        if scene is None:
            raise HTTPException(
                status_code=404,
                detail=f"Escena no registrada: {project_id}#{scene_index}",
            )
        try:
            html_path = _resolve_scene_html(workspace, project, scene)
        except (PathTraversalError, WorkspaceError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not html_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"No existe el HTML de la escena: {scene.html_path}",
            )
        try:
            prepared = inject_harness(html_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SceneValidationError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return HTMLResponse(prepared, headers=NO_STORE_HEADERS)

    @app.get("/preview/{project_id}/audio")
    def get_audio(project_id: str) -> FileResponse:
        project = get_project_or_404(project_id)
        try:
            audio_path = _find_master_audio(workspace, project)
        except (PathTraversalError, WorkspaceError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if audio_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"No existe el audio maestro del proyecto: {project_id}",
            )
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            headers=NO_STORE_HEADERS,
        )

    @app.post("/api/projects/{project_id}/render")
    def render_project(
        project_id: str,
        payload: RenderRequest | None = None,
    ) -> dict[str, Any]:
        project = get_project_or_404(project_id)
        request = payload or RenderRequest()
        if (
            request.scene_index is not None
            and scenes_repo.get(project.id, request.scene_index) is None
        ):
            raise HTTPException(
                status_code=404,
                detail=f"Escena no registrada: {project_id}#{request.scene_index}",
            )
        try:
            return dispatch_render(project, request.burn_subtitles)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (SceneNotFoundError, RenderJobNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SceneNotValidatedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (RenderEngineError, RepositoryError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_preview_app()


__all__ = [
    "INDEX_FILENAME",
    "PREVIEW_STATIC_DIR",
    "RenderRequest",
    "app",
    "create_preview_app",
]
