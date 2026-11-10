"""Puente de integración híbrida para HyperFrames (@hyperframes/cli).

Gestiona la exportación de proyectos a composiciones HyperFrames autocontenidas,
la auditoría visual paralela y el renderizado híbrido con fallback a Playwright.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from mcp_videos.config import AppSettings
from mcp_videos.core.models import (
    Project,
    ProjectStatus,
    RenderJob,
    RenderStatus,
    Scene,
    WordTimestamp,
    total_frames_for_duration,
)
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import ProjectRepository, RenderJobRepository
from mcp_videos.render.engine import DeterministicRenderEngine, RenderEngineError
from mcp_videos.scenes.validator import SceneValidator, inject_harness

logger = logging.getLogger(__name__)

RenderBackend = Literal["auto", "hyperframes", "playwright"]


def _extract_json_block(raw: str) -> dict[str, Any]:
    """Extrae y parsea el primer bloque JSON válido presente en el texto."""
    raw = raw.strip()
    if not raw:
        return {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class HyperFramesBridge:
    """Puente de exportación, auditoría visual y renderizado híbrido HyperFrames."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        db_path: Path | None = None,
        render_engine: DeterministicRenderEngine | None = None,
        workspace: WorkspaceManager | None = None,
        scene_validator: SceneValidator | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.db = DatabaseManager(db_path or self.settings.mcp_videos_db_path)
        self.workspace = workspace or WorkspaceManager(
            self.settings.mcp_videos_workspaces_dir,
            self.settings.mcp_videos_voices_dir,
        )
        self.projects = ProjectRepository(self.db)
        self.jobs = RenderJobRepository(self.db)
        self.render_engine = render_engine or DeterministicRenderEngine(
            self.db,
            self.workspace,
            self.settings,
        )
        self.scene_validator = scene_validator

    def _resolve_command_prefix(self) -> list[str] | None:
        """Determina el comando ejecutable de HyperFrames en el sistema."""
        # 1. Configuración explícita en settings o variable de entorno
        env_dir = None
        if self.settings.hyperframes_dir:
            env_dir = str(self.settings.hyperframes_dir)
        elif os.environ.get("MCP_VIDEOS_HYPERFRAMES_DIR"):
            env_dir = os.environ["MCP_VIDEOS_HYPERFRAMES_DIR"]

        if env_dir:
            base = Path(env_dir)
            if base.is_file():
                return [str(base)]
            if base.is_dir():
                candidates = (
                    base / "node_modules" / ".bin" / "hyperframes.cmd",
                    base / "node_modules" / ".bin" / "hyperframes",
                    base / "hyperframes.cmd",
                    base / "hyperframes",
                    base / "bin" / "hyperframes.js",
                )
                for cand in candidates:
                    if cand.is_file():
                        return [str(cand)]

        # 2. Binario directo en PATH
        located = shutil.which("hyperframes")
        if located:
            return [located]

        # 3. node_modules local en directorio actual
        for local_cand in (
            Path.cwd() / "node_modules" / ".bin" / "hyperframes.cmd",
            Path.cwd() / "node_modules" / ".bin" / "hyperframes",
        ):
            if local_cand.is_file():
                return [str(local_cand)]

        return None

    def is_available(self) -> bool:
        """Indica si existe un binario ejecutable de HyperFrames instalado."""
        return self._resolve_command_prefix() is not None

    def export_composition(
        self,
        project: Project,
        scenes: Sequence[Scene],
        cues_data: Sequence[WordTimestamp] | Sequence[dict[str, Any]] | None = None,
        workspace_root: Path | None = None,
        *,
        burn_subtitles: bool = True,
    ) -> Path:
        """Exporta un proyecto a una composición HyperFrames autocontenida."""
        ws_root = workspace_root or self.workspace.project_dir(project.id)
        hf_dir = ws_root / "hyperframes"
        hf_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copiar audio maestro localmente a hyperframes/dialogue_master.wav
        master_src: Path | None = None
        if project.master_audio_path:
            cand = Path(project.master_audio_path)
            master_src = cand if cand.is_absolute() else ws_root / cand
        if master_src is None or not master_src.is_file():
            default_audio = ws_root / "audio" / "dialogue_master.wav"
            if default_audio.is_file():
                master_src = default_audio

        has_audio = False
        if master_src is not None and master_src.is_file():
            audio_dest = hf_dir / "dialogue_master.wav"
            try:
                shutil.copy2(master_src, audio_dest)
                has_audio = True
            except OSError as exc:
                logger.warning("No se pudo copiar audio maestro a hyperframes: %s", exc)

        # 2. Copiar escenas individuales inyectando harness.js
        ordered_scenes = sorted(scenes, key=lambda s: s.scene_index)
        for scene in ordered_scenes:
            cand = Path(scene.html_path)
            scene_src = cand if cand.is_absolute() else ws_root / cand
            if scene_src.is_file():
                try:
                    content = scene_src.read_text(encoding="utf-8")
                    instrumented = inject_harness(content)
                    dest = hf_dir / f"scene_{scene.scene_index:02d}.html"
                    dest.write_text(instrumented, encoding="utf-8")
                except OSError as exc:
                    logger.warning("No se pudo preparar escena %d: %s", scene.scene_index, exc)

        # 3. Preparar palabras de subtítulos (cues)
        cue_list: list[dict[str, Any]] = []
        if cues_data:
            for item in cues_data:
                if isinstance(item, WordTimestamp):
                    cue_list.append(
                        {
                            "word": item.word,
                            "start_ms": item.start_ms,
                            "end_ms": item.end_ms,
                        }
                    )
                elif isinstance(item, dict):
                    cue_list.append(
                        {
                            "word": str(item.get("word", "")),
                            "start_ms": int(item.get("start_ms", 0)),
                            "end_ms": int(item.get("end_ms", 0)),
                        }
                    )

        # 4. Construir metadata de clips de escenas
        scenes_meta = [
            {
                "index": s.scene_index,
                "startMs": s.start_ms,
                "endMs": s.end_ms,
                "elId": f"scene-clip-{s.scene_index}",
            }
            for s in ordered_scenes
        ]

        duration_sec = max(0.001, project.total_duration_ms / 1000.0)
        cues_json = json.dumps(cue_list, ensure_ascii=False)
        scenes_json = json.dumps(scenes_meta, ensure_ascii=False)

        # 5. Generar index.html autocontenido con el contrato HyperFrames
        audio_tag = (
            f'<audio id="dialogue-master" src="dialogue_master.wav" '
            f'data-start="0" data-duration="{duration_sec:.3f}" data-track-index="1"></audio>'
            if has_audio
            else ""
        )

        clips_html = []
        for s in ordered_scenes:
            s_start = s.start_ms / 1000.0
            s_dur = max(0.001, (s.end_ms - s.start_ms) / 1000.0)
            initial_display = "block" if s.scene_index == 0 else "none"
            clips_html.append(
                f'<div class="clip scene-clip" id="scene-clip-{s.scene_index}" '
                f'data-scene-index="{s.scene_index}" data-start="{s_start:.3f}" '
                f'data-duration="{s_dur:.3f}" '
                f'style="position:absolute;top:0;left:0;width:100%;height:100%;'
                f'display:{initial_display};">'
                f'<iframe src="scene_{s.scene_index:02d}.html" '
                f'style="width:100%;height:100%;border:none;pointer-events:none;"></iframe>'
                f"</div>"
            )
        clips_str = "\n    ".join(clips_html)

        captions_tag = (
            '<div id="captions-track" data-track-kind="captions" '
            'style="position:absolute;bottom:10%;left:5%;width:90%;text-align:center;'
            "font-family:system-ui,-apple-system,sans-serif;font-size:42px;font-weight:bold;"
            'color:#ffffff;text-shadow:2px 2px 4px rgba(0,0,0,0.8);'
            'pointer-events:none;z-index:9999;">'
            "</div>"
            if burn_subtitles
            else ""
        )

        root_attrs = (
            f'data-composition-id="main" data-width="{project.width}" '
            f'data-height="{project.height}" data-fps="{project.fps}" '
            f'data-duration="{duration_sec:.3f}"'
        )

        index_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>{project.title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ width: 100%; height: 100%; overflow: hidden; background: #000; }}
    #root {{
      position: relative;
      width: {project.width}px;
      height: {project.height}px;
      overflow: hidden;
      background: #000;
    }}
  </style>
</head>
<body>
  <div id="root" {root_attrs}>
    {audio_tag}
    {clips_str}
    {captions_tag}
  </div>

  <script>
    (function () {{
      var scenes = {scenes_json};
      var cueWords = {cues_json};
      var totalDurationSec = {duration_sec:.3f};

      function updateComposition(timeMs) {{
        // 1. Activar escena correspondiente
        for (var i = 0; i < scenes.length; i++) {{
          var sc = scenes[i];
          var el = document.getElementById(sc.elId);
          var active = (timeMs >= sc.startMs && (timeMs < sc.endMs || i === scenes.length - 1));
          if (el) {{
            if (active) {{
              el.style.display = "block";
              var iframe = el.querySelector("iframe");
              var win = iframe ? iframe.contentWindow : null;
              if (win && typeof win.__setFrameTime === "function") {{
                win.__setFrameTime(Math.max(0, timeMs - sc.startMs));
              }}
            }} else {{
              el.style.display = "none";
            }}
          }}
        }}

        // 2. Subtítulos dinámicos
        var capTrack = document.getElementById("captions-track");
        if (capTrack && cueWords.length > 0) {{
          var activeWord = "";
          for (var w = 0; w < cueWords.length; w++) {{
            var wordObj = cueWords[w];
            if (timeMs >= wordObj.start_ms && timeMs <= wordObj.end_ms) {{
              activeWord = wordObj.word;
              break;
            }}
          }}
          capTrack.textContent = activeWord;
        }}
      }}

      window.__setFrameTime = updateComposition;
      if (typeof window.__registerSceneRender === "function") {{
        window.__registerSceneRender(updateComposition);
      }}

      window.__timelines = window.__timelines || {{}};
      window.__timelines["main"] = {{
        _timeSec: 0,
        seek: function (sec) {{
          this._timeSec = Number(sec) || 0;
          updateComposition(this._timeSec * 1000);
        }},
        time: function (sec) {{
          if (sec !== undefined) this.seek(sec);
          return this._timeSec;
        }},
        duration: function () {{
          return totalDurationSec;
        }},
        pause: function () {{}}
      }};
    }})();
  </script>
</body>
</html>
"""
        index_path = hf_dir / "index.html"
        index_path.write_text(index_content, encoding="utf-8")
        return index_path

    def check_project(
        self,
        project: Project,
        scenes: Sequence[Scene],
        cues_data: Sequence[WordTimestamp] | Sequence[dict[str, Any]] | None = None,
        workspace_root: Path | None = None,
        *,
        backend: RenderBackend = "auto",
    ) -> dict[str, Any]:
        """Audita visualmente la composición y las escenas con HyperFrames o Playwright."""
        ws_root = workspace_root or self.workspace.project_dir(project.id)
        index_html = self.export_composition(project, scenes, cues_data, ws_root)
        hf_dir = index_html.parent

        if backend == "hyperframes" and not self.is_available():
            raise RenderEngineError(
                "El backend 'hyperframes' fue solicitado explícitamente pero @hyperframes/cli "
                "no está disponible."
            )

        if backend in ("auto", "hyperframes") and self.is_available():
            cmd_prefix = self._resolve_command_prefix()
            assert cmd_prefix is not None
            command = [*cmd_prefix, "check", str(hf_dir), "--json"]
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                parsed = _extract_json_block(proc.stdout)
                ok = bool(parsed.get("ok", proc.returncode == 0))
                return {
                    "ok": ok,
                    "backend_used": "hyperframes",
                    "composition_html_path": str(index_html),
                    "report": parsed or {"raw_output": proc.stdout},
                }
            except (OSError, subprocess.SubprocessError) as exc:
                if backend == "hyperframes":
                    raise RenderEngineError(f"HyperFrames check falló: {exc}") from exc
                logger.warning("Fallo al ejecutar HyperFrames check, cayendo a Playwright: %s", exc)

        # Fallback a Playwright (SceneValidator)
        errors: list[str] = []
        scenes_checked = 0
        validator = self.scene_validator
        if validator is not None:
            for s in scenes:
                cand = Path(s.html_path)
                p = cand if cand.is_absolute() else ws_root / cand
                if p.is_file():
                    duration = max(1, s.end_ms - s.start_ms)
                    try:
                        res = validator.validate_file(p, duration_ms=duration)
                        scenes_checked += 1
                        if not res.valid:
                            errors.extend(res.errors)
                    except Exception as val_exc:
                        errors.append(f"Escena {s.scene_index}: {val_exc}")

        return {
            "ok": len(errors) == 0,
            "backend_used": "playwright",
            "composition_html_path": str(index_html),
            "report": {
                "scenes_checked": scenes_checked,
                "errors": errors,
            },
        }

    def render_project(
        self,
        project: Project,
        scenes: Sequence[Scene],
        cues_data: Sequence[WordTimestamp] | Sequence[dict[str, Any]] | None = None,
        workspace_root: Path | None = None,
        output_mp4_path: Path | None = None,
        *,
        burn_subtitles: bool = True,
        backend: RenderBackend = "auto",
        fallback_engine: DeterministicRenderEngine | None = None,
    ) -> dict[str, Any]:
        """Renderiza la composición a MP4 usando HyperFrames o el motor Playwright base."""
        ws_root = workspace_root or self.workspace.project_dir(project.id)
        out_mp4 = output_mp4_path or self.workspace.final_mp4_path(project.id)
        index_html = self.export_composition(
            project,
            scenes,
            cues_data,
            ws_root,
            burn_subtitles=burn_subtitles,
        )
        hf_dir = index_html.parent

        if backend == "hyperframes" and not self.is_available():
            raise RenderEngineError(
                "El backend 'hyperframes' fue solicitado explícitamente pero @hyperframes/cli "
                "no está disponible."
            )

        if backend in ("auto", "hyperframes") and self.is_available():
            cmd_prefix = self._resolve_command_prefix()
            assert cmd_prefix is not None
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            command = [
                *cmd_prefix,
                "render",
                str(hf_dir),
                "-o",
                str(out_mp4),
                "--fps",
                str(project.fps),
            ]
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                if proc.returncode == 0 and out_mp4.is_file() and out_mp4.stat().st_size > 0:
                    total_frames = total_frames_for_duration(
                        project.total_duration_ms, project.fps
                    )
                    job = RenderJob(
                        project_id=project.id,
                        status=RenderStatus.COMPLETED,
                        total_frames=total_frames,
                        rendered_frames=total_frames,
                        burn_subtitles=burn_subtitles,
                    )
                    saved_job = self.jobs.create(job)
                    with contextlib.suppress(Exception):
                        self.projects.update_status(project.id, ProjectStatus.RENDERED)

                    relative_output = str(
                        Path(
                            self.workspace.relative_project_path(project.id, out_mp4)
                        ).as_posix()
                    )
                    return {
                        "job_id": saved_job.id,
                        "project_id": project.id,
                        "status": "completed",
                        "project_status": "rendered",
                        "output_mp4_path": str(out_mp4),
                        "output_mp4_relative_path": relative_output,
                        "fps": project.fps,
                        "total_frames": total_frames,
                        "rendered_frames": total_frames,
                        "duration_ms": project.total_duration_ms,
                        "subtitles_path": str(index_html) if burn_subtitles else None,
                        "burn_subtitles": burn_subtitles,
                        "backend_used": "hyperframes",
                        "composition_html_path": str(index_html),
                    }

                # Si falló la ejecución
                error_detail = proc.stderr.strip() or proc.stdout.strip()
                if backend == "hyperframes":
                    raise RenderEngineError(
                        f"HyperFrames render falló con código {proc.returncode}: {error_detail}"
                    )
                logger.warning(
                    "HyperFrames render falló (%s), cayendo a fallback Playwright.",
                    error_detail,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                if backend == "hyperframes":
                    raise RenderEngineError(f"HyperFrames render falló: {exc}") from exc
                logger.warning("Error ejecutando HyperFrames render, cayendo a Playwright: %s", exc)

        # Fallback a Playwright
        engine = fallback_engine or self.render_engine
        words_list = None
        if cues_data:
            words_list = [c for c in cues_data if isinstance(c, WordTimestamp)]

        try:
            res = engine.render_project(
                project,
                scenes,
                burn_subtitles=burn_subtitles,
                words=words_list,
            )
        except TypeError:
            res = engine.render_project(
                project,
                scenes,
                burn_subtitles=burn_subtitles,
            )

        relative_output = str(
            Path(self.workspace.relative_project_path(project.id, res.output_path)).as_posix()
        )
        sub_path = str(res.subtitles_path) if res.subtitles_path else None
        return {
            "job_id": res.job.id,
            "project_id": project.id,
            "status": res.job.status.value,
            "project_status": "rendered",
            "output_mp4_path": str(res.output_path),
            "output_mp4_relative_path": relative_output,
            "fps": project.fps,
            "total_frames": res.total_frames,
            "rendered_frames": res.rendered_frames,
            "duration_ms": project.total_duration_ms,
            "subtitles_path": sub_path,
            "burn_subtitles": burn_subtitles,
            "backend_used": "playwright",
            "composition_html_path": str(index_html),
        }


__all__ = [
    "HyperFramesBridge",
    "RenderBackend",
]
