"""Suite de pruebas E2E del flujo híbrido HyperFrames.

Valida la integración integral de Audio-First, composición HyperFrames autocontenida,
auditoría visual y renderizado determinista tanto desde la interfaz MCP como desde la CLI.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright
from typer.testing import CliRunner

from mcp_videos import cli
from mcp_videos import mcp_server as server
from mcp_videos.audio.aligner import WordAligner
from mcp_videos.audio.dialogue_synth import DialogueSynthesizer
from mcp_videos.config import AppSettings
from mcp_videos.core.models import ProjectStatus
from mcp_videos.core.orchestrator import ProjectOrchestrator
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import SceneRepository
from mcp_videos.render.engine import DeterministicRenderEngine
from mcp_videos.scenes.generator import SceneGenerator, build_technical_explainer_scene_html
from mcp_videos.scenes.validator import SceneValidator, find_chromium_executable

WIDTH = 360
HEIGHT = 640
FPS = 60
SAMPLE_RATE = 24_000

DIALOGUE = [
    {
        "speaker_label": "Narrador",
        "voice_id": "preset-narrador-es",
        "text_content": "La NASA probó una antena imposible.",
        "pause_after_ms": 100,
    },
    {
        "speaker_label": "Ingeniero",
        "voice_id": "preset-ingeniero-es",
        "text_content": "Un algoritmo encontró la forma.",
        "pause_after_ms": 100,
    },
]

DURATIONS_MS = {
    "La NASA probó una antena imposible.": 1100,
    "Un algoritmo encontró la forma.": 900,
}

VOICES = [
    ("preset-narrador-es", "Narrador"),
    ("preset-ingeniero-es", "Ingeniero"),
]


class FixedDurationProvider:
    """Proveedor determinista de audio sintético para pruebas rápidas."""

    def synthesize_to_wav(self, text: str, voice: object, output_path: Path | str) -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = DURATIONS_MS.get(text, 1000)
        frames = round(SAMPLE_RATE * duration_ms / 1000)
        frequency = 195.0 if getattr(voice, "mode", None) == "preset" else 150.0
        samples = bytearray()
        for index in range(frames):
            value = 0.11 * math.sin(2.0 * math.pi * frequency * index / SAMPLE_RATE)
            samples.extend(round(value * 32767).to_bytes(2, "little", signed=True))
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(bytes(samples))
        return target


def chromium_available() -> bool:
    """Verifica si Chromium está operativo en el entorno actual."""
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError:
                executable = find_chromium_executable()
                if executable is None:
                    return False
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=executable,
                )
            browser.close()
    except Exception:
        return False
    return True


def probe_mp4(path: Path) -> dict[str, dict[str, object]]:
    """Extrae la información técnica de pistas de audio y video con ffprobe."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,nb_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    return {stream["codec_type"]: stream for stream in payload["streams"]}


def setup_mock_hyperframes_cli(tmp_path: Path) -> Path:
    """Crea un binario mock de @hyperframes/cli para check y render."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    py_script = bin_dir / "mock_hyperframes.py"
    py_script.write_text(
        "import sys, json, subprocess\n"
        "args = sys.argv[1:]\n"
        "if not args:\n"
        "    sys.exit(0)\n"
        "subcmd = args[0]\n"
        "if subcmd == 'check':\n"
        "    print(json.dumps({'ok': True, 'scenes_checked': 2, 'errors': []}))\n"
        "    sys.exit(0)\n"
        "elif subcmd == 'render':\n"
        "    out_idx = args.index('-o') + 1\n"
        "    out_path = args[out_idx]\n"
        "    cmd = [\n"
        "        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',\n"
        "        '-f', 'lavfi', '-i', 'color=c=blue:s=360x640:r=60:d=1',\n"
        "        '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono',\n"
        "        '-t', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',\n"
        "        '-c:a', 'aac', str(out_path)\n"
        "    ]\n"
        "    res = subprocess.run(cmd, check=False)\n"
        "    sys.exit(res.returncode)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )

    if os.name == "nt":
        cmd_file = bin_dir / "hyperframes.cmd"
        cmd_file.write_text(
            f'@echo off\n"{sys.executable}" "%~dp0mock_hyperframes.py" %*\n',
            encoding="utf-8",
        )
    else:
        sh_file = bin_dir / "hyperframes"
        sh_file.write_text(
            f'#!/bin/sh\n"{sys.executable}" "$(dirname "$0")/mock_hyperframes.py" "$@"\n',
            encoding="utf-8",
        )
        sh_file.chmod(0o755)

    return bin_dir


def build_environment(tmp_path: Path) -> tuple[ProjectOrchestrator, WorkspaceManager]:
    """Inicializa la infraestructura completa de orquestación sobre un directorio temporal."""
    settings = AppSettings(
        _env_file=None,
        mcp_videos_db_path=tmp_path / "data" / "mcp_videos.sqlite",
        mcp_videos_workspaces_dir=tmp_path / "workspaces",
        mcp_videos_voices_dir=tmp_path / "data" / "voices",
        ffmpeg_crf=28,
    )
    db = DatabaseManager(settings.mcp_videos_db_path)
    db.apply_migrations()
    workspace = WorkspaceManager(
        workspaces_root=settings.mcp_videos_workspaces_dir,
        voices_root=settings.mcp_videos_voices_dir,
    )
    synthesizer = DialogueSynthesizer(
        db,
        workspace,
        settings,
        provider_factory=lambda voice, settings: FixedDurationProvider(),
    )
    aligner = WordAligner(db, workspace, settings, enable_whisper=False)
    validator = SceneValidator(
        SceneRepository(db),
        workspace=workspace,
        width=WIDTH,
        height=HEIGHT,
        timeout_ms=20_000,
    )
    generator = SceneGenerator(settings, provider="template", validator=validator)
    engine = DeterministicRenderEngine(db, workspace, settings)
    orchestrator = ProjectOrchestrator(
        settings,
        db=db,
        workspace=workspace,
        dialogue_synthesizer=synthesizer,
        aligner=aligner,
        scene_validator=validator,
        scene_generator=generator,
        render_engine=engine,
    )
    return orchestrator, workspace


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg y ffprobe son obligatorios para la prueba E2E",
)
def test_e2e_mcp_hybrid_pipeline_export_audit_and_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valida el pipeline híbrido completo invocado a través de herramientas MCP."""
    if not chromium_available():
        pytest.skip("Chromium no disponible para validación de escenas en Playwright")

    mock_bin_dir = setup_mock_hyperframes_cli(tmp_path)
    monkeypatch.setenv("MCP_VIDEOS_HYPERFRAMES_DIR", str(mock_bin_dir))

    orchestrator, workspace = build_environment(tmp_path)
    server.set_orchestrator(orchestrator)
    project_id = "short-hybrid-mcp-e2e"

    try:
        # 1. Registrar voces
        for voice_id, name in VOICES:
            res_voice = server.register_voice(
                voice_id=voice_id,
                name=name,
                engine="kokoro",
                mode="preset",
            )
            assert res_voice.get("status") != "error", res_voice

        # 2. Crear proyecto
        res_proj = server.create_project(
            project_id=project_id,
            title="Antena NASA Híbrida",
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            description="Demostración híbrida HyperFrames",
        )
        assert res_proj.get("status") != "error", res_proj
        assert res_proj["project_id"] == project_id

        # 3. Sintetizar diálogo multi-voz y verificar dialogue_master.wav y cues.json
        res_dialogue = server.synthesize_dialogue(project_id, DIALOGUE)
        assert res_dialogue.get("status") != "error", res_dialogue
        assert res_dialogue["status"] == ProjectStatus.AUDIO_READY.value
        assert res_dialogue["words_count"] > 0

        master_wav = workspace.dialogue_master_path(project_id)
        assert master_wav.is_file(), "dialogue_master.wav debe haber sido generado"
        assert master_wav.stat().st_size > 1000

        cues_path = workspace.project_dir(project_id) / "cues.json"
        assert cues_path.is_file(), "cues.json debe haber sido generado"
        cues_data = json.loads(cues_path.read_text(encoding="utf-8"))
        assert len(cues_data["words"]) == res_dialogue["words_count"]
        assert all("start_ms" in w and "end_ms" in w for w in cues_data["words"])

        total_duration_ms = res_dialogue["total_duration_ms"]
        half_ms = total_duration_ms // 2

        # 4. Guardar y validar Escena 0 con contrato estándar __setFrameTime
        words_scene0 = orchestrator.dialogues.word_timestamps_in_window(
            0, half_ms, project_id=project_id
        )
        html_scene0 = build_technical_explainer_scene_html(
            "LA ANTENA IMPOSIBLE",
            words=words_scene0,
            start_ms=0,
            end_ms=half_ms,
            subtitle="Diseñada mediante algoritmos genéticos.",
            eyebrow="MISIÓN NASA",
            width=WIDTH,
            height=HEIGHT,
            accent="#67e8f9",
            words_are_absolute=True,
        )
        res_scene0 = server.save_and_validate_scene(
            project_id=project_id,
            scene_index=0,
            title="Hook Antena",
            start_ms=0,
            end_ms=half_ms,
            html_content=html_scene0,
            prompt="mcp-scene0",
        )
        assert res_scene0.get("status") != "error", res_scene0
        assert res_scene0["validation_status"] == "valid"
        assert res_scene0["scenes_ready"] is False

        # 5. Guardar y validar Escena 1 con contrato data-composition-id="main"
        # y window.__timelines["main"]
        dur_sec = max(0.001, (total_duration_ms - half_ms) / 1000.0)
        html_scene1 = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ width: 100%; height: 100%; background: #0b1120; color: #fff; }}
    #composition {{
      width: {WIDTH}px;
      height: {HEIGHT}px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    h1 {{ font-size: 32px; color: #38bdf8; }}
  </style>
</head>
<body>
  <div id="composition" data-composition-id="main" data-duration="{dur_sec:.3f}">
    <h1 id="title">FORMA EVOLUTIVA</h1>
  </div>
  <script>
    window.__timelines = window.__timelines || {{}};
    window.__timelines["main"] = {{
      _t: 0,
      seek: function(sec) {{
        this._t = sec;
        var el = document.getElementById("title");
        if (el) {{
          el.style.opacity = Math.min(1, sec);
        }}
      }},
      time: function(sec) {{
        if (sec !== undefined) this.seek(sec);
        return this._t;
      }},
      duration: function() {{ return {dur_sec:.3f}; }},
      pause: function() {{}}
    }};
    window.__setFrameTime = function(ms) {{
      window.__timelines["main"].seek(ms / 1000.0);
    }};
  </script>
</body>
</html>"""
        res_scene1 = server.save_and_validate_scene(
            project_id=project_id,
            scene_index=1,
            title="Forma Evolutiva",
            start_ms=half_ms,
            end_ms=total_duration_ms,
            html_content=html_scene1,
            prompt="mcp-scene1",
        )
        assert res_scene1.get("status") != "error", res_scene1
        assert res_scene1["validation_status"] == "valid"
        assert res_scene1["scenes_ready"] is True

        # 6. Auditoría visual de la composición HyperFrames
        audit_res = server.audit_project_visuals(project_id=project_id, backend="auto")
        assert audit_res.get("status") != "error", audit_res
        assert audit_res["ok"] is True
        assert audit_res["backend_used"] == "hyperframes"
        assert "composition_html_path" in audit_res

        comp_path = Path(audit_res["composition_html_path"])
        assert comp_path.is_file(), "index.html de HyperFrames debe existir"
        comp_audio = comp_path.parent / "dialogue_master.wav"
        assert comp_audio.is_file(), "hyperframes/dialogue_master.wav debe existir"

        # 7. Renderizado híbrido a MP4 determinista
        render_res = server.render_project_video(
            project_id=project_id,
            burn_subtitles=True,
            backend="auto",
        )
        assert render_res.get("status") != "error", render_res
        assert render_res["status"] == "completed"
        assert render_res["project_status"] == ProjectStatus.RENDERED.value
        assert render_res["backend_used"] == "hyperframes"
        assert "composition_html_path" in render_res

        out_mp4 = Path(render_res["output_mp4_path"])
        assert out_mp4.is_file()
        assert out_mp4.stat().st_size > 1000

        streams = probe_mp4(out_mp4)
        assert "video" in streams
        assert streams["video"]["codec_name"] == "h264"
        assert "audio" in streams
        assert streams["audio"]["codec_name"] == "aac"

    finally:
        server.set_orchestrator(None)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg y ffprobe son obligatorios para la prueba E2E",
)
def test_e2e_cli_check_and_render_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valida el flujo de auditoría y renderizado completo conducido por la CLI."""
    if not chromium_available():
        pytest.skip("Chromium no disponible para validación de escenas")

    mock_bin_dir = setup_mock_hyperframes_cli(tmp_path)
    monkeypatch.setenv("MCP_VIDEOS_HYPERFRAMES_DIR", str(mock_bin_dir))

    orchestrator, _workspace = build_environment(tmp_path)
    monkeypatch.setattr(cli, "get_orchestrator", lambda: orchestrator)
    runner = CliRunner()
    project_id = "short-hybrid-cli-e2e"

    # 1. Ejecutar doctor con --json
    doctor_res = runner.invoke(cli.app, ["doctor", "--json"])
    assert doctor_res.exit_code == 0, doctor_res.output
    doctor_payload = json.loads(doctor_res.output)
    assert doctor_payload["status"] == "ok"
    assert doctor_payload["healthy"] is True
    assert doctor_payload["binaries"]["hyperframes"] is not None

    # Registrar voces
    for voice_id, name in VOICES:
        v_res = runner.invoke(
            cli.app,
            ["register-voice", "--voice-id", voice_id, "--name", name, "--json"],
        )
        assert v_res.exit_code == 0, v_res.output

    # Crear proyecto
    p_res = runner.invoke(
        cli.app,
        [
            "create-project",
            "--project-id",
            project_id,
            "--title",
            "Antena CLI Híbrida",
            "--width",
            str(WIDTH),
            "--height",
            str(HEIGHT),
            "--fps",
            str(FPS),
            "--json",
        ],
    )
    assert p_res.exit_code == 0, p_res.output

    # Sintetizar diálogo
    script_file = tmp_path / "script.json"
    script_file.write_text(json.dumps(DIALOGUE, ensure_ascii=False), encoding="utf-8")
    s_res = runner.invoke(
        cli.app,
        ["synthesize", "--project-id", project_id, "--script-file", str(script_file), "--json"],
    )
    assert s_res.exit_code == 0, s_res.output
    s_payload = json.loads(s_res.output)
    assert s_payload["status"] == ProjectStatus.AUDIO_READY.value

    # Guardar escena
    total_duration_ms = s_payload["total_duration_ms"]
    words = orchestrator.dialogues.word_timestamps_in_window(
        0, total_duration_ms, project_id=project_id
    )
    html_content = build_technical_explainer_scene_html(
        "LA ANTENA IMPOSIBLE",
        words=words,
        start_ms=0,
        end_ms=total_duration_ms,
        subtitle="Optimizada por cálculo evolutivo.",
        eyebrow="MISIÓN NASA",
        width=WIDTH,
        height=HEIGHT,
        accent="#67e8f9",
        words_are_absolute=True,
    )
    html_file = tmp_path / "scene.html"
    html_file.write_text(html_content, encoding="utf-8")

    save_res = runner.invoke(
        cli.app,
        [
            "save-scene",
            "--project-id",
            project_id,
            "--scene-index",
            "0",
            "--start-ms",
            "0",
            "--end-ms",
            str(total_duration_ms),
            "--title",
            "Escena Única",
            "--html-file",
            str(html_file),
            "--prompt",
            "cli-test",
            "--json",
        ],
    )
    assert save_res.exit_code == 0, save_res.output
    assert json.loads(save_res.output)["validation_status"] == "valid"

    # 2. Ejecutar check --project-id ... --json
    check_res = runner.invoke(
        cli.app,
        ["check", "--project-id", project_id, "--json"],
    )
    assert check_res.exit_code == 0, check_res.output
    check_payload = json.loads(check_res.output)
    assert check_payload["ok"] is True
    assert check_payload["backend_used"] == "hyperframes"
    assert "composition_html_path" in check_payload
    assert Path(check_payload["composition_html_path"]).is_file()

    # 3. Ejecutar render --project-id ... --backend ... --json
    render_res = runner.invoke(
        cli.app,
        ["render", "--project-id", project_id, "--backend", "hyperframes", "--json"],
    )
    assert render_res.exit_code == 0, render_res.output
    render_payload = json.loads(render_res.output)
    assert render_payload["status"] == "completed"
    assert render_payload["backend_used"] == "hyperframes"
    assert Path(render_payload["output_mp4_path"]).is_file()
