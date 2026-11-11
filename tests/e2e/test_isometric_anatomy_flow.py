"""Suite de pruebas E2E del flujo de figuras isométricas interactivas (IsoKit / Anatomy).

Valida de punta a punta la integración de Audio-First, generación de escenas
isométricas interactivas con `window.IsoKit`, validación geométrica estricta de
líneas flotantes (`window.lineEnds`), exportación de composición en HyperFrames
y renderizado final determinista a MP4 con Playwright y FFmpeg.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from mcp_videos.audio.aligner import WordAligner
from mcp_videos.audio.dialogue_synth import DialogueSynthesizer
from mcp_videos.config import AppSettings
from mcp_videos.core.models import ProjectStatus, ValidationStatus
from mcp_videos.core.orchestrator import ProjectOrchestrator
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import SceneRepository
from mcp_videos.render.engine import DeterministicRenderEngine
from mcp_videos.render.hyperframes_bridge import HyperFramesBridge
from mcp_videos.scenes.generator import SceneGenerator
from mcp_videos.scenes.validator import SceneValidator, find_chromium_executable

WIDTH = 540
HEIGHT = 960
FPS = 30
SAMPLE_RATE = 24_000

DIALOGUE = [
    {
        "speaker_label": "Narrador",
        "voice_id": "preset-narrador-es",
        "text_content": "Iniciando calibración del instrumento isométrico.",
        "pause_after_ms": 100,
    },
    {
        "speaker_label": "Ingeniero",
        "voice_id": "preset-ingeniero-es",
        "text_content": "Ejes ortogonales y resortes cinemáticos alineados.",
        "pause_after_ms": 100,
    },
]

DURATIONS_MS = {
    "Iniciando calibración del instrumento isométrico.": 1200,
    "Ejes ortogonales y resortes cinemáticos alineados.": 1100,
}

VOICES = [
    ("preset-narrador-es", "Narrador"),
    ("preset-ingeniero-es", "Ingeniero"),
]


class FixedDurationProvider:
    """Proveedor determinista de audio sintético para pruebas rápidas y reproducibles."""

    def synthesize_to_wav(self, text: str, voice: object, output_path: Path | str) -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = DURATIONS_MS.get(text, 1000)
        frames = round(SAMPLE_RATE * duration_ms / 1000)
        frequency = 210.0 if getattr(voice, "mode", None) == "preset" else 165.0
        samples = bytearray()
        for index in range(frames):
            value = 0.12 * math.sin(2.0 * math.pi * frequency * index / SAMPLE_RATE)
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


def build_environment(
    tmp_path: Path,
) -> tuple[ProjectOrchestrator, WorkspaceManager, SceneValidator, SceneGenerator]:
    """Inicializa la infraestructura completa de orquestación con proveedores deterministas."""
    settings = AppSettings(
        _env_file=None,
        mcp_videos_db_path=tmp_path / "data" / "mcp_videos.sqlite",
        mcp_videos_workspaces_dir=tmp_path / "workspaces",
        mcp_videos_voices_dir=tmp_path / "data" / "voices",
        ffmpeg_crf=28,
        default_video_fps=FPS,
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
        timeout_ms=25_000,
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
    return orchestrator, workspace, validator, generator


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg y ffprobe son obligatorios para la prueba E2E",
)
def test_e2e_isometric_anatomy_flow(tmp_path: Path) -> None:
    """Valida el flujo E2E completo con figuras isométricas interactivas y render MP4."""
    if not chromium_available():
        pytest.skip("Chromium no disponible para validación y renderizado con Playwright")

    orchestrator, workspace, validator, generator = build_environment(tmp_path)
    project_id = "short-iso-e2e"

    # 1. Registrar catálogo de voces
    for voice_id, name in VOICES:
        res_voice = orchestrator.register_voice(
            voice_id=voice_id,
            name=name,
            engine="kokoro",
            mode="preset",
        )
        assert res_voice["voice_id"] == voice_id

    # 2. Inicializar proyecto vertical 540x960 @ 30 FPS
    res_proj = orchestrator.create_project(
        project_id=project_id,
        title="Calibración Isométrica de Alta Precisión",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        description="Flujo E2E demostrando máquinas isométricas IsoKit",
    )
    assert res_proj["project_id"] == project_id

    # 3. Sintetizar diálogo multivoz y alinear timestamps de palabras
    res_dialogue = orchestrator.synthesize_and_align(project_id, DIALOGUE)
    assert res_dialogue["status"] == ProjectStatus.AUDIO_READY.value
    assert res_dialogue["words_count"] > 0

    master_wav = workspace.dialogue_master_path(project_id)
    assert master_wav.is_file(), "dialogue_master.wav debe haberse generado"
    assert master_wav.stat().st_size > 1000

    cues_path = workspace.project_dir(project_id) / "cues.json"
    assert cues_path.is_file(), "cues.json debe haberse generado"
    cues_data = json.loads(cues_path.read_text(encoding="utf-8"))
    assert len(cues_data["words"]) == res_dialogue["words_count"]

    total_duration_ms: int = res_dialogue["total_duration_ms"]
    half_ms = total_duration_ms // 2

    # 4. Generar y validar 2 escenas isométricas interactivas
    words_scene0 = orchestrator.dialogues.word_timestamps_in_window(
        0, half_ms, project_id=project_id
    )
    scene_0 = generator.generate_scene(
        project_id=project_id,
        scene_index=0,
        title="Calibrador Isométrico",
        words=words_scene0,
        start_ms=0,
        end_ms=half_ms,
        workspace=workspace,
        template_name="isometric_instrument.html",
        check_iso_lines=True,
        width=WIDTH,
        height=HEIGHT,
    )
    assert scene_0.validation_status is ValidationStatus.VALID
    assert scene_0.validation_errors == []

    words_scene1 = orchestrator.dialogues.word_timestamps_in_window(
        half_ms, total_duration_ms, project_id=project_id
    )
    scene_1 = generator.generate_scene(
        project_id=project_id,
        scene_index=1,
        title="Cinemática y Resortes",
        words=words_scene1,
        start_ms=half_ms,
        end_ms=total_duration_ms,
        workspace=workspace,
        template_name="isometric_instrument.html",
        check_iso_lines=True,
        width=WIDTH,
        height=HEIGHT,
    )
    assert scene_1.validation_status is ValidationStatus.VALID
    assert scene_1.validation_errors == []

    # 5. Comprobar auditoría geométrica estricta de líneas con window.lineEnds
    file_p0 = workspace.resolve_project_path(project_id, scene_0.html_path)
    val_res_0 = validator.validate_file(
        file_p0,
        duration_ms=half_ms,
        check_iso_lines=True,
        width=WIDTH,
        height=HEIGHT,
    )
    assert val_res_0.valid is True
    assert val_res_0.iso_line_problems == []

    file_p1 = workspace.resolve_project_path(project_id, scene_1.html_path)
    val_res_1 = validator.validate_file(
        file_p1,
        duration_ms=total_duration_ms - half_ms,
        check_iso_lines=True,
        width=WIDTH,
        height=HEIGHT,
    )
    assert val_res_1.valid is True
    assert val_res_1.iso_line_problems == []

    content_0 = file_p0.read_text(encoding="utf-8")
    assert "window.IsoKit" in content_0 or "IsoKit" in content_0
    assert ".iso-plate" in content_0

    # 6. Exportar composición HyperFrames autocontenida
    project_model = orchestrator.projects.get_or_raise(project_id)
    all_words = orchestrator.dialogues.word_timestamps_in_window(
        0, total_duration_ms, project_id=project_id
    )
    comp_html = orchestrator.hyperframes_bridge.export_composition(
        project_model,
        [scene_0, scene_1],
        cues_data=all_words,
    )
    assert comp_html.is_file()
    assert (comp_html.parent / "scene_00.html").is_file()
    assert (comp_html.parent / "scene_01.html").is_file()
    assert (comp_html.parent / "dialogue_master.wav").is_file()

    comp_text = comp_html.read_text(encoding="utf-8")
    assert "scene_00.html" in comp_text
    assert "scene_01.html" in comp_text

    # 7. Auditar visualmente el proyecto con el backend Playwright
    audit_res = orchestrator.audit_project_visuals(project_id, backend="playwright")
    assert audit_res["ok"] is True
    assert audit_res["backend_used"] == "playwright"
    assert audit_res["report"]["scenes_checked"] == 2
    assert audit_res["report"]["errors"] == []

    # 8. Renderizar el video final determinista a 30 FPS con Playwright y FFmpeg
    render_res = orchestrator.render_project(
        project_id,
        burn_subtitles=True,
        backend="playwright",
    )
    assert render_res["status"] == "completed"
    assert render_res["project_status"] == "rendered"
    assert render_res["backend_used"] == "playwright"

    out_mp4 = Path(render_res["output_mp4_path"])
    assert out_mp4.is_file(), f"El archivo {out_mp4} debe existir"
    assert out_mp4.stat().st_size > 1000, "El MP4 debe contener datos de video"

    # 9. Inspeccionar contenedor y codecs con ffprobe
    streams = probe_mp4(out_mp4)
    assert "video" in streams, "El contenedor MP4 debe contener una pista de video"
    video_stream = streams["video"]
    assert video_stream["codec_name"] == "h264"
    assert int(video_stream["width"]) == WIDTH
    assert int(video_stream["height"]) == HEIGHT

    assert "audio" in streams, "El contenedor MP4 debe contener una pista de audio"
    audio_stream = streams["audio"]
    assert audio_stream["codec_name"] == "aac"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg y ffprobe son obligatorios para la prueba E2E",
)
def test_e2e_isometric_prompt_keyword_detection_and_export(tmp_path: Path) -> None:
    """Valida que un prompt con palabras clave isométricas active IsoKit y pase la auditoría."""
    if not chromium_available():
        pytest.skip("Chromium no disponible para validación con Playwright")

    orchestrator, workspace, validator, generator = build_environment(tmp_path)
    project_id = "short-iso-keyword-e2e"

    for voice_id, name in VOICES:
        orchestrator.register_voice(voice_id=voice_id, name=name, engine="kokoro", mode="preset")

    orchestrator.create_project(
        project_id=project_id,
        title="Dial Isométrico",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
    )

    synth_res = orchestrator.synthesize_and_align(
        project_id,
        [
            {
                "speaker_label": "Narrador",
                "voice_id": "preset-narrador-es",
                "text_content": "Iniciando calibración del instrumento isométrico.",
                "pause_after_ms": 50,
            }
        ],
    )
    total_ms = synth_res["total_duration_ms"]
    words = orchestrator.dialogues.word_timestamps_in_window(0, total_ms, project_id=project_id)

    # Generación sin template_name explícito pero con prompt isométrico
    scene = generator.generate_scene(
        project_id=project_id,
        scene_index=0,
        title="Calibrador de Precisión con Dial",
        words=words,
        start_ms=0,
        end_ms=total_ms,
        workspace=workspace,
        prompt_used="calibrador isométrico con dial de medición",
        check_iso_lines=True,
        width=WIDTH,
        height=HEIGHT,
    )

    assert scene.validation_status is ValidationStatus.VALID
    assert scene.validation_errors == []

    file_path = workspace.resolve_project_path(project_id, scene.html_path)
    content = file_path.read_text(encoding="utf-8")
    assert "60 FPS · ISOKIT" in content or "window.IsoKit" in content or ".iso-plate" in content

    # Validación geométrica directa de líneas
    val_res = validator.validate_file(
        file_path,
        duration_ms=total_ms,
        check_iso_lines=True,
        width=WIDTH,
        height=HEIGHT,
    )
    assert val_res.valid is True
    assert val_res.iso_line_problems == []

    # Exportación HyperFrames
    project_model = orchestrator.projects.get_or_raise(project_id)
    bridge = HyperFramesBridge(
        AppSettings(_env_file=None),
        db_path=orchestrator.settings.mcp_videos_db_path,
        workspace=workspace,
        scene_validator=validator,
    )
    comp_html = bridge.export_composition(project_model, [scene], cues_data=words)
    assert comp_html.is_file()
    assert (comp_html.parent / "scene_00.html").is_file()
    assert (comp_html.parent / "dialogue_master.wav").is_file()
