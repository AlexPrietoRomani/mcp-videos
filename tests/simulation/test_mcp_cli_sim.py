"""User-simulation tests for the MCP tool surface and the CLI studio flow."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright
from typer.testing import CliRunner

import mcp_videos.cli as cli
import mcp_videos.mcp_server as server
from mcp_videos.audio.aligner import WordAligner
from mcp_videos.audio.dialogue_synth import DialogueSynthesizer
from mcp_videos.config import AppSettings
from mcp_videos.core.models import ProjectStatus
from mcp_videos.core.orchestrator import ProjectOrchestrator
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import SceneRepository
from mcp_videos.render.engine import DeterministicRenderEngine
from mcp_videos.scenes.generator import (
    SceneGenerator,
    build_technical_explainer_scene_html,
)
from mcp_videos.scenes.validator import SceneValidator, find_chromium_executable

WIDTH = 360
HEIGHT = 640
FPS = 24
SAMPLE_RATE = 24000
DIALOGUE = [
    {
        "voice_id": "preset-narrador-es",
        "speaker_label": "Narrador",
        "text_content": "La NASA probó una antena imposible.",
        "pause_after_ms": 150,
    },
    {
        "voice_id": "preset-ingeniero-es",
        "speaker_label": "Ingeniero",
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
    def synthesize_to_wav(self, text: str, voice: object, output_path: Path | str) -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = DURATIONS_MS[text]
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


def build_environment(tmp_path: Path) -> tuple[ProjectOrchestrator, WorkspaceManager]:
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


def scene_html(orchestrator: ProjectOrchestrator, project_id: str) -> tuple[str, int]:
    project = orchestrator.get_project(project_id)
    words = orchestrator.dialogues.word_timestamps_in_window(
        0,
        project.total_duration_ms,
        project_id=project_id,
    )
    html = build_technical_explainer_scene_html(
        "LA ANTENA IMPOSIBLE",
        words=words,
        start_ms=0,
        end_ms=project.total_duration_ms,
        subtitle="Ligera, flexible y desplegable en órbita.",
        eyebrow="MISIÓN NASA",
        width=WIDTH,
        height=HEIGHT,
        accent="#67e8f9",
        words_are_absolute=True,
    )
    return html, project.total_duration_ms


def probe_mp4(path: Path) -> dict[str, dict[str, object]]:
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


def mcp_payload(result: Any) -> Any:
    payload = result.structured_content
    if payload is None:
        return json.loads(result.content[0].text)
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    return asyncio.run(server.mcp.call_tool(name, arguments))


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg y ffprobe son obligatorios para la simulación",
)
def test_user_drives_cli_from_voices_to_final_mp4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not chromium_available():
        pytest.skip("Chromium no disponible para validar escenas")

    orchestrator, workspace = build_environment(tmp_path)
    monkeypatch.setattr(cli, "get_orchestrator", lambda: orchestrator)
    runner = CliRunner()
    project_id = "short-cli-sim"

    for voice_id, name in VOICES:
        result = runner.invoke(
            cli.app,
            [
                "register-voice",
                "--voice-id",
                voice_id,
                "--name",
                name,
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output

    created = runner.invoke(
        cli.app,
        [
            "create-project",
            "--project-id",
            project_id,
            "--title",
            "La antena imposible",
            "--width",
            str(WIDTH),
            "--height",
            str(HEIGHT),
            "--fps",
            str(FPS),
            "--json",
        ],
    )
    assert created.exit_code == 0, created.output

    script = tmp_path / "script.json"
    script.write_text(json.dumps(DIALOGUE, ensure_ascii=False), encoding="utf-8")
    synthesized = runner.invoke(
        cli.app,
        [
            "synthesize",
            "--project-id",
            project_id,
            "--script-file",
            str(script),
            "--json",
        ],
    )
    assert synthesized.exit_code == 0, synthesized.output
    assert json.loads(synthesized.output)["status"] == ProjectStatus.AUDIO_READY.value

    html, total_duration_ms = scene_html(orchestrator, project_id)
    html_file = tmp_path / "scene.html"
    html_file.write_text(html, encoding="utf-8")
    saved = runner.invoke(
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
            "LA ANTENA IMPOSIBLE",
            "--html-file",
            str(html_file),
            "--prompt",
            "cli-sim",
            "--json",
        ],
    )
    assert saved.exit_code == 0, saved.output
    assert json.loads(saved.output)["validation_status"] == "valid"

    rendered = runner.invoke(
        cli.app,
        ["render", "--project-id", project_id, "--json"],
    )
    assert rendered.exit_code == 0, rendered.output
    render_payload = json.loads(rendered.output)
    assert render_payload["status"] == "completed"
    assert render_payload["project_status"] == ProjectStatus.RENDERED.value

    output_path = Path(render_payload["output_mp4_path"])
    assert output_path == workspace.final_mp4_path(project_id)
    assert output_path.is_file()
    streams = probe_mp4(output_path)
    assert (streams["video"]["width"], streams["video"]["height"]) == (WIDTH, HEIGHT)
    assert streams["video"]["r_frame_rate"] == f"{FPS}/1"
    assert streams["audio"]["codec_name"] == "aac"
    subtitles = Path(render_payload["subtitles_path"]).read_text(encoding="utf-8")
    assert "NASA" in subtitles
    assert "algoritmo" in subtitles
    assert orchestrator.get_project(project_id).status is ProjectStatus.RENDERED

    voices = runner.invoke(cli.app, ["voices", "--json"])
    assert voices.exit_code == 0
    assert {voice["voice_id"] for voice in json.loads(voices.output)} == {
        voice_id for voice_id, _ in VOICES
    }


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg y ffprobe son obligatorios para la simulación",
)
def test_agent_drives_every_mcp_tool_to_final_mp4(tmp_path: Path) -> None:
    if not chromium_available():
        pytest.skip("Chromium no disponible para validar escenas")

    orchestrator, workspace = build_environment(tmp_path)
    server.set_orchestrator(orchestrator)
    project_id = "short-mcp-sim"
    try:
        for voice_id, name in VOICES:
            registered = call_tool(
                "register_voice",
                {"voice_id": voice_id, "name": name, "language": "es"},
            )
            assert registered.is_error is False
            assert mcp_payload(registered)["status"] == "ready"

        created = mcp_payload(
            call_tool(
                "create_project",
                {
                    "project_id": project_id,
                    "title": "La antena imposible",
                    "width": WIDTH,
                    "height": HEIGHT,
                    "fps": FPS,
                },
            )
        )
        assert created["project_id"] == project_id

        synthesized = mcp_payload(
            call_tool(
                "synthesize_dialogue",
                {"project_id": project_id, "lines": DIALOGUE},
            )
        )
        assert synthesized["status"] == ProjectStatus.AUDIO_READY.value
        assert synthesized["words_count"] > 0

        html, total_duration_ms = scene_html(orchestrator, project_id)
        saved = mcp_payload(
            call_tool(
                "save_and_validate_scene",
                {
                    "project_id": project_id,
                    "scene_index": 0,
                    "title": "LA ANTENA IMPOSIBLE",
                    "start_ms": 0,
                    "end_ms": total_duration_ms,
                    "html_content": html,
                    "prompt": "mcp-sim",
                },
            )
        )
        assert saved["validation_status"] == "valid"
        assert saved["scenes_ready"] is True

        rendered = mcp_payload(
            call_tool(
                "render_project_video",
                {"project_id": project_id, "burn_subtitles": True},
            )
        )
        assert rendered["status"] == "completed"
        assert rendered["project_status"] == ProjectStatus.RENDERED.value
        output_path = Path(rendered["output_mp4_path"])
        assert output_path == workspace.final_mp4_path(project_id)
        assert output_path.is_file()
        streams = probe_mp4(output_path)
        assert streams["video"]["codec_name"] == "h264"
        assert streams["video"]["nb_frames"] == str(rendered["total_frames"])
        assert streams["audio"]["codec_name"] == "aac"

        voices = mcp_payload(call_tool("list_voices", {"language": "es"}))
        assert {voice["voice_id"] for voice in voices} == {voice_id for voice_id, _ in VOICES}
    finally:
        server.set_orchestrator(None)
