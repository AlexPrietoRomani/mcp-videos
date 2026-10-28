"""User-simulation test for the complete ProjectOrchestrator demo flow."""

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
from mcp_videos.core.models import ProjectStatus
from mcp_videos.core.orchestrator import ProjectOrchestrator
from mcp_videos.core.workspace import WorkspaceManager
from mcp_videos.db.connection import DatabaseManager
from mcp_videos.db.repositories import SceneRepository
from mcp_videos.render.engine import DeterministicRenderEngine
from mcp_videos.scenes.generator import SceneGenerator
from mcp_videos.scenes.validator import SceneValidator, find_chromium_executable

PROJECT_ID = "short-demo-orchestrator"
WIDTH = 360
HEIGHT = 640
FPS = 24
SAMPLE_RATE = 24000
DIALOGUE = [
    {
        "voice_id": "preset-narrador-es",
        "speaker_label": "Narrador",
        "text_content": "La NASA probó una antena imposible.",
        "pause_after_ms": 180,
    },
    {
        "voice_id": "preset-ingeniero-es",
        "speaker_label": "Ingeniero",
        "text_content": "Un algoritmo encontró la forma.",
        "pause_after_ms": 100,
    },
]
DURATIONS_MS = {
    "La NASA probó una antena imposible.": 1150,
    "Un algoritmo encontró la forma.": 950,
}


class FixedDurationProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize_to_wav(self, text: str, voice: object, output_path: Path | str) -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = DURATIONS_MS[text]
        frames = round(SAMPLE_RATE * duration_ms / 1000)
        frequency = 190.0 if getattr(voice, "mode", None) == "preset" else 145.0
        samples = bytearray()
        for index in range(frames):
            value = 0.12 * math.sin(2.0 * math.pi * frequency * index / SAMPLE_RATE)
            samples.extend(round(value * 32767).to_bytes(2, "little", signed=True))
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(bytes(samples))
        self.calls.append(text)
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


def probe_mp4(path: Path) -> tuple[dict[str, dict[str, object]], float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,nb_frames,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = {stream["codec_type"]: stream for stream in payload["streams"]}
    return streams, float(payload["format"]["duration"])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg y ffprobe son obligatorios para la simulación",
)
def test_user_builds_demo_short_end_to_end(tmp_path: Path) -> None:
    if not chromium_available():
        pytest.skip("Chromium no disponible para validar escenas")

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
    provider = FixedDurationProvider()
    synthesizer = DialogueSynthesizer(
        db,
        workspace,
        settings,
        provider_factory=lambda voice, settings: provider,
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

    payload = orchestrator.build_demo_short(
        PROJECT_ID,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        scene_count=2,
        burn_subtitles=True,
        render=True,
        dialogue=DIALOGUE,
    )

    assert provider.calls == [turn["text_content"] for turn in DIALOGUE]
    assert payload["synthesis"]["words_count"] > 0
    assert payload["project"]["status"] == ProjectStatus.RENDERED.value
    assert [scene["validation_status"] for scene in payload["scenes"]] == ["valid", "valid"]
    assert all(scene["scenes_ready"] for scene in payload["scenes"])

    render = payload["render"]
    assert render is not None
    assert render["status"] == "completed"
    assert render["project_status"] == ProjectStatus.RENDERED.value
    output_path = Path(render["output_mp4_path"])
    assert output_path == workspace.final_mp4_path(PROJECT_ID)
    assert output_path.is_file()
    assert output_path.stat().st_size > 1000
    assert render["rendered_frames"] == render["total_frames"] > 0

    subtitles_path = Path(render["subtitles_path"])
    assert subtitles_path.is_file()
    subtitles = subtitles_path.read_text(encoding="utf-8")
    assert "NASA" in subtitles
    assert "algoritmo" in subtitles

    streams, duration_s = probe_mp4(output_path)
    video = streams["video"]
    assert video["codec_name"] == "h264"
    assert (video["width"], video["height"]) == (WIDTH, HEIGHT)
    assert video["r_frame_rate"] == f"{FPS}/1"
    assert video["nb_frames"] == str(render["total_frames"])
    assert streams["audio"]["codec_name"] == "aac"
    assert duration_s == pytest.approx(payload["synthesis"]["total_duration_ms"] / 1000, abs=0.15)

    manifest = orchestrator.get_project_manifest(PROJECT_ID)
    assert manifest["project"]["status"] == ProjectStatus.RENDERED.value
    assert manifest["counts"]["lines"] == len(DIALOGUE)
    assert manifest["counts"]["words"] == payload["synthesis"]["words_count"]
    assert manifest["counts"]["scenes"] == 2
    cues_path = workspace.project_dir(PROJECT_ID) / "cues.json"
    assert cues_path.is_file()
    cues = json.loads(cues_path.read_text(encoding="utf-8"))
    assert len(cues["words"]) == payload["synthesis"]["words_count"]
