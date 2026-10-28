"""User-simulation tests for the Golden project domain and workspaces."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mcp_videos.core.models import (
    DialogueLine,
    Project,
    RenderJob,
    Scene,
    TimelineCoverageError,
    VoiceProfile,
    WordTimestamp,
    compute_dialogue_timeline,
    frame_to_time_ms,
    total_frames_for_duration,
    validate_scenes_timeline_coverage,
)
from mcp_videos.core.workspace import WorkspaceManager

PROJECT_ID = "short-antena-st5"
TOTAL_DURATION_MS = 4500
TOTAL_FRAMES = 270


@pytest.fixture
def manager(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(
        workspaces_root=tmp_path / "workspaces",
        voices_root=tmp_path / "data" / "voices",
    )


def build_golden_project(
    manager: WorkspaceManager,
) -> tuple[
    Project,
    list[VoiceProfile],
    list[DialogueLine],
    list[WordTimestamp],
    list[Scene],
    RenderJob,
]:
    manager.ensure_project_dirs(PROJECT_ID)
    manager.ensure_voices_dir()

    preset_voice = VoiceProfile(
        id="preset-narrador-es",
        name="Narrador",
        engine="kokoro",
        mode="preset",
        language="es",
        config={"voice_code": "ef_dora", "speed": 1.0},
    )
    clone_voice = VoiceProfile(
        id="clon-ingeniero-es",
        name="Ingeniero clonado",
        engine="f5",
        mode="clone",
        language="es",
        sample_wav_path="data/voices/clon-ingeniero-es.wav",
        config={"ref_text": "Muestra de voz del ingeniero", "speed": 1.0},
    )
    manager.ensure_voices_dir()
    manager.voice_sample_path("clon-ingeniero-es").write_bytes(b"RIFF....WAVE")

    timeline = compute_dialogue_timeline(
        durations_ms=[2000, 2200],
        pauses_after_ms=[200, 100],
    )
    assert timeline.total_duration_ms == TOTAL_DURATION_MS

    line_texts = [
        "En dos mil seis la NASA lanzó una antena que ningún humano diseñó.",
        "Un algoritmo evolutivo probó miles de formas hasta hallar la óptima.",
    ]
    lines = [
        DialogueLine(
            id=index + 1,
            project_id=PROJECT_ID,
            line_index=index,
            voice_id=voice.id,
            speaker_label="Narrador" if index == 0 else "Ingeniero",
            text_content=text,
            pause_after_ms=timing_pause,
            start_ms=timing.start_ms,
            end_ms=timing.end_ms,
            audio_path=manager.relative_project_path(
                PROJECT_ID,
                manager.line_audio_path(PROJECT_ID, index),
            ),
        )
        for index, (timing, text, voice, timing_pause) in enumerate(
            zip(
                timeline.lines,
                line_texts,
                [preset_voice, clone_voice],
                [200, 100],
                strict=True,
            )
        )
    ]
    for line in lines:
        manager.write_project_file(
            PROJECT_ID,
            line.audio_path or "",
            b"RIFF....WAVE",
        )

    words = [
        WordTimestamp(
            line_id=1,
            word_index=index,
            word=word,
            start_ms=index * 500,
            end_ms=(index + 1) * 500,
            confidence=0.97,
        )
        for index, word in enumerate(["En", "dos", "mil", "seis"])
    ]

    html = (
        "<html><body><h1>La antena más rara de la NASA</h1>"
        "<script>"
        "window.__setFrameTime = function(timeMs) {"
        "  document.body.style.setProperty('--progress', timeMs / 4500);"
        "};"
        "</script></body></html>"
    )
    scene_path = manager.write_scene_html(PROJECT_ID, 0, html)
    scene = Scene(
        id=1,
        project_id=PROJECT_ID,
        scene_index=0,
        title="Hook Antena Evolucionada",
        start_ms=0,
        end_ms=TOTAL_DURATION_MS,
        html_path=manager.relative_project_path(PROJECT_ID, scene_path),
        validation_status="valid",
    )

    project = Project(
        id=PROJECT_ID,
        title="La antena más rara de la NASA",
        description="Antena evolucionada por algoritmo genético.",
        width=1080,
        height=1920,
        fps=60,
        total_duration_ms=timeline.total_duration_ms,
        master_audio_path="audio/dialogue_master.wav",
        status="scenes_ready",
    )
    manager.write_project_file(
        PROJECT_ID,
        project.master_audio_path or "",
        b"RIFF....WAVE",
    )

    render_job = RenderJob(
        project_id=PROJECT_ID,
        total_frames=total_frames_for_duration(
            project.total_duration_ms,
            project.fps,
        ),
        output_mp4_path=manager.relative_project_path(
            PROJECT_ID,
            manager.final_mp4_path(PROJECT_ID),
        ),
    )

    return project, [preset_voice, clone_voice], lines, words, [scene], render_job


def test_golden_project_end_to_end(manager: WorkspaceManager) -> None:
    project, voices, lines, words, scenes, render_job = build_golden_project(manager)

    assert project.total_duration_ms == 4500
    assert render_job.total_frames == 270
    assert [(line.start_ms, line.end_ms) for line in lines] == [
        (0, 2000),
        (2200, 4400),
    ]
    assert [voice.id for voice in voices] == [
        "preset-narrador-es",
        "clon-ingeniero-es",
    ]
    assert len(words) == 4
    assert scenes[0].html_path == "scenes/scene_00.html"

    validate_scenes_timeline_coverage(scenes, project.total_duration_ms)

    frame_times = [frame_to_time_ms(index, project.fps) for index in range(render_job.total_frames)]
    assert frame_times[0] == 0.0
    assert frame_times[-1] < project.total_duration_ms
    assert frame_to_time_ms(render_job.total_frames, project.fps) == pytest.approx(
        project.total_duration_ms
    )
    assert len(frame_times) == 270

    project_dir = manager.project_dir(PROJECT_ID)
    expected_files = {
        "audio/line_00.wav",
        "audio/line_01.wav",
        "audio/dialogue_master.wav",
        "scenes/scene_00.html",
    }
    actual_files = {
        path.relative_to(project_dir).as_posix()
        for path in project_dir.rglob("*")
        if path.is_file() and path.name != ".keep"
    }
    assert expected_files <= actual_files
    assert manager.voice_sample_path("clon-ingeniero-es").is_file()

    manifest = {
        "project": project.model_dump(mode="json"),
        "voices": [voice.model_dump(mode="json") for voice in voices],
        "lines": [line.model_dump(mode="json") for line in lines],
        "words": [word.model_dump(mode="json") for word in words],
        "scenes": [scene.model_dump(mode="json") for scene in scenes],
        "render_job": render_job.model_dump(mode="json"),
    }
    manifest_path = manager.write_project_file(
        PROJECT_ID,
        "output/project_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert Project.model_validate(reloaded["project"]) == project
    assert [DialogueLine.model_validate(item) for item in reloaded["lines"]] == lines
    assert [Scene.model_validate(item) for item in reloaded["scenes"]] == scenes
    assert RenderJob.model_validate(reloaded["render_job"]) == render_job


def test_mcp_json_payload_can_be_rebuilt_as_domain_objects() -> None:
    payload = {
        "project": {
            "id": "antena-nasa",
            "title": "Antena NASA",
            "description": "Short de prueba",
            "width": 1080,
            "height": 1920,
            "fps": 60,
            "total_duration_ms": 0,
            "status": "draft",
            "created_at": "2026-10-28T12:00:00Z",
            "updated_at": "2026-10-28T12:00:00Z",
        },
        "voices": [
            {
                "id": "preset-narrador-es",
                "name": "Narrador",
                "engine": "kokoro",
                "mode": "preset",
                "language": "es",
                "sample_wav_path": None,
                "config": {"voice_code": "ef_dora"},
                "created_at": "2026-10-28T12:00:00Z",
            },
            {
                "id": "clon-ingeniero-es",
                "name": "Ingeniero",
                "engine": "f5",
                "mode": "clone",
                "language": "es",
                "sample_wav_path": "data/voices/clon-ingeniero-es.wav",
                "config": {},
                "created_at": "2026-10-28T12:00:00Z",
            },
        ],
        "lines": [
            {
                "project_id": "antena-nasa",
                "line_index": 0,
                "voice_id": "preset-narrador-es",
                "speaker_label": "Narrador",
                "text_content": "En dos mil seis la NASA lanzó una antena.",
                "pause_after_ms": 200,
                "start_ms": 0,
                "end_ms": 2000,
                "audio_path": "audio/line_00.wav",
            }
        ],
        "scenes": [
            {
                "project_id": "antena-nasa",
                "scene_index": 0,
                "title": "Hook",
                "start_ms": 0,
                "end_ms": 4500,
                "html_path": "scenes/scene_00.html",
                "prompt_used": "",
                "validation_status": "pending",
                "validation_errors": [],
                "rendered_clip_path": None,
                "updated_at": "2026-10-28T12:00:00Z",
            }
        ],
    }

    project = Project.model_validate(payload["project"])
    voices = [VoiceProfile.model_validate(item) for item in payload["voices"]]
    lines = [DialogueLine.model_validate(item) for item in payload["lines"]]
    scenes = [Scene.model_validate(item) for item in payload["scenes"]]

    assert project.id == "antena-nasa"
    assert voices[1].mode == "clone"
    assert lines[0].voice_id == "preset-narrador-es"
    assert scenes[0].validation_status == "pending"


def test_gap_report_uses_real_project_scenes() -> None:
    scenes = [
        Scene(
            project_id=PROJECT_ID,
            scene_index=0,
            title="Introducción",
            start_ms=0,
            end_ms=2000,
            html_path="scenes/scene_00.html",
        ),
        Scene(
            project_id=PROJECT_ID,
            scene_index=1,
            title="Desarrollo",
            start_ms=2100,
            end_ms=4500,
            html_path="scenes/scene_01.html",
        ),
    ]

    with pytest.raises(TimelineCoverageError) as exc_info:
        validate_scenes_timeline_coverage(scenes, TOTAL_DURATION_MS)

    assert exc_info.value.gap_start_ms == 2000
    assert exc_info.value.gap_end_ms == 2100

    scenes[1] = scenes[1].model_copy(update={"start_ms": 2000})
    validate_scenes_timeline_coverage(scenes, TOTAL_DURATION_MS)


def test_multiple_workspaces_are_created_and_cleaned_up(
    manager: WorkspaceManager,
) -> None:
    def build(index: int) -> tuple[str, str]:
        project_id = f"proyecto-{index}"
        manager.ensure_project_dirs(project_id)
        path = manager.write_scene_html(
            project_id,
            0,
            f"<html><body>Proyecto {index}</body></html>",
        )
        return project_id, manager.relative_project_path(project_id, path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(build, range(6)))

    assert [relative for _, relative in results] == ["scenes/scene_00.html"] * 6
    for project_id, _ in results:
        project_dir = manager.project_dir(project_id)
        assert (project_dir / "audio").is_dir()
        assert (project_dir / "scenes" / "scene_00.html").is_file()
        assert (project_dir / "output").is_dir()

    shutil.rmtree(manager.workspaces_root)
    assert not manager.workspaces_root.exists()
