"""Unit tests for domain models and temporal calculations."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mcp_videos.core.models import (
    DialogueLine,
    Project,
    RenderJob,
    RenderStatus,
    Scene,
    TimelineCoverageError,
    ValidationStatus,
    VoiceEngine,
    VoiceMode,
    VoiceProfile,
    WordTimestamp,
    compute_dialogue_timeline,
    frame_to_time_ms,
    total_frames_for_duration,
    validate_scenes_timeline_coverage,
)


def make_scene(
    scene_index: int,
    start_ms: int,
    end_ms: int,
    *,
    project_id: str = "short-antena-st5",
) -> Scene:
    return Scene(
        project_id=project_id,
        scene_index=scene_index,
        title=f"Escena {scene_index}",
        start_ms=start_ms,
        end_ms=end_ms,
        html_path=f"scenes/scene_{scene_index:02d}.html",
    )


class TestVoiceProfile:
    def test_preset_voice_is_valid_without_sample(self) -> None:
        voice = VoiceProfile(
            id="preset-narrador-es",
            name="Narrador",
            engine=VoiceEngine.KOKORO,
            mode=VoiceMode.PRESET,
            language="es",
            config={"voice_code": "ef_dora", "speed": 1.0},
        )

        assert voice.sample_wav_path is None
        assert voice.language == "es"
        assert voice.config["voice_code"] == "ef_dora"

    def test_clone_voice_requires_sample_path(self) -> None:
        with pytest.raises(ValidationError, match="sample_wav_path"):
            VoiceProfile(
                id="clon-alex",
                name="Alex clonado",
                engine="f5",
                mode="clone",
            )

    @pytest.mark.parametrize("sample_path", ["", "   ", None])
    def test_clone_voice_rejects_blank_sample_path(
        self,
        sample_path: str | None,
    ) -> None:
        with pytest.raises(ValidationError):
            VoiceProfile(
                id="clon-alex",
                name="Alex clonado",
                engine="f5",
                mode="clone",
                sample_wav_path=sample_path,
            )

    def test_clone_voice_accepts_sample_path(self) -> None:
        voice = VoiceProfile(
            id="clon-alex",
            name="Alex clonado",
            engine="xtts",
            mode="clone",
            sample_wav_path="data/voices/clon-alex.wav",
        )

        assert voice.mode is VoiceMode.CLONE

    @pytest.mark.parametrize(
        "slug",
        [
            "",
            "Mayusculas",
            "con_guion_bajo",
            "-inicio",
            "fin-",
            "doble--guion",
            "con espacio",
            "../escape",
            "con/punto",
        ],
    )
    def test_invalid_slugs_are_rejected(self, slug: str) -> None:
        with pytest.raises(ValidationError):
            VoiceProfile(
                id=slug,
                name="Voz",
                engine="kokoro",
                mode="preset",
            )

    def test_invalid_engine_and_mode_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VoiceProfile(
                id="voz-uno",
                name="Voz",
                engine="desconocido",
                mode="preset",
            )

        with pytest.raises(ValidationError):
            VoiceProfile(
                id="voz-uno",
                name="Voz",
                engine="kokoro",
                mode="otro",
            )

    def test_config_json_alias_is_accepted(self) -> None:
        voice = VoiceProfile.model_validate(
            {
                "id": "voz-uno",
                "name": "Uno",
                "engine": "kokoro",
                "mode": "preset",
                "config_json": '{"voice_code": "ef_dora", "speed": 1.1}',
            }
        )

        assert voice.config == {"voice_code": "ef_dora", "speed": 1.1}

    def test_config_defaults_are_not_shared(self) -> None:
        first = VoiceProfile(
            id="voz-uno",
            name="Uno",
            engine="kokoro",
            mode="preset",
        )
        second = VoiceProfile(
            id="voz-dos",
            name="Dos",
            engine="kokoro",
            mode="preset",
        )

        first.config["speed"] = 1.3

        assert second.config == {}


class TestProject:
    def test_defaults_match_short_video_profile(self) -> None:
        project = Project(id="short-antena-st5", title="Antena NASA")

        assert project.width == 1080
        assert project.height == 1920
        assert project.fps == 60
        assert project.total_duration_ms == 0
        assert project.status == "draft"
        assert project.master_audio_path is None

    @pytest.mark.parametrize("fps", [24, 30, 60])
    def test_supported_fps_values(self, fps: int) -> None:
        assert Project(id="proyecto-uno", title="T", fps=fps).fps == fps

    @pytest.mark.parametrize("fps", [0, 25, 45, 120])
    def test_unsupported_fps_is_rejected(self, fps: int) -> None:
        with pytest.raises(ValidationError):
            Project(id="proyecto-uno", title="T", fps=fps)

    @pytest.mark.parametrize(
        "payload",
        [
            {"width": 0},
            {"height": -1},
            {"total_duration_ms": -1},
        ],
    )
    def test_invalid_dimensions_and_duration(self, payload: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            Project(id="proyecto-uno", title="T", **payload)

    def test_json_round_trip_preserves_enums_and_dates(self) -> None:
        project = Project(
            id="short-antena-st5",
            title="La antena más rara de la NASA",
            status="audio_ready",
            total_duration_ms=4500,
        )

        dumped = project.model_dump(mode="json")
        restored = Project.model_validate(dumped)

        assert dumped["status"] == "audio_ready"
        assert isinstance(dumped["created_at"], str)
        assert restored == project

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Project(
                id="proyecto-uno",
                title="T",
                campo_desconocido=True,
            )


class TestDialogueLine:
    def test_valid_line(self) -> None:
        line = DialogueLine(
            project_id="short-antena-st5",
            line_index=0,
            voice_id="preset-narrador-es",
            speaker_label="Narrador",
            text_content="En dos mil seis la NASA lanzó una antena.",
            pause_after_ms=200,
            start_ms=0,
            end_ms=2000,
            audio_path="audio/line_00.wav",
        )

        assert line.end_ms > line.start_ms
        assert line.pause_after_ms == 200

    @pytest.mark.parametrize("pause", [-250, 0, 150, 2000])
    def test_pause_boundaries_are_accepted(self, pause: int) -> None:
        line = DialogueLine(
            project_id="proyecto-uno",
            line_index=0,
            voice_id="voz-uno",
            speaker_label="Narrador",
            text_content="Hola",
            pause_after_ms=pause,
            start_ms=0,
            end_ms=100,
        )

        assert line.pause_after_ms == pause

    @pytest.mark.parametrize("pause", [-251, 2001, 5000])
    def test_pause_out_of_range_is_rejected(self, pause: int) -> None:
        with pytest.raises(ValidationError):
            DialogueLine(
                project_id="proyecto-uno",
                line_index=0,
                voice_id="voz-uno",
                speaker_label="Narrador",
                text_content="Hola",
                pause_after_ms=pause,
                start_ms=0,
                end_ms=100,
            )

    @pytest.mark.parametrize(
        ("start_ms", "end_ms"),
        [(0, 0), (100, 100), (200, 100)],
    )
    def test_invalid_interval_is_rejected(self, start_ms: int, end_ms: int) -> None:
        with pytest.raises(ValidationError, match="end_ms"):
            DialogueLine(
                project_id="proyecto-uno",
                line_index=0,
                voice_id="voz-uno",
                speaker_label="Narrador",
                text_content="Hola",
                start_ms=start_ms,
                end_ms=end_ms,
            )

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_text_is_rejected(self, text: str) -> None:
        with pytest.raises(ValidationError):
            DialogueLine(
                project_id="proyecto-uno",
                line_index=0,
                voice_id="voz-uno",
                speaker_label="Narrador",
                text_content=text,
                start_ms=0,
                end_ms=100,
            )

    def test_negative_start_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DialogueLine(
                project_id="proyecto-uno",
                line_index=0,
                voice_id="voz-uno",
                speaker_label="Narrador",
                text_content="Hola",
                start_ms=-1,
                end_ms=100,
            )


class TestWordTimestamp:
    def test_valid_timestamp(self) -> None:
        word = WordTimestamp(
            line_id=1,
            word_index=0,
            word="NASA",
            start_ms=100,
            end_ms=420,
            confidence=0.98,
        )

        assert word.confidence == pytest.approx(0.98)

    @pytest.mark.parametrize("confidence", [-0.1, 1.01, 1.5])
    def test_confidence_out_of_range_is_rejected(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            WordTimestamp(
                line_id=1,
                word_index=0,
                word="NASA",
                start_ms=100,
                end_ms=420,
                confidence=confidence,
            )

    @pytest.mark.parametrize(
        ("start_ms", "end_ms"),
        [(100, 100), (500, 100)],
    )
    def test_invalid_interval_is_rejected(self, start_ms: int, end_ms: int) -> None:
        with pytest.raises(ValidationError):
            WordTimestamp(
                line_id=1,
                word_index=0,
                word="NASA",
                start_ms=start_ms,
                end_ms=end_ms,
            )

    def test_blank_word_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WordTimestamp(
                line_id=1,
                word_index=0,
                word="   ",
                start_ms=0,
                end_ms=10,
            )


class TestScene:
    def test_scene_defaults(self) -> None:
        scene = make_scene(0, 0, 4500)

        assert scene.validation_status == ValidationStatus.PENDING
        assert scene.validation_errors == []
        assert scene.validation_errors_json == "[]"
        assert scene.rendered_clip_path is None

    def test_validation_errors_serialize_to_json(self) -> None:
        scene = make_scene(0, 0, 4500)
        scene.validation_errors = ["ReferenceError: x is not defined"]

        assert json.loads(scene.validation_errors_json) == ["ReferenceError: x is not defined"]

    @pytest.mark.parametrize(
        ("start_ms", "end_ms"),
        [(4500, 4500), (4500, 0)],
    )
    def test_invalid_interval_is_rejected(self, start_ms: int, end_ms: int) -> None:
        with pytest.raises(ValidationError):
            make_scene(0, start_ms, end_ms)

    def test_blank_html_path_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Scene(
                project_id="proyecto-uno",
                scene_index=0,
                title="Escena",
                start_ms=0,
                end_ms=100,
                html_path="   ",
            )


class TestRenderJob:
    def test_render_job_generates_uuid_and_defaults(self) -> None:
        job = RenderJob(project_id="short-antena-st5", total_frames=270)

        uuid.UUID(job.id)
        assert job.status == RenderStatus.QUEUED
        assert job.rendered_frames == 0
        assert job.burn_subtitles is True
        assert job.completed_at is None

    def test_invalid_uuid_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="UUID"):
            RenderJob(
                id="job-123",
                project_id="proyecto-uno",
                total_frames=10,
            )

    @pytest.mark.parametrize("total_frames", [0, -1])
    def test_total_frames_must_be_positive(self, total_frames: int) -> None:
        with pytest.raises(ValidationError):
            RenderJob(project_id="proyecto-uno", total_frames=total_frames)

    def test_rendered_frames_cannot_exceed_total(self) -> None:
        with pytest.raises(ValidationError, match="rendered_frames"):
            RenderJob(
                project_id="proyecto-uno",
                total_frames=270,
                rendered_frames=271,
            )

    def test_completed_job_metadata(self) -> None:
        completed_at = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
        job = RenderJob(
            project_id="proyecto-uno",
            status="completed",
            total_frames=270,
            rendered_frames=270,
            output_mp4_path="output/final.mp4",
            completed_at=completed_at,
        )

        assert job.status is RenderStatus.COMPLETED
        assert job.completed_at == completed_at


class TestComputeDialogueTimeline:
    def test_golden_data(self) -> None:
        timeline = compute_dialogue_timeline(
            durations_ms=[2000, 2200],
            pauses_after_ms=[200, 100],
        )

        assert timeline.total_duration_ms == 4500
        assert [(line.start_ms, line.end_ms) for line in timeline.lines] == [
            (0, 2000),
            (2200, 4400),
        ]

    def test_negative_pause_creates_controlled_overlap(self) -> None:
        timeline = compute_dialogue_timeline(
            durations_ms=[1000, 1000],
            pauses_after_ms=[-250, 0],
        )

        assert [(line.start_ms, line.end_ms) for line in timeline.lines] == [
            (0, 1000),
            (750, 1750),
        ]
        assert timeline.total_duration_ms == 1750

    def test_pauses_default_to_zero(self) -> None:
        timeline = compute_dialogue_timeline([500, 500])

        assert timeline.total_duration_ms == 1000
        assert timeline.lines[1].start_ms == 500

    @pytest.mark.parametrize(
        "durations",
        [[], [0], [-1], [1000, 0]],
    )
    def test_invalid_durations_are_rejected(self, durations: list[int]) -> None:
        with pytest.raises(ValueError):
            compute_dialogue_timeline(durations, [0] * len(durations))

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="longitud"):
            compute_dialogue_timeline([1000, 1000], [0])

    @pytest.mark.parametrize("duration", ["1000", 10.5, True])
    def test_non_integer_durations_are_rejected(self, duration: object) -> None:
        with pytest.raises(ValueError, match="entero"):
            compute_dialogue_timeline([duration], [0])

    @pytest.mark.parametrize("pause", [-251, 2001])
    def test_invalid_pause_is_rejected(self, pause: int) -> None:
        with pytest.raises(ValueError):
            compute_dialogue_timeline([1000], [pause])

    def test_negative_total_timeline_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negativo"):
            compute_dialogue_timeline([100], [-250])


class TestFrameMath:
    @pytest.mark.parametrize(
        ("duration_ms", "fps", "expected"),
        [
            (4500, 60, 270),
            (1000, 24, 24),
            (1000, 30, 30),
            (1001, 30, 31),
            (123, 60, 8),
            (16, 60, 1),
        ],
    )
    def test_total_frames(
        self,
        duration_ms: int,
        fps: int,
        expected: int,
    ) -> None:
        assert total_frames_for_duration(duration_ms, fps) == expected

    @pytest.mark.parametrize("duration_ms", [0, -1])
    def test_total_frames_rejects_non_positive_duration(
        self,
        duration_ms: int,
    ) -> None:
        with pytest.raises(ValueError):
            total_frames_for_duration(duration_ms, 60)

    @pytest.mark.parametrize("fps", [1, 25, 45, 120])
    def test_total_frames_rejects_unsupported_fps(self, fps: int) -> None:
        with pytest.raises(ValueError, match="fps"):
            total_frames_for_duration(1000, fps)

    @pytest.mark.parametrize(
        ("frame_index", "fps", "expected"),
        [
            (0, 60, 0.0),
            (1, 60, 1000.0 / 60.0),
            (269, 60, 4483.333333333333),
            (24, 24, 1000.0),
            (30, 30, 1000.0),
        ],
    )
    def test_frame_to_time(
        self,
        frame_index: int,
        fps: int,
        expected: float,
    ) -> None:
        assert frame_to_time_ms(frame_index, fps) == pytest.approx(expected)

    def test_frame_index_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="frame_index"):
            frame_to_time_ms(-1, 60)

    def test_frame_math_rejects_unsupported_fps(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            frame_to_time_ms(0, 45)


class TestSceneCoverage:
    def test_contiguous_scenes_are_valid(self) -> None:
        scenes = [
            make_scene(0, 0, 2000),
            make_scene(1, 2000, 4500),
        ]

        validate_scenes_timeline_coverage(scenes, 4500)

    def test_scenes_may_be_passed_out_of_order(self) -> None:
        scenes = [
            make_scene(1, 2000, 4500),
            make_scene(0, 0, 2000),
        ]

        validate_scenes_timeline_coverage(scenes, 4500)

    def test_gap_is_reported_with_exact_bounds(self) -> None:
        scenes = [
            make_scene(0, 0, 2000),
            make_scene(1, 2100, 4500),
        ]

        with pytest.raises(TimelineCoverageError) as exc_info:
            validate_scenes_timeline_coverage(scenes, 4500)

        error = exc_info.value
        assert error.gap_start_ms == 2000
        assert error.gap_end_ms == 2100
        assert "[2000, 2100]" in str(error)

    def test_overlap_is_reported_with_exact_bounds(self) -> None:
        scenes = [
            make_scene(0, 0, 2100),
            make_scene(1, 2000, 4500),
        ]

        with pytest.raises(TimelineCoverageError) as exc_info:
            validate_scenes_timeline_coverage(scenes, 4500)

        error = exc_info.value
        assert error.overlap_start_ms == 2000
        assert error.overlap_end_ms == 2100
        assert "[2000, 2100]" in str(error)

    def test_initial_gap_is_reported(self) -> None:
        with pytest.raises(TimelineCoverageError) as exc_info:
            validate_scenes_timeline_coverage([make_scene(0, 500, 4500)], 4500)

        assert exc_info.value.gap_start_ms == 0
        assert exc_info.value.gap_end_ms == 500

    def test_trailing_gap_is_reported(self) -> None:
        with pytest.raises(TimelineCoverageError) as exc_info:
            validate_scenes_timeline_coverage([make_scene(0, 0, 4000)], 4500)

        assert exc_info.value.gap_start_ms == 4000
        assert exc_info.value.gap_end_ms == 4500

    def test_scene_exceeding_total_duration_is_reported(self) -> None:
        with pytest.raises(TimelineCoverageError) as exc_info:
            validate_scenes_timeline_coverage([make_scene(0, 0, 5000)], 4500)

        assert exc_info.value.overlap_start_ms == 4500
        assert exc_info.value.overlap_end_ms == 5000

    def test_empty_scene_list_covers_nothing(self) -> None:
        with pytest.raises(TimelineCoverageError) as exc_info:
            validate_scenes_timeline_coverage([], 4500)

        assert exc_info.value.gap_start_ms == 0
        assert exc_info.value.gap_end_ms == 4500

    @pytest.mark.parametrize("total_duration_ms", [0, -1])
    def test_non_positive_total_duration_is_rejected(
        self,
        total_duration_ms: int,
    ) -> None:
        with pytest.raises(ValueError):
            validate_scenes_timeline_coverage(
                [make_scene(0, 0, 100)],
                total_duration_ms,
            )

    def test_non_contiguous_indices_are_rejected(self) -> None:
        scenes = [
            make_scene(0, 0, 2000),
            make_scene(2, 2000, 4500),
        ]

        with pytest.raises(ValueError, match="scene_index"):
            validate_scenes_timeline_coverage(scenes, 4500)
