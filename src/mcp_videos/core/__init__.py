"""Core domain models, workspace management and application orchestration."""

from typing import TYPE_CHECKING, Any

from mcp_videos.core.models import (
    SLUG_RE,
    SUPPORTED_FPS,
    DialogueLine,
    DialogueTimeline,
    DomainModel,
    LanguageCode,
    LineTiming,
    Project,
    ProjectStatus,
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

if TYPE_CHECKING:
    from mcp_videos.core.orchestrator import (
        DEMO_DIALOGUE,
        DEMO_PROJECT_ID,
        DEMO_SCENES,
        DEMO_TITLE,
        OrchestratorError,
        ProjectNotReadyError,
        ProjectOrchestrator,
        SceneGenerationUnavailableError,
    )

from mcp_videos.core.workspace import (
    DIALOGUE_MASTER_FILENAME,
    FINAL_VIDEO_FILENAME,
    PROJECT_SUBDIRS,
    InvalidArtifactIndexError,
    InvalidSlugError,
    PathTraversalError,
    WorkspaceError,
    WorkspaceManager,
)

__all__ = [
    "DEMO_DIALOGUE",
    "DEMO_PROJECT_ID",
    "DEMO_SCENES",
    "DEMO_TITLE",
    "DIALOGUE_MASTER_FILENAME",
    "FINAL_VIDEO_FILENAME",
    "PROJECT_SUBDIRS",
    "SLUG_RE",
    "SUPPORTED_FPS",
    "DialogueLine",
    "DialogueTimeline",
    "DomainModel",
    "InvalidArtifactIndexError",
    "InvalidSlugError",
    "LanguageCode",
    "LineTiming",
    "OrchestratorError",
    "PathTraversalError",
    "Project",
    "ProjectNotReadyError",
    "ProjectOrchestrator",
    "ProjectStatus",
    "RenderJob",
    "RenderStatus",
    "Scene",
    "SceneGenerationUnavailableError",
    "TimelineCoverageError",
    "ValidationStatus",
    "VoiceEngine",
    "VoiceMode",
    "VoiceProfile",
    "WordTimestamp",
    "WorkspaceError",
    "WorkspaceManager",
    "compute_dialogue_timeline",
    "frame_to_time_ms",
    "total_frames_for_duration",
    "validate_scenes_timeline_coverage",
]

_ORCHESTRATOR_EXPORTS = frozenset(
    {
        "DEMO_DIALOGUE",
        "DEMO_PROJECT_ID",
        "DEMO_SCENES",
        "DEMO_TITLE",
        "OrchestratorError",
        "ProjectNotReadyError",
        "ProjectOrchestrator",
        "SceneGenerationUnavailableError",
    }
)


def __getattr__(name: str) -> Any:
    if name in _ORCHESTRATOR_EXPORTS:
        from mcp_videos.core import orchestrator

        value = getattr(orchestrator, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
