"""SQLite persistence layer: connections, migrations and repositories."""

from mcp_videos.db.connection import DEFAULT_DB_PATH, DEFAULT_MIGRATIONS_DIR, DatabaseManager
from mcp_videos.db.repositories import (
    DialogueLineNotFoundError,
    DialogueRepository,
    ProjectNotFoundError,
    ProjectRepository,
    RenderJobNotFoundError,
    RenderJobRepository,
    RepositoryError,
    SceneNotFoundError,
    SceneRepository,
    VoiceNotFoundError,
    VoiceRepository,
    replace_project_dialogue,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_MIGRATIONS_DIR",
    "DatabaseManager",
    "DialogueLineNotFoundError",
    "DialogueRepository",
    "ProjectNotFoundError",
    "ProjectRepository",
    "RenderJobNotFoundError",
    "RenderJobRepository",
    "RepositoryError",
    "SceneNotFoundError",
    "SceneRepository",
    "VoiceNotFoundError",
    "VoiceRepository",
    "replace_project_dialogue",
]
