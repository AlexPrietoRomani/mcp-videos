"""Carga y validacion centralizada de la configuracion de mcp-videos."""

import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

InferenceDevice = Literal["cpu", "cuda", "auto"]
ResolvedDevice = Literal["cpu", "cuda"]
VideoFps = Literal[24, 30, 60]


class AppSettings(BaseSettings):
    """Configuracion tipada e inmutable leida de variables de entorno y `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    mcp_videos_db_path: Path = Path("data/mcp_videos.sqlite")
    mcp_videos_workspaces_dir: Path = Path("workspaces")
    mcp_videos_voices_dir: Path = Path("data/voices")

    local_inference_device: InferenceDevice = "auto"
    whisper_model_size: str = "small"
    whisper_compute_type: str = "int8"

    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen2.5-coder:14b"

    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None

    preview_port: int = Field(default=8080, ge=1, le=65535)
    default_video_fps: VideoFps = 60
    ffmpeg_crf: int = Field(default=18, ge=0, le=51)

    @field_validator("default_video_fps", mode="before")
    @classmethod
    def _coerce_video_fps(cls, value: object) -> object:
        """Convierte el FPS leido de `.env`/entorno a entero antes de validar el literal."""
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return value

    @property
    def resolved_device(self) -> ResolvedDevice:
        """Dispositivo efectivo de inferencia tras resolver `auto`."""
        return resolve_device(self.local_inference_device)

    def ensure_base_dirs(self) -> None:
        """Crea de forma idempotente los directorios base de datos y workspaces."""
        self.mcp_videos_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.mcp_videos_workspaces_dir.mkdir(parents=True, exist_ok=True)
        self.mcp_videos_voices_dir.mkdir(parents=True, exist_ok=True)


def _cuda_available() -> bool:
    """Detecta CUDA de forma barata mediante el binario `nvidia-smi`."""
    return shutil.which("nvidia-smi") is not None


def resolve_device(preferred: InferenceDevice | None = None) -> ResolvedDevice:
    """Resuelve el dispositivo de inferencia solicitado o leido del entorno.

    `auto` usa CUDA cuando `nvidia-smi` esta disponible y cae a CPU en caso contrario.
    """
    requested = preferred
    if requested is None:
        requested = os.getenv("LOCAL_INFERENCE_DEVICE", "auto")
    requested = requested.strip().lower()

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    if requested == "auto":
        return "cuda" if _cuda_available() else "cpu"
    raise ValueError(f"Dispositivo de inferencia no soportado: {requested!r}")
