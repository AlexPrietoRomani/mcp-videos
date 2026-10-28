"""User-simulation tests for offline and cloud TTS voice synthesis."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest

from mcp_videos.audio.ffmpeg_utils import (
    canonicalize_voice_sample,
    concat_audio_segments_with_pauses,
    get_wav_duration_ms,
    normalize_audio_lufs,
)
from mcp_videos.audio.providers import (
    CloudAPITTSProvider,
    extract_f0_hz,
    get_tts_provider,
)
from mcp_videos.config import AppSettings
from mcp_videos.core.models import VoiceProfile
from mcp_videos.core.workspace import WorkspaceManager

SAMPLE_RATE = 24000


def render_reference_voice(path: Path, duration_ms: int = 6000) -> Path:
    sample_rate = SAMPLE_RATE
    frames = round(sample_rate * duration_ms / 1000)
    time_axis = np.arange(frames, dtype=np.float64) / sample_rate
    signal = np.zeros(frames, dtype=np.float64)
    for harmonic in range(1, 8):
        signal += (0.28 / harmonic) * np.sin(2.0 * np.pi * 158.0 * harmonic * time_axis)
    signal *= 0.5 + 0.5 * np.sin(2.0 * np.pi * 1.7 * time_axis)
    envelope = np.ones(frames, dtype=np.float64)
    ramp = round(sample_rate * 0.05)
    envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
    envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)
    signal = np.clip(signal * envelope, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.round(signal * 32767.0).astype("<i2").tobytes())
    return path


def wav_payload(duration_ms: int, frequency_hz: float = 195.0) -> bytes:
    frames = round(SAMPLE_RATE * duration_ms / 1000)
    time_axis = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    signal = 0.35 * np.sin(2.0 * np.pi * frequency_hz * time_axis)
    signal += 0.18 * np.sin(2.0 * np.pi * frequency_hz * 2.0 * time_axis)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.round(signal * 32767.0).astype("<i2").tobytes())
    return buffer.getvalue()


def sample_stats(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        payload = handle.readframes(handle.getnframes())
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    return rate, channels, width, rms


@pytest.fixture
def manager(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(
        workspaces_root=tmp_path / "workspaces",
        voices_root=tmp_path / "data" / "voices",
    )


def test_user_registers_cloned_voice_and_builds_two_voice_dialogue(
    manager: WorkspaceManager,
) -> None:
    manager.ensure_project_dirs("short-antena-st5")
    manager.ensure_voices_dir()
    raw_sample = render_reference_voice(manager.voices_root / "raw-ingreso.wav")
    canonical_sample = manager.voice_sample_path("clon-ingeniero-es")

    effective_ms = canonicalize_voice_sample(raw_sample, canonical_sample)
    assert 5900 <= effective_ms <= 6000
    rate, channels, width, rms = sample_stats(canonical_sample)
    assert (rate, channels, width) == (SAMPLE_RATE, 1, 2)
    assert rms > 0.01

    settings = AppSettings(_env_file=None)
    preset = VoiceProfile(
        id="preset-narrador-es",
        name="Narrador",
        engine="kokoro",
        mode="preset",
        language="es",
        config={"voice_code": "ef_dora", "pitch_hz": 215, "speed": 1.0},
    )
    clone = VoiceProfile(
        id="clon-ingeniero-es",
        name="Ingeniero",
        engine="f5",
        mode="clone",
        language="es",
        sample_wav_path=str(canonical_sample),
        config={"ref_text": "Muestra de voz del ingeniero", "speed": 1.0},
    )
    audio_dir = manager.audio_dir("short-antena-st5")
    raw_preset = audio_dir / "line_00_raw.wav"
    raw_clone = audio_dir / "line_01_raw.wav"

    get_tts_provider(preset, settings).synthesize_to_wav(
        "En dos mil seis la NASA lanzó una antena imposible.",
        preset,
        raw_preset,
    )
    get_tts_provider(clone, settings).synthesize_to_wav(
        "Un algoritmo evolutivo encontró una forma que nadie había imaginado.",
        clone,
        raw_clone,
    )

    for raw in (raw_preset, raw_clone):
        rate, channels, width, rms = sample_stats(raw)
        assert (rate, channels, width) == (SAMPLE_RATE, 1, 2)
        assert rms > 0.01
        assert get_wav_duration_ms(raw) > 500

    preset_f0 = extract_f0_hz(raw_preset)
    clone_f0 = extract_f0_hz(raw_clone)
    assert clone_f0 == pytest.approx(158.0, abs=8.0)
    assert abs(clone_f0 - preset_f0) > 25.0

    normalized_preset = audio_dir / "line_00.wav"
    normalized_clone = audio_dir / "line_01.wav"
    normalize_audio_lufs(raw_preset, normalized_preset)
    normalize_audio_lufs(raw_clone, normalized_clone)
    preset_ms = get_wav_duration_ms(normalized_preset)
    clone_ms = get_wav_duration_ms(normalized_clone)

    master = manager.dialogue_master_path("short-antena-st5")
    concat_audio_segments_with_pauses(
        [normalized_preset, normalized_clone],
        master,
        [250],
    )

    assert get_wav_duration_ms(master) == preset_ms + clone_ms + 250
    assert get_wav_duration_ms(master) > 2000
    rate, channels, width, rms = sample_stats(master)
    assert (rate, channels, width) == (SAMPLE_RATE, 1, 2)
    assert rms > 0.01


def test_user_synthesizes_cloud_voice_through_mock_http_transport(
    manager: WorkspaceManager,
) -> None:
    manager.ensure_project_dirs("short-antena-st5")
    settings = AppSettings(_env_file=None, openai_api_key="sk-simulation")
    voice = VoiceProfile(
        id="preset-api-openai",
        name="Voz API",
        engine="openai",
        mode="preset",
        language="es",
        config={"voice": "nova", "model": "gpt-4o-mini-tts", "speed": 1.1},
    )
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(
            200,
            content=wav_payload(1350),
            headers={"content-type": "audio/wav"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = CloudAPITTSProvider(settings, client=client)
    line_path = manager.audio_dir("short-antena-st5") / "line_00.wav"

    result = provider.synthesize_to_wav(
        "La antena evolucionó sola durante diez años.",
        voice,
        line_path,
    )

    assert len(requests_seen) == 1
    assert requests_seen[0].url.path.endswith("/v1/audio/speech")
    assert result == line_path
    rate, channels, width, rms = sample_stats(result)
    assert (rate, channels, width) == (SAMPLE_RATE, 1, 2)
    assert rms > 0.01
    assert get_wav_duration_ms(result) == 1350


def test_user_keeps_working_offline_when_model_weights_are_absent(
    manager: WorkspaceManager,
) -> None:
    manager.ensure_project_dirs("short-antena-st5")
    settings = AppSettings(_env_file=None)
    voice = VoiceProfile(
        id="preset-offline-es",
        name="Offline",
        engine="kokoro",
        mode="preset",
        language="es",
        config={
            "voice_code": "ef_dora",
            "model_path": str(manager.workspaces_root / "missing-model.onnx"),
            "voices_path": str(manager.workspaces_root / "missing-voices.bin"),
        },
    )
    output = manager.audio_dir("short-antena-st5") / "line_00.wav"

    result = get_tts_provider(voice, settings).synthesize_to_wav(
        "El pipeline completo funciona sin descargar pesos externos.",
        voice,
        output,
    )

    assert result.is_file()
    rate, channels, width, rms = sample_stats(result)
    assert (rate, channels, width) == (SAMPLE_RATE, 1, 2)
    assert rms > 0.01
    assert get_wav_duration_ms(result) > 1000
