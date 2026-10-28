"""Unit tests for the FFmpeg audio utilities and TTS provider adapters."""

from __future__ import annotations

import io
import json
import re
import subprocess
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest

from mcp_videos.audio.ffmpeg_utils import (
    AudioProcessingError,
    canonicalize_voice_sample,
    concat_audio_segments_with_pauses,
    get_wav_duration_ms,
    normalize_audio_lufs,
)
from mcp_videos.audio.providers import (
    CloudAPITTSProvider,
    PresetLocalProvider,
    TTSConfigurationError,
    TTSProviderError,
    ZeroShotCloneProvider,
    extract_f0_hz,
    get_tts_provider,
)
from mcp_videos.config import AppSettings
from mcp_videos.core.models import VoiceProfile

SAMPLE_RATE = 24000


def tone_samples(
    duration_ms: int,
    *,
    frequency_hz: float = 180.0,
    amplitude: float = 0.35,
    sample_rate: int = SAMPLE_RATE,
    leading_silence_ms: int = 0,
    trailing_silence_ms: int = 0,
) -> np.ndarray:
    total_ms = leading_silence_ms + duration_ms + trailing_silence_ms
    total_frames = round(sample_rate * total_ms / 1000)
    signal = np.zeros(total_frames, dtype=np.float64)
    start = round(sample_rate * leading_silence_ms / 1000)
    frames = round(sample_rate * duration_ms / 1000)
    time_axis = np.arange(frames, dtype=np.float64) / sample_rate
    voice = np.zeros(frames, dtype=np.float64)
    for harmonic in range(1, 6):
        voice += (amplitude / harmonic) * np.sin(2.0 * np.pi * frequency_hz * harmonic * time_axis)
    signal[start : start + frames] = voice
    return np.clip(signal, -1.0, 1.0)


def write_wav_file(
    path: Path,
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)
    return path


def write_tone(
    path: Path,
    duration_ms: int,
    *,
    frequency_hz: float = 180.0,
    amplitude: float = 0.35,
    leading_silence_ms: int = 0,
    trailing_silence_ms: int = 0,
) -> Path:
    return write_wav_file(
        path,
        tone_samples(
            duration_ms,
            frequency_hz=frequency_hz,
            amplitude=amplitude,
            leading_silence_ms=leading_silence_ms,
            trailing_silence_ms=trailing_silence_ms,
        ),
    )


def wav_bytes(samples: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    payload = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)
    return buffer.getvalue()


def read_samples(path: Path) -> tuple[np.ndarray, wave._wave_params]:
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        payload = handle.readframes(handle.getnframes())
    return np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0, params


def rms(path: Path) -> float:
    samples, _ = read_samples(path)
    return float(np.sqrt(np.mean(samples * samples)))


def integrated_lufs(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = re.findall(r"I:\s+(-?\d+(?:\.\d+)?) LUFS", completed.stderr)
    assert matches
    return float(matches[-1])


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(_env_file=None)


@pytest.fixture
def preset_voice() -> VoiceProfile:
    return VoiceProfile(
        id="preset-narrador-es",
        name="Narrador",
        engine="kokoro",
        mode="preset",
        language="es",
        config={"voice_code": "ef_dora", "pitch_hz": 210, "speed": 1.0},
    )


@pytest.fixture
def clone_voice(tmp_path: Path) -> VoiceProfile:
    sample = write_tone(tmp_path / "reference.wav", 6000, frequency_hz=165.0)
    return VoiceProfile(
        id="clon-ingeniero-es",
        name="Ingeniero",
        engine="f5",
        mode="clone",
        language="es",
        sample_wav_path=str(sample),
        config={"ref_text": "Muestra de referencia", "speed": 1.0},
    )


class TestGetWavDuration:
    @pytest.mark.parametrize("duration_ms", [1, 250, 1000, 1234, 6000])
    def test_duration_is_exact(self, tmp_path: Path, duration_ms: int) -> None:
        path = write_tone(tmp_path / "tone.wav", duration_ms)

        assert get_wav_duration_ms(path) == duration_ms

    def test_missing_or_corrupt_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(AudioProcessingError, match="no existe"):
            get_wav_duration_ms(tmp_path / "missing.wav")

        corrupt = tmp_path / "corrupt.wav"
        corrupt.write_bytes(b"not a wav file")
        with pytest.raises(AudioProcessingError, match="WAV inválido"):
            get_wav_duration_ms(corrupt)


class TestCanonicalizeVoiceSample:
    def test_accepts_six_seconds_and_publishes_canonical_format(
        self,
        tmp_path: Path,
    ) -> None:
        source = write_tone(tmp_path / "source.wav", 6000)
        target = tmp_path / "voices" / "clon-uno.wav"

        duration_ms = canonicalize_voice_sample(source, target)

        assert duration_ms == 6000
        samples, params = read_samples(target)
        assert params.framerate == SAMPLE_RATE
        assert params.nchannels == 1
        assert params.sampwidth == 2
        assert abs(samples.size - 6000 * SAMPLE_RATE // 1000) <= 1

    @pytest.mark.parametrize("duration_ms", [1500, 35000])
    def test_rejects_samples_outside_three_to_thirty_seconds(
        self,
        tmp_path: Path,
        duration_ms: int,
    ) -> None:
        source = write_tone(tmp_path / "source.wav", duration_ms)
        target = tmp_path / "target.wav"

        with pytest.raises(AudioProcessingError, match="fuera de rango"):
            canonicalize_voice_sample(source, target)

        assert not target.exists()

    def test_trims_leading_and_trailing_silence(self, tmp_path: Path) -> None:
        source = write_tone(
            tmp_path / "source.wav",
            2000,
            leading_silence_ms=500,
            trailing_silence_ms=500,
        )
        target = tmp_path / "trimmed.wav"

        duration_ms = canonicalize_voice_sample(
            source,
            target,
            min_duration_ms=0,
        )

        assert 1900 <= duration_ms <= 2050
        assert get_wav_duration_ms(target) == duration_ms

    def test_invalid_bounds_are_rejected(self, tmp_path: Path) -> None:
        source = write_tone(tmp_path / "source.wav", 6000)

        with pytest.raises(ValueError, match="max_duration_ms"):
            canonicalize_voice_sample(
                source,
                tmp_path / "target.wav",
                min_duration_ms=5000,
                max_duration_ms=5000,
            )


class TestNormalizeAudioLufs:
    def test_normalizes_to_minus_sixteen_lufs_and_preserves_duration(
        self,
        tmp_path: Path,
    ) -> None:
        source = write_tone(tmp_path / "quiet.wav", 3000, amplitude=0.02)
        target = tmp_path / "normalized.wav"
        source_rms = rms(source)

        result = normalize_audio_lufs(source, target)

        assert result == target
        assert get_wav_duration_ms(target) == 3000
        assert rms(target) > source_rms * 4
        assert -17.0 <= integrated_lufs(target) <= -15.0
        _, params = read_samples(target)
        assert params.framerate == SAMPLE_RATE
        assert params.nchannels == 1

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"target_lufs": -80.0}, "target_lufs"),
            ({"true_peak_db": 3.0}, "true_peak_db"),
            ({"loudness_range_lu": 0.0}, "loudness_range_lu"),
        ],
    )
    def test_invalid_loudness_parameters_are_rejected(
        self,
        tmp_path: Path,
        kwargs: dict[str, float],
        message: str,
    ) -> None:
        source = write_tone(tmp_path / "source.wav", 1000)

        with pytest.raises(ValueError, match=message):
            normalize_audio_lufs(source, tmp_path / "out.wav", **kwargs)


class TestConcatAudioSegments:
    def test_one_second_segments_with_250_ms_pause_measure_2250_ms(
        self,
        tmp_path: Path,
    ) -> None:
        first = write_tone(tmp_path / "first.wav", 1000, frequency_hz=200)
        second = write_tone(tmp_path / "second.wav", 1000, frequency_hz=400)
        target = tmp_path / "master.wav"

        result = concat_audio_segments_with_pauses(
            [first, second],
            target,
            [250],
        )

        assert result == target
        assert get_wav_duration_ms(target) == 2250
        _, params = read_samples(target)
        assert params.framerate == SAMPLE_RATE
        assert params.nchannels == 1

    def test_one_pause_per_segment_adds_a_trailing_pause(
        self,
        tmp_path: Path,
    ) -> None:
        first = write_tone(tmp_path / "first.wav", 1000)
        second = write_tone(tmp_path / "second.wav", 1000)
        target = tmp_path / "master.wav"

        concat_audio_segments_with_pauses([first, second], target, [250, 100])

        assert get_wav_duration_ms(target) == 2350

    def test_integer_pause_applies_between_all_segments(
        self,
        tmp_path: Path,
    ) -> None:
        first = write_tone(tmp_path / "first.wav", 500)
        second = write_tone(tmp_path / "second.wav", 500)
        third = write_tone(tmp_path / "third.wav", 500)
        target = tmp_path / "master.wav"

        concat_audio_segments_with_pauses([first, second, third], target, 125)

        assert get_wav_duration_ms(target) == 1750

    def test_empty_segment_list_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="al menos un segmento"):
            concat_audio_segments_with_pauses([], tmp_path / "out.wav")

    def test_mismatched_pause_list_is_rejected(self, tmp_path: Path) -> None:
        first = write_tone(tmp_path / "first.wav", 500)
        second = write_tone(tmp_path / "second.wav", 500)

        with pytest.raises(ValueError, match="pauses_ms"):
            concat_audio_segments_with_pauses([first, second], tmp_path / "out.wav", [1, 2, 3])

    def test_negative_pause_is_rejected(self, tmp_path: Path) -> None:
        segment = write_tone(tmp_path / "one.wav", 500)

        with pytest.raises(ValueError, match="negativo"):
            concat_audio_segments_with_pauses(
                [segment],
                tmp_path / "out.wav",
                [-1],
            )


class TestProviderFactory:
    @pytest.mark.parametrize(
        ("engine", "mode", "sample", "expected"),
        [
            ("kokoro", "preset", False, PresetLocalProvider),
            ("piper", "preset", False, PresetLocalProvider),
            ("xtts", "clone", True, ZeroShotCloneProvider),
            ("f5", "clone", True, ZeroShotCloneProvider),
            ("elevenlabs", "preset", False, CloudAPITTSProvider),
            ("openai", "preset", False, CloudAPITTSProvider),
        ],
    )
    def test_dispatch_matches_engine_and_mode(
        self,
        tmp_path: Path,
        settings: AppSettings,
        engine: str,
        mode: str,
        sample: bool,
        expected: type,
    ) -> None:
        voice = VoiceProfile(
            id=f"voz-{engine}",
            name=engine,
            engine=engine,
            mode=mode,
            sample_wav_path=str(tmp_path / "ref.wav") if sample else None,
        )

        provider = get_tts_provider(voice, settings)
        try:
            assert isinstance(provider, expected)
        finally:
            if isinstance(provider, CloudAPITTSProvider):
                provider.close()


class TestPresetLocalProvider:
    def test_fallback_generates_audible_24khz_audio_and_scales_with_text(
        self,
        tmp_path: Path,
        settings: AppSettings,
        preset_voice: VoiceProfile,
    ) -> None:
        provider = PresetLocalProvider(settings)
        short = provider.synthesize_to_wav("Hola.", preset_voice, tmp_path / "short.wav")
        long = provider.synthesize_to_wav(
            "En dos mil seis la NASA lanzó una antena que ningún humano diseñó.",
            preset_voice,
            tmp_path / "long.wav",
        )

        _, short_params = read_samples(short)
        _, long_params = read_samples(long)
        assert short_params.framerate == SAMPLE_RATE
        assert short_params.nchannels == 1
        assert short_params.sampwidth == 2
        assert long_params.framerate == SAMPLE_RATE
        assert rms(short) > 0.01
        assert rms(long) > 0.01
        assert get_wav_duration_ms(long) > get_wav_duration_ms(short) * 2

    def test_fallback_is_deterministic(
        self,
        tmp_path: Path,
        settings: AppSettings,
        preset_voice: VoiceProfile,
    ) -> None:
        provider = PresetLocalProvider(settings)
        first = provider.synthesize_to_wav("Texto determinista.", preset_voice, tmp_path / "a.wav")
        second = provider.synthesize_to_wav("Texto determinista.", preset_voice, tmp_path / "b.wav")

        assert first.read_bytes() == second.read_bytes()

    def test_speed_shortens_the_generated_signal(
        self,
        tmp_path: Path,
        settings: AppSettings,
    ) -> None:
        slow_voice = VoiceProfile(
            id="voz-lenta",
            name="Lenta",
            engine="kokoro",
            mode="preset",
            config={"speed": 0.7},
        )
        fast_voice = VoiceProfile(
            id="voz-rapida",
            name="Rápida",
            engine="kokoro",
            mode="preset",
            config={"speed": 1.5},
        )
        provider = PresetLocalProvider(settings)
        slow = provider.synthesize_to_wav("Una frase de prueba.", slow_voice, tmp_path / "s.wav")
        fast = provider.synthesize_to_wav("Una frase de prueba.", fast_voice, tmp_path / "f.wav")

        assert get_wav_duration_ms(slow) > get_wav_duration_ms(fast)

    def test_blank_text_is_rejected(
        self,
        tmp_path: Path,
        settings: AppSettings,
        preset_voice: VoiceProfile,
    ) -> None:
        with pytest.raises(TTSProviderError, match="vacío"):
            PresetLocalProvider(settings).synthesize_to_wav(
                "   ",
                preset_voice,
                tmp_path / "out.wav",
            )


class TestZeroShotCloneProvider:
    def test_estimates_f0_from_reference_sample(self, tmp_path: Path) -> None:
        sample = write_tone(tmp_path / "reference.wav", 4000, frequency_hz=145.0)

        assert extract_f0_hz(sample) == pytest.approx(145.0, abs=6.0)

    def test_fallback_clones_reference_pitch_and_timbre(
        self,
        tmp_path: Path,
        settings: AppSettings,
        preset_voice: VoiceProfile,
    ) -> None:
        sample = write_tone(tmp_path / "reference.wav", 5000, frequency_hz=155.0)
        clone = VoiceProfile(
            id="clon-voz-grave",
            name="Clon grave",
            engine="f5",
            mode="clone",
            sample_wav_path=str(sample),
        )
        provider = ZeroShotCloneProvider(settings)
        clone_path = provider.synthesize_to_wav(
            "Esta es una frase con la voz clonada.",
            clone,
            tmp_path / "clone.wav",
        )
        preset_path = PresetLocalProvider(settings).synthesize_to_wav(
            "Esta es una frase con la voz clonada.",
            preset_voice,
            tmp_path / "preset.wav",
        )

        clone_f0 = extract_f0_hz(clone_path)
        preset_f0 = extract_f0_hz(preset_path)
        assert clone_f0 == pytest.approx(155.0, abs=8.0)
        assert abs(clone_f0 - preset_f0) > 25.0
        assert rms(clone_path) > 0.01
        _, params = read_samples(clone_path)
        assert params.framerate == SAMPLE_RATE
        assert params.nchannels == 1

    def test_missing_reference_sample_is_rejected(
        self,
        tmp_path: Path,
        settings: AppSettings,
    ) -> None:
        clone = VoiceProfile(
            id="clon-sin-muestra",
            name="Clon",
            engine="xtts",
            mode="clone",
            sample_wav_path=str(tmp_path / "missing.wav"),
        )

        with pytest.raises(TTSProviderError, match="no encontrada"):
            ZeroShotCloneProvider(settings).synthesize_to_wav(
                "Hola.",
                clone,
                tmp_path / "out.wav",
            )

    def test_silent_reference_cannot_estimate_f0(self, tmp_path: Path) -> None:
        silent = write_wav_file(tmp_path / "silence.wav", np.zeros(SAMPLE_RATE))

        with pytest.raises(TTSProviderError, match="F0"):
            extract_f0_hz(silent)


class TestCloudAPITTSProvider:
    def test_elevenlabs_request_and_canonical_output(
        self,
        tmp_path: Path,
        preset_voice: VoiceProfile,
    ) -> None:
        audio = wav_bytes(tone_samples(1200, frequency_hz=220.0))
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["key"] = request.headers.get("xi-api-key")
            captured["payload"] = request.read().decode("utf-8")
            return httpx.Response(
                200,
                content=audio,
                headers={"content-type": "audio/wav"},
            )

        settings = AppSettings(_env_file=None, elevenlabs_api_key="xi-test")
        voice = preset_voice.model_copy(
            update={
                "engine": "elevenlabs",
                "config": {"voice_id": "voice-123", "model_id": "eleven_multilingual_v2"},
            }
        )
        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = CloudAPITTSProvider(settings, client=client)

        result = provider.synthesize_to_wav("Hola ElevenLabs.", voice, tmp_path / "out.wav")

        assert str(captured["url"]).endswith("/voice-123")
        assert captured["key"] == "xi-test"
        assert "Hola ElevenLabs." in str(captured["payload"])
        assert get_wav_duration_ms(result) == 1200
        _, params = read_samples(result)
        assert params.framerate == SAMPLE_RATE
        assert params.nchannels == 1

    def test_openai_request_and_canonical_output(
        self,
        tmp_path: Path,
        preset_voice: VoiceProfile,
    ) -> None:
        audio = wav_bytes(tone_samples(1000, frequency_hz=190.0))
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            captured["payload"] = request.read().decode("utf-8")
            return httpx.Response(
                200,
                content=audio,
                headers={"content-type": "audio/wav"},
            )

        settings = AppSettings(_env_file=None, openai_api_key="sk-test")
        voice = preset_voice.model_copy(update={"engine": "openai", "config": {"voice": "nova"}})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = CloudAPITTSProvider(settings, client=client)

        result = provider.synthesize_to_wav("Hola OpenAI.", voice, tmp_path / "out.wav")

        assert str(captured["url"]).endswith("/v1/audio/speech")
        assert captured["auth"] == "Bearer sk-test"
        payload = json.loads(str(captured["payload"]))
        assert payload["voice"] == "nova"
        assert payload["response_format"] == "wav"
        assert payload["input"] == "Hola OpenAI."
        assert get_wav_duration_ms(result) == 1000

    @pytest.mark.parametrize(
        ("engine", "missing_key"),
        [
            ("elevenlabs", "ELEVENLABS_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
        ],
    )
    def test_missing_credentials_are_rejected(
        self,
        tmp_path: Path,
        preset_voice: VoiceProfile,
        engine: str,
        missing_key: str,
    ) -> None:
        settings = AppSettings(_env_file=None)
        voice = preset_voice.model_copy(
            update={
                "engine": engine,
                "config": {"voice_id": "voice-123", "voice": "alloy"},
            }
        )
        provider = CloudAPITTSProvider(
            settings,
            client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
        )

        with pytest.raises(TTSConfigurationError, match=missing_key):
            provider.synthesize_to_wav("Hola.", voice, tmp_path / "out.wav")

    def test_http_error_is_wrapped(
        self,
        tmp_path: Path,
        preset_voice: VoiceProfile,
    ) -> None:
        settings = AppSettings(_env_file=None, openai_api_key="sk-test")
        voice = preset_voice.model_copy(update={"engine": "openai", "config": {"voice": "alloy"}})

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "invalid key"})

        provider = CloudAPITTSProvider(
            settings,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(TTSProviderError, match="HTTP 401"):
            provider.synthesize_to_wav("Hola.", voice, tmp_path / "out.wav")

    def test_non_audio_payload_is_wrapped(
        self,
        tmp_path: Path,
        preset_voice: VoiceProfile,
    ) -> None:
        settings = AppSettings(_env_file=None, openai_api_key="sk-test")
        voice = preset_voice.model_copy(update={"engine": "openai", "config": {"voice": "alloy"}})

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"plain text",
                headers={"content-type": "text/plain"},
            )

        provider = CloudAPITTSProvider(
            settings,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(TTSProviderError, match="decodificable"):
            provider.synthesize_to_wav("Hola.", voice, tmp_path / "out.wav")
