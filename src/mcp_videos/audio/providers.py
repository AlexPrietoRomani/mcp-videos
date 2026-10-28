"""Pluggable text-to-speech providers for preset, cloned and cloud voices."""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from mcp_videos.audio.ffmpeg_utils import (
    SAMPLE_RATE_HZ,
    AudioProcessingError,
    convert_to_pcm_wav,
)
from mcp_videos.config import AppSettings
from mcp_videos.core.models import VoiceEngine, VoiceMode, VoiceProfile

DEFAULT_F0_HZ = 165.0
MIN_F0_HZ = 60.0
MAX_F0_HZ = 400.0
ELEVENLABS_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
OPENAI_SPEECH_ENDPOINT = "https://api.openai.com/v1/audio/speech"
VOWEL_FORMANTS: dict[str, tuple[float, float, float]] = {
    "a": (800.0, 1200.0, 2500.0),
    "e": (500.0, 1800.0, 2500.0),
    "i": (300.0, 2300.0, 3000.0),
    "o": (500.0, 900.0, 2400.0),
    "u": (320.0, 800.0, 2300.0),
}
DEFAULT_FORMANTS: tuple[float, float, float] = (500.0, 1500.0, 2500.0)
VOWELS = "aeiouáéíóúü"
WORD_PATTERN = re.compile(r"[^\W\d_]+|[.,;:!?…]", flags=re.UNICODE)


class TTSProviderError(RuntimeError):
    """Raised when a TTS backend cannot synthesize the requested line."""


class TTSConfigurationError(TTSProviderError):
    """Raised when a provider is missing required configuration or credentials."""


@dataclass(frozen=True)
class AcousticProfile:
    """Deterministic acoustic parameters used by the local formant synthesizer."""

    f0_hz: float = DEFAULT_F0_HZ
    formant_scale: float = 1.0
    harmonic_rolloff: float = 0.95
    breathiness: float = 0.02


def _resolve_engine(voice: VoiceProfile) -> VoiceEngine:
    try:
        return VoiceEngine(voice.engine)
    except ValueError as exc:
        raise TTSProviderError(f"motor TTS no soportado: {voice.engine!r}") from exc


def _resolve_mode(voice: VoiceProfile) -> VoiceMode:
    try:
        return VoiceMode(voice.mode)
    except ValueError as exc:
        raise TTSProviderError(f"modo de voz no soportado: {voice.mode!r}") from exc


def _validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise TTSProviderError("el texto a sintetizar no puede estar vacío")
    return text.strip()


def _config_float(
    config: dict[str, Any],
    keys: tuple[str, ...],
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    for key in keys:
        value = config.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        return min(maximum, max(minimum, number))
    return default


def _stable_seed(voice: VoiceProfile) -> int:
    payload = f"{voice.id}|{voice.engine.value}|{voice.language.value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _decode_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(wav_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise TTSProviderError(f"no se pudo leer la muestra de voz {wav_path}: {exc}") from exc

    if sample_rate <= 0 or channels <= 0:
        raise TTSProviderError(f"formato WAV inválido en {wav_path}")

    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values >= 1 << 23, values - (1 << 24), values)
        data = values.astype(np.float64) / float(1 << 23)
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float64) / float(1 << 31)
    else:
        raise TTSProviderError(f"ancho de muestra no soportado: {sample_width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def extract_f0_hz(
    wav_path: Path | str,
    *,
    min_hz: float = MIN_F0_HZ,
    max_hz: float = MAX_F0_HZ,
) -> float:
    """Estimate the fundamental frequency of a voice sample via autocorrelation."""

    if min_hz <= 0 or max_hz <= min_hz:
        raise ValueError("rango de F0 inválido")
    path = Path(wav_path).expanduser()
    if not path.is_file():
        raise TTSProviderError(f"muestra de voz no encontrada: {path}")

    samples, sample_rate = _decode_wav_mono(path)
    if samples.size == 0:
        raise TTSProviderError(f"muestra de voz vacía: {path}")

    max_hz = min(max_hz, sample_rate / 3.0)
    frame_length = max(64, int(sample_rate * 0.04))
    hop = max(32, frame_length // 2)
    max_frames = 120
    total_frames = max(1, (samples.size - frame_length) // hop + 1)
    starts = np.linspace(
        0,
        max(0, samples.size - frame_length),
        num=min(max_frames, total_frames),
    ).astype(int)
    window = np.hanning(frame_length)
    min_lag = max(1, int(sample_rate / max_hz))
    max_lag = min(frame_length - 1, int(sample_rate / min_hz))
    if max_lag <= min_lag:
        raise TTSProviderError("muestra demasiado corta para estimar F0")

    estimates: list[float] = []
    for start in starts:
        frame = samples[start : start + frame_length]
        if frame.size < frame_length:
            continue
        frame = frame - float(frame.mean())
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < 1e-4:
            continue
        spectrum = frame * window
        autocorrelation = np.correlate(spectrum, spectrum, mode="full")[frame_length - 1 :]
        if autocorrelation[0] <= 0:
            continue
        autocorrelation = autocorrelation / autocorrelation[0]
        region = autocorrelation[min_lag : max_lag + 1]
        if region.size == 0:
            continue
        strongest = float(region.max())
        if strongest < 0.25:
            continue
        candidates = np.flatnonzero(region >= strongest * 0.85)
        lag = int(candidates[0] + min_lag) if candidates.size else int(np.argmax(region) + min_lag)
        if 1 <= lag < autocorrelation.size - 1:
            left = float(autocorrelation[lag - 1])
            center = float(autocorrelation[lag])
            right = float(autocorrelation[lag + 1])
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                lag = lag + round(0.5 * (left - right) / denominator)
        frequency = sample_rate / max(1, lag)
        if min_hz <= frequency <= max_hz:
            estimates.append(frequency)

    if not estimates:
        raise TTSProviderError(f"no se pudo estimar F0 en la muestra {path}")
    return float(np.median(estimates))


def _spectral_centroid_hz(wav_path: Path) -> float:
    samples, sample_rate = _decode_wav_mono(wav_path)
    limit = min(samples.size, sample_rate * 3)
    if limit < 64:
        return 1400.0
    window = samples[:limit] * np.hanning(limit)
    magnitudes = np.abs(np.fft.rfft(window))
    frequencies = np.fft.rfftfreq(limit, d=1.0 / sample_rate)
    mask = frequencies >= 50.0
    total = float(magnitudes[mask].sum())
    if total <= 0:
        return 1400.0
    return float((frequencies[mask] * magnitudes[mask]).sum() / total)


def _profile_from_sample(voice: VoiceProfile) -> AcousticProfile:
    if not voice.sample_wav_path:
        raise TTSProviderError(f"la voz clonada {voice.id!r} no tiene muestra de voz")
    sample_path = Path(voice.sample_wav_path).expanduser()
    f0_hz = extract_f0_hz(sample_path)
    f0_hz = _config_float(
        voice.config,
        ("pitch_hz", "f0_hz", "base_f0_hz"),
        f0_hz,
        minimum=MIN_F0_HZ,
        maximum=MAX_F0_HZ,
    )
    centroid = _spectral_centroid_hz(sample_path)
    formant_scale = _config_float(
        voice.config,
        ("formant_scale",),
        min(1.18, max(0.85, centroid / 1400.0)),
        minimum=0.7,
        maximum=1.4,
    )
    brightness = min(1.0, max(0.0, (centroid - 500.0) / 2500.0))
    harmonic_rolloff = _config_float(
        voice.config,
        ("harmonic_rolloff",),
        1.15 - 0.35 * brightness,
        minimum=0.6,
        maximum=1.4,
    )
    return AcousticProfile(
        f0_hz=f0_hz,
        formant_scale=formant_scale,
        harmonic_rolloff=harmonic_rolloff,
        breathiness=_config_float(
            voice.config,
            ("breathiness",),
            0.015 + 0.02 * brightness,
            minimum=0.0,
            maximum=0.12,
        ),
    )


def _profile_from_preset(voice: VoiceProfile) -> AcousticProfile:
    seed = _stable_seed(voice)
    gender = str(voice.config.get("gender", "")).strip().lower()
    default_f0 = 120.0 + (seed % 1400) / 20.0
    if gender in {"female", "femenino", "mujer"}:
        default_f0 = 185.0 + (seed % 450) / 10.0
    elif gender in {"male", "masculino", "hombre"}:
        default_f0 = 105.0 + (seed % 350) / 10.0
    f0_hz = _config_float(
        voice.config,
        ("pitch_hz", "f0_hz", "base_f0_hz"),
        default_f0,
        minimum=MIN_F0_HZ,
        maximum=MAX_F0_HZ,
    )
    formant_scale = _config_float(
        voice.config,
        ("formant_scale",),
        0.9 + (seed % 220) / 1000.0,
        minimum=0.7,
        maximum=1.4,
    )
    return AcousticProfile(
        f0_hz=f0_hz,
        formant_scale=formant_scale,
        harmonic_rolloff=_config_float(
            voice.config,
            ("harmonic_rolloff",),
            0.85 + (seed % 300) / 1000.0,
            minimum=0.6,
            maximum=1.4,
        ),
        breathiness=_config_float(
            voice.config,
            ("breathiness",),
            0.01 + (seed % 30) / 1000.0,
            minimum=0.0,
            maximum=0.12,
        ),
    )


def _syllables(word: str) -> list[str]:
    groups = re.findall(f"[{VOWELS}]+", word.lower())
    return groups if groups else [word.lower() or "a"]


def _vowel_key(group: str) -> str:
    for character in group:
        normalized = (
            character.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ü", "u")
        )
        if normalized in VOWEL_FORMANTS:
            return normalized
    return "a"


def _segments_from_text(
    text: str,
    speed: float,
    f0_hz: float,
) -> list[tuple[float, float, float, tuple[float, float, float]]]:
    tokens = WORD_PATTERN.findall(text)
    if not tokens:
        tokens = [text]

    base_syllable_s = 0.115 / speed
    segments: list[tuple[float, float, float, tuple[float, float, float]]] = []
    cursor = 0.0
    for token in tokens:
        if token in ".,;:!?…":
            pause = 0.22 if token in ".!?…" else 0.12
            cursor += pause / speed
            continue
        syllables = _syllables(token)
        for index, syllable in enumerate(syllables):
            duration = base_syllable_s * (0.9 + 0.12 * min(len(syllable), 4))
            contour = 1.0 + 0.045 * math.sin(index + 0.5) - 0.02 * index
            formants = VOWEL_FORMANTS[_vowel_key(syllable)]
            segments.append((cursor, duration, f0_hz * contour, formants))
            cursor += duration + (0.012 / speed)
        cursor += 0.045 / speed
    return segments


def _synthesize_acoustic(
    text: str,
    profile: AcousticProfile,
    speed: float,
) -> tuple[np.ndarray, int]:
    segments = _segments_from_text(text, speed, profile.f0_hz)
    if not segments:
        segments = [(0.0, 0.2 / speed, profile.f0_hz, DEFAULT_FORMANTS)]

    total_seconds = segments[-1][0] + segments[-1][1] + 0.06
    sample_count = max(1, math.ceil(total_seconds * SAMPLE_RATE_HZ))
    audio = np.zeros(sample_count, dtype=np.float64)
    max_harmonic = max(1, min(40, int((SAMPLE_RATE_HZ / 2.0) / profile.f0_hz)))

    for start_s, duration_s, f0_hz, formants in segments:
        start = int(start_s * SAMPLE_RATE_HZ)
        count = max(1, int(duration_s * SAMPLE_RATE_HZ))
        end = min(sample_count, start + count)
        if end <= start:
            continue
        time_axis = np.arange(end - start, dtype=np.float64) / SAMPLE_RATE_HZ
        attack = max(1, int(0.012 * SAMPLE_RATE_HZ))
        release = max(1, int(min(0.04, duration_s * 0.35) * SAMPLE_RATE_HZ))
        envelope = np.ones(end - start, dtype=np.float64)
        attack_samples = min(attack, envelope.size)
        release_samples = min(release, envelope.size)
        envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, endpoint=False)
        envelope[-release_samples:] *= np.linspace(1.0, 0.0, release_samples)
        envelope *= np.sin(np.pi * np.linspace(0.0, 1.0, envelope.size)) ** 0.25

        pitch = f0_hz * (1.0 + 0.035 * np.sin(2.0 * np.pi * 2.5 * time_axis))
        phase = 2.0 * np.pi * np.cumsum(pitch) / SAMPLE_RATE_HZ
        syllable = np.zeros(end - start, dtype=np.float64)
        for harmonic in range(1, max_harmonic + 1):
            harmonic_hz = harmonic * f0_hz
            resonance = 0.0
            for formant in formants:
                shifted = formant * profile.formant_scale
                bandwidth = max(80.0, shifted * 0.28)
                resonance += math.exp(-0.5 * ((harmonic_hz - shifted) / bandwidth) ** 2)
            amplitude = (1.0 / harmonic**profile.harmonic_rolloff) * (0.45 + 1.6 * resonance)
            syllable += amplitude * np.sin(harmonic * phase)

        if profile.breathiness > 0:
            noise = np.sin(
                2.0 * np.pi * 4700.0 * time_axis + np.sin(2.0 * np.pi * 31.0 * time_axis)
            )
            syllable += (
                profile.breathiness * noise * (0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * time_axis))
            )

        audio[start:end] += syllable * envelope

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-9:
        raise TTSProviderError("la síntesis acústica produjo una señal vacía")
    audio = np.tanh((audio / peak) * 1.4) * 0.72
    return audio, SAMPLE_RATE_HZ


def _write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    payload = np.round(clipped * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)
    return path


def _write_temporary_audio(suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix="mcp-videos-tts-", suffix=suffix)
    os.close(descriptor)
    return Path(name)


def _canonicalize_generated_audio(source: Path, output_path: Path) -> Path:
    if source.resolve() == output_path.resolve():
        return output_path
    convert_to_pcm_wav(source, output_path, sample_rate=SAMPLE_RATE_HZ)
    return output_path


class TTSProvider(ABC):
    """Strategy interface implemented by every TTS backend."""

    @abstractmethod
    def synthesize_to_wav(
        self,
        text: str,
        voice: VoiceProfile,
        output_path: Path | str,
    ) -> Path:
        """Synthesize ``text`` with ``voice`` into a PCM WAV file."""


class PresetLocalProvider(TTSProvider):
    """Kokoro/Piper adapter with a deterministic local formant fallback."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings()

    def synthesize_to_wav(
        self,
        text: str,
        voice: VoiceProfile,
        output_path: Path | str,
    ) -> Path:
        content = _validate_text(text)
        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        speed = _config_float(
            voice.config,
            ("speed", "rate"),
            1.0,
            minimum=0.5,
            maximum=2.0,
        )

        engine = _resolve_engine(voice)
        if engine is VoiceEngine.KOKORO and self._synthesize_kokoro(content, voice, target, speed):
            return target
        if engine is VoiceEngine.PIPER and self._synthesize_piper(content, voice, target):
            return target

        profile = _profile_from_preset(voice)
        samples, sample_rate = _synthesize_acoustic(content, profile, speed)
        return _write_pcm16_wav(target, samples, sample_rate)

    def _synthesize_kokoro(
        self,
        text: str,
        voice: VoiceProfile,
        target: Path,
        speed: float,
    ) -> bool:
        model_path = voice.config.get("model_path") or voice.config.get("kokoro_model_path")
        voices_path = voice.config.get("voices_path") or voice.config.get("kokoro_voices_path")
        if not model_path or not voices_path:
            return False
        model_file = Path(str(model_path)).expanduser()
        voices_file = Path(str(voices_path)).expanduser()
        if not model_file.is_file() or not voices_file.is_file():
            return False
        try:
            module = importlib.import_module("kokoro_onnx")
        except ImportError:
            return False

        voice_code = str(
            voice.config.get("voice_code")
            or voice.config.get("voice")
            or voice.config.get("speaker_id")
            or "ef_dora"
        )
        language = str(voice.config.get("lang") or voice.language.value)
        try:
            engine = module.Kokoro(str(model_file), str(voices_file))
            samples, sample_rate = engine.create(
                text,
                voice=voice_code,
                speed=speed,
                lang=language,
            )
        except Exception as exc:
            raise TTSProviderError(f"Kokoro no pudo sintetizar la línea: {exc}") from exc
        return self._publish_samples(samples, int(sample_rate), target)

    def _synthesize_piper(
        self,
        text: str,
        voice: VoiceProfile,
        target: Path,
    ) -> bool:
        model_path = voice.config.get("model_path") or voice.config.get("piper_model_path")
        binary = voice.config.get("piper_binary") or shutil.which("piper")
        if not model_path or not binary or not Path(str(model_path)).expanduser().is_file():
            return False
        descriptor = subprocess.run(
            [
                str(binary),
                "--model",
                str(Path(str(model_path)).expanduser()),
                "--output_file",
                str(target),
            ],
            input=text,
            text=True,
            capture_output=True,
            check=False,
        )
        if descriptor.returncode != 0:
            detail = (descriptor.stderr or "").strip()
            raise TTSProviderError(f"Piper falló: {detail[-600:]}")
        if not target.is_file() or target.stat().st_size == 0:
            raise TTSProviderError("Piper no generó un archivo WAV válido")
        temporary = _write_temporary_audio(".wav")
        try:
            convert_to_pcm_wav(target, temporary, sample_rate=SAMPLE_RATE_HZ)
            target.write_bytes(temporary.read_bytes())
        finally:
            temporary.unlink(missing_ok=True)
        return True

    @staticmethod
    def _publish_samples(samples: Any, sample_rate: int, target: Path) -> bool:
        array = np.asarray(samples, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return False
        if sample_rate == SAMPLE_RATE_HZ:
            _write_pcm16_wav(target, array, sample_rate)
            return True
        temporary = _write_temporary_audio(".wav")
        try:
            _write_pcm16_wav(temporary, array, sample_rate)
            convert_to_pcm_wav(temporary, target, sample_rate=SAMPLE_RATE_HZ)
        finally:
            temporary.unlink(missing_ok=True)
        return True


class ZeroShotCloneProvider(TTSProvider):
    """XTTS/F5 adapter with local F0-guided formant cloning as offline fallback."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings()

    def synthesize_to_wav(
        self,
        text: str,
        voice: VoiceProfile,
        output_path: Path | str,
    ) -> Path:
        content = _validate_text(text)
        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if _resolve_mode(voice) is not VoiceMode.CLONE or not voice.sample_wav_path:
            raise TTSProviderError("ZeroShotCloneProvider requiere una voz en modo clone")

        sample_path = Path(voice.sample_wav_path).expanduser()
        if not sample_path.is_file():
            raise TTSProviderError(f"muestra de voz no encontrada: {sample_path}")

        speed = _config_float(
            voice.config,
            ("speed", "rate"),
            1.0,
            minimum=0.5,
            maximum=2.0,
        )
        engine = _resolve_engine(voice)
        if engine is VoiceEngine.XTTS and self._synthesize_xtts(
            content, voice, sample_path, target, speed
        ):
            return target
        if engine is VoiceEngine.F5 and self._synthesize_f5(content, voice, sample_path, target):
            return target

        profile = _profile_from_sample(voice)
        samples, sample_rate = _synthesize_acoustic(content, profile, speed)
        return _write_pcm16_wav(target, samples, sample_rate)

    def _synthesize_xtts(
        self,
        text: str,
        voice: VoiceProfile,
        sample_path: Path,
        target: Path,
        speed: float,
    ) -> bool:
        model_name = voice.config.get("xtts_model") or voice.config.get("model_name")
        if not model_name or not bool(voice.config.get("allow_model_download", False)):
            return False
        try:
            module = importlib.import_module("TTS.api")
        except ImportError:
            return False

        temporary = _write_temporary_audio(".wav")
        try:
            try:
                engine = module.TTS(
                    model_name=str(model_name),
                    gpu=self.settings.resolved_device == "cuda",
                )
                engine.tts_to_file(
                    text=text,
                    speaker_wav=str(sample_path),
                    language=voice.language.value,
                    file_path=str(temporary),
                    speed=speed,
                )
            except Exception as exc:
                raise TTSProviderError(f"XTTS no pudo sintetizar la línea: {exc}") from exc
            _canonicalize_generated_audio(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def _synthesize_f5(
        self,
        text: str,
        voice: VoiceProfile,
        sample_path: Path,
        target: Path,
    ) -> bool:
        checkpoint = voice.config.get("f5_checkpoint") or voice.config.get("ckpt_file")
        if not checkpoint or not Path(str(checkpoint)).expanduser().is_file():
            return False
        try:
            module = importlib.import_module("f5_tts.api")
        except ImportError:
            return False

        temporary = _write_temporary_audio(".wav")
        try:
            try:
                engine = module.F5TTS(
                    model=str(voice.config.get("f5_model", "F5TTS_v1_Base")),
                    ckpt_file=str(Path(str(checkpoint)).expanduser()),
                    vocab_file=voice.config.get("f5_vocab_file"),
                )
                if hasattr(engine, "infer"):
                    engine.infer(
                        ref_file=str(sample_path),
                        ref_text=str(voice.config.get("ref_text", "")),
                        gen_text=text,
                        file_wave=str(temporary),
                    )
                else:
                    engine.sample(
                        ref_file=str(sample_path),
                        ref_text=str(voice.config.get("ref_text", "")),
                        gen_text=text,
                        file_wave=str(temporary),
                    )
            except Exception as exc:
                raise TTSProviderError(f"F5-TTS no pudo sintetizar la línea: {exc}") from exc
            _canonicalize_generated_audio(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return True


class CloudAPITTSProvider(TTSProvider):
    """ElevenLabs and OpenAI speech adapters over HTTP."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.settings = settings or AppSettings()
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> CloudAPITTSProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def synthesize_to_wav(
        self,
        text: str,
        voice: VoiceProfile,
        output_path: Path | str,
    ) -> Path:
        content = _validate_text(text)
        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        engine = _resolve_engine(voice)
        if engine is VoiceEngine.ELEVENLABS:
            response = self._request_elevenlabs(content, voice)
        elif engine is VoiceEngine.OPENAI:
            response = self._request_openai(content, voice)
        else:
            raise TTSConfigurationError(f"motor de API no soportado: {engine.value}")
        return self._persist_response(response, target)

    def _request_elevenlabs(self, text: str, voice: VoiceProfile) -> httpx.Response:
        api_key = (self.settings.elevenlabs_api_key or "").strip()
        if not api_key:
            raise TTSConfigurationError("ELEVENLABS_API_KEY es obligatoria")
        voice_id = str(
            voice.config.get("voice_id")
            or voice.config.get("voice_code")
            or voice.config.get("speaker_id")
            or ""
        ).strip()
        if not voice_id:
            raise TTSConfigurationError("la voz de ElevenLabs requiere config['voice_id']")
        payload: dict[str, Any] = {
            "text": text,
            "model_id": str(voice.config.get("model_id", "eleven_multilingual_v2")),
        }
        if isinstance(voice.config.get("voice_settings"), dict):
            payload["voice_settings"] = voice.config["voice_settings"]
        try:
            return self._client.post(
                ELEVENLABS_ENDPOINT.format(voice_id=voice_id),
                headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise TTSProviderError(f"ElevenLabs no respondió: {exc}") from exc

    def _request_openai(self, text: str, voice: VoiceProfile) -> httpx.Response:
        api_key = (self.settings.openai_api_key or "").strip()
        if not api_key:
            raise TTSConfigurationError("OPENAI_API_KEY es obligatoria")
        payload: dict[str, Any] = {
            "model": str(voice.config.get("model", "gpt-4o-mini-tts")),
            "voice": str(voice.config.get("voice", "alloy")),
            "input": text,
            "response_format": "wav",
        }
        speed = _config_float(
            voice.config,
            ("speed", "rate"),
            1.0,
            minimum=0.25,
            maximum=4.0,
        )
        if speed != 1.0:
            payload["speed"] = speed
        try:
            return self._client.post(
                OPENAI_SPEECH_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise TTSProviderError(f"OpenAI no respondió: {exc}") from exc

    @staticmethod
    def _persist_response(response: httpx.Response, target: Path) -> Path:
        if response.status_code >= 400:
            detail = response.text[:600] if response.content else ""
            raise TTSProviderError(f"la API TTS respondió HTTP {response.status_code}: {detail}")
        payload = response.content
        if not payload:
            raise TTSProviderError("la API TTS devolvió una respuesta vacía")

        content_type = response.headers.get("content-type", "").lower()
        if payload.startswith(b"RIFF") or "wav" in content_type:
            suffix = ".wav"
        elif "mpeg" in content_type or "mp3" in content_type or payload.startswith(b"ID3"):
            suffix = ".mp3"
        elif "ogg" in content_type:
            suffix = ".ogg"
        elif "webm" in content_type:
            suffix = ".webm"
        else:
            suffix = ".bin"

        temporary = _write_temporary_audio(suffix)
        try:
            temporary.write_bytes(payload)
            try:
                convert_to_pcm_wav(temporary, target, sample_rate=SAMPLE_RATE_HZ)
            except AudioProcessingError as exc:
                raise TTSProviderError(
                    f"la respuesta TTS no es un audio decodificable: {exc}"
                ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        return target


LocalPresetProvider = PresetLocalProvider
APITTSProvider = CloudAPITTSProvider


def get_tts_provider(
    voice: VoiceProfile,
    settings: AppSettings | None = None,
) -> TTSProvider:
    """Return the adapter that matches the voice engine and registration mode."""

    engine = _resolve_engine(voice)
    mode = _resolve_mode(voice)
    if engine in {VoiceEngine.ELEVENLABS, VoiceEngine.OPENAI}:
        return CloudAPITTSProvider(settings)
    if mode is VoiceMode.CLONE or engine in {VoiceEngine.XTTS, VoiceEngine.F5}:
        return ZeroShotCloneProvider(settings)
    return PresetLocalProvider(settings)


__all__ = [
    "APITTSProvider",
    "AcousticProfile",
    "CloudAPITTSProvider",
    "LocalPresetProvider",
    "PresetLocalProvider",
    "TTSConfigurationError",
    "TTSProvider",
    "TTSProviderError",
    "ZeroShotCloneProvider",
    "extract_f0_hz",
    "get_tts_provider",
]
