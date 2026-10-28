"""Deterministic FFmpeg and WAV utilities for the audio pipeline."""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import wave
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

SAMPLE_RATE_HZ = 24000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
MIN_VOICE_SAMPLE_MS = 3000
MAX_VOICE_SAMPLE_MS = 30000
SILENCE_THRESHOLD_DB = -50.0
TARGET_LUFS = -16.0
TRUE_PEAK_DB = -1.5
LOUDNESS_RANGE_LU = 11.0


class AudioProcessingError(ValueError):
    """Raised when FFmpeg cannot read, trim, normalize or concatenate audio."""


def _as_path(path: Path | str, kind: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise AudioProcessingError(f"{kind} no existe: {candidate}")
    if not candidate.is_file():
        raise AudioProcessingError(f"{kind} no es un archivo: {candidate}")
    return candidate


def _ffmpeg_binary() -> str:
    binary = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    return binary


def _run_ffmpeg(args: Sequence[str]) -> None:
    command = [_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise AudioProcessingError(f"no se pudo ejecutar FFmpeg: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise AudioProcessingError(
            f"FFmpeg falló con código {completed.returncode}: {detail[-800:]}"
        )


@contextmanager
def _atomic_wav_target(output_path: Path | str) -> Iterator[Path]:
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=".wav",
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        yield temporary_path
        os.replace(temporary_path, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()


def _validate_int_ms(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} debe ser un entero en milisegundos")
    if value < 0:
        raise ValueError(f"{name} no puede ser negativo")
    return value


def get_wav_duration_ms(wav_path: Path | str) -> int:
    """Return the exact duration of a PCM WAV file measured from its frames."""

    path = _as_path(wav_path, "archivo WAV")
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if rate <= 0:
                raise AudioProcessingError(f"frecuencia de muestreo inválida en {path}")
            return round((frames / float(rate)) * 1000)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioProcessingError(f"WAV inválido: {path}: {exc}") from exc


def convert_to_pcm_wav(
    src_path: Path | str,
    dst_path: Path | str,
    *,
    sample_rate: int = SAMPLE_RATE_HZ,
    channels: int = CHANNELS,
) -> Path:
    """Transcode any audio source into PCM 16-bit WAV at the requested format."""

    source = _as_path(src_path, "audio de entrada")
    if sample_rate <= 0:
        raise ValueError("sample_rate debe ser mayor que 0")
    if channels <= 0:
        raise ValueError("channels debe ser mayor que 0")
    target = Path(dst_path).expanduser()

    with _atomic_wav_target(target) as temporary:
        _run_ffmpeg(
            [
                "-i",
                str(source),
                "-vn",
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(temporary),
            ]
        )
    return target


def _silence_trim_filter(threshold_db: float) -> str:
    threshold = f"{threshold_db:g}dB"
    return (
        "silenceremove="
        f"start_periods=1:start_duration=0:start_threshold={threshold},"
        "areverse,"
        "silenceremove="
        f"start_periods=1:start_duration=0:start_threshold={threshold},"
        "areverse"
    )


def canonicalize_voice_sample(
    src_path: Path | str,
    dst_path: Path | str,
    *,
    min_duration_ms: int = MIN_VOICE_SAMPLE_MS,
    max_duration_ms: int = MAX_VOICE_SAMPLE_MS,
    trim_silence: bool = True,
    silence_threshold_db: float = SILENCE_THRESHOLD_DB,
) -> int:
    """Trim extreme silence and convert a voice sample to 24 kHz mono PCM16.

    Returns the effective duration in milliseconds after trimming. Samples whose
    effective duration falls outside the configured range are rejected before the
    destination file is published.
    """

    source = _as_path(src_path, "muestra de voz")
    min_duration_ms = _validate_int_ms(min_duration_ms, "min_duration_ms")
    max_duration_ms = _validate_int_ms(max_duration_ms, "max_duration_ms")
    if max_duration_ms <= min_duration_ms:
        raise ValueError("max_duration_ms debe ser mayor que min_duration_ms")
    if silence_threshold_db >= 0:
        raise ValueError("silence_threshold_db debe ser negativo")

    target = Path(dst_path).expanduser()
    filters = ["aresample=async=1:first_pts=0"]
    if trim_silence:
        filters.append(_silence_trim_filter(silence_threshold_db))
    audio_filter = ",".join(filters)

    with _atomic_wav_target(target) as temporary:
        _run_ffmpeg(
            [
                "-i",
                str(source),
                "-vn",
                "-af",
                audio_filter,
                "-ac",
                str(CHANNELS),
                "-ar",
                str(SAMPLE_RATE_HZ),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(temporary),
            ]
        )
        duration_ms = get_wav_duration_ms(temporary)
        if duration_ms < min_duration_ms or duration_ms > max_duration_ms:
            raise AudioProcessingError(
                "duración efectiva de la muestra fuera de rango: "
                f"{duration_ms} ms; se requiere entre "
                f"{min_duration_ms / 1000:.1f} s y {max_duration_ms / 1000:.1f} s"
            )
    return duration_ms


def normalize_audio_lufs(
    src_path: Path | str,
    dst_path: Path | str,
    *,
    target_lufs: float = TARGET_LUFS,
    true_peak_db: float = TRUE_PEAK_DB,
    loudness_range_lu: float = LOUDNESS_RANGE_LU,
) -> Path:
    """Normalize audio loudness to the EBU R128 target using FFmpeg loudnorm."""

    source = _as_path(src_path, "audio de entrada")
    if not -70.0 <= target_lufs <= -5.0:
        raise ValueError("target_lufs debe estar entre -70 y -5")
    if not -9.0 <= true_peak_db <= 0.0:
        raise ValueError("true_peak_db debe estar entre -9 y 0")
    if not 1.0 <= loudness_range_lu <= 50.0:
        raise ValueError("loudness_range_lu debe estar entre 1 y 50")

    target = Path(dst_path).expanduser()
    loudnorm = f"loudnorm=I={target_lufs:g}:TP={true_peak_db:g}:LRA={loudness_range_lu:g}"

    with _atomic_wav_target(target) as temporary:
        _run_ffmpeg(
            [
                "-i",
                str(source),
                "-vn",
                "-af",
                loudnorm,
                "-ac",
                str(CHANNELS),
                "-ar",
                str(SAMPLE_RATE_HZ),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(temporary),
            ]
        )
    return target


def _write_silence_wav(
    path: Path,
    duration_ms: int,
    *,
    sample_rate: int,
    channels: int,
) -> None:
    frames = round(sample_rate * duration_ms / 1000)
    payload = b"\x00" * (frames * channels * SAMPLE_WIDTH_BYTES)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(SAMPLE_WIDTH_BYTES)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


def _resolve_pauses(
    pauses_ms: Sequence[int] | int | None,
    segment_count: int,
) -> list[int]:
    if isinstance(pauses_ms, bool):
        raise ValueError("pauses_ms debe ser un entero o una secuencia de enteros")
    if pauses_ms is None:
        return [0] * segment_count
    if isinstance(pauses_ms, int):
        _validate_int_ms(pauses_ms, "pauses_ms")
        return [pauses_ms] * (segment_count - 1) + [0]

    pauses = [
        _validate_int_ms(value, f"pauses_ms[{index}]") for index, value in enumerate(pauses_ms)
    ]
    if len(pauses) == segment_count - 1:
        pauses.append(0)
    if len(pauses) != segment_count:
        raise ValueError(
            "pauses_ms debe contener una pausa por segmento o una menos "
            f"(recibidas {len(pauses)} para {segment_count} segmentos)"
        )
    return pauses


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def concat_audio_segments_with_pauses(
    segment_paths: Sequence[Path | str],
    output_path: Path | str,
    pauses_ms: Sequence[int] | int | None = None,
    *,
    sample_rate: int = SAMPLE_RATE_HZ,
    channels: int = CHANNELS,
) -> Path:
    """Concatenate WAV segments with exact silence pauses between turns.

    ``pauses_ms`` may be one non-negative pause per segment (the last one is a
    trailing pause), one pause less, or an integer applied between segments.
    All inputs are homogenized to PCM16 at ``sample_rate`` and ``channels``.
    """

    segments = [_as_path(path, "segmento de audio") for path in segment_paths]
    if not segments:
        raise ValueError("segment_paths debe contener al menos un segmento")
    if sample_rate <= 0:
        raise ValueError("sample_rate debe ser mayor que 0")
    if channels <= 0:
        raise ValueError("channels debe ser mayor que 0")

    pauses = _resolve_pauses(pauses_ms, len(segments))
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mcp-videos-concat-") as workspace:
        workdir = Path(workspace)
        concat_entries: list[Path] = []
        for index, segment in enumerate(segments):
            normalized = convert_to_pcm_wav(
                segment,
                workdir / f"segment_{index:03d}.wav",
                sample_rate=sample_rate,
                channels=channels,
            )
            concat_entries.append(normalized)
            if pauses[index] > 0:
                silence = workdir / f"pause_{index:03d}.wav"
                _write_silence_wav(
                    silence,
                    pauses[index],
                    sample_rate=sample_rate,
                    channels=channels,
                )
                concat_entries.append(silence)

        concat_list = workdir / "concat.txt"
        concat_list.write_text(
            "".join(f"file '{_escape_concat_path(path)}'\n" for path in concat_entries),
            encoding="utf-8",
        )

        with _atomic_wav_target(target) as temporary:
            _run_ffmpeg(
                [
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    "-c:a",
                    "pcm_s16le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    str(channels),
                    "-f",
                    "wav",
                    str(temporary),
                ]
            )
    return target


__all__ = [
    "CHANNELS",
    "LOUDNESS_RANGE_LU",
    "MAX_VOICE_SAMPLE_MS",
    "MIN_VOICE_SAMPLE_MS",
    "SAMPLE_RATE_HZ",
    "SAMPLE_WIDTH_BYTES",
    "SILENCE_THRESHOLD_DB",
    "TARGET_LUFS",
    "TRUE_PEAK_DB",
    "AudioProcessingError",
    "canonicalize_voice_sample",
    "concat_audio_segments_with_pauses",
    "convert_to_pcm_wav",
    "get_wav_duration_ms",
    "normalize_audio_lufs",
]
