from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass
import math
from typing import Callable

import av
import numpy as np

from local_ai.slices.voice.shared.audio_processing import normalize_audio_format


def _frame_channel_count(frame: av.AudioFrame) -> int:
    layout = getattr(frame, "layout", None)
    if layout is None:
        return 1
    nb_channels = getattr(layout, "nb_channels", None)
    if isinstance(nb_channels, int) and nb_channels > 0:
        return nb_channels
    channels = getattr(layout, "channels", None)
    if channels is None:
        return 1
    try:
        count = len(channels)
    except TypeError:
        count = 0
    return count if count > 0 else 1


def decode_audio_frame(frame: av.AudioFrame) -> np.ndarray:
    """Normalize planar or packed audio frames to mono float32 samples."""
    array = frame.to_ndarray()
    if array.ndim == 2:
        if array.shape[0] > 1:
            audio = array.mean(axis=0)
        else:
            channels = _frame_channel_count(frame)
            if channels > 1 and array.shape[1] % channels == 0:
                audio = array.reshape(-1, channels).mean(axis=1)
            else:
                audio = array.reshape(-1)
    else:
        audio = array
    audio = normalize_audio_format(audio)
    if np.issubdtype(array.dtype, np.integer):
        type_info = np.iinfo(array.dtype)
        if np.issubdtype(array.dtype, np.unsignedinteger):
            midpoint = float(type_info.max // 2 + 1)
            audio = (audio - midpoint) / midpoint
        else:
            max_value = max(abs(type_info.min), type_info.max)
            audio = audio / float(max_value)
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class MediaChunk:
    """A bounded mono audio window positioned on the media timeline."""

    audio: np.ndarray
    sample_rate: int
    start_seconds: float


def iter_media_chunks(
    path: Path,
    *,
    chunk_seconds: float = 30.0,
    open_container: Callable[..., object] = av.open,
) -> Iterator[MediaChunk]:
    """Decode only the first audio stream, keeping at most one window buffered."""
    if not math.isfinite(chunk_seconds) or not 0 < chunk_seconds <= 30:
        raise ValueError("chunk_seconds must be greater than 0 and at most 30.")
    with open_container(str(path)) as container:
        origin = (getattr(container, "start_time", None) or 0) / av.time_base
        buffer = np.empty(0, dtype=np.float32)
        used = 0
        sample_rate = 0
        window_start = 0.0
        next_time = 0.0
        decoded = False
        for frame in container.decode(audio=0):
            mono = decode_audio_frame(frame)
            if mono.size == 0:
                continue
            rate = int(frame.sample_rate)
            if rate <= 0:
                raise ValueError(f"Invalid audio sample rate: {rate}")
            frame_time = getattr(frame, "time", None)
            frame_start = float(frame_time) - origin if frame_time is not None else next_time
            if not math.isfinite(frame_start):
                raise ValueError("Invalid audio frame timestamp.")
            # Preserve audio/video offsets and gaps instead of squeezing out silence.
            discontinuity = abs(frame_start - next_time) > max(0.01, 1 / rate)
            if rate != sample_rate or discontinuity:
                if used:
                    yield MediaChunk(buffer[:used].copy(), sample_rate, window_start)
                    used = 0
                buffer = np.empty(max(1, int(rate * chunk_seconds)), dtype=np.float32)
                sample_rate = rate
            position = 0
            while position < mono.size:
                if used == 0:
                    window_start = frame_start + position / rate
                count = min(buffer.size - used, mono.size - position)
                buffer[used:used + count] = mono[position:position + count]
                used += count
                position += count
                if used == buffer.size:
                    yield MediaChunk(buffer.copy(), sample_rate, window_start)
                    used = 0
            next_time = frame_start + mono.size / rate
            decoded = True
        if used:
            yield MediaChunk(buffer[:used].copy(), sample_rate, window_start)
        if not decoded:
            raise ValueError(f"No decodable audio stream found: {path}")
