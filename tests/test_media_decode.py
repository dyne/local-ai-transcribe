from __future__ import annotations

import pathlib

import numpy as np
import pytest

from local_ai.slices.voice.shared.media_decode import decode_audio_frame, iter_media_chunks


class FakeFrame:
    class _Layout:
        def __init__(self, channels: int) -> None:
            self.channels = [object() for _ in range(channels)]
            self.nb_channels = channels

    def __init__(self, array: np.ndarray, sample_rate: int, channels: int = 1) -> None:
        self._array = array
        self.sample_rate = sample_rate
        self.layout = self._Layout(channels)

    def to_ndarray(self) -> np.ndarray:
        return self._array


def test_decode_audio_frame_downmixes_and_scales_integer_audio() -> None:
    frame = FakeFrame(np.array([[0, 32767], [0, 32767]], dtype=np.int16), 16000)

    audio = decode_audio_frame(frame)

    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0.0, 1.0], rel=1e-4)


def test_decode_audio_frame_downmixes_packed_stereo_interleaved_audio() -> None:
    frame = FakeFrame(np.array([[32767, -32768, -32768, 32767]], dtype=np.int16), 16000, channels=2)

    audio = decode_audio_frame(frame)

    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([-1.5258789e-05, -1.5258789e-05], rel=1e-4, abs=1e-6)


def test_decode_audio_frame_offsets_unsigned_integer_audio_before_scaling() -> None:
    frame = FakeFrame(np.array([0, 128, 255], dtype=np.uint8), 16000)

    audio = decode_audio_frame(frame)

    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([-1.0, 0.0, 0.9921875], rel=1e-4, abs=1e-6)


def test_decode_media_file_concatenates_decoded_frames(tmp_path: pathlib.Path) -> None:
    media_path = tmp_path / "sample.mp3"
    media_path.write_bytes(b"stub")
    frames = [
        FakeFrame(np.array([[0, 32767]], dtype=np.int16), 16000),
        FakeFrame(np.array([[32767, 0]], dtype=np.int16), 16000),
    ]

    class FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def decode(self, audio: int):
            return iter(frames)

    result = list(iter_media_chunks(media_path, open_container=lambda source: FakeContainer()))

    assert len(result) == 1
    audio, sample_rate = result[0].audio, result[0].sample_rate
    assert sample_rate == 16000
    assert audio.tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0], rel=1e-4)


def test_decode_media_file_raises_when_no_audio_stream(tmp_path: pathlib.Path) -> None:
    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(b"stub")

    class FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def decode(self, audio: int):
            return iter([])

    with pytest.raises(ValueError, match="No decodable audio stream found"):
        list(iter_media_chunks(media_path, open_container=lambda source: FakeContainer()))


def test_decode_audio_frame_clips_float_values() -> None:
    frame = FakeFrame(np.array([1.3, -1.6, 0.25], dtype=np.float32), 16000)

    audio = decode_audio_frame(frame)

    assert audio.tolist() == pytest.approx([1.0, -1.0, 0.25], rel=1e-5)


def test_decode_audio_frame_handles_empty_frame() -> None:
    frame = FakeFrame(np.array([], dtype=np.int16), 16000)
    audio = decode_audio_frame(frame)
    assert audio.size == 0


class ChunkContainer:
    """Track lazy frame decoding and deterministic resource closure."""

    def __init__(self, frames, start_time=None):
        self.frames = frames
        self.start_time = start_time
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def decode(self, audio):
        return iter(self.frames)


def test_chunk_iterator_is_lazy_and_closes_when_stopped():
    decoded = []

    def frames():
        for i in range(10000):
            decoded.append(i)
            yield FakeFrame(np.ones(16000, dtype=np.float32), 16000)

    container = ChunkContainer(frames())
    chunks = iter_media_chunks(pathlib.Path("large.mp4"), open_container=lambda path: container)
    first = next(chunks)
    assert first.audio.size == 30 * 16000
    assert len(decoded) == 30
    assert first.start_seconds == 0
    chunks.close()
    assert container.closed


def test_chunks_keep_exact_samples_and_final_partial_window():
    frames = [FakeFrame(np.arange(11, dtype=np.float32) / 10, 4)]
    chunks = list(iter_media_chunks(pathlib.Path("media.mp4"), chunk_seconds=1, open_container=lambda path: ChunkContainer(frames)))
    assert [chunk.audio.size for chunk in chunks] == [4, 4, 3]
    assert [chunk.start_seconds for chunk in chunks] == [0, 1, 2]
    np.testing.assert_allclose(np.concatenate([chunk.audio for chunk in chunks]), frames[0].to_ndarray())


def test_audio_pts_preserves_video_offset_and_gap():
    frames = [FakeFrame(np.ones(2, dtype=np.float32), 4) for _ in range(2)]
    frames[0].time = 7
    frames[1].time = 12
    container = ChunkContainer(frames, start_time=5_000_000)
    chunks = list(iter_media_chunks(pathlib.Path("media.mp4"), open_container=lambda path: container))
    assert [chunk.start_seconds for chunk in chunks] == [2, 7]
    assert [chunk.audio.size for chunk in chunks] == [2, 2]


def test_sample_rate_change_flushes_previous_window():
    frames = [FakeFrame(np.ones(4, dtype=np.float32), 4), FakeFrame(np.ones(8, dtype=np.float32), 8)]
    chunks = list(iter_media_chunks(pathlib.Path("media.mp4"), open_container=lambda path: ChunkContainer(frames)))
    assert [chunk.sample_rate for chunk in chunks] == [4, 8]
    assert [chunk.start_seconds for chunk in chunks] == [0, 1]


def test_invalid_sample_rate_closes_container():
    container = ChunkContainer([FakeFrame(np.ones(2, dtype=np.float32), 0)])
    with pytest.raises(ValueError, match="sample rate"):
        list(iter_media_chunks(pathlib.Path("media.mp4"), open_container=lambda path: container))
    assert container.closed
