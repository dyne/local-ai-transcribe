"""Check streaming transcription, errors, and media-relative subtitle timing."""

from types import SimpleNamespace

import numpy as np
import pytest

from local_ai.slices.voice.shared.media_decode import MediaChunk
from local_ai.slices.voice.transcribe_file import service
from local_ai.slices.voice.transcribe_file.request import TranscribeFileRequest


def result(text="hello", start=0.25, end=1.5):
    """Build a timestamped Whisper result matching OpenVINO's public contract."""
    return SimpleNamespace(texts=[text], chunks=[SimpleNamespace(text=text, start_ts=start, end_ts=end)])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"media")
    windows = [MediaChunk(np.ones(32000, dtype=np.float32), 16000, 0)]
    calls = []
    cues = []

    def generate(audio, **kwargs):
        calls.append((audio, kwargs))
        return result()

    monkeypatch.setattr(service, "iter_media_chunks", lambda *args, **kwargs: (chunk for chunk in windows))
    kwargs = dict(
        request=TranscribeFileRequest(input_path=path),
        pipe=SimpleNamespace(generate=generate),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda *args: None,
        runtime_error_details=lambda exc: [f"Runtime error: {exc}"],
        on_cue=cues.append,
    )
    return kwargs, windows, calls, cues


def test_streams_timed_cues_and_forces_timestamps(setup):
    kwargs, windows, calls, cues = setup
    kwargs["generate_kwargs"] = {"return_timestamps": False, "task": "translate"}
    response = service.execute_transcribe_file(**kwargs)
    assert response.exit_code == 0
    assert cues[0].text == "hello"
    assert cues[0].start_seconds == 0.25
    assert cues[0].end_seconds == 1.5
    assert calls[0][1] == {"return_timestamps": True, "task": "translate"}


def test_chunk_offsets_survive_skipped_silence(monkeypatch, setup):
    kwargs, windows, calls, cues = setup
    windows[:] = [
        MediaChunk(np.zeros(32000, dtype=np.float32), 16000, 0),
        MediaChunk(np.ones(32000, dtype=np.float32), 16000, 3600),
    ]
    monkeypatch.setattr(service, "preprocess_audio", lambda audio, *args: audio if audio.any() else np.array([], dtype=np.float32))
    assert service.execute_transcribe_file(**kwargs).exit_code == 0
    assert len(calls) == 1
    assert cues[0].start_seconds == 3600.25


def test_audio_is_resampled_before_inference(setup):
    kwargs, windows, calls, cues = setup
    windows[:] = [MediaChunk(np.ones(48000, dtype=np.float32), 24000, 0)]
    assert service.execute_transcribe_file(**kwargs).exit_code == 0
    assert len(calls[0][0]) == 32000


def test_decode_failure_after_first_chunk_reports_error_and_closes(monkeypatch, setup):
    kwargs, windows, calls, cues = setup
    closed = []

    def decode(*args, **kwargs):
        try:
            yield windows[0]
            raise ValueError("corrupt packet")
        finally:
            closed.append(True)

    monkeypatch.setattr(service, "iter_media_chunks", decode)
    response = service.execute_transcribe_file(**kwargs)
    assert response.exit_code == 2
    assert response.reason == "Audio decode failed."
    assert response.details == ["corrupt packet"]
    assert closed == [True]


@pytest.mark.parametrize("stage,code,reason", [
    ("preprocess_audio", 6, "Audio preprocessing failed."),
    ("ensure_sample_rate", 6, "Audio resampling failed."),
])
def test_processing_failures_are_diagnostic(monkeypatch, setup, stage, code, reason):
    kwargs, windows, calls, cues = setup

    def broken(*args):
        raise RuntimeError("failure")

    monkeypatch.setattr(service, stage, broken)
    response = service.execute_transcribe_file(**kwargs)
    assert response.exit_code == code
    assert response.reason == reason


def test_inference_failure_closes_decoder(monkeypatch, setup):
    kwargs, windows, calls, cues = setup
    closed = []

    def decode(*args, **kwargs):
        try:
            yield windows[0]
            raise AssertionError("Do not decode ahead after inference failure")
        finally:
            closed.append(True)

    def broken(*args, **kwargs):
        raise RuntimeError("bad inference")

    monkeypatch.setattr(service, "iter_media_chunks", decode)
    kwargs["pipe"] = SimpleNamespace(generate=broken)
    response = service.execute_transcribe_file(**kwargs)
    assert response.exit_code == 5
    assert response.details == ["Runtime error: bad inference"]
    assert closed == [True]


def test_subtitle_disk_error_is_reported(setup):
    kwargs, windows, calls, cues = setup

    def disk_full(cue):
        raise OSError("disk full")

    kwargs["on_cue"] = disk_full
    response = service.execute_transcribe_file(**kwargs)
    assert response.exit_code == 2
    assert response.reason == "Subtitle write failed."


def test_silent_file_returns_no_speech(monkeypatch, setup):
    kwargs, windows, calls, cues = setup
    monkeypatch.setattr(service, "preprocess_audio", lambda *args: np.array([], dtype=np.float32))
    assert service.execute_transcribe_file(**kwargs).reason == "No speech detected."
    assert not calls


def test_missing_file_is_diagnostic(setup):
    kwargs, windows, calls, cues = setup
    kwargs["request"].input_path.unlink()
    assert service.execute_transcribe_file(**kwargs).exit_code == 2
    assert not calls
