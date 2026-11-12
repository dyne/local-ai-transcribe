from __future__ import annotations

import pathlib
import wave

import numpy as np
import pytest

from local_ai.slices.voice.shared import audio_processing


def create_test_wav(path: pathlib.Path, *, sample_rate: int = 8000, channels: int = 2) -> None:
    samples = np.array([0, 32767, -32768, 1000], dtype=np.int16)
    if channels == 2:
        samples = np.column_stack((samples, samples))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def test_read_wav_mono_float32_downmixes_to_mono(tmp_path: pathlib.Path) -> None:
    wav_path = tmp_path / "sample.wav"
    create_test_wav(wav_path)

    audio, sample_rate = audio_processing.read_wav_mono_float32(wav_path)

    assert sample_rate == 8000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] == 4


def test_normalize_audio_format_returns_flat_float32() -> None:
    normalized = audio_processing.normalize_audio_format(np.array([[1, 2], [3, 4]], dtype=np.int16))
    assert normalized.dtype == np.float32
    assert normalized.shape == (4,)


def test_resample_audio_linear_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError, match="Invalid sample rate"):
        audio_processing.resample_audio_linear(np.array([0.0], dtype=np.float32), 0, 16000)


def test_ensure_sample_rate_logs_and_resamples() -> None:
    messages: list[str] = []
    audio = np.array([0.0, 1.0], dtype=np.float32)

    resampled, rate = audio_processing.ensure_sample_rate(
        audio,
        8000,
        True,
        1.0,
        lambda message, verbose, start: messages.append(message),
    )

    assert rate == 16000
    assert resampled.dtype == np.float32
    assert messages == ["Resampling audio from 8000 Hz to 16000 Hz"]


def test_create_audio_preprocessor_disabled_returns_none() -> None:
    assert audio_processing.create_audio_preprocessor(False) is None


def test_create_audio_preprocessor_rejects_invalid_params() -> None:
    with pytest.raises(RuntimeError, match="min_speech_frames must be > 0"):
        audio_processing.create_audio_preprocessor(True, min_speech_frames=0)


def test_prepare_vad_audio_uses_closest_supported_rate() -> None:
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    prepared, rate = audio_processing.prepare_vad_audio(audio, 11025)
    assert rate in audio_processing.SUPPORTED_VAD_SAMPLE_RATES
    assert prepared.dtype == np.float32


def test_preprocess_audio_without_preprocessor_normalizes_input() -> None:
    audio = audio_processing.preprocess_audio(
        np.array([[1, 2], [3, 4]], dtype=np.int16),
        16000,
        None,
        False,
        0.0,
        lambda message, verbose, start: None,
    )
    assert audio.dtype == np.float32
    assert audio.shape == (4,)


def test_sanitize_audio_replaces_non_finite_values_and_logs() -> None:
    messages: list[str] = []

    audio = audio_processing.sanitize_audio(
        np.array([0.0, np.nan, np.inf, -np.inf], dtype=np.float32),
        context="noise reduction",
        verbose=True,
        start=0.0,
        logger=lambda message, verbose, start: messages.append(message),
    )

    assert audio.tolist() == [0.0, 0.0, 1.0, -1.0]
    assert messages == ["Audio sanitation: replaced 3 non-finite samples after noise reduction"]


def make_test_preprocessor(reduce_noise):
    """Build preprocessing state without importing an audio device or model."""
    from types import SimpleNamespace

    return audio_processing.AudioPreprocessor(
        nr=SimpleNamespace(reduce_noise=reduce_noise),
        vad=object(),
        vad_mode=1,
        min_speech_frames=3,
        min_speech_ratio=0.02,
        min_utterance_ms=180,
        hangover_ms=300,
    )


@pytest.mark.parametrize("samples", [0, 16000, 30 * 48000])
def test_digital_silence_skips_denoiser_and_clears_stale_speech_state(samples):
    def unexpected_denoise(**kwargs):
        raise AssertionError("Digital silence must not enter noise reduction")

    preprocessor = make_test_preprocessor(unexpected_denoise)
    preprocessor.hangover_frames = 10
    preprocessor.last_speech_frames = 5
    preprocessor.last_total_frames = 10
    preprocessor.last_max_run = 5
    preprocessor.last_was_hangover = True
    messages = []
    output = audio_processing.preprocess_audio(
        np.zeros(samples, dtype=np.float32), 48000, preprocessor,
        True, 0, lambda message, *args: messages.append(message),
    )
    assert output.size == 0
    assert output.dtype == np.float32
    assert preprocessor.hangover_frames == 0
    assert preprocessor.last_speech_frames == 0
    assert preprocessor.last_total_frames == 0
    assert preprocessor.last_max_run == 0
    assert preprocessor.last_was_hangover is False
    assert messages == ["Audio skipped: digital silence"]


@pytest.mark.parametrize("operation", ["invalid", "divide"])
def test_invalid_noise_reduction_falls_back_without_warning(monkeypatch, operation):
    import warnings

    original = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    old_error_policy = np.geterr().copy()
    messages = []
    gated = []

    def unstable_denoise(**kwargs):
        numerator = 0.0 if operation == "invalid" else 1.0
        return np.divide(np.array([numerator]), np.zeros(1))

    def speech_detected(original_audio, reduced_audio, *args):
        gated.append(reduced_audio.copy())
        return True

    monkeypatch.setattr(audio_processing, "_speech_detected", speech_detected)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        output = audio_processing.preprocess_audio(
            original, 16000, make_test_preprocessor(unstable_denoise),
            True, 0, lambda message, *args: messages.append(message),
        )
    np.testing.assert_array_equal(output, original)
    np.testing.assert_array_equal(gated[0], original)
    assert np.isfinite(output).all()
    assert np.geterr() == old_error_policy
    assert any("Noise reduction bypassed:" in message for message in messages)


def test_nonfinite_denoiser_output_preserves_original_speech(monkeypatch):
    original = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    preprocessor = make_test_preprocessor(lambda **kwargs: np.array([np.nan, np.inf, -np.inf], dtype=np.float32))
    monkeypatch.setattr(audio_processing, "_speech_detected", lambda *args: True)
    output = audio_processing.preprocess_audio(original, 16000, preprocessor, False, 0, lambda *args: None)
    np.testing.assert_array_equal(output, original)


def test_near_silent_audio_is_not_discarded_by_amplitude_threshold(monkeypatch):
    original = np.full(160, 1e-30, dtype=np.float32)
    calls = []

    def denoise(**kwargs):
        calls.append(kwargs["y"].copy())
        return kwargs["y"]

    monkeypatch.setattr(audio_processing, "_speech_detected", lambda *args: True)
    output = audio_processing.preprocess_audio(original, 16000, make_test_preprocessor(denoise), False, 0, lambda *args: None)
    np.testing.assert_array_equal(calls[0], original)
    np.testing.assert_array_equal(output, original)


def test_other_noise_reduction_errors_remain_diagnostic():
    def broken_denoise(**kwargs):
        raise RuntimeError("unrelated failure")

    with pytest.raises(RuntimeError, match="Noise reduction failed"):
        audio_processing.preprocess_audio(
            np.ones(160, dtype=np.float32), 16000, make_test_preprocessor(broken_denoise),
            False, 0, lambda *args: None,
        )
