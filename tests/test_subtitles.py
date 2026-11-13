"""Verify SRT timestamps, Whisper edge cases, and incremental serialization."""

import io
from types import SimpleNamespace

import pytest

from local_ai.slices.voice.transcribe_file.subtitles import SubtitleCue, SrtWriter, TxtWriter, format_srt_time, result_to_cues


def result(segments, text="speech"):
    """Build a Whisper-like timestamp result for conversion tests."""
    return SimpleNamespace(texts=[text], chunks=[SimpleNamespace(start_ts=a, end_ts=b, text=t) for a, b, t in segments])


def test_maps_timestamps_and_clamps_open_ended_segment():
    cues = result_to_cues(result([(0.5, 2, " hi "), (2, -1, "bye")]), offset=3600, duration=3)
    assert cues == [SubtitleCue(3600.5, 3602, "hi"), SubtitleCue(3602, 3603, "bye")]


def test_clamps_to_audio_bounds_and_drops_empty_intervals():
    cues = result_to_cues(result([(-1, 1, "start"), (1, 5, "end"), (4, 5, "past"), (0, 1, " ")]), offset=0, duration=2)
    assert cues == [SubtitleCue(0, 1, "start"), SubtitleCue(1, 2, "end")]


def test_negative_media_offset_clips_before_zero():
    assert result_to_cues(result([(0, 1, "hi")]), offset=-0.5, duration=2) == [SubtitleCue(0, 0.5, "hi")]


def test_text_without_timestamps_is_an_error():
    with pytest.raises(ValueError, match="without segment timestamps"):
        result_to_cues(SimpleNamespace(texts=["hello"], chunks=None), offset=0, duration=30)


def test_empty_result_has_no_cues():
    assert result_to_cues(SimpleNamespace(texts=[""], chunks=None), offset=0, duration=30) == []


def test_rejects_nonfinite_model_timestamps():
    with pytest.raises(ValueError, match="non-finite"):
        result_to_cues(result([(float("nan"), 1, "bad")]), offset=0, duration=30)


def test_srt_rounding_numbering_and_overlap():
    output = io.StringIO()
    writer = SrtWriter(output)
    writer.write(SubtitleCue(0.9996, 2.1, "hello"))
    writer.write(SubtitleCue(2, 3, "world"))
    writer.write(SubtitleCue(2, 2.5, "already ended"))
    assert writer.count == 2
    assert output.getvalue() == "1\n00:00:01,000 --> 00:00:02,100\nhello\n\n2\n00:00:02,100 --> 00:00:03,000\nworld\n\n"


def test_hours_do_not_wrap_at_twenty_four():
    assert format_srt_time(100 * 3600000 + 1234) == "100:00:01,234"


def test_txt_writer_emits_only_speech_and_preserves_repetitions():
    output = io.StringIO()
    writer = TxtWriter(output)
    writer.write(SubtitleCue(0, 1, "  hello\nworld "))
    writer.write(SubtitleCue(1, 2, "hello world"))
    writer.write(SubtitleCue(2, 3, " "))
    assert output.getvalue() == "hello world\nhello world\n"
    assert writer.count == 2
