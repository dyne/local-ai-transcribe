"""Verify standalone SRT conversion without importing transcription dependencies."""

import pytest

from srt2txt import iter_srt_text, main, srt_to_text


SAMPLE = "1\n00:00:01,000 --> 00:00:02,500\nHello\nworld.\n\n2\n00:00:03,000 --> 00:00:04,000\n2026\n\n"


def test_conversion_removes_headers_but_preserves_numeric_caption_text():
    assert srt_to_text(SAMPLE) == "Hello world.\n2026\n"


def test_handles_bom_crlf_styles_and_entities():
    source = '\ufeff1\r\n00:00:00,000 --> 00:00:01,000\r\n<i>caffè</i> &amp; <font color="red">tea</font>\r\n'
    assert srt_to_text(source) == "caffè & tea\n"


def test_allows_missing_indices_dot_timestamps_and_long_hours():
    assert srt_to_text("100:00:00.000 --> 100:00:01.000\nHello") == "Hello\n"


def test_empty_subtitles_produce_empty_text():
    assert srt_to_text("\n\n") == ""


def test_invalid_timing_is_diagnostic():
    with pytest.raises(ValueError, match="Invalid SRT timing in cue 1"):
        srt_to_text("1\nnot timestamps\nhello")


def test_cli_prints_to_stdout_by_default(tmp_path, capsys):
    source = tmp_path / "video.srt"
    source.write_text(SAMPLE, encoding="utf-8")
    assert main([str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Hello world.\n2026\n"
    assert captured.err == ""


def test_cli_creates_output_without_overwriting(tmp_path, capsys):
    source = tmp_path / "video.srt"
    target = tmp_path / "video.txt"
    source.write_text(SAMPLE, encoding="utf-8")
    assert main([str(source), "-o", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == "Hello world.\n2026\n"
    target.write_text("keep", encoding="utf-8")
    assert main([str(source), "-o", str(target)]) == 2
    assert target.read_text(encoding="utf-8") == "keep"
    assert "Error:" in capsys.readouterr().err


def test_invalid_input_does_not_create_output(tmp_path):
    source = tmp_path / "invalid.srt"
    target = tmp_path / "out.txt"
    source.write_text("not SRT", encoding="utf-8")
    assert main([str(source), "-o", str(target)]) == 2
    assert not target.exists()


def test_conversion_does_not_read_ahead_into_later_cues():
    def source():
        yield from ["1\n", "00:00:00,000 --> 00:00:01,000\n", "hello\n", "\n"]
        raise AssertionError("The next cue must not be read before the first is emitted")

    converted = iter_srt_text(source())
    assert next(converted) == "hello\n"
    converted.close()


def test_later_invalid_cue_removes_incomplete_output(tmp_path):
    source = tmp_path / "invalid.srt"
    target = tmp_path / "out.txt"
    source.write_text(SAMPLE + "3\ninvalid\nbad cue", encoding="utf-8")
    assert main([str(source), "-o", str(target)]) == 2
    assert not target.exists()


def test_single_file_output_replaces_empty_text_file(tmp_path):
    source = tmp_path / "video.srt"
    target = tmp_path / "video.txt"
    source.write_text(SAMPLE, encoding="utf-8")
    target.touch()
    assert main([str(source), "-o", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == "Hello world.\n2026\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_folder_conversion_handles_nested_srt_and_skips_nonempty_txt(tmp_path, capsys, monkeypatch):
    nested = tmp_path / "nested"
    nested.mkdir()
    completed = tmp_path / "done.srt"
    empty = nested / "empty.SRT"
    missing = nested / "missing.srt"
    for path in (completed, empty, missing):
        path.write_text(SAMPLE, encoding="utf-8")
    completed.with_suffix(".txt").write_text("keep text", encoding="utf-8")
    empty.with_suffix(".txt").touch()
    (nested / "notes.txt").write_text("not subtitles", encoding="utf-8")

    def unexpected_input():
        raise AssertionError("Recursive conversion must not ask questions")

    monkeypatch.setattr("builtins.input", unexpected_input)
    assert main([str(tmp_path)]) == 0
    assert completed.with_suffix(".txt").read_text(encoding="utf-8") == "keep text"
    assert empty.with_suffix(".txt").read_text(encoding="utf-8") == "Hello world.\n2026\n"
    assert missing.with_suffix(".txt").read_text(encoding="utf-8") == "Hello world.\n2026\n"
    assert (nested / "notes.txt").read_text(encoding="utf-8") == "not subtitles"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Skipped:" in captured.err
    assert captured.err.count("Saved text:") == 2


def test_recursive_conversion_continues_after_invalid_srt(tmp_path):
    (tmp_path / "bad.srt").write_text("not SRT", encoding="utf-8")
    (tmp_path / "good.srt").write_text(SAMPLE, encoding="utf-8")
    assert main([str(tmp_path)]) == 2
    assert not (tmp_path / "bad.txt").exists()
    assert (tmp_path / "good.txt").read_text(encoding="utf-8") == "Hello world.\n2026\n"


def test_failed_conversion_preserves_empty_placeholder(tmp_path):
    source = tmp_path / "bad.srt"
    target = tmp_path / "bad.txt"
    source.write_text("not SRT", encoding="utf-8")
    target.touch()
    assert main([str(tmp_path)]) == 2
    assert target.exists() and target.stat().st_size == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_folder_rejects_single_output_override(tmp_path):
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), "-o", str(tmp_path / "combined.txt")])
    assert error.value.code == 2
    assert not (tmp_path / "combined.txt").exists()


def test_converter_cannot_overwrite_its_input(tmp_path):
    source = tmp_path / "video.srt"
    source.write_text(SAMPLE, encoding="utf-8")
    assert main([str(source), "-o", str(source)]) == 2
    assert source.read_text(encoding="utf-8") == SAMPLE
