"""Exercise batch subtitle output without downloads or inference hardware."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from local_ai import cli
from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError
from local_ai.slices.voice.transcribe_file.response import TranscribeFileResponse
from local_ai.slices.voice.transcribe_file.subtitles import SubtitleCue


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    value = SimpleNamespace(selected_device="CPU", model_dir=Path("model"), pipe=object(), generate_kwargs={})
    monkeypatch.setattr(cli, "configure_openvino_runtime_env", lambda: None)
    monkeypatch.setattr(cli, "create_whisper_runtime", lambda **kwargs: calls.append(kwargs) or value)
    monkeypatch.setattr(cli, "create_audio_preprocessor", lambda *args, **kwargs: object())
    return calls, value


def media(tmp_path, name):
    """Create a media placeholder for orchestration tests."""
    path = tmp_path / name
    path.write_bytes(b"media")
    return path


def transcribe(**kwargs):
    """Emit a real timed cue through the supplied callback."""
    kwargs["on_cue"](SubtitleCue(0.25, 1.5, "caffè"))
    return TranscribeFileResponse(exit_code=0)


def test_default_writes_matching_srt_beside_input(tmp_path, monkeypatch, runtime, capsys):
    path = media(tmp_path, "speech.mp4")
    monkeypatch.setattr(cli, "execute_transcribe_file", transcribe)
    assert cli.main([str(path), "--offline"]) == 0
    assert path.with_suffix(".srt").read_text(encoding="utf-8") == "1\n00:00:00,250 --> 00:00:01,500\ncaffè\n\n"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Using device: CPU" in captured.err
    assert "Using model: model" in captured.err
    assert "Saved subtitles:" in captured.err
    assert runtime[0][0]["args"].offline is True


def test_batch_reuses_runtime_and_isolates_preprocessors(tmp_path, monkeypatch, runtime):
    paths = [media(tmp_path, name) for name in ("one.wav", "two.mp3")]
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return transcribe(**kwargs)

    monkeypatch.setattr(cli, "execute_transcribe_file", capture)
    assert cli.main([str(path) for path in paths]) == 0
    assert len(runtime[0]) == 1
    assert [call["pipe"] for call in calls] == [runtime[1].pipe] * 2
    assert calls[0]["audio_preprocessor"] is not calls[1]["audio_preprocessor"]
    assert all(path.with_suffix(".srt").exists() for path in paths)


def test_options_between_files_and_chunk_size_forwarded(tmp_path, monkeypatch, runtime):
    paths = [media(tmp_path, name) for name in ("one.wav", "two.wav")]
    calls = []

    def capture(**kwargs):
        calls.append(kwargs["request"])
        return transcribe(**kwargs)

    monkeypatch.setattr(cli, "execute_transcribe_file", capture)
    assert cli.main([str(paths[0]), "--offline", str(paths[1]), "--chunk-seconds", "15"]) == 0
    assert [call.chunk_seconds for call in calls] == [15, 15]


def test_batch_continues_after_errors_and_removes_failed_output(tmp_path, monkeypatch, runtime, capsys):
    broken, good = [media(tmp_path, name) for name in ("broken.wav", "good.wav")]

    def capture(**kwargs):
        if kwargs["request"].input_path == broken:
            kwargs["on_cue"](SubtitleCue(0, 1, "incomplete"))
            return TranscribeFileResponse(exit_code=5, reason="Inference failed.")
        return transcribe(**kwargs)

    monkeypatch.setattr(cli, "execute_transcribe_file", capture)
    missing = [tmp_path / "missing.wav", tmp_path / "also-missing.wav"]
    assert cli.main([str(p) for p in [broken, *missing, good]]) == 5
    assert not broken.with_suffix(".srt").exists()
    assert good.with_suffix(".srt").exists()
    captured = capsys.readouterr()
    assert "Inference failed" in captured.err
    assert all(str(path) in captured.err for path in missing)


def test_output_directory_overrides_location(tmp_path, monkeypatch, runtime):
    path = media(tmp_path, "speech.mp4")
    output_dir = tmp_path / "subtitles"
    monkeypatch.setattr(cli, "execute_transcribe_file", transcribe)
    assert cli.main([str(path), "--output-dir", str(output_dir)]) == 0
    assert (output_dir / "speech.srt").exists()
    assert not path.with_suffix(".srt").exists()


def test_existing_srt_is_preserved_without_inference_and_batch_continues(tmp_path, monkeypatch, runtime, capsys):
    paths = [media(tmp_path, name) for name in ("one.wav", "two.wav")]
    existing = paths[0].with_suffix(".srt")
    existing.write_text("keep", encoding="utf-8")
    calls = []

    def capture(**kwargs):
        calls.append(kwargs["request"].input_path)
        return transcribe(**kwargs)

    monkeypatch.setattr(cli, "execute_transcribe_file", capture)
    assert cli.main([str(path) for path in paths]) == 2
    assert existing.read_text(encoding="utf-8") == "keep"
    assert calls == [paths[1]]
    assert "Cannot write subtitles" in capsys.readouterr().err


def test_same_stem_cannot_overwrite_first_subtitle(tmp_path, monkeypatch, runtime):
    paths = [media(tmp_path, name) for name in ("speech.mp4", "speech.wav")]
    monkeypatch.setattr(cli, "execute_transcribe_file", transcribe)
    assert cli.main([str(path) for path in paths]) == 2
    assert paths[0].with_suffix(".srt").read_text(encoding="utf-8").count("caffè") == 1


def test_same_filename_in_different_folders_is_allowed(tmp_path, monkeypatch, runtime):
    folders = [tmp_path / "a", tmp_path / "b"]
    for folder in folders:
        folder.mkdir()
    paths = [media(folder, "speech.mp4") for folder in folders]
    monkeypatch.setattr(cli, "execute_transcribe_file", transcribe)
    assert cli.main([str(path) for path in paths]) == 0
    assert all(path.with_suffix(".srt").exists() for path in paths)


def test_missing_inputs_do_not_initialize_runtime(runtime):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    assert not runtime[0]


@pytest.mark.parametrize("duration", ["0", "-1", "31", "nan", "inf"])
def test_invalid_chunk_size_rejected_before_runtime(runtime, duration):
    with pytest.raises(SystemExit) as exc:
        cli.main(["video.mp4", "--chunk-seconds", duration])
    assert exc.value.code == 2
    assert not runtime[0]


def test_device_listing_does_not_require_files(monkeypatch, runtime, capsys):
    def list_devices(**kwargs):
        raise DeviceListRequested(["CPU", "GPU"])

    monkeypatch.setattr(cli, "create_whisper_runtime", list_devices)
    assert cli.main(["--device", "list"]) == 0
    assert capsys.readouterr().err.splitlines() == ["CPU", "GPU"]


def test_model_setup_error_stops_batch(tmp_path, monkeypatch, runtime, capsys):
    path = media(tmp_path, "speech.wav")

    def setup_error(**kwargs):
        raise PipelineSetupError("Model download required but offline mode is enabled.", ["use a local model"])

    monkeypatch.setattr(cli, "create_whisper_runtime", setup_error)
    assert cli.main([str(path), "--offline"]) == 3
    assert "offline" in capsys.readouterr().err


def test_preprocessing_error_is_diagnostic(tmp_path, monkeypatch, runtime, capsys):
    path = media(tmp_path, "speech.wav")

    def broken_preprocessor(*args, **kwargs):
        raise RuntimeError("VAD unavailable")

    monkeypatch.setattr(cli, "create_audio_preprocessor", broken_preprocessor)
    assert cli.main([str(path)]) == 6
    assert "VAD unavailable" in capsys.readouterr().err


def test_removed_interfaces_are_rejected():
    for option in ("--server", "--web", "--cli"):
        with pytest.raises(SystemExit) as exc:
            cli.main([option])
        assert exc.value.code == 2


def test_keyboard_interrupt_removes_partial_srt_and_stops_profile(tmp_path, monkeypatch, runtime):
    stopped = []
    path = media(tmp_path, "speech.wav")
    monkeypatch.setattr(cli, "start_py_spy_profile", lambda **kwargs: "profile")
    monkeypatch.setattr(cli, "stop_py_spy_profile", stopped.append)

    def interrupt(**kwargs):
        kwargs["on_cue"](SubtitleCue(0, 1, "partial"))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "execute_transcribe_file", interrupt)
    assert cli.main([str(path)]) == 130
    assert stopped == ["profile"]
    assert not path.with_suffix(".srt").exists()
