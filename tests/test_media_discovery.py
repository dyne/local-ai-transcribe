"""Verify recursive input selection without decoding media or loading models."""

import pytest

from local_ai.slices.voice.transcribe_file.discovery import discover_media_files


def test_finds_nested_media_in_stable_order_and_ignores_other_files(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    for name in ("b.MP4", "a.mp3", "notes.txt", "b.srt", "cover.jpg", "settings.json"):
        (tmp_path / name).touch()
    for name in ("speech.FLAC", "recording.webm", "ignore.srt"):
        (nested / name).touch()
    assert discover_media_files([tmp_path]) == [
        tmp_path / "a.mp3", tmp_path / "b.MP4",
        nested / "recording.webm", nested / "speech.FLAC",
    ]


def test_overlapping_and_repeated_roots_do_not_duplicate_inputs(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    path = nested / "video.mp4"
    path.touch()
    assert discover_media_files([tmp_path, nested, tmp_path]) == [path]


def test_media_named_directory_is_not_selected(tmp_path):
    directory = tmp_path / "archive.mp4"
    directory.mkdir()
    assert discover_media_files([tmp_path]) == []


def test_rejects_file_and_missing_directory_roots(tmp_path):
    path = tmp_path / "video.mp4"
    path.touch()
    for root in (path, tmp_path / "missing"):
        with pytest.raises(ValueError, match="not a directory"):
            discover_media_files([root])


def test_directory_scan_errors_are_not_silently_ignored(tmp_path, monkeypatch):
    def denied(root, *, onerror, followlinks):
        onerror(PermissionError("Cannot read protected directory"))

    monkeypatch.setattr("local_ai.slices.voice.transcribe_file.discovery.os.walk", denied)
    with pytest.raises(PermissionError, match="protected directory"):
        discover_media_files([tmp_path])


def test_does_not_follow_directory_or_file_symlinks(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    target = outside / "video.mp4"
    target.touch()
    try:
        (root / "linked-directory").symlink_to(outside, target_is_directory=True)
        (root / "linked-file.mp4").symlink_to(target)
    except OSError:
        pytest.skip("Symbolic link creation is unavailable on this host")
    assert discover_media_files([root]) == []
