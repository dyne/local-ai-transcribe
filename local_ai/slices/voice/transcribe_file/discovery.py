"""Local directory traversal for batch media transcription."""

import os
from pathlib import Path


MEDIA_EXTENSIONS = frozenset({
    ".3g2", ".3gp", ".aac", ".ac3", ".aif", ".aiff", ".amr", ".ape",
    ".asf", ".au", ".avi", ".caf", ".flac", ".flv", ".m2ts", ".m4a",
    ".m4b", ".m4v", ".mka", ".mkv", ".mov", ".mp2", ".mp3", ".mp4",
    ".mpeg", ".mpg", ".mts", ".mxf", ".oga", ".ogg", ".ogv", ".opus",
    ".ts", ".vob", ".wav", ".wave", ".weba", ".webm", ".wma", ".wmv",
})


def discover_media_files(roots: list[Path]) -> list[Path]:
    """Find media in stable order, skipping symbolic links and repeated directories."""
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Recursive input is not a directory: {root}")
    files: list[Path] = []
    visited_directories: set[Path] = set()

    def report_error(error: OSError) -> None:
        """Fail visibly if a directory could not be scanned."""
        raise error

    for root in roots:
        for directory, subdirectories, filenames in os.walk(root, onerror=report_error, followlinks=False):
            resolved_directory = Path(directory).resolve()
            if resolved_directory in visited_directories:
                subdirectories.clear()
                continue
            visited_directories.add(resolved_directory)
            subdirectories.sort(key=str.casefold)
            for name in sorted(filenames, key=str.casefold):
                path = Path(directory) / name
                if path.suffix.lower() not in MEDIA_EXTENSIONS or path.is_symlink() or not path.is_file():
                    continue
                files.append(path)
    return files
