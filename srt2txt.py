#!/usr/bin/env python3
"""Convert an SRT file or folder to text using only the Python standard library."""

import argparse
from collections.abc import Iterable, Iterator
import html
import io
from itertools import chain
import os
from pathlib import Path
import re
import sys
import tempfile


TIMING = re.compile(r"^\d+:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d+:\d{2}:\d{2}[,.]\d{3}(?:\s+.*)?$")
STYLE_TAG = re.compile(r"</?(?:b|i|u|font)(?:\s[^>]*)?>", re.IGNORECASE)


def iter_srt_text(source: Iterable[str]) -> Iterator[str]:
    """Convert one cue at a time, keeping memory bounded for long subtitles."""
    cue: list[str] = []
    number = 0
    for line in chain(source, ["\n"]):
        if line.strip():
            cue.append(line.strip())
            continue
        if not cue:
            continue
        number += 1
        timing_index = 1 if cue[0].strip().isdecimal() else 0
        if timing_index >= len(cue) or not TIMING.fullmatch(cue[timing_index].strip()):
            raise ValueError(f"Invalid SRT timing in cue {number}.")
        text = " ".join(cue[timing_index + 1:])
        text = " ".join(html.unescape(STYLE_TAG.sub("", text)).split())
        cue.clear()
        if text:
            yield text + "\n"


def srt_to_text(source: str) -> str:
    """Convert an in-memory SRT string using the same streaming cue parser."""
    return "".join(iter_srt_text(io.StringIO(source.lstrip("\ufeff"), newline=None)))


def iter_srt_files(root: Path) -> Iterator[Path]:
    """Find SRT files recursively in stable order without following links."""
    visited: set[Path] = set()

    def report_error(error: OSError) -> None:
        """Report unreadable directories instead of silently omitting them."""
        raise error

    for directory, subdirectories, filenames in os.walk(root, onerror=report_error, followlinks=False):
        resolved = Path(directory).resolve()
        if resolved in visited:
            subdirectories.clear()
            continue
        visited.add(resolved)
        subdirectories.sort(key=str.casefold)
        for name in sorted(filenames, key=str.casefold):
            path = Path(directory) / name
            if path.suffix.lower() == ".srt" and path.is_file() and not path.is_symlink():
                yield path


def convert_to_file(source: Path, target: Path, *, skip_existing: bool = False) -> bool:
    """Stream to a new or empty TXT file; optionally skip non-empty outputs."""
    if source.resolve() == target.resolve():
        raise ValueError("Input and output paths must be different.")
    try:
        output = target.open("x", encoding="utf-8", newline="\n")
    except FileExistsError:
        if not target.is_file():
            raise
        if target.stat().st_size > 0:
            if skip_existing:
                return False
            raise
        output = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False,
            dir=target.parent, prefix=".srt2txt-", suffix=".txt.tmp",
        )
    working_path = Path(output.name)
    completed = False
    try:
        with output, source.open(encoding="utf-8-sig") as subtitles:
            output.writelines(iter_srt_text(subtitles))
        if working_path != target:
            working_path.replace(target)
        completed = True
        return True
    finally:
        if not completed:
            working_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Convert one SRT, or recursively create adjacent TXT files for a folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 SRT file, or a folder to convert recursively.")
    parser.add_argument("-o", "--output", type=Path, help="Single-file mode: write TXT instead of stdout; replace empty files only.")
    args = parser.parse_args(argv)
    if args.input.is_dir() and args.output is not None:
        parser.error("--output is only supported for a single input file")
    try:
        if args.input.is_dir():
            status = 0
            for path in iter_srt_files(args.input):
                target = path.with_suffix(".txt")
                try:
                    converted = convert_to_file(path, target, skip_existing=True)
                    print(f"{'Saved text' if converted else 'Skipped'}: {target}", file=sys.stderr)
                except (OSError, ValueError) as exc:
                    print(f"Error: {path}: {exc}", file=sys.stderr)
                    status = 2
            return status
        if args.output is None:
            with args.input.open(encoding="utf-8-sig") as source:
                sys.stdout.writelines(iter_srt_text(source))
        else:
            convert_to_file(args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
