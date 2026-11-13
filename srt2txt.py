#!/usr/bin/env python3
"""Convert an SRT file to plain text using only the Python standard library."""

import argparse
from collections.abc import Iterable, Iterator
import html
import io
from itertools import chain
from pathlib import Path
import re
import sys


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


def main(argv: list[str] | None = None) -> int:
    """Print converted text, or create a new UTF-8 file with --output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input SRT file (UTF-8).")
    parser.add_argument("-o", "--output", type=Path, help="Write a new UTF-8 text file instead of stdout; never overwrite.")
    args = parser.parse_args(argv)
    created = False
    try:
        with args.input.open(encoding="utf-8-sig") as source:
            if args.output is None:
                sys.stdout.writelines(iter_srt_text(source))
            else:
                with args.output.open("x", encoding="utf-8", newline="\n") as output:
                    created = True
                    output.writelines(iter_srt_text(source))
    except (OSError, ValueError) as exc:
        if created:
            args.output.unlink(missing_ok=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
