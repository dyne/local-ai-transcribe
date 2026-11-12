"""Whisper segment mapping and incremental SubRip serialization."""

from dataclasses import dataclass
import math
from typing import TextIO

from local_ai.slices.voice.shared.transcript_policy import result_to_text


@dataclass(frozen=True)
class SubtitleCue:
    """A timed piece of speech in seconds on the original media timeline."""

    start_seconds: float
    end_seconds: float
    text: str


def result_to_cues(result: object, *, offset: float, duration: float) -> list[SubtitleCue]:
    """Clamp Whisper's local segment timestamps to the decoded audio window."""
    segments = getattr(result, "chunks", None)
    if not segments and result_to_text(result).strip():
        raise ValueError("Whisper returned text without segment timestamps; check model timestamp support.")
    cues = []
    for segment in segments or ():
        text = " ".join(segment.text.split())
        if not text:
            continue
        start, end = float(segment.start_ts), float(segment.end_ts)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Whisper returned non-finite segment timestamps.")
        start = min(duration, max(0.0, start))
        # Whisper can leave the last segment open with end_ts=-1.
        end = duration if end < 0 else min(duration, end)
        absolute_start = max(0.0, offset + start)
        absolute_end = max(0.0, offset + end)
        if absolute_end > absolute_start:
            cues.append(SubtitleCue(absolute_start, absolute_end, text))
    return cues


def format_srt_time(milliseconds: int) -> str:
    """Format integer milliseconds, including media lasting more than 24 hours."""
    seconds, millis = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


class SrtWriter:
    """Write sequential, non-overlapping cues without retaining subtitle history."""

    def __init__(self, output: TextIO) -> None:
        self.output = output
        self.count = 0
        self.last_end_ms = 0

    def write(self, cue: SubtitleCue) -> None:
        """Append one cue, dropping empty or sub-millisecond intervals."""
        start_ms = max(self.last_end_ms, round(cue.start_seconds * 1000))
        end_ms = round(cue.end_seconds * 1000)
        if end_ms <= start_ms or not cue.text.strip():
            return
        self.count += 1
        self.output.write(
            f"{self.count}\n{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}\n{cue.text}\n\n"
        )
        self.output.flush()
        self.last_end_ms = end_ms
