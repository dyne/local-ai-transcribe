from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TranscribeFileResponse:
    exit_code: int
    reason: str | None = None
    details: list[str] = field(default_factory=list)
