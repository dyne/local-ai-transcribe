from __future__ import annotations

from collections.abc import Callable
from contextlib import closing

from local_ai.slices.voice.shared.audio_processing import ensure_sample_rate, preprocess_audio
from local_ai.slices.voice.shared.media_decode import iter_media_chunks
from local_ai.slices.voice.shared.transcript_policy import result_to_text, should_suppress_transcript
from local_ai.slices.voice.transcribe_file.request import TranscribeFileRequest
from local_ai.slices.voice.transcribe_file.response import TranscribeFileResponse
from local_ai.slices.voice.transcribe_file.subtitles import SubtitleCue, result_to_cues


def execute_transcribe_file(
    *,
    request: TranscribeFileRequest,
    pipe: object,
    audio_preprocessor: object | None,
    generate_kwargs: dict[str, object],
    start: float,
    logger: Callable[[str, bool, float | None], None],
    runtime_error_details: Callable[[Exception], list[str]],
    on_cue: Callable[[SubtitleCue], None],
) -> TranscribeFileResponse:
    """Stream audio windows through Whisper and emit timestamped subtitle cues."""
    if not request.input_path.is_file():
        return TranscribeFileResponse(exit_code=2, reason=f"Input file not found: {request.input_path}")
    emitted = False
    reason, code = "Audio decode failed.", 2
    try:
        with closing(iter_media_chunks(request.input_path, chunk_seconds=request.chunk_seconds)) as chunks:
            for chunk in chunks:
                logger(f"Processing {request.input_path} at {chunk.start_seconds:.2f}s", request.verbose, start)
                reason, code = "Audio preprocessing failed.", 6
                audio = preprocess_audio(chunk.audio, chunk.sample_rate, audio_preprocessor, request.verbose, start, logger)
                if audio.size:
                    reason, code = "Audio resampling failed.", 6
                    audio, _ = ensure_sample_rate(audio, chunk.sample_rate, request.verbose, start, logger)
                    reason, code = "Transcription failed.", 5
                    result = pipe.generate(audio.tolist(), **{**generate_kwargs, "return_timestamps": True})
                    if not should_suppress_transcript(result_to_text(result).strip(), audio_preprocessor):
                        cues = result_to_cues(
                            result,
                            offset=chunk.start_seconds,
                            duration=chunk.audio.size / chunk.sample_rate,
                        )
                        reason, code = "Subtitle write failed.", 2
                        for cue in cues:
                            on_cue(cue)
                            emitted = True
                reason, code = "Audio decode failed.", 2
    except Exception as exc:
        details = runtime_error_details(exc) if code == 5 else [str(exc)]
        return TranscribeFileResponse(exit_code=code, reason=reason, details=details)
    if not emitted:
        return TranscribeFileResponse(exit_code=2, reason="No speech detected.")
    return TranscribeFileResponse(exit_code=0)
