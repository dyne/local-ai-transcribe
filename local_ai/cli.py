"""Command-line interface for local Whisper file transcription."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

from local_ai.infrastructure.openvino.runtime_env import configure_openvino_runtime_env
from local_ai.infrastructure.openvino.whisper import create_whisper_runtime, likely_reason_details
from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError
from local_ai.slices.voice.shared.audio_processing import (
    create_audio_preprocessor,
    VAD_MODE,
    VAD_MIN_SPEECH_FRAMES,
    VAD_MIN_SPEECH_RATIO,
    VAD_MIN_UTTERANCE_MS,
    VAD_HANGOVER_MS,
)
from local_ai.slices.voice.shared.transcript_policy import setup_error_exit_code
from local_ai.slices.voice.transcribe_file.request import TranscribeFileRequest
from local_ai.slices.voice.transcribe_file.service import execute_transcribe_file
from local_ai.slices.voice.transcribe_file.subtitles import SrtWriter
from pyspy_profile import start_py_spy_profile, stop_py_spy_profile


def log(message: str, verbose: bool, start_time: float | None = None) -> None:
    """Write optional progress to stderr."""
    if verbose:
        elapsed = "" if start_time is None else f" t+{time.perf_counter() - start_time:.2f}s"
        print(f"[transcribe{elapsed}] {message}", file=sys.stderr, flush=True)


def fail(reason: str, details: list[str] | None = None, exit_code: int = 1) -> int:
    """Report a concise diagnostic and return its exit status."""
    print(f"Error: {reason}", file=sys.stderr)
    for detail in details or ():
        print(f"- {detail}", file=sys.stderr)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Describe file inputs and the shared OpenVINO Whisper options."""
    parser = argparse.ArgumentParser(
        prog="local-ai-transcribe",
        description="Transcribe local audio/video files to .srt subtitles beside each input using OpenVINO Whisper.",
    )
    parser.add_argument("input_paths", type=Path, nargs="*", help="Local media files, processed in order.")
    parser.add_argument("--device", default="NPU,GPU,CPU", help="Device preference order, or 'list' (default: NPU,GPU,CPU).")
    parser.add_argument("--model", help="OpenVINO model directory or Hugging Face org/repo; otherwise download the device default.")
    parser.add_argument("--offline", action="store_true", help="Use local/cached models only; never download.")
    parser.add_argument("--language", help="Whisper language token, e.g. <|en|>.")
    parser.add_argument("--task", choices=["transcribe", "translate"], help="Whisper task.")
    parser.add_argument("--timestamps", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-seconds", type=float, default=30.0, help="Audio window in seconds, greater than 0 and at most 30 (default: 30).")
    silence = parser.add_mutually_exclusive_group()
    silence.add_argument("--silence-detect", dest="silence_detect", action="store_true", help="Enable noise reduction and speech detection (default).")
    silence.add_argument("--no-silence-detect", dest="silence_detect", action="store_false", help="Disable noise reduction and speech detection.")
    parser.set_defaults(silence_detect=True)
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=VAD_MODE)
    parser.add_argument("--vad-min-speech-frames", type=int, default=VAD_MIN_SPEECH_FRAMES)
    parser.add_argument("--vad-min-speech-ratio", type=float, default=VAD_MIN_SPEECH_RATIO)
    parser.add_argument("--vad-min-utterance-ms", type=int, default=VAD_MIN_UTTERANCE_MS)
    parser.add_argument("--vad-hangover-ms", type=int, default=VAD_HANGOVER_MS)
    parser.add_argument("--output-dir", type=Path, help="Override the default input directory for .srt files; never overwrite.")
    parser.add_argument("--profile", action="store_true", help="Record this run using optional py-spy.")
    parser.add_argument("--profile-output", type=Path, help="Profiling SVG path (default: profiles/<timestamp>.svg).")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr.")
    return parser


def run_files(args: argparse.Namespace) -> int:
    """Reuse one runtime across files, isolating preprocessing and failures."""
    start = time.perf_counter()
    configure_openvino_runtime_env()
    try:
        runtime = create_whisper_runtime(
            args=args,
            base_dir=Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1],
            logger=log,
            verbose=args.verbose,
            start_time=start,
        )
    except DeviceListRequested as exc:
        if not exc.devices:
            return fail("No OpenVINO devices detected.")
        for device in exc.devices:
            print(device, file=sys.stderr)
        return 0
    except PipelineSetupError as exc:
        return fail(exc.reason, exc.details, setup_error_exit_code(exc.reason))
    except ValueError as exc:
        return fail(str(exc), exit_code=2)

    print(f"Using device: {runtime.selected_device}", file=sys.stderr, flush=True)
    print(f"Using model: {runtime.model_dir}", file=sys.stderr, flush=True)
    if args.output_dir is not None:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return fail("Cannot create output directory.", [str(exc)], 2)

    status = 0
    for path in args.input_paths:
        if not path.is_file():
            fail(f"Input file not found or not a regular file: {path}", exit_code=2)
            status = status or 2
            continue
        output_path = (args.output_dir / path.name if args.output_dir else path).with_suffix(".srt")
        if output_path.resolve() == path.resolve():
            fail(f"Subtitle output would replace input: {path}", exit_code=2)
            status = status or 2
            continue
        try:
            preprocessor = create_audio_preprocessor(
                args.silence_detect,
                vad_mode=args.vad_mode,
                min_speech_frames=args.vad_min_speech_frames,
                min_speech_ratio=args.vad_min_speech_ratio,
                min_utterance_ms=args.vad_min_utterance_ms,
                hangover_ms=args.vad_hangover_ms,
            )
        except Exception as exc:
            return fail("Audio preprocessing failed.", [str(exc)], 6)
        created = False
        completed = False
        try:
            with output_path.open("x", encoding="utf-8", newline="\n") as output:
                created = True
                writer = SrtWriter(output)
                response = execute_transcribe_file(
                    request=TranscribeFileRequest(input_path=path, verbose=args.verbose, chunk_seconds=args.chunk_seconds),
                    pipe=runtime.pipe,
                    audio_preprocessor=preprocessor,
                    generate_kwargs=runtime.generate_kwargs,
                    start=start,
                    logger=log,
                    runtime_error_details=likely_reason_details,
                    on_cue=writer.write,
                )
                if response.exit_code:
                    fail(f"{path}: {response.reason}", response.details, response.exit_code)
                    status = status or response.exit_code
            completed = response.exit_code == 0
            if completed:
                print(f"Saved subtitles: {output_path} ({writer.count} cues)", file=sys.stderr)
        except OSError as exc:
            fail(f"Cannot write subtitles: {output_path}", [str(exc)], 2)
            status = status or 2
        finally:
            if created and not completed:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError as exc:
                    fail(f"Cannot remove incomplete subtitles: {output_path}", [str(exc)], 2)
                    status = status or 2
    log(f"Done in {time.perf_counter() - start:.2f}s", args.verbose, start)
    return status


def main(argv: list[str] | None = None) -> int:
    """Validate the invocation and run transcription with optional profiling."""
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)
    if not args.input_paths and args.device.strip().lower() != "list":
        parser.error("provide one or more input files, or use --device list")
    if not math.isfinite(args.chunk_seconds) or not 0 < args.chunk_seconds <= 30:
        parser.error("--chunk-seconds must be greater than 0 and at most 30")
    profile = start_py_spy_profile(enabled=args.profile, label="local-ai-transcribe", output_path=args.profile_output)
    try:
        return run_files(args)
    except KeyboardInterrupt:
        return 130
    finally:
        stop_py_spy_profile(profile)
