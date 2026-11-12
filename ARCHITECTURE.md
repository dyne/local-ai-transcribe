# Architecture

local-ai-transcribe is a file-only OpenVINO Whisper CLI that creates SubRip
subtitles beside each local media file. It avoids an application server, UI
framework, or service container.

- `local-ai-transcribe.py`: executable entry shell.
- `local_ai/cli.py`: validation, one-runtime batch execution, exclusive output
  creation, and cleanup of incomplete subtitles.
- `local_ai/shared/domain/`: device selection, model artifact validation,
  Hugging Face cache/download policy, and setup errors.
- `local_ai/infrastructure/openvino/`: runtime environment and Whisper creation.
- `local_ai/slices/voice/transcribe_file/`: per-file request/service/response,
  Whisper timestamp mapping, and incremental SRT serialization.
- `local_ai/slices/voice/shared/`: PyAV audio decoding, mono float32 normalization,
  noise reduction, VAD, 16 kHz resampling, and transcript suppression policy.
- `pyspy_profile.py`: optional profiling adapter.

The per-file use case accepts an `on_cue` output port. The CLI supplies an SRT
writer. The service never collects the full transcript or audio track. Existing
internal `voice` module names remain to avoid unnecessary renaming of the core.

A batch constructs one Whisper runtime and processes input paths in order. Each
file gets a new preprocessor so VAD/hangover state cannot leak between files.
PyAV decodes the first audio stream into bounded windows of at most 30 seconds.
It does not decode video. Frame presentation timestamps are normalized to the
container timeline, preserving A/V offsets and timestamp discontinuities.

Each window is denoised and speech-gated before final resampling. Silence does
not compress the media timeline. Whisper is always called with segment timestamps.
Local segment times are clipped to the window and shifted to the media position;
an open-ended last segment ends at the window boundary. SRT serialization rounds
to milliseconds, keeps cues ordered and non-overlapping, and writes immediately.

File failures remove incomplete output and allow the batch to continue. Existing
subtitles are never overwritten. Model setup failures stop the batch. Resume is
not implemented; failed or interrupted files must restart. Independent windows
can split utterances and lose cross-window context.

Preserved invariants:

- OpenVINO GenAI executes on NPU/GPU/CPU.
- Explicit model selection never silently falls back.
- Offline mode never downloads; cached snapshots precede downloads.
- Model artifacts are validated before runtime construction.
- Inference audio is mono float32 at 16000 Hz.
- Decoded audio and subtitle memory remain bounded by the processing window.
- Default output replaces the input extension with `.srt` beside the input.
- UTF-8 subtitle content goes to disk; diagnostics go to stderr.

No new dependencies are required. The frozen CLI bundles dynamic inference,
tokenizer, audio preprocessing, and Hugging Face Xet dependencies through
`local-ai-transcribe.spec`.
