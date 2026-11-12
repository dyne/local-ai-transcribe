# AGENTS.md

## Scope and approach

This repository builds **local-ai-transcribe**, a Python 3.11 CLI for one or more
local audio/video files. Prefer the simplest viable approach, native docstrings,
and existing algorithms. Do not add dependencies unless asked.

Use pragmatic VSA + REPR + ports/adapters: keep file transcription under its
slice, domain rules in shared/domain, and OpenVINO/PyAV IO in adapters. Avoid
adding infrastructure or abstractions without a concrete need.

## Entrypoints and boundaries

- `local-ai-transcribe.py`: thin public entrypoint.
- `local_ai/cli.py`: CLI parsing, batch execution, and transcript output.
- `local_ai/shared/domain/devices.py`: NPU/GPU/CPU preference and enumeration.
- `local_ai/shared/domain/models.py`: model validation, cache, and download.
- `local_ai/infrastructure/openvino/`: runtime setup and Whisper construction.
- `local_ai/slices/voice/transcribe_file/`: per-file request/service/response.
- `local_ai/slices/voice/shared/`: decoding, preprocessing, transcript policy.

Desktop/browser UI, microphone streaming, Documents, and OCR are intentionally
out of scope. Do not restore these dependencies for file transcription.

## Invariants

- Use Python 3.11 for install, test, profiling, and packaging.
- Keep inference on OpenVINO GenAI and audio mono float32 at 16000 Hz.
- Always use Whisper's `transcribe` task; do not expose translation options.
- `-R` / `--recursive` takes directories and discovers eligible media recursively,
  with case-insensitive extensions and no duplicate paths from overlapping roots.
- Decode and preprocess before final resampling.
- Explicit `--model` must never silently fall back to a different model.
- `--offline` must never download. Reuse the shared Hugging Face cache first.
- Support `HF_TOKEN`, local OpenVINO exports, and Hugging Face org/repo model IDs.
- Default devices: NPU,GPU,CPU. Default NPU model:
  OpenVINO/whisper-large-v3-turbo-int8-ov; GPU/CPU: OpenVINO/whisper-tiny-fp16-ov.
- Print selected device and resolved model path to stderr.
- Load one runtime per batch, reset preprocessing per file, continue after file
  errors, and return nonzero if any file fails.
- Write UTF-8 SubRip subtitles beside each input, replacing its extension with
  `.srt`. `--output-dir` is an optional location override. Ask `Overwrite? (Y/n)`
  before replacing existing subtitles; Enter means yes, No skips without error.
  `-y` / `--yes` bypasses prompts. Preserve the original until replacement succeeds.
- Decode only the first audio stream in bounded windows of at most 30 seconds.
  Do not accumulate complete audio or subtitle history in memory.
- Always request Whisper segment timestamps and preserve the original media
  timeline, including audio/video offsets and skipped silence.
- Remove incomplete subtitles on failure or interruption. Diagnostics go to stderr.
- Render terminal progress with tqdm on stderr, disabled for redirected output;
  keep progress callbacks independent of terminal presentation in the service.

## Validation and packaging

Use pytest for changed logic. Run:

1. `.venv\Scripts\python.exe -m pytest`
2. `.venv\Scripts\python.exe -m py_compile local-ai-transcribe.py local_ai/cli.py`
3. A local CPU file smoke test when inference behavior changes:
   `.venv\Scripts\python.exe local-ai-transcribe.py <file> --device CPU --model <model_dir> --offline`
4. For packaging changes, build `local-ai-transcribe.spec` and run the resulting
   executable with `--help` and a local file when available.

On Linux use `.venv/bin/python`. `make test` runs pytest; `make build` freezes the
CLI. Keep runtime installs in GNUmakefile, README.md, and the release workflow
aligned with the checked-in spec. The workflow uses that spec directly. Preserve
OpenVINO/tokenizer libraries and Hugging Face Xet binaries/data in frozen builds.

Untracked models, tools, and runtime data are user-owned. Do not remove them as
part of implementation cleanup.
