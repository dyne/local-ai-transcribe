# local-ai-transcribe

Create **SRT subtitles beside your local audio/video files**, using OpenVINO
GenAI Whisper on NPU, GPU, or CPU. Audio is decoded and processed on the machine.
Network access is needed only to download a missing model; `--offline` prohibits
downloads.

This is a file-only application. No desktop/browser UI, microphone recording,
Documents RAG, or OCR services are required.

## Install

Use Python **3.11**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy noisereduce webrtcvad-wheels av "huggingface_hub[hf_xet]" hf_xet openvino openvino-genai openvino-tokenizers
```

On Linux, use `python3.11 -m venv .venv` and `.venv/bin/python`.
Alternatively, `make install` prepares the environment and runtime dependencies.
Node.js and web-server packages are not required.

## Create subtitles

```powershell
.\.venv\Scripts\python.exe .\local-ai-transcribe.py movie.mp4
.\.venv\Scripts\python.exe .\local-ai-transcribe.py first.mp4 second.mp3 recording.wav
```

Outputs are written beside each input, with the same basename:

```text
videos/movie.mp4       -> videos/movie.srt
recordings/meeting.wav -> recordings/meeting.srt
```

The input files are never changed. Subtitles use UTF-8, sequential cue numbers,
and `HH:MM:SS,mmm` timestamps. Diagnostics and completion messages go to stderr;
stdout is unused. Paths containing spaces must be quoted. Supply files explicitly;
shell glob expansion is shell-dependent. Directories are not accepted.

Existing `.srt` files are never overwritten. If two inputs would produce the
same subtitle path, the first successful output is preserved and the second is
reported as an output error. A failed file does not prevent remaining inputs
from running. Incomplete subtitles are removed after failure or Ctrl+C; resuming
partway through a file is not implemented.

To place subtitles in a separate directory:

```powershell
.\.venv\Scripts\python.exe .\local-ai-transcribe.py movie.mp4 --output-dir subtitles
```

This writes `subtitles/movie.srt`. Exit status is zero only when every input
succeeds; otherwise it is the first file error code, or a model/setup error.
Ctrl+C exits with status 130. Files with no detected speech return an error
without leaving an empty `.srt`.

## Large and long MP4 files

The decoder reads **only the first audio stream** with PyAV; video frames are
not decoded. Supported formats depend on its bundled codecs. Files are processed
sequentially using a single shared Whisper runtime.

Audio is decoded in windows of **30 seconds**, denoised and checked for speech,
then resampled to mono float32 at 16 kHz for inference. Cues are written as each
window completes. Memory use for decoded audio and subtitle output stays bounded
by the window size, rather than growing with the duration of the recording.
There is no application-level file-size or duration limit; codec/container
support, available disk space, and processing time still apply.

Whisper's segment timestamps are shifted onto the original media timeline,
including audio/video start offsets and timestamp gaps. Silent windows advance
the timeline even when no inference runs. The final short window is processed.
SRT hours can exceed 24 for multi-day recordings.

```powershell
.\.venv\Scripts\python.exe .\local-ai-transcribe.py long-recording.mp4 --verbose
.\.venv\Scripts\python.exe .\local-ai-transcribe.py long-recording.mp4 --chunk-seconds 15
```

`--chunk-seconds` must be greater than 0 and at most 30. Window boundaries can
split speech and lose context, and subtitle accuracy depends on the model and
recording. These are model-predicted segment times, not forced alignment or
word-level timestamps. Use `--no-silence-detect` if speech detection rejects quiet
or sparse speech; this also disables noise reduction.

## Models and devices

```powershell
.\.venv\Scripts\python.exe .\local-ai-transcribe.py --device list
.\.venv\Scripts\python.exe .\local-ai-transcribe.py movie.mp4 --device CPU
.\.venv\Scripts\python.exe .\local-ai-transcribe.py movie.mp4 --model OpenVINO/whisper-small-fp16-ov
.\.venv\Scripts\python.exe .\local-ai-transcribe.py movie.mp4 --offline --model .\whisper-tiny-fp16-ov
```

The default device order is `NPU,GPU,CPU`, selecting the first available device.
Default models are:

| Selected device | Hugging Face model |
| --- | --- |
| NPU | `OpenVINO/whisper-base.en-int8-ov` |
| GPU / CPU | `OpenVINO/whisper-tiny-fp16-ov` |

Models are resolved from local directories or the shared Hugging Face cache
before downloading. `HF_TOKEN` is supported. An explicit model never silently
falls back to a different model. `--offline` fails if the requested/default model
is unavailable locally. Local models must be complete OpenVINO Whisper exports.
Frozen executables can find default model directories beside themselves.

Retained options include `--language`, `--task transcribe|translate`,
`--no-silence-detect`, the `--vad-*` controls, and `--verbose`. Segment timestamps
are always enabled for SRT output. Use a multilingual model for non-English
transcription or translation; the default NPU model is English-only.
Run `--help` for all options.

## Validate and package

```powershell
.\.venv\Scripts\python.exe -m pip install pytest pyinstaller
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m PyInstaller --clean local-ai-transcribe.spec
.\dist\local-ai-transcribe.exe --help
.\dist\local-ai-transcribe.exe movie.mp4 --device CPU --model .\whisper-tiny-fp16-ov --offline
```

`make test` runs pytest. `make build` installs dependencies and freezes the CLI.
`make run ARGS="movie.mp4 --device CPU"` invokes the source entrypoint.
Windows and Linux release jobs test and build the same checked-in PyInstaller
spec. The executable is named `local-ai-transcribe` (`.exe` on Windows).
Models are downloaded/cached separately, rather than embedded in the executable.

Pushing to `main` starts the release workflow. `ietf-tools/semver-action` calculates
the next version from conventional commits and tags that exact commit. Both
platforms must pass tests, build with PyInstaller, and run the frozen CLI before
the workflow publishes a GitHub release containing the Windows ZIP and Linux
tar.gz assets. The workflow can also be started manually on `main`; rerunning
the same commit reuses its tag.

Optional profiling uses `--profile` and `--profile-output path.svg`; install
`py-spy` separately when needed.

## License

Copyright (C) 2026 by the Dyne.org Foundation.
Distributed under the Affero GNU General Public License v3.
