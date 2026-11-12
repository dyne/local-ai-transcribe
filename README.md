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
.\.venv\Scripts\python.exe -m pip install numpy noisereduce webrtcvad-wheels av tqdm "huggingface_hub[hf_xet]" hf_xet openvino openvino-genai openvino-tokenizers
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
shell glob expansion is shell-dependent. Use `-R` / `--recursive` for directories.

Each file starts with a separate `Transcribing: <path>` line, followed by its
own progress bar. An interactive terminal shows a `tqdm` progress bar labeled with
its position in the batch and filename. It advances through processed audio
seconds, including silent windows. When duration metadata is available, it shows
percentage and estimated time remaining; otherwise it shows processed seconds
and elapsed time. Bars are written to stderr and disabled automatically when
stderr is redirected. Verbose logs and errors do not overwrite the progress bar.

After model resolution or downloading, a separate preparation indicator shows
elapsed time while OpenVINO loads and compiles Whisper for the selected device.
OpenVINO does not expose a compilation percentage here, so this indicator marks
completion only when the runtime is ready. With redirected stderr, a plain
preparation message is printed instead of the live bar.

Recursively transcribe audio/video files inside one or more directories:

```powershell
.\.venv\Scripts\python.exe .\local-ai-transcribe.py -R "C:\Videos"
.\.venv\Scripts\python.exe .\local-ai-transcribe.py --recursive "C:\Videos" "D:\Recordings"
```

Recursive mode requires directories and searches their subdirectories in stable
order. It selects common media extensions case-insensitively (including MP4,
MKV, MOV, AVI, WebM, WAV, MP3, M4A, AAC, FLAC, Ogg, and Opus), ignoring subtitles,
images, and other files. Codec support and the presence of an audio track are
checked during decoding. Symbolic links within input directories are not followed, and
overlapping roots do not transcribe the same path twice. An unreadable directory
or a scan with no eligible files is reported before loading a model. Direct file
inputs remain supported without `-R`, regardless of their extension.

Each discovered file gets its `.srt` beside it. With `--output-dir`, outputs use
that single directory; matching basenames can collide and trigger an overwrite prompt.

When a subtitle file already exists, the CLI asks:

```text
File exists: videos/movie.srt
Overwrite? (Y/n)
```

Press Enter or answer `y` / `yes` to replace it, or `n` / `no` to skip that input
and continue the batch. Skipping is not an error. If no answer can be read, the
file is skipped. Use `-y` / `--yes` to answer yes to all prompts:

```powershell
.\.venv\Scripts\python.exe .\local-ai-transcribe.py -R "C:\Videos" --yes
```

Approved overwrites are generated in a temporary file beside the original and
replace it only after transcription succeeds. A failed file does not prevent
remaining inputs from running. Incomplete outputs are removed after failure or
Ctrl+C; existing subtitles remain intact if replacement fails. Resuming partway
through a file is not implemented.

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
Digital silence bypasses denoising and inference. If noise reduction encounters
invalid arithmetic or non-finite output, the original valid audio is used for
speech detection and transcription instead; `--verbose` reports the fallback.
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

VAD defaults to mode `1` and a minimum speech ratio of `0.02`, favoring retention
of quiet or sparse dialogue. Override these with `--vad-mode` and
`--vad-min-speech-ratio` when needed.

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
| NPU | `OpenVINO/whisper-large-v3-turbo-int8-ov` |
| GPU / CPU | `OpenVINO/whisper-tiny-fp16-ov` |

Models are resolved from local directories or the shared Hugging Face cache
before downloading. `HF_TOKEN` is supported. An explicit model never silently
falls back to a different model. `--offline` fails if the requested/default model
is unavailable locally. Local models must be complete OpenVINO Whisper exports.
Frozen executables can find default model directories beside themselves.

OpenVINO compilation results are also cached automatically, separately from
the Hugging Face download cache:

- Windows: `%LOCALAPPDATA%\local-ai-transcribe\cache\openvino\<device>`
- Linux: `$XDG_CACHE_HOME/local-ai-transcribe/openvino/<device>`, defaulting to
  `~/.cache/local-ai-transcribe/openvino/<device>`.

The first run compiles the model; later compatible runs can load compiled
artifacts from disk. OpenVINO manages cache keys and compatibility for the model,
device, and compilation settings. Software or driver updates may require
recompilation, and cache hits still require some loading and device initialization.
`--verbose` prints the compilation cache directory. This works with `--offline`
and adds no network access or Python dependencies.

Options include `--language`, `-R` / `--recursive`, `--no-silence-detect`, the
`--vad-*` controls, and `--verbose`. The Whisper task is always `transcribe`;
translation is not exposed. Segment timestamps are always enabled for SRT output.
Use a multilingual model for non-English transcription; the NPU default uses large-v3-turbo.
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
