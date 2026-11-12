from __future__ import annotations

import pathlib
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Callable

from local_ai.infrastructure.openvino.runtime_env import compiled_model_cache_dir
from local_ai.shared.domain.devices import DeviceSelection, resolve_device_selection
from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError
from local_ai.shared.domain.models import resolve_model_artifact


@dataclass
class WhisperRuntime:
    pipe: object
    generate_kwargs: dict[str, object]
    selected_device: str
    model_dir: pathlib.Path
    preferred_devices: tuple[str, ...]
    available_devices: list[str]


def build_generate_kwargs(args: object) -> dict[str, object]:
    """Always transcribe speech, with optional language and timestamp settings."""
    kwargs: dict[str, object] = {"task": "transcribe"}
    language = getattr(args, "language", None)
    timestamps = bool(getattr(args, "timestamps", False))
    if language:
        kwargs["language"] = language
    if timestamps:
        kwargs["return_timestamps"] = True
    return kwargs


def likely_reason_details(exc: Exception) -> list[str]:
    message = str(exc)
    details = [f"Runtime error: {message}"]
    if "Upper bounds were not specified" in message:
        details.append("NPU compiler rejected dynamic bounds for this model/runtime combination.")
    if "OpenVINO and OpenVINO Tokenizers versions are not binary compatible" in message:
        details.append("Align package versions: pip install -U openvino openvino-genai openvino-tokenizers")
    if "openvino_tokenizers.dll" in message:
        details.append("Frozen build is missing tokenizer runtime library; rebuild with tokenizer binaries/data included.")
    if "Unsupported property STATIC_PIPELINE by CPU plugin" in message:
        details.append("This OpenVINO CPU plugin version does not support STATIC_PIPELINE.")
    details.append("Ensure OpenVINO runtime, drivers, and model export are compatible.")
    return details


def _create_pipeline(model_dir: pathlib.Path, selected: str, cache_dir: pathlib.Path) -> object:
    """Create a device pipeline while preserving caching through CPU fallback."""
    try:
        import openvino_genai as ov_genai
    except Exception as exc:
        raise PipelineSetupError("openvino_genai is not available.", [f"Import error: {exc}", "Install OpenVINO GenAI and retry."]) from exc

    pipeline_kwargs: dict[str, object] = {"STATIC_PIPELINE": True, "CACHE_DIR": str(cache_dir)}
    if selected == "CPU":
        try:
            return ov_genai.WhisperPipeline(str(model_dir), selected, **pipeline_kwargs)
        except Exception:
            try:
                # Keep the static attempt's settings separate from the fallback.
                return ov_genai.WhisperPipeline(str(model_dir), selected, CACHE_DIR=str(cache_dir / "fallback"))
            except Exception as exc:
                raise PipelineSetupError(f"Failed to create WhisperPipeline on {selected}.", likely_reason_details(exc)) from exc

    try:
        return ov_genai.WhisperPipeline(str(model_dir), selected, **pipeline_kwargs)
    except Exception as exc:
        raise PipelineSetupError(f"Failed to create WhisperPipeline on {selected}.", likely_reason_details(exc)) from exc


def create_whisper_runtime(
    *,
    args: object,
    base_dir: pathlib.Path,
    logger: Callable[[str, bool, float | None], None] | None = None,
    verbose: bool = False,
    start_time: float | None = None,
    preparation_progress: Callable[[str], AbstractContextManager[None]] | None = None,
) -> WhisperRuntime:
    device_selection: DeviceSelection = resolve_device_selection(getattr(args, "device"))
    model_artifact = resolve_model_artifact(
        model_arg=getattr(args, "model", None),
        selected_device=device_selection.selected,
        base_dir=base_dir,
        offline=bool(getattr(args, "offline", False)),
    )
    cache_dir = compiled_model_cache_dir(device_selection.selected)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PipelineSetupError(
            "Cannot create model compilation cache.",
            [f"Cache directory: {cache_dir}", f"Filesystem error: {exc}"],
        ) from exc
    if logger is not None:
        logger(f"Compilation cache: {cache_dir}", verbose, start_time)
    with preparation_progress(device_selection.selected) if preparation_progress else nullcontext():
        pipe = _create_pipeline(model_artifact.path, device_selection.selected, cache_dir)

    if logger is not None:
        logger(f"Requested device order: {', '.join(device_selection.requested)}", verbose, start_time)
        logger("Backend: openvino", verbose, start_time)
        logger(f"Selected device: {device_selection.selected}", verbose, start_time)
        logger(f"Model directory: {model_artifact.path}", verbose, start_time)

    return WhisperRuntime(
        pipe=pipe,
        generate_kwargs=build_generate_kwargs(args),
        selected_device=device_selection.selected,
        model_dir=model_artifact.path,
        preferred_devices=device_selection.requested,
        available_devices=device_selection.available,
    )
