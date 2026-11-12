from __future__ import annotations

import pathlib
import sys
import types
from contextlib import contextmanager

import pytest

from local_ai.infrastructure.openvino import whisper
from local_ai.shared.domain.devices import DeviceSelection
from local_ai.shared.domain.errors import PipelineSetupError
from local_ai.shared.domain.models import ModelArtifact


@pytest.fixture(autouse=True)
def compilation_cache(monkeypatch, tmp_path):
    """Keep compilation cache writes inside each test's temporary directory."""
    monkeypatch.setattr(whisper, "compiled_model_cache_dir", lambda device: tmp_path / "cache" / device)


def test_build_generate_kwargs_always_transcribes_with_requested_options() -> None:
    args = types.SimpleNamespace(language="<|en|>", task="translate", timestamps=True)

    kwargs = whisper.build_generate_kwargs(args)

    assert kwargs == {"language": "<|en|>", "task": "transcribe", "return_timestamps": True}


def test_likely_reason_details_adds_known_hints() -> None:
    details = whisper.likely_reason_details(Exception("Unsupported property STATIC_PIPELINE by CPU plugin"))
    assert any("STATIC_PIPELINE" in line for line in details)


def test_preparation_indicator_starts_after_model_resolution(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        whisper, "resolve_device_selection",
        lambda raw: DeviceSelection(requested=("CPU",), available=["CPU"], selected="CPU"),
    )

    def resolve_model(**kwargs):
        events.append("model resolved")
        return ModelArtifact(path=tmp_path / "model", source="download")

    @contextmanager
    def preparation(device):
        assert device == "CPU"
        events.append("preparing")
        yield
        events.append("ready")

    monkeypatch.setattr(whisper, "resolve_model_artifact", resolve_model)
    monkeypatch.setattr(whisper, "_create_pipeline", lambda *args: events.append("compiled") or object())
    whisper.create_whisper_runtime(
        args=types.SimpleNamespace(device="CPU"),
        base_dir=tmp_path,
        preparation_progress=preparation,
    )
    assert events == ["model resolved", "preparing", "compiled", "ready"]


@pytest.mark.parametrize("device", ["GPU", "NPU"])
def test_create_whisper_runtime_uses_selected_device_and_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, device: str) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakePipeline:
        def __init__(self, model_dir: str, device: str, **kwargs: object) -> None:
            calls.append((model_dir, device, kwargs))

    fake_module = types.SimpleNamespace(WhisperPipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake_module)
    monkeypatch.setattr(
        whisper,
        "resolve_device_selection",
        lambda raw: DeviceSelection(requested=(device, "CPU"), available=[f"{device}.0"], selected=device),
    )
    monkeypatch.setattr(
        whisper,
        "resolve_model_artifact",
        lambda **kwargs: ModelArtifact(path=tmp_path / "model", source="local"),
    )

    logs: list[str] = []
    args = types.SimpleNamespace(device=f"{device},CPU", model=None, offline=False, language=None, task=None, timestamps=False)

    runtime = whisper.create_whisper_runtime(
        args=args,
        base_dir=tmp_path,
        logger=lambda message, verbose, start: logs.append(message),
        verbose=True,
        start_time=1.0,
    )

    assert runtime.selected_device == device
    assert runtime.model_dir == tmp_path / "model"
    assert calls == [(str(tmp_path / "model"), device, {
        "STATIC_PIPELINE": True, "CACHE_DIR": str(tmp_path / "cache" / device),
    })]
    assert (tmp_path / "cache" / device).is_dir()
    assert any(f"Selected device: {device}" == line for line in logs)
    assert f"Compilation cache: {tmp_path / 'cache' / device}" in logs


def test_create_whisper_runtime_falls_back_for_cpu_without_static_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakePipeline:
        def __init__(self, model_dir: str, device: str, **kwargs: object) -> None:
            calls.append((model_dir, device, kwargs))
            if kwargs.get("STATIC_PIPELINE") is True:
                raise RuntimeError("Unsupported property STATIC_PIPELINE by CPU plugin")

    fake_module = types.SimpleNamespace(WhisperPipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake_module)
    monkeypatch.setattr(
        whisper,
        "resolve_device_selection",
        lambda raw: DeviceSelection(requested=("CPU",), available=["CPU"], selected="CPU"),
    )
    monkeypatch.setattr(
        whisper,
        "resolve_model_artifact",
        lambda **kwargs: ModelArtifact(path=tmp_path / "model", source="local"),
    )

    args = types.SimpleNamespace(device="CPU", model=None, offline=False, language=None, task=None, timestamps=False)
    runtime = whisper.create_whisper_runtime(args=args, base_dir=tmp_path)

    assert runtime.selected_device == "CPU"
    assert calls == [
        (str(tmp_path / "model"), "CPU", {
            "STATIC_PIPELINE": True, "CACHE_DIR": str(tmp_path / "cache" / "CPU"),
        }),
        (str(tmp_path / "model"), "CPU", {"CACHE_DIR": str(tmp_path / "cache" / "CPU" / "fallback")}),
    ]


def test_create_whisper_runtime_raises_pipeline_setup_error_when_pipeline_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class FakePipeline:
        def __init__(self, model_dir: str, device: str, **kwargs: object) -> None:
            raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(WhisperPipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake_module)
    monkeypatch.setattr(
        whisper,
        "resolve_device_selection",
        lambda raw: DeviceSelection(requested=("GPU",), available=["GPU"], selected="GPU"),
    )
    monkeypatch.setattr(
        whisper,
        "resolve_model_artifact",
        lambda **kwargs: ModelArtifact(path=tmp_path / "model", source="local"),
    )

    args = types.SimpleNamespace(device="GPU", model=None, offline=False, language=None, task=None, timestamps=False)
    with pytest.raises(PipelineSetupError, match="Failed to create WhisperPipeline on GPU"):
        whisper.create_whisper_runtime(args=args, base_dir=tmp_path)


def test_cache_directory_failure_is_diagnostic_before_compilation(monkeypatch, tmp_path):
    invalid_cache = tmp_path / "cache-file"
    invalid_cache.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(whisper, "compiled_model_cache_dir", lambda device: invalid_cache)
    monkeypatch.setattr(
        whisper, "resolve_device_selection",
        lambda raw: DeviceSelection(requested=("CPU",), available=["CPU"], selected="CPU"),
    )
    monkeypatch.setattr(
        whisper, "resolve_model_artifact",
        lambda **kwargs: ModelArtifact(path=tmp_path / "model", source="local"),
    )

    def unexpected_compile(*args):
        raise AssertionError("Compilation must not run without a usable cache directory")

    monkeypatch.setattr(whisper, "_create_pipeline", unexpected_compile)
    with pytest.raises(PipelineSetupError, match="Cannot create model compilation cache") as exc:
        whisper.create_whisper_runtime(args=types.SimpleNamespace(device="CPU"), base_dir=tmp_path)
    assert str(invalid_cache) in exc.value.details[0]
