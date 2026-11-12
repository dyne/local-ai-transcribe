from __future__ import annotations

import os
import pathlib
import sys
from types import SimpleNamespace

from local_ai.infrastructure.openvino import runtime_env
from local_ai.infrastructure.openvino.runtime_env import configure_openvino_runtime_env


def test_configure_openvino_runtime_env_noops_when_not_frozen(monkeypatch) -> None:
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.delenv("OPENVINO_LIB_PATHS", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    configure_openvino_runtime_env()

    assert os.environ.get("OPENVINO_LIB_PATHS") is None


def test_configure_openvino_runtime_env_prepends_existing_paths(tmp_path: pathlib.Path, monkeypatch) -> None:
    meipass = tmp_path / "bundle"
    libs = meipass / "openvino" / "libs"
    tokenizers = meipass / "openvino_tokenizers" / "libs"
    exe_dir = tmp_path / "dist"
    exe_libs = exe_dir / "openvino_genai"
    libs.mkdir(parents=True)
    tokenizers.mkdir(parents=True)
    exe_libs.mkdir(parents=True)
    fake_exe = exe_dir / "local-ai-transcribe.exe"
    fake_exe.write_text("x", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe), raising=False)
    monkeypatch.setenv("PATH", "existing-path")
    monkeypatch.setenv("OPENVINO_LIB_PATHS", "existing-lib-path")

    configure_openvino_runtime_env()

    path_parts = os.environ["PATH"].split(os.pathsep)
    lib_parts = os.environ["OPENVINO_LIB_PATHS"].split(os.pathsep)
    assert str(libs) in path_parts
    assert str(tokenizers) in path_parts
    assert str(exe_libs) in path_parts
    assert path_parts[-1] == "existing-path"
    assert lib_parts[-1] == "existing-lib-path"


def test_windows_cache_uses_local_appdata_and_separates_devices(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_env, "sys", SimpleNamespace(platform="win32", frozen=True))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    base = tmp_path / "local-ai-transcribe" / "cache" / "openvino"
    assert runtime_env.compiled_model_cache_dir("CPU") == base / "CPU"
    assert runtime_env.compiled_model_cache_dir("GPU") == base / "GPU"
    assert runtime_env.compiled_model_cache_dir("NPU") == base / "NPU"


def test_linux_cache_uses_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_env, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert runtime_env.compiled_model_cache_dir("CPU") == tmp_path / "local-ai-transcribe" / "openvino" / "CPU"


def test_cache_location_falls_back_to_user_home(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(runtime_env, "sys", SimpleNamespace(platform="win32"))
    assert runtime_env.compiled_model_cache_dir("CPU") == (
        tmp_path / "AppData" / "Local" / "local-ai-transcribe" / "cache" / "openvino" / "CPU"
    )
    monkeypatch.setattr(runtime_env, "sys", SimpleNamespace(platform="linux"))
    assert runtime_env.compiled_model_cache_dir("CPU") == tmp_path / ".cache" / "local-ai-transcribe" / "openvino" / "CPU"
