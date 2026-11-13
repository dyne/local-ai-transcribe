"""Keep CLI packaging independent of removed application services."""

from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace


def test_spec_collects_only_transcription_runtime(monkeypatch):
    collected = []
    analyses = []
    executables = []
    hooks = ModuleType("PyInstaller.utils.hooks")
    hooks.collect_data_files = lambda name: collected.append(("data", name)) or []
    hooks.collect_dynamic_libs = lambda name: collected.append(("binary", name)) or []
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)

    def analysis(scripts, **kwargs):
        analyses.append((scripts, kwargs))
        return SimpleNamespace(pure=[], scripts=scripts, binaries=[], datas=[])

    runpy.run_path("local-ai-transcribe.spec", init_globals={
        "Analysis": analysis,
        "PYZ": lambda pure: pure,
        "EXE": lambda *args, **kwargs: executables.append(kwargs),
    })
    assert analyses[0][0] == ["local-ai-transcribe.py"]
    assert executables[0]["name"] == "local-ai-transcribe"
    for package in ("openvino", "openvino_genai", "openvino_tokenizers", "hf_xet"):
        assert ("data", package) in collected
        assert ("binary", package) in collected
    assert "webrtcvad" in analyses[0][1]["hiddenimports"]
    assert "hf_xet" in analyses[0][1]["hiddenimports"]
    assert "tqdm" in analyses[0][1]["hiddenimports"]
    assert analyses[0][1]["datas"] == []
    assert "fastapi" in analyses[0][1]["excludes"]
    assert "torch" in analyses[0][1]["excludes"]


def test_release_and_make_use_shared_spec_without_web_dependencies():
    workflow = Path(".github/workflows/build-and-release.yaml").read_text(encoding="utf-8")
    makefile = Path("GNUmakefile").read_text(encoding="utf-8")
    assert "--clean local-ai-transcribe.spec" in workflow
    assert "--clean $(SPEC)" in makefile
    assert "windows-latest" in workflow and "ubuntu-latest" in workflow
    for text in (workflow, makefile):
        for removed in ("uvicorn", "faiss", "pywebview", "sounddevice", "frontend", "npm"):
            assert removed not in text
        for required in ("av", "tqdm", "noisereduce", "webrtcvad-wheels", "hf_xet", "openvino-genai", "openvino-tokenizers"):
            assert required in text


def test_release_versions_main_and_publishes_only_after_both_builds():
    workflow = Path(".github/workflows/build-and-release.yaml").read_text(encoding="utf-8")
    assert "branches: [main]" in workflow
    assert "uses: ietf-tools/semver-action@v1" in workflow
    assert "branch: ${{ github.sha }}" in workflow
    assert 'git tag "$RELEASE_TAG" "$GITHUB_SHA"' in workflow
    assert "needs: version" in workflow
    assert "needs: [version, build]" in workflow
    assert "uses: actions/download-artifact@v4" in workflow
    assert "tag_name: ${{ needs.version.outputs.tag }}" in workflow
    assert "fail_on_unmatched_files: true" in workflow
    assert "release-assets/local-ai-transcribe-windows-x64.zip" in workflow
    assert "release-assets/local-ai-transcribe-linux-x64.tar.gz" in workflow
    assert "Copy-Item srt2txt.py $stage" in workflow
