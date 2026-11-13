"""Verify preparation activity, native-call responsiveness, and cleanup."""

import io
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from local_ai import model_preparation


@pytest.fixture
def renderer(monkeypatch):
    """Record helper lifecycle without launching a child for cleanup tests."""
    processes = []

    class Process:
        def __init__(self, *, target, args, daemon):
            self.args = args
            self.started = False
            self.closed = False
            self.alive = False
            processes.append(self)

        def start(self):
            self.started = True
            self.alive = True
            self.args[2].set()

        def join(self, timeout=None):
            assert self.args[1].is_set()
            self.alive = False

        def is_alive(self):
            return self.alive

        def close(self):
            self.closed = True

    def context(method):
        assert method == "spawn"
        return SimpleNamespace(Event=Event, Process=Process)

    monkeypatch.setattr(model_preparation.multiprocessing, "get_context", context)
    output = io.StringIO()
    output.isatty = lambda: True
    monkeypatch.setattr(model_preparation, "sys", SimpleNamespace(stderr=output))
    return SimpleNamespace(processes=processes, output=output)


def test_preparation_helper_starts_once_and_announces_success(renderer):
    with model_preparation.model_preparation_progress("NPU"):
        assert len(renderer.processes) == 1
        assert renderer.processes[0].started
        assert not renderer.processes[0].args[1].is_set()
    assert renderer.processes[0].closed
    assert not renderer.processes[0].alive
    output = renderer.output.getvalue()
    assert output == "Preparing Whisper model on NPU...\nModel ready (NPU).\n"


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failure_stops_renderer_without_claiming_success(renderer, error):
    with pytest.raises(error):
        with model_preparation.model_preparation_progress("GPU"):
            raise error()
    assert renderer.processes[0].closed
    assert not renderer.processes[0].alive
    assert "Model ready" not in renderer.output.getvalue()


def test_redirected_output_uses_plain_messages_without_a_process(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(model_preparation.sys, "stderr", output)

    def unexpected_process(*args):
        raise AssertionError("Redirected output must not create a renderer process")

    monkeypatch.setattr(model_preparation.multiprocessing, "get_context", unexpected_process)
    with model_preparation.model_preparation_progress("CPU"):
        pass
    assert output.getvalue() == "Preparing Whisper model on CPU...\nModel ready (CPU).\n"


def test_spinner_moves_while_native_code_holds_the_gil():
    script = '''
import ctypes
import sys
from local_ai.model_preparation import model_preparation_progress

if __name__ == "__main__":
    sys.stderr.isatty = lambda: True
    with model_preparation_progress("NPU"):
        # PyDLL deliberately holds the GIL; CDLL would release it.
        if sys.platform == "win32":
            ctypes.PyDLL("kernel32").Sleep(1000)
        else:
            ctypes.PyDLL(None).sleep(1)
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    frames = [line.split(" elapsed ")[0] for line in result.stderr.splitlines() if line.startswith("Preparing model (NPU):")]
    assert len(set(frames)) > 1, result.stderr
    assert "elapsed" in result.stderr
    assert "%" not in result.stderr
    assert "Model ready (NPU)." in result.stderr
