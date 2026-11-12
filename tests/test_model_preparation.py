"""Verify live preparation feedback and refresher cleanup."""

import io
from threading import Event, Thread

import pytest

from local_ai import model_preparation


@pytest.fixture
def live_progress(monkeypatch):
    bars, threads = [], []
    refreshed = Event()

    class Progress:
        """Record progress while allowing the real refresher thread to run."""

        write = staticmethod(lambda *args, **kwargs: None)

        def __init__(self, **kwargs):
            self.options = kwargs
            self.disable = False
            self.n = 0
            self.closed = False
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def refresh(self):
            refreshed.set()

        def update(self, amount):
            self.n += amount

    def create_thread(**kwargs):
        thread = Thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(model_preparation, "tqdm", Progress)
    monkeypatch.setattr(model_preparation, "Thread", create_thread)
    return bars, threads, refreshed


def test_elapsed_time_refreshes_until_preparation_completes(live_progress):
    bars, threads, refreshed = live_progress
    with model_preparation.model_preparation_progress("NPU"):
        assert refreshed.wait(2), "Preparation indicator did not refresh"
        assert bars[0].n == 0
    assert bars[0].n == 1
    assert bars[0].closed
    assert "NPU" in bars[0].options["desc"]
    assert all(not thread.is_alive() for thread in threads)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failure_stops_indicator_without_claiming_completion(live_progress, error):
    bars, threads, refreshed = live_progress
    with pytest.raises(error):
        with model_preparation.model_preparation_progress("GPU"):
            raise error()
    assert bars[0].n == 0
    assert bars[0].closed
    assert all(not thread.is_alive() for thread in threads)


def test_redirected_output_gets_plain_preparation_message(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(model_preparation.sys, "stderr", output)
    with model_preparation.model_preparation_progress("CPU"):
        pass
    assert output.getvalue() == "Preparing Whisper model on CPU...\n"
