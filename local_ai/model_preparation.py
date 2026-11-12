"""Terminal feedback while OpenVINO loads and compiles the Whisper model."""

from collections.abc import Iterator
from contextlib import contextmanager
import sys
from threading import Event, Thread

from tqdm import tqdm


def _refresh_progress(progress: tqdm, stopped: Event) -> None:
    """Refresh elapsed time without estimating unreported compilation progress."""
    while not stopped.wait(0.2):
        progress.refresh()


@contextmanager
def model_preparation_progress(device: str) -> Iterator[None]:
    """Show preparation activity, marking completion only when loading succeeds."""
    tqdm.write(f"Preparing Whisper model on {device}...", file=sys.stderr)
    with tqdm(
        total=1,
        desc=f"Preparing model ({device})",
        bar_format="{desc}: |{bar}| elapsed {elapsed}",
        file=sys.stderr,
        disable=None,
        dynamic_ncols=True,
    ) as progress:
        stopped = Event()
        refresher = None
        if not progress.disable:
            refresher = Thread(target=_refresh_progress, args=(progress, stopped), daemon=True)
            refresher.start()
        try:
            yield
            progress.update(1)
        finally:
            stopped.set()
            if refresher is not None:
                refresher.join()
