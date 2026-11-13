"""Terminal activity feedback while OpenVINO loads and compiles Whisper."""

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import cycle
import multiprocessing
import sys

from tqdm import tqdm


def _render_preparation(device: str, stopped, ready) -> None:
    """Animate independently of native model loading holding the parent's GIL."""
    label = f"Preparing model ({device})"
    with tqdm(
        total=None,
        desc=f"{label}: |",
        bar_format="{desc} elapsed {elapsed}",
        file=sys.stderr,
        disable=False,
        dynamic_ncols=True,
    ) as progress:
        ready.set()
        frames = cycle("/-\\|")
        try:
            while not stopped.wait(0.2):
                progress.set_description_str(f"{label}: {next(frames)}")
        except KeyboardInterrupt:
            # The parent owns cancellation and stops this renderer in its finally.
            pass


@contextmanager
def model_preparation_progress(device: str) -> Iterator[None]:
    """Show a spinner and elapsed time, then announce readiness only on success."""
    tqdm.write(f"Preparing Whisper model on {device}...", file=sys.stderr)
    renderer = None
    if sys.stderr.isatty():
        context = multiprocessing.get_context("spawn")
        stopped = context.Event()
        ready = context.Event()
        renderer = context.Process(target=_render_preparation, args=(device, stopped, ready), daemon=True)
        renderer.start()
    try:
        if renderer is not None:
            ready.wait(timeout=5)
        yield
    finally:
        if renderer is not None:
            stopped.set()
            renderer.join(timeout=5)
            if renderer.is_alive():
                renderer.terminate()
                renderer.join()
            renderer.close()
    tqdm.write(f"Model ready ({device}).", file=sys.stderr)
