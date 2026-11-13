#!/usr/bin/env python
"""Run the local file transcription CLI."""

if __name__ == "__main__":
    from multiprocessing import freeze_support

    # Dispatch the frozen progress renderer before importing the application.
    freeze_support()

    from local_ai.cli import main

    raise SystemExit(main())
