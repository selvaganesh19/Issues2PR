"""Worker entrypoint: ``python -m app.workers``.

Builds the best available queue for the environment (real Redis, ``fake://``
in-process Redis, or the in-memory fallback) and consumes jobs, handing each to
:func:`app.workers.tasks.process_job`. Runs until interrupted (Ctrl-C).
"""

from __future__ import annotations

import logging
import signal
import threading

from app.config import get_settings
from app.workers.queue import Job, get_queue, run_worker
from app.workers.tasks import process_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("issue2pr.worker")


def main() -> None:
    """Run the job-consuming worker loop until a stop signal is received."""
    settings = get_settings()
    queue = get_queue(settings)
    logger.info(
        "worker starting (queue=%s, sandbox=%s)",
        type(queue).__name__,
        settings.sandbox_backend,
    )

    stop = threading.Event()

    def _handle_signal(signum, _frame) -> None:  # noqa: ANN001
        logger.info("received signal %s, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def _handler(job: Job) -> object:
        logger.info("processing job delivery_id=%s", job.delivery_id)
        return process_job(job, settings)

    run_worker(queue, _handler, stop=stop)
    logger.info("worker stopped")


if __name__ == "__main__":
    main()
