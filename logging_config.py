"""Central logging setup for the backend.

The pipeline runs matching jobs on background threads (see ``server.py``), so
operational visibility hinges on two things: every line is timestamped and tagged
with the thread it came from (``job-XXXX``), and the same stream lands both on the
uvicorn console (watch a run live) and in a rotating file (grep it afterwards).

This is deliberately separate from ``services.audit_logger`` — that writes the
per-record *domain* audit trail; this is *operational* logging (what ran, how
long, what failed).

Usage:
    from logging_config import setup_logging, get_logger
    setup_logging()                     # once, at process start
    log = get_logger("server")          # -> logger "mmv.server"
    log.info("job %s started", job_id)

Env overrides (via ``config``): ``MMV_LOG_LEVEL`` (default INFO),
``MMV_LOG_PATH`` (default logs/backend.log).
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config

_ROOT_NAME = "mmv"
_FORMAT = "%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"

# Guard so repeated calls (uvicorn --reload re-imports modules) don't stack
# duplicate handlers and double every log line.
_configured = False


def setup_logging() -> logging.Logger:
    """Configure the ``mmv`` logger tree once; return the root ``mmv`` logger.

    Idempotent: safe to call from every entrypoint and under ``--reload``.
    """
    global _configured
    logger = logging.getLogger(_ROOT_NAME)
    if _configured:
        return logger

    level = getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False  # don't double-emit through the root logger

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_path = Path(config.LOG_PATH)
    if not log_path.is_absolute():
        log_path = Path(__file__).resolve().parent / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _configured = True
    logger.info("logging initialized (level=%s, file=%s)", logging.getLevelName(level), log_path)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``mmv`` tree, e.g. ``mmv.server``.

    Calls ``setup_logging`` lazily so importing a module and logging from it
    works even if the entrypoint forgot to configure logging explicitly.
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
