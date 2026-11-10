"""Readable console output and a JSON-lines file for each pipeline run."""

import logging
import sys
from pathlib import Path

import structlog

LOGGER_NAME = "aipi_pipeline"


def get_logger(name: str = "pipeline") -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(f"{LOGGER_NAME}.{name}")


def configure_logging(log_file: Path, level: str = "INFO", log_format: str = "console") -> None:
    """Configure our logger namespace without replacing application/root handlers."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    close_logging()
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    structlog.configure(
        processors=[structlog.stdlib.filter_by_level, *shared,
                    structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    console = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    for handler, renderer in (
        (console, structlog.processors.JSONRenderer() if log_format == "json"
         else structlog.dev.ConsoleRenderer(colors=False)),
        (file_handler, structlog.processors.JSONRenderer()),
    ):
        handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        ))
        logger.addHandler(handler)


def flush_logs() -> None:
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()


def close_logging() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
