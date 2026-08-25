"""Logging configuration for the medical triage application."""

import logging
import sys

DEFAULT_LOG_LEVEL = "INFO"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(
    log_level: str = DEFAULT_LOG_LEVEL,
) -> None:
    """Configure application-wide logging.

    The configuration sends logs to standard output, which is suitable
    for local execution, Docker containers, and cloud environments.

    Args:
        log_level: Logging level such as DEBUG, INFO, WARNING, ERROR,
            or CRITICAL.
    """
    normalized_level = log_level.upper()

    numeric_level = getattr(
        logging,
        normalized_level,
        logging.INFO,
    )

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance for a given module.

    Args:
        name: Logger name, normally the module ``__name__``.

    Returns:
        Configured Python logger instance.
    """
    return logging.getLogger(name)
