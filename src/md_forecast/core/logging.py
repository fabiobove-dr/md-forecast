"""Standard logging configured explicitly by applications, never on import."""

import logging

from md_forecast.core.constants import DEFAULT_LOG_LEVEL, LOG_FORMAT, LogLevel


def configure_logging(level: LogLevel = DEFAULT_LOG_LEVEL) -> None:
    """Configure root logging without replacing existing application handlers.

    Args:
        level: Standard logging severity.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)
