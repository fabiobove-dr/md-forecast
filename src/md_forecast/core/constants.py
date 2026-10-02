"""Stable defaults shared by configuration and runtime helpers."""

from pathlib import Path
from typing import Final, Literal

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_DATA_DIR: Final = Path("data")
DEFAULT_LOG_LEVEL: Final[LogLevel] = "INFO"
ENV_PREFIX: Final = "MD_FORECAST_"
LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(name)s %(message)s"
