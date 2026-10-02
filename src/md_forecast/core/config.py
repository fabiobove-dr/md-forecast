"""Validated configuration loaded explicitly at application boundaries."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from md_forecast.core.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_LOG_LEVEL,
    ENV_PREFIX,
    LogLevel,
)


class Settings(BaseSettings):
    """Runtime paths and diagnostics, overridable with MD_FORECAST_ variables.

    Attributes:
        data_dir: Local data root; construction does not create it.
        log_level: Standard Python logging severity.
    """

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="forbid")

    data_dir: Path = DEFAULT_DATA_DIR
    log_level: LogLevel = DEFAULT_LOG_LEVEL
