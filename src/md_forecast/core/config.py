"""Validated configuration loaded explicitly at application boundaries."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from md_forecast.core import constants
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


class DownloadSettings(Settings):
    """Bounded acquisition settings; environment names use MD_FORECAST_."""

    model_config = SettingsConfigDict(allow_inf_nan=False)

    destination: Path | None = None
    concurrency: int = Field(default=constants.DOWNLOAD_CONCURRENCY, gt=0)
    chunk_bytes: int = Field(default=constants.DOWNLOAD_CHUNK_BYTES, gt=0)
    disk_reserve_bytes: int = Field(default=constants.DOWNLOAD_DISK_RESERVE_BYTES, ge=0)
    connect_timeout: float = Field(default=constants.DOWNLOAD_CONNECT_TIMEOUT, gt=0)
    read_timeout: float = Field(default=constants.DOWNLOAD_READ_TIMEOUT, gt=0)
    write_timeout: float = Field(default=constants.DOWNLOAD_WRITE_TIMEOUT, gt=0)
    pool_timeout: float = Field(default=constants.DOWNLOAD_POOL_TIMEOUT, gt=0)
    retries: int = Field(default=constants.DOWNLOAD_RETRIES, ge=0)
    retry_delay: float = Field(default=constants.DOWNLOAD_RETRY_DELAY, ge=0)
