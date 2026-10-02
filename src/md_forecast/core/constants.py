"""Stable defaults shared by configuration and runtime helpers."""

from pathlib import Path
from typing import Final, Literal

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_DATA_DIR: Final = Path("data")
DEFAULT_LOG_LEVEL: Final[LogLevel] = "INFO"
ENV_PREFIX: Final = "MD_FORECAST_"
LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(name)s %(message)s"

type DownloadMode = Literal["metadata", "md"]
MISATO_CONFIG: Final = Path("configs/datasets/misato.yaml")
MISATO_DATA_SUBDIR: Final = Path("external/misato")
SOURCE_CONFIG_VERSION: Final = 1
DOWNLOAD_CONCURRENCY: Final = 2
DOWNLOAD_CHUNK_BYTES: Final = 1024 * 1024
DOWNLOAD_DISK_RESERVE_BYTES: Final = 64 * 1024 * 1024
DOWNLOAD_CONNECT_TIMEOUT: Final = 10.0
DOWNLOAD_READ_TIMEOUT: Final = 60.0
DOWNLOAD_WRITE_TIMEOUT: Final = 30.0
DOWNLOAD_POOL_TIMEOUT: Final = 10.0
DOWNLOAD_RETRIES: Final = 2
DOWNLOAD_RETRY_DELAY: Final = 1.0
PARTIAL_SUFFIX: Final = ".part"
RECEIPT_SUFFIX: Final = ".receipt.json"
CONTENT_RANGE_PATTERN: Final = r"bytes (\d+)-(\d+)/(\d+)"
CHECKSUM_PATTERN: Final = r"(?:md5:[0-9a-f]{32}|sha256:[0-9a-f]{64})"
ARTIFACT_NAME_PATTERN: Final = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
ACQUISITION_LOCK: Final = ".acquisition.lock"
