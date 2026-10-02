"""Stable defaults shared by configuration and runtime helpers."""

from enum import StrEnum
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

CANONICAL_SCHEMA_VERSION: Final = 1
PS_PER_NS: Final = 1000.0
TIME_RTOL: Final = 1e-9
TIME_ATOL: Final = 1e-10
SERIES_METADATA_KEY: Final = b"md_forecast.series"
TIME_COLUMN: Final = "time"
RATIO_ATOL: Final = 1e-12
DEFAULT_GROUP_FIELD: Final = "system_id"
SHA256_PATTERN: Final = r"^sha256:[0-9a-f]{64}$"
CONSTANT_FEATURE_SCALE: Final = 1.0


class DatasetId(StrEnum):
    """Supported public dataset namespaces."""

    MISATO = "misato"
    MDBIND = "mdbind"


class Split(StrEnum):
    """Closed vocabulary for dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TimeUnit(StrEnum):
    """Frame indices are explicitly not physical time."""

    FRAME = "frame"
    PS = "ps"
    NS = "ns"


class SamplingStatus(StrEnum):
    """Distinguish confirmed sampling from assumptions or missing metadata."""

    VERIFIED = "verified"
    ASSUMED = "assumed"
    UNAVAILABLE = "unavailable"


class FeatureUnit(StrEnum):
    """Units admitted by the initial observable contract."""

    ANGSTROM = "angstrom"
    ANGSTROM_SQUARED = "angstrom_squared"
    KCAL_PER_MOL = "kcal_per_mol"
    DIMENSIONLESS = "dimensionless"
    COUNT = "count"


class ModelId(StrEnum):
    """Baseline identities; persistence is the mandatory reference."""

    PERSISTENCE = "persistence"
    CONTEXT_MEAN = "context-mean"
    LINEAR = "linear-extrapolation"
    AR = "autoregression"
    VAR = "var"
    NLINEAR = "nlinear"
    CHRONOS2 = "chronos-2"
