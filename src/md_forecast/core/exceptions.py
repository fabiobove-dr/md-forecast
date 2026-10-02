"""Base error for actionable failures in future dataset and model code."""


class MDForecastError(Exception):
    """Base class for project-specific errors; preserve causes with chaining."""


class AcquisitionError(MDForecastError):
    """Source metadata, transport, storage, or integrity prevents acquisition."""


class DataContractError(MDForecastError):
    """Canonical registry or numerical series violates the data contract."""


class ForecastError(MDForecastError):
    """Forecast configuration, fitting or prediction cannot be completed safely."""
