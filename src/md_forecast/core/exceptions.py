"""Base error for actionable failures in future dataset and model code."""


class MDForecastError(Exception):
    """Base class for project-specific errors; preserve causes with chaining."""
