import logging
from importlib.metadata import version
from pathlib import Path

import pytest
from pydantic import ValidationError

from md_forecast import __version__
from md_forecast.cli import main
from md_forecast.core.config import Settings
from md_forecast.core.constants import DEFAULT_DATA_DIR, DEFAULT_LOG_LEVEL
from md_forecast.core.exceptions import MDForecastError
from md_forecast.core.logging import configure_logging


def test_package_version() -> None:
    assert version("md-forecast") == __version__
    assert issubclass(MDForecastError, Exception)


def test_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MD_FORECAST_DATA_DIR", raising=False)
    monkeypatch.delenv("MD_FORECAST_LOG_LEVEL", raising=False)
    defaults = Settings()
    assert defaults.data_dir == DEFAULT_DATA_DIR
    assert defaults.log_level == DEFAULT_LOG_LEVEL

    data_dir = tmp_path / "not-created"
    monkeypatch.setenv("MD_FORECAST_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MD_FORECAST_LOG_LEVEL", "DEBUG")
    settings = Settings()
    assert settings.data_dir == data_dir
    assert settings.log_level == "DEBUG"
    assert not data_dir.exists()

    monkeypatch.setenv("MD_FORECAST_LOG_LEVEL", "invalid")
    with pytest.raises(ValidationError, match="log_level"):
        Settings()
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Settings.model_validate({"unknown": "value"})


def test_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate the root logger so pytest's own handlers are not changed.
    root = logging.RootLogger(logging.WARNING)
    monkeypatch.setattr(logging, "root", root)
    configure_logging("DEBUG")
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    formatter = root.handlers[0].formatter
    assert formatter is not None
    record = logging.LogRecord("test", logging.INFO, "", 0, "ready", (), None)
    assert "INFO test ready" in formatter.format(record)
    configure_logging()
    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG
    root.handlers[0].close()


def test_cli_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["md-forecast"])
    main()
    assert "--version" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argument", "exit_code", "output"),
    [("--help", 0, "usage:"), ("--version", 0, __version__), ("unknown", 2, "")],
)
def test_cli_arguments(
    argument: str,
    exit_code: int,
    output: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["md-forecast", argument])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == exit_code
    captured = capsys.readouterr()
    assert output in captured.out
    if exit_code:
        assert "invalid choice" in captured.err
