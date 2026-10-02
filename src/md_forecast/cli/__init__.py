"""Command-line entry point; forecasting commands arrive with their issues."""

import argparse
import asyncio
from pathlib import Path

from pydantic import ValidationError

from md_forecast import __version__
from md_forecast.core import constants
from md_forecast.core.config import DownloadSettings
from md_forecast.core.exceptions import MDForecastError
from md_forecast.core.logging import configure_logging


def main() -> None:
    """Expose help, version, and explicit dataset acquisition."""
    parser = argparse.ArgumentParser(
        prog="md-forecast",
        description="Forecast MD observables from partial trajectories.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    download = commands.add_parser("download", help="Acquire verified MISATO artifacts")
    download.add_argument("--source", type=Path, default=constants.MISATO_CONFIG)
    download.add_argument("--destination", type=Path)
    download.add_argument("--mode", choices=("metadata", "md"), default="metadata")
    args = parser.parse_args()
    if args.command != "download":
        parser.print_help()
        return
    try:
        _download(args)
    except (MDForecastError, ValidationError) as error:
        parser.exit(2, f"md-forecast: {error}\n")


def _download(args: argparse.Namespace) -> None:
    from md_forecast.data.acquisition import acquire, load_source

    overrides = {} if args.destination is None else {"destination": args.destination}
    settings = DownloadSettings(**overrides)
    configure_logging(settings.log_level)
    asyncio.run(acquire(load_source(args.source), settings, args.mode))
