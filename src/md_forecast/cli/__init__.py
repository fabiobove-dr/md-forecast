"""Command-line entry point; forecasting commands arrive with their issues."""

import argparse
import asyncio
from pathlib import Path

from pydantic import ValidationError

from md_forecast import __version__
from md_forecast.core import constants
from md_forecast.core.config import DownloadSettings, Settings
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
    _add_extraction_parser(commands)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "download":
            _download(args)
        else:
            _extract(args)
    except (MDForecastError, ValidationError) as error:
        parser.exit(2, f"md-forecast: {error}\n")


def _download(args: argparse.Namespace) -> None:
    from md_forecast.data.acquisition import acquire, load_source

    overrides = {} if args.destination is None else {"destination": args.destination}
    settings = DownloadSettings(**overrides)
    configure_logging(settings.log_level)
    asyncio.run(acquire(load_source(args.source), settings, args.mode))


def _add_extraction_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    extract = commands.add_parser(
        "extract-misato", help="Export selected native MISATO series"
    )
    extract.add_argument("--source", type=Path, default=constants.MISATO_CONFIG)
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--splits", type=Path, required=True)
    extract.add_argument("--artifact", choices=("md", "sample"), default="md")
    extract.add_argument("--systems", nargs="+", required=True)


def _extract(args: argparse.Namespace) -> None:
    from md_forecast.data.public.misato import (
        ExtractionConfig,
        extract_misato,
        load_misato_source,
    )

    settings = Settings()
    configure_logging(settings.log_level)
    config = ExtractionConfig(
        input_path=args.input,
        output_dir=args.output,
        splits_dir=args.splits,
        artifact=args.artifact,
        system_ids=tuple(args.systems),
    )
    extract_misato(load_misato_source(args.source), config)
