"""Command-line entry point; forecasting commands arrive with their issues."""

import argparse

from md_forecast import __version__


def main() -> None:
    """Expose package help and version without initiating data or model work."""
    parser = argparse.ArgumentParser(
        prog="md-forecast",
        description="Forecast MD observables from partial trajectories.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
    parser.print_help()
