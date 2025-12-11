import csv
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from dotenv import load_dotenv

import typer
import logging

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

from . import run_rabbit, RetryableError

load_dotenv()

console_ui = Console(stderr=True)

app = typer.Typer(
    help="RABBIT is an Activity Based Bot Identification Tool that identifies bots "
    "based on their recent activities in GitHub",
    add_completion=False,
)


class OutputFormat(str, Enum):
    TERMINAL = "term"
    CSV = "csv"


def setup_logger(debug: bool = False):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            RichHandler(console=console_ui, rich_tracebacks=True, show_path=False)
        ],
    )

    # Use only warning logs for urllib3 to reduce verbosity
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _get_all_contributors(
    arg_contributors: list[str], input_file: Optional[Path]
) -> list[str]:
    """Combine CLI arguments and file content into a unique list."""
    contributors = arg_contributors.copy() if arg_contributors else []

    if input_file is not None:
        # Read txt file and extract contributors
        file_content = input_file.read_text(encoding="utf-8")
        file_contributors = [
            line.strip() for line in file_content.splitlines() if line.strip()
        ]
        contributors.extend(file_contributors)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(contributors))


class RabbitRenderer:
    """
    Class to render RABBIT progress and results in the terminal using Rich. (on stderr)
    """

    def __init__(self, total_items: int, quiet: bool = False):
        self.quiet = quiet
        self.total_items = total_items

        self.progress = Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• {task.completed}/{task.total}"),
            console=console_ui,
            transient=True,
        )
        self.task_id = self.progress.add_task("", total=self.total_items)

        if not self.quiet:
            self.table = Table(box=box.ROUNDED)
            self.table.add_column("Contributor", style="bold cyan", no_wrap=True)
            self.table.add_column("Type")
            self.table.add_column("Confidence", justify="right")

    def get_renderable(self):
        if not self.quiet:
            return Group(self.progress, self.table)
        return self.progress

    def update(self, result: dict):
        self.progress.advance(self.task_id)
        if not self.quiet:
            self.table.add_row(
                result["contributor"],
                result["type"],
                f"{result['confidence']}",
            )

    def stop(self):
        self.progress.stop()


class DataWriter:
    """
    Class to write RABBIT results to stdout
    """

    def __init__(self, fmt: OutputFormat):
        self.fmt = fmt
        self.active = not sys.stdout.isatty()
        self.output = sys.stdout

        if self.active:
            if self.fmt == OutputFormat.CSV:
                self.csv_writer = csv.DictWriter(
                    sys.stdout, fieldnames=["contributor", "type", "confidence"]
                )
                self.csv_writer.writeheader()
            else:  # TSV format
                print("contributor\ttype\tconfidence")
            sys.stdout.flush()

    def write(self, result: dict):
        if not self.active:
            return
        if self.fmt == OutputFormat.CSV:
            self.csv_writer.writerow(result)
        else:
            print(
                f"{result['contributor']}\t{result['type']}\t{result['confidence']}",
                file=self.output,
            )
        sys.stdout.flush()


@app.command()
def cli(
    # ---- INPUTS ----
    contributors: Annotated[
        list[str],
        typer.Argument(
            help="Login names of contributors to analyze.",
            show_default=False,
        ),
    ] = None,
    input_file: Annotated[
        Optional[Path],
        typer.Option(
            "--input-file",
            "-i",
            help="Path to a file containing login names (one per line).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            rich_help_panel="Inputs",
        ),
    ] = None,
    # ---- CONFIGURATION ----
    key: Annotated[
        Optional[str],
        typer.Option(
            "--key",
            "-k",
            envvar="GITHUB_API_KEY",
            help="GitHub API key (either in command line or in GITHUB_API_KEY env variable).",
            rich_help_panel="Configuration",
        ),
    ] = None,
    min_events: Annotated[
        int,
        typer.Option(
            min=1,
            max=300,
            help="Min number of events required.",
            rich_help_panel="Configuration",
        ),
    ] = 5,
    min_confidence: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Confidence threshold to stop querying.",
            rich_help_panel="Configuration",
        ),
    ] = 1.0,
    max_queries: Annotated[
        int,
        typer.Option(
            min=1,
            max=3,
            help="Max API queries per contributor.",
            rich_help_panel="Configuration",
        ),
    ] = 3,
    # ---- OUTPUTS ----
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Path to save the results.",
            rich_help_panel="Output",
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            case_sensitive=False,
            help="Format of the output.",
            rich_help_panel="Output",
        ),
    ] = OutputFormat.TERMINAL,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Save/Report results on the fly.",
            rich_help_panel="Output",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed features extraction.",
            rich_help_panel="Output",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            envvar="RABBIT_DEBUG",
            help="Enable debug logging.",
            rich_help_panel="Output",
        ),
    ] = False,
):
    """Identify bot contributors based on their activity sequences in GitHub."""

    setup_logger(debug)
    logger = logging.getLogger("rabbit.cli")

    contributors = _get_all_contributors(contributors, input_file)
    if len(contributors) == 0:
        logger.error(
            "No contributors provided. Provide at least one contributor or an input file. (--help for more info)"
        )
        raise typer.Exit(code=1)
    if key is None:
        logger.warning("No API key provided. Rate limits will be low (60/hr).")

    renderer = RabbitRenderer(total_items=len(contributors), quiet=quiet)

    writer = DataWriter(fmt=output_format)

    try:
        rabbit_generator = run_rabbit(
            contributors=contributors,
            api_key=key,
            min_events=min_events,
            min_confidence=min_confidence,
            max_queries=max_queries,
            _verbose=verbose,
        )
        with Live(renderer.get_renderable(), console=console_ui) as live:
            for result in rabbit_generator:
                renderer.update(result)
                writer.write(result)

            if not renderer.quiet:
                live.update(renderer.table)
            else:
                live.stop()
            logger.info("RABBIT completed successfully.")
    except RetryableError as e:
        logger.error(f"API rate limit or network issue: {e}")
        logger.info("Please try again later or provide a GitHub API key.")
        raise typer.Exit(code=2)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=debug)
        raise typer.Exit(code=3)


if __name__ == "__main__":
    app()
