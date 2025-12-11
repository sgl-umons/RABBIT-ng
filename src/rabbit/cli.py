import csv
import io
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from dotenv import load_dotenv

import typer
import logging

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.theme import Theme

from . import run_rabbit, RetryableError

load_dotenv()

custom_theme = Theme(
    {
        "header": "bold underline",
        "login": "bold cyan",
        "Bot": "red",
        "Human": "green",
        "Unknown": "dim",
        "Invalid": "dim yellow",
    }
)

console_err = Console(
    stderr=True, no_color="NO_COLOR" in os.environ, theme=custom_theme
)

app = typer.Typer(
    help="RABBIT is an Activity Based Bot Identification Tool that identifies bots "
    "based on their recent activities in GitHub",
    add_completion=False,
)


class OutputFormat(str, Enum):
    TERMINAL = "term"
    CSV = "csv"


def setup_logger(verbose: int):
    levels = [
        logging.CRITICAL,  # 0 - default
        logging.INFO,  # 1 - -v
        logging.DEBUG,  # 2 - -vv
    ]

    log_level = levels[verbose]
    logging.basicConfig(
        level=log_level,
        format="%(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            RichHandler(console=console_err, rich_tracebacks=True, show_path=False)
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


class RabbitUI:
    """Manages incremental display with progress bar for CLI output."""

    COLUMN_WIDTHS = {"login": 30, "type": 10, "confidence": 10}

    def __init__(self, total: int, fmt: OutputFormat):
        self.fmt = fmt
        self.total = total
        self._is_interactive = sys.stdout.isatty()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console_err,
            transient=True,
        )
        self._task_id = self._progress.add_task("Analyzing...", total=self.total)

    def __enter__(self):
        self._progress.start()
        self._print_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._progress.stop()

    def advance(self):
        """Advance the progress bar by one step."""
        self._progress.advance(self._task_id)

    def print_row(self, result: dict):
        """Print a single result row (CSV or terminal format)."""
        content = (
            self._format_csv_row(result)
            if self.fmt == OutputFormat.CSV
            else self._format_terminal_row(result)
        )
        self._output(content)

    def _print_header(self):
        """Print the header row."""
        header = (
            "contributor,type,confidence"
            if self.fmt == OutputFormat.CSV
            else self._build_terminal_header()
        )
        self._output(header)

    def _format_csv_row(self, result: dict) -> str:
        """Format a result as a CSV row."""
        output = io.StringIO()
        csv.writer(output).writerow(
            [result["contributor"], result["type"], result["confidence"]]
        )
        return output.getvalue().strip()

    def _format_terminal_row(self, result: dict) -> Text:
        """Format a result as a rich terminal row."""
        login = result["contributor"]
        rtype = result["type"]
        confidence = result["confidence"]

        # Truncate login if too long
        w_login = self.COLUMN_WIDTHS["login"]
        display_login = f"{login[: w_login - 1]}…" if len(login) > w_login else login

        text = Text()
        text.append(f"{display_login:<{w_login}}", style="login")
        text.append("  ")
        text.append(f"{rtype:<{self.COLUMN_WIDTHS['type']}}", style=rtype)
        text.append("  ")
        text.append(f"{confidence:>{self.COLUMN_WIDTHS['confidence']}}")

        return text

    def _build_terminal_header(self) -> Text:
        """Build the terminal header row."""
        text = Text()
        text.append(f"{'CONTRIBUTOR':<{self.COLUMN_WIDTHS['login']}}", style="header")
        text.append("  ")
        text.append(f"{'TYPE':<{self.COLUMN_WIDTHS['type']}}", style="header")
        text.append("  ")
        text.append(
            f"{'CONFIDENCE':>{self.COLUMN_WIDTHS['confidence']}}", style="header"
        )
        return text

    def _output(self, content):
        """Write content to appropriate output stream."""
        if self._is_interactive:
            self._progress.console.print(content)
        else:
            print(content)
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
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity level (can be used multiple times. e.g -vv).",
            rich_help_panel="Output",
        ),
    ] = 0,
):
    """Identify bot contributors based on their activity sequences in GitHub."""

    setup_logger(verbose)
    logger = logging.getLogger("rabbit.cli")

    contributors = _get_all_contributors(contributors, input_file)
    if len(contributors) == 0:
        logger.error(
            "No contributors provided. Provide at least one contributor or an input file. (--help for more info)"
        )
        raise typer.Exit(code=1)
    if key is None:
        logger.warning("No API key provided. Rate limits will be low (60/hr).")

    try:
        with RabbitUI(len(contributors), output_format) as ui:
            for result in run_rabbit(
                contributors=contributors,
                api_key=key,
                min_events=min_events,
                min_confidence=min_confidence,
                max_queries=max_queries,
            ):
                ui.print_row(result)

                ui.advance()

    except RetryableError as e:
        logger.error(f"API rate limit or network issue: {e}")
        logger.info("Please try again later or provide a GitHub API key.")
        raise typer.Exit(code=2)
    except Exception as e:
        logger.critical(
            f"Unexpected error: {e}", exc_info=True if verbose > 0 else False
        )
        raise typer.Exit(code=3)


if __name__ == "__main__":
    app()
