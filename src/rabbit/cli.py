import csv
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from dotenv import load_dotenv

import typer
import logging

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.theme import Theme

from .predictor import ContributorResult, FEATURE_NAMES
from . import run_rabbit, RetryableError

load_dotenv()

custom_theme = Theme(
    {
        "header": "bold underline",
        "login": "bold cyan",
        "Bot": "red",
        "Organization": "yellow",
        "Human": "green",
        "Unknown": "dim",
        "Invalid": "dim yellow",
    }
)

console_err = Console(
    stderr=True, no_color="NO_COLOR" in os.environ, theme=custom_theme
)

app = typer.Typer(
    help="RABBIT is an Activity Based Bot Identification Tool that identifies bots.",
    add_completion=False,
)


class OutputFormat(str, Enum):
    TEXT = "text"
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
            RichHandler(console=console_err, rich_tracebacks=True, show_path=False),
        ],
    )

    # Use only warning logs for urllib3 to reduce verbosity
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _concat_all_contributors(
    arg_contributors: list[str] | None, input_file: Path | None
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

    return list(dict.fromkeys(contributors))


class RabbitUI:
    """Manages incremental display with progress bar for CLI output."""

    COLUMN_WIDTHS = {"login": 30, "type": 12, "confidence": 10}

    def __init__(self, total: int, fmt: OutputFormat, display_features: bool = False):
        self.fmt = fmt
        self.total = total
        self.display_features = display_features
        self._is_interactive = sys.stdout.isatty()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console_err,
            transient=True,
            redirect_stdout=False,  # Avoid capturing print statements
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

    def print_row(self, result: ContributorResult):
        """Print a single result row (CSV or terminal format)."""
        content = (
            self._format_csv_row(result)
            if self.fmt == OutputFormat.CSV
            else self._format_terminal_row(result)
        )
        self._output(content)

    def _print_header(self):
        """Print the header row."""
        if self.fmt == OutputFormat.CSV:
            header = "contributor,type,confidence"
            if self.display_features:
                header += "," + ",".join(FEATURE_NAMES)
        else:
            header = self._build_terminal_header()
        self._output(header)

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

        if self.display_features:
            for feature_name in FEATURE_NAMES:
                text.append(" ")
                text.append(f"{feature_name.upper()}", style="header")

        return text

    def _format_csv_row(self, result: ContributorResult) -> str:
        """Format a result as a CSV row."""
        import io

        output = io.StringIO()

        row = [result.contributor, result.user_type, result.confidence]
        if self.display_features:
            # Append feature values in the order of FEATURE_NAMES
            feature_values = [result.features.get(name, "") for name in FEATURE_NAMES]
            row.extend(feature_values)

        csv.writer(output).writerow(row)
        return output.getvalue().strip()

    def _format_terminal_row(self, result: ContributorResult) -> Text:
        """Format a result as a rich terminal row."""
        login = result.contributor
        rtype = result.user_type
        confidence = result.confidence

        # Truncate login if too long
        w_login = self.COLUMN_WIDTHS["login"]
        display_login = f"{login[: w_login - 1]}…" if len(login) > w_login else login

        text = Text()
        text.append(f"{display_login:<{w_login}}", style="login")
        text.append("  ")
        text.append(f"{rtype:<{self.COLUMN_WIDTHS['type']}}", style=rtype)
        text.append("  ")
        text.append(f"{confidence:>{self.COLUMN_WIDTHS['confidence']}}")

        if self.display_features:
            for feature_name in FEATURE_NAMES:
                feature_value = result.features.get(feature_name, "-")
                text.append(" ")
                text.append(f"{feature_value}")

        return text

    def _output(self, content):
        """Write content to appropriate output stream."""
        if self._is_interactive:
            self._progress.console.print(content)
        else:
            print(content, flush=True)


@app.command()
def cli(
    # ---- INPUTS ----
    contributors: Annotated[
        list[str] | None,
        typer.Argument(
            help="Login names of contributors to analyze. (Ex: 'user1 user2 ...')",
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
    no_wait: Annotated[
        bool,
        typer.Option(
            "--no-wait",
            help="Do not wait when rate limit is reached; exit immediately.",
            rich_help_panel="Configuration",
        ),
    ] = False,
    # ---- OUTPUTS ----
    display_features: Annotated[
        bool,
        typer.Option(
            "--features",
            help="Display computed features for each contributor.",
            rich_help_panel="Output",
        ),
    ] = False,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            case_sensitive=False,
            help="Format of the output.",
            rich_help_panel="Output",
        ),
    ] = OutputFormat.TEXT,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity level (can be used multiple times. -v or -vv).",
            rich_help_panel="Output",
        ),
    ] = 0,
):
    """
    RABBIT is an Activity Based Bot Identification Tool that identifies bots
    based on their recent activities in GitHub.

    The simplest way to use RABBIT is to provide a list of GitHub usernames (e.g. rabbit user1 user2 ...)
    """
    setup_logger(verbose)
    logger = logging.getLogger("rabbit.cli")

    contributors = _concat_all_contributors(contributors, input_file)
    if len(contributors) == 0:
        logger.critical(
            "No contributors provided. Provide at least one contributor or an input file. (--help for more info)"
        )
        raise typer.Exit(code=1)
    if key is None:
        logger.warning("No API key provided. Rate limits will be low (60/hr).")

    try:
        with RabbitUI(len(contributors), output_format, display_features) as ui:
            for result in run_rabbit(
                contributors=contributors,
                api_key=key,
                min_events=min_events,
                min_confidence=min_confidence,
                max_queries=max_queries,
                no_wait=no_wait,
            ):
                ui.print_row(result)

                ui.advance()

    except RetryableError as e:
        logger.error(f"Network issue occurred: {e}")
    except Exception as e:
        logger.critical(
            f"Unexpected error: {e}", exc_info=True if verbose > 0 else False
        )


if __name__ == "__main__":
    app()
