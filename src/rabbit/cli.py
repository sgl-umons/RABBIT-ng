from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from rabbit.main import OutputFormat
from . import run_rabbit

app = typer.Typer(
    help="RABBIT is an Activity Based Bot Identification Tool that identifies bots "
    "based on their recent activities in GitHub"
)


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
            help="Path to .txt file containing login names (one per line).",
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
            help="GitHub API key (required for >60 queries/hour)",
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
    incremental: Annotated[
        bool,
        typer.Option(
            "--incremental",
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
):
    """Identify bot contributors based on their activity sequences in GitHub."""
    if contributors is None:
        contributors = []

    if input_file is None and len(contributors) == 0:
        typer.echo(
            "Error: Provide either contributor names or an input file.", err=True
        )
        raise typer.Exit(code=1)

    if key is None or len(key) < 40:
        typer.echo(
            "Warning: A valid GitHub API key is recommended for higher rate limits.",
            err=True,
        )

    if input_file is not None:
        # Read txt file and extract contributors
        with open(input_file, "r") as f:
            file_contributors = [line.strip() for line in f if line.strip()]
        contributors.extend(file_contributors)

    if format is OutputFormat.CSV or format is OutputFormat.JSON:
        if output_path is None:
            typer.echo(
                f"Error: Output path must be specified for {format.value} format.",
                err=True,
            )
            raise typer.Exit(code=1)

    run_rabbit(
        contributors=contributors,
        api_key=key,
        min_events=min_events,
        min_confidence=min_confidence,
        max_queries=max_queries,
        output_type=output_format,
        output_path=str(output_path),
        _verbose=verbose,
        incremental=incremental,
    )


if __name__ == "__main__":
    app()
