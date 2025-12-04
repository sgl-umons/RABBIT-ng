from pathlib import Path

import typer

from . import run_rabbit

app = typer.Typer(
    help="RABBIT is an Activity Based Bot Identification Tool that identifies bots "
    "based on their recent activities in GitHub"
)


@app.command()
def cli(
    contributors: list[str] = typer.Argument(
        [], help="Login names of contributors to analyze"
    ),
    input_file: Path = typer.Option(
        None,
        "--input-file",
        help="Path to .txt file containing login names (one per line)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Report events, activities, and features used for determination",
    ),
    min_events: int = typer.Option(
        5,
        "--min-events",
        min=1,
        max=300,
        help="Minimum number of events required to determine contributor type",
    ),
    min_confidence: float = typer.Option(
        1.0,
        "--min-confidence",
        min=0.0,
        max=1.0,
        help="Minimum confidence threshold to stop further querying",
    ),
    max_queries: int = typer.Option(
        3,
        "--max-queries",
        min=1,
        max=3,
        help="Maximum number of queries to GitHub Events API per contributor",
    ),
    key: str = typer.Option(
        None, "--key", help="GitHub API key (required for >60 queries/hour)"
    ),
    csv: Path = typer.Option(
        None, "--csv", help="Save results in CSV format to specified file"
    ),
    json: Path = typer.Option(
        None, "--json", help="Save results in JSON format to specified file"
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Report results incrementally instead of all at once",
    ),
):
    """Identify bot contributors based on their activity sequences in GitHub."""

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

    # Determine output type
    if csv is not None:
        output_type = "csv"
        output_path = str(csv)
    elif json is not None:
        output_type = "json"
        output_path = str(json)
    else:
        output_type = "text"
        output_path = ""

    if input_file is not None:
        # Read txt file and extract contributors
        with open(input_file, "r") as f:
            file_contributors = [line.strip() for line in f if line.strip()]
        contributors.extend(file_contributors)

    run_rabbit(
        contributors=contributors,
        api_key=key,
        min_events=min_events,
        min_confidence=min_confidence,
        max_queries=max_queries,
        output_type=output_type,
        output_path=output_path,
        _verbose=verbose,
        incremental=incremental,
    )


if __name__ == "__main__":
    app()
