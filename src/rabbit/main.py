from enum import Enum

import pandas as pd

from .sources import GitHubAPIExtractor
from .predictor import predict_user_type
from .errors import RabbitErrors, NotFoundError

from rich.progress import track


class OutputFormat(str, Enum):
    TERMINAL = "term"
    CSV = "csv"
    JSON = "json"


def _save_results(all_results, output_type: OutputFormat, save_path: str):
    """Save the result in the specified format and path."""

    if output_type == OutputFormat.CSV:
        all_results.to_csv(save_path, index=False)
    elif output_type == OutputFormat.JSON:
        all_results.to_json(save_path, orient="records", indent=4)
    else:  # Print to console
        print(all_results.to_string(index=False))


def _process_single_contributor(
    contributor: str,
    gh_api_client: GitHubAPIExtractor,
    min_events: int,
):
    """Process a single contributor to determine their type."""
    try:
        events = gh_api_client.query_events(contributor)
        if len(events) < min_events:
            return {
                "contributor": contributor,
                "type": "Unknown",
                "confidence": "-",
            }
        else:
            user_type, confidence = predict_user_type(contributor, events)
            return {
                "contributor": contributor,
                "type": user_type,
                "confidence": confidence,
            }
    except NotFoundError as not_found_err:
        print(not_found_err)
        return {
            "contributor": contributor,
            "type": "Invalid",
            "confidence": "-",
        }
    except RabbitErrors as err:
        raise err from err
    except Exception as err:
        raise RabbitErrors(f"A critical error occurred: {str(err)}") from err


def run_rabbit(
    contributors: list[str],
    api_key: str = None,
    min_events: int = 5,
    min_confidence: float = 1.0,
    max_queries: int = 3,
    output_type: OutputFormat = OutputFormat.TERMINAL,
    output_path: str = "",
    _verbose: bool = False,
    incremental: bool = False,
):
    """
    Orchestrates the RABBIT bot identification process for GitHub contributors.

    Args:
        contributors (list[str]): A list of GitHub contributor login names to analyze.
        api_key (str, optional): GitHub API key for authentication. Defaults to None.
        min_events (int, optional): Minimum number of events required to analyze a contributor. Defaults to 5.
        min_confidence (float, optional): minimum confidence on type of contributor to stop further querying. Defaults to 1.0.
        max_queries (int, optional): Maximum number of API queries allowed per contributor. Defaults to 3.
        output_type (str, optional): Format for saving results ('text', 'csv', or 'json'). Defaults to 'text'.
        output_path (str, optional): Path to save the output file. Defaults to an empty string.
        _verbose (bool, optional): If True, displays the features that were used to determine the type of contributor. Defaults to False.
        incremental (bool, optional): Update the output file/print on terminal once the type is determined for new contributors. If False, results will be accessible only after the type is determined for all the contributors Defaults to False.

    Returns:
        None

    Description:
        Processes each contributor by:
        1. Querying GitHub Events API to fetch events.
        2. Computing activity sequences from the events. (using ghmap)
        3. Extracting features from the activity sequences.
        4. Predicting whether the contributor is a bot or human based on the features.
        5. Saving the results in the specified format and location.
    """

    gh_api_client = GitHubAPIExtractor(api_key=api_key, max_queries=max_queries)
    all_results = pd.DataFrame()

    try:
        for contributor in track(
            contributors, description="Processing contributors..."
        ):
            result = _process_single_contributor(contributor, gh_api_client, min_events)
            all_results = pd.concat(
                [all_results, pd.DataFrame([result])], ignore_index=True
            )

            if incremental:
                _save_results(all_results, output_type, output_path)

    except Exception as e:
        # TODO: Maybe just let the exception propagate and handle it at a higher level
        print(e)

    if not incremental:
        _save_results(all_results, output_type, output_path)
