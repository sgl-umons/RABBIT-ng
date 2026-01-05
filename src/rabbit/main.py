import logging
from collections.abc import Iterator
from typing import Optional

from .predictor.models import Predictor, ONNXPredictor
from .sources import GitHubAPIExtractor
from .predictor import ContributorResult, predict_user_type
from .errors import RabbitErrors, NotFoundError


logger = logging.getLogger(__name__)


def _process_single_contributor(
    contributor: str,
    gh_api_client: GitHubAPIExtractor,
    predictor: Predictor,
    min_events: int,
    min_confidence: float,
) -> ContributorResult:
    """Process a single contributor to determine their type."""
    try:
        all_events = []
        result = ContributorResult(contributor, "Unknown")

        github_type = gh_api_client.query_user_type(contributor)
        if github_type != "User":
            logger.debug(
                f"Contributor {contributor} is of type {github_type}. Skipping prediction."
            )
            return ContributorResult(contributor, github_type, 1.0)

        for event_batch in gh_api_client.query_events(contributor):
            all_events.extend(event_batch)
            logger.debug(
                f"Fetched {len(event_batch)} events for contributor {contributor} (Total: {len(all_events)})"
            )
            if len(all_events) < min_events:
                continue

            result = predict_user_type(contributor, all_events, predictor)

            if (
                isinstance(result.confidence, float)
                and result.confidence >= min_confidence
            ):
                logger.debug(
                    f"Early stopping for {contributor} with confidence {result.confidence}"
                )
                break

        if len(all_events) < min_events:
            logger.debug(f"Not enough events for contributor {contributor}")
            return ContributorResult(contributor, "Unknown", "-")

        return result
    except NotFoundError as not_found_err:
        logger.error(not_found_err)
        return ContributorResult(contributor, "Invalid", "-")
    except RabbitErrors as _:
        raise
    except Exception as err:
        raise RabbitErrors(f"A critical error occurred: {str(err)}") from err


def run_rabbit(
    contributors: list[str],
    api_key: Optional[str] = None,
    min_events: int = 5,
    min_confidence: float = 1.0,
    max_queries: int = 3,
) -> Iterator[ContributorResult]:
    """
    Run rabbit on a list of contributors to determiner their type.

    This is the main entry point for using RABBIT as a library.
    It yields results incrementally, allowing to process contributors one at a time.

    Args:
        contributors: List of GitHub usernames to analyze.
        api_key: GitHub personal access token. If None, rate limits are lower (60/hr).
        min_events: Minimum number of events required to make a prediction.
        min_confidence: Confidence threshold (0.0-1.0). Stop querying once reached.
        max_queries: Maximum number of API queries per contributor (max 300 events).

    Yields:
        ContributorResult: The result for each contributor.

    Raises:
        RabbitErrors: For general API or prediction errors.
        RetryableError: For network-related issues that did not resolve after retries.
        RateLimitExceededError: If the API rate limit is exceeded (and reset time is not known. Typically, when no API key is given).

    Example:
        >>> for result in run_rabbit(["alice"], api_key="token"):
        >>>     print(f"{result.contributor}: {result.user_type} ({result.confidence})")
        alice: Human (0.95)
    """
    gh_api_client = GitHubAPIExtractor(api_key=api_key, max_queries=max_queries)

    try:
        predictor = ONNXPredictor()
        for contributor in contributors:
            result = _process_single_contributor(
                contributor, gh_api_client, predictor, min_events, min_confidence
            )
            yield result

    except RuntimeError as err:
        rabbit_errors = RabbitErrors(str(err))
        raise rabbit_errors from err
    except Exception as _:
        raise
