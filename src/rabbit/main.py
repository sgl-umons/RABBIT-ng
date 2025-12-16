import logging
from collections.abc import Iterator


from .predictor.models import Predictor, ONNXPredictor
from .sources import GitHubAPIExtractor
from .predictor import predict_user_type, ContributorResult
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
    api_key: str = None,
    min_events: int = 5,
    min_confidence: float = 1.0,
    max_queries: int = 3,
    _verbose: bool = False,
) -> Iterator[ContributorResult]:
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
