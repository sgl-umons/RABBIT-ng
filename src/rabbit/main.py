import logging
from typing import Generator


from .predictor.models import Predictor, ONNXPredictor
from .sources import GitHubAPIExtractor
from .predictor import predict_user_type
from .errors import RabbitErrors, NotFoundError


logger = logging.getLogger(__name__)


def _process_single_contributor(
    contributor: str,
    gh_api_client: GitHubAPIExtractor,
    predictor: Predictor,
    min_events: int,
) -> dict[str, str | float]:
    """Process a single contributor to determine their type."""
    try:
        events = gh_api_client.query_events(contributor)
        logger.debug(f"Found {len(events)} events for contributor {contributor}")
        if len(events) < min_events:
            logger.debug(f"Not enough events for contributor {contributor}")
            return {
                "contributor": contributor,
                "type": "Unknown",
                "confidence": "-",
            }
        else:
            user_type, confidence = predict_user_type(contributor, events, predictor)
            return {
                "contributor": contributor,
                "type": user_type,
                "confidence": confidence,
            }
    except NotFoundError as not_found_err:
        logger.error(not_found_err)
        return {
            "contributor": contributor,
            "type": "Invalid",
            "confidence": "-",
        }
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
) -> Generator[dict[str, float | str]]:
    gh_api_client = GitHubAPIExtractor(api_key=api_key, max_queries=max_queries)

    try:
        predictor = ONNXPredictor()
        for contributor in contributors:
            result = _process_single_contributor(
                contributor, gh_api_client, predictor, min_events
            )
            yield result

    except RuntimeError as err:
        rabbit_errors = RabbitErrors(str(err))
        raise rabbit_errors from err
    except Exception as _:
        raise
