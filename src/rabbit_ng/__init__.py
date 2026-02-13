import importlib.metadata
import logging

from .main import run_rabbit
from .errors import (
    RabbitErrors,
    APIRequestError,
    RetryableError,
    RateLimitExceededError,
    NotFoundError,
)
from .predictor import ContributorResult

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = importlib.metadata.version("rabbit_ng")


__all__ = [
    "run_rabbit",
    "ContributorResult",
    "RabbitErrors",
    "APIRequestError",
    "RetryableError",
    "RateLimitExceededError",
    "NotFoundError",
    "__version__",
]
