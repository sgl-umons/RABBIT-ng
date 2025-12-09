import importlib.metadata
import logging

from .main import run_rabbit
from .errors import RabbitErrors, APIRequestError, RetryableError

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = importlib.metadata.version("rabbit")
