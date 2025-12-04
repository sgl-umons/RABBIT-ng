import importlib.metadata

__version__ = importlib.metadata.version("rabbit")

from .main import run_rabbit
from .errors import RabbitErrors, APIRequestError, RetryableError
