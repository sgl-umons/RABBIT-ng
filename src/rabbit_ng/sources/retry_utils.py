import logging
from ..errors import RetryableError

logger = logging.getLogger(__name__)


def retry(max_attempts: int = 3, delay: int = 10, backoff: float = 2.0):
    """
    Decorator to retry a function on network-related errors.

    Parameters:
        max_attempts: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier to increase the delay after each attempt.
    """

    def decorator(func):
        import time
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_error = None

            if max_attempts <= 0:
                return func(*args, **kwargs)

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except RetryableError as e:
                    if attempt < max_attempts - 1:
                        logger.info(f"{e} - Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        logger.info("Retrying...")
                        current_delay *= backoff
                    else:
                        last_error = e

            logger.error(
                f"Max attempts reached. Function failed with error: {last_error}"
            )
            raise last_error

        return wrapper

    return decorator
