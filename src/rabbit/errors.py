"""Custom errors for API operations."""

from requests import Response


class RabbitErrors(Exception):
    """Base error class for API related issues"""

    def __init__(self, message: str = "An error occurred with the GitHub API"):
        super().__init__(message)

    def __str__(self):
        return f"[{self.__class__.__name__}] {self.args[0] if self.args else ''}"


class RateLimitExceededError(RabbitErrors):
    """Error raised when the API rate limit is exceeded."""

    def __init__(self, reset_time: str):
        self.reset_time = reset_time
        super().__init__(f"API rate limit exceeded. Reset at {reset_time}.")

    def wait_reset(self):
        """Wait until the reset time for rate limiting."""
        from datetime import datetime
        import time

        reset_time = datetime.strptime(self.reset_time, "%Y-%m-%d %H:%M:%S")
        time_diff = (reset_time - datetime.now()).total_seconds()
        if time_diff > 0:
            print(
                f"Waiting for {time_diff} seconds until rate limit reset. "
                f"Reset at {reset_time}."
            )
            time.sleep(time_diff)


class NotFoundError(RabbitErrors):
    """Error raised when a requested resource is not found."""

    def __init__(self, contributor: str):
        self.resource = contributor
        super().__init__(f"Contributor not found: {contributor}.")


class RetryableError(RabbitErrors):
    """Error indicating that the operation can be retried."""

    def __init__(self, error_message: str):
        super().__init__(error_message)


class APIRequestError(RabbitErrors):
    """Error raised for general API request failures."""

    def __init__(self, response: Response, message: str = "API request failed"):
        super().__init__(
            f"{message}. Status code: {response.status_code}. Reason: {response.reason}"
        )
