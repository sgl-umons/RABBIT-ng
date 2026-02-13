from datetime import datetime, timedelta
from collections.abc import Iterator

import requests

from ..errors import (
    APIRequestError,
    NotFoundError,
    RateLimitExceededError,
    RetryableError,
)
from .retry_utils import retry

import logging

logger = logging.getLogger(__name__)


class GitHubAPIExtractor:
    """
    Extract GitHub event data for contributors via the GitHub REST API.

    This class handles pagination, rate limiting, and retries for fetching
    user events. It yields events incrementally to support early stopping
    based on prediction confidence.

    Args:
        api_key: GitHub personal access token. If None, rate limits are
            lower (60 requests/hour vs 5000/hour).
        max_queries: Maximum number of API pages to fetch per contributor.
            Each page contains up to 100 events.
        no_wait: If True, do not wait for rate limit reset, raise error instead.

    Attributes:
        api_key: The GitHub API token for authenticated requests.
        max_queries: Maximum number of pages to query.
        no_wait: Whether to wait for rate limit reset or raise error.
        query_root: Base URL for GitHub API (https://api.github.com).

    Example:
        >>> extractor = GitHubAPIExtractor(api_key="token", max_queries=3)
        >>> for events in extractor.query_events("alice"):
        ...     print(f"Fetched {len(events)} events")
        Fetched 100 events
        Fetched 100 events
    """

    def __init__(self, api_key=None, max_queries=3, no_wait=False):
        self.api_key = api_key
        self.max_queries = max_queries
        self.no_wait = no_wait

        self.query_root = "https://api.github.com"

    @staticmethod
    def _check_events_left(events):
        """Return True if there might be more events to fetch."""
        return len(events) == 100

    def _handle_api_response(self, contributor, response):
        match response.status_code:
            case 200:  # OK
                logger.debug(
                    f"GitHub API Rate Limit Remaining: {response.headers.get('x-ratelimit-remaining')}"
                )
                return response.json()

            case (
                403 | 429
            ):  # Forbidden or Too Many Request (Following GitHub best practices)
                if response.headers.get("retry-after"):
                    retry_after = int(response.headers.get("retry-after"))
                    reset_time = (
                        datetime.now() + timedelta(seconds=retry_after)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    raise RateLimitExceededError(reset_time)
                if response.headers.get("x-ratelimit-remaining") == 0:
                    reset_time = response.headers.get("x-ratelimit-reset")
                    reset_time = datetime.fromtimestamp(int(reset_time)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    raise RateLimitExceededError(reset_time)
                if (
                    not self.api_key
                    and response.reason
                    and "rate limit" in response.reason.lower()
                ):
                    # If no API key is provided, we cannot determine reset time
                    raise RateLimitExceededError(reset_time=None)
                raise RetryableError(response.reason)

            case 404:  # Not Found
                raise NotFoundError(contributor)

            case 500 | 504 | 408:  # Timeout or server errors
                raise RetryableError(response.reason)

            case _:
                raise APIRequestError(response, f"Error while querying {contributor}.")

    @retry(max_attempts=3, delay=10, backoff=2.5)
    def _query_event_page(self, contributor, page):
        """Fetch a single page of GitHub events for a contributor."""
        query = f"{self.query_root}/users/{contributor}/events"
        response = requests.get(
            query,
            headers={"Authorization": f"token {self.api_key}"} if self.api_key else {},
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        return self._handle_api_response(contributor, response)

    @retry(max_attempts=3, delay=10, backoff=2.5)
    def query_user_type(self, contributor: str) -> str:
        """
        Fetch GitHub user data for a contributor to determine their type.

        This method queries the GitHub /users/{contributor} endpoint to retrieve
        the type of the contributor according to GitHub (Bot, User or Organization).

        Args:
            contributor: GitHub username to query.

        Returns:
            The type of the contributor ("Bot", "User", "Organization").
        """
        query = f"{self.query_root}/users/{contributor}"
        response = requests.get(
            query,
            headers={"Authorization": f"token {self.api_key}"} if self.api_key else {},
            timeout=30,
        )
        try:
            user_data = self._handle_api_response(contributor, response)
            return user_data.get("type", "Unknown")
        except RateLimitExceededError as rate_limit_e:
            if self.no_wait or not rate_limit_e.reset_time:
                raise
            rate_limit_e.wait_reset()
            return self.query_user_type(contributor)

    def query_events(self, contributor: str) -> Iterator[list[dict]]:
        """
        Fetch GitHub events for a contributor, yielding page-by-page.

        This method queries the GitHub API incrementally, allowing callers to
        stop early once sufficient data is collected. Rate limit errors are
        handled automatically by waiting until the limit resets.
        (Or raising an error if `no_wait` is True).

        The method stops querying automatically when either:
        - The maximum number of queries (`max_queries`) is reached.
        - A page returns fewer than 100 events, indicating no more events are available.

        Be aware that GitHub API returns a maximum of 300 events (3 pages of 100)

        Args:
            contributor: GitHub username to query.

        Yields:
            Lists of event dictionaries. Each list contains up to 100 events.

        Raises:
            NotFoundError: If the contributor does not exist.
            RetryableError: If network errors persist after retries.
            RateLimitExceededError: If rate limit is hit and reset time is
                unknown (typically when no API key is provided) or when `no_wait` is True.

        Example:
            >>> extractor = GitHubAPIExtractor(api_key="token", max_queries=3)
            >>> for events in extractor.query_events("alice"):
            ...     print(f"Fetched {len(events)} events")
            Fetched 100 events
        """
        page = 1
        while page <= self.max_queries:
            try:
                page_events = self._query_event_page(contributor, page)

                yield page_events

                if not self._check_events_left(page_events):
                    break
                page += 1
            except RateLimitExceededError as rate_limit_e:
                if self.no_wait or not rate_limit_e.reset_time:
                    raise
                rate_limit_e.wait_reset()
