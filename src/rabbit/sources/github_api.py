from datetime import datetime, timedelta
from typing import Generator

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
    def __init__(self, api_key=None, max_queries=3):
        self.api_key = api_key
        self.max_queries = max_queries

        self.query_root = "https://api.github.com"

    @staticmethod
    def _check_events_left(events):
        return len(events) == 100

    @staticmethod
    def _handle_api_response(contributor, response):
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
                raise RetryableError(response.reason)

            case 404:  # Not Found
                raise NotFoundError(contributor)

            case 500 | 504 | 408:  # Timeout or server errors
                raise RetryableError(response.reason)

            case _:
                raise APIRequestError(response, f"Error while querying {contributor}.")

    @retry(max_attempts=3, delay=10, backoff=2.5)
    def _query_event_page(self, contributor, page):
        query = f"{self.query_root}/users/{contributor}/events"
        response = requests.get(
            query,
            headers={"Authorization": f"token {self.api_key}"} if self.api_key else {},
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        return self._handle_api_response(contributor, response)

    def query_events(self, contributor: str) -> Generator[list[dict]]:
        page = 1
        while page <= self.max_queries:
            try:
                page_events = self._query_event_page(contributor, page)

                yield page_events

                if not self._check_events_left(page_events):
                    break
                page += 1
            except RateLimitExceededError as rate_limit_e:
                rate_limit_e.wait_reset()