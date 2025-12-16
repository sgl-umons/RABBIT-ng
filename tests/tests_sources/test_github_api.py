import itertools
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from rabbit.sources import GitHubAPIExtractor
from rabbit.errors import RetryableError


class TestGitHubAPIExtractor:
    @pytest.fixture
    def extractor(self):
        """Create a basic extractor instance for testing."""
        return GitHubAPIExtractor(api_key="test_api_key", max_queries=3)

    def test_check_events_left_true(self):
        """Test _check_events_left returns True when 100 events."""
        events = [{"id": i} for i in range(100)]
        assert GitHubAPIExtractor._check_events_left(events) is True

    def test_check_events_left_false(self):
        """Test _check_events_left returns False when less than 100 events."""
        events = [{"id": i} for i in range(50)]
        assert GitHubAPIExtractor._check_events_left(events) is False

    @patch("rabbit.sources.github_api.requests.get")
    def test_query_events_without_api_key(self, mock_get):
        """Test if _query_event_page works without API key."""
        extractor_no_key = GitHubAPIExtractor(api_key=None)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": 1}]
        mock_get.return_value = mock_response

        next(extractor_no_key.query_events("testuser"))

        _, kwargs = mock_get.call_args
        assert kwargs["headers"] == {}

    @patch("rabbit.sources.github_api.requests.get")
    def test_query_events_single_page(self, mock_get, extractor):
        """Test if _query_event_page handles request parameters correctly."""
        test_user = "testuser"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": i} for i in range(50)]
        mock_get.return_value = mock_response

        events = list(itertools.chain.from_iterable(extractor.query_events(test_user)))

        # Check response is as expected
        assert len(events) == 50
        assert events == [{"id": i} for i in range(50)]

        # Check that the request was made with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert test_user in args[0]
        assert kwargs["params"]["page"] == 1
        assert kwargs["params"]["per_page"] == 100

    @patch("rabbit.sources.github_api.requests.get")
    def test_query_events_all_pages(self, mock_get, extractor):
        """Test if query_events handles multiple pages correctly."""
        test_user = "testuser"

        # Mock responses for 2 pages
        mock_responses = []
        for i in range(2):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [{"id": j + i * 100} for j in range(100)]
            mock_responses.append(mock_response)

        # Last page with less than 100 events
        last_response = Mock()
        last_response.status_code = 200
        last_response.json.return_value = [{"id": j + 200} for j in range(100)]
        mock_responses.append(last_response)

        mock_get.side_effect = mock_responses

        events = list(itertools.chain.from_iterable(extractor.query_events(test_user)))

        # Check response is as expected
        assert len(events) == 300
        assert events[0]["id"] == 0
        assert events[-1]["id"] == 299

        # Check that the request was made correct number of times
        assert mock_get.call_count == 3


class TestGitHubAPIExtractorAPIResponses(TestGitHubAPIExtractor):
    @pytest.fixture
    def mock_success(self):
        mock_success = Mock()
        mock_success.status_code = 200
        mock_success.json.return_value = [{"id": i} for i in range(50)]
        return mock_success

    @patch("rabbit.sources.github_api.requests.get")
    def test_handle_404_not_found(self, mock_get, extractor):
        """Test if query_events() raises NotFoundError on 404."""
        from rabbit.errors import NotFoundError

        test_user = "nonexistentuser"

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_get.return_value = mock_response

        with pytest.raises(NotFoundError) as exc_info:
            next(extractor.query_events(test_user))

        assert test_user in str(exc_info.value)

    @patch("rabbit.sources.github_api.requests.get")
    @patch("time.sleep")
    def test_handle_429_rate_limit_exceeded(
        self, mock_sleep, mock_get, extractor, mock_success
    ):
        """Test if query_events() raises RateLimitExceededError on rate limit exceeded."""

        future_time = int((datetime.now() + timedelta(hours=1)).timestamp())
        mock_fail = Mock()
        mock_fail.status_code = 429
        mock_fail.headers.get = (
            lambda key: 0
            if key == "x-ratelimit-remaining"
            else (str(future_time) if key == "x-ratelimit-reset" else None)
        )

        mock_get.side_effect = [mock_fail, mock_success]

        result = next(extractor.query_events("testuser"))

        assert len(result) == 50
        assert mock_sleep.call_count == 1
        assert mock_get.call_count == 2

    @patch("rabbit.sources.github_api.requests.get")
    @patch("time.sleep")
    def test_handle_403_rate_limit_with_retry_after(
        self, mock_sleep, mock_get, extractor, mock_success
    ):
        """Test if query_events() raises RateLimitExceededError on 403 with retry-after."""
        mock_fail = Mock()
        mock_fail.status_code = 403
        mock_fail.headers.get = lambda key: "10" if key == "retry-after" else None

        mock_get.side_effect = [mock_fail, mock_success]

        result = next(extractor.query_events("testuser"))

        assert len(result) == 50
        assert mock_sleep.call_count == 1
        assert mock_get.call_count == 2

    @pytest.mark.parametrize(
        "status_code,reason,error_type",
        [
            (500, "Internal Server Error", RetryableError),
            (504, "Gateway Timeout", RetryableError),
            (408, "Request Timeout", RetryableError),
            (429, "Too Many Requests", RetryableError),
        ],
    )
    @patch("rabbit.sources.github_api.requests.get")
    @patch("time.sleep")
    def test_handle_retryable_errors(
        self, mock_sleep, mock_get, extractor, status_code, reason, error_type
    ):
        """Test all retryable errors (500/504/408) raise RetryableError."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.reason = reason
        mock_response.headers.get = lambda key: None
        mock_get.return_value = mock_response

        with pytest.raises(error_type) as exc_info:
            extractor._query_event_page("testuser", 1)

        assert reason in str(exc_info.value)
        assert mock_get.call_count == 3  # retries
        assert mock_sleep.call_count == 2  # sleeps between retries

    @patch("rabbit.sources.github_api.requests.get")
    def test_raises_api_request_error_on_unknown_status(self, mock_get, extractor):
        """Test if APIRequestError is raised on unknown status codes."""
        from rabbit.errors import APIRequestError

        mock_response = Mock()
        mock_response.status_code = 418  # I'm a teapot
        mock_response.reason = "I'm a teapot"
        mock_get.return_value = mock_response

        with pytest.raises(APIRequestError) as exc_info:
            next(extractor.query_events("testuser"))

        assert "I'm a teapot" in str(exc_info.value)
