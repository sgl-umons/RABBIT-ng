from unittest.mock import patch

import pytest

from rabbit.main import (
    _process_single_contributor,
    run_rabbit,
)
from rabbit.errors import RabbitErrors


class TestProcessSingleContributor:
    """Tests for the _process_single_contributor function."""

    @patch("rabbit.main.ONNXPredictor")
    @patch("rabbit.main.GitHubAPIExtractor")
    @patch("rabbit.main.predict_user_type")
    def test_process_contributor_success(
        self, mock_predict, mock_gh_extractor, mock_predictor
    ):
        """Test _process_single_contributor returns correct type and confidence."""

        mock_gh_extractor.query_events.return_value = [[{"event": "test"}] * 10]  # Simulate 10 events
        mock_predict.return_value = ("Human", 0.95)

        result = _process_single_contributor(
            "testuser",
            mock_gh_extractor,
            predictor=mock_predictor,
            min_events=5,
            min_confidence=1,
        )

        assert result["contributor"] == "testuser"
        assert result["type"] == "Human"
        assert result["confidence"] == 0.95

    @patch("rabbit.main.ONNXPredictor")
    @patch("rabbit.main.GitHubAPIExtractor")
    @patch("rabbit.main.predict_user_type")
    def test_process_contributor_with_more_than_100_event_minimum(
            self, mock_predict, mock_gh_extractor, mock_predictor
    ):
        """Users can set the min_events parameter to more than 100, which requires multiple API calls."""
        # Simulate 250 events returned in 3 pages
        mock_gh_extractor.query_events.return_value = [
            [{"event": "test"}] * 100,
            [{"event": "test"}] * 100,
            [{"event": "test"}] * 50,
        ]
        mock_predict.return_value = ("Bot", 0.9)

        result = _process_single_contributor(
            "testuser",
            mock_gh_extractor,
            predictor=mock_predictor,
            min_events=200,
            min_confidence=1,
        )

        assert result["contributor"] == "testuser"
        assert result["type"] == "Bot"
        assert result["confidence"] == 0.90

    @patch("rabbit.main.ONNXPredictor")
    @patch("rabbit.main.GitHubAPIExtractor")
    def test_process_contributor_returns_unknown_when_no_events(
        self, mock_gh_extractor, mock_predictor
    ):
        """Test _process_single_contributor returns 'Unknown' when events are below min_events."""

        mock_gh_extractor.query_events.return_value = [
            {}
        ]  # Simulate less than min_events

        result = _process_single_contributor(
            "testuser", mock_gh_extractor, predictor=mock_predictor, min_events=5, min_confidence=1
        )

        assert result["contributor"] == "testuser"
        assert result["type"] == "Unknown"
        assert result["confidence"] == "-"

    @patch("rabbit.main.ONNXPredictor")
    @patch("rabbit.main.GitHubAPIExtractor")
    def test_process_contributor_returns_invalid_when_not_found(
        self, mock_gh_extractor, mock_predictor
    ):
        """Test _process_single_contributor handles NotFoundError correctly."""
        from rabbit.errors import NotFoundError

        mock_gh_extractor.query_events.side_effect = NotFoundError("User not found")

        result = _process_single_contributor(
            "invaliduser", mock_gh_extractor, predictor=mock_predictor, min_events=5, min_confidence=1
        )

        assert result["contributor"] == "invaliduser"
        assert result["type"] == "Invalid"
        assert result["confidence"] == "-"

    @patch("rabbit.main.ONNXPredictor")
    @patch("rabbit.main.GitHubAPIExtractor")
    @patch("rabbit.main.predict_user_type")
    def test_process_contributor_with_early_stopping(
            self, mock_predict, mock_gh_extractor, mock_predictor
    ):
        """Test _process_single_contributor stops early when min_confidence is reached."""

        # Simulate events returned in pages
        mock_gh_extractor.query_events.return_value = [
            [{"event": "test"}] * 100,
            [{"event": "test"}] * 100,
        ]

        # First call returns low confidence, second call returns high confidence
        mock_predict.side_effect = [("Human", 0.6), ("Human", 0.95)]

        result = _process_single_contributor(
            "testuser",
            mock_gh_extractor,
            predictor=mock_predictor,
            min_events=5,
            min_confidence=0.5,
        )

        assert result["contributor"] == "testuser"
        assert result["type"] == "Human"
        assert result["confidence"] == 0.6

        # Make sure that it was called only once
        assert mock_gh_extractor.query_events.call_count == 1

    @patch("rabbit.main.ONNXPredictor")
    @patch("rabbit.main.GitHubAPIExtractor")
    def test_process_contributor_forwards_api_errors(
        self, mock_gh_extractor, mock_predictor
    ):
        """Test _process_single_contributor forwards GitHubAPIError exceptions."""
        from rabbit.errors import RabbitErrors

        mock_gh_extractor.query_events.side_effect = RabbitErrors("API error")

        with pytest.raises(RabbitErrors):
            _process_single_contributor(
                "testuser", mock_gh_extractor, predictor=mock_predictor, min_events=5, min_confidence=1
            )

    @patch("rabbit.main.ONNXPredictor")
    @patch("rabbit.main.GitHubAPIExtractor")
    def test_process_contributor_forwards_unexpected_errors(
        self, mock_gh_extractor, mock_predictor
    ):
        """Test _process_single_contributor forwards unexpected exceptions."""

        mock_gh_extractor.query_events.side_effect = Exception("Unexpected error")

        with pytest.raises(RabbitErrors):
            _process_single_contributor(
                "testuser", mock_gh_extractor, predictor=mock_predictor, min_events=5, min_confidence=1
            )


class TestRunRabbit:
    """Tests for the run_rabbit function."""

    @patch("rabbit.main._process_single_contributor")
    def test_run_rabbit_multiple_contributors(self, mock_process):
        """Test run_rabbit processes multiple contributors correctly."""
        sample_result = {
            "contributor": "testuser",
            "type": "Human",
            "confidence": 0.95,
        }
        mock_process.return_value = sample_result

        contributors = ["user1", "user2", "user3"]

        results = list(run_rabbit(contributors))

        assert len(results) == 3
        assert mock_process.call_count == 3

        # Vérifier le contenu des résultats
        for result in results:
            assert result["type"] == "Human"
            assert result["confidence"] == 0.95
