from unittest.mock import patch

import pytest
import pandas as pd

from rabbit.main import _save_results, _process_single_contributor, run_rabbit
from rabbit.errors import RabbitErrors


class TestSaveResults:
    """Tests for the _save_results function."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            [
                {"contributor": "user1", "type": "Bot", "confidence": 0.9},
                {"contributor": "user2", "type": "Human", "confidence": 1.0},
            ]
        )

    def test_save_results_csv(self, sample_data, tmp_path):
        """Test if results are saved correctly in CSV format."""
        output_file = tmp_path / "results.csv"

        _save_results(sample_data, "csv", output_file)

        assert output_file.exists()
        df_res = pd.read_csv(output_file)
        assert len(df_res) == 2
        assert "user1" in df_res["contributor"].values

    def test_save_results_json(self, sample_data, tmp_path):
        """Test if results are saved correctly in JSON format."""
        output_file = tmp_path / "results.json"

        _save_results(sample_data, "json", output_file)

        assert output_file.exists()
        df_res = pd.read_json(output_file)
        assert len(df_res) == 2
        assert "user2" in df_res["contributor"].values

    def test_save_results_text(self, capsys, sample_data):
        """Test if results are printed in the console in text format."""
        _save_results(sample_data, "text", None)

        captured = capsys.readouterr()
        assert "user1" in captured.out
        assert "Bot" in captured.out


class TestProcessSingleContributor:
    """Tests for the _process_single_contributor function."""

    @patch("rabbit.main.GitHubAPIExtractor")
    @patch("rabbit.main.predict_user_type")
    def test_process_contributor_success(self, mock_predict, mock_gh_extractor):
        """Test _process_single_contributor returns correct type and confidence."""

        mock_gh_extractor.query_events.return_value = [{}] * 10  # Simulate 10 events
        mock_predict.return_value = ("Human", 0.95)

        result = _process_single_contributor(
            "testuser", mock_gh_extractor, min_events=5
        )

        assert result["contributor"] == "testuser"
        assert result["type"] == "Human"
        assert result["confidence"] == 0.95

    @patch("rabbit.main.GitHubAPIExtractor")
    def test_process_contributor_returns_unknown_when_no_events(
        self, mock_gh_extractor
    ):
        """Test _process_single_contributor returns 'Unknown' when events are below min_events."""

        mock_gh_extractor.query_events.return_value = [
            {}
        ]  # Simulate less than min_events

        result = _process_single_contributor(
            "testuser", mock_gh_extractor, min_events=5
        )

        assert result["contributor"] == "testuser"
        assert result["type"] == "Unknown"
        assert result["confidence"] == "-"

    @patch("rabbit.main.GitHubAPIExtractor")
    def test_process_contributor_handles_not_found_error(self, mock_gh_extractor):
        """Test _process_single_contributor handles NotFoundError correctly."""
        from rabbit.errors import NotFoundError

        mock_gh_extractor.query_events.side_effect = NotFoundError("User not found")

        result = _process_single_contributor(
            "invaliduser", mock_gh_extractor, min_events=5
        )

        assert result["contributor"] == "invaliduser"
        assert result["type"] == "Invalid"
        assert result["confidence"] == "-"

    @patch("rabbit.main.GitHubAPIExtractor")
    def test_process_contributor_forwards_api_errors(self, mock_gh_extractor):
        """Test _process_single_contributor forwards GitHubAPIError exceptions."""
        from rabbit.errors import RabbitErrors

        mock_gh_extractor.query_events.side_effect = RabbitErrors("API error")

        with pytest.raises(RabbitErrors):
            _process_single_contributor("testuser", mock_gh_extractor, min_events=5)

    @patch("rabbit.main.GitHubAPIExtractor")
    def test_process_contributor_forwards_unexpected_errors(self, mock_gh_extractor):
        """Test _process_single_contributor forwards unexpected exceptions."""

        mock_gh_extractor.query_events.side_effect = Exception("Unexpected error")

        with pytest.raises(RabbitErrors):
            _process_single_contributor("testuser", mock_gh_extractor, min_events=5)


class TestRunRabbit:
    """Tests for the run_rabbit function."""

    @patch("rabbit.main._save_results")
    @patch("rabbit.main._process_single_contributor")
    @patch("rabbit.main.track", side_effect=lambda x, description: x)
    def test_run_rabbit_multiple_contributors(
        self, _mock_track, mock_process, mock_save
    ):
        """Test run_rabbit processes multiple contributors correctly."""
        sample_result = {
            "contributor": "testuser",
            "type": "Human",
            "confidence": 0.95,
        }
        mock_process.return_value = sample_result

        contributors = ["user1", "user2", "user3"]

        run_rabbit(contributors)

        assert mock_process.call_count == 3

        # Make sure results are stored once (default is not incremental)
        mock_save.assert_called_once()

        # Results that will be saved should have 3 entries
        args, _ = mock_save.call_args
        assert len(args[0]) == 3  # all_results DataFrame has 3 rows
        assert args[1] == "text"  # output_type

    @patch("rabbit.main._save_results")
    @patch("rabbit.main._process_single_contributor")
    @patch("rabbit.main.track", side_effect=lambda x, description: x)
    def test_run_rabbit_incremental_output(self, _mock_track, mock_process, mock_save):
        """Test run_rabbit with incremental output."""
        sample_result = {
            "contributor": "testuser",
            "type": "Human",
            "confidence": 0.95,
        }
        mock_process.return_value = sample_result

        contributors = ["user1", "user2"]

        run_rabbit(contributors, incremental=True)

        assert mock_process.call_count == 2

        # Make sure results are stored twice (incremental)
        assert mock_save.call_count == 2
