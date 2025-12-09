from pathlib import Path
from unittest.mock import patch

import pytest
import responses
from typer.testing import CliRunner

from rabbit.cli import app

runner = CliRunner()


class TestCLI:
    @pytest.fixture
    def mock_run_rabbit(self):
        """Mock run_rabbit to verify parameters are passed correctly."""
        with patch("rabbit.cli.run_rabbit") as mock:
            yield mock

    def test_app_help(self):
        """Test if the CLI can be invoked and shows help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.stdout

    def test_error_when_no_contributors_given(self):
        """Test if an error is raised when no contributors or input file is provided."""
        result = runner.invoke(app, [])
        assert result.exit_code != 0

    def test_text_output(self, mock_run_rabbit):
        result = runner.invoke(app, ["octocat"])

        assert result.exit_code == 0
        assert mock_run_rabbit.call_args.kwargs["output_type"] == "term"
        assert mock_run_rabbit.call_args.kwargs["output_path"] == ""

    def test_csv_output(self, mock_run_rabbit, tmp_path):
        csv_file = tmp_path / "test_output.csv"
        result = runner.invoke(app, ["octocat", "-f", "csv", "-o", str(csv_file)])

        assert result.exit_code == 0
        assert mock_run_rabbit.call_args.kwargs["output_type"] == "csv"
        assert mock_run_rabbit.call_args.kwargs["output_path"] == str(csv_file)

    def test_json_output(self, mock_run_rabbit, tmp_path):
        json_file = tmp_path / "test_output.json"
        result = runner.invoke(app, ["octocat", "-f", "json", "-o", str(json_file)])

        assert result.exit_code == 0
        assert mock_run_rabbit.call_args.kwargs["output_type"] == "json"
        assert mock_run_rabbit.call_args.kwargs["output_path"] == str(json_file)

    def test_cli_extend_contributors_list_with_file(self, mock_run_rabbit, tmp_path):
        """Test if the list of the contributors is extended with the content of the input file."""
        # Create a temporary input file with contributor names
        input_file = tmp_path / "contributors.txt"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("contributor1\ncontributor2\n")

        result = runner.invoke(
            app,
            [
                "--input-file",
                str(input_file),
                "contributor3",
                "--key",
                "valid_github_api_key_which_is_long_enough_1234567890",
            ],
        )

        assert result.exit_code == 0
        # Check if run_rabbit was called with the correct list of contributors
        expected_contributors = ["contributor3", "contributor1", "contributor2"]
        kwargs = mock_run_rabbit.call_args.kwargs
        assert sorted(kwargs["contributors"]) == sorted(expected_contributors)
        assert (
            kwargs["api_key"] == "valid_github_api_key_which_is_long_enough_1234567890"
        )


class TestIntegration:
    """Complete test suite"""

    @pytest.fixture
    def github_events(self):
        """Charge le JSON depuis le fichier une seule fois pour les tests."""
        import json

        data_file = Path(__file__).parent / "data" / "events_human.json"
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @responses.activate
    def test_real_execution(self, tmp_path, github_events):
        """Test a real execution of the CLI with mocked GitHub API responses."""
        test_user = "testuser"

        # Mock GitHub API response
        responses.add(
            responses.GET,
            f"https://api.github.com/users/{test_user}/events",
            json=github_events,
            status=200,
        )

        result = runner.invoke(
            app,
            [
                test_user,
                "--key",
                "valid_github_api_key_which_is_long_enough_1234567890",
                "-f",
                "json",
                "-o",
                f"/{tmp_path}/output.json",
            ],
        )

        assert result.exit_code == 0

        # Check if the output is saved in tmp_path
        output_file = tmp_path / "output.json"
        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            output_data = f.read()
        assert "testuser" in output_data  # Basic check to see if username is in
        assert (
            "Human" in output_data or "Bot" in output_data
        )  # Check for classification
