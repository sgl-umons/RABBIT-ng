import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rabbit.predictor.features import ActivityFeatureExtractor


class TestValidation:
    """Test input validation logic in ActivityFeatureExtractor."""

    def test_raises_with_multiple_contributors(self):
        """Should raise ValueError if activities belong to different users."""
        activities = [
            {
                "start_date": "2024-01-01T10:00:00Z",
                "activity": "PushEvent",
                "actor": {"login": "user1"},
                "repository": {"id": 1, "name": "o/r"},
            },
            {
                "start_date": "2024-01-02T10:00:00Z",
                "activity": "PushEvent",
                "actor": {"login": "user2"},  # Different user
                "repository": {"id": 1, "name": "o/r"},
            },
        ]

        with pytest.raises(ValueError, match="Expected activities for one contributor"):
            ActivityFeatureExtractor("user1", activities)


class TestGini:
    """Tests for the static Gini coefficient calculation."""

    def test_gini_with_uniform_distribution(self):
        array = np.array([5, 5, 5, 5])
        assert ActivityFeatureExtractor._compute_gini(array) == pytest.approx(
            0.0, abs=0.01
        )

    def test_gini_with_high_inequality(self):
        array = np.array([1, 2, 5, 10, 50, 100])
        assert ActivityFeatureExtractor._compute_gini(array) == pytest.approx(
            0.63, abs=0.01
        )

    def test_gini_with_moderate_inequality(self):
        array = np.array([10, 20, 30, 40])
        assert ActivityFeatureExtractor._compute_gini(array) == pytest.approx(
            0.25, abs=0.01
        )

    def test_gini_with_zeros_filtered(self):
        """Zeros should be filtered before computation."""
        array = np.array([0, 0, 5, 10])
        assert ActivityFeatureExtractor._compute_gini(array) == pytest.approx(
            0.16, abs=0.01
        )

    def test_gini_with_all_zeros(self):
        array = np.array([0, 0, 0])
        assert ActivityFeatureExtractor._compute_gini(array) == 0.0

    def test_gini_with_single_nonzero(self):
        array = np.array([0, 0, 0, 100])
        assert ActivityFeatureExtractor._compute_gini(array) == 0.0


class TestFeatureExtraction:
    """Integration tests for the ActivityFeatureExtractor class."""

    @pytest.fixture
    def github_sample_activities(self):
        return [
            {
                "start_date": "2024-01-01T10:00:00Z",
                "activity": "PushEvent",
                "actor": {"login": "testuser"},
                "repository": {"id": 1, "name": "owner1/repo1"},
            },
            {
                "start_date": "2024-01-01T11:00:00Z",
                "activity": "PushEvent",
                "actor": {"login": "testuser"},
                "repository": {"id": 1, "name": "owner1/repo1"},
            },
            {
                "start_date": "2024-01-01T13:00:00Z",
                "activity": "IssuesEvent",
                "actor": {"login": "testuser"},
                "repository": {"id": 2, "name": "owner2/repo2"},
            },
            {
                "start_date": "2024-01-01T14:00:00Z",
                "activity": "PullRequestEvent",
                "actor": {"login": "testuser"},
                "repository": {"id": 2, "name": "owner2/repo2"},
            },
        ]

    def test_initialization_prepares_dataframe(self, github_sample_activities):
        """Test that __init__ correctly converts list of dicts to DataFrame."""
        extractor = ActivityFeatureExtractor("testuser", github_sample_activities)
        df = extractor.activity_df

        assert len(df) == 4
        assert list(df.columns) == [
            "date",
            "activity",
            "contributor",
            "repository",
            "owner",
        ]
        assert df["contributor"].unique().tolist() == ["testuser"]
        assert df["repository"].iloc[0] == 1
        assert df["owner"].iloc[0] == "owner1"
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_compute_features_structure(self, github_sample_activities):
        """Test that compute_features returns the expected shape and basic values."""
        extractor = ActivityFeatureExtractor("testuser", github_sample_activities)
        features_df = extractor.compute_features()

        assert len(features_df) == 1
        features = features_df.iloc[0]

        # Basic counting features
        assert features["NA"] == 4
        assert features["NT"] == 3  # PushEvent, IssuesEvent, PullRequestEvent
        assert features["NOR"] == 2
        assert features["ORR"] == 2 / 2

    def test_compute_single_activity(self):
        """Test behavior with a single activity (should handle NaNs/zeros)."""
        activities = [
            {
                "start_date": "2024-01-01T10:00:00Z",
                "activity": "PushEvent",
                "actor": {"login": "testuser"},
                "repository": {"id": 1, "name": "owner1/repo1"},
            }
        ]

        extractor = ActivityFeatureExtractor("testuser", activities)
        features = extractor.compute_features().iloc[0]

        assert features["NA"] == 1
        assert features["DCA_mean"] == 0.0

    def test_compute_features_real_example(self):
        """Regression test using real data files."""
        # Locate data files relative to this test file
        base_dir = Path(__file__).parent.parent / "data"
        data_file = base_dir / "human_activities.json"
        features_file = base_dir / "human_features.csv"

        if not data_file.exists() or not features_file.exists():
            pytest.skip("Test data files not found")

        with open(data_file, "r", encoding="utf-8") as f:
            activities = json.load(f)

        # Compute
        extractor = ActivityFeatureExtractor("testuser", activities)
        computed_features = extractor.compute_features().iloc[0]

        # Compare
        expected_features = pd.read_csv(features_file, index_col=0)

        for col in expected_features.columns:
            expected_val = expected_features.iloc[0][col]
            computed_val = computed_features[col]

            if pd.api.types.is_float_dtype(type(expected_val)):
                assert computed_val == pytest.approx(expected_val, abs=0.01), (
                    f"Mismatch in column {col}"
                )
            else:
                assert computed_val == expected_val, f"Mismatch in column {col}"
