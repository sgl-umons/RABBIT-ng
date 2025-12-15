import numpy as np
import pandas as pd
import pytest

from rabbit.predictor.features import (
    compute_user_features,
    _compute_gini,
    _compute_descriptive_stats,
    _validate_single_contributor,
    _convert_activities_to_dataframe,
)


class TestValidation:
    """Test input validation"""

    def test_pass_with_single_contributor(self):
        df = pd.DataFrame({"contributor": ["user1", "user1", "user1"]})
        _validate_single_contributor(df)  # Should not raise

    def test_raises_with_multiple_contributors(self):
        df = pd.DataFrame({"contributor": ["user1", "user2", "user1"]})

        with pytest.raises(ValueError, match="Expected activities for one contributor"):
            _validate_single_contributor(df)


class TestDescriptiveStats:
    """Tests for descriptive statistics computation."""

    def test_gini_with_uniform_distribution(self):
        """Uniform distribution should have Gini ≈ 0."""
        array = np.array([5, 5, 5, 5])
        assert _compute_gini(array) == pytest.approx(0.0, abs=0.01)

    def test_gini_with_high_inequality(self):
        """High inequality should have Gini closer to 1."""
        # Example: [1, 2, 5, 10, 50, 100] has Gini ≈ 0.639
        array = np.array([1, 2, 5, 10, 50, 100])
        gini = _compute_gini(array)
        assert gini == pytest.approx(0.63, abs=0.01)

    def test_gini_with_moderate_inequality(self):
        """Moderate inequality example."""
        array = np.array([10, 20, 30, 40])
        gini = _compute_gini(array)
        assert gini == pytest.approx(0.25, abs=0.01)

    def test_gini_with_zeros_filtered(self):
        """Zeros should be filtered before computation."""
        # After filtering: [5, 10] → Gini ≈ 0.167
        array = np.array([0, 0, 5, 10])
        gini = _compute_gini(array)
        assert gini == pytest.approx(0.16, abs=0.01)

    def test_gini_with_all_zeros(self):
        """All zeros should return 0 (no inequality among non-zeros)."""
        array = np.array([0, 0, 0])
        assert _compute_gini(array) == 0.0

    def test_gini_with_single_nonzero(self):
        """Single non-zero value → perfect equality → Gini = 0."""
        array = np.array([0, 0, 0, 100])
        assert _compute_gini(array) == 0.0

    def test_basic_stats(self):
        """Verify mean, median, std, IQR calculations."""
        series = pd.Series([1, 2, 3, 4, 5])
        stats = _compute_descriptive_stats(series)

        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["std"] == pytest.approx(1.58, abs=0.01)
        assert stats["IQR"] == 2.0  # Q3 (4) - Q1 (2)
        assert 0.0 <= stats["gini"] <= 1.0

    def test_single_value(self):
        """Single value should have std = 0."""
        series = pd.Series([5])
        stats = _compute_descriptive_stats(series)

        assert stats["mean"] == 5.0
        assert stats["median"] == 5.0
        assert stats["std"] == 0.0
        assert stats["IQR"] == 0.0

class TestComputeFeatures:

    @pytest.fixture
    def github_sample_activities(self):
        """Sample activity data from GitHub for testing."""
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

    def test_can_compute_df_from_activity_sequences(self, github_sample_activities):
        df = _convert_activities_to_dataframe(github_sample_activities)

        assert len(df) == 4
        assert list(df.columns) == ["date", "activity", "contributor", "repository", "owner"]
        assert df["contributor"].unique().tolist() == ["testuser"]
        assert df["repository"].iloc[0] == 1
        assert df["owner"].iloc[0] == "owner1"
        assert df["date"].iloc[0] == pd.Timestamp("2024-01-01 10:00:00")

    def test_compute_counting_features(self, github_sample_activities):
        features = compute_user_features("testuser", github_sample_activities).iloc[0]

        assert features["NA"] == 4
        assert features["NT"] == 3  # PushEvent, IssuesEvent, PullRequestEvent
        assert features["NOR"] == 2
        assert features["ORR"] == 2 / 2

    def test_compute_single_activity(self):
        activities = [
            {
                "start_date": "2024-01-01T10:00:00Z",
                "activity": "PushEvent",
                "actor": {"login": "testuser"},
                "repository": {"id": 1, "name": "owner1/repo1"},
            }
        ]

        features = compute_user_features("testuser", activities).iloc[0]

        assert features["NA"] == 1

        # Time differences should be NaN since there's only one activity
        assert pd.isna(features["DAAR_mean"])
        assert pd.isna(features["DCA_mean"])
        assert pd.isna(features["DCAT_mean"])

    def test_compute_features_real_example(self):
        import json
        from pathlib import Path
        data_file = Path(__file__).parent.parent / "data" / "human_activities.json"
        with open(data_file, "r", encoding="utf-8") as f:
            activities = json.load(f)

        features = compute_user_features("testuser", activities).iloc[0]

        # Read csv
        features_file = Path(__file__).parent.parent / "data" / "human_features.csv"
        expected_features = pd.read_csv(features_file, index_col=0)

        for col in expected_features.columns:
            expected_value = expected_features.iloc[0][col]
            computed_value = features[col]
            if pd.api.types.is_float_dtype(type(expected_value)):
                assert computed_value == pytest.approx(expected_value, abs=0.01), f"Mismatch in column {col}"
            else:
                assert computed_value == expected_value, f"Mismatch in column {col}"
