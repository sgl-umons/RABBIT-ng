"""
Feature extraction module for RABBIT bot detection.

This module is used to computes 38 behavioral features from user activity sequences:
- NA: number of activities
- NT: number of activity types
- NOR: number of repository owners
- ORR: owner/repository ratio
- DCA: time difference between consecutive activities
- NAR: number of activities per repository
- NTR: number of activity types per repository
- NCAR: number of continuous activities in a repo
- DCAR: time spent in each repository
- DAAR: time taken to switch repos
- DCAT: time taken to switch activity type
- NAT: number of activities per type

Each feature (except NA, NT, NOR, ORR) includes statistical aggregations:
mean, median, std, Gini coefficient, and/or IQR.
"""

import numpy as np
import pandas as pd

__all__ = ["ActivityFeatureExtractor", "FEATURE_NAMES"]

# Constants
TIME_UNIT = "1 hour"
FEATURE_CONFIG = {
    "NA": [],
    "NT": [],
    "NOR": [],
    "ORR": [],
    "DCA": ["mean", "median", "std", "gini"],
    "NAR": ["mean", "median", "gini", "IQR"],
    "NTR": ["mean", "median", "std", "gini"],
    "NCAR": ["mean", "std", "IQR"],
    "DCAR": ["mean", "median", "std", "IQR"],
    "DAAR": ["mean", "median", "std", "gini", "IQR"],
    "DCAT": ["mean", "median", "std", "gini", "IQR"],
    "NAT": ["mean", "median", "std", "gini", "IQR"],
}

FEATURE_NAMES = [
    f"{feature}_{stat}" if stat else feature
    for feature, stats in FEATURE_CONFIG.items()
    for stat in (stats if stats else [""])
]
"""List of all 38 features names used by the BIMBAS model. (e.g NA, DCA_mean, NAR_gini, etc.)"""

INTEGER_FEATURES = ["NA", "NT", "NOR"]


class ActivityFeatureExtractor:
    """
    Extract behavioral features from a single contributor's activity sequence.

    This class processes GitHub activity data and computes 38 statistical features
    used by the BIMBAS machine learning model to distinguish bots from humans.

    Args:
        username: The GitHub username of the contributor.
        activity_sequences: List of activity dictionaries from ghmap transformation.

    Attributes:
        username: The contributor's GitHub username.
        activity_df: DataFrame containing processed activity data. (Computed at init)

    Raises:
        ValueError: If activities for multiple contributors are found.

    Example:
        >>> activities = [
        ...     {"activity": "Push", "start_date": "2024-01-01T10:00:00Z",
        ...      "actor": {"login": "alice"}, "repository": {"id": 123, "name": "owner/repo"}}
        ... ]
        >>> extractor = ActivityFeatureExtractor("alice", activities)
        >>> features = extractor.compute_features()
        >>> print(features.columns.tolist())
        ['NA', 'NT', 'NOR', 'ORR', 'DCA_mean', ...]
    """

    COL_DATE = "date"
    COL_ACTIVITY = "activity"
    COL_CONTRIBUTOR = "contributor"
    COL_REPOSITORY = "repository"
    COL_OWNER = "owner"

    def __init__(self, username: str, activity_sequences: list[dict]):
        """
        Initialize the extractor, validate data and prepare the activity DataFrame.

        Args:
            username: The username of the contributor.
            activity_sequences: List of activity dictionaries for the user.

        Raises:
            ValueError: If activities for multiple contributors are found.
        """
        self.username = username
        self.activity_df = self._prepare_dataframe(activity_sequences)
        self._validate_date()

    def compute_features(self) -> pd.DataFrame:
        """
        Compute all 38 behavioral features for the contributor.

        Returns:
            DataFrame with one row containing all features. Columns are named
            according to FEATURE_NAMES constant. Index is the username.

        Example:
            >>> activities = [
            ...     {"activity": "Push", "start_date": "2024-01-01T10:00:00Z",
            ...      "actor": {"login": "alice"}, "repository": {"id": 123, "name": "owner/repo"}}
            ... ]
            >>> extractor = ActivityFeatureExtractor("alice", activities)
            >>> features = extractor.compute_features()
            >>> features.loc["alice", "NA"]  # Number of activities
            1
        """
        counting_features = self._compute_counting_features()
        aggregated_features = self._compute_aggregated_features()

        all_features = {**counting_features, **aggregated_features}

        features_df = pd.json_normalize(all_features, sep="_")

        features_df = features_df.astype("float").round(3)
        for col in INTEGER_FEATURES:
            if col in features_df.columns:
                features_df = features_df.astype({col: "int"})

        return features_df[FEATURE_NAMES].set_index(pd.Index([self.username]))

    def _prepare_dataframe(self, activity_sequences: list[dict]) -> pd.DataFrame:
        """
        Convert activity sequences to a DataFrame for feature extraction.
        """
        activities_data = []
        for activity in activity_sequences:
            repo_name = activity["repository"]["name"]
            owner = repo_name.split("/")[0] if "/" in repo_name else "unknown"

            activities_data.append(
                {
                    self.COL_DATE: activity["start_date"],
                    self.COL_ACTIVITY: activity["activity"],
                    self.COL_CONTRIBUTOR: activity["actor"]["login"],
                    self.COL_REPOSITORY: activity["repository"]["id"],
                    self.COL_OWNER: owner,
                }
            )
        activities_df = pd.DataFrame(activities_data)

        if not activities_df.empty:
            activities_df[self.COL_DATE] = pd.to_datetime(
                activities_df[self.COL_DATE],
                errors="coerce",
                format="%Y-%m-%dT%H:%M:%SZ",
            ).dt.tz_localize(None)

            # Sort by date (Important for time-based features)
            activities_df = activities_df.sort_values(self.COL_DATE).reset_index(
                drop=True
            )

        return activities_df

    def _validate_date(self) -> None:
        """
        Ensures the DataFrame contains data for exactly one contributor. If not, raises a ValueError.
        """
        if self.activity_df.empty:
            return
        unique_contributors = self.activity_df[self.COL_CONTRIBUTOR].unique()
        if len(unique_contributors) != 1:
            raise ValueError(
                f"Expected activities for one contributor, found {len(unique_contributors)}: "
                f"{unique_contributors}"
            )

    def _compute_counting_features(self) -> dict[str, int | float]:
        """Computes simple counts (NA, NT, NOR, ORR)."""
        n_owners = self.activity_df[self.COL_OWNER].nunique()
        n_repos = self.activity_df[self.COL_REPOSITORY].nunique()

        return {
            "NA": len(self.activity_df),
            "NT": self.activity_df[self.COL_ACTIVITY].nunique(),
            "NOR": n_owners,
            "ORR": (n_owners / n_repos) if n_repos > 0 else 0.0,
        }

    def _compute_aggregated_features(self) -> dict[str, dict[str, float]]:
        """Computes aggregated features (DCA, NAR, NTR, NCAR, DCAR, DAAR, DCAT, NAT)."""
        if self.activity_df.empty:
            return {
                key: {stat: np.nan for stat in stats}
                for key, stats in FEATURE_CONFIG.items()
                if stats
            }

        features = {
            "DCA": self._compute_dca(),
            "NAR": self._compute_nar(),
            "NTR": self._compute_ntr(),
            "NAT": self._compute_nat(),
            "DCAT": self._compute_dcat(),
        }

        ncar, dcar, daar = self._compute_repo_switching_metrics()
        features["NCAR"] = ncar
        features["DCAR"] = dcar
        features["DAAR"] = daar

        return features

    def _compute_dca(self) -> dict[str, float]:
        """DCA: Time difference between consecutive activities."""
        times = self.activity_df[self.COL_DATE]
        # Compute time diffs with next activity
        time_diffs = (times.shift(-1) - times).dropna() / pd.to_timedelta(TIME_UNIT)
        return self._compute_stats(time_diffs)

    def _compute_nar(self) -> dict[str, float]:
        """NAR: Activities per repository"""
        counts = self.activity_df.groupby(self.COL_REPOSITORY, sort=False)[
            self.COL_ACTIVITY
        ].count()
        if isinstance(counts, pd.DataFrame):
            counts = counts.iloc[:, 0]
        return self._compute_stats(counts)

    def _compute_ntr(self) -> dict[str, float]:
        """NTR: Number of activity types per repository"""
        counts = self.activity_df.groupby(self.COL_REPOSITORY, sort=False)[
            self.COL_ACTIVITY
        ].nunique()
        return self._compute_stats(counts)

    def _compute_nat(self) -> dict[str, float]:
        """NAT: Number of activities per activity type."""
        counts = self.activity_df.groupby(self.COL_ACTIVITY, sort=False)[
            self.COL_REPOSITORY
        ].count()
        if isinstance(counts, pd.DataFrame):
            counts = counts.iloc[:, 0]
        return self._compute_stats(counts)

    def _compute_dcat(self) -> dict[str, float]:
        """DCAT: Time diff between consecutive activities of different types"""
        metrics = self._get_switching_metrics(self.COL_ACTIVITY)
        return self._compute_stats(metrics["time_to_switch"])

    def _compute_repo_switching_metrics(self) -> tuple[dict, dict, dict]:
        """Computes NCAR, DCAR and DAAR"""
        metrics = self._get_switching_metrics(self.COL_REPOSITORY)

        ncar = self._compute_stats(metrics["activities_count"])
        dcar = self._compute_stats(metrics["time_spent"])
        daar = self._compute_stats(metrics["time_to_switch"])

        return ncar, dcar, daar

    def _get_switching_metrics(self, group_col: str) -> pd.DataFrame:
        """
        Compute metrics for consecutive activity groupings.

        This function groups consecutive activities that share the same value
        in the specified column (e.g., same repository or same activity type).

        Used to compute:
        - NCAR, DCAR, DAAR (when group_by_column='repository')
        - DCAT (when group_by_column='activity')
        """
        is_new_group = self.activity_df[group_col] != self.activity_df[group_col].shift(
            1
        )
        group_ids = is_new_group.cumsum()

        grouped = self.activity_df.groupby(group_ids, sort=False)

        aggs = grouped.agg(
            activities_count=(self.COL_ACTIVITY, "count"),
            start=(self.COL_DATE, "first"),
            end=(self.COL_DATE, "last"),
        )

        time_unit = pd.to_timedelta(TIME_UNIT)

        # Time spent in the group
        aggs["time_spent"] = (aggs["end"] - aggs["start"]) / time_unit

        # Time to switch to the NEXT group
        aggs["next_start"] = aggs["start"].shift(-1)
        aggs["time_to_switch"] = (aggs["next_start"] - aggs["end"]) / time_unit

        return aggs[["activities_count", "time_spent", "time_to_switch"]]

    @staticmethod
    def _compute_gini(array: np.ndarray) -> float:
        """Calculates Gini coefficient."""
        array = array[array != 0]
        if len(array) == 0:
            return 0.0
        array = array.flatten()
        array = np.sort(array)
        n = array.shape[0]
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    def _compute_stats(self, series: pd.Series) -> dict[str, float]:
        """Computes mean, median, std, gini, IQR for a series."""
        if series.empty:
            return {"mean": 0, "median": 0, "std": 0, "gini": 0, "IQR": 0}

        q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
        std = series.std()

        return {
            "mean": series.mean(),
            "median": median,
            "std": 0.0 if np.isnan(std) else std,
            "gini": self._compute_gini(series.dropna().to_numpy()),
            "IQR": q3 - q1,
        }
