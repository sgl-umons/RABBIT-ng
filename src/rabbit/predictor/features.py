"""
Feature extraction module for RABBIT bot detection.

This module computes 38 behavioral features from user activity sequences:
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

__all__ = ["compute_user_features", "FEATURE_NAMES"]

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
INTEGER_FEATURES = ["NA", "NT", "NOR"]


def _validate_single_contributor(activities_df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains activities for only one contributor.

    Args:
        activities_df: DataFrame with 'contributor' column

    Raises:
        ValueError: If activities for multiple contributors are found
    """
    unique_contributors = activities_df.contributor.unique()
    if len(unique_contributors) != 1:
        raise ValueError(
            f"Expected activities for one contributor, found {len(unique_contributors)}: "
            f"{unique_contributors}"
        )


def _compute_gini(array: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a numpy array.

    Args:
        array: Numpy array of numeric values

    Returns:
        Gini coefficient as a float
    """
    array = array[array != 0]
    if len(array) == 0:
        return 0.0

    array = array.flatten()
    array = np.sort(array)
    n = array.shape[0]
    index = np.arange(1, n + 1)
    gini_coef = (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    return gini_coef


def _compute_descriptive_stats(series: pd.Series) -> dict:
    """
    Compute descriptive statistics for a numeric series.

    Args:
        series: Pandas Series of numeric values

    Returns:
        Dictionary with keys: 'mean', 'median', 'std', 'gini', 'IQR'
    """
    q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
    std = series.std()

    return {
        "mean": series.mean(),
        "median": median,
        "std": 0.0 if np.isnan(std) else std,
        "gini": _compute_gini(series.dropna().to_numpy()),
        "IQR": q3 - q1,
    }


def _compute_time_between_activities(activities_df: pd.DataFrame) -> dict:
    """
    Compute DCA: Time difference between consecutive activities.

    Args:
        activities_df: DataFrame with 'date', 'activity', and 'repository' columns

    Returns:
        Dictionary of descriptive statistics (mean, median, std, gini and IQR)
    """
    time_diffs = (
        activities_df[["date", "activity", "repository"]]
        .sort_values("date")
        .assign(
            next_timestamp=lambda d: d.date.shift(-1)
        )  # Add column with date of next activity
        .assign(time_diff=lambda d: d.next_timestamp - d.date)
        .assign(time_diff=lambda d: d.time_diff / pd.to_timedelta(TIME_UNIT))[
            "time_diff"
        ]
        .dropna()
    )

    return _compute_descriptive_stats(time_diffs)


def _compute_activities_per_repository(activities_df: pd.DataFrame) -> dict:
    """
    Compute NAR: Number of activities per repository.

    Args:
        activities_df: DataFrame with 'repository' and 'activity' columns

    Returns:
        Dictionary of descriptive statistics for NAR (mean, median, gini, IQR)
    """
    activities_per_repo = activities_df.groupby("repository", sort=False).agg(
        count=("activity", "count")
    )["count"]

    return _compute_descriptive_stats(activities_per_repo)


def _compute_activity_types_per_repository(activities_df: pd.DataFrame) -> dict:
    """
    Compute NTR: Number of activity types per repository.

    Args:
        activities_df: DataFrame with 'repository' and 'activity' columns

    Returns:
        Dictionary of descriptive statistics for NTR (mean, median, std, gini)
    """
    activity_types_per_repo = activities_df.groupby("repository", sort=False).agg(
        unique_activities=("activity", "nunique")
    )["unique_activities"]

    return _compute_descriptive_stats(activity_types_per_repo)


def _compute_switching_metrics(
    activities_df: pd.DataFrame, group_by_column: str
) -> pd.DataFrame:
    """
    Compute metrics for consecutive activity groupings.

    This function groups consecutive activities that share the same value
    in the specified column (e.g., same repository or same activity type).

    Used to calculate:
    - NCAR, DCAR, DAAR (when group_by_column='repository')
    - DCAT (when group_by_column='activity')

    Args:
        activities_df: DataFrame with 'date' and the specified column
        group_by_column: Column name to group by ('repository' or 'activity')

    Returns:
        DataFrame with columns:
        - activities: count of consecutive activities in each group
        - time_spent: duration between first and last activity in group
        - time_to_switch: time gap before switching to next group
    """

    # Detect group boundaries: True when value changes from previous row
    group_changes = activities_df[group_by_column] != activities_df[
        group_by_column
    ].shift(1)

    count_column = "activity" if group_by_column == "repository" else "repository"
    time_unit = pd.to_timedelta(TIME_UNIT)

    return (
        activities_df
        # Create group IDs: increment when value changes from previous row
        .assign(group=group_changes.cumsum())
        .groupby("group", sort=False)
        .agg(
            activities=(count_column, "count"),
            first_time=("date", "first"),
            last_time=("date", "last"),
        )
        # TODO: Optimize to allow vectorized computation without .assign()
        .assign(
            next_first_time=lambda d: d.first_time.shift(-1),
            # Time spent working within this group (DCAR for repos)
            time_spent=lambda d: (d.last_time - d.first_time) / time_unit,
            # Time gap before switching to next group (DAAR/DCAT)
            time_to_switch=lambda d: (d.next_first_time - d.last_time) / time_unit,
        )[["activities", "time_spent", "time_to_switch"]]  # Drop intermediate cols
    )


def _compute_repository_switching_metrics(
    activities_df: pd.DataFrame,
) -> tuple[dict, dict, dict]:
    """
    Compute NCAR, DCAR, and DAAR for repository switching behavior.

    Returns:
        Tuple of (NCAR_stats, DCAR_stats, DAAR_stats)
    """
    switching_data = _compute_switching_metrics(activities_df, "repository")

    ncar_stats = _compute_descriptive_stats(switching_data["activities"])
    dcar_stats = _compute_descriptive_stats(switching_data["time_spent"])
    daar_stats = _compute_descriptive_stats(switching_data["time_to_switch"])

    return ncar_stats, dcar_stats, daar_stats


def _compute_activity_type_switching_time_metrics(activities_df: pd.DataFrame) -> dict:
    """
    Compute DCAT: Time taken to switch activity types.

    Args:
        activities_df: DataFrame with 'date' and 'activity' columns

    Returns:
        Dictionary of descriptive statistics (mean, median, std, gini, IQR)
    """
    switching_data = _compute_switching_metrics(activities_df, "activity")

    return _compute_descriptive_stats(switching_data["time_to_switch"])


def _compute_activities_per_type(activities_df: pd.DataFrame) -> dict:
    """
    Compute NAT: Number of activities per activity type.

    Args:
        activities_df: DataFrame with 'activity' column

    Returns:
        Dictionary of descriptive statistics (mean, median, std, gini, IQR)
    """
    activities_per_type = activities_df.groupby("activity", sort=False).agg(
        count=("repository", "count")
    )["count"]

    return _compute_descriptive_stats(activities_per_type)


def _compute_aggregated_features(activities_df: pd.DataFrame) -> dict:
    """
    Compute all aggregated features from activity data.

    Computes the following features with statistical aggregations (mean, median, std, gini, IQR):
    - DCA: Time difference between consecutive activities
    - NAR: Number of activities per repository
    - NTR: Number of activity types per repository
    - NCAR: Number of continuous activities in a repo
    - DCAR: Time spent in each repository
    - DAAR: Time taken to switch repos
    - DCAT: Time taken to switch activity type
    - NAT: Number of activities per type

    Args:
        activities_df: DataFrame with activity data for one contributor

    Returns:
        Dictionary with feature names as keys and stat dictionaries as values
        (e.g., {"DCA": { "mean": ..., "median": ..., ... }, ...})
    """
    features = {
        "DCA": _compute_time_between_activities(activities_df),
        "NAR": _compute_activities_per_repository(activities_df),
        "NTR": _compute_activity_types_per_repository(activities_df),
    }

    # Repository switching features
    ncar_stats, dcar_stats, daar_stats = _compute_repository_switching_metrics(
        activities_df
    )
    features["NCAR"] = ncar_stats
    features["DCAR"] = dcar_stats
    features["DAAR"] = daar_stats

    # Activity type features
    features["DCAT"] = _compute_activity_type_switching_time_metrics(activities_df)
    features["NAT"] = _compute_activities_per_type(activities_df)

    return features


def _compute_counting_features(activities_df: pd.DataFrame) -> dict:
    """
    Compute counting features:
    - NA: number of activities
    - NT: number of activity types
    - NOR: number of repository owners
    - ORR: owner/repository ratio

    Args:
        activities_df: DataFrame with activity data for one contributor

    Returns:
        Dictionary with feature names as keys
    """
    return {
        "NA": np.int64(len(activities_df)),
        "NT": np.int64(activities_df.activity.nunique()),
        "NOR": np.int64(activities_df.owner.nunique()),
        "ORR": np.float64(
            activities_df.owner.nunique() / activities_df.repository.nunique()
        ),
    }


def _compute_all_features(activities_df: pd.DataFrame) -> dict:
    """
    Compute all features from activity data.

    Args:
        activities_df: DataFrame with activity data for one contributor

    Returns:
        Dictionary with all feature statistics
    """

    _validate_single_contributor(activities_df)

    # Compute basic features
    basic_features = _compute_counting_features(activities_df)

    # Compute aggregated features
    aggregated_features = _compute_aggregated_features(activities_df)

    # Combine all features
    all_features = {**basic_features}
    all_features.update(aggregated_features)

    return all_features


def _convert_activities_to_dataframe(activity_sequences: list) -> pd.DataFrame:
    """
    Convert activity sequences to a DataFrame for feature extraction.

    The returned DataFrame will have the following columns:
    - 'date': datetime of the activity
    - 'activity': type of activity performed
    - 'contributor': username of the actor
    - 'repository': repository ID
    - 'owner': repository owner name

    Args:
        activity_sequences: List of activity dictionaries with fields:
            - start_date: ISO 8601 timestamp
            - activity: activity type
            - actor: dict with 'login' field
            - repository: dict with 'id' and 'name' fields

    Returns:
        DataFrame with normalized activity data
    """
    activities_data = []

    for activity in activity_sequences:
        # Extract owner from repository name (format: "owner/repo")
        # Note: This is GitHub-specific and may need adaptation for other platforms
        # TODO: Modify this if supporting other platforms in the future
        owner = activity["repository"]["name"].split("/")[0]

        activities_data.append(
            {
                "date": activity["start_date"],
                "activity": activity["activity"],
                "contributor": activity["actor"]["login"],
                "repository": activity["repository"]["id"],
                "owner": owner,
            }
        )

    activities_df = pd.DataFrame(activities_data)

    # Parse and normalize dates
    activities_df["date"] = pd.to_datetime(
        activities_df["date"], errors="coerce", format="%Y-%m-%dT%H:%M:%SZ"
    ).dt.tz_localize(None)

    return activities_df


def _extract_features_from_df(username: str, activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all features from activity DataFrame for a single user.

    Args:
        username: Username to set as the DataFrame index
        activity_df: DataFrame with activity data

    Returns:
        DataFrame with one row containing all 38 features, indexed by username
    """
    activity_df = activity_df.sort_values("date")
    all_features = _compute_all_features(activity_df)

    # Flatten nested dictionary to DataFrame (e.g., DCA_mean, DCA_median, etc.)
    features_df = pd.json_normalize(all_features, sep="_")

    features_df = features_df.astype("float").round(3)
    for col in INTEGER_FEATURES:
        features_df = features_df.astype({col: "int"})

    return features_df[FEATURE_NAMES].set_index([[username]])


def compute_user_features(username: str, activity_sequences: list) -> pd.DataFrame:
    """
    Compute behavioral features from user activity sequences.

    This function computes 38 behavioral features for bot detection:
    - NA: number of activities
    - NT: number of activity types
    - NOR: number of repository owners
    - ORR: owner/repository ratio
    - DCA: time difference between consecutive activities (mean, median, std, gini)
    - NAR: number of activities per repository (mean, median, gini, IQR)
    - NTR: number of activity types per repository (mean, median, std, gini)
    - NCAR: number of continuous activities in a repo (mean, std, IQR)
    - DCAR: time spent in each repository (mean, median, std, IQR)
    - DAAR: time taken to switch repos (mean, median, std, gini, IQR)
    - DCAT: time taken to switch activity type (mean, median, std, gini, IQR)
    - NAT: number of activities per type (mean, median, std, gini, IQR)

    Args:
        username: The username of the contributor
        activity_sequences: List of activity dictionaries for the user

    Returns:
        DataFrame with one row containing all features, indexed by username
    """
    activity_df = _convert_activities_to_dataframe(activity_sequences)
    user_features_df = _extract_features_from_df(username, activity_df)

    return user_features_df
