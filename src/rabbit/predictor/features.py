import numpy as np
import pandas as pd


# TODO:  Refactor everything below
# TODO: Should we remove the dependency on pandas/numpy?


def __compute_gini(array):
    """
    Compute the Gini coefficient of a numpy array
    """
    array = array[array != 0]
    if len(array) == 0:
        return 0
    array = array.flatten()
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    gini_coef = (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    return gini_coef


def __my_describe(num_series):
    """
    args: num_series - Series; a series of numbers corresponding to each feature

    returns: num_desc - Dictionart; a dictionary of aggregated values (aggregate term as key)

    method:
    For the given set of numbers, provide
    1. mean
    2. median
    3. std (if there is only one datapoint, then DAAR and DCAT NaN, as std(NaN) is nan, replacing it with 0.0)
    4. gini coefficient
    5. IQR
    """

    num_desc = {}
    num_desc["mean"] = num_series.mean()  # 1
    num_desc["median"] = num_series.quantile(q=0.5)  # 2
    num_desc["std"] = num_series.std()  # 3
    if np.isnan(num_desc["std"]):
        num_desc["std"] = 0.0
    num_desc["gini"] = __compute_gini(num_series.dropna().to_numpy())  # 4
    Q1 = num_series.quantile(q=0.25)
    Q3 = num_series.quantile(q=0.75)
    num_desc["IQR"] = Q3 - Q1  # 5

    return num_desc


def __aggregate(df_loc):
    """
    args: df_loc - dataframe of activities for a contributor

    return: agg_characteristics - dictionary with charateristics name as key

    method:
    calculate mean, median, std, Gini and IQR for
    1. DCA: Time difference between consecutive activities
    2. NAR: Number of activities per repository
    3. NTR: Number of activity type per repository
    4. NCAR: Number of consecutive activities performed in each repository before switching to the next repository
    5. DCAR: Time spent in each repository before switching to the next repository
    6. DAAR: Time taken to switch from one repository to another
    7. DCAT: Time taken to switch from one activity type to another
    8. NAT: num of activities made per activity type
    """

    agg_characteristics = {}

    # 1
    time_diff_consec_act = (
        df_loc[["date", "activity", "repository"]]
        .sort_values("date")
        .assign(next_time_stamp=lambda d: d.date.shift(-1))
        .assign(time_diff=lambda d: d.next_time_stamp - d.date)
        .assign(time_diff=lambda d: d.time_diff / pd.to_timedelta("1 hour"))[
            "time_diff"
        ]
        .dropna()
    )
    time_desc = __my_describe(time_diff_consec_act)

    # 2
    act_per_repo = df_loc.groupby("repository").agg(act_count=("activity", "count"))[
        "act_count"
    ]
    act_per_repo_desc = __my_describe(act_per_repo)

    # 3
    act_type_per_repo = df_loc.groupby("repository").agg(
        act_type_count=("activity", "nunique")
    )["act_type_count"]
    act_type_per_repo_desc = __my_describe(act_type_per_repo)

    # 4, #5, #6
    continuous_act_repo = (
        df_loc.sort_values("date")
        .assign(group=lambda d: (d.repository != d.repository.shift(1)).cumsum())
        .groupby("group")
        .agg(
            activities=("activity", "count"),
            first_time=("date", "first"),
            last_time=("date", "last"),
        )
        .assign(next_first_time=lambda d: d.first_time.shift(-1))
        .assign(
            time_spent=lambda d: (d.last_time - d.first_time)
            / pd.to_timedelta("1 hour")
        )
        .assign(
            time_to_switch=lambda d: (d.next_first_time - d.last_time)
            / pd.to_timedelta("1 hour")
        )
    )
    # 4
    cont_act_repo_counter_desc = __my_describe(continuous_act_repo["activities"])
    # 5
    time_spent_in_repo_desc = __my_describe(continuous_act_repo["time_spent"])
    # 6
    time_to_switch_repo_desc = __my_describe(
        continuous_act_repo["time_to_switch"]
    )  # .dropna())

    # 7
    continous_act_type = (
        df_loc.sort_values("date")
        .assign(group=lambda d: (d.activity != d.activity.shift(1)).cumsum())
        .groupby("group")
        .agg(
            activities=("repository", "count"),
            first_time=("date", "first"),
            last_time=("date", "last"),
        )
        .assign(next_first_time=lambda d: d.first_time.shift(-1))
        .assign(
            time_spent=lambda d: (d.last_time - d.first_time)
            / pd.to_timedelta("1 hour")
        )
        .assign(
            time_to_switch=lambda d: (d.next_first_time - d.last_time)
            / pd.to_timedelta("1 hour")
        )
    )

    time_to_switch_act_type_desc = __my_describe(
        continous_act_type["time_to_switch"]
    )  # .dropna())

    # 8
    act_per_act_type = df_loc.groupby("activity").agg(
        act_count=("repository", "count")
    )["act_count"]
    act_per_act_type_desc = __my_describe(act_per_act_type)

    agg_characteristics["DCA"] = time_desc  # 1
    agg_characteristics["NAR"] = act_per_repo_desc  # 2
    agg_characteristics["NTR"] = act_type_per_repo_desc  # 3
    agg_characteristics["NCAR"] = cont_act_repo_counter_desc  # 4
    agg_characteristics["DCAR"] = time_spent_in_repo_desc  # 5
    agg_characteristics["DAAR"] = time_to_switch_repo_desc  # 6
    agg_characteristics["DCAT"] = time_to_switch_act_type_desc  # 7
    agg_characteristics["NAT"] = act_per_act_type_desc  # 8

    return agg_characteristics


def __stats(df_loc):
    """
    args: df_loc - dataframe of activities for a contributor

    return: characteristics - dictionary with the charateristics name as key

    method:
    counting metrics:
    1. NA: number of activities
    2. NT: number of activity types
    3. NOR: number of owners involved with
    4. ORR: ratio of number of owners to repos

    aggregated metrics:
    invoke aggregate function to calculate the aggregation values for aggregated metrics
    """

    # Raise assertion error if activities corresponding to more than one contributor is given
    unq_contribs = df_loc.contributor.unique()
    assert len(unq_contribs) == 1, (
        f"Provide activity details for one contributor at a time. The provided contributors are {unq_contribs}"
    )

    individ_characteristics = {}  # to store the characteristics that are represented as a single number
    characteristics = {}  # to store the characteristics that are represented as a single number

    num_activities = np.int64(
        df_loc.groupby("contributor", as_index=False)
        .count()
        .rename(columns={"activity": "activities"})["activities"][0]
    )
    num_activity_type = np.int64(df_loc.activity.nunique())
    num_owner = np.int64(df_loc.owner.nunique())
    ratio_owner_repo = np.float64(df_loc.owner.nunique() / df_loc.repository.nunique())

    individ_characteristics["NA"] = num_activities  # 1
    individ_characteristics["NT"] = num_activity_type  # 2
    individ_characteristics["NOR"] = num_owner  # 3
    individ_characteristics["ORR"] = ratio_owner_repo  # 4

    characteristics["feat"] = individ_characteristics
    characteristics.update(__aggregate(df_loc))

    return characteristics


def __convert_col_type(df_loc):
    """
    args: df_loc - DataFrame; a dataframe of all the features and their values for all the contributors

    returns: df_loc - DataFrame; a dataframe of all the features and their values (converted to required type) for all the contributors

    method: converts all the column to float then converts specific columns to int
    """
    int_feat_col = ["feat_NA", "feat_NT", "feat_NOR"]

    df_loc = df_loc.astype("float").round(3)
    for int_cols in int_feat_col:
        df_loc = df_loc.astype({int_cols: "int"})

    return df_loc


def _activity_to_df(activity_sequences: list) -> pd.DataFrame:
    """
    Convert the activities to a DataFrame compatible with the RABBIT feature extractor.

    The returned DataFrame will have the following columns:
    - 'date'
    - 'activity'
    - 'contributor'
    - 'repository` (id)
    - 'owner`

    Args:
        activity_sequences: A list of dictionaries corresponding to the activities of a contributor

    Returns:
        A DataFrame with the columns 'date', 'activity', 'contributor', 'repository' and 'owner'.
    """
    activities_df = pd.DataFrame()
    for activity in activity_sequences:
        new_row = pd.DataFrame(
            [
                [
                    activity["start_date"],
                    activity["activity"],
                    activity["actor"]["login"],
                    activity["repository"]["id"],
                    activity["repository"]["name"].split("/")[0],
                    # TODO: This should be changed since it's hardcoded to work with GitHub only
                ]
            ],
            columns=["date", "activity", "contributor", "repository", "owner"],
        )
        activities_df = pd.concat([activities_df, new_row], ignore_index=True)
    activities_df["date"] = pd.to_datetime(
        activities_df["date"], errors="coerce", format="%Y-%m-%dT%H:%M:%SZ"
    ).dt.tz_localize(None)

    return activities_df


def _extract_features_from_df(username: str, activity_df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "NA",
        "NT",
        "NOR",
        "ORR",
        "DCA_mean",
        "DCA_median",
        "DCA_std",
        "DCA_gini",
        "NAR_mean",
        "NAR_median",
        "NAR_gini",
        "NAR_IQR",
        "NTR_mean",
        "NTR_median",
        "NTR_std",
        "NTR_gini",
        "NCAR_mean",
        "NCAR_std",
        "NCAR_IQR",
        "DCAR_mean",
        "DCAR_median",
        "DCAR_std",
        "DCAR_IQR",
        "DAAR_mean",
        "DAAR_median",
        "DAAR_std",
        "DAAR_gini",
        "DAAR_IQR",
        "DCAT_mean",
        "DCAT_median",
        "DCAT_std",
        "DCAT_gini",
        "DCAT_IQR",
        "NAT_mean",
        "NAT_median",
        "NAT_std",
        "NAT_gini",
        "NAT_IQR",
    ]

    df_feat = pd.json_normalize(__stats(activity_df), sep="_")

    df_feat = (
        __convert_col_type(df_feat)
        .rename(
            columns={
                "feat_NA": "NA",
                "feat_NT": "NT",
                "feat_NOR": "NOR",
                "feat_ORR": "ORR",
            }
        )[features]  # Reorder the columns
        .sort_index()
        .set_index([[username]])
    )

    return df_feat


def compute_user_features(username: str, activity_sequences: list) -> pd.DataFrame:
    """
    Compute user features based on activity sequences.

    This function computes, 38 such features, namely
        NA: number of activities,
        NT: number of activity types, NOR: number of owners of repositories,
        ORR: Owner repository ratio,
        DCA: time difference between consecutive activities (mean, median, std and gini),
        NAR: number of activities per repository (mean, median, gini and IQR),
        NTR: number of activity types per repository (mean, median, std and gini),
        NCAR: number of continuous activities in a repo (mean, std and IQR),
        DCAR: total time taken to perform consecutive activities in a repo (mean, median, std and IQR),
        DAAR: time taken to switch repos (mean, median, std, gini and IQR),
        DCAT: time taken to switch activity type (mean, median, std, gini and IQR),
        NAT: number of activities per type (mean, median, std, gini and IQR).

    Args:
        username (str): The username of the user
        activity_sequences (list): List of activity sequences for the user (List of dicts)
    Returns:
        DataFrame With computed features of the user.
    """

    activity_df = _activity_to_df(activity_sequences)
    user_features_df = _extract_features_from_df(username, activity_df)

    return user_features_df
