import numpy as np
import pandas as pd


def data_merge(cfa_df, rf_df, idv_list, config):
    """
    Merge CFA and RF data based on pillars and metrics.

    Args
    ----------
        cfa_df (pd.DataFrame): CFA results dataframe.
        rf_df (pd.DataFrame): RF model results dataframe.
        idv_list (pd.DataFrame): List of independent variables and their corresponding pillars.
        config (dict): Configuration dictionary.

    Returns
    ----------
        pd.DataFrame
            Merged dataframe containing combined CFA and RF results.
    """

    data_group = config["data_prep_group_var"]
    # FIXME: should we do this filtering in cfa itself?
    cfa_filtered = cfa_df[
        cfa_df["lhs"].isin(idv_list["equity_pillar"].unique())
        & (cfa_df["op"] == "=~")
        & (cfa_df["seed"] == config["cfa_seed"])
    ]

    cfa_filtered.rename(
        columns={"lhs": "pillar", "rhs": "metric"}, inplace=True
    )

    cfa_filtered[config["cfa_target_col"]] = cfa_filtered[
        config["cfa_target_col"]
    ].abs()

    cfa_filtered = cfa_filtered[
        data_group + ["pillar", "metric"] + [config["cfa_target_col"]]
    ]

    rf_df["feature_importance"] = rf_df["feature_importance"].abs()

    rf_filtered = rf_df[
        data_group + ["pillar", "metric"] + [config["rf_target_col"]]
    ]

    merged_df = cfa_filtered.merge(
        rf_filtered, on=data_group + ["pillar", "metric"], how="inner"
    )

    return merged_df


def weight_creation(merged_df, config):
    """
    Create weights by normalizing CFA and RF values within each group.

    Args
    ----------
        merged_df (pd.DataFrame): Dataframe containing merged CFA and RF values.
        config (dict): Configuration dictionary.

    Returns
    ----------
        pd.DataFrame
            Dataframe with calculated weights.
    """

    weight_creation_df = merged_df.copy()
    # Normalize cfa_value and shap_values within each group
    weight_creation_df[config["cfa_target_col"]] = weight_creation_df.groupby(
        config["data_prep_group_var"] + ["pillar"]
    )[config["cfa_target_col"]].transform(lambda x: x / x.sum())

    weight_creation_df[config["rf_target_col"]] = weight_creation_df.groupby(
        config["data_prep_group_var"] + ["pillar"]
    )[config["rf_target_col"]].transform(lambda x: x / x.sum())

    weight_creation_df["weight"] = (
        weight_creation_df[config["cfa_target_col"]]
        + weight_creation_df[config["rf_target_col"]]
    ) / 2

    weight_creation_df.drop(
        [config["cfa_target_col"], config["rf_target_col"]],
        axis=1,
        inplace=True,
    )
    return weight_creation_df


def pillar_creation(weight_df, scaled_data, config):
    pillar_score_inter_df = pd.merge(
        weight_df,
        scaled_data,
        on=config["data_prep_group_var"] + ["metric"],
        how="inner",  # Use 'inner' to drop metrics without weights
    )

    pillar_score_inter_df["score"] = (
        pillar_score_inter_df["value"] * pillar_score_inter_df["weight"]
    )
    pillar_scores = (
        pillar_score_inter_df.groupby(
            [config["date_column"]]
            + config["data_prep_group_var"]
            + ["pillar"]
        )["score"]
        .sum()
        .reset_index()
    )
    return pillar_scores


def trend_past_creation(pillar_data, config):
    """
    Compute past trend values using a rolling average.

    Args
    ----------
        pillar_data (pd.DataFrame): Dataframe containing pillar scores over time.
        config (dict): Configuration dictionary with rolling window size.

    Returns
    ----------
        pd.DataFrame
            Dataframe with an additional 'trend_past' column.
    """

    trend_past = pillar_data.sort_values(
        by=config["data_prep_group_var"] + ["pillar", config["date_column"]]
    ).copy()

    # Define window size (e.g., 2-day rolling average)
    window_size = config[
        "trend_past_rolling_window"
    ]  # Adjust this value as needed

    # Calculate rolling average within groups
    trend_past["trend_past"] = trend_past.groupby(
        ["brand", "category", "pillar"]
    )["score"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )

    return trend_past


def scaled_score_creation(pillar_data, config):
    """
    Create scaled scores by normalizing pillar scores within each time period.

    Args
    ----------
        pillar_data (pd.DataFrame): Dataframe containing pillar scores.
        config (dict): Configuration dictionary with required column names.

    Returns
    ----------
        pd.DataFrame
            Dataframe with an additional 'scaled_score' column.
    """

    df = pillar_data.copy()

    df["group_mean"] = df.groupby([config["date_column"], "pillar"])[
        "score"
    ].transform("mean")

    # Scale scores such that group mean = 100
    df["scaled_score"] = (df["score"] / df["group_mean"]) * 100

    # Drop intermediate column (optional)
    df = df.drop(columns=["group_mean"])

    return df


def scoring(cfa_df, rf_df, scaled_data, idv_list, config, paths):
    """
    Execute the full scoring process, including data merging, weight creation,
    pillar computation, trend calculation, and score scaling.

    Args
    ----------
        cfa_df (pd.DataFrame): CFA results dataframe.
        rf_df (pd.DataFrame): Random forest feature importance dataframe.
        scaled_data (pd.DataFrame): Scaled input data for modeling.
        idv_list (pd.DataFrame): List of independent variables and their mappings.
        config (dict): Configuration dictionary.
        paths (dict): Dictionary containing file paths for saving results.

    Returns
    ----------
        tuple
            Dataframes for pillar weights, pillar data, trend past data, and scaled scores.
    """

    merged_df = data_merge(cfa_df, rf_df, idv_list, config)
    pillar_weights = weight_creation(merged_df, config)
    pillar_data = pillar_creation(pillar_weights, scaled_data, config)
    trend_past_data = trend_past_creation(pillar_data, config)
    scaled_score_data = scaled_score_creation(pillar_data, config)

    pillar_weights.to_csv(paths["pillar_weights_path"], index=False)
    pillar_data.to_csv(paths["pillar_data_path"], index=False)
    trend_past_data.to_csv(paths["trend_past_data_path"], index=False)
    scaled_score_data.to_csv(paths["scaled_score_data_path"], index=False)

    return pillar_weights, pillar_data, trend_past_data, scaled_score_data
