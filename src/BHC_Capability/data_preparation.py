import os

import pandas as pd
import yaml


# functions
def config_loading(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_output_paths(config):
    """
    Build full paths for output files based on the configuration.

    Args:
        config (dict): Configuration dictionary with keys:
            - root_path: Root directory path.
            - output_folder: Folder where output files will be saved.
            - filtered_data_filename: Filename for filtered data.
            - no_null_imputed_data_filename: Filename for no-null imputed data.
            - scaled_data_filename: Filename for scaled data.

    Returns:
        dict: A dictionary with keys 'filtered_data_path', 'no_null_imputed_data_path',
              and 'scaled_data_path', containing the full paths for the respective files.
    """
    # Get the root path and output folder
    root_path = config["root_path"]
    output_folder = config["output_folder"]

    # Ensure the output folder exists
    output_path = os.path.join(root_path, output_folder)
    os.makedirs(output_path, exist_ok=True)

    # Build full paths for the output files
    paths = {
        "filtered_data_path": os.path.join(
            output_path, config["filtered_data"]
        ),
        "no_null_imputed_data_path": os.path.join(
            output_path, config["no_null_imputed_data"]
        ),
        "scaled_data_path": os.path.join(output_path, config["scaled_data"]),
    }

    return paths


def date_dv_columns_check(data: pd.DataFrame, config: dict):
    required_columns = [config["date_column"], config["dv_column"]] + config[
        "data_prep_group_var"
    ]
    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The following required columns are missing from the DataFrame: {missing_columns}"
        )
    else:
        print("All required columns are present in the DataFrame.")


def data_date_conversion(data: pd.DataFrame, config: dict):
    data[config["date_column"]] = pd.to_datetime(
        data[config["date_column"]], format=config["date_format"]
    )
    return data


def idv_list_loading(config: dict):
    idv = pd.read_csv(config["idv_list"])
    return idv


def check_idv_columns_in_data(
    data: pd.DataFrame, idv_list: pd.DataFrame, column_name: str, config
):
    """
    Checks if all columns in df1 are present as rows in a specified column of df2.

    Parameters:
    - data (pd.DataFrame): The first DataFrame whose columns need to be checked.
    - idv_list (pd.DataFrame): The second DataFrame with the reference column.
    - column_name (str): The column in df2 that should contain all column names of df1.

    Returns:
    - None: If all columns are found, the function silently passes.

    Raises:
    - ValueError: If any columns are missing, it raises an error with the missing columns.
    """
    # Get the list of columns from df1
    df1_columns = set(data.columns)

    # Get the unique values in the specified column of df2
    df2_values = set(idv_list[column_name])

    # Find the missing columns
    missing_columns = df2_values - df1_columns

    # Raise an error if there are missing columns
    if missing_columns:
        raise ValueError(
            f"The following independent variables are missing in data: {missing_columns}"
        )
    else:
        print("All independent variables in idv_list are present in the data.")
    required_columns = [config["date_column"], config["dv_column"]] + config[
        "data_prep_group_var"
    ]
    return data[required_columns + list(df2_values)]


def data_loading(config: dict):

    input_data = pd.read_csv(config["input_data"])
    date_dv_columns_check(input_data, config)
    data = data_date_conversion(input_data, config)
    idv_list = idv_list_loading(config)
    data = check_idv_columns_in_data(data, idv_list, "idv", config)

    return data, idv_list


def column_arrangement(config: dict, idv_list: pd.DataFrame):
    sorted_idv_list = sorted(idv_list["idv"].tolist())
    sorted_idv_list
    cols_arrangement = (
        [config["date_column"]]
        + config["data_prep_group_var"]
        + sorted_idv_list
        + [config["dv_column"]]
    )
    return cols_arrangement


def filter_by_data_date_range(data: pd.DataFrame, config: dict):
    """Filtering date range for the data processing and further analysis

    Args:
        data (DataFrame): Harmonized_processed_data to filter out the date range
        config (dict): configuration dictionary
    Returns:
        DataFrame: Data with filtered date range
    """
    data["date"] = pd.to_datetime(data["date"], utc=False)

    # Print the minimum and maximum date values for verification
    print("Minimum date:", data["date"].min(skipna=True))
    print("Maximum date:", data["date"].max(skipna=True))

    # Define the date range from run_config
    date1 = pd.to_datetime(config["start_date"], format="%Y-%m-%d")
    date2 = pd.to_datetime(config["end_date"], format="%Y-%m-%d")

    # Filter the DataFrame based on the date range
    data = data[(data["date"] >= date1) & (data["date"] <= date2)]
    if data[config["dv_column"]].isna().sum() != 0:
        raise ValueError(
            f"The dependent variable {config['dv_column']} is having null values"
        )
    return data


### preprocessing functions ####


def impute_groups(
    df: pd.DataFrame,
    group_by_columns: list,
    null_threshold: float,
    imputation_method: str = "mean",
):
    """
    Drop rows for specific combinations of columns where more than the specified percentage of the 'value' column is null,
    and impute missing values for groups with less than the threshold.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - group_by_columns (list): List of columns to group by.
    - null_threshold (float): The threshold of null percentage for which rows will be dropped.
    - imputation_method (str): The method for imputing missing values ('mean', 'median', 'mode').

    Returns:
    - pd.DataFrame: The DataFrame with rows removed and missing values imputed.
    """
    # Step 1: Group by specified columns
    grouped = df.groupby(group_by_columns)

    # Step 2: Calculate the percentage of nulls for each group
    null_percentage = grouped["value"].apply(lambda x: x.isnull().mean())

    # Step 3: Identify groups with more than the null threshold percentage of null values
    groups_to_drop = null_percentage[null_percentage > null_threshold].index
    groups_to_drop_list = groups_to_drop.to_list()

    print(groups_to_drop_list)
    # Step 4: Filter out rows belonging to the identified groups
    df_filtered = df[
        ~df.set_index(group_by_columns).index.isin(groups_to_drop)
    ]

    # Step 5: Impute missing values for the remaining groups (those with less than the threshold null percentage)
    for group, group_df in df_filtered.groupby(group_by_columns):
        if group not in groups_to_drop:
            if imputation_method == "mean":
                fill_value = group_df["value"].mean()
            elif imputation_method == "median":
                fill_value = group_df["value"].median()
            elif imputation_method == "mode":
                fill_value = group_df["value"].mode()[0]
            else:
                raise ValueError(
                    f"Unsupported imputation method: {imputation_method}"
                )

            # Impute the missing values in the group
            df_filtered.loc[group_df.index, "value"] = group_df[
                "value"
            ].fillna(fill_value)

    return df_filtered


# Scaling Function
def scale_metrics(melted_data, idv_list, config):
    def apply_scaling(group):
        metric_name = group["metric"].iloc[0]
        # Skip scaling for the dependent variable
        if metric_name == config["dv_column"]:
            group["value"] = group["value"]
            return group
        if config["scaling"] == "minmax":
            # Min-Max Scaling: Scale between 0 and 1
            min_val = group["value"].min()
            max_val = group["value"].max()
            group["value"] = (group["value"] - min_val) / (max_val - min_val)

        elif config["scaling"] == "standard":
            # Standard Scaling: Scale with mean and standard deviation
            mean_val = group["value"].mean()
            std_dev = group["value"].std()
            group["value"] = (group["value"] - mean_val) / std_dev

        elif config["scaling"] == "custom":
            # Custom Scaling: Use min and max from df2
            custom_min = idv_list.loc[
                idv_list["idv"] == metric_name, "min"
            ].values[0]
            custom_max = idv_list.loc[
                idv_list["idv"] == metric_name, "max"
            ].values[0]
            group["value"] = (group["value"] - custom_min) / (
                custom_max - custom_min
            )

        else:
            raise ValueError(
                "Invalid scaling method. Choose 'minmax', 'standard', or 'custom'."
            )

        return group

    # Group by 'metric' and apply scaling
    scaled_df = melted_data.groupby("metric", group_keys=False).apply(
        apply_scaling
    )
    return scaled_df


def main():
    config_file_path = "config.yml"
    config = config_loading(config_file_path)

    paths = build_output_paths(config)
    print(paths)

    data, idv_list = data_loading(config)
    cols_arrangement = column_arrangement(config, idv_list)
    group_list = [config["date_column"]] + config["data_prep_group_var"]

    filtered_data = filter_by_data_date_range(data, config)

    filtered_data = filtered_data[cols_arrangement]
    filtered_data.to_csv(paths["filtered_data_path"], index=False)

    melted_df = pd.melt(
        filtered_data,
        id_vars=group_list,  # Columns to keep as-is
        var_name="metric",
        value_name="value",
    )

    no_null_imputed_data = impute_groups(
        melted_df, config["data_prep_group_var"] + ["metric"], 0.5
    )
    pivoted_no_null_imputed_data = no_null_imputed_data.pivot(
        index=group_list,
        columns="metric",
        values="value",
    ).reset_index()
    pivoted_no_null_imputed_data = pivoted_no_null_imputed_data[
        [
            col
            for col in cols_arrangement
            if col in pivoted_no_null_imputed_data.columns
        ]
    ]
    pivoted_no_null_imputed_data.to_csv(
        paths["no_null_imputed_data_path"], index=False
    )

    scaled_data = scale_metrics(no_null_imputed_data, idv_list, config)
    pivoted_scaled_data = scaled_data.pivot(
        index=group_list,
        columns="metric",
        values="value",
    ).reset_index()
    pivoted_scaled_data = pivoted_scaled_data[
        [col for col in cols_arrangement if col in pivoted_scaled_data.columns]
    ]
    pivoted_scaled_data.to_csv(paths["scaled_data_path"], index=False)


if __name__ == "__main__":
    main()
