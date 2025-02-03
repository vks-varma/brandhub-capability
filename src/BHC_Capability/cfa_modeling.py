import os

import numpy as np
import pandas as pd
import semopy as sp
import yaml
from semopy import Model

columns_list = [
    "seed",
    "lhs",
    "op",
    "rhs",
    "est.std",
    "se",
    "z",
    "pvalue",
    "factor_str",
    "cfi",
    "tli",
    "rmsea",
]


def cfa_py(fa_str, scaled_data):

    model = Model(fa_str, cov_diag=False)
    model.fit(scaled_data, solver="L-BFGS-B")  # , estimator="GLS")

    # Retrieve fit statistics (method may vary)
    stats = sp.calc_stats(model)

    try:
        fit_indices = model.fit_stats()
    except AttributeError:
        # If fit_stats() is not available, check other attributes
        fit_indices = (
            model.statistics_ if hasattr(model, "statistics_") else {}
        )
    cfa_fit_indices = pd.DataFrame(
        fit_indices.items(), columns=["fitmeasure", "value"]
    )
    cfa_fit_indices = cfa_fit_indices[
        cfa_fit_indices["fitmeasure"].isin(["cfi", "tli", "rmsea"])
    ]
    cfa_fit_indices["value"] = cfa_fit_indices["value"].astype(float)

    cfa_fit_indices_t = cfa_fit_indices.set_index("fitmeasure").T

    # Extract parameter estimates
    cfa_estimates = model.inspect()

    # CFA summary table
    cfa_summary = pd.concat([cfa_estimates, cfa_fit_indices_t], axis=1)

    # Store the results
    cfa_summary["factor_str"] = fa_str
    # cfa_summary["Seed"] = seed
    cfa_summary["cfi"] = stats.loc["Value"]["CFI"]

    if stats.loc["Value"]["RMSEA"] == np.inf:
        cfa_summary["tli"] = 1
        cfa_summary["rmsea"] = 0
    else:
        cfa_summary["tli"] = stats.loc["Value"]["TLI"]
        cfa_summary["rmsea"] = stats.loc["Value"]["RMSEA"]
    cfa_summary = cfa_summary.rename(
        columns={
            "lval": "rhs",
            "rval": "lhs",
            "Estimate": "est.std",
            "Std. Err": "se",
            "z-value": "z",
            "p-value": "pvalue",
        }
    )
    cfa_summary.replace({"op": {"~": "=~"}}, inplace=True)

    # cfa_summary = cfa_summary[columns_list]

    return cfa_summary


def process_cfa_samples(
    cfa_py, factor_str, cfa_data_filtered, config, group_tuple
):
    """
    Processes CFA samples, fits CFA to each sample, and concatenates the results.

    Args:
        cfa_py (function): Function to perform CFA.
        factor_str (str): The CFA factor string.
        cfa_data_filtered (DataFrame): The original filtered DataFrame.
        config (dict): Configuration dictionary with sampling seeds and group variables.
        group_tuple (tuple): Tuple containing values for `data_prep_group_var`.

    Returns:
        DataFrame: Concatenated results with additional columns.
    """
    cfa_data_filtered_list = []

    # Perform sampling and fit CFA for each sample
    for seed in config["cfa_sampling_seeding"]:
        # Sample 95% of the data
        sampled_df = cfa_data_filtered.sample(frac=0.95, random_state=seed)

        # Fit CFA and drop rows with missing 'op' values
        fit_cfa_df = cfa_py(factor_str, sampled_df).dropna(subset=["op"])

        # Add group variables as columns
        for i, x in enumerate(config["data_prep_group_var"]):
            fit_cfa_df[x] = group_tuple[i]

        fit_cfa_df["seed"] = seed

        # Append the processed DataFrame to the list
        cfa_data_filtered_list.append(fit_cfa_df)

    # Concatenate all results
    concatenated_results = pd.concat(cfa_data_filtered_list, ignore_index=True)

    return concatenated_results


def perform_cfa_analysis(scaled_data, idv_list, config, cfa_py, paths):
    """
    Performs CFA analysis across multiple groups, equity pillars, and samples.
    """
    group_list = [config["date_column"]] + config["data_prep_group_var"]
    final_results = []  # To store results from all groups and equity pillars
    pillar_idv_dict = (
        idv_list.groupby("equity_pillar")["idv"].apply(list).to_dict()
    )
    for group, group_df in scaled_data.groupby(config["data_prep_group_var"]):

        # Pivot the data
        cfa_data = group_df.pivot(
            index=group_list, columns="metric", values="value"
        ).reset_index()
        group_value_tuple = group  # Capture group values for column assignment

        for pillar in pillar_idv_dict.keys():
            # Filter columns available for the current pillar
            col_available = [
                col for col in cfa_data if col in pillar_idv_dict[pillar]
            ]

            if col_available:  # Proceed only if there are relevant columns
                # Filter data for CFA
                cfa_data_filtered = cfa_data[col_available]

                # Create factor string
                factor_str = f"{pillar} =~ " + " + ".join(col_available)

                # Process CFA samples and collect results
                results = process_cfa_samples(
                    cfa_py=cfa_py,
                    factor_str=factor_str,
                    cfa_data_filtered=cfa_data_filtered,
                    config=config,
                    group_tuple=group_value_tuple,
                )
                final_results.append(results)

    cfa_fit_data = pd.concat(final_results, ignore_index=True)
    print(cfa_fit_data.columns)
    col_arrangement = config["data_prep_group_var"] + columns_list
    cfa_fit_data = cfa_fit_data[col_arrangement]
    cfa_fit_data.to_csv(paths["cfa_fit_data_path"], index=False)

    # Concatenate all results from all groups and pillars
    return cfa_fit_data
