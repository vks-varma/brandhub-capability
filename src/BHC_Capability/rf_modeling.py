import gc
import os

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_percentage_error

# Machine learning imports
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)


def model_data_prep(
    model_idv_dv_df, col_available, config, test_size_ratio=0.1, shuffle=True
):
    """
    Prepare model data by splitting into training and testing sets.

    Parameters
    ----------
    model_idv_dv_df (DataFrame): The dataset containing independent and dependent variables.
    col_available (list): List of column names to be used as independent variables.
    config (dict): Configuration dictionary containing the dependent variable column name.
    test_size_ratio (float): Proportion of the dataset to be used as the test set.
    shuffle (bool): Whether to shuffle the data before splitting.

    Returns
    ----------
        tuple
            train_x, test_x, train_y, test_y
    """
    idvs = model_idv_dv_df[col_available]
    dv = model_idv_dv_df[config["dv_column"]]

    train_x, test_x, train_y, test_y = train_test_split(
        idvs,
        dv,
        test_size=int(test_size_ratio * idvs.shape[0]),
        shuffle=shuffle,
    )

    return train_x, test_x, train_y, test_y, idvs, dv


def train_and_evaluate_model(config, train_x, train_y):
    """
    Train and evaluate a machine learning model based on the provided configuration.

    Parameters
    ----------
    config (dict): Configuration dictionary containing model details and hyperparameters.
    train_x (DataFrame): Training features.
    train_y (Series): Training target.

    Returns
    ----------
        regressor : object
            Trained model.
        feat_importance : DataFrame
            Feature importance values.
        feat_df : DataFrame
            SHAP feature importance values.

    """
    # Select model class based on config
    if config["model_type"] == "RandomForest":
        model_class = RandomForestRegressor

    # Extract hyperparameters
    param_grid = {
        key: config["model_config"][config["model_type"]]["grid_search"][key]
        for key in config["model_config"][config["model_type"]]["grid_search"]
        if key != "eval_metrics"  # Exclude eval_metrics
    }

    # Initialize model with random state
    regressor = model_class(random_state=param_grid["random_state"])

    # Perform Grid Search
    search = GridSearchCV(
        regressor,
        param_grid,
        cv=config["cross_validation_number"],
        scoring=["r2", "neg_mean_absolute_percentage_error"],
        refit="neg_mean_absolute_percentage_error",
    )

    search.fit(train_x, train_y)

    print(
        f"The best hyperparameters for {config['model_type']} are {search.best_params_}"
    )

    # Train model with best parameters
    regressor = model_class(**search.best_params_)
    regressor.fit(train_x, train_y)

    # Feature importance and SHAP values (only for RandomForest)
    feat_importance = None
    shap_df = None
    if config["model_type"] == "RandomForest":
        features = list(train_x.columns)
        f_i = list(zip(features, regressor.feature_importances_))
        f_i.sort(key=lambda x: x[1], reverse=True)

        rfe = RFECV(
            regressor,
            cv=config["cross_validation_number"],
            scoring="neg_mean_absolute_percentage_error",
        )
        rfe.fit(train_x, train_y)
        selected_features = list(np.array(features)[rfe.get_support()])
        print(selected_features)

        feat_importance = pd.DataFrame(
            f_i, columns=["metric", "Feature Importance"]
        )
        feat_importance.set_index("metric", inplace=True)
        print(feat_importance)

        # Compute SHAP values
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(train_x)
        shap_df = pd.DataFrame(
            np.abs(pd.DataFrame(shap_values, columns=train_x.columns)).mean(),
            columns=["shap values"],
        )
        print(
            f"{config['model_type']} SHAP importance",
            shap_df.sort_values(by="shap values", ascending=False),
        )

    return regressor, feat_importance, shap_df, search


def evaluate_model_performance(
    config,
    regressor,
    train_x,
    train_y,
    test_x,
    test_y,
    idvs,
    dv,
    feat_importance,
    shap_df,
    search,
    group,
    pillar,
):
    """
    Evaluate model performance and compute various metrics.

    Parameters
    ----------
    config (dict): Configuration dictionary containing model details and hyperparameters.
    regressor (object): Trained model.
    train_x (DataFrame): Training features.
    train_y (Series): Training target.
    test_x (DataFrame): Test features.
    test_y (Series): Test target.
    idvs (DataFrame): Independent variables used for the entire dataset.
    dv (Series): Dependent variable.
    feat_importance (DataFrame): Feature importance values.
    shap_df (DataFrame): SHAP feature importance values.
    search (GridSearchCV): Grid search object containing best parameters.

    Returns
    ----------
        results_all_model : DataFrame
            Model evaluation results and feature importance.
        actual_vs_predicted : DataFrame
            Actual vs predicted values for the full dataset.
    """
    # Generate Predictions
    y_pred_train = regressor.predict(train_x)
    y_pred_test = regressor.predict(test_x)
    y_pred_all = regressor.predict(idvs)

    # Compute Metrics
    mae_train = metrics.mean_absolute_error(train_y, y_pred_train)
    mse_train = metrics.mean_squared_error(train_y, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = metrics.r2_score(train_y, y_pred_train)
    mape_train = mean_absolute_percentage_error(train_y, y_pred_train)

    # Merge Feature Importance and SHAP values
    results_all_model = (
        pd.concat([feat_importance, shap_df], axis=1)
        .reset_index()
        .rename(
            columns={
                "Feature Importance": "feature_importance",
                "index": "shap_features",
                "shap values": "shap_values",
            }
        )
    )

    # Add model performance metrics
    results_all_model["model_type"] = config["model_type"]
    results_all_model["latest_dv"] = dv.values[-1]
    results_all_model["r2_score_train"] = r2_train
    results_all_model["mape_train"] = mape_train

    # Cross-validation metrics
    results_all_model["r2_score_fold"] = cross_val_score(
        regressor,
        train_x,
        train_y,
        cv=config["cross_validation_number"],
        scoring="r2",
    ).mean()
    results_all_model["mape_fold"] = (
        cross_val_score(
            regressor,
            train_x,
            train_y,
            cv=config["cross_validation_number"],
            scoring="neg_mean_absolute_percentage_error",
        ).mean()
        * -1
    )

    # Hold-out test set metrics
    results_all_model["r2_score_hold_out"] = metrics.r2_score(
        test_y, y_pred_test
    )
    results_all_model["mape_hold_out"] = mean_absolute_percentage_error(
        test_y, y_pred_test
    )

    # Overall dataset metrics
    results_all_model["r2_score_all"] = metrics.r2_score(dv, y_pred_all)
    results_all_model["mape_all"] = mean_absolute_percentage_error(
        dv, y_pred_all
    )

    # Store best parameters from grid search
    results_all_model["best_params_gridsearchcv"] = str(search.best_params_)

    # Create Actual vs Predicted DataFrame
    actual_vs_predicted = pd.DataFrame({"actual": dv, "predicted": y_pred_all})
    actual_vs_predicted["model_type"] = config["model_type"]

    for i, x in enumerate(config["data_prep_group_var"]):
        results_all_model[x] = group[i]
        actual_vs_predicted[x] = group[i]
    results_all_model["pillar"] = pillar
    actual_vs_predicted["pillar"] = pillar

    return results_all_model, actual_vs_predicted


def train_and_evaluate_group_models(config, scaled_data, idv_list, paths):
    """
    Function to train and evaluate models for each group in the scaled data,
    then return concatenated results.

    Args
    ----------
        config (dict): Configuration dictionary with necessary columns and parameters.
        scaled_data (pd.DataFrame): The input data to train and evaluate the models on.
        idv_list (pd.DataFrame): The individual variable data for different equity pillars.

    Returns
    ----------
        pd.DataFrame
            Concatenated results for both models and actual vs predicted values.
    """
    # Prepare the group list and initialize result containers
    group_list = [config["date_column"]] + config["data_prep_group_var"]
    final_rf_results = (
        []
    )  # To store results from all groups and equity pillars
    final_act_pred_results = []

    # Create the pillar to IDV dictionary
    pillar_idv_dict = (
        idv_list.groupby("equity_pillar")["idv"].apply(list).to_dict()
    )

    # Loop through each group in the scaled data
    for group, group_df in scaled_data.groupby(config["data_prep_group_var"]):
        print(f"Processing group: {group}")
        model_data = group_df.copy()
        model_data = model_data.pivot(
            index=group_list, columns="metric", values="value"
        ).reset_index()

        # Loop through each pillar
        for pillar in pillar_idv_dict.keys():
            if group == ("TEMPTATIONS", "CAT FOOD") and pillar == "advocacy":
                print(f"Processing pillar: {pillar}")
                # Filter columns available for the current pillar
                col_available = [
                    col for col in model_data if col in pillar_idv_dict[pillar]
                ]
                columns_model = (
                    group_list + col_available + [config["dv_column"]]
                )

                if col_available:  # Proceed only if there are relevant columns
                    # Filter data for CFA
                    print("Training model...")
                    model_idv_dv_df = model_data[columns_model]
                    train_x, test_x, train_y, test_y, idvs, dv = (
                        model_data_prep(model_idv_dv_df, col_available, config)
                    )

                    # Train and evaluate the model
                    regressor, feat_importance, shap_df, search = (
                        train_and_evaluate_model(config, train_x, train_y)
                    )
                    results_all_model, actual_vs_predicted = (
                        evaluate_model_performance(
                            config,
                            regressor,
                            train_x,
                            train_y,
                            test_x,
                            test_y,
                            idvs,
                            dv,
                            feat_importance,
                            shap_df,
                            search,
                            group,
                            pillar,
                        )
                    )

                    # Append results to the final lists
                    final_rf_results.append(results_all_model)
                    final_act_pred_results.append(actual_vs_predicted)

    # Concatenate the results
    rf_act_pred_columns_list = ["pillar", "actual", "predicted", "model_type"]

    rf_fit_columns_list = [
        "pillar",
        "metric",
        "feature_importance",
        "shap_values",
        "model_type",
        "latest_dv",
        "r2_score_train",
        "mape_train",
        "r2_score_fold",
        "mape_fold",
        "r2_score_hold_out",
        "mape_hold_out",
        "r2_score_all",
        "mape_all",
        "best_params_gridsearchcv",
        "brand",
        "category",
    ]
    rf_fit_col_arrangement = (
        config["data_prep_group_var"] + rf_fit_columns_list
    )
    rf_act_pred_arrangement = (
        config["data_prep_group_var"] + rf_act_pred_columns_list
    )
    final_rf_results_df = pd.concat(
        final_rf_results, axis=0, ignore_index=True
    )
    final_rf_results_df = final_rf_results_df[rf_fit_col_arrangement]

    final_act_pred_results_df = pd.concat(
        final_act_pred_results, axis=0, ignore_index=True
    )
    final_act_pred_results_df = final_act_pred_results_df[
        rf_act_pred_arrangement
    ]

    final_rf_results_df.to_csv(paths["rf_fit_data_path"], index=False)
    final_act_pred_results_df.to_csv(
        paths["rf_act_pred_data_path"], index=False
    )

    return final_rf_results_df, final_act_pred_results_df


# DS
def train_and_evaluate_group_models_parallel(
    config, scaled_data, idv_list, paths
):
    """
    Train and evaluate models for different groups in parallel.

    Args
    ----------
        config (dict): Configuration dictionary.
        scaled_data (pd.DataFrame): Scaled dataset containing metrics.
        idv_list (pd.DataFrame): List mapping independent variables to pillars.
        paths (dict): Dictionary containing file paths for saving results.

    Returns
    ----------
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Dataframe containing random forest model results.
            - Dataframe containing actual vs predicted values.
    """

    # Prepare data once upfront
    group_list = [config["date_column"]] + config["data_prep_group_var"]
    pillar_idv_dict = (
        idv_list.groupby("equity_pillar")["idv"].apply(list).to_dict()
    )

    # Pre-pivot entire dataset (critical optimization)
    pivoted_data = scaled_data.pivot(
        index=group_list, columns="metric", values="value"
    ).reset_index()

    # Parallel processing of groups
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(process_single_group)(
            group, group_df, config, pillar_idv_dict, group_list
        )
        for group, group_df in pivoted_data.groupby(
            config["data_prep_group_var"]
        )
    )

    # Combine results from all workers
    final_rf_results, final_act_pred_results = zip(*results)
    final_rf_results_df = pd.concat(final_rf_results, ignore_index=True)
    final_act_pred_results_df = pd.concat(
        final_act_pred_results, ignore_index=True
    )

    rf_act_pred_columns_list = ["pillar", "actual", "predicted", "model_type"]

    rf_fit_columns_list = [
        "pillar",
        "metric",
        "feature_importance",
        "shap_values",
        "model_type",
        "latest_dv",
        "r2_score_train",
        "mape_train",
        "r2_score_fold",
        "mape_fold",
        "r2_score_hold_out",
        "mape_hold_out",
        "r2_score_all",
        "mape_all",
        "best_params_gridsearchcv",
    ]
    rf_fit_col_arrangement = (
        config["data_prep_group_var"] + rf_fit_columns_list
    )
    rf_act_pred_arrangement = (
        config["data_prep_group_var"] + rf_act_pred_columns_list
    )

    final_rf_results_df = final_rf_results_df[rf_fit_col_arrangement]
    final_act_pred_results_df = final_act_pred_results_df[
        rf_act_pred_arrangement
    ]
    # Save results
    final_rf_results_df.to_csv(paths["rf_fit_data_path"], index=False)
    final_act_pred_results_df.to_csv(
        paths["rf_act_pred_data_path"], index=False
    )

    return final_rf_results_df, final_act_pred_results_df


def process_single_group(
    group, model_data, config, pillar_idv_dict, group_list
):
    """
    Process a single group in parallel for model training and evaluation.

    Args
    ----------
        group (tuple): The specific group being processed.
        model_data (pd.DataFrame): Data for the given group.
        config (dict): Configuration dictionary.
        pillar_idv_dict (dict): Mapping of pillars to their independent variables.
        group_list (list): List of grouping variables.

    Returns
    ----------
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Dataframe containing model evaluation results.
            - Dataframe containing actual vs predicted values.
    """

    print(f"Processing group: {group}")
    group_results = []
    group_act_pred = []
    model_data = model_data.dropna(axis=1, how="all")
    for pillar in pillar_idv_dict.keys():
        print(f"Processing pillar: {pillar}")
        col_available = [
            col for col in model_data if col in pillar_idv_dict[pillar]
        ]

        if not col_available:
            continue

        try:
            # Efficient data preparation
            model_idv_dv_df = model_data[
                group_list + col_available + [config["dv_column"]]
            ]
            train_x, test_x, train_y, test_y, idvs, dv = model_data_prep(
                model_idv_dv_df, col_available, config
            )

            # Optimized model training
            regressor, feat_importance, shap_df, search = (
                train_and_evaluate_model(config, train_x, train_y)
            )

            # Memory-efficient evaluation
            results, actual_pred = evaluate_model_performance(
                config,
                regressor,
                train_x,
                train_y,
                test_x,
                test_y,
                idvs,
                dv,
                feat_importance,
                shap_df,
                search,
                group,
                pillar,
            )

            group_results.append(results)
            group_act_pred.append(actual_pred)

        except Exception as e:
            print(f"Error processing {group}-{pillar}: {str(e)}")

        # Clean up memory
        del train_x, test_x, train_y, test_y
        gc.collect()

    return (
        (
            pd.concat(group_results, ignore_index=True)
            if group_results
            else pd.DataFrame()
        ),
        (
            pd.concat(group_act_pred, ignore_index=True)
            if group_act_pred
            else pd.DataFrame()
        ),
    )
