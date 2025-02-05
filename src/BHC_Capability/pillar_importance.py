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


def importance_model_data_prep(
    model_idv_dv_df, col_available, config, test_size_ratio=0.1, shuffle=True
):
    """
    Prepare model data by splitting into training and testing sets.

    Parameters:
    model_idv_dv_df (DataFrame): The dataset containing independent and dependent variables.
    col_available (list): List of column names to be used as independent variables.
    config (dict): Configuration dictionary containing the dependent variable column name.
    test_size_ratio (float): Proportion of the dataset to be used as the test set.
    shuffle (bool): Whether to shuffle the data before splitting.

    Returns:
    tuple: train_x, test_x, train_y, test_y
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


def importance_train_model(config, train_x, train_y):
    """
    Train and evaluate a machine learning model based on the provided configuration.

    Parameters:
    config (dict): Configuration dictionary containing model details and hyperparameters.
    train_x (DataFrame): Training features.
    train_y (Series): Training target.

    Returns:
    regressor (object): Trained model.
    feat_importance (DataFrame): Feature importance values.
    feat_df (DataFrame): SHAP feature importance values.
    """
    # Select model class based on config
    if config["importance_model_type"] == "RandomForest":
        model_class = RandomForestRegressor
        params = {
            "max_depth": config["importance_model_config"]["RandomForest"][
                "grid_search"
            ]["max_depth"],
            "n_estimators": config["importance_model_config"]["RandomForest"][
                "grid_search"
            ]["n_estimators"],
            "max_features": config["importance_model_config"]["RandomForest"][
                "grid_search"
            ]["max_features"],
            "random_state": [
                config["importance_model_config"]["RandomForest"][
                    "grid_search"
                ]["random_state"]
            ],
        }
        model = RandomForestRegressor(
            random_state=[
                config["importance_model_config"]["RandomForest"][
                    "random_state"
                ]
            ]
        )
        grid_search = GridSearchCV(
            model,
            params,
            cv=config["cross_validation_number"],
            scoring=["r2", "neg_mean_absolute_percentage_error"],
            refit="neg_mean_absolute_percentage_error",
        )

    grid_search.fit(train_x, train_y)

    print(
        f"The best hyperparameters for {config['importance_model_type']} are {grid_search.best_params_}"
    )

    # Train model with best parameters
    regressor = model_class(**grid_search.best_params_)
    regressor.fit(train_x, train_y)

    # Feature importance and SHAP values (only for RandomForest)
    feat_importance = None
    shap_df = None
    if config["importance_model_type"] == "RandomForest":
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

    return regressor, feat_importance, shap_df, grid_search


def importance_evaluate_model_performance(
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
):
    """
    Evaluate model performance and compute various metrics.

    Parameters:
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

    Returns:
    results_all_model (DataFrame): Model evaluation results and feature importance.
    actual_vs_predicted (DataFrame): Actual vs predicted values for the full dataset.
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

    return results_all_model, actual_vs_predicted


def importance_train_and_evaluate_models(
    scaled_data, trend_data, idv_list, config
):
    """Train and evaluate models sequentially and store results in DataFrames."""

    # Prepare dependent variable data
    dv_data = (
        scaled_data[scaled_data["metric"] == config["dv_column"]]
        .pivot(
            index=[config["date_column"]] + config["data_prep_group_var"],
            columns="metric",
            values="value",
        )
        .reset_index()
    )

    # Prepare pillar trend data
    pillar_pivot = trend_data.pivot(
        index=[config["date_column"]] + config["data_prep_group_var"],
        columns="pillar",
        values="trend_past",
    ).reset_index()

    # Merge datasets
    model_dv_idv_df = pd.merge(
        pillar_pivot,
        dv_data,
        on=[config["date_column"]] + config["data_prep_group_var"],
        how="inner",
    )

    # Identify available importance columns
    importance_col_available = idv_list["equity_pillar"].unique()

    final_rf_results = []
    final_act_pred_results = []

    for group, group_df in model_dv_idv_df.groupby(
        config["data_prep_group_var"]
    ):
        train_x, test_x, train_y, test_y, idvs, dv = (
            importance_model_data_prep(
                group_df, importance_col_available, config
            )
        )
        regressor, feat_importance, shap_df, search = importance_train_model(
            config, train_x, train_y
        )
        results_all_model, actual_vs_predicted = (
            importance_evaluate_model_performance(
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
            )
        )

        final_rf_results.append(results_all_model)
        final_act_pred_results.append(actual_vs_predicted)

    # Concatenating all results into single DataFrames
    final_rf_results_df = pd.concat(final_rf_results, ignore_index=True)
    final_act_pred_results_df = pd.concat(
        final_act_pred_results, ignore_index=True
    )

    return final_rf_results_df, final_act_pred_results_df


def importance_process_group(
    group_df, group, importance_col_available, config
):
    """Process a single group to train and evaluate the model."""
    # Prepare data
    train_x, test_x, train_y, test_y, idvs, dv = importance_model_data_prep(
        group_df, importance_col_available, config
    )
    # Train model
    regressor, feat_importance, shap_df, search = importance_train_model(
        config, train_x, train_y
    )
    # Evaluate performance
    results_all_model, actual_vs_predicted = (
        importance_evaluate_model_performance(
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
        )
    )
    return results_all_model, actual_vs_predicted


def importance_run_parallel_processing(
    scaled_data, trend_data, idv_list, config, paths
):
    """
    Execute the data preparation and model training/evaluation
    with parallel processing across groups.
    """
    # Prepare dependent variable data
    dv_data = (
        scaled_data[scaled_data["metric"] == config["dv_column"]]
        .pivot(
            index=[config["date_column"]] + config["data_prep_group_var"],
            columns="metric",
            values="value",
        )
        .reset_index()
    )

    # Prepare pillar trend data
    pillar_pivot = trend_data.pivot(
        index=[config["date_column"]] + config["data_prep_group_var"],
        columns="pillar",
        values="trend_past",
    ).reset_index()

    # Merge datasets
    model_dv_idv_df = pd.merge(
        pillar_pivot,
        dv_data,
        on=[config["date_column"]] + config["data_prep_group_var"],
        how="inner",
    )

    # Identify available importance columns
    importance_col_available = idv_list["equity_pillar"].unique()

    # Parallel execution across groups
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(importance_process_group)(
            group_df, group, importance_col_available, config
        )
        for group, group_df in model_dv_idv_df.groupby(
            config["data_prep_group_var"]
        )
    )

    # Unpack results
    final_rf_results, final_act_pred_results = zip(*results)

    final_rf_results_df = pd.concat(final_rf_results, ignore_index=True)
    final_act_pred_results_df = pd.concat(
        final_act_pred_results, ignore_index=True
    )

    # Save results
    final_rf_results_df.to_csv(paths["imp_rf_fit_data_path"], index=False)
    final_act_pred_results_df.to_csv(
        paths["imp_rf_act_pred_data_path"], index=False
    )

    return final_rf_results_df, final_act_pred_results_df


def scorecard_format(
    config, pillar_weights, scaled_data, scaled_pillar_data, imp_rf_df, paths
):
    """
    Computes the score card and importance model results.

    Args:
        config (dict): Configuration settings.
        weight_scores (pd.DataFrame): DataFrame containing weights for metrics.
        scaled_data (pd.DataFrame): DataFrame with scaled values.
        scaled_pillar_data (pd.DataFrame): DataFrame containing pillar data.
        imp_rf_df (pd.DataFrame): DataFrame containing model importance results.

    Returns:
        score_card_final_df (pd.DataFrame): Final score card with calculated metric contributions.
        filtered_imp_model_results (pd.DataFrame): Processed importance model results with relative importance.
    """

    # Merge weight scores with scaled data
    score_card_df = pd.merge(
        pillar_weights,
        scaled_data,
        on=config["data_prep_group_var"] + ["metric"],
        how="inner",  # Use 'inner' to drop metrics without weights
    )

    # Compute metric contribution
    score_card_df["metric_contribution"] = (
        score_card_df["value"] * score_card_df["weight"]
    )

    # Convert date column to datetime format and extract year/month
    score_card_df[config["date_column"]] = pd.to_datetime(
        score_card_df["date"], format="%Y-%m-%d"
    )
    score_card_df["year"] = score_card_df["date"].dt.year
    score_card_df["month"] = score_card_df["date"].dt.month

    # Merge with scaled pillar data
    score_card_final_df = pd.merge(
        score_card_df, scaled_pillar_data, how="inner"
    )

    # Filter model importance results
    filtered_imp_model_results = imp_rf_df[
        imp_rf_df["model_type"] == config["importance_model_type"]
    ][["brand", "category", "shap_features", "shap_values"]].copy()

    # Calculate the sum of SHAP values and relative importance
    sum_shap_values = filtered_imp_model_results.groupby(
        config["data_prep_group_var"]
    )["shap_values"].transform("sum")
    filtered_imp_model_results["relative_importance"] = (
        filtered_imp_model_results["shap_values"] / sum_shap_values
    )

    score_card_final_df.to_csv(paths["score_card_final_df_path"], index=False)
    filtered_imp_model_results.to_csv(
        paths["relative_imp_model_results_path"], index=False
    )

    return score_card_final_df, filtered_imp_model_results
