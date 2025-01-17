# Databricks notebook source
# MAGIC %run ./library_installation

import os

# COMMAND ----------
from datetime import datetime

platform_type = "local"

# Define interactive widgets in Databricks to accept user inputs for key configuration parameters.
# These widgets allow dynamic parameterization of the notebook, making it reusable for different configurations.

if platform_type == "databricks":
    dbutils.widgets.text(
        "account_name", "npusdvdatalakesta"
    )  # Widget for specifying the storage account name.
    dbutils.widgets.text(
        "keyvault_name", "npusdvbrandhubidkey"
    )  # Widget for specifying the Azure Key Vault name.
    dbutils.widgets.text(
        "refresh_type", "model_refresh"
    )  # Widget for defining the type of refresh (e.g., model scoring/model refresh).
    dbutils.widgets.text(
        "pre_validation_check", "True"
    )  # Widget for enabling or disabling pre-validation checks.
    dbutils.widgets.text(
        "scoring_refresh_check", "True"
    )  # Widget for enabling or disabling scoring refresh checks.
    dbutils.widgets.text(
        "post_validation_check", "True"
    )  # Widget for enabling or disabling post-validation checks.

    # COMMAND ----------

    # Retrieve the values entered or selected in the widgets.
    # These values will be used in the subsequent logic to control the notebook's behavior dynamically.

    account_name = dbutils.widgets.get(
        "account_name"
    )  # Fetch the storage account name from the widget.
    keyvault_name = dbutils.widgets.get(
        "keyvault_name"
    )  # Fetch the Azure Key Vault name from the widget.
    refresh_type = dbutils.widgets.get(
        "refresh_type"
    )  # Fetch the type of refresh selected by the user.
    pre_validation_check = dbutils.widgets.get(
        "pre_validation_check"
    )  # Retrieve the pre-validation check setting (True/False).
    scoring_refresh_check = dbutils.widgets.get(
        "scoring_refresh_check"
    )  # Retrieve the scoring refresh check setting (True/False).
    post_validation_check = dbutils.widgets.get(
        "post_validation_check"
    )  # Retrieve the post-validation check setting (True/False).

    # COMMAND ----------

    client_secret = dbutils.secrets.get(
        scope=keyvault_name, key="brandhub-spn-secret"
    )
    client_id = dbutils.secrets.get(scope=keyvault_name, key="brandhub-spn-id")
    tenant_id = dbutils.secrets.get(scope=keyvault_name, key="tenant-id")

    dbs_sql_hostname = dbutils.secrets.get(
        scope=keyvault_name, key="nppc-dh-databricks-hostname"
    )
    dbs_sql_http_path = dbutils.secrets.get(
        scope=keyvault_name, key="nppc-dh-databricks-http-path"
    )
    dbs_sql_token = dbutils.secrets.get(
        scope=keyvault_name, key="nppc-dh-databricks-token"
    )

    Adlsg2_authentication(
        account_name, keyvault_name, client_secret, client_id, tenant_id
    )

    # COMMAND ----------

    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    fs = AzureBlobFileSystem(account_name=account_name, credential=credential)
else:
    account_name = None  # Fetch the storage account name from the widget.
    keyvault_name = None  # Fetch the Azure Key Vault name from the widget.
    refresh_type = "model_refresh"  # Fetch the type of refresh selected by the user.(model_refresh/model_scoring)
    pre_validation_check = (
        True  # Retrieve the pre-validation check setting (True/False).
    )
    scoring_refresh_check = (
        False  # Retrieve the scoring refresh check setting (True/False).
    )
    post_validation_check = (
        True  # Retrieve the post-validation check setting (True/False).
    )

    # COMMAND ----------

    client_secret = None
    client_id = None
    tenant_id = None

    dbs_sql_hostname = None
    dbs_sql_http_path = None
    dbs_sql_token = None

    # COMMAND ----------

    credential = None
    fs = None


# COMMAND ----------

inputdata_blob_name = "restricted-published"
dataoperations_name = "restricted-dataoperations"
outputdata_blob_name = "general-published"
current_date = datetime.now().replace(day=1).strftime("%Y-%m-%d")
prev_date = "2024-11-01"
dv_folder = "scorecard_refresh"

time_granularity = "monthly"
group_vars = ["vendor", "brand_group_expanded", "category", "date"]


if platform_type == "databricks":
    main_dir = (
        f"abfss://{dataoperations_name}@{account_name}.dfs.core.windows.net/"
    )
    storage_options = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    sql_options = {
        "dbs_sql_hostname": dbs_sql_hostname,
        "http_path": dbs_sql_http_path,
        "token": dbs_sql_token,
    }
else:
    main_dir = "../"
    storage_options = None
    sql_options = None

apps_dir = "apps/cmi_brand_hub/"
staging_dir = "staging/cmi_brand_hub/"
solution_dir = "solution/cmi_brand_hub/"

# COMMAND ----------

directory_config = {
    "support_dir": f"{main_dir}{apps_dir}{'brand_health_scorecard_mapping/'}",
    "prev_staging_dir": f"{main_dir}{staging_dir}{dv_folder}/{prev_date}/",
    "current_staging_dir": f"{main_dir}{staging_dir}{dv_folder}/{current_date}/",
}

# COMMAND ----------

mapping_config = {
    "metrics_rename_mapping": f"{directory_config['support_dir']}/column_name_mapping_file_updated.csv",
    "inverse_logic_mapping": f"{directory_config['support_dir']}/inverse_metrics_creation_logic.csv",
    "idv_list": f"{directory_config['support_dir']}/idv_list.csv",
    "brand_list": f"{directory_config['support_dir']}/brand_select_list_updated.csv",
    "dashboard_metric_names_mapping": f"{directory_config['support_dir']}/dashborad_metric_names_mapping.xlsx",
    "price_class_mapping": f"{directory_config['support_dir']}/price_class_mapping.csv",
    "panel_rename_mapping": f"{directory_config['support_dir']}/panel_new_names_mapping.csv",
}


# COMMAND ----------

filter_config = {
    "high_null_count_categories": ["CAT TREATS ONLY", "DOG TREATS ONLY"],
    "scaling": {
        "transform_using_standard_scaler": False,
        "transform_using_min_max_scaler": False,
        "custom_min_max": True,
        "cap_min_max_from_idv_file": True,
    },
    "equal_weightage": {
        "give_awareness_metrics_equal_weightage": True,
        "give_advocacy_metrics_equal_weightage": False,
    },
    "scaled_score": {"only_pillars": True},
}

# COMMAND ----------

refresh_config = {
    "dv": "market_share",
    "time_granularity": time_granularity,
    "platform": platform_type,
    "sql_options": sql_options,
    "start_date": "2019-09-01",
    "end_date": "2023-12-30",
    "run_importance_model_for_scoring_refresh": False,
    "pillars": {
        "all_category_pillars": [
            "awareness_pillar",
            "loyalty_pillar",
            "advocacy_pillar",
            "consideration_pillar",
        ],
        "by_category_pillars": [
            "brand_perceptions_pillar",
            "product_feedback_pillar",
        ],
    },
    "weights_models": {
        "CFA": {"run": True},  # only True for now
        "RandomForest": {"run": True},  # default
        "XGBoost": {"run": False},
        # "RF_Ridge": {
        #     "run": False
        # },
        # "Corr_ridge": {
        #     "run": False
        # },
        # "Brute_force": {
        #     "run": False
        # }
    },
    "importance_model": {
        "RandomForest": {"run": True},  # default
        "XGBoost": {"run": False},
    },
}

# COMMAND ----------

input_config = {
    "current_sales_data": f"{directory_config['current_staging_dir']}raw_input_data/rms_tenten_monthly_19_9_24_blue.csv",
    "prev_sales_data": f"{directory_config['prev_staging_dir']}raw_input_data/nielsen_rms_data.csv",
    "current_harmonized_data": f"{directory_config['current_staging_dir']}raw_input_data/harmonized_data_rms_tenten_blue_19_9_24_monthly.csv",
    "prev_harmonized_data": f"{directory_config['prev_staging_dir']}processed_input_data/harmonized_data_processed.csv",
}

# COMMAND ----------

if platform_type == "databricks":
    dbutils.fs.mkdirs(
        f"{directory_config['current_staging_dir']}/processed_input_data"
    )
    dbutils.fs.mkdirs(
        f"{directory_config['current_staging_dir']}/raw_input_data"
    )
    dbutils.fs.mkdirs(f"{directory_config['current_staging_dir']}/data_prep")
    dbutils.fs.mkdirs(f"{directory_config['current_staging_dir']}/cfa")
    dbutils.fs.mkdirs(
        f"{directory_config['current_staging_dir']}/weights_model"
    )
    dbutils.fs.mkdirs(
        f"{directory_config['current_staging_dir']}/pillar_creation"
    )
    dbutils.fs.mkdirs(
        f"{directory_config['current_staging_dir']}/trend_pillar"
    )
    dbutils.fs.mkdirs(
        f"{directory_config['current_staging_dir']}/importance_model"
    )
    dbutils.fs.mkdirs(
        f"{directory_config['current_staging_dir']}/scaled_scores"
    )
    dbutils.fs.mkdirs(
        f"{directory_config['current_staging_dir']}/pillar_importances"
    )
    dbutils.fs.mkdirs(f"{directory_config['current_staging_dir']}/scorecard")
    dbutils.fs.mkdirs(
        f"{directory_config['current_staging_dir']}/updated_scorecard"
    )

    print("output folders created:", directory_config["current_staging_dir"])

if platform_type == "local":

    # Local directory creation using the same format as Databricks code
    os.makedirs(
        f"{directory_config['current_staging_dir']}/processed_input_data",
        exist_ok=True,
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/raw_input_data",
        exist_ok=True,
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/data_prep", exist_ok=True
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/cfa", exist_ok=True
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/weights_model",
        exist_ok=True,
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/pillar_creation",
        exist_ok=True,
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/trend_pillar",
        exist_ok=True,
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/importance_model",
        exist_ok=True,
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/scaled_scores",
        exist_ok=True,
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/pillar_importances",
        exist_ok=True,
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/scorecard", exist_ok=True
    )
    os.makedirs(
        f"{directory_config['current_staging_dir']}/updated_scorecard",
        exist_ok=True,
    )

    print("Output folders created:", directory_config["current_staging_dir"])

# COMMAND ----------

print("config - staging_output_path:", directory_config["current_staging_dir"])
output_config = {
    "processed_input_data": f"{directory_config['current_staging_dir']}/processed_input_data/harmonized_data_processed.csv",
    "raw_input_data": f"{directory_config['current_staging_dir']}/raw_input_data/harmonized_data_raw.csv",
    "processed_sales_data": f"{directory_config['current_staging_dir']}/processed_input_data/neilsen_sales_data.csv",
    "data_prep": {
        "eq_sub_scale": f"{directory_config['current_staging_dir']}/data_prep/eq_sub_scale_merged_brand_g_new.csv",
        "modeling_data": f"{directory_config['current_staging_dir']}/data_prep/modeling_data.csv",
        "equity_dt": f"{directory_config['current_staging_dir']}/data_prep/equity_dt.csv",
    },
    "cfa": {
        "model_results_all_category": f"{directory_config['current_staging_dir']}/cfa/cfa_fit_summary_all_category.csv",
        "model_results_by_category": f"{directory_config['current_staging_dir']}/cfa/cfa_fit_summary_all_brand.csv",
    },
    "weights_model": {
        "model_results": f"{directory_config['current_staging_dir']}/weights_model/model_results_all_cat.csv",
        "corr_results": f"{directory_config['current_staging_dir']}/weights_model/corr_results_all_cat.csv",
    },
    "pillar_creation": {
        "weights_sheet": f"{directory_config['current_staging_dir']}/pillar_creation/weights_sheet.csv",
        "pillars": f"{directory_config['current_staging_dir']}/pillar_creation/pillars.csv",
        "pillars_long_format": f"{directory_config['current_staging_dir']}/pillar_creation/pillars_long_format.csv",
    },
    "trend_pillar": {
        "trend_pillars": f"{directory_config['current_staging_dir']}/trend_pillar/trend_pillar_data.csv",
        "trend_pillars_long_format": f"{directory_config['current_staging_dir']}/trend_pillar/trend_pillar_long_format.csv",
    },
    "importance_model": {
        "model_results": f"{directory_config['current_staging_dir']}/importance_model/model_results_all_cat.csv",
    },
    "scaled_scores": {
        "scaled_pillars": f"{directory_config['current_staging_dir']}/scaled_scores/scaled_pillars.csv",
        "scaled_pillars_long_format": f"{directory_config['current_staging_dir']}/scaled_scores/scaled_pillars_long_format.csv",
    },
    "pillar_importances": {
        "pillar_importances": f"{directory_config['current_staging_dir']}/pillar_importances/relative_importance_shapely_values_new.csv",
        "hub_data": f"{directory_config['current_staging_dir']}/pillar_importances/hub_data.csv",
        "variable_mapping": f"{directory_config['current_staging_dir']}/pillar_importances/var_map.csv",
    },
    "scorecard": {
        "detailed": f"{directory_config['current_staging_dir']}/scorecard/brand_health_scorecard_detailed.csv",
        "summary": f"{directory_config['current_staging_dir']}/scorecard/brand_health_scorecard_summary.csv",
        "pillar_importances": f"{directory_config['current_staging_dir']}/scorecard/brand_health_pillar_relative_importance.csv",
    },
    "updated_scorecard": {
        "updated_summary": f"{directory_config['current_staging_dir']}/scorecard/brand_health_scorecard_summary.csv",
        "updated_pillar_importances": f"{directory_config['current_staging_dir']}/scorecard/brand_health_pillar_relative_importance.csv",
        "post_validation_report": f"{directory_config['current_staging_dir']}/scorecard/post_validation_report.html",
        "pre_validation_report": f"{directory_config['current_staging_dir']}/scorecard/pre_validation_report.html",
    },
}


# COMMAND ----------

feat_eng_config = {
    "weights_model": {
        "DV": (
            "market_share_total_sales"
            if refresh_config["dv"] == "market_share"
            else refresh_config["dv"]
        ),
        "Temp": "new_brand",
        "hyperparameters": {
            "RandomForest": {
                "grid_search": {
                    "max_depth": [2, 3, 4],
                    "n_estimators": [15, 50, 100, 300, 500],
                    "max_features": [2, 4, 10],
                    "random_state": [42],
                    "eval_metrics": [],
                },
                "random_state": 42,
            },
            "XGBoost": {
                "grid_search": {
                    "max_depth": [2, 3],
                    "n_estimators": [100, 500, 1000, 1500],
                    "learning_rate": [0.01, 0.02],
                    "random_state": [42],
                    "eval_metrics": [],
                },
                "random_state": 42,
            },
            "RF_Ridge": {
                "grid_search": {
                    "alpha": [0.001, 0.001, 0.01, 1, 5, 10, 15],
                    "random_state": [42],
                },
                "positive": False,
                "random_state": 42,
            },
            "Corr_ridge": {
                "grid_search": {
                    "alpha": [0.001, 0.01, 0.1, 1, 5, 10],
                    "random_state": [42],
                },
                "positive": False,
                "random_state": 42,
            },
            "Brute_force": {
                "grid_search": {
                    "alpha": [0.001, 0.01, 0.1, 1, 5, 10],
                    "random_state": [42],
                },
                "positive": False,
                "random_state": 42,
            },
        },
        "cross_validation_number": 4,  # Model Cross validation score
        "P_N_check": False,
        "Counter_Intuitive_cut_off": 0.30,
        "Time_series_split": False,  # For spliting the train and test data
        "Random_seed_split": True,
        "PCA_Transform": False,  # PCA Transformation on model Data
        "Corr_file_generation": True,  # Whether we need correlation file to be generated
        "log_convert_DV": False,  # Log convertion on DV. Set this to True if log(DV) is to be taken. Else keep False
        "is_lag_considered": False,
        "Cols_force_sel": False,  # Force to select the columns for modeling - only the columns in the below list will be used for modeling
        "pillars_list": [
            pillar.replace("_pillar", "")
            for pillar in (
                refresh_config["pillars"]["all_category_pillars"]
                + refresh_config["pillars"]["by_category_pillars"]
            )
        ],
        "Force_cols": [
            "rms_acv_selling_rt",
            "average_price_rt",
            "daily_users_pet_owners_mc",
            "infrequent_users_pet_owners_mc",
            "nps_detractors_pet_owners_mc",
            "nps_passives_pet_owners_mc",
            "nps_promoters_pet_owners_mc",
            "total_aware_all_respondents_mc",
            "total_buzz_pet_owners_mc",
            "total_considering_pet_owners_mc",
            "total_distrust_pet_owners_mc",
            "total_favorable_pet_owners_mc",
            "total_good_value_pet_owners_mc",
            "total_not_considering_pet_owners_mc",
            "total_poor_value_pet_owners_mc",
            "total_trust_pet_owners_mc",
            "total_unfavorable_pet_owners_mc",
            "weekly_users_pet_owners_mc",
        ],
        "Force_cols_net": [
            "rms_acv_selling_rt",
            "average_price_rt",
            "daily_users_pet_owners_mc",
            "infrequent_users_pet_owners_mc",
            "net_buzz_pet_owners_mc",
            "net_favorability_pet_owners_mc",
            "net_promoter_score_pet_owners_mc",
            "net_purchase_consideration_pet_owners_mc",
            "net_trust_pet_owners_mc",
            "net_value_pet_owners_mc",
            "total_aware_all_respondents_mc",
        ],
        "standardize": False,  # Whether we need to standardize the data
        "drop_mean_attributes": False,  # Droping of Rank and mean features for modeling
        "drop_rank1st_attributes": False,
    },
    "cfa": {
        "perform_weighted_average_DV": True,
        "pillars": {
            "by_category_pillars": refresh_config["pillars"][
                "by_category_pillars"
            ],
            "all_category_pillars": refresh_config["pillars"][
                "all_category_pillars"
            ],
        },
        "exclude_pillars": ["advocacy_pillar"],
        "std_lv": True,
        "check_gradient": False,
        "standardized_": True,
        "fit_measures": True,
        "sample_seeds": [2, 3, 5, 7, 11, 13, 17, 19],
    },
    "importance_model": {
        "DV": (
            "market_share_total_sales"
            if refresh_config["dv"] == "market_share"
            else refresh_config["dv"]
        ),
        "cross_validation_number": 3,
        "price_and_acv_added": False,
        "Time_series_split": False,
        "Random_seed_split": True,
        "lags_added": False,
        "log_convert_DV": False,
        "Corr_file_generation": True,
        "standardize": True,
        "DV": refresh_config["dv"],
        "hyperparameters": {
            "RandomForest": {
                "grid_search": {
                    "max_depth": [2, 3, 4, 5, 6],
                    "n_estimators": [15, 50, 100, 500],
                    "max_features": [2, 4, 10],
                    "random_state": [42],
                },
                "random_state": 42,
            },
            "XGBoost": {
                "grid_search": {
                    "max_depth": [2, 3],
                    "n_estimators": [500, 1000, 1500],
                    "learning_rate": [0.01, 0.005],
                    "random_state": [42],
                },
                "random_state": 42,
            },
            "Brute_force": {
                "grid_search": {
                    "alpha": [0.001, 0.01, 0.1, 1, 5, 10],
                    "random_state": [42],
                },
                "positive": True,
                "random_state": 42,
            },
        },
    },
}

print("config file complete")
# COMMAND ----------

# %run ./pre_validation

# COMMAND ----------

# %run ./data_preparation

# COMMAND ----------

# %run ./modelling

# COMMAND ----------

# %run ./post_modelling

# COMMAND ----------

# %run ./post_validation

# COMMAND ----------


# COMMAND ----------

# MAGIC %md
# MAGIC **Guidelines for Writing Production Grade for BHC/MMX**
# MAGIC
# MAGIC To ensure consistency and maintain best practices, please follow these steps when working on brand hub tasks:
# MAGIC (you will better understand these, once you have looked at latest BHC codes)
# MAGIC
# MAGIC Possible Conditions
# MAGIC   - time_granularity = 'weekly' or 'monthly'
# MAGIC   - refresh_type = 'model_scoring' or 'model_refresh'
# MAGIC
# MAGIC 1. **Time Granularity**
# MAGIC    - Always set `time_granularity` to either `monthly` or `weekly`.
# MAGIC    - Use lowercase letters with underscores as separators (e.g., `monthly`, `weekly`).
# MAGIC
# MAGIC 2. **File Configuration**
# MAGIC    - Declare all files created in a previous stage and used in the next stage under `input_config`.
# MAGIC    - Any common files that remain unchanged (e.g., mappings) should be declared under `mapping_config`.
# MAGIC    - Place all output files under `output_config`.
# MAGIC    - Pass `storage_options` as an additional layer for file handling.
# MAGIC
# MAGIC 3. **Refresh Configuration**
# MAGIC    - Place all refresh-related configurations under `refresh_config` (e.g., `start_date`, `end_date`, `time_granularity`, `scoring`, `modelling`).
# MAGIC
# MAGIC 4. **Feature Engineering and Additional Configurations**
# MAGIC    - If additional configurations are needed (e.g., for data preparation, feature engineering, or modelling), feel free to add them.
# MAGIC    - Append new configurations at the end of the flow, maintaining the order: `input_config`, `output_config`, `mapping_config`, `refresh_config`, and `feat_eng_config`.
# MAGIC
# MAGIC 5. **Code Structure**
# MAGIC    - Ensure that functions do not exceed 100–150 lines of code.
# MAGIC    - Avoid using loops within functions whenever possible; use vectorization techniques instead.
# MAGIC    - For running operations across multiple entities (e.g., brands or categories), use loops to call the function externally rather than embedding loops within the function.
# MAGIC
# MAGIC 6. **Intermediate and Final Data Guidelines**
# MAGIC    - Ensure the following structure for data:
# MAGIC      - All keys should be placed on the left, ordered hierarchically (e.g., `vendor x brand x sub_brand x category`).
# MAGIC      - Independent variables (IDVs) should be on the right.
# MAGIC      - Date column should be seperate these (e.g., `brand x category x date x eq_volume x amazon_spends ...`).
# MAGIC
# MAGIC 7. **Key or Group Variables**
# MAGIC    - For tasks involving model iterations (e.g., running a model for `brand x category x pillar`), create a key or `group_var`.
# MAGIC    - Pass this variable into the code to enable automatic configuration for future iterations.
# MAGIC    - If implementing this is overly complex, it can be skipped.
# MAGIC
# MAGIC By following these guidelines, we can ensure clarity, consistency, and scalability in our processes.
# MAGIC
# MAGIC

# COMMAND ----------
