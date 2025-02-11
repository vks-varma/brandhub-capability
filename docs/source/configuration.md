# Configuration File Guide

## Overview
This configuration file (`config.yml`) defines the parameters for data preparation, modeling, scoring, and output generation for the Brand Hub Project. The settings control data processing, model training, hyperparameter tuning, and result storage.

---

## Data Preparation
### Input Files:
- **`input_data`**: Path to the harmonized dataset.
- **`idv_list`**: Path to the independent variable list.

### Data Processing:
- **`granularity`**: Defines the time granularity for data processing (`weekly` or `monthly`).
- **`data_prep_group_var`**: List of variables used for grouping data (vendor, brand, category).
- **`date_column`**: Name of the date column in the dataset.
- **`date_format`**: Expected format of the date column.
- **`start_date`**: Start date for data filtering.
- **`end_date`**: End date for data filtering.
- **`dv_column`**: Dependent variable (market_share) used in modeling.
- **`null_percentage`**: Threshold for allowed null values in the dataset before imputation.
- **`scaling`**: Defines scaling type (`custom`).

---

## CFA Modeling
- **`cfa_sampling_seeding`**: List of seed values used for CFA model sampling.

---

## Random Forest (RF) Model
- **`model_type`**: Defines the primary model (`RandomForest`).
- **`model_config`**:
  - **RandomForest**: Hyperparameter tuning settings, including `max_depth`, `n_estimators`, and `max_features`.
  - **XGBoost**: Grid search parameters for `max_depth`, `n_estimators`, `learning_rate`.
  - **RF_Ridge**, **Corr_ridge**, **Brute_force**: Ridge regression settings with hyperparameters (`alpha`).
- **`cross_validation_number`**: Number of cross-validation folds.

---

## Scoring
- **`cfa_seed`**: Seed value for CFA scoring.
- **`cfa_target_col`**: Target column for CFA scoring (`est.std`).
- **`rf_target_col`**: Target column for RF scoring (`shap_values`).
- **`trend_past_rolling_window`**: Rolling window size for trend analysis.
- **`scale_level`**: Defines the level at which data is scaled (`category`).

---

## Importance Modeling
- **`importance_model_type`**: Specifies model type for feature importance analysis (`RandomForest`).
- **`importance_model_config`**:
  - **RandomForest**: Hyperparameters for importance modeling, including `max_depth`, `n_estimators`, and `max_features`.

---

## Output Paths
### Root & Output Directories:
- **`root_path`**: Root directory for all project files.
- **`output_folder`**: Folder where processed outputs will be stored.

### Specific Output Files:
- `filtered_data.csv`: Data after initial filtering.
- `no_null_imputed_data.csv`: Data after null value imputation.
- `scaled_data.csv`: Scaled dataset.
- `cfa_fit_data.csv`: CFA model fitted data.
- `rf_fit_data.csv`: Random Forest model fitted data.
- `rf_act_pred_data.csv`: Random Forest actual vs. predicted values.
- `pillar_weights.csv`: Weights assigned to different pillars.
- `pillar_data.csv`: Pillar-wise processed data.
- `trend_data.csv`: Trend analysis data.
- `scaled_score_data.csv`: Scaled scoring output.
- `imp_rf_fit_data.csv`: Feature importance RF fitted data.
- `imp_rf_act_pred_data.csv`: Importance model actual vs. predicted values.
- `score_card_final_df.csv`: Final scorecard data.
- `relative_imp_model_results.csv`: Feature importance results.

---

## Usage Instructions
1. **Modify Configuration**: Update paths and parameters as needed.
2. **Run Data Preparation**: Process data according to the defined parameters.
3. **Train Models**: Use specified configurations for training RF and CFA models.
4. **Score and Analyze**: Perform scoring and importance analysis.
5. **Review Outputs**: Check generated files in the output directory.

---

## Notes
- Ensure that file paths are correct and accessible.
- Adjust hyperparameters based on dataset size and computational capacity.
- Model selection can be changed in `model_type`.

This configuration provides flexibility for end-to-end data preparation, modeling, and result storage.

