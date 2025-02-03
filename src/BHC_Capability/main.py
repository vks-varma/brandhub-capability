from cfa_modeling import cfa_py, perform_cfa_analysis
from data_preparation import data_prepare
from rf_modeling import (
    train_and_evaluate_group_models_parallel,  # (Parallel process function)
)

# from rf_modeling import  train_and_evaluate_group_models # (single process function)


def main():
    scaled_data, idv_list, config, paths = data_prepare()
    cfa_df = perform_cfa_analysis(scaled_data, idv_list, config, cfa_py, paths)
    rf_df, rf_act_pred_df = train_and_evaluate_group_models_parallel(
        config, scaled_data, idv_list, paths
    )


if __name__ == "__main__":
    main()
