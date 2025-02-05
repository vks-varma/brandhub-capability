from cfa_modeling import cfa_py, perform_cfa_analysis
from data_preparation import data_prepare
from pillar_importance import (
    importance_run_parallel_processing,  # (Parallel process function)
)
from pillar_importance import scorecard_format
from rf_modeling import (
    train_and_evaluate_group_models_parallel,  # (Parallel process function)
)
from score import scoring

# from rf_modeling import  train_and_evaluate_group_models # (single process function)
# from pillar_importance import importance_train_and_evaluate_models  # (single process function)


def main():
    scaled_data, idv_list, config, paths = data_prepare()
    cfa_df = perform_cfa_analysis(scaled_data, idv_list, config, cfa_py, paths)
    rf_df, rf_act_pred_df = train_and_evaluate_group_models_parallel(
        config, scaled_data, idv_list, paths
    )
    pillar_weights, pillar_data, trend_past_data, scaled_score_data = scoring(
        cfa_df, rf_df, scaled_data, idv_list, config, paths
    )

    imp_rf_df, imp_rf_act_pred_df = importance_run_parallel_processing(
        scaled_data, trend_past_data, idv_list, config, paths
    )

    scorecard, pillar_relative_importance = scorecard_format(
        config,
        pillar_weights,
        scaled_data,
        scaled_score_data,
        imp_rf_df,
        paths,
    )


if __name__ == "__main__":
    main()
