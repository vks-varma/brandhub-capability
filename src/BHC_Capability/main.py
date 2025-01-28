from cfa_modeling import cfa_py, perform_cfa_analysis
from data_preparation import data_prepare


def main():
    scaled_data, idv_list, config, paths = data_prepare()
    cfa_df = perform_cfa_analysis(scaled_data, idv_list, config, cfa_py, paths)


if __name__ == "__main__":
    main()
