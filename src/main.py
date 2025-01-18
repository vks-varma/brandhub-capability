# Databricks notebook source
# MAGIC %md
# MAGIC # Brand Heath Centre Code Module

# COMMAND ----------

# MAGIC %md
# MAGIC installation of packages/importing libraries

# COMMAND ----------

# MAGIC %run ./library_installation

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Main Function**

# COMMAND ----------

# MAGIC %run ./configuration_function

# COMMAND ----------

# MAGIC %run ./pre_validation

# COMMAND ----------


from configuration_function import *
from data_preparation import data_preparation
from modelling import modelling
from post_modelling import post_modelling, scoring
from pre_validation import pre_validation


# if  True:
def main(
    input_config,
    output_config,
    mapping_config,
    storage_options,
    refresh_config,
    feat_eng_config,
    filter_config,
    refresh_type,
):
    pre_validation(
        input_config,
        output_config,
        mapping_config,
        refresh_config,
        storage_options,
    )

    data_preparation(
        input_config,
        output_config,
        mapping_config,
        refresh_config,
        filter_config,
        storage_options,
        refresh_type,
    )

    # if scoring_refresh_check == False:
    modelling(
        input_config,
        output_config,
        mapping_config,
        refresh_config,
        feat_eng_config,
        filter_config,
        storage_options,
        refresh_type,
    )

    scoring(
        input_config,
        output_config,
        mapping_config,
        storage_options,
        refresh_config,
        feat_eng_config,
        filter_config,
        refresh_type,
    )

    post_modelling(
        input_config,
        output_config,
        mapping_config,
        storage_options,
        refresh_config,
        feat_eng_config,
        filter_config,
        refresh_type,
    )


if __name__ == "__main__":
    main(
        input_config,
        output_config,
        mapping_config,
        storage_options,
        refresh_config,
        feat_eng_config,
        filter_config,
        refresh_type,
    )
