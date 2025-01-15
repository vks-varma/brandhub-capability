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

if  pre_validation_check:
    pre_validation(input_config,output_config,mapping_config,refresh_config,storage_options)

# COMMAND ----------

import time

while True:
    print("Keeping the cluster alive...")
    time.sleep(1500)  # Sleep for 30 minutes (1800 seconds)

# COMMAND ----------

# MAGIC %run ./data_preparation

# COMMAND ----------

data_preparation(input_config,output_config,mapping_config,refresh_config,filter_config,storage_options, refresh_type)

# COMMAND ----------

# MAGIC %run ./modelling

# COMMAND ----------

# if scoring_refresh_check == False:
modelling(input_config,output_config,mapping_config,refresh_config, feat_eng_config,filter_config,storage_options, refresh_type)

# COMMAND ----------

# MAGIC %run ./post_modelling

# COMMAND ----------

scoring(input_config, output_config, mapping_config, storage_options, refresh_config, feat_eng_config, filter_config, refresh_type)

# COMMAND ----------

weights_sheet = pd.read_csv(output_config["pillar_creation"]["weights_sheet"], storage_options = storage_options)

# COMMAND ----------

index_df_long = pd.read_csv(output_config["pillar_creation"]["pillars_long_format"], storage_options = storage_options)

# COMMAND ----------

final_merged_df = pd.read_csv(output_config["trend_pillar"]["trend_pillars"], storage_options = storage_options)

# COMMAND ----------

scaled_scores = pd.read_csv(output_config["scaled_scores"]["scaled_pillars"], storage_options = storage_options)
scaled_scores_long = pd.read_csv(output_config["scaled_scores"]["scaled_pillars_long_format"], storage_options = storage_options)

# COMMAND ----------

scaled_scores.display()

# COMMAND ----------

scaled_scores_long.display()

# COMMAND ----------

post_modelling(input_config,output_config, mapping_config, storage_options, refresh_config, feat_eng_config, filter_config, refresh_type)


# COMMAND ----------

final_merged_df = pd.read_csv(output_config["trend_pillar"]["trend_pillars"], storage_options = storage_options)

# COMMAND ----------

if post_validation_check: 
    post_validation(input_config,output_config)

# COMMAND ----------


