# Databricks notebook source
# MAGIC %md
# MAGIC **Installing Libraries**

# COMMAND ----------

# MAGIC %pip install fsspec==2024.10.0
# MAGIC %pip install xlrd==2.0.1
# MAGIC %pip install adal==1.2.7
# MAGIC %pip install adlfs==2024.7.0
# MAGIC %pip install numpy==1.22
# MAGIC # %pip install scikit-learn==1.6.0
# MAGIC # %pip install shap==0.46.0
# MAGIC %pip install xgboost==2.1.3
# MAGIC %pip install rpy2==3.5.17
# MAGIC %pip install openpyxl==3.1.5
# MAGIC %pip install databricks-sql-connector==3.6.0
# MAGIC # Install specific versions
# MAGIC %pip install scikit-learn==1.4.2 shap==0.45.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC **Importing packages**

# COMMAND ----------

import calendar
import configparser
import glob
import json
import math

# Standard library imports
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from calendar import month_abbr
from datetime import datetime, timedelta
from functools import reduce
from itertools import chain

# External library imports
import fsspec
import matplotlib.pyplot as plt
import numpy as np

# Data science and numerical computation
import pandas as pd
import pyodbc

# import rpy2.robjects as ro
import seaborn as sns
import shap
import xgboost as xgb
from matplotlib import pyplot

# R and Python integration
# from rpy2 import robjects
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr
from scipy import stats
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge
from sklearn.metrics import (
    get_scorer,
    mean_absolute_percentage_error,
    r2_score,
)

# Machine learning imports
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)
from statsmodels.tsa.stattools import adfuller

# import adal





# # Visualization and interactive tools
# import ipywidgets as widgets
# from IPython.display import display

# # Azure and Databricks-specific imports
# from azure.identity import ClientSecretCredential
# from azure.storage.blob import BlobServiceClient
# from adlfs import AzureBlobFileSystem
# import databricks.sql as sql

# # PySpark imports
# from pyspark.sql import functions as F
# from pyspark.sql.functions import (
#     concat_ws, lower, col, translate, mean, sum, expr, concat,
#     lit, to_date, lpad, year
# )
# from pyspark.sql.window import Window



print("All required libraries are installed.")


# COMMAND ----------

#authentication function
def Adlsg2_authentication(account_name,kv_name,client_secret,client_id, tenant_id):
    """
    Authenticate Azure Data Lake Storage with service priciple to read and write data
    :param account_name azure storage account name
    :param kv_name azure keyvault name (secret scope linked to databricks)
    :param client_secret azure service principle client secret
    :param client_id azure service principle client id
    :param tenant_id azure service principle tenant id

    :returns
    None
    """
    spark.conf.set("fs.azure.account.auth.type.{}.dfs.core.windows.net".format(account_name), "OAuth")
    spark.conf.set("fs.azure.account.oauth.provider.type.{}.dfs.core.windows.net".format(account_name), "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
    spark.conf.set("fs.azure.account.oauth2.client.id.{}.dfs.core.windows.net".format(account_name), client_id)
    spark.conf.set("fs.azure.account.oauth2.client.secret.{}.dfs.core.windows.net".format(account_name), client_secret)
    spark.conf.set("fs.azure.account.oauth2.client.endpoint.{}.dfs.core.windows.net".format(account_name), "https://login.microsoftonline.com/{}/oauth2/token".format(tenant_id))
