# Databricks notebook source
# MAGIC %md
# MAGIC **Data Preparation**

# COMMAND ----------

# MAGIC %run ./configuration_function

# COMMAND ----------

# MAGIC %md
# MAGIC before cleaning(v1)

# COMMAND ----------

# def data_preparation(input_config,output_config,mapping_config,refresh_config,filter_config,storage_options):
#     processed_harmonized_data= pd.read_csv(output_config["processed_input_data"], storage_options=storage_options)
#     #display(processed_harmonized_data)
#     df=processed_harmonized_data.copy()
#     req_cols = pd.read_csv(mapping_config["idv_list"], storage_options=storage_options)

#     brand_category_to_run = pd.read_csv(mapping_config["brand_list"], storage_options=storage_options)
#     category_to_filter = "CAT LITTER"

#     # Filter the dataframe for the specific category
#     brand_category_to_run = brand_category_to_run[brand_category_to_run['category'] == category_to_filter]


#     dashboard_metric_names_mapping = pd.read_excel(mapping_config["dashboard_metric_names_mapping"], storage_options=storage_options)

#     # Normalize the 'idv_for_model_corrected' column to lowercase
#     req_cols['idv_for_model_corrected'] = req_cols['idv_for_model_corrected'].str.lower()

#     # Create a copy of the original DataFrame for further analysis
#     equity_dt = df.copy()

#     # Get unique product categories
#     #category_list = req_cols['product_category_idv'].unique()
#     category_list = ["CAT LITTER"]

#     # Initialize DataFrames to store results
#     fit_summary_all_brands = pd.DataFrame()
#     corr_pillar_all_brands = pd.DataFrame()

#     # Initialize a log DataFrame with specified columns
#     log_columns = [
#         "brand", "category", "start_date", "end_date",
#         "l_shape_rows", "l_shape_cols", "l_shape1_rows", "l_shape1_cols",
#         "l_shape5_rows", "l_shape5_cols", "l_shape2_rows", "l_shape2_cols",
#         "l_n_complete_null_columns", "l_shape3_rows", "l_shape3_cols",
#         "l_n_high_null_columns", "l_shape4_rows", "l_shape4_cols",
#         "l_n_partially_null_columns", "pillar_", "l_n_pillar_metrics",
#         "l_n_pillar_a_metrics"
#     ]
#     log_dt = pd.DataFrame(columns=log_columns)

#     # Convert the 'date' column to datetime format
#     equity_dt['date'] = pd.to_datetime(equity_dt['date'], utc=False)

#     # Print the minimum and maximum date values for verification
#     print("Minimum date:", equity_dt['date'].min(skipna=True))
#     print("Maximum date:", equity_dt['date'].max(skipna=True))

#     # Define the date range from run_config
#     date1 = pd.to_datetime(refresh_config["start_date"], format='%Y-%m-%d')
#     date2 = pd.to_datetime(refresh_config["end_date"], format='%Y-%m-%d')

#     # Filter the DataFrame based on the date range
#     equity_dt = equity_dt[(equity_dt['date'] >= date1) & (equity_dt['date'] <= date2)]

#     # Normalize column names to lowercase
#     equity_dt.columns = equity_dt.columns.str.lower()

#     req_cols1 = req_cols[req_cols['Select'] == 'Y']
#     req_cols_ = req_cols1['idv_for_model_corrected']
#     req_cols_ =list(req_cols_.unique())+["brand_group_expanded", "category","date"]
#     equity_dt_sel = equity_dt[req_cols_]

#     def scaled_data_prep(df, category_list, req_cols, brand_category_to_run):
#         eq_sub_scale_merged_brand = pd.DataFrame()
#         for category_ in category_list:
#         # for category_ in ["CAT FOOD"]:
#             try:
#                 print(category_)
#                 equity_dt_b = df[df['category'] == category_]  # subsetting for category
#                 brand_list = equity_dt_b['brand_group_expanded'].unique()  # finding the unique brands
#                 brands_list = brand_category_to_run[brand_category_to_run['category'] == category_]['brand_group_expanded'].unique()  # obtaining the brands list in the category
#                 category_brands_list = set(brand_list).intersection(brands_list)

#                 # Taking the entire brand list
#                 equity_dt_stack = equity_dt_b[equity_dt_b['brand_group_expanded'].isin(category_brands_list)]

#                 equity_dt_stack_date = equity_dt_stack['date']
#                 equity_dt_stack_brand_list = equity_dt_stack['brand_group_expanded']

#                 # Use regex to filter the columns
#                 directions_columns = [col for col in equity_dt_stack.columns if re.match(r'^directions_brand_personality_.*_net$', col)]

#                 equity_dt_stack = equity_dt_stack.drop(columns=directions_columns)

#                 # selecting the req cols for the category from the IDV file
#                 req_cols1 = req_cols[(req_cols['Select'] == 'Y') & (req_cols['product_category_idv'] == category_)]
#                 req_cols_ = req_cols1['idv_for_model_corrected']
#                 equity_dt_stack = equity_dt_stack[equity_dt_stack.columns.intersection(req_cols_)]

#                 if filter_config['scaling']['cap_min_max_from_idv_file']:
#                     # capping the max and min values from the idv file
#                     for col in equity_dt_stack.columns:
#                         if col in req_cols1['idv_for_model_corrected'].values:

#                             max_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'max'].unique()[0]
#                             min_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'min'].unique()[0]

#                             equity_dt_stack[col] = equity_dt_stack[col].astype(float)
#                             equity_dt_stack.loc[equity_dt_stack[col] > max_value, col] = max_value
#                             equity_dt_stack.loc[equity_dt_stack[col] < min_value, col] = min_value

#                 l_complete_null_columns_stack = equity_dt_stack.columns[equity_dt_stack.isnull().sum() == len(equity_dt_stack)].tolist()
#                 eq_sub_stack = equity_dt_stack.drop(columns=l_complete_null_columns_stack)

#                 if category_ in filter_config["high_null_count_categories"]:
#                     l_high_null_columns_stack = eq_sub_stack.columns[eq_sub_stack.isnull().sum() >= len(eq_sub_stack) * 0.9].to_list()
#                 else:
#                     l_high_null_columns_stack = eq_sub_stack.columns[eq_sub_stack.isnull().sum() >= len(eq_sub_stack) * 0.5].to_list()

#                 print(l_high_null_columns_stack)

#                 l_n_high_null_columns_stack = len(l_high_null_columns_stack)

#                 if l_n_high_null_columns_stack > 0:
#                     eq_sub_stack = eq_sub_stack.drop(columns=l_high_null_columns_stack)

#                 eq_sub_stack = eq_sub_stack.apply(pd.to_numeric, errors='coerce')
#                 l_partially_null_columns = eq_sub_stack.columns[eq_sub_stack.isnull().mean() > 0].tolist()
#                 l_n_partially_null_columns = len(l_partially_null_columns)
#                 eq_sub_stack = eq_sub_stack.fillna(eq_sub_stack.mean())

#                 correlation = eq_sub_stack.corr()

#                 if filter_config["scaling"]['transform_using_min_max_scaler']:
#                     # normalise data using custom function
#                     eq_sub_scaled_stack = eq_sub_stack.apply(minMax)
#                 if filter_config["scaling"]['custom_min_max'] == True:
#                     eq_sub_scaled_stack = eq_sub_stack.copy()
#                     columns = eq_sub_stack.columns
#                     for col in columns:
#                         print(col)
#                         max_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'max'].unique()[0]
#                         min_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'min'].unique()[0]
#                         print(max_value)

#                         eq_sub_scaled_stack[col] = (eq_sub_scaled_stack[col] - min_value) / (max_value - min_value)

#                 if filter_config["scaling"]['transform_using_standard_scaler'] == True:
#                     scaler = StandardScaler()
#                     eq_sub_scaled = scaler.fit_transform(eq_sub_stack)
#                     eq_sub_scaled_stack = pd.DataFrame(eq_sub_scaled, columns=eq_sub_stack.columns)

#                 eq_sub_scaled_stack['Brand'] = equity_dt_stack_brand_list
#                 eq_sub_scaled_stack['New_Brand'] = "Stacked Brand"
#                 eq_sub_scaled_stack['Category'] = category_
#                 eq_sub_scaled_stack['date'] = equity_dt_stack_date

#                 for brand in category_brands_list:

#                     print(brand)

#                     # initialize for log
#                     l_shape_rows = l_shape_cols = l_shape1_rows = l_shape1_cols = l_shape5_rows = l_shape5_cols = \
#                     l_shape2_rows = l_shape2_cols = l_n_complete_null_columns = l_shape3_rows = l_shape3_cols = l_colnames_completely_null = \
#                     l_n_high_null_columns = l_shape4_rows = l_shape4_cols = l_colnames_50_percent_null = l_n_partially_null_columns = \
#                     pillar_ = l_n_pillar_metrics = l_n_pillar_a_metrics = 0

#                     ### Filtering for a brand
#                     equity_dt_ = equity_dt_b[equity_dt_b['brand_group_expanded'] == brand]
#                     l_shape_rows = equity_dt_.shape[0]
#                     l_shape_cols = equity_dt_.shape[1]

#                     equity_dt_date = equity_dt_['date']
#                     l_shape1_rows = equity_dt_.shape[0]
#                     l_shape1_cols = equity_dt_.shape[1]

#                     l_shape5_rows, l_shape5_cols = equity_dt_.shape
#                     equity_dt_ = equity_dt_[equity_dt_.columns[equity_dt_.columns.isin(req_cols_)].tolist()]

#                     l_shape2_rows, l_shape2_cols = equity_dt_.shape

#                     if filter_config["scaling"]['cap_min_max_from_idv_file']:
#                         # Capping the max and min values from the idv file
#                         for col in equity_dt_.columns:
#                             if col in req_cols1['idv_for_model_corrected'].values:
#     #                             print(f"Column: {col}")
#                                 max_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'max'].unique()[0]
#                                 min_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'min'].unique()[0]
#     #                             print(f"max from data: {equity_dt_[col].max()}")
#                                 equity_dt_[col] = equity_dt_[col].astype(float)
#                                 equity_dt_[col] = equity_dt_[col].clip(lower=min_value, upper=max_value)

#                     # Some data prep

#                     # Completely null columns
#                     l_complete_null_columns = equity_dt_.columns[equity_dt_.isnull().sum() == len(equity_dt_)]

#                     l_n_complete_null_cols = len(l_complete_null_columns)

#                     # Remove completely NULL columns
#                     eq_sub = equity_dt_.drop(columns=l_complete_null_columns)
#                     l_shape3_rows, l_shape3_cols = eq_sub.shape

#                     # Getting column names of completely null
#                     l_colnames_completely_null = ', '.join(sorted(eq_sub.columns))

#                     # Columns > 50% missing values
#                     if category_ in filter_config["high_null_count_categories"]:
#                         l_high_null_columns = eq_sub.columns[eq_sub.isnull().sum() >= len(eq_sub) * 0.9].to_list()
#                     else:
#                         l_high_null_columns = eq_sub.columns[eq_sub.isnull().sum() >= len(eq_sub) * 0.5].to_list()

#                     l_n_high_null_columns = len(l_high_null_columns)

#                     print(l_high_null_columns)

#                     # Remove columns with more than 50% NA
#                     if l_n_high_null_columns > 0:
#                         eq_sub = eq_sub.drop(columns=l_high_null_columns)

#                     l_shape4_rows, l_shape4_cols = eq_sub.shape

#                     # Getting column names of completely null
#                     l_colnames_50_percent_null = ', '.join(sorted(eq_sub.columns))

#                     # Converting equity columns to numeric
#                     eq_sub = eq_sub.apply(pd.to_numeric, errors='coerce')

#                     # Columns with any missing values
#                     l_partially_null_columns = eq_sub.columns[eq_sub.isnull().mean() > 0]
#                     l_n_partially_null_columns = len(l_partially_null_columns)
#                     eq_sub = eq_sub.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)

#                     # Calculate correlation
#                     correlation = eq_sub.corr()

#                     if filter_config["scaling"]["transform_using_min_max_scaler"]:
#                         # Normalize data using MinMaxScaler
#                         scaler = MinMaxScaler()
#                         eq_sub_scaled = pd.DataFrame(scaler.fit_transform(eq_sub), columns=eq_sub.columns)

#                     if filter_config["scaling"]['custom_min_max']:
#                         eq_sub_scaled = eq_sub.copy()
#                         for col in eq_sub.columns:
#     #                         print(col)
#                             max_value = float(req_cols1[req_cols1['idv_for_model_corrected'] == col]['max'].unique())
#                             min_value = float(req_cols1[req_cols1['idv_for_model_corrected'] == col]['min'].unique())
#     #                         print(max_value)

#                             eq_sub_scaled[col] = (eq_sub_scaled[col] - min_value) / (max_value - min_value)

#                     if filter_config["scaling"]["transform_using_standard_scaler"]:
#                         # Standard scaler on the df
#                         scaler = StandardScaler()
#                         eq_sub_scaled = pd.DataFrame(scaler.fit_transform(eq_sub), columns=eq_sub.columns)

#                     eq_sub_scaled['Brand'] = brand
#                     eq_sub_scaled['New_Brand'] = brand
#                     eq_sub_scaled['Category'] = category_
#                     eq_sub_scaled['date'] = equity_dt_date
#                     eq_sub_scale_merged_brand = pd.concat([eq_sub_scale_merged_brand, eq_sub_scaled], ignore_index=True)

#             except:
#                     # Capture and store the error message
#                     print(f"Warning: Error in preprocessing in brand: {category_}")
#                     print(f"Error in preprocessing in brand: {category_}")
#             eq_sub_scale_merged_brand = pd.concat([eq_sub_scale_merged_brand,eq_sub_scaled_stack])
#         return eq_sub_scale_merged_brand

#     def dv_data_prep(equity_dt, eq_sub_scale_merged_brand_copy):
#         if refresh_config['dv'] == "market_share":
#             dv_data = equity_dt[['date', 'brand_group_expanded', 'category', 'market_share_total_sales']].copy()
#         elif refresh_config['dv'] == "equalized_volume":
#             dv_data = equity_dt[['date', 'brand_group_expanded', 'category', 'equalized_volume']].copy()
#         elif refresh_config['dv'] == "total_sales":
#             dv_data = equity_dt[['date', 'brand_group_expanded', 'category', 'total_sales']].copy()

#         # Rename multiple columns for old to new
#         dv_data.rename(columns={'brand_group_expanded': 'brand', 'category': 'category', 'market_share_total_sales': 'market_share'}, inplace=True)
#         eq_sub_scale_merged_brand_copy['date'] = pd.to_datetime(eq_sub_scale_merged_brand_copy['date'], format="%Y-%m-%d", utc=False)
#         dv_data['date'] = pd.to_datetime(dv_data['date'], format="%Y-%m-%d")
#         eq_sub_scale_merged_brand_copy['date'] = eq_sub_scale_merged_brand_copy['date'].astype(str)
#         dv_data['date'] = dv_data['date'].astype(str)
#         eq_sub_scale_merged_brand_copy.columns = map(str.lower, eq_sub_scale_merged_brand_copy.columns)
#         index_df_final1 = pd.merge(eq_sub_scale_merged_brand_copy, dv_data, on=['date', 'brand', 'category'], how='left')
#         return index_df_final1

#     eq_sub_scale_merged_brand = scaled_data_prep(equity_dt, category_list, req_cols, brand_category_to_run)

#     eq_sub_scale_merged_brand_copy = eq_sub_scale_merged_brand.copy()

#     modeling_data = dv_data_prep(equity_dt, eq_sub_scale_merged_brand_copy)

#     modeling_data_copy = modeling_data.copy()
#     equity_dt.to_csv(output_config["data_prep"]["equity_dt"], index=False, storage_options=storage_options)
#     eq_sub_scale_merged_brand.to_csv(output_config["data_prep"]["eq_sub_scale"], index=False, storage_options=storage_options)
#     modeling_data.to_csv(output_config["data_prep"]["modeling_data"], index=False, storage_options=storage_options)

#     eq_sub_scale_merged_brand["date"] = pd.to_datetime(eq_sub_scale_merged_brand["date"], utc=False)
#     modeling_data["date"] = pd.to_datetime(modeling_data["date"], utc=False)




# COMMAND ----------

# def scaled_data_prep(df, category_list, req_cols, brand_category_to_run):
#     eq_sub_scale_merged_brand = pd.DataFrame()
#     for category_ in category_list:
#     # for category_ in ["CAT FOOD"]:
#         try:
#             print(category_)
#             equity_dt_b = df[df['category'] == category_]  # subsetting for category
#             brand_list = equity_dt_b['brand_group_expanded'].unique()  # finding the unique brands
#             brands_list = brand_category_to_run[brand_category_to_run['category'] == category_]['brand_group_expanded'].unique()  # obtaining the brands list in the category
#             category_brands_list = set(brand_list).intersection(brands_list)

#             # Taking the entire brand list
#             equity_dt_stack = equity_dt_b[equity_dt_b['brand_group_expanded'].isin(category_brands_list)]

#             equity_dt_stack_date = equity_dt_stack['date']
#             equity_dt_stack_brand_list = equity_dt_stack['brand_group_expanded']

#             # Use regex to filter the columns
#             directions_columns = [col for col in equity_dt_stack.columns if re.match(r'^directions_brand_personality_.*_net$', col)]

#             equity_dt_stack = equity_dt_stack.drop(columns=directions_columns)

#             # selecting the req cols for the category from the IDV file
#             req_cols1 = req_cols[(req_cols['Select'] == 'Y') & (req_cols['product_category_idv'] == category_)]
#             req_cols_ = req_cols1['idv_for_model_corrected']
#             equity_dt_stack = equity_dt_stack[equity_dt_stack.columns.intersection(req_cols_)]

#             if filter_config['scaling']['cap_min_max_from_idv_file']:
#                 # capping the max and min values from the idv file
#                 for col in equity_dt_stack.columns:
#                     if col in req_cols1['idv_for_model_corrected'].values:

#                         max_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'max'].unique()[0]
#                         min_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'min'].unique()[0]

#                         equity_dt_stack[col] = equity_dt_stack[col].astype(float)
#                         equity_dt_stack.loc[equity_dt_stack[col] > max_value, col] = max_value
#                         equity_dt_stack.loc[equity_dt_stack[col] < min_value, col] = min_value

#             l_complete_null_columns_stack = equity_dt_stack.columns[equity_dt_stack.isnull().sum() == len(equity_dt_stack)].tolist()
#             eq_sub_stack = equity_dt_stack.drop(columns=l_complete_null_columns_stack)

#             if category_ in filter_config["high_null_count_categories"]:
#                 l_high_null_columns_stack = eq_sub_stack.columns[eq_sub_stack.isnull().sum() >= len(eq_sub_stack) * 0.9].to_list()
#             else:
#                 l_high_null_columns_stack = eq_sub_stack.columns[eq_sub_stack.isnull().sum() >= len(eq_sub_stack) * 0.5].to_list()

#             print(l_high_null_columns_stack)

#             l_n_high_null_columns_stack = len(l_high_null_columns_stack)

#             if l_n_high_null_columns_stack > 0:
#                 eq_sub_stack = eq_sub_stack.drop(columns=l_high_null_columns_stack)

#             eq_sub_stack = eq_sub_stack.apply(pd.to_numeric, errors='coerce')
#             l_partially_null_columns = eq_sub_stack.columns[eq_sub_stack.isnull().mean() > 0].tolist()
#             l_n_partially_null_columns = len(l_partially_null_columns)
#             eq_sub_stack = eq_sub_stack.fillna(eq_sub_stack.mean())

#             correlation = eq_sub_stack.corr()

#             if filter_config["scaling"]['transform_using_min_max_scaler']:
#                 # normalise data using custom function
#                 eq_sub_scaled_stack = eq_sub_stack.apply(minMax)
#             if filter_config["scaling"]['custom_min_max'] == True:
#                 eq_sub_scaled_stack = eq_sub_stack.copy()
#                 columns = eq_sub_stack.columns
#                 for col in columns:
#                     print(col)
#                     max_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'max'].unique()[0]
#                     min_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'min'].unique()[0]
#                     print(max_value)

#                     eq_sub_scaled_stack[col] = (eq_sub_scaled_stack[col] - min_value) / (max_value - min_value)

#             if filter_config["scaling"]['transform_using_standard_scaler'] == True:
#                 scaler = StandardScaler()
#                 eq_sub_scaled = scaler.fit_transform(eq_sub_stack)
#                 eq_sub_scaled_stack = pd.DataFrame(eq_sub_scaled, columns=eq_sub_stack.columns)

#             eq_sub_scaled_stack['Brand'] = equity_dt_stack_brand_list
#             eq_sub_scaled_stack['New_Brand'] = "Stacked Brand"
#             eq_sub_scaled_stack['Category'] = category_
#             eq_sub_scaled_stack['date'] = equity_dt_stack_date

#             for brand in category_brands_list:

#                 print(brand)

#                 # initialize for log
#                 l_shape_rows = l_shape_cols = l_shape1_rows = l_shape1_cols = l_shape5_rows = l_shape5_cols = \
#                 l_shape2_rows = l_shape2_cols = l_n_complete_null_columns = l_shape3_rows = l_shape3_cols = l_colnames_completely_null = \
#                 l_n_high_null_columns = l_shape4_rows = l_shape4_cols = l_colnames_50_percent_null = l_n_partially_null_columns = \
#                 pillar_ = l_n_pillar_metrics = l_n_pillar_a_metrics = 0

#                 ### Filtering for a brand
#                 equity_dt_ = equity_dt_b[equity_dt_b['brand_group_expanded'] == brand]
#                 l_shape_rows = equity_dt_.shape[0]
#                 l_shape_cols = equity_dt_.shape[1]

#                 equity_dt_date = equity_dt_['date']
#                 l_shape1_rows = equity_dt_.shape[0]
#                 l_shape1_cols = equity_dt_.shape[1]

#                 l_shape5_rows, l_shape5_cols = equity_dt_.shape
#                 equity_dt_ = equity_dt_[equity_dt_.columns[equity_dt_.columns.isin(req_cols_)].tolist()]

#                 l_shape2_rows, l_shape2_cols = equity_dt_.shape

#                 if filter_config["scaling"]['cap_min_max_from_idv_file']:
#                     # Capping the max and min values from the idv file
#                     for col in equity_dt_.columns:
#                         if col in req_cols1['idv_for_model_corrected'].values:
# #                             print(f"Column: {col}")
#                             max_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'max'].unique()[0]
#                             min_value = req_cols1.loc[req_cols1['idv_for_model_corrected'] == col, 'min'].unique()[0]
# #                             print(f"max from data: {equity_dt_[col].max()}")
#                             equity_dt_[col] = equity_dt_[col].astype(float)
#                             equity_dt_[col] = equity_dt_[col].clip(lower=min_value, upper=max_value)

#                 # Some data prep

#                 # Completely null columns
#                 l_complete_null_columns = equity_dt_.columns[equity_dt_.isnull().sum() == len(equity_dt_)]

#                 l_n_complete_null_cols = len(l_complete_null_columns)

#                 # Remove completely NULL columns
#                 eq_sub = equity_dt_.drop(columns=l_complete_null_columns)
#                 l_shape3_rows, l_shape3_cols = eq_sub.shape

#                 # Getting column names of completely null
#                 l_colnames_completely_null = ', '.join(sorted(eq_sub.columns))

#                 # Columns > 50% missing values
#                 if category_ in filter_config["high_null_count_categories"]:
#                     l_high_null_columns = eq_sub.columns[eq_sub.isnull().sum() >= len(eq_sub) * 0.9].to_list()
#                 else:
#                     l_high_null_columns = eq_sub.columns[eq_sub.isnull().sum() >= len(eq_sub) * 0.5].to_list()

#                 l_n_high_null_columns = len(l_high_null_columns)

#                 print(l_high_null_columns)

#                 # Remove columns with more than 50% NA
#                 if l_n_high_null_columns > 0:
#                     eq_sub = eq_sub.drop(columns=l_high_null_columns)

#                 l_shape4_rows, l_shape4_cols = eq_sub.shape

#                 # Getting column names of completely null
#                 l_colnames_50_percent_null = ', '.join(sorted(eq_sub.columns))

#                 # Converting equity columns to numeric
#                 eq_sub = eq_sub.apply(pd.to_numeric, errors='coerce')

#                 # Columns with any missing values
#                 l_partially_null_columns = eq_sub.columns[eq_sub.isnull().mean() > 0]
#                 l_n_partially_null_columns = len(l_partially_null_columns)
#                 eq_sub = eq_sub.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)

#                 # Calculate correlation
#                 correlation = eq_sub.corr()

#                 if filter_config["scaling"]["transform_using_min_max_scaler"]:
#                     # Normalize data using MinMaxScaler
#                     scaler = MinMaxScaler()
#                     eq_sub_scaled = pd.DataFrame(scaler.fit_transform(eq_sub), columns=eq_sub.columns)

#                 if filter_config["scaling"]['custom_min_max']:
#                     eq_sub_scaled = eq_sub.copy()
#                     for col in eq_sub.columns:
# #                         print(col)
#                         max_value = float(req_cols1[req_cols1['idv_for_model_corrected'] == col]['max'].unique())
#                         min_value = float(req_cols1[req_cols1['idv_for_model_corrected'] == col]['min'].unique())
# #                         print(max_value)

#                         eq_sub_scaled[col] = (eq_sub_scaled[col] - min_value) / (max_value - min_value)

#                 if filter_config["scaling"]["transform_using_standard_scaler"]:
#                     # Standard scaler on the df
#                     scaler = StandardScaler()
#                     eq_sub_scaled = pd.DataFrame(scaler.fit_transform(eq_sub), columns=eq_sub.columns)

#                 eq_sub_scaled['Brand'] = brand
#                 eq_sub_scaled['New_Brand'] = brand
#                 eq_sub_scaled['Category'] = category_
#                 eq_sub_scaled['date'] = equity_dt_date
#                 eq_sub_scale_merged_brand = pd.concat([eq_sub_scale_merged_brand, eq_sub_scaled], ignore_index=True)

#         except:
#                 # Capture and store the error message
#                 print(f"Warning: Error in preprocessing in brand: {category_}")
#                 print(f"Error in preprocessing in brand: {category_}")
#         eq_sub_scale_merged_brand = pd.concat([eq_sub_scale_merged_brand,eq_sub_scaled_stack])
#     return eq_sub_scale_merged_brand


# COMMAND ----------

# MAGIC %md
# MAGIC after cleaning(under v3)

# COMMAND ----------
from library_installation import *


def cap_values(df, col, min_val, max_val):
    df[col] = df[col].astype(float).clip(lower=min_val, upper=max_val)
    return df


# COMMAND ----------

def process_level(df, req_cols, brand_category_to_run, filter_config, level, level_value, parent_level=None):
    """
    Processes data for a specific level (category or brand).

    Parameters:
        df (DataFrame): Input DataFrame.
        req_cols (DataFrame): Required columns configuration.
        filter_config (dict): Configuration for filtering and scaling.
        level (str): The level to process ('category' or 'brand').
        level_value (str): The value of the level being processed.
        parent_level (str, optional): The parent level value for brand processing (e.g., category for brands).

    Returns:
        DataFrame: Processed and scaled data for the specified level.
    """
    # try:
    print(f"Processing {level}: {level_value}")
    df["date"] = pd.to_datetime(df["date"])
    # Filter data for the specified level
    if level == 'category':
        # df_filtered = df[df['category'] == level_value]
        category_ = level_value
        df_filtered = df[df['category'] == category_]  # subsetting for category
    elif level == 'brand':
        # df_filtered = df[(df['category'] == parent_level) & (df['brand_group_expanded'] == level_value)]
        category_ = parent_level
        brand = level_value
        df_filtered = df[(df['brand_group_expanded'] == brand) & (df['category'] == category_)]  # subsetting for category
    else:
        raise ValueError(f"Invalid level: {level}. Expected 'category' or 'brand'.")

    brand_list = df_filtered['brand_group_expanded'].unique()  # finding the unique brands
    brands_list = brand_category_to_run[brand_category_to_run['category'] == category_]['brand_group_expanded'].unique()  # obtaining the brands
    category_brands_list = set(brand_list).intersection(brands_list)
    # Taking the entire brand list
    df_filtered = df_filtered[df_filtered['brand_group_expanded'].isin(category_brands_list)]

    # Extract metadata columns
    metadata_cols = ['date', 'brand_group_expanded', 'category']
    metadata = df_filtered[metadata_cols]

    # Filter required columns
    req_cols_level = req_cols[(req_cols['Select'] == 'Y') &
                                (req_cols['product_category_idv'] == category_)]

    # if level == 'brand':
    #     req_cols_level = req_cols[
    #         (req_cols['Select'] == 'Y') &
    #         (req_cols['product_category_idv'] == parent_level)
    #     ]
    # else:
    #     req_cols_level = req_cols[
    #         (req_cols['Select'] == 'Y') &
    #         (req_cols['product_category_idv'] == level_value)
    #     ]


    df_filtered = df_filtered[df_filtered.columns.intersection(req_cols_level['idv_for_model_corrected'])]

    # Cap values if configured
    if filter_config['scaling']['cap_min_max_from_idv_file']:
        for col in df_filtered.columns:
            if col in req_cols_level['idv_for_model_corrected'].values:
                max_value = req_cols_level.loc[req_cols_level['idv_for_model_corrected'] == col, 'max'].unique()[0]
                min_value = req_cols_level.loc[req_cols_level['idv_for_model_corrected'] == col, 'min'].unique()[0]
                df_filtered = cap_values(df_filtered, col, min_value, max_value)

    # Handle null values
    high_null_threshold = 0.9 if parent_level in filter_config["high_null_count_categories"] else 0.5
    df_filtered = df_filtered.dropna(axis=1, thresh=int(len(df_filtered) * (1 - high_null_threshold)))  # Remove high-null columns
    df_filtered = df_filtered.fillna(df_filtered.mean())  # Fill partial nulls

    # Scale data
    if filter_config["scaling"]['transform_using_min_max_scaler']:
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_filtered), columns=df_filtered.columns)
    elif filter_config["scaling"]['custom_min_max']:
        df_scaled = df_filtered.copy()
        for col in df_filtered.columns:
            max_value = req_cols_level.loc[req_cols_level['idv_for_model_corrected'] == col, 'max'].unique()[0]
            min_value = req_cols_level.loc[req_cols_level['idv_for_model_corrected'] == col, 'min'].unique()[0]
            df_scaled[col] = (df_scaled[col] - min_value) / (max_value - min_value)
    elif filter_config["scaling"]['transform_using_standard_scaler']:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_filtered), columns=df_filtered.columns)

    # Add metadata columns
    df_scaled['Brand'] = brand if level == 'brand' else  metadata['brand_group_expanded']
    df_scaled['New_Brand'] = brand if level == 'brand' else "Stacked Brand"
    df_scaled['Category'] = category_
    df_scaled['date'] = metadata['date']

    return df_scaled

    # except Exception as e:
    #     print(f"Error processing {level} {level_value}: {e}")
        # return pd.DataFrame()  # Return an empty DataFrame on error

# COMMAND ----------

# def scaled_data_prep(df, category_list, req_cols, brand_category_to_run, filter_config):
#     eq_sub_scale_merged_brand = pd.DataFrame()

#     for category_ in category_list:
#         # Process category level
#         eq_category_scaled = process_level(df, req_cols, filter_config, level='category', level_value=category_)
#         eq_sub_scale_merged_brand = pd.concat([eq_sub_scale_merged_brand, eq_category_scaled], ignore_index=True)

#         # Get brands within the category
#         equity_dt_b = df[df['category'] == category_]
#         req_cols1 = req_cols[(req_cols['Select'] == 'Y') & (req_cols['product_category_idv'] == category_)]
#         brand_list = brand_category_to_run[brand_category_to_run['category'] == category_]['brand_group_expanded'].unique()

#         for brand in brand_list:
#             # Process brand level
#             eq_brand_scaled = process_level(df, req_cols1, filter_config, level='brand', level_value=brand, parent_level=category_)
#             eq_sub_scale_merged_brand = pd.concat([eq_sub_scale_merged_brand, eq_brand_scaled], ignore_index=True)

#     return eq_sub_scale_merged_brand


# COMMAND ----------

def dv_data_prep(equity_dt, eq_sub_scale_merged_brand_copy, refresh_config):
    if refresh_config['dv'] == "market_share":
        dv = 'market_share_total_sales'
    else:
        dv = refresh_config['dv']

    dv_data = equity_dt[['date', 'brand_group_expanded', 'category', dv]].copy()
    # Rename multiple columns for old to new
    dv_data.rename(columns={'brand_group_expanded': 'brand', 'category': 'category', 'market_share_total_sales': 'market_share'}, inplace=True)
    eq_sub_scale_merged_brand_copy['date'] = pd.to_datetime(eq_sub_scale_merged_brand_copy['date'], format="%Y-%m-%d", utc=False)
    dv_data['date'] = pd.to_datetime(dv_data['date'], format="%Y-%m-%d")
    eq_sub_scale_merged_brand_copy['date'] = eq_sub_scale_merged_brand_copy['date'].astype(str)
    dv_data['date'] = dv_data['date'].astype(str)
    eq_sub_scale_merged_brand_copy.columns = map(str.lower, eq_sub_scale_merged_brand_copy.columns)
    index_df_final1 = pd.merge(eq_sub_scale_merged_brand_copy, dv_data, on=['date', 'brand', 'category'], how='left')
    return index_df_final1

# COMMAND ----------

def filtering_idv_list(df):
    df['idv_for_model_corrected'] = df['idv_for_model_corrected'].str.lower()
    df1 = df[df['Select'] == 'Y']
    df_ = df1['idv_for_model_corrected']
    df_ = ["brand_group_expanded", "category","date"] + list(df_.unique())
    return df_

# COMMAND ----------

def filter_by_date_range(df, start_date, end_date):
    df['date'] = pd.to_datetime(df['date'], utc=False)

    # Print the minimum and maximum date values for verification
    print("Minimum date:", df['date'].min(skipna=True))
    print("Maximum date:", df['date'].max(skipna=True))

    # Define the date range from run_config
    date1 = pd.to_datetime(start_date, format='%Y-%m-%d')
    date2 = pd.to_datetime(end_date, format='%Y-%m-%d')

    # Filter the DataFrame based on the date range
    df = df[(df['date'] >= date1) & (df['date'] <= date2)]
    return df

# COMMAND ----------

# #v1
# def data_preparation(input_config,output_config,mapping_config,refresh_config,filter_config,storage_options):
#     equity_dt = pd.read_csv(output_config["processed_input_data"], storage_options=storage_options)

#     req_cols = pd.read_csv(mapping_config["idv_list"], storage_options=storage_options)

#     brand_category_to_run = pd.read_csv(mapping_config["brand_list"], storage_options=storage_options)

#     dashboard_metric_names_mapping = pd.read_excel(mapping_config["dashboard_metric_names_mapping"], storage_options=storage_options)

#     equity_dt = filter_by_date_range(equity_dt,refresh_config["start_date"],refresh_config["end_date"])

#     # Normalize column names to lowercase
#     equity_dt.columns = equity_dt.columns.str.lower()

#     category_list = req_cols['product_category_idv'].unique()
#     eq_sub_scale_merged_brand = scaled_data_prep(equity_dt, category_list, req_cols, brand_category_to_run, filter_config)

#     eq_sub_scale_merged_brand_copy = eq_sub_scale_merged_brand.copy()

#     modeling_data = dv_data_prep(equity_dt, eq_sub_scale_merged_brand_copy)

#     equity_dt.to_csv(output_config["data_prep"]["equity_dt"], index=False, storage_options=storage_options)
#     eq_sub_scale_merged_brand.to_csv(output_config["data_prep"]["eq_sub_scale"], index=False, storage_options=storage_options)
#     modeling_data.to_csv(output_config["data_prep"]["modeling_data"], index=False, storage_options=storage_options)


# COMMAND ----------

#v2
def data_preparation(input_config,output_config,mapping_config,refresh_config,filter_config,storage_options, refresh_type):

    equity_dt = pd.read_csv(output_config["processed_input_data"], storage_options= storage_options )
    equity_dt["date"] = pd.to_datetime(equity_dt["date"], utc=False)
    req_cols = pd.read_csv(mapping_config["idv_list"], storage_options=storage_options)

    brand_category_to_run = pd.read_csv(mapping_config["brand_list"], storage_options=storage_options)

    dashboard_metric_names_mapping = pd.read_excel(mapping_config["dashboard_metric_names_mapping"], storage_options=storage_options)

    equity_dt = filter_by_date_range(equity_dt,refresh_config["start_date"],refresh_config["end_date"])

    # Normalize column names to lowercase
    equity_dt.columns = equity_dt.columns.str.lower()

    category_list = req_cols['product_category_idv'].unique()
    # eq_sub_scale_merged_brand = scaled_data_prep(equity_dt, category_list, req_cols, brand_category_to_run, filter_config)
    eq_sub_scale_merged = pd.DataFrame()
    eq_sub_scale_merged_brand = pd.DataFrame()
    for category_ in category_list:
        # Process category level
        eq_sub_scaled_stack = process_level(equity_dt, req_cols,brand_category_to_run, filter_config, level='category', level_value=category_)
        # eq_sub_scale_merged_brand = pd.concat([eq_sub_scale_merged_brand, eq_sub_scaled_stack], ignore_index=True)

        # Get brands within the category
        equity_dt_b = equity_dt[equity_dt['category'] == category_]
        req_cols1 = req_cols[(req_cols['Select'] == 'Y') & (req_cols['product_category_idv'] == category_)]
        brand_list = equity_dt_b['brand_group_expanded'].unique()  # finding the unique brands
        brands_list = brand_category_to_run[brand_category_to_run['category'] == category_]['brand_group_expanded'].unique()
        category_brands_list = set(brand_list).intersection(brands_list)
        eq_sub_scale_all_brand = pd.DataFrame()
        for brand in category_brands_list:
            # Process brand level
            eq_brand_scaled = pd.DataFrame()
            eq_brand_scaled = process_level(equity_dt, req_cols1, brand_category_to_run, filter_config, level='brand', level_value=brand, parent_level=category_)
            eq_sub_scale_merged_brand = pd.concat([eq_sub_scale_merged_brand, eq_brand_scaled], ignore_index=True)
        # eq_sub_scale_merged = pd.concat([eq_sub_scale_merged,eq_sub_scale_all_brand,eq_sub_scaled_stack], ignore_index=True)

        eq_sub_scale_merged_brand = pd.concat([eq_sub_scale_merged_brand,eq_sub_scaled_stack])

    eq_sub_scale_merged_brand_copy = eq_sub_scale_merged_brand.copy()
    modeling_data = dv_data_prep(equity_dt, eq_sub_scale_merged_brand_copy, refresh_config)

    equity_dt.to_csv(output_config["data_prep"]["equity_dt"], index=False, storage_options=storage_options)
    eq_sub_scale_merged_brand_copy.to_csv(output_config["data_prep"]["eq_sub_scale"], index=False, storage_options=storage_options)
    modeling_data.to_csv(output_config["data_prep"]["modeling_data"], index=False, storage_options=storage_options)


# COMMAND ----------


