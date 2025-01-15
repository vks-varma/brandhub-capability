# Databricks notebook source
# MAGIC %run ./configuration_function

# COMMAND ----------

# MAGIC %md
# MAGIC before cleaning(v1)

# COMMAND ----------

# def post_modelling(input_config,output_config):
#     def dv_weighted_shap(df, model_results_sub, end_date, categories_list, DV_previous_years_to_take=1):
#         def calculate_category_sums(equity_data, dv_column):
#             """
#             Aggregates DV column to calculate mean for each brand/category 
#             and sum for each category.
#             """
#             equity_data_agg = equity_data.groupby(['category', 'brand_group_expanded'])[dv_column].mean().reset_index()
#             if dv_column == "market_share_total_sales":
#                 equity_data_agg.rename(columns={dv_column: 'Market share - mean'}, inplace=True)
#                 category_sums = equity_data_agg.groupby('category')['Market share - mean'].sum().reset_index()
#                 category_sums.rename(columns={'Market share - mean': 'category_Market share - sum'}, inplace=True)
#             elif dv_column == "equalized_volume":
#                 equity_data_agg.rename(columns={dv_column: 'Equalized Volume - mean'}, inplace=True)
#                 category_sums = equity_data_agg.groupby('category')['Equalized Volume - mean'].sum().reset_index()
#                 category_sums.rename(columns={'Equalized Volume - mean': 'category_Equalized Volume - sum'}, inplace=True)
#             elif dv_column == "total_sales":
#                 equity_data_agg.rename(columns={dv_column: 'Total Sales - mean'}, inplace=True)
#                 category_sums = equity_data_agg.groupby('category')['Total Sales - mean'].sum().reset_index()
#                 category_sums.rename(columns={'Total Sales - mean': 'category_Total Sales - sum'}, inplace=True)
            
#             return equity_data_agg, category_sums

#         # Preprocess and filter the DataFrame
#         dv_column = model_config["weights_model"]["DV"]
#         df_sub = df[['date', 'brand_group_expanded', 'category', dv_column]].copy()
#         df_sub = df_sub[df_sub['category'].isin(categories_list)]
#         df_sub[dv_column] = pd.to_numeric(df_sub[dv_column], errors='coerce')
#         df_sub.dropna(inplace=True)

#         # Filter data based on date range
#         end_date = pd.to_datetime(end_date)
#         start_date = end_date - pd.DateOffset(years=DV_previous_years_to_take)
#         df_sub = df_sub[(df_sub['date'] >= start_date) & (df_sub['date'] <= end_date)]

#         # Calculate category sums and merge them with aggregated data
#         equity_data_agg, category_sums = calculate_category_sums(df_sub, dv_column)
#         equity_data_agg = equity_data_agg.merge(category_sums, on='category', how='left')

#         # Prepare model results DataFrame for merging
#         model_results_sub.columns = model_results_sub.columns.str.lower()
#         model_results_sub.rename(columns={'category': 'category', 'brand_group_expanded': 'brands'}, inplace=True)
#         merged_results = model_results_sub.merge(equity_data_agg, left_on=['brands', 'category'], right_on=['brand_group_expanded', 'category'], how='left')

#         # Calculate weighted SHAP values based on DV type
#         shap_column_map = {
#             "market_share_total_sales": ('Market share - mean', 'category_Market share - sum'),
#             "equalized_volume": ('Equalized Volume - mean', 'category_Equalized Volume - sum'),
#             "total_sales": ('Total Sales - mean', 'category_Total Sales - sum')
#         }

#         mean_column, sum_column = shap_column_map.get(dv_column, (None, None))
#         if mean_column and sum_column:
#             merged_results['weighted_shap'] = (merged_results['shap values'] * merged_results[mean_column]) / merged_results[sum_column]

#         # Aggregate results at category and overall levels
#         category_agg = merged_results.groupby(['category', 'pillar', 'var'])['weighted_shap'].mean().reset_index()
#         category_agg.rename(columns={'weighted_shap': 'Shap Value - mean'}, inplace=True)

#         overall_agg = category_agg.groupby(['pillar', 'var'])['Shap Value - mean'].mean().reset_index()

#         return category_agg, overall_agg

#     def prepare_weights(all_cat_pillars, by_cat_pillars, cfa_results_all_cat_sub, cfa_results_by_cat_sub,
#                         model_results_pillar_all_cat, model_results_sub_agg):

#         # Standardize column names and format 'var' columns
#         def standardize_columns(*dfs):
#             for df in dfs:
#                 df.columns = [col.lower() for col in df.columns]
#                 if 'var' in df.columns:
#                     df['var'] = df['var'].str.lower()

#         standardize_columns(model_results_sub_agg, cfa_results_by_cat_sub, model_results_pillar_all_cat, cfa_results_all_cat_sub)
        
#         # Merge model and CFA results and filter by specified pillars
#         def merge_and_filter(model_df, cfa_df, pillars, by_category=False):
#             if by_category:
#                 merged_df = pd.merge(model_df, cfa_df, on=['var', 'pillar', 'category'], how='outer')
#             else:
#                 merged_df = pd.merge(model_df, cfa_df, on=['var', 'pillar'], how='outer')
#             return merged_df[merged_df['pillar'].isin(pillars)]
        
#         all_cat_weights = merge_and_filter(model_results_pillar_all_cat, cfa_results_all_cat_sub, all_cat_pillars)
#         by_cat_weights_merged = merge_and_filter(model_results_sub_agg, cfa_results_by_cat_sub, by_cat_pillars, by_category=True)
        
#         # Calculate percentage weights
#         def calculate_weights(df, shap_col='shap value - mean', cfa_col='est.std'):
#             df['abs_shap'] = df[shap_col].abs()
#             df['abs_cfa_est'] = df[cfa_col].abs()
            
#             sum_shap = df.groupby('pillar')['abs_shap'].sum().rename('metrics_sum_shap').reset_index()
#             sum_cfa = df.groupby('pillar')['abs_cfa_est'].sum().rename('metrics_sum_cfa').reset_index()
            
#             df = df.merge(sum_shap, on='pillar', how='left').merge(sum_cfa, on='pillar', how='left')
#             df['perc_shap'] = df['abs_shap'] / df['metrics_sum_shap']
#             df['perc_cfa'] = df['abs_cfa_est'] / df['metrics_sum_cfa']
#             df['weight'] = df[['perc_cfa', 'perc_shap']].mean(axis=1, skipna=True)
#             return df
        
#         # Calculate weights for each category
#         by_cat_weights = pd.concat([
#             calculate_weights(by_cat_weights_merged[by_cat_weights_merged['category'] == category])
#             for category in by_cat_weights_merged['category'].unique()
#         ], ignore_index=True)
        
#         # Calculate weights for all categories with equal weighting options
#         all_cat_weights = calculate_weights(all_cat_weights)
#         def apply_equal_weights(df, pillar_name, config_key):
#             if feature_engineering_config["equal_weightage"].get(config_key, False):
#                 n = df['pillar'].value_counts().get(pillar_name, 0)
#                 if n > 0:
#                     df.loc[df['pillar'] == pillar_name, 'weight'] = 1 / n

#         apply_equal_weights(all_cat_weights, 'awareness_pillar', 'give_awareness_metrics_equal_weightage')
#         apply_equal_weights(all_cat_weights, 'advocacy_pillar', 'give_advocacy_metrics_equal_weightage')
        
#         # Replicate all category weights across individual categories
#         all_cat_weights_expanded = pd.concat(
#             [all_cat_weights.assign(category=category) for category in cfa_results_by_cat_sub['category'].unique()],
#             ignore_index=True
#         )
        
#         # Combine all weights and format output
#         all_weights = pd.concat([by_cat_weights, all_cat_weights_expanded], ignore_index=True).rename(columns={'var': 'metric'})
#         weights_sheet = all_weights[['category', 'pillar', 'metric', 'weight']]
        
#         return weights_sheet

#     def weights_creation(fit_summary_all_cat, fit_summary_all_brands):
#         # Helper function to prepare the cfa_results subset based on model config criteria
#         def prepare_cfa_subset(cfa_data, exclude_pillars):
#             return cfa_data[
#                 (cfa_data['op'] == "=~") & 
#                 (cfa_data['Seed'] == 2) & 
#                 (~cfa_data['lhs'].isin(exclude_pillars))
#             ][['lhs', 'rhs', 'est.std', 'Brands', 'Category']].rename(columns={'lhs': 'Pillar', 'rhs': 'Var'})
        
#         # Initialize category and model configuration variables
#         category_list_pillar = eq_sub_scale_merged_brand['Category'].unique()
#         exclude_pillars = model_config["cfa"]["exclude_pillars"]
#         all_cat_pillars = model_config["cfa"]["pillars"]["all_category_pillars"]
#         by_cat_pillars = model_config["cfa"]["pillars"]["by_category_pillars"]

#         # Prepare subsets and rename columns as per config criteria
#         model_results = all_pillar_results.copy()
#         model_results['pillar'] = model_results['pillar'].apply(
#             lambda x: f"{x}_pillar" if not x.endswith("_pillar") else x
#         )

#         cfa_results_all_cat_sub = prepare_cfa_subset(fit_summary_all_cat, exclude_pillars)
#         cfa_results_by_cat_sub = prepare_cfa_subset(fit_summary_all_brands, exclude_pillars)

#         model_results_sub = model_results[['Shap Features', 'shap values', 'Brand', 'Category', 'pillar']].drop_duplicates()
#         model_results_sub.rename(columns={'Brand': 'brands', 'Shap Features': 'Var', 'pillar': 'Pillar'}, inplace=True)

#         # Aggregate model results based on configuration
#         if model_config['cfa']['perform_weighted_average_DV']:
#             model_results_sub_agg, model_results_pillar_all_cat = dv_weighted_shap(
#                 equity_dt, model_results_sub, run_config["end_date"], category_list_pillar
#             )
#         else:
#             model_results_sub_agg = model_results_sub.groupby(['category', 'pillar', 'var'], as_index=False)['shap values'].mean()
#             model_results_sub_agg.rename(columns={'shap values': 'Shap Value - mean'}, inplace=True)
#             model_results_pillar_all_cat = model_results_sub_agg.groupby(['pillar', 'var'], as_index=False)['Shap Value - mean'].mean()

#         # Call the helper function to prepare weights
#         weights_sheet = prepare_weights(
#             all_cat_pillars, by_cat_pillars, cfa_results_all_cat_sub, cfa_results_by_cat_sub,
#             model_results_pillar_all_cat, model_results_sub_agg
#         )

#         return weights_sheet



#     def create_pillar_scores(scaled_df, weights_sheet,req_cols):
#         # Initialize an empty DataFrame for the output
#         eq_sub2 = pd.DataFrame()
        
#         # Standardize metric names to lowercase
#         scaled_df.columns = scaled_df.columns.str.lower()
#         weights_sheet['metric'] = weights_sheet['metric'].str.lower()
        
#         # Calculate weighted sum for each category and pillar
#         for category in weights_sheet['category'].unique():
#             scaled_df_sub = scaled_df[scaled_df['category'] == category].copy().reindex(
#                 sorted(scaled_df.columns), axis=1
#             )
            
#             for pillar in weights_sheet['pillar'].unique():
#                 # Select weights for the current category and pillar
#                 metric_weights = weights_sheet[(weights_sheet['pillar'] == pillar) & (weights_sheet['category'] == category)]
#                 metrics_to_use = metric_weights['metric'].values
                
#                 # Get columns in data that match metrics from weights sheet
#                 relevant_cols = scaled_df_sub.columns.intersection(metrics_to_use)
#                 relevant_cols = sorted(relevant_cols)
#                 metric_weights = metric_weights[metric_weights['metric'].isin(relevant_cols)].sort_values(by='metric')
                
#                 # Calculate weighted scores for the pillar
#                 scaled_df_sub[pillar] = scaled_df_sub[relevant_cols].fillna(0).dot(metric_weights['weight'].values)
            
#             # Append the results to eq_sub2
#             eq_sub2 = pd.concat([eq_sub2, scaled_df_sub], ignore_index=True)
        
#         # Drop unnecessary columns if present
#         eq_sub2.drop(columns=['new_brand', 'v1'], inplace=True, errors='ignore')
        
#         # Transform to long format
#         index_df_long = pd.melt(eq_sub2, id_vars=['date', 'brand', 'category'], var_name='var')

#         # Prepare mapping for merging pillars
#         mapping_subset = req_cols[['idv_for_model_corrected', 'Equity Pillar', 'product_category_idv']].rename(
#             columns={'idv_for_model_corrected': 'var', 'product_category_idv': 'category'}
#         )
#         mapping_subset['var'] = mapping_subset['var'].str.lower()

#         # Merge pillar mappings
#         index_df_long = pd.merge(index_df_long, mapping_subset, on=['var', 'category'], how='left')
#         index_df_long['Equity Pillar'].fillna(index_df_long['var'], inplace=True)
        
#         # Prepare DV data and merge
#         dv_data = equity_dt[['date', 'brand_group_expanded', 'category', model_config["weights_model"]["DV"]]].copy()
#         dv_data.rename(columns={
#             'brand_group_expanded': 'brand', 
#             'market_share_total_sales': 'market_share'
#         }, inplace=True)
        
#         # Ensure date columns are properly formatted for merging
#         for df in [eq_sub2, dv_data, index_df_long]:
#             df['date'] = pd.to_datetime(df['date'], utc=False, format="%Y-%m-%d")
        
#         # Merge DV data with the pillar scores data
#         index_df_final1 = pd.merge(eq_sub2, dv_data, on=['date', 'brand', 'category'], how='left')
#         index_df_long = pd.merge(index_df_long, dv_data, on=['date', 'brand', 'category'], how='left')
        
#         return index_df_final1, index_df_long


#     def pillar_trend_creation(index_df):
#         # Determine rolling mean period based on time granularity
#         rolling_mean = 3 if run_config["time_granularity"] == "Monthly" else 13
        
#         df_data = index_df.copy()
#         # Identify pillar columns
#         pillar_cols = [col for col in df_data.columns if col.endswith('_pillar')]
#         # print('pillar_cols:', pillar_cols)

#         # Keep only relevant columns
#         df_data = df_data[['date', 'brand', 'category'] + pillar_cols + [run_config["DV"]]]
        
#         # Drop existing trend columns
#         trend_cols = [col for col in df_data.columns if col.endswith('_trend')]
#         df_data.drop(columns=trend_cols, inplace=True)
        
#         final_merged_df = pd.DataFrame()

#         # Iterate through each unique brand and category to calculate trends
#         for brand in df_data['brand'].unique():
#             brand_data = df_data[df_data['brand'] == brand]
            
#             for category in brand_data['category'].unique():
#                 br_cat_df = brand_data[brand_data['category'] == category]
                
#                 # Calculate rolling mean for each pillar
#                 for pillar in pillar_cols:
#                     past_3_ma = br_cat_df[pillar].rolling(rolling_mean).mean()
#                     past_future_3_ma = past_3_ma.shift(-1)
                    
#                     br_cat_df = br_cat_df.assign(
#                         **{f"{pillar}_trend_past": past_3_ma,
#                         f"{pillar}_trend_past_future": past_future_3_ma}
#                     )
                    
#                 final_merged_df = pd.concat([final_merged_df, br_cat_df], ignore_index=True)
        
#         return final_merged_df



#     def lag_addition(dataset):
#         # Initialize an empty DataFrame to store results
#         ret_df = pd.DataFrame()
        
#         # Identify columns that will not have lags
#         cols_no_lag = ['date', 'category', 'brand', run_config["DV"]]
#         basecols = dataset.columns.difference(cols_no_lag).tolist()

#         # Iterate over each unique brand and category to create lagged features
#         for brand in dataset['brand'].unique():
#             brand_data = dataset[dataset['brand'] == brand]
            
#             for category in brand_data['category'].unique():
#                 category_data = brand_data[brand_data['category'] == category]

#                 # Create lagged features for each base column
#                 for col in basecols:
#                     for lag in range(1, 4):  # Create lags 1 to 3
#                         category_data[f"{col}_lag_{lag}"] = category_data[col].shift(lag)
                        
#                 # Concatenate the modified DataFrame to the result
#                 ret_df = pd.concat([ret_df, category_data], ignore_index=True)
        
#         return ret_df

#     def pillar_importance_model(trend_pillars_data, sales_data):
#         df_data = trend_pillars_data
#         brand_df = df_data.copy()
#         brand_df['date'] = pd.to_datetime(brand_df['date'], format="%Y-%m-%d")
#         CV_NO = model_config["importance_model"]["cross_validation_number"]

#         # print("brand_df.shape :",brand_df.shape)

#         results_all_model = pd.DataFrame()
#         final_df = lag_addition(brand_df)
#         final_df.rename(columns={'category':'category_new'},inplace=True)

#         final_df.loc[final_df.category_new == "CAT TREATS ONLY","category"] = 'CAT FOOD'
#         final_df.loc[final_df.category_new == "DOG TREATS ONLY","category"] = 'DOG FOOD'
#         final_df.loc[final_df.category_new == "CAT FOOD","category"] = 'CAT FOOD'
#         final_df.loc[final_df.category_new == "DOG FOOD","category"] = 'DOG FOOD'
#         final_df.loc[final_df.category_new == "CAT LITTER","category"] = 'CAT LITTER'

#         rms_teneten_df = sales_data

#         rms_teneten_df['date'] = pd.to_datetime(rms_teneten_df['date'])

#         rms_teneten_df.rename(columns={'brand_group_expanded':'brand'},inplace=True)

#         final_df.drop(columns=['equalized_volume'],inplace=True)

#         final_df = final_df.merge(rms_teneten_df,on=['date', 'category', 'brand'],how='left')

#         # print(list(final_df.columns))

#         past_future_trend_cols = [cols for cols in final_df.columns if "_trend_past_future" in cols]
#         past_trend_cols = [cols for cols in final_df.columns if "_trend_past" in cols and cols not in past_future_trend_cols]
#         raw_cols = [cols for cols in final_df.columns if cols not in past_trend_cols and cols not in past_future_trend_cols]
#         corr_raw_df_full =pd.DataFrame()

#         ###run without price and distribution
#         if model_config["importance_model"]["price_and_acv_added"]:
#             final_df.drop(columns=['average_price','acv_selling'],inplace=True)

#         for brand in final_df.brand.unique():
#             br_df = final_df[final_df['brand'] == brand]
#             for category in br_df['category_new'].unique():
#                 cat_br_df = br_df[br_df['category_new'] == category]
#                 null_count_df = pd.DataFrame(cat_br_df.isnull().sum(),columns=['null count'])
#                 null_cols = null_count_df[null_count_df['null count'] > cat_br_df.shape[0] * 0.5].index
#                 if len(null_cols) > 0:
#                     cat_br_df.drop(columns=null_cols,inplace=True)
#                 cat_br_df = cat_br_df.iloc[3:,:]
#                 cat_br_df = cat_br_df.fillna(cat_br_df.mean())

#                 brand_perceptions_pillar = [cols for cols in cat_br_df.columns if cols.startswith('brand_perceptions')]
#                 product_feedback_pillar =  [cols for cols in cat_br_df.columns if cols.startswith('product_feedback')]
#                 advocacy_pillar = [cols for cols in cat_br_df.columns if cols.startswith('advocacy')]
#                 awareness_pillar = [cols for cols in cat_br_df.columns if cols.startswith('awareness')]
#                 consideration_pillar = [cols for cols in cat_br_df.columns if cols.startswith('consideration')]
#                 loyalty_pillar = [cols for cols in cat_br_df.columns if cols.startswith('loyalty')]

#                 pillars=[brand_perceptions_pillar,product_feedback_pillar,advocacy_pillar,awareness_pillar,consideration_pillar,loyalty_pillar]

#                 cat_br_pillar_df = cat_br_df.copy()
#                 flatten_pillars_list = list(chain.from_iterable(pillars))
#                 if model_config["importance_model"]["lags_added"]==False:
#                     flatten_pillars_list = [cols for cols in flatten_pillars_list if ("_lag") not in cols] # dropping lag metrics
#                 past_future_trend_cols = [cols for cols in flatten_pillars_list if "_trend_past_future" in cols]
#                 past_trend_cols = [cols for cols in flatten_pillars_list if "_trend_past" in cols and cols not in past_future_trend_cols]
#                 raw_cols = [cols for cols in flatten_pillars_list if cols not in past_trend_cols and cols not in past_future_trend_cols]

#                 brand_name = brand
#                 category_name = category


#                 k=2

#                 if model_config["importance_model"]["price_and_acv_added"]:
#                     cols_to_select = past_trend_cols+['average_price','rms_acv_selling']
#                 else:
#                     cols_to_select = past_trend_cols
#                 for j in [cols_to_select]:
#                     # print('j:',j,"-",brand,"-",category)
#                     k=k+1
#                     matched_cols=[cols for cols in j if cols in cat_br_pillar_df.columns]
#                     print("cat_br_pillar_df cols:",list(cat_br_pillar_df.columns))
#                     modeldf = cat_br_pillar_df[matched_cols+[model_config["importance_model"]["DV"]]]

#                     idvs = modeldf.drop(model_config["importance_model"]["DV"], 1)       # feature matrix

#                     if model_config["importance_model"]["log_convert_DV"]:
#                         dv = np.log1p(modeldf[model_config["importance_model"]["DV"]])
#                     else:
#                         dv = modeldf[model_config["importance_model"]["DV"]]
#                     if idvs.shape[1] > 2:
#                         if model_config["importance_model"]["standardize"]:
#                             mmscaler = StandardScaler() # Do even before feature selection
#                             idvs_scaled = pd.DataFrame(mmscaler.fit_transform(idvs), columns = idvs.columns)
#                             idvs = idvs_scaled.copy()
#                             if len(idvs_scaled.columns) >= 2:
#                                 for i in [2]:
#                                     # print(i)
#                                     if model_config["importance_model"]["Time_series_split"]:
#                                         train_x = idvs[idvs_scaled.columns].iloc[:-6,:]
#                                         train_y = dv[:-6]
#                                         test_x = idvs[idvs_scaled.columns].iloc[-6:,:]
#                                         test_y = dv[-6:]
#                                     if model_config["importance_model"]["Random_seed_split"]:
#                                         train_x, test_x, train_y, test_y = train_test_split(idvs[idvs_scaled.columns],dv,test_size=6, random_state=i, shuffle=True)
#                                     train_x_all =  idvs[idvs_scaled.columns]
#                                     train_y_all = dv

#                                     X_test_hold = test_x.copy()
#                                     y_test_hold = test_y.copy()


#                                     X_train = train_x.copy()
#                                     y_train = train_y.copy()

#                                     feat_importance = pd.DataFrame()
#                                     feat_df=pd.DataFrame()

#                                     feat_importance = pd.DataFrame()
#                                     feat_df=pd.DataFrame()

#                                     if model_config["importance_model"]["Corr_file_generation"]:
#                                         correlation_mat = modeldf.corr()
#                                         corr_pairs = correlation_mat.unstack()
#                                         corr_pairs = corr_pairs.reset_index()
#                                         corr_pairs.rename(columns = {'level_0':'Variable 1','level_1':'Variable 2',0:'Corelation Co efficient'},inplace=True)
#                                         corr_pairs = corr_pairs[corr_pairs['Variable 1']!=corr_pairs['Variable 2']]
#         #                                 corr_raw_df_full[corr_raw_df_full['Variable 1'] == 'market_share']
#                                         corr_pairs['Brand'] = brand
#                                         corr_pairs['Category'] = category
#                                         corr_raw_df_full = pd.concat([corr_raw_df_full,corr_pairs])

#                                     if run_config["importance_model"]["Brute_force"]["run"]:
#                                             param_rf_ridge = {'alpha':model_config["importance_model"]["hyperparameters"]["Brute_force"]["grid_search"]["alpha"], "random_state":model_config["importance_model"]["hyperparameters"]["Brute_force"]["grid_search"]["random_state"]}
#                                             ridg3=Ridge(positive=model_config["importance_model"]["hyperparameters"]["Brute_force"]["positive"],random_state=model_config["importance_model"]["hyperparameters"]["Brute_force"]["random_state"])
#                                             search_brute = GridSearchCV(ridg3,param_rf_ridge,cv=CV_NO,scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train, y_train)
#                                             fit_model_brute=Ridge(alpha=search_brute.best_params_["alpha"],random_state=model_config["importance_model"]["hyperparameters"]["Brute_force"]["random_state"]).fit(X_train, y_train)



#                                             stat_df = pd.DataFrame()
#                                             params = np.append(fit_model_brute.intercept_,fit_model_brute.coef_)
#                                             stat_df["coefficients"] = list(params)
#                                             features = ['intercept']+list(X_train.columns)
#                                             stat_df.insert(0,"features", features)
#                                             feat_import = stat_df.sort_values(by='coefficients',ascending=False)
#                                             # print(feat_import)

#                                             explainer = shap.LinearExplainer(fit_model_brute,X_train)
#                                             shap_values = explainer.shap_values(X_train)
#                                             feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train.columns)).mean(),columns=['shap values'])
#                                             # print("Brute force shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                             feat_df = feat_df.sort_values(by='shap values',ascending=False)

#                                             y_pred_train = fit_model_brute.predict(X_train)
#                                             y_pred_test = fit_model_brute.predict(X_test_hold)
#                                             y_pred_all = fit_model_brute.predict(train_x_all[list(X_train.columns)])


#                                             feat_importance = feat_import.reset_index().rename(columns={'features':'Features','coefficients':'Feature Importance/coefficient'})
#                                             feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                             mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                             mse_train = metrics.mean_squared_error(y_train, y_pred_train)
#                                             rmse_train = np.sqrt(mse_train) #mse**(0.5)
#                                             r2_train = metrics.r2_score(y_train, y_pred_train)
#                                             mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                             results_all_model_brute = pd.concat([feat_importance,feat_df], axis=1)
#                                             results_all_model_brute['Model'] = "Brute Force Model"
#                                             results_all_model_brute['Brand'] = brand
#                                             results_all_model_brute['Category'] = category
#                                             results_all_model_brute['Latest MS'] = dv.values[-1]

#                                             results_all_model_brute['R2_Score_Train'] = r2_train
#                                             results_all_model_brute['MAPE_Train'] = mape_train
#                                             results_all_model_brute['R2_score_fold'] = cross_val_score(fit_model_brute,X_train,y_train,cv=CV_NO,scoring='r2').mean()
#                                             results_all_model_brute['MAPE_fold'] = cross_val_score(fit_model_brute,X_train,y_train,cv=CV_NO,scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                             results_all_model_brute['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                             results_all_model_brute['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                             results_all_model_brute['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                             results_all_model_brute['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)
#                                             results_all_model_brute['Pillar'] = 'All Pillars'
#                                             results_all_model_brute['Best_Params_Gridsearchcv']=str(search_brute.best_params_)
#                                             results_all_model = pd.concat([results_all_model, results_all_model_brute],axis=0)

#                                     feat_importance = pd.DataFrame()
#                                     feat_df=pd.DataFrame()
#                                     if run_config["importance_model"]["RandomForest"]["run"]:
#                                         param_grid_rf = {"max_depth":model_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["max_depth"],"n_estimators": model_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["n_estimators"],'max_features': model_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["max_features"],"random_state":model_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["random_state"]}
#                                         rf1=RandomForestRegressor(random_state=model_config["importance_model"]["hyperparameters"]["RandomForest"]["random_state"])
#                                         search_rf = GridSearchCV(rf1, param_grid_rf,cv=CV_NO,scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train, y_train)
#                                         rf=RandomForestRegressor(n_estimators  = search_rf.best_params_["n_estimators"],max_depth = search_rf.best_params_["max_depth"],random_state=model_config["importance_model"]["hyperparameters"]["RandomForest"]["random_state"])
#                                         rf.fit(X_train,y_train)
#                                         features = list(X_train.columns)
#                                         f_i = list(zip(features,rf.feature_importances_))
#                                         f_i.sort(key = lambda x : x[1],reverse=True)

#                                         rfe = RFECV(rf,cv=CV_NO,scoring='neg_mean_absolute_percentage_error')
#                                         rfe.fit(X_train,y_train)
#                                         selected_features = list(np.array(features)[rfe.get_support()])
#                                         # print(selected_features)
#                                         feat_importance = pd.DataFrame(f_i,columns=['Features','Feature Importance'])
#                                         feat_importance.set_index('Features',inplace=True)
#                                         feat_importance = feat_importance.iloc[:20,:]
#                                         # print(feat_importance)
#                                         best_features = list(feat_importance.index)

#                                         explainer = shap.TreeExplainer(rf)
#                                         shap_values = explainer.shap_values(X_train)
#                                         feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train.columns)).mean(),columns=['shap values'])
#                                         # print("Random Forest shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                         feat_df = feat_df.sort_values(by='shap values',ascending=False)


#                                         y_pred_test = rf.predict(X_test_hold)
#                                         y_pred_train = rf.predict(X_train)
#                                         y_pred_all = rf.predict(train_x_all[list(X_train.columns)])

#                                         feat_importance = feat_importance.reset_index().rename(columns={'Feature Importance':'Feature Importance/coefficient'})
#                                         feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                         mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                         mse_train = metrics.mean_squared_error(y_train, y_pred_train)

#                                         rmse_train = np.sqrt(mse_train)
#                                         r2_train = metrics.r2_score(y_train, y_pred_train)
#                                         mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                         feat_df

#                                         results_all_model_RF = pd.concat([feat_importance,feat_df], axis=1)
#                                         results_all_model_RF['Model'] = "Random Forest"
#                                         results_all_model_RF['Brand'] = brand
#                                         results_all_model_RF['Category'] = category

#                                         results_all_model_RF['Latest DV'] = dv.values[-1]

#                                         results_all_model_RF['R2_Score_Train'] = r2_train
#                                         results_all_model_RF['MAPE_Train'] = mape_train
#                                         results_all_model_RF['R2_score_fold'] = cross_val_score(rf,X_train,y_train,cv=CV_NO,scoring='r2').mean()
#                                         results_all_model_RF['MAPE_fold'] = cross_val_score(rf,X_train,y_train,cv=CV_NO,scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                         results_all_model_RF['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                         results_all_model_RF['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                         results_all_model_RF['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                         results_all_model_RF['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)
#                                         results_all_model_RF['Pillar'] = 'All Pillars'
#                                         results_all_model_RF['Best_Params_Gridsearchcv']=str(search_rf.best_params_)
#                                         results_all_model = pd.concat([results_all_model, results_all_model_RF],axis=0)
#                                         title=(brand_name+"-"+category_name+"-"+"Random_Forest")


#                                     feat_importance = pd.DataFrame()
#                                     feat_df=pd.DataFrame()
#                                     if run_config["importance_model"]["XGBoost"]["run"]:
#                                         param_grid = {"max_depth": model_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["max_depth"],"n_estimators": model_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["n_estimators"],"learning_rate": model_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["learning_rate"],"random_state":model_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["random_state"]}
#                                         regressor=xgb.XGBRegressor(eval_metric='mape',random_state=model_config["importance_model"]["hyperparameters"]["XGBoost"]["random_state"])
#                                         search_xgb = GridSearchCV(regressor, param_grid,cv=CV_NO,scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train, y_train)
#                                         # print("The best hyperparameters are ",search_xgb.best_params_)

#                                         regressor=xgb.XGBRegressor(learning_rate = search_xgb.best_params_["learning_rate"],
#                                                     n_estimators  = search_xgb.best_params_["n_estimators"],
#                                                     max_depth     = search_xgb.best_params_["max_depth"],
#                                                     eval_metric='mape',random_state=model_config["importance_model"]["hyperparameters"]["XGBoost"]["random_state"])

#                                         regressor.fit(X_train, y_train)
#                                         dict_result = regressor.get_booster().get_score(importance_type='gain')
#                                         # print("Feature importance XGBoost",pd.DataFrame(dict_result.items(),columns=['Feature','gain']).sort_values(by ='gain',ascending=False))
#                                         feat_importance = pd.DataFrame(dict_result.items(),columns=['Feature','gain']).sort_values(by ='gain',ascending=False)
#                                         explainer = shap.TreeExplainer(regressor)
#                                         shap_values = explainer.shap_values(X_train)
#                                         feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train.columns)).mean(),columns=['shap values'])
#                                         # print("XGBoost shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                         feat_df = feat_df.sort_values(by='shap values',ascending=False)

#                                         y_pred_test = regressor.predict(X_test_hold)
#                                         y_pred_train = regressor.predict(X_train)
#                                         y_pred_all = regressor.predict(train_x_all[list(X_train.columns)])

#                                         feat_importance = feat_importance.reset_index().rename(columns={'Feature':'Features','gain':'Feature Importance/coefficient'})
#                                         feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                         mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                         mse_train = metrics.mean_squared_error(y_train, y_pred_train)
#                                         rmse_train = np.sqrt(mse_train) #mse**(0.5)
#                                         r2_train = metrics.r2_score(y_train, y_pred_train)
#                                         mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                         results_all_model_XGB = pd.concat([feat_importance,feat_df], axis=1)
#                                         results_all_model_XGB['Model'] = "XGBoost"
#                                         results_all_model_XGB['Brand'] = brand
#                                         results_all_model_XGB['Category'] = category
#                                         results_all_model_XGB['Latest DV'] = dv.values[-1]

#                                         results_all_model_XGB['R2_Score_Train'] = r2_train
#                                         results_all_model_XGB['MAPE_Train'] = mape_train
#                                         results_all_model_XGB['R2_score_fold'] = cross_val_score(regressor,X_train,y_train,cv=CV_NO,scoring='r2').mean()
#                                         results_all_model_XGB['MAPE_fold'] = cross_val_score(regressor,X_train,y_train,cv=CV_NO,scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                         results_all_model_XGB['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                         results_all_model_XGB['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                         results_all_model_XGB['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                         results_all_model_XGB['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)
#                                         results_all_model_XGB['Pillar'] = 'All Pillars'
#                                         results_all_model_XGB['Best_Params_Gridsearchcv']=str(search_xgb.best_params_)
#                                         results_all_model = pd.concat([results_all_model, results_all_model_XGB],axis=0)

#         return results_all_model



#     def scaled_scores(pillar_data):
#         # Convert date column to datetime
#         pillar_data['date'] = pd.to_datetime(pillar_data['date'], format="%Y-%m-%d")
        
#         # Define relevant columns for grouping and scaling
#         pillar_columns = [col for col in pillar_data.columns if col.endswith('_pillar')]
#         additional_columns = ['date', 'category', run_config["DV"], 'brand']

#         if feature_engineering_config["scaled_score"]["only_pillars"]:
#             grouped = pillar_data.groupby(['date', 'category'])[pillar_columns].mean().reset_index()
#             df1 = pillar_data[additional_columns + pillar_columns]
#         else:
#             columns_to_mean = pillar_data.columns.difference(['date', 'brand', 'category', run_config["DV"]]).tolist()
#             grouped = pillar_data.groupby(['date', 'category'])[columns_to_mean].mean().reset_index()
#             df1 = pillar_data

#         # Prepare to store processed dataframes
#         processed_dataframes = []

#         # Extract unique combinations of date and category
#         unique_combinations = df1[['date', 'category']].drop_duplicates()

#         # Scaling logic
#         for _, group in unique_combinations.iterrows():
#             date = group["date"]
#             category = group['category']

#             subset1 = df1[(df1['date'] == date) & (df1['category'] == category)]
#             subset = grouped[(grouped['date'] == date) & (grouped['category'] == category)]

#             for column_name in (pillar_columns if feature_engineering_config["scaled_score"]["only_pillars"] else columns_to_mean):
#                 new_column_name = f"{column_name}_scores"
#                 average = subset[column_name].values[0]  # Assumes there's at least one row in subset
#                 subset1[new_column_name] = (subset1[column_name] - average) * 100 + 100
            
#             processed_dataframes.append(subset1)

#         result_df = pd.concat(processed_dataframes, ignore_index=True)

#         # Melt the result DataFrame
#         columns_to_pivot = [col for col in result_df.columns if col not in ['date', 'brand', 'category', run_config["DV"]]]
#         melted_df = pd.melt(result_df, id_vars=['date', 'brand', 'category'], value_vars=columns_to_pivot, value_name='column_value')

#         return result_df, melted_df



#     def data_hub_data_format_code(pillars_long_data, weights_sheet, scaled_scores_long, imp_model_results, idv_list):
#         # Preprocess pillars_long_data
#         attr_df = pillars_long_data

#         # Preprocess weights_sheet
#         attr_df1 = weights_sheet.copy()
#         attr_df1.columns = (attr_df1.columns
#                             .str.replace(' - ', '_')
#                             .str.replace(' ', '_')
#                             .str.lower())

#         # Rename columns for merging
#         df = attr_df1.rename(columns={'metric': 'var', 'pillar': 'Equity Pillar'})
#         df['Equity Pillar'] = df['Equity Pillar'].str.replace('_pillar', '')
        
#         # Merge pillar data with weights
#         merged_df = pd.merge(attr_df, df[['category', 'var', 'Equity Pillar', 'weight']], 
#                             on=['category', 'var', 'Equity Pillar'], how='left')

#         # Calculate metric contribution
#         merged_df['metric contribution'] = merged_df['value'] * merged_df['weight']
        
#         # Date processing
#         merged_df['date'] = pd.to_datetime(merged_df['date'], format="%Y-%m-%d")
#         merged_df['year'] = merged_df['date'].dt.year
#         merged_df['month'] = merged_df['date'].dt.month

#         # Calculate category average and differences
#         merged_df['Category_MC_Avg'] = merged_df.groupby(['date', 'category', 'var'])['metric contribution'].transform('mean')
#         merged_df['MC diff to catg avg'] = merged_df['metric contribution'] - merged_df['Category_MC_Avg']

#         # Identify if Equity Pillar
#         merged_df['Is Pillar'] = merged_df['Equity Pillar'].str.endswith('_pillar') & (merged_df['Equity Pillar'] != '')

#         # Preprocess scaled scores
#         filtered_df = scaled_scores_long[scaled_scores_long['variable'].str.endswith('_scores')].copy()
#         filtered_df['var'] = filtered_df['variable'].str.replace('_scores$', '', regex=True)
#         filtered_df['date'] = pd.to_datetime(filtered_df['date'], format="%Y-%m-%d")

#         # Merge with scaled scores
#         merged_df1 = pd.merge(merged_df, filtered_df[['date', 'brand', 'category', 'var', 'column_value']],
#                             on=['date', 'brand', 'category', 'var'], how='left')
#         merged_df1 = merged_df1.rename(columns={'column_value': 'Scaled Values'})

#         # Preprocess importance model results
#         filtered_imp_model_results = imp_model_results[imp_model_results['Model'] == 'Random Forest'][['Brand', 'Category', 'Shap Features', 'shap values']].copy()
        
#         # Calculate the sum of SHAP values and relative importance
#         sum_shap_values = filtered_imp_model_results.groupby(['Brand', 'Category'])['shap values'].transform('sum')
#         filtered_imp_model_results['Relative Importance'] = filtered_imp_model_results['shap values'] / sum_shap_values

#         # Preprocess IDV list
#         attr_df4 = idv_list.rename(columns={'idv_for_model_corrected': 'var'}).drop_duplicates()
#         selected_columns = merged_df1[['var']]
        
#         # Merge to get metric details
#         merged_df3 = pd.merge(selected_columns, attr_df4[['var', 'metric_name', 'data_source', 'Vendor Metric Group']], on=['var'], how='left')

#         # Classify metric type
#         merged_df3['Metric Type'] = np.select(
#             [merged_df3['var'].str.contains('_mean'),
#             merged_df3['var'].str.contains('_rank_1st'),
#             merged_df3['var'].str.contains('_net')],
#             ['mean', 'rank_1st', 'net'],
#             default=''
#         )

#         return merged_df1, filtered_imp_model_results, merged_df3



#     def scorecard_summary(df):
#         df = df.drop(columns=['value', 'metric contribution','Category_MC_Avg', 'MC diff to catg avg'])
#         df["Is Pillar"] = np.where(df["Is Pillar"] == False, "Metric", "Pillar")
#         return df


#     def updated_scorecard_format(detailed, harmonized_data, pillar_importances, dashboard_metric_names_mapping, price_class_mapping):
        
#         def prepare_harmonized_data(harmonized_data):
#             harmonized_data["date"] = pd.to_datetime(harmonized_data["date"])
#             df_long = pd.melt(harmonized_data, id_vars=['brand_group_expanded', 'category', 'date'], 
#                             var_name='var', value_name='metric_raw_value')
#             df_long = df_long.rename(columns={"brand_group_expanded": "brand"})
#             return df_long
        
#         def merge_and_prepare_summary(detailed, df_long, price_class_mapping, dashboard_metric_names_mapping):
#             price_class_mapping["Price Class"] = price_class_mapping["Price Class"].fillna("#NA")
#             detailed["date"] = pd.to_datetime(detailed["date"])
            
#             summary_hd = pd.merge(detailed, df_long, on=["brand", "category", "var", "date"], how="left")
#             summary_hd_price = pd.merge(summary_hd, price_class_mapping, 
#                                         left_on=["brand", "category"], right_on=["Brand", "Category"], how="left")
            
#             dashboard_metric_names_mapping = dashboard_metric_names_mapping.drop(
#                 columns=['Harmonized_attribute', 'Transformation_logic', 'Modelled'])
            
#             summary_renamed = pd.merge(summary_hd_price, dashboard_metric_names_mapping, 
#                                     left_on=["var"], right_on=["Scorecard_attribute"], how="left")
#             summary_renamed['var'] = summary_renamed['Scorecard_attribute_updated'].fillna(summary_renamed['var'])
#             summary_renamed = summary_renamed.drop(columns=['Scorecard_attribute', 'Scorecard_attribute_updated'])
            
#             summary_renamed['Equity Pillar'] = summary_renamed['Equity Pillar'].str.replace('_pillar', '', regex=False) \
#                                                                             .str.replace('_', ' ') \
#                                                                             .str.title()
#             return summary_renamed

#         def clean_and_rename_summary(summary_df):
#             pillar_mapping = {
#                 'brand_perceptions_pillar': 'Brand Perceptions',
#                 'loyalty_pillar': 'Loyalty',
#                 'advocacy_pillar': 'Advocacy',
#                 'awareness_pillar': 'Awareness',
#                 'consideration_pillar': 'Consideration',
#                 'product_feedback_pillar': 'Product Feedback'
#             }
            
#             summary_df['var'] = summary_df['var'].replace(pillar_mapping)
#             summary_df = summary_df.rename(columns={
#                 'Equity Pillar': 'equity_pillar',
#                 'Is Pillar': 'metric_type',
#                 'var': 'metric_name',
#                 'Scaled Values': 'scaled_scores',
#                 'Price Class': 'price_class'
#             })
            
#             summary_df = summary_df.drop(columns=["Brand", "Category", "year", "month"])
            
#             selected_columns = ['brand', 'category', 'price_class', 'equity_pillar', 'metric_type', 
#                                 'metric_name', 'date', 'metric_raw_value', 'weight', 'scaled_scores']
#             summary_df = summary_df[selected_columns]
#             summary_df = summary_df.sort_values(by=['brand', 'category', 'equity_pillar', 
#                                                     'metric_type', 'metric_name', 'date'], ignore_index=True)
#             summary_df = summary_df[~(summary_df['weight'].isnull() & (summary_df['metric_type'] == 'Metric'))]
            
#             return summary_df

#         def round_columns(summary_df):
#             numeric_cols = ['metric_raw_value', 'weight', 'scaled_scores']
#             for col, decimals in zip(numeric_cols, [5, 5, 2]):
#                 summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').round(decimals)
#             return summary_df

#         def prepare_pillar_importances(pillar_importances, price_class_mapping):
#             updated_pillars = pd.merge(pillar_importances, price_class_mapping, on=["Brand", "Category"], how="left")
#             updated_pillars = updated_pillars.rename(columns={
#                 "Shap Features": "equity_pillar",
#                 "Relative Importance": "relative_importance",
#                 "Price Class": "price_class",
#                 "Brand": "brand",
#                 "Category": "category"
#             })
            
#             updated_pillars = updated_pillars[["brand", "category", "price_class", "equity_pillar", "relative_importance"]]
#             updated_pillars['equity_pillar'] = updated_pillars['equity_pillar'].str.replace('_pillar_trend_past', '', regex=False) \
#                                                                             .str.replace('_', ' ') \
#                                                                             .str.title()
#             updated_pillars = updated_pillars.sort_values(by=['brand', 'category', 'price_class', 'equity_pillar'], ignore_index=True)
#             updated_pillars['relative_importance'] = pd.to_numeric(updated_pillars['relative_importance'], errors='coerce').round(4)
            
#             return updated_pillars

#         # Main processing
#         df_long = prepare_harmonized_data(harmonized_data)
#         summary_hd_price_renamed = merge_and_prepare_summary(detailed, df_long, price_class_mapping, dashboard_metric_names_mapping)
#         updated_summary = clean_and_rename_summary(summary_hd_price_renamed)
#         updated_summary = round_columns(updated_summary)
#         updated_pillar_importances = prepare_pillar_importances(pillar_importances, price_class_mapping)
        
#         return updated_summary, updated_pillar_importances
#     print("post modelling 1- staging_output_path:",staging_output_path)
#     eq_sub_scale_merged_brand = pd.read_csv(output_config["data_prep"]["eq_sub_scale"], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     equity_dt = pd.read_csv(output_config["data_prep"]["equity_dt"], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))
#     # Convert the 'date' column to datetime format
#     equity_dt['date'] = pd.to_datetime(equity_dt['date'], utc=False)

#     processed_harmonized_data = pd.read_csv(output_config['processed_input_data'], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     output_file_path = f"{static_output_path}/dashboard_metric_names_mapping.csv"
#     dashboard_metric_names_mapping = pd.read_csv(output_file_path, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     output_file_path = f"{static_output_path}/price_class_mapping.csv"
#     price_class_mapping = pd.read_csv(output_file_path, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     output_file_path = f"{static_output_path}/idv_list.csv"
#     req_cols = pd.read_csv(output_file_path, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     if run_config["refresh_type"] == "full":
#         all_pillar_results = pd.read_csv(output_config["weights_model"]["model_results"], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#         nielsen_rms_data = pd.read_csv(output_config['processed_sales_data'], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     print("post modelling 2- staging_output_path:",staging_output_path)

#     if run_config["refresh_type"] == "full":
#         fit_summary_all_cat_py = pd.read_csv(output_config["cfa"]["model_results_all_category"], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#         fit_summary_all_brands_py = pd.read_csv(output_config["cfa"]["model_results_by_category"], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#         weights_sheet = weights_creation(fit_summary_all_cat_py,fit_summary_all_brands_py)

#     if run_config["refresh_type"] == "full":
#         weights_sheet.to_csv(output_config["pillar_creation"]["weights_sheet"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     if run_config["refresh_type"] == "scoring":
#         weights_sheet = pd.read_csv(input_config["weights_sheet"], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))
#     # Rename all columns to lowercase
#     weights_sheet.columns = weights_sheet.columns.str.lower()
#     eq_sub_scale_merged_brand_stacked = eq_sub_scale_merged_brand[eq_sub_scale_merged_brand['New_Brand'] == "Stacked Brand"]

#     index_df, index_df_long = create_pillar_scores(eq_sub_scale_merged_brand_stacked, weights_sheet,req_cols)

#     index_df_long.to_csv(output_config["pillar_creation"]["pillars_long_format"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))
#     index_df.to_csv(output_config["pillar_creation"]["pillars"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     index_df = pd.read_csv(output_config["pillar_creation"]["pillars"], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     final_merged_df = pillar_trend_creation(index_df)
#     final_merged_df.to_csv(output_config["trend_pillar"]["trend_pillars"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     if (run_config["refresh_type"] == "full") | ((run_config["refresh_type"] == "scoring") & (run_config["run_importance_model_for_scoring_refresh"] == True)):
#         results_all_model = pillar_importance_model(final_merged_df, nielsen_rms_data)

#     if (run_config["refresh_type"] == "full") | ((run_config["refresh_type"] == "scoring") & (run_config["run_importance_model_for_scoring_refresh"] == True)):
#         results_all_model.to_csv(output_config["importance_model"]["model_results"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     if ((run_config["refresh_type"] == "scoring") & (run_config["run_importance_model_for_scoring_refresh"] == False)):
#         results_all_model = pd.read_csv('abfss://restricted-dataoperations@npusdvdatalakesta.dfs.core.windows.net/staging/cmi_brand_hub/score_card_data/output/random_forest2/model_results_all_cat.csv', **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     scaled_scores, scaled_scores_long = scaled_scores(index_df)

#     scaled_scores.to_csv(output_config["scaled_scores"]["scaled_pillars"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))
#     scaled_scores_long.to_csv(output_config["scaled_scores"]["scaled_pillars_long_format"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     index_df_long = pd.read_csv(output_config["pillar_creation"]["pillars_long_format"], **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))



#     hub_data, pillar_importances, var_map = data_hub_data_format_code(pillars_long_data=index_df_long, weights_sheet=weights_sheet, scaled_scores_long=scaled_scores_long, imp_model_results=results_all_model, idv_list=req_cols)

#     hub_data, pillar_importances, var_map = data_hub_data_format_code(pillars_long_data=index_df_long, weights_sheet=weights_sheet, scaled_scores_long=scaled_scores_long, imp_model_results=results_all_model, idv_list=req_cols)

#     hub_data.to_csv(output_config["pillar_importances"]["hub_data"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))
#     pillar_importances.to_csv(output_config["pillar_importances"]["pillar_importances"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))
#     var_map.to_csv(output_config["pillar_importances"]["variable_mapping"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     scorecard_detailed = hub_data.copy()
#     scorecard_summary = scorecard_summary(scorecard_detailed)

#     hub_data.to_csv(output_config["scorecard"]["detailed"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     scorecard_summary.to_csv(output_config["scorecard"]["summary"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     pillar_importances.to_csv(output_config["scorecard"]["pillar_importances"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))

#     updated_summary, updated_pillar_importances = updated_scorecard_format(scorecard_detailed, processed_harmonized_data, pillar_importances, dashboard_metric_names_mapping, price_class_mapping)

#     print("post modelling 3- staging_output_path:",staging_output_path)
#     print("summary path:",output_config["updated_scorecard"]["updated_summary"])
#     updated_summary.to_csv(output_config["updated_scorecard"]["updated_summary"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))
#     updated_pillar_importances.to_csv(output_config["updated_scorecard"]["updated_pillar_importances"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))



# COMMAND ----------

# MAGIC %md
# MAGIC after cleaning(v3)

# COMMAND ----------

def calculate_category_sums(equity_data, dv_column):
    """
    Aggregates DV column to calculate mean for each brand/category 
    and sum for each category.
    """
    equity_data_agg = equity_data.groupby(['category', 'brand_group_expanded'])[dv_column].mean().reset_index()
    if dv_column == "market_share_total_sales":
        equity_data_agg.rename(columns={dv_column: 'Market share - mean'}, inplace=True)
        category_sums = equity_data_agg.groupby('category')['Market share - mean'].sum().reset_index()
        category_sums.rename(columns={'Market share - mean': 'category_Market share - sum'}, inplace=True)
    elif dv_column == "equalized_volume":
        equity_data_agg.rename(columns={dv_column: 'Equalized Volume - mean'}, inplace=True)
        category_sums = equity_data_agg.groupby('category')['Equalized Volume - mean'].sum().reset_index()
        category_sums.rename(columns={'Equalized Volume - mean': 'category_Equalized Volume - sum'}, inplace=True)
    elif dv_column == "total_sales":
        equity_data_agg.rename(columns={dv_column: 'Total Sales - mean'}, inplace=True)
        category_sums = equity_data_agg.groupby('category')['Total Sales - mean'].sum().reset_index()
        category_sums.rename(columns={'Total Sales - mean': 'category_Total Sales - sum'}, inplace=True)
    
    return equity_data_agg, category_sums

# COMMAND ----------

def dv_weighted_shap(df, model_results_sub, end_date, categories_list, DV_previous_years_to_take=1):
    # Preprocess and filter the DataFrame
    dv_column = feat_eng_config["weights_model"]["DV"]
    df_sub = df[['date', 'brand_group_expanded', 'category', dv_column]].copy()
    df_sub = df_sub[df_sub['category'].isin(categories_list)]
    df_sub[dv_column] = pd.to_numeric(df_sub[dv_column], errors='coerce')
    df_sub.dropna(inplace=True)

    # Filter data based on date range
    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.DateOffset(years=DV_previous_years_to_take)
    df_sub = df_sub[(df_sub['date'] >= start_date) & (df_sub['date'] <= end_date)]

    # Calculate category sums and merge them with aggregated data
    equity_data_agg, category_sums = calculate_category_sums(df_sub, dv_column)
    equity_data_agg = equity_data_agg.merge(category_sums, on='category', how='left')

    # Prepare model results DataFrame for merging
    model_results_sub.columns = model_results_sub.columns.str.lower()
    model_results_sub.rename(columns={'category': 'category', 'brand_group_expanded': 'brands'}, inplace=True)
    merged_results = model_results_sub.merge(equity_data_agg, left_on=['brands', 'category'], right_on=['brand_group_expanded', 'category'], how='left')

    # Calculate weighted SHAP values based on DV type
    shap_column_map = {
        "market_share_total_sales": ('Market share - mean', 'category_Market share - sum'),
        "equalized_volume": ('Equalized Volume - mean', 'category_Equalized Volume - sum'),
        "total_sales": ('Total Sales - mean', 'category_Total Sales - sum')
    }

    mean_column, sum_column = shap_column_map.get(dv_column, (None, None))
    if mean_column and sum_column:
        merged_results['weighted_shap'] = (merged_results['shap values'] * merged_results[mean_column]) / merged_results[sum_column]

    # Aggregate results at category and overall levels
    category_agg = merged_results.groupby(['category', 'pillar', 'var'])['weighted_shap'].mean().reset_index()
    category_agg.rename(columns={'weighted_shap': 'Shap Value - mean'}, inplace=True)

    overall_agg = category_agg.groupby(['pillar', 'var'])['Shap Value - mean'].mean().reset_index()

    return category_agg, overall_agg

# COMMAND ----------

# Standardize column names and format 'var' columns
def standardize_columns(*dfs):
    for df in dfs:
        df.columns = [col.lower() for col in df.columns]
        if 'var' in df.columns:
            df['var'] = df['var'].str.lower()

# COMMAND ----------

# Merge model and CFA results and filter by specified pillars
def merge_and_filter(model_df, cfa_df, pillars, by_category=False):
    if by_category:
        merged_df = pd.merge(model_df, cfa_df, on=['var', 'pillar', 'category'], how='outer')
    else:
        merged_df = pd.merge(model_df, cfa_df, on=['var', 'pillar'], how='outer')
    return merged_df[merged_df['pillar'].isin(pillars)]

# COMMAND ----------

# Calculate percentage weights
def calculate_weights(df, shap_col='shap value - mean', cfa_col='est.std'):
    df['abs_shap'] = df[shap_col].abs()
    df['abs_cfa_est'] = df[cfa_col].abs()
    
    sum_shap = df.groupby('pillar')['abs_shap'].sum().rename('metrics_sum_shap').reset_index()
    sum_cfa = df.groupby('pillar')['abs_cfa_est'].sum().rename('metrics_sum_cfa').reset_index()
    
    df = df.merge(sum_shap, on='pillar', how='left').merge(sum_cfa, on='pillar', how='left')
    df['perc_shap'] = df['abs_shap'] / df['metrics_sum_shap']
    df['perc_cfa'] = df['abs_cfa_est'] / df['metrics_sum_cfa']
    df['weight'] = df[['perc_cfa', 'perc_shap']].mean(axis=1, skipna=True)
    return df

# COMMAND ----------

def apply_equal_weights(df, pillar_name, config_key):
    if filter_config["equal_weightage"].get(config_key, False):
        n = df['pillar'].value_counts().get(pillar_name, 0)
        if n > 0:
            df.loc[df['pillar'] == pillar_name, 'weight'] = 1 / n

# COMMAND ----------

def prepare_weights(all_cat_pillars, by_cat_pillars, cfa_results_all_cat_sub, cfa_results_by_cat_sub,
                        model_results_pillar_all_cat, model_results_sub_agg):

    standardize_columns(model_results_sub_agg, cfa_results_by_cat_sub, model_results_pillar_all_cat, cfa_results_all_cat_sub)
    
    all_cat_weights = merge_and_filter(model_results_pillar_all_cat, cfa_results_all_cat_sub, all_cat_pillars)
    by_cat_weights_merged = merge_and_filter(model_results_sub_agg, cfa_results_by_cat_sub, by_cat_pillars, by_category=True)
    
    # Calculate weights for each category
    by_cat_weights = pd.concat([
        calculate_weights(by_cat_weights_merged[by_cat_weights_merged['category'] == category])
        for category in by_cat_weights_merged['category'].unique()
    ], ignore_index=True)
    
    # Calculate weights for all categories with equal weighting options
    all_cat_weights = calculate_weights(all_cat_weights)

    apply_equal_weights(all_cat_weights, 'awareness_pillar', 'give_awareness_metrics_equal_weightage')
    apply_equal_weights(all_cat_weights, 'advocacy_pillar', 'give_advocacy_metrics_equal_weightage')
    
    # Replicate all category weights across individual categories
    all_cat_weights_expanded = pd.concat(
        [all_cat_weights.assign(category=category) for category in cfa_results_by_cat_sub['category'].unique()],
        ignore_index=True
    )
    
    # Combine all weights and format output
    all_weights = pd.concat([by_cat_weights, all_cat_weights_expanded], ignore_index=True).rename(columns={'var': 'metric'})
    weights_sheet = all_weights[['category', 'pillar', 'metric', 'weight']]
    
    return weights_sheet

# COMMAND ----------

def prepare_cfa_subset(cfa_data, exclude_pillars):
    return cfa_data[
        (cfa_data['op'] == "=~") & 
        (cfa_data['Seed'] == 2) & 
        (~cfa_data['lhs'].isin(exclude_pillars))
    ][['lhs', 'rhs', 'est.std', 'Brands', 'Category']].rename(columns={'lhs': 'Pillar', 'rhs': 'Var'})

# COMMAND ----------

def weights_creation(fit_summary_all_cat, fit_summary_all_brands, refresh_config, feat_eng_config, eq_sub_scale_merged_brand, all_pillar_results, equity_dt):

    # Initialize category and model configuration variables
    category_list_pillar = eq_sub_scale_merged_brand['category'].unique()
    # category_list_pillar = refresh_config["pillars"]["all_category_pillars"] + refresh_config["pillars"]["by_category_pillars"]
    exclude_pillars = feat_eng_config["cfa"]["exclude_pillars"]
    all_cat_pillars = feat_eng_config["cfa"]["pillars"]["all_category_pillars"]
    by_cat_pillars = feat_eng_config["cfa"]["pillars"]["by_category_pillars"]

    # Prepare subsets and rename columns as per config criteria
    model_results = all_pillar_results.copy()
    model_results['pillar'] = model_results['pillar'].apply(
        lambda x: f"{x}_pillar" if not x.endswith("_pillar") else x
    )

    cfa_results_all_cat_sub = prepare_cfa_subset(fit_summary_all_cat, exclude_pillars)
    cfa_results_by_cat_sub = prepare_cfa_subset(fit_summary_all_brands, exclude_pillars)

    model_results_sub = model_results[['Shap Features', 'shap values', 'Brand', 'Category', 'pillar']].drop_duplicates()
    model_results_sub.rename(columns={'Brand': 'brands', 'Shap Features': 'Var', 'pillar': 'Pillar'}, inplace=True)

    # Aggregate model results based on configuration
    if feat_eng_config['cfa']['perform_weighted_average_DV']:
        model_results_sub_agg, model_results_pillar_all_cat = dv_weighted_shap(
            equity_dt, model_results_sub, refresh_config["end_date"], category_list_pillar
        )
    else:
        model_results_sub_agg = model_results_sub.groupby(['category', 'pillar', 'var'], as_index=False)['shap values'].mean()
        model_results_sub_agg.rename(columns={'shap values': 'Shap Value - mean'}, inplace=True)
        model_results_pillar_all_cat = model_results_sub_agg.groupby(['pillar', 'var'], as_index=False)['Shap Value - mean'].mean()

    # Call the helper function to prepare weights
    weights_sheet = prepare_weights(
        all_cat_pillars, by_cat_pillars, cfa_results_all_cat_sub, cfa_results_by_cat_sub,
        model_results_pillar_all_cat, model_results_sub_agg
    )

    return weights_sheet

# COMMAND ----------

def create_pillar_scores(scaled_df, weights_sheet,req_cols, category, pillar, equity_dt, feat_eng_config):
    # Initialize an empty DataFrame for the output
    eq_sub2 = pd.DataFrame()
    
    # Standardize metric names to lowercase
    scaled_df.columns = scaled_df.columns.str.lower()
    weights_sheet['metric'] = weights_sheet['metric'].str.lower()
    
    # Calculate weighted sum for each category and pillar
    # for category in weights_sheet['category'].unique():
    scaled_df_sub = scaled_df[scaled_df['category'] == category].copy().reindex(
        sorted(scaled_df.columns), axis=1
    )
        
        # for pillar in weights_sheet['pillar'].unique():
            # Select weights for the current category and pillar
    metric_weights = weights_sheet[(weights_sheet['pillar'] == pillar) & (weights_sheet['category'] == category)]
    metrics_to_use = metric_weights['metric'].values
    
    # Get columns in data that match metrics from weights sheet
    relevant_cols = scaled_df_sub.columns.intersection(metrics_to_use)
    relevant_cols = sorted(relevant_cols)
    metric_weights = metric_weights[metric_weights['metric'].isin(relevant_cols)].sort_values(by='metric')
    
    # Calculate weighted scores for the pillar
    scaled_df_sub[pillar] = scaled_df_sub[relevant_cols].dot(metric_weights['weight'].values)
        
    # Append the results to eq_sub2
    eq_sub2 = pd.concat([eq_sub2, scaled_df_sub], ignore_index=True)

    # Drop unnecessary columns if present
    eq_sub2.drop(columns=['new_brand', 'v1'], inplace=True, errors='ignore')
    
    # Transform to long format
    index_df_long = pd.melt(eq_sub2, id_vars=['date', 'brand', 'category'], var_name='var')

    # Prepare mapping for merging pillars
    mapping_subset = req_cols[['idv_for_model_corrected', 'Equity Pillar', 'product_category_idv']].rename(
        columns={'idv_for_model_corrected': 'var', 'product_category_idv': 'category'}
    )
    mapping_subset['var'] = mapping_subset['var'].str.lower()

    # Merge pillar mappings
    index_df_long = pd.merge(index_df_long, mapping_subset, on=['var', 'category'], how='left')
    index_df_long['Equity Pillar'].fillna(index_df_long['var'], inplace=True)
    
    # Prepare DV data and merge
    dv_data = equity_dt[['date', 'brand_group_expanded', 'category', feat_eng_config["weights_model"]["DV"]]].copy()
    dv_data.rename(columns={
        'brand_group_expanded': 'brand', 
        'market_share_total_sales': 'market_share'
    }, inplace=True)
    
    # Ensure date columns are properly formatted for merging
    for df in [eq_sub2, dv_data, index_df_long]:
        df['date'] = pd.to_datetime(df['date'], utc=False, format="%Y-%m-%d")
    
    # Merge DV data with the pillar scores data
    index_df_final1 = pd.merge(eq_sub2, dv_data, on=['date', 'brand', 'category'], how='left')
    index_df_long = pd.merge(index_df_long, dv_data, on=['date', 'brand', 'category'], how='left')
    
    return index_df_final1, index_df_long

# COMMAND ----------

# def create_pillar_scores(scaled_df, weights_sheet,req_cols):
#     # Initialize an empty DataFrame for the output
#     eq_sub2 = pd.DataFrame()
    
#     # Standardize metric names to lowercase
#     scaled_df.columns = scaled_df.columns.str.lower()
#     weights_sheet['metric'] = weights_sheet['metric'].str.lower()
    
#     # Calculate weighted sum for each category and pillar
#     for category in weights_sheet['category'].unique():
#         scaled_df_sub = scaled_df[scaled_df['category'] == category].copy().reindex(
#             sorted(scaled_df.columns), axis=1
#         )
        
#         for pillar in weights_sheet['pillar'].unique():
#             # Select weights for the current category and pillar
#             metric_weights = weights_sheet[(weights_sheet['pillar'] == pillar) & (weights_sheet['category'] == category)]
#             metrics_to_use = metric_weights['metric'].values
            
#             # Get columns in data that match metrics from weights sheet
#             relevant_cols = scaled_df_sub.columns.intersection(metrics_to_use)
#             relevant_cols = sorted(relevant_cols)
#             metric_weights = metric_weights[metric_weights['metric'].isin(relevant_cols)].sort_values(by='metric')
            
#             # Calculate weighted scores for the pillar
#             scaled_df_sub[pillar] = scaled_df_sub[relevant_cols].fillna(0).dot(metric_weights['weight'].values)
        
#         # Append the results to eq_sub2
#         eq_sub2 = pd.concat([eq_sub2, scaled_df_sub], ignore_index=True)
    
#     # Drop unnecessary columns if present
#     eq_sub2.drop(columns=['new_brand', 'v1'], inplace=True, errors='ignore')
    
#     # Transform to long format
#     index_df_long = pd.melt(eq_sub2, id_vars=['date', 'brand', 'category'], var_name='var')

#     # Prepare mapping for merging pillars
#     mapping_subset = req_cols[['idv_for_model_corrected', 'Equity Pillar', 'product_category_idv']].rename(
#         columns={'idv_for_model_corrected': 'var', 'product_category_idv': 'category'}
#     )
#     mapping_subset['var'] = mapping_subset['var'].str.lower()

#     # Merge pillar mappings
#     index_df_long = pd.merge(index_df_long, mapping_subset, on=['var', 'category'], how='left')
#     index_df_long['Equity Pillar'].fillna(index_df_long['var'], inplace=True)
    
#     # Prepare DV data and merge
#     dv_data = equity_dt[['date', 'brand_group_expanded', 'category', model_config["weights_model"]["DV"]]].copy()
#     dv_data.rename(columns={
#         'brand_group_expanded': 'brand', 
#         'market_share_total_sales': 'market_share'
#     }, inplace=True)
    
#     # Ensure date columns are properly formatted for merging
#     for df in [eq_sub2, dv_data, index_df_long]:
#         df['date'] = pd.to_datetime(df['date'], utc=False, format="%Y-%m-%d")
    
#     # Merge DV data with the pillar scores data
#     index_df_final1 = pd.merge(eq_sub2, dv_data, on=['date', 'brand', 'category'], how='left')
#     index_df_long = pd.merge(index_df_long, dv_data, on=['date', 'brand', 'category'], how='left')
    
#     return index_df_final1, index_df_long

# COMMAND ----------

def pillar_trend_creation(index_df, brand, category, pillar, refresh_config):
    # Determine rolling mean period based on time granularity
    rolling_mean = 3 if refresh_config["time_granularity"] == "monthly" else 13
    
    df_data = index_df.copy()
    # Identify pillar columns
    pillar_cols = [col for col in df_data.columns if col.endswith('_pillar')]
    # print('pillar_cols:', pillar_cols)

    # Keep only relevant columns
    df_data = df_data[['date', 'brand', 'category'] + pillar_cols + [refresh_config["dv"]]]
    
    # Drop existing trend columns
    trend_cols = [col for col in df_data.columns if col.endswith('_trend')]
    df_data.drop(columns=trend_cols, inplace=True)
    
    final_merged_df = pd.DataFrame()

    # Iterate through each unique brand and category to calculate trends
    # for brand in df_data['brand'].unique():
    # brand_data = df_data[df_data['brand'] == brand]
        
        # for category in brand_data['category'].unique():
    br_cat_df = df_data[(df_data['brand'] == brand) & (df_data['category'] == category)]
            
            # Calculate rolling mean for each pillar
            # for pillar in pillar_cols:
    past_3_ma = br_cat_df[pillar].rolling(rolling_mean).mean()
    past_future_3_ma = past_3_ma.shift(-1)
    
    br_cat_df = br_cat_df.assign(
        **{f"{pillar}_trend_past": past_3_ma,
        f"{pillar}_trend_past_future": past_future_3_ma}
    )
                
    return br_cat_df

# COMMAND ----------

# def pillar_trend_creation(index_df):
#     # Determine rolling mean period based on time granularity
#     rolling_mean = 3 if run_config["time_granularity"] == "Monthly" else 13
    
#     df_data = index_df.copy()
#     # Identify pillar columns
#     pillar_cols = [col for col in df_data.columns if col.endswith('_pillar')]
#     # print('pillar_cols:', pillar_cols)

#     # Keep only relevant columns
#     df_data = df_data[['date', 'brand', 'category'] + pillar_cols + [run_config["DV"]]]
    
#     # Drop existing trend columns
#     trend_cols = [col for col in df_data.columns if col.endswith('_trend')]
#     df_data.drop(columns=trend_cols, inplace=True)
    
#     final_merged_df = pd.DataFrame()

#     # Iterate through each unique brand and category to calculate trends
#     for brand in df_data['brand'].unique():
#         brand_data = df_data[df_data['brand'] == brand]
        
#         for category in brand_data['category'].unique():
#             br_cat_df = brand_data[brand_data['category'] == category]
            
#             # Calculate rolling mean for each pillar
#             for pillar in pillar_cols:
#                 past_3_ma = br_cat_df[pillar].rolling(rolling_mean).mean()
#                 past_future_3_ma = past_3_ma.shift(-1)
                
#                 br_cat_df = br_cat_df.assign(
#                     **{f"{pillar}_trend_past": past_3_ma,
#                     f"{pillar}_trend_past_future": past_future_3_ma}
#                 )
                
#             final_merged_df = pd.concat([final_merged_df, br_cat_df], ignore_index=True)
    
#     return final_merged_df

# COMMAND ----------

def lag_addition(dataset):
    # Initialize an empty DataFrame to store results
    ret_df = pd.DataFrame()
    
    # Identify columns that will not have lags
    cols_no_lag = ['date', 'category', 'brand', refresh_config["dv"]]
    basecols = dataset.columns.difference(cols_no_lag).tolist()

    # Iterate over each unique brand and category to create lagged features
    for brand in dataset['brand'].unique():
        brand_data = dataset[dataset['brand'] == brand]
        
        for category in brand_data['category'].unique():
            category_data = brand_data[brand_data['category'] == category]

            # Create lagged features for each base column
            for col in basecols:
                for lag in range(1, 4):  # Create lags 1 to 3
                    category_data[f"{col}_lag_{lag}"] = category_data[col].shift(lag)
                    
            # Concatenate the modified DataFrame to the result
            ret_df = pd.concat([ret_df, category_data], ignore_index=True)
    
    return ret_df

# COMMAND ----------

def run_model_analysis(model_type, feat_eng_config, X_train, y_train, X_test_hold, train_x_all, train_y_all, brand, category, dv, CV_NO):
    feat_importance = pd.DataFrame()
    feat_df = pd.DataFrame()
    results_all_model = pd.DataFrame()
    
    if model_type == "Brute_force":
        params = {
            'alpha': feat_eng_config["importance_model"]["hyperparameters"]["Brute_force"]["grid_search"]["alpha"],
            "random_state": feat_eng_config["importance_model"]["hyperparameters"]["Brute_force"]["grid_search"]["random_state"]
        }
        model = Ridge(
            positive=feat_eng_config["importance_model"]["hyperparameters"]["Brute_force"]["positive"],
            random_state=feat_eng_config["importance_model"]["hyperparameters"]["Brute_force"]["random_state"]
        )
        grid_search = GridSearchCV(model, params, cv=CV_NO, scoring=['r2', 'neg_mean_absolute_percentage_error'], refit='neg_mean_absolute_percentage_error')
    
    elif model_type == "RandomForest":
        params = {
            "max_depth": feat_eng_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["max_depth"],
            "n_estimators": feat_eng_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["n_estimators"],
            "max_features": feat_eng_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["max_features"],
            "random_state": feat_eng_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["random_state"]
        }
        model = RandomForestRegressor(random_state=feat_eng_config["importance_model"]["hyperparameters"]["RandomForest"]["random_state"])
        grid_search = GridSearchCV(model, params, cv=CV_NO, scoring=['r2', 'neg_mean_absolute_percentage_error'], refit='neg_mean_absolute_percentage_error')
    
    elif model_type == "XGBoost":
        params = {
            "max_depth": feat_eng_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["max_depth"],
            "n_estimators": feat_eng_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["n_estimators"],
            "learning_rate": feat_eng_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["learning_rate"],
            "random_state": feat_eng_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["random_state"]
        }
        model = xgb.XGBRegressor(eval_metric='mape', random_state=feat_eng_config["importance_model"]["hyperparameters"]["XGBoost"]["random_state"])
        grid_search = GridSearchCV(model, params, cv=CV_NO, scoring=['r2', 'neg_mean_absolute_percentage_error'], refit='neg_mean_absolute_percentage_error')

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Fit the model
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    # Feature Importance
    if model_type == "Brute_force":
        params = np.append(model.intercept_, model.coef_)
        feat_importance = pd.DataFrame({'Features': ['intercept'] + list(X_train.columns), 'Feature Importance/coefficient': params})
    elif model_type == "RandomForest":
        feat_importance = pd.DataFrame(list(zip(X_train.columns, model.feature_importances_)), columns=['Features', 'Feature Importance/coefficient']).sort_values(by='Feature Importance/coefficient', ascending=False)
    elif model_type == "XGBoost":
        feat_importance = pd.DataFrame(model.get_booster().get_score(importance_type='gain').items(), columns=['Features', 'Feature Importance/coefficient']).sort_values(by='Feature Importance/coefficient', ascending=False)

    # SHAP Values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train).values
    feat_df = pd.DataFrame(np.abs(shap_values).mean(axis=0), index=X_train.columns, columns=['shap values']).sort_values(by='shap values', ascending=False)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test_hold)
    y_pred_all = model.predict(train_x_all[X_train.columns])

    # Metrics
    mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
    r2_train = metrics.r2_score(y_train, y_pred_train)
    mape_train = metrics.mean_absolute_percentage_error(y_train, y_pred_train)
    
    # Results DataFrame
    results_all_model = pd.concat([feat_importance, feat_df], axis=1)
    results_all_model['Model'] = model_type
    results_all_model['Brand'] = brand
    results_all_model['Category'] = category
    results_all_model['Latest DV'] = dv.values[-1]
    results_all_model['R2_Score_Train'] = r2_train
    results_all_model['MAPE_Train'] = mape_train

    return results_all_model


# COMMAND ----------

def pillar_importance_model(trend_pillars_data, sales_data, feat_eng_config, brand, category):
    df_data = trend_pillars_data
    brand_df = df_data.copy()
    brand_df['date'] = pd.to_datetime(brand_df['date'], format="%Y-%m-%d")
    CV_NO = feat_eng_config["importance_model"]["cross_validation_number"]

    # print("brand_df.shape :",brand_df.shape)

    results_all_model = pd.DataFrame()
    final_df = lag_addition(brand_df)
    final_df.rename(columns={'category':'category_new'},inplace=True)

    final_df.loc[final_df.category_new == "CAT TREATS ONLY","category"] = 'CAT FOOD'
    final_df.loc[final_df.category_new == "DOG TREATS ONLY","category"] = 'DOG FOOD'
    final_df.loc[final_df.category_new == "CAT FOOD","category"] = 'CAT FOOD'
    final_df.loc[final_df.category_new == "DOG FOOD","category"] = 'DOG FOOD'
    final_df.loc[final_df.category_new == "CAT LITTER","category"] = 'CAT LITTER'

    rms_teneten_df = sales_data

    rms_teneten_df['date'] = pd.to_datetime(rms_teneten_df['date'])

    rms_teneten_df.rename(columns={'brand_group_expanded':'brand'},inplace=True)

    # final_df.drop(columns=['equalized_volume'],inplace=True)

    final_df = final_df.merge(rms_teneten_df,on=['date', 'category', 'brand'],how='left')

    # print(list(final_df.columns))

    past_future_trend_cols = [cols for cols in final_df.columns if "_trend_past_future" in cols]
    past_trend_cols = [cols for cols in final_df.columns if "_trend_past" in cols and cols not in past_future_trend_cols]
    raw_cols = [cols for cols in final_df.columns if cols not in past_trend_cols and cols not in past_future_trend_cols]
    corr_raw_df_full =pd.DataFrame()

    ###run without price and distribution
    if feat_eng_config["importance_model"]["price_and_acv_added"]:
        final_df.drop(columns=['average_price','acv_selling'],inplace=True)

    # for brand in final_df.brand.unique():
    # br_df = final_df[final_df['brand'] == brand]
        # for category in br_df['category_new'].unique():
    cat_br_df = final_df[(final_df['brand'] == brand) & (final_df['category_new'] == category)]
    null_count_df = pd.DataFrame(cat_br_df.isnull().sum(),columns=['null count'])
    null_cols = null_count_df[null_count_df['null count'] > cat_br_df.shape[0] * 0.5].index
    if len(null_cols) > 0:
        cat_br_df.drop(columns=null_cols,inplace=True)
    cat_br_df = cat_br_df.iloc[3:,:]
    cat_br_df = cat_br_df.fillna(cat_br_df.mean())

    brand_perceptions_pillar = [cols for cols in cat_br_df.columns if cols.startswith('brand_perceptions')]
    product_feedback_pillar =  [cols for cols in cat_br_df.columns if cols.startswith('product_feedback')]
    advocacy_pillar = [cols for cols in cat_br_df.columns if cols.startswith('advocacy')]
    awareness_pillar = [cols for cols in cat_br_df.columns if cols.startswith('awareness')]
    consideration_pillar = [cols for cols in cat_br_df.columns if cols.startswith('consideration')]
    loyalty_pillar = [cols for cols in cat_br_df.columns if cols.startswith('loyalty')]

    pillars=[brand_perceptions_pillar,product_feedback_pillar,advocacy_pillar,awareness_pillar,consideration_pillar,loyalty_pillar]

    cat_br_pillar_df = cat_br_df.copy()
    flatten_pillars_list = list(chain.from_iterable(pillars))
    if feat_eng_config["importance_model"]["lags_added"]==False:
        flatten_pillars_list = [cols for cols in flatten_pillars_list if ("_lag") not in cols] # dropping lag metrics
    past_future_trend_cols = [cols for cols in flatten_pillars_list if "_trend_past_future" in cols]
    past_trend_cols = [cols for cols in flatten_pillars_list if "_trend_past" in cols and cols not in past_future_trend_cols]
    raw_cols = [cols for cols in flatten_pillars_list if cols not in past_trend_cols and cols not in past_future_trend_cols]

    brand_name = brand
    category_name = category


    k=2

    if feat_eng_config["importance_model"]["price_and_acv_added"]:
        cols_to_select = past_trend_cols+['average_price','rms_acv_selling']
    else:
        cols_to_select = past_trend_cols
    for j in [cols_to_select]:
        # print('j:',j,"-",brand,"-",category)
        k=k+1
        matched_cols=[cols for cols in j if cols in cat_br_pillar_df.columns]
        print("cat_br_pillar_df cols:",list(cat_br_pillar_df.columns))
        modeldf = cat_br_pillar_df[matched_cols+[feat_eng_config["importance_model"]["DV"]]]

        idvs = modeldf.drop(feat_eng_config["importance_model"]["DV"], 1)       # feature matrix

        if feat_eng_config["importance_model"]["log_convert_DV"]:
            dv = np.log1p(modeldf[feat_eng_config["importance_model"]["DV"]])
        else:
            dv = modeldf[feat_eng_config["importance_model"]["DV"]]
        if idvs.shape[1] > 2:
            if feat_eng_config["importance_model"]["standardize"]:
                mmscaler = StandardScaler() # Do even before feature selection
                idvs_scaled = pd.DataFrame(mmscaler.fit_transform(idvs), columns = idvs.columns)
                idvs = idvs_scaled.copy()
                if len(idvs_scaled.columns) >= 2:
                    for i in [2]:
                        # print(i)
                        if feat_eng_config["importance_model"]["Time_series_split"]:
                            train_x = idvs[idvs_scaled.columns].iloc[:-6,:]
                            train_y = dv[:-6]
                            test_x = idvs[idvs_scaled.columns].iloc[-6:,:]
                            test_y = dv[-6:]
                        if feat_eng_config["importance_model"]["Random_seed_split"]:
                            train_x, test_x, train_y, test_y = train_test_split(idvs[idvs_scaled.columns],dv,test_size=6, random_state=i, shuffle=True)
                        train_x_all =  idvs[idvs_scaled.columns]
                        train_y_all = dv

                        X_test_hold = test_x.copy()
                        y_test_hold = test_y.copy()

                        X_train = train_x.copy()
                        y_train = train_y.copy()

                        feat_importance = pd.DataFrame()
                        feat_df=pd.DataFrame()

                        if feat_eng_config["importance_model"]["Corr_file_generation"]:
                            correlation_mat = modeldf.corr()
                            corr_pairs = correlation_mat.unstack()
                            corr_pairs = corr_pairs.reset_index()
                            corr_pairs.rename(columns = {'level_0':'Variable 1','level_1':'Variable 2',0:'Corelation Co efficient'},inplace=True)
                            corr_pairs = corr_pairs[corr_pairs['Variable 1']!=corr_pairs['Variable 2']]
#                                 corr_raw_df_full[corr_raw_df_full['Variable 1'] == 'market_share']
                            corr_pairs['Brand'] = brand
                            corr_pairs['Category'] = category
                            corr_raw_df_full = pd.concat([corr_raw_df_full,corr_pairs])

                        if feat_eng_config["importance_model"]["Brute_force"]["run"]:
                            results = run_model_analysis(
                                model_type="Brute_force",
                                feat_eng_config=feat_eng_config,
                                X_train=X_train,
                                y_train=y_train,
                                X_test_hold=X_test_hold,
                                train_x_all=train_x_all,
                                train_y_all=train_y_all,
                                brand=brand,
                                category=category,
                                dv=dv,
                                CV_NO=5
                            )

                        feat_importance = pd.DataFrame()
                        feat_df=pd.DataFrame()
                        if feat_eng_config["importance_model"]["RandomForest"]["run"]:
                            results = run_model_analysis(
                                model_type="RandomForest",
                                feat_eng_config=feat_eng_config,
                                X_train=X_train,
                                y_train=y_train,
                                X_test_hold=X_test_hold,
                                train_x_all=train_x_all,
                                train_y_all=train_y_all,
                                brand=brand,
                                category=category,
                                dv=dv,
                                CV_NO=5
                            )

                        feat_importance = pd.DataFrame()
                        feat_df=pd.DataFrame()
                        if feat_eng_config["importance_model"]["XGBoost"]["run"]:
                            results = run_model_analysis(
                                model_type="XGBoost",
                                feat_eng_config=feat_eng_config,
                                X_train=X_train,
                                y_train=y_train,
                                X_test_hold=X_test_hold,
                                train_x_all=train_x_all,
                                train_y_all=train_y_all,
                                brand=brand,
                                category=category,
                                dv=dv,
                                CV_NO=5
                            )

    return results

# COMMAND ----------

# def pillar_importance_model(trend_pillars_data, sales_data):
#     df_data = trend_pillars_data
#     brand_df = df_data.copy()
#     brand_df['date'] = pd.to_datetime(brand_df['date'], format="%Y-%m-%d")
#     CV_NO = model_config["importance_model"]["cross_validation_number"]

#     # print("brand_df.shape :",brand_df.shape)

#     results_all_model = pd.DataFrame()
#     final_df = lag_addition(brand_df)
#     final_df.rename(columns={'category':'category_new'},inplace=True)

#     final_df.loc[final_df.category_new == "CAT TREATS ONLY","category"] = 'CAT FOOD'
#     final_df.loc[final_df.category_new == "DOG TREATS ONLY","category"] = 'DOG FOOD'
#     final_df.loc[final_df.category_new == "CAT FOOD","category"] = 'CAT FOOD'
#     final_df.loc[final_df.category_new == "DOG FOOD","category"] = 'DOG FOOD'
#     final_df.loc[final_df.category_new == "CAT LITTER","category"] = 'CAT LITTER'

#     rms_teneten_df = sales_data

#     rms_teneten_df['date'] = pd.to_datetime(rms_teneten_df['date'])

#     rms_teneten_df.rename(columns={'brand_group_expanded':'brand'},inplace=True)

#     final_df.drop(columns=['equalized_volume'],inplace=True)

#     final_df = final_df.merge(rms_teneten_df,on=['date', 'category', 'brand'],how='left')

#     # print(list(final_df.columns))

#     past_future_trend_cols = [cols for cols in final_df.columns if "_trend_past_future" in cols]
#     past_trend_cols = [cols for cols in final_df.columns if "_trend_past" in cols and cols not in past_future_trend_cols]
#     raw_cols = [cols for cols in final_df.columns if cols not in past_trend_cols and cols not in past_future_trend_cols]
#     corr_raw_df_full =pd.DataFrame()

#     ###run without price and distribution
#     if model_config["importance_model"]["price_and_acv_added"]:
#         final_df.drop(columns=['average_price','acv_selling'],inplace=True)

#     for brand in final_df.brand.unique():
#         br_df = final_df[final_df['brand'] == brand]
#         for category in br_df['category_new'].unique():
#             cat_br_df = br_df[br_df['category_new'] == category]
#             null_count_df = pd.DataFrame(cat_br_df.isnull().sum(),columns=['null count'])
#             null_cols = null_count_df[null_count_df['null count'] > cat_br_df.shape[0] * 0.5].index
#             if len(null_cols) > 0:
#                 cat_br_df.drop(columns=null_cols,inplace=True)
#             cat_br_df = cat_br_df.iloc[3:,:]
#             cat_br_df = cat_br_df.fillna(cat_br_df.mean())

#             brand_perceptions_pillar = [cols for cols in cat_br_df.columns if cols.startswith('brand_perceptions')]
#             product_feedback_pillar =  [cols for cols in cat_br_df.columns if cols.startswith('product_feedback')]
#             advocacy_pillar = [cols for cols in cat_br_df.columns if cols.startswith('advocacy')]
#             awareness_pillar = [cols for cols in cat_br_df.columns if cols.startswith('awareness')]
#             consideration_pillar = [cols for cols in cat_br_df.columns if cols.startswith('consideration')]
#             loyalty_pillar = [cols for cols in cat_br_df.columns if cols.startswith('loyalty')]

#             pillars=[brand_perceptions_pillar,product_feedback_pillar,advocacy_pillar,awareness_pillar,consideration_pillar,loyalty_pillar]

#             cat_br_pillar_df = cat_br_df.copy()
#             flatten_pillars_list = list(chain.from_iterable(pillars))
#             if model_config["importance_model"]["lags_added"]==False:
#                 flatten_pillars_list = [cols for cols in flatten_pillars_list if ("_lag") not in cols] # dropping lag metrics
#             past_future_trend_cols = [cols for cols in flatten_pillars_list if "_trend_past_future" in cols]
#             past_trend_cols = [cols for cols in flatten_pillars_list if "_trend_past" in cols and cols not in past_future_trend_cols]
#             raw_cols = [cols for cols in flatten_pillars_list if cols not in past_trend_cols and cols not in past_future_trend_cols]

#             brand_name = brand
#             category_name = category


#             k=2

#             if model_config["importance_model"]["price_and_acv_added"]:
#                 cols_to_select = past_trend_cols+['average_price','rms_acv_selling']
#             else:
#                 cols_to_select = past_trend_cols
#             for j in [cols_to_select]:
#                 # print('j:',j,"-",brand,"-",category)
#                 k=k+1
#                 matched_cols=[cols for cols in j if cols in cat_br_pillar_df.columns]
#                 print("cat_br_pillar_df cols:",list(cat_br_pillar_df.columns))
#                 modeldf = cat_br_pillar_df[matched_cols+[model_config["importance_model"]["DV"]]]

#                 idvs = modeldf.drop(model_config["importance_model"]["DV"], 1)       # feature matrix

#                 if model_config["importance_model"]["log_convert_DV"]:
#                     dv = np.log1p(modeldf[model_config["importance_model"]["DV"]])
#                 else:
#                     dv = modeldf[model_config["importance_model"]["DV"]]
#                 if idvs.shape[1] > 2:
#                     if model_config["importance_model"]["standardize"]:
#                         mmscaler = StandardScaler() # Do even before feature selection
#                         idvs_scaled = pd.DataFrame(mmscaler.fit_transform(idvs), columns = idvs.columns)
#                         idvs = idvs_scaled.copy()
#                         if len(idvs_scaled.columns) >= 2:
#                             for i in [2]:
#                                 # print(i)
#                                 if model_config["importance_model"]["Time_series_split"]:
#                                     train_x = idvs[idvs_scaled.columns].iloc[:-6,:]
#                                     train_y = dv[:-6]
#                                     test_x = idvs[idvs_scaled.columns].iloc[-6:,:]
#                                     test_y = dv[-6:]
#                                 if model_config["importance_model"]["Random_seed_split"]:
#                                     train_x, test_x, train_y, test_y = train_test_split(idvs[idvs_scaled.columns],dv,test_size=6, random_state=i, shuffle=True)
#                                 train_x_all =  idvs[idvs_scaled.columns]
#                                 train_y_all = dv

#                                 X_test_hold = test_x.copy()
#                                 y_test_hold = test_y.copy()


#                                 X_train = train_x.copy()
#                                 y_train = train_y.copy()

#                                 feat_importance = pd.DataFrame()
#                                 feat_df=pd.DataFrame()

#                                 feat_importance = pd.DataFrame()
#                                 feat_df=pd.DataFrame()

#                                 if model_config["importance_model"]["Corr_file_generation"]:
#                                     correlation_mat = modeldf.corr()
#                                     corr_pairs = correlation_mat.unstack()
#                                     corr_pairs = corr_pairs.reset_index()
#                                     corr_pairs.rename(columns = {'level_0':'Variable 1','level_1':'Variable 2',0:'Corelation Co efficient'},inplace=True)
#                                     corr_pairs = corr_pairs[corr_pairs['Variable 1']!=corr_pairs['Variable 2']]
#     #                                 corr_raw_df_full[corr_raw_df_full['Variable 1'] == 'market_share']
#                                     corr_pairs['Brand'] = brand
#                                     corr_pairs['Category'] = category
#                                     corr_raw_df_full = pd.concat([corr_raw_df_full,corr_pairs])

#                                 if run_config["importance_model"]["Brute_force"]["run"]:
#                                         param_rf_ridge = {'alpha':model_config["importance_model"]["hyperparameters"]["Brute_force"]["grid_search"]["alpha"], "random_state":model_config["importance_model"]["hyperparameters"]["Brute_force"]["grid_search"]["random_state"]}
#                                         ridg3=Ridge(positive=model_config["importance_model"]["hyperparameters"]["Brute_force"]["positive"],random_state=model_config["importance_model"]["hyperparameters"]["Brute_force"]["random_state"])
#                                         search_brute = GridSearchCV(ridg3,param_rf_ridge,cv=CV_NO,scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train, y_train)
#                                         fit_model_brute=Ridge(alpha=search_brute.best_params_["alpha"],random_state=model_config["importance_model"]["hyperparameters"]["Brute_force"]["random_state"]).fit(X_train, y_train)



#                                         stat_df = pd.DataFrame()
#                                         params = np.append(fit_model_brute.intercept_,fit_model_brute.coef_)
#                                         stat_df["coefficients"] = list(params)
#                                         features = ['intercept']+list(X_train.columns)
#                                         stat_df.insert(0,"features", features)
#                                         feat_import = stat_df.sort_values(by='coefficients',ascending=False)
#                                         # print(feat_import)

#                                         explainer = shap.LinearExplainer(fit_model_brute,X_train)
#                                         shap_values = explainer.shap_values(X_train)
#                                         feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train.columns)).mean(),columns=['shap values'])
#                                         # print("Brute force shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                         feat_df = feat_df.sort_values(by='shap values',ascending=False)

#                                         y_pred_train = fit_model_brute.predict(X_train)
#                                         y_pred_test = fit_model_brute.predict(X_test_hold)
#                                         y_pred_all = fit_model_brute.predict(train_x_all[list(X_train.columns)])


#                                         feat_importance = feat_import.reset_index().rename(columns={'features':'Features','coefficients':'Feature Importance/coefficient'})
#                                         feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                         mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                         mse_train = metrics.mean_squared_error(y_train, y_pred_train)
#                                         rmse_train = np.sqrt(mse_train) #mse**(0.5)
#                                         r2_train = metrics.r2_score(y_train, y_pred_train)
#                                         mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                         results_all_model_brute = pd.concat([feat_importance,feat_df], axis=1)
#                                         results_all_model_brute['Model'] = "Brute Force Model"
#                                         results_all_model_brute['Brand'] = brand
#                                         results_all_model_brute['Category'] = category
#                                         results_all_model_brute['Latest MS'] = dv.values[-1]

#                                         results_all_model_brute['R2_Score_Train'] = r2_train
#                                         results_all_model_brute['MAPE_Train'] = mape_train
#                                         results_all_model_brute['R2_score_fold'] = cross_val_score(fit_model_brute,X_train,y_train,cv=CV_NO,scoring='r2').mean()
#                                         results_all_model_brute['MAPE_fold'] = cross_val_score(fit_model_brute,X_train,y_train,cv=CV_NO,scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                         results_all_model_brute['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                         results_all_model_brute['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                         results_all_model_brute['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                         results_all_model_brute['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)
#                                         results_all_model_brute['Pillar'] = 'All Pillars'
#                                         results_all_model_brute['Best_Params_Gridsearchcv']=str(search_brute.best_params_)
#                                         results_all_model = pd.concat([results_all_model, results_all_model_brute],axis=0)

#                                 feat_importance = pd.DataFrame()
#                                 feat_df=pd.DataFrame()
#                                 if run_config["importance_model"]["RandomForest"]["run"]:
#                                     param_grid_rf = {"max_depth":model_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["max_depth"],"n_estimators": model_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["n_estimators"],'max_features': model_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["max_features"],"random_state":model_config["importance_model"]["hyperparameters"]["RandomForest"]["grid_search"]["random_state"]}
#                                     rf1=RandomForestRegressor(random_state=model_config["importance_model"]["hyperparameters"]["RandomForest"]["random_state"])
#                                     search_rf = GridSearchCV(rf1, param_grid_rf,cv=CV_NO,scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train, y_train)
#                                     rf=RandomForestRegressor(n_estimators  = search_rf.best_params_["n_estimators"],max_depth = search_rf.best_params_["max_depth"],random_state=model_config["importance_model"]["hyperparameters"]["RandomForest"]["random_state"])
#                                     rf.fit(X_train,y_train)
#                                     features = list(X_train.columns)
#                                     f_i = list(zip(features,rf.feature_importances_))
#                                     f_i.sort(key = lambda x : x[1],reverse=True)

#                                     rfe = RFECV(rf,cv=CV_NO,scoring='neg_mean_absolute_percentage_error')
#                                     rfe.fit(X_train,y_train)
#                                     selected_features = list(np.array(features)[rfe.get_support()])
#                                     # print(selected_features)
#                                     feat_importance = pd.DataFrame(f_i,columns=['Features','Feature Importance'])
#                                     feat_importance.set_index('Features',inplace=True)
#                                     feat_importance = feat_importance.iloc[:20,:]
#                                     # print(feat_importance)
#                                     best_features = list(feat_importance.index)

#                                     explainer = shap.TreeExplainer(rf)
#                                     shap_values = explainer.shap_values(X_train)
#                                     feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train.columns)).mean(),columns=['shap values'])
#                                     # print("Random Forest shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                     feat_df = feat_df.sort_values(by='shap values',ascending=False)


#                                     y_pred_test = rf.predict(X_test_hold)
#                                     y_pred_train = rf.predict(X_train)
#                                     y_pred_all = rf.predict(train_x_all[list(X_train.columns)])

#                                     feat_importance = feat_importance.reset_index().rename(columns={'Feature Importance':'Feature Importance/coefficient'})
#                                     feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                     mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                     mse_train = metrics.mean_squared_error(y_train, y_pred_train)

#                                     rmse_train = np.sqrt(mse_train)
#                                     r2_train = metrics.r2_score(y_train, y_pred_train)
#                                     mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                     feat_df

#                                     results_all_model_RF = pd.concat([feat_importance,feat_df], axis=1)
#                                     results_all_model_RF['Model'] = "Random Forest"
#                                     results_all_model_RF['Brand'] = brand
#                                     results_all_model_RF['Category'] = category

#                                     results_all_model_RF['Latest DV'] = dv.values[-1]

#                                     results_all_model_RF['R2_Score_Train'] = r2_train
#                                     results_all_model_RF['MAPE_Train'] = mape_train
#                                     results_all_model_RF['R2_score_fold'] = cross_val_score(rf,X_train,y_train,cv=CV_NO,scoring='r2').mean()
#                                     results_all_model_RF['MAPE_fold'] = cross_val_score(rf,X_train,y_train,cv=CV_NO,scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                     results_all_model_RF['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                     results_all_model_RF['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                     results_all_model_RF['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                     results_all_model_RF['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)
#                                     results_all_model_RF['Pillar'] = 'All Pillars'
#                                     results_all_model_RF['Best_Params_Gridsearchcv']=str(search_rf.best_params_)
#                                     results_all_model = pd.concat([results_all_model, results_all_model_RF],axis=0)
#                                     title=(brand_name+"-"+category_name+"-"+"Random_Forest")


#                                 feat_importance = pd.DataFrame()
#                                 feat_df=pd.DataFrame()
#                                 if run_config["importance_model"]["XGBoost"]["run"]:
#                                     param_grid = {"max_depth": model_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["max_depth"],"n_estimators": model_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["n_estimators"],"learning_rate": model_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["learning_rate"],"random_state":model_config["importance_model"]["hyperparameters"]["XGBoost"]["grid_search"]["random_state"]}
#                                     regressor=xgb.XGBRegressor(eval_metric='mape',random_state=model_config["importance_model"]["hyperparameters"]["XGBoost"]["random_state"])
#                                     search_xgb = GridSearchCV(regressor, param_grid,cv=CV_NO,scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train, y_train)
#                                     # print("The best hyperparameters are ",search_xgb.best_params_)

#                                     regressor=xgb.XGBRegressor(learning_rate = search_xgb.best_params_["learning_rate"],
#                                                 n_estimators  = search_xgb.best_params_["n_estimators"],
#                                                 max_depth     = search_xgb.best_params_["max_depth"],
#                                                 eval_metric='mape',random_state=model_config["importance_model"]["hyperparameters"]["XGBoost"]["random_state"])

#                                     regressor.fit(X_train, y_train)
#                                     dict_result = regressor.get_booster().get_score(importance_type='gain')
#                                     # print("Feature importance XGBoost",pd.DataFrame(dict_result.items(),columns=['Feature','gain']).sort_values(by ='gain',ascending=False))
#                                     feat_importance = pd.DataFrame(dict_result.items(),columns=['Feature','gain']).sort_values(by ='gain',ascending=False)
#                                     explainer = shap.TreeExplainer(regressor)
#                                     shap_values = explainer.shap_values(X_train)
#                                     feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train.columns)).mean(),columns=['shap values'])
#                                     # print("XGBoost shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                     feat_df = feat_df.sort_values(by='shap values',ascending=False)

#                                     y_pred_test = regressor.predict(X_test_hold)
#                                     y_pred_train = regressor.predict(X_train)
#                                     y_pred_all = regressor.predict(train_x_all[list(X_train.columns)])

#                                     feat_importance = feat_importance.reset_index().rename(columns={'Feature':'Features','gain':'Feature Importance/coefficient'})
#                                     feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                     mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                     mse_train = metrics.mean_squared_error(y_train, y_pred_train)
#                                     rmse_train = np.sqrt(mse_train) #mse**(0.5)
#                                     r2_train = metrics.r2_score(y_train, y_pred_train)
#                                     mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                     results_all_model_XGB = pd.concat([feat_importance,feat_df], axis=1)
#                                     results_all_model_XGB['Model'] = "XGBoost"
#                                     results_all_model_XGB['Brand'] = brand
#                                     results_all_model_XGB['Category'] = category
#                                     results_all_model_XGB['Latest DV'] = dv.values[-1]

#                                     results_all_model_XGB['R2_Score_Train'] = r2_train
#                                     results_all_model_XGB['MAPE_Train'] = mape_train
#                                     results_all_model_XGB['R2_score_fold'] = cross_val_score(regressor,X_train,y_train,cv=CV_NO,scoring='r2').mean()
#                                     results_all_model_XGB['MAPE_fold'] = cross_val_score(regressor,X_train,y_train,cv=CV_NO,scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                     results_all_model_XGB['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                     results_all_model_XGB['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                     results_all_model_XGB['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                     results_all_model_XGB['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)
#                                     results_all_model_XGB['Pillar'] = 'All Pillars'
#                                     results_all_model_XGB['Best_Params_Gridsearchcv']=str(search_xgb.best_params_)
#                                     results_all_model = pd.concat([results_all_model, results_all_model_XGB],axis=0)

#     return results_all_model

# COMMAND ----------

# def scaled_scores(pillar_data, feat_eng_config, filter_config, refresh_config):
#     # Convert date column to datetime
#     pillar_data['date'] = pd.to_datetime(pillar_data['date'], format="%Y-%m-%d")
    
#     # Define relevant columns for grouping and scaling
#     pillar_columns = [col for col in pillar_data.columns if col.endswith('_pillar')]
#     additional_columns = ['date', 'category', refresh_config["DV"], 'brand']

#     if feat_eng_config["scaled_score"]["only_pillars"]:
#         grouped = pillar_data.groupby(['date', 'category'])[pillar_columns].mean().reset_index()
#         df1 = pillar_data[additional_columns + pillar_columns]
#     else:
#         columns_to_mean = pillar_data.columns.difference(['date', 'brand', 'category', refresh_config["DV"]]).tolist()
#         grouped = pillar_data.groupby(['date', 'category'])[columns_to_mean].mean().reset_index()
#         df1 = pillar_data

#     # Prepare to store processed dataframes
#     processed_dataframes = []

#     # Extract unique combinations of date and category
#     unique_combinations = df1[['date', 'category']].drop_duplicates()

#     # Scaling logic
#     for _, group in unique_combinations.iterrows():
#         date = group["date"]
#         category = group['category']

#         subset1 = df1[(df1['date'] == date) & (df1['category'] == category)]
#         subset = grouped[(grouped['date'] == date) & (grouped['category'] == category)]

#         for column_name in (pillar_columns if filter_config["scaled_score"]["only_pillars"] else columns_to_mean):
#             new_column_name = f"{column_name}_scores"
#             average = subset[column_name].values[0]  # Assumes there's at least one row in subset
#             subset1[new_column_name] = (subset1[column_name] - average) * 100 + 100
        
#         processed_dataframes.append(subset1)

#     result_df = pd.concat(processed_dataframes, ignore_index=True)

#     # Melt the result DataFrame
#     columns_to_pivot = [col for col in result_df.columns if col not in ['date', 'brand', 'category', refresh_config["DV"]]]
#     melted_df = pd.melt(result_df, id_vars=['date', 'brand', 'category'], value_vars=columns_to_pivot, value_name='column_value')

#     return result_df, melted_df

# COMMAND ----------

import pandas as pd

def compute_scaled_scores(date, category, df1, grouped, pillar_columns, columns_to_mean, only_pillars):
    """
    Process data for a single combination of date and category.

    Parameters:
        date (datetime): The date to filter on.
        category (str): The category to filter on.
        df1 (DataFrame): The main DataFrame.
        grouped (DataFrame): The grouped DataFrame with means.
        pillar_columns (list): List of pillar columns.
        columns_to_mean (list): List of columns to scale.
        only_pillars (bool): Whether to process only pillar columns.

    Returns:
        DataFrame: The processed subset DataFrame with scaled scores.
    """
    subset1 = df1[(df1['date'] == date) & (df1['category'] == category)].copy()
    subset = grouped[(grouped['date'] == date) & (grouped['category'] == category)]

    for column_name in (pillar_columns if only_pillars else columns_to_mean):
        new_column_name = f"{column_name}_scores"
        average = subset[column_name].values[0]  # Assumes there's at least one row in subset
        subset1[new_column_name] = (subset1[column_name] - average) * 100 + 100
    return subset1

# Main function to set up the data and prepare for processing
def scaled_scores_prep(pillar_data, feat_eng_config, filter_config, refresh_config):
    # Convert date column to datetime
    pillar_data['date'] = pd.to_datetime(pillar_data['date'], format="%Y-%m-%d")

    # Define relevant columns for grouping and scaling
    pillar_columns = [col for col in pillar_data.columns if col.endswith('_pillar')]
    additional_columns = ['date', 'category', refresh_config["dv"], 'brand']

    if filter_config["scaled_score"]["only_pillars"]:
        grouped = pillar_data.groupby(['date', 'category'])[pillar_columns].mean().reset_index()
        df1 = pillar_data[additional_columns + pillar_columns]
    else:
        columns_to_mean = pillar_data.columns.difference(['date', 'brand', 'category', refresh_config["dv"]]).tolist()
        grouped = pillar_data.groupby(['date', 'category'])[columns_to_mean].mean().reset_index()
        df1 = pillar_data

    # Extract unique combinations of date and category
    unique_combinations = df1[['date', 'category']].drop_duplicates()

    return df1, grouped, pillar_columns, columns_to_mean, unique_combinations


# COMMAND ----------

def data_hub_data_format_code(pillars_long_data, weights_sheet, scaled_scores_long, imp_model_results, idv_list):
    # Preprocess pillars_long_data
    attr_df = pillars_long_data

    # Preprocess weights_sheet
    attr_df1 = weights_sheet.copy()
    attr_df1.columns = (attr_df1.columns
                        .str.replace(' - ', '_')
                        .str.replace(' ', '_')
                        .str.lower())

    # Rename columns for merging
    df = attr_df1.rename(columns={'metric': 'var', 'pillar': 'Equity Pillar'})
    df['Equity Pillar'] = df['Equity Pillar'].str.replace('_pillar', '')
    
    # Merge pillar data with weights
    merged_df = pd.merge(attr_df, df[['category', 'var', 'Equity Pillar', 'weight']], 
                        on=['category', 'var', 'Equity Pillar'], how='left')

    # Calculate metric contribution
    merged_df['metric contribution'] = merged_df['value'] * merged_df['weight']
    
    # Date processing
    merged_df['date'] = pd.to_datetime(merged_df['date'], format="%Y-%m-%d")
    merged_df['year'] = merged_df['date'].dt.year
    merged_df['month'] = merged_df['date'].dt.month

    # Calculate category average and differences
    merged_df['Category_MC_Avg'] = merged_df.groupby(['date', 'category', 'var'])['metric contribution'].transform('mean')
    merged_df['MC diff to catg avg'] = merged_df['metric contribution'] - merged_df['Category_MC_Avg']

    # Identify if Equity Pillar
    merged_df['Is Pillar'] = merged_df['Equity Pillar'].str.endswith('_pillar') & (merged_df['Equity Pillar'] != '')

    # Preprocess scaled scores
    filtered_df = scaled_scores_long[scaled_scores_long['variable'].str.endswith('_scores')].copy()
    filtered_df['var'] = filtered_df['variable'].str.replace('_scores$', '', regex=True)
    filtered_df['date'] = pd.to_datetime(filtered_df['date'], format="%Y-%m-%d")

    # Merge with scaled scores
    merged_df1 = pd.merge(merged_df, filtered_df[['date', 'brand', 'category', 'var', 'column_value']],
                        on=['date', 'brand', 'category', 'var'], how='left')
    merged_df1 = merged_df1.rename(columns={'column_value': 'Scaled Values'})

    # Preprocess importance model results
    filtered_imp_model_results = imp_model_results[imp_model_results['Model'] == 'Random Forest'][['Brand', 'Category', 'Shap Features', 'shap values']].copy()
    
    # Calculate the sum of SHAP values and relative importance
    sum_shap_values = filtered_imp_model_results.groupby(['Brand', 'Category'])['shap values'].transform('sum')
    filtered_imp_model_results['Relative Importance'] = filtered_imp_model_results['shap values'] / sum_shap_values

    # Preprocess IDV list
    attr_df4 = idv_list.rename(columns={'idv_for_model_corrected': 'var'}).drop_duplicates()
    selected_columns = merged_df1[['var']]
    
    # Merge to get metric details
    merged_df3 = pd.merge(selected_columns, attr_df4[['var', 'metric_name', 'data_source', 'Vendor Metric Group']], on=['var'], how='left')

    # Classify metric type
    merged_df3['Metric Type'] = np.select(
        [merged_df3['var'].str.contains('_mean'),
        merged_df3['var'].str.contains('_rank_1st'),
        merged_df3['var'].str.contains('_net')],
        ['mean', 'rank_1st', 'net'],
        default=''
    )

    return merged_df1, filtered_imp_model_results, merged_df3

# COMMAND ----------

 def scorecard_summary(df):
    df = df.drop(columns=['value', 'metric contribution','Category_MC_Avg', 'MC diff to catg avg'])
    df["Is Pillar"] = np.where(df["Is Pillar"] == False, "Metric", "Pillar")
    return df

# COMMAND ----------

def prepare_harmonized_data(harmonized_data):
    harmonized_data["date"] = pd.to_datetime(harmonized_data["date"])
    df_long = pd.melt(harmonized_data, id_vars=['brand_group_expanded', 'category', 'date'], 
                    var_name='var', value_name='metric_raw_value')
    df_long = df_long.rename(columns={"brand_group_expanded": "brand"})
    return df_long

# COMMAND ----------

def merge_and_prepare_summary(detailed, df_long, price_class_mapping, dashboard_metric_names_mapping):
    price_class_mapping["Price Class"] = price_class_mapping["Price Class"].fillna("#NA")
    detailed["date"] = pd.to_datetime(detailed["date"])
    
    summary_hd = pd.merge(detailed, df_long, on=["brand", "category", "var", "date"], how="left")
    summary_hd_price = pd.merge(summary_hd, price_class_mapping, 
                                left_on=["brand", "category"], right_on=["Brand", "Category"], how="left")
    
    dashboard_metric_names_mapping = dashboard_metric_names_mapping.drop(
        columns=['Harmonized_attribute', 'Transformation_logic', 'Modelled'])
    
    summary_renamed = pd.merge(summary_hd_price, dashboard_metric_names_mapping, 
                            left_on=["var"], right_on=["Scorecard_attribute"], how="left")
    summary_renamed['var'] = summary_renamed['Scorecard_attribute_updated'].fillna(summary_renamed['var'])
    summary_renamed = summary_renamed.drop(columns=['Scorecard_attribute', 'Scorecard_attribute_updated'])
    
    summary_renamed['Equity Pillar'] = summary_renamed['Equity Pillar'].str.replace('_pillar', '', regex=False) \
                                                                    .str.replace('_', ' ') \
                                                                    .str.title()
    return summary_renamed

# COMMAND ----------

def clean_and_rename_summary(summary_df):
    pillar_mapping = {
        'brand_perceptions_pillar': 'Brand Perceptions',
        'loyalty_pillar': 'Loyalty',
        'advocacy_pillar': 'Advocacy',
        'awareness_pillar': 'Awareness',
        'consideration_pillar': 'Consideration',
        'product_feedback_pillar': 'Product Feedback'
    }
    
    summary_df['var'] = summary_df['var'].replace(pillar_mapping)
    summary_df = summary_df.rename(columns={
        'Equity Pillar': 'equity_pillar',
        'Is Pillar': 'metric_type',
        'var': 'metric_name',
        'Scaled Values': 'scaled_scores',
        'Price Class': 'price_class'
    })
    
    summary_df = summary_df.drop(columns=["Brand", "Category", "year", "month"])
    
    selected_columns = ['brand', 'category', 'price_class', 'equity_pillar', 'metric_type', 
                        'metric_name', 'date', 'metric_raw_value', 'weight', 'scaled_scores']
    summary_df = summary_df[selected_columns]
    summary_df = summary_df.sort_values(by=['brand', 'category', 'equity_pillar', 
                                            'metric_type', 'metric_name', 'date'], ignore_index=True)
    summary_df = summary_df[~(summary_df['weight'].isnull() & (summary_df['metric_type'] == 'Metric'))]
    
    return summary_df

# COMMAND ----------

def round_columns(summary_df):
    numeric_cols = ['metric_raw_value', 'weight', 'scaled_scores']
    for col, decimals in zip(numeric_cols, [5, 5, 2]):
        summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').round(decimals)
    return summary_df

# COMMAND ----------

def prepare_pillar_importances(pillar_importances, price_class_mapping):
    updated_pillars = pd.merge(pillar_importances, price_class_mapping, on=["Brand", "Category"], how="left")
    updated_pillars = updated_pillars.rename(columns={
        "Shap Features": "equity_pillar",
        "Relative Importance": "relative_importance",
        "Price Class": "price_class",
        "Brand": "brand",
        "Category": "category"
    })
    
    updated_pillars = updated_pillars[["brand", "category", "price_class", "equity_pillar", "relative_importance"]]
    updated_pillars['equity_pillar'] = updated_pillars['equity_pillar'].str.replace('_pillar_trend_past', '', regex=False) \
                                                                    .str.replace('_', ' ') \
                                                                    .str.title()
    updated_pillars = updated_pillars.sort_values(by=['brand', 'category', 'price_class', 'equity_pillar'], ignore_index=True)
    updated_pillars['relative_importance'] = pd.to_numeric(updated_pillars['relative_importance'], errors='coerce').round(4)
    
    return updated_pillars

# COMMAND ----------

def updated_scorecard_format(detailed, harmonized_data, pillar_importances, dashboard_metric_names_mapping, price_class_mapping):
    # Main processing
    df_long = prepare_harmonized_data(harmonized_data)
    summary_hd_price_renamed = merge_and_prepare_summary(detailed, df_long, price_class_mapping, dashboard_metric_names_mapping)
    updated_summary = clean_and_rename_summary(summary_hd_price_renamed)
    updated_summary = round_columns(updated_summary)
    updated_pillar_importances = prepare_pillar_importances(pillar_importances, price_class_mapping)
    
    return updated_summary, updated_pillar_importances

# COMMAND ----------

def scoring(input_config, output_config, mapping_config, storage_options, refresh_config, feat_eng_config, filter_config, refresh_type):
    '''
    scoring()
    1. weights creation (using cfa and rf1 results)
    2. pillar creation (using weights)
    3. trend creation (using pillar)
    4. scaled scores creation (using pillar)

    post_modeling()
    1. pillar importance model (using trend pillars)
    2. scorecard creation (detailed, summary and relative importance files)
    3. updated scorecard format
    '''

    #input
    # print("post modelling 1- staging_output_path:",staging_output_path)
    eq_sub_scale_merged_brand = pd.read_csv(output_config["data_prep"]["eq_sub_scale"], storage_options = storage_options)

    equity_dt = pd.read_csv(output_config["data_prep"]["equity_dt"], storage_options = storage_options)
    # Convert the 'date' column to datetime format
    eq_sub_scale_merged_brand['date'] = pd.to_datetime(eq_sub_scale_merged_brand['date'], utc=False)
    equity_dt['date'] = pd.to_datetime(equity_dt['date'], utc=False)

    processed_harmonized_data = pd.read_csv(output_config['processed_input_data'], storage_options = storage_options)
    processed_harmonized_data['date'] = pd.to_datetime(processed_harmonized_data['date'], utc=False)
    # output_file_path = f"{static_output_path}/dashboard_metric_names_mapping.csv"
    dashboard_metric_names_mapping = pd.read_excel(mapping_config["dashboard_metric_names_mapping"], storage_options = storage_options)

    # output_file_path = f"{static_output_path}/price_class_mapping.csv"
    price_class_mapping = pd.read_csv(mapping_config["price_class_mapping"], storage_options = storage_options)

    # output_file_path = f"{static_output_path}/idv_list.csv"
    req_cols = pd.read_csv(mapping_config["idv_list"], storage_options = storage_options)

    if refresh_type == "model_refresh":
        all_pillar_results = pd.read_csv(output_config["weights_model"]["model_results"], storage_options = storage_options)
        # all_pillar_results["date"] = pd.to_datetime(all_pillar_results["date"], utc=False)
        nielsen_rms_data = pd.read_csv(output_config['processed_sales_data'], storage_options = storage_options)
        nielsen_rms_data["date"] = pd.to_datetime(nielsen_rms_data["date"], utc=False)
        # print("post modelling 2- staging_output_path:",staging_output_path)

        fit_summary_all_cat = pd.read_csv(output_config["cfa"]["model_results_all_category"], storage_options = storage_options)

        fit_summary_all_brands = pd.read_csv(output_config["cfa"]["model_results_by_category"], storage_options = storage_options)

        weights_sheet = weights_creation(fit_summary_all_cat, fit_summary_all_brands, refresh_config, feat_eng_config, eq_sub_scale_merged_brand, all_pillar_results, equity_dt) #1. done

        weights_sheet.to_csv(output_config["pillar_creation"]["weights_sheet"],index=False, storage_options = storage_options)
        

    if refresh_type == "model_scoring":
        weights_sheet = pd.read_csv(input_config["weights_sheet"], storage_options = storage_options)
    # Rename all columns to lowercase
    weights_sheet.columns = weights_sheet.columns.str.lower()
    eq_sub_scale_merged_brand_stacked = eq_sub_scale_merged_brand[eq_sub_scale_merged_brand['new_brand'] == "Stacked Brand"]

    index_df = pd.DataFrame()
    index_df_long = pd.DataFrame()
    # index_df, index_df_long = create_pillar_scores(eq_sub_scale_merged_brand_stacked, weights_sheet,req_cols)
    for category in weights_sheet['category'].unique():
        for pillar in weights_sheet['pillar'].unique():
            index_df_final1, index_df_long1 = create_pillar_scores(eq_sub_scale_merged_brand_stacked, weights_sheet,req_cols, category, pillar, equity_dt, feat_eng_config) #2. done
            index_df = pd.concat([index_df, index_df_final1], ignore_index=True)
            index_df_long = pd.concat([index_df_long, index_df_long1], ignore_index=True)


    index_df_long.to_csv(output_config["pillar_creation"]["pillars_long_format"],index=False, storage_options = storage_options)
    index_df.to_csv(output_config["pillar_creation"]["pillars"],index=False, storage_options = storage_options)

    # index_df = pd.read_csv(output_config["pillar_creation"]["pillars"], storage_options = storage_options)

    # final_merged_df = pillar_trend_creation(index_df)
    final_merged_df = pd.DataFrame()
    for brand in index_df['brand'].unique():
        brand_data = index_df[index_df['brand'] == brand]
        for category in brand_data['category'].unique():
            # pillar_cols = [col for col in index_df.columns if col.endswith('_pillar')]
            for pillar in refresh_config["pillars"]["by_category_pillars"] + refresh_config["pillars"]["all_category_pillars"]:
                br_cat_df = pillar_trend_creation(index_df, brand, category, pillar, refresh_config) #3. done
                final_merged_df = pd.concat([final_merged_df, br_cat_df], ignore_index=True)
    final_merged_df.to_csv(output_config["trend_pillar"]["trend_pillars"],index=False, storage_options = storage_options)

    # scaled_scores, scaled_scores_long = scaled_scores(index_df)
    # Initialize processed dataframes list
    processed_dataframes = []

    df1, grouped, pillar_columns, columns_to_mean, unique_combinations = scaled_scores_prep(
        index_df, feat_eng_config, filter_config, refresh_config
    )

    Iterate over unique combinations of date and category
    for _, group in unique_combinations.iterrows():
        date = group["date"]
        category = group['category']

        # Process a single combination
        processed_subset = compute_scaled_scores(
            date, category, df1, grouped, pillar_columns, columns_to_mean, filter_config["scaled_score"]["only_pillars"]
        ) #5. done

        # Append the processed subset to the list
        processed_dataframes.append(processed_subset)

    # Combine all processed dataframes
    scaled_scores = pd.concat(processed_dataframes, ignore_index=True)

    # Melt the result DataFrame
    columns_to_pivot = [col for col in scaled_scores.columns if col not in ['date', 'brand', 'category', refresh_config["dv"]]]
    scaled_scores_long = pd.melt(scaled_scores, id_vars=['date', 'brand', 'category'], value_vars=columns_to_pivot, value_name='column_value')

    scaled_scores.to_csv(output_config["scaled_scores"]["scaled_pillars"],index=False, storage_options = storage_options)
    scaled_scores_long.to_csv(output_config["scaled_scores"]["scaled_pillars_long_format"],index=False, storage_options = storage_options)

# COMMAND ----------

def post_modelling(input_config,output_config, mapping_config, storage_options, refresh_config, feat_eng_config, filter_config, refresh_type):

    final_merged_df = pd.read_csv(output_config["trend_pillar"]["trend_pillars"], storage_options = storage_options)
    final_merged_df["date"] = pd.to_datetime(final_merged_df["date"])
    nielsen_rms_data = pd.read_csv(output_config['processed_sales_data'], storage_options = storage_options)
    nielsen_rms_data["date"] = pd.to_datetime(nielsen_rms_data["date"])
    index_df = pd.read_csv(output_config["pillar_creation"]["pillars"], storage_options = storage_options)
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df_long = pd.read_csv(output_config["pillar_creation"]["pillars_long_format"], storage_options = storage_options)
    index_df_long["date"] = pd.to_datetime(index_df_long["date"])
    weights_sheet = pd.read_csv(output_config["pillar_creation"]["weights_sheet"], storage_options = storage_options)
    scaled_scores_long = pd.read_csv(output_config["scaled_scores"]["scaled_pillars_long_format"], storage_options = storage_options)
    scaled_scores_long["date"] = pd.to_datetime(scaled_scores_long["date"])
    req_cols = pd.read_csv(mapping_config["idv_list"], storage_options = storage_options)
    processed_harmonized_data = pd.read_csv(output_config['processed_input_data'], storage_options = storage_options)
    processed_harmonized_data["date"] = pd.to_datetime(processed_harmonized_data["date"])

    if (refresh_type == "model_refresh") | ((refresh_type == "model_scoring") & (refresh_config["run_importance_model_for_scoring_refresh"] == True)):
        # results_all_model = pillar_importance_model(final_merged_df, nielsen_rms_data)
        results_all_model = pd.DataFrame()
        for brand in final_merged_df.brand.unique():
            br_df = final_merged_df[final_merged_df['brand'] == brand]
            for category in br_df['category'].unique():
                results = pillar_importance_model(final_merged_df, nielsen_rms_data, feat_eng_config, brand, category) #4. done
                results_all_model = pd.concat([results_all_model, results])

    if (refresh_type == "model_refresh") | ((refresh_type == "model_scoring") & (refresh_config["run_importance_model_for_scoring_refresh"] == True)):
        results_all_model.to_csv(output_config["importance_model"]["model_results"],index=False, storage_options = storage_options)

    if ((refresh_type == "model_scoring") & (refresh_config["run_importance_model_for_scoring_refresh"] == False)):
        results_all_model = pd.read_csv(output_config["importance_model"]["model_results"], storage_options = storage_options)

    hub_data, pillar_importances, var_map = data_hub_data_format_code(pillars_long_data=index_df_long, weights_sheet=weights_sheet, scaled_scores_long=scaled_scores_long, imp_model_results=results_all_model, idv_list=req_cols)

    hub_data.to_csv(output_config["pillar_importances"]["hub_data"],index=False, storage_options = storage_options)
    pillar_importances.to_csv(output_config["pillar_importances"]["pillar_importances"],index=False, storage_options = storage_options)
    var_map.to_csv(output_config["pillar_importances"]["variable_mapping"],index=False, storage_options = storage_options)

    scorecard_detailed = hub_data.copy()
    scorecard_summary = scorecard_summary(scorecard_detailed)

    hub_data.to_csv(output_config["scorecard"]["detailed"],index=False, storage_options = storage_options)

    scorecard_summary.to_csv(output_config["scorecard"]["summary"],index=False, storage_options = storage_options)

    pillar_importances.to_csv(output_config["scorecard"]["pillar_importances"],index=False, storage_options = storage_options)

    updated_summary, updated_pillar_importances = updated_scorecard_format(scorecard_detailed, processed_harmonized_data, pillar_importances, mapping_config)

    print("post modelling 3- staging_output_path:",staging_output_path)
    print("summary path:",output_config["updated_scorecard"]["updated_summary"])
    updated_summary.to_csv(output_config["updated_scorecard"]["updated_summary"],index=False, storage_options = storage_options)
    updated_pillar_importances.to_csv(output_config["updated_scorecard"]["updated_pillar_importances"],index=False, storage_options = storage_options)


# COMMAND ----------

# def post_modelling(input_config,output_config, storage_options, refresh_config, feat_eng_config, filter_config, refresh_type, staging_output_path, static_output_path):
#     #input
#     print("post modelling 1- staging_output_path:",staging_output_path)
#     eq_sub_scale_merged_brand = pd.read_csv(output_config["data_prep"]["eq_sub_scale"], storage_options = storage_options)

#     equity_dt = pd.read_csv(output_config["data_prep"]["equity_dt"], storage_options = storage_options)
#     # Convert the 'date' column to datetime format
#     equity_dt['date'] = pd.to_datetime(equity_dt['date'], utc=False)

#     processed_harmonized_data = pd.read_csv(output_config['processed_input_data'], storage_options = storage_options)

#     output_file_path = f"{static_output_path}/dashboard_metric_names_mapping.csv"
#     dashboard_metric_names_mapping = pd.read_csv(output_file_path, storage_options = storage_options)

#     output_file_path = f"{static_output_path}/price_class_mapping.csv"
#     price_class_mapping = pd.read_csv(output_file_path, storage_options = storage_options)

#     output_file_path = f"{static_output_path}/idv_list.csv"
#     req_cols = pd.read_csv(output_file_path, storage_options = storage_options)

#     if refresh_type == "model_refresh":
#         all_pillar_results = pd.read_csv(output_config["weights_model"]["model_results"], storage_options = storage_options)

#         nielsen_rms_data = pd.read_csv(output_config['processed_sales_data'], storage_options = storage_options)

#         print("post modelling 2- staging_output_path:",staging_output_path)

#         fit_summary_all_cat_py = pd.read_csv(output_config["cfa"]["model_results_all_category"], storage_options = storage_options)

#         fit_summary_all_brands_py = pd.read_csv(output_config["cfa"]["model_results_by_category"], storage_options = storage_options)

#         weights_sheet = weights_creation(fit_summary_all_cat, fit_summary_all_brands, category_list, refresh_config, feat_eng_config, eq_sub_scale_merged_brand, all_pillar_results, equity_dt) #1. done

#         weights_sheet.to_csv(output_config["pillar_creation"]["weights_sheet"],index=False, storage_options = storage_options)
        

#     if refresh_type == "model_scoring":
#         weights_sheet = pd.read_csv(input_config["weights_sheet"], storage_options = storage_options)
#     # Rename all columns to lowercase
#     weights_sheet.columns = weights_sheet.columns.str.lower()
#     eq_sub_scale_merged_brand_stacked = eq_sub_scale_merged_brand[eq_sub_scale_merged_brand['New_Brand'] == "Stacked Brand"]

#     index_df = pd.DataFrame()
#     index_df_long = pd.DataFrame()
#     # index_df, index_df_long = create_pillar_scores(eq_sub_scale_merged_brand_stacked, weights_sheet,req_cols)
#     for category in weights_sheet['category'].unique():
#         for pillar in weights_sheet['pillar'].unique():
#             index_df_final1, index_df_long1 = create_pillar_scores(eq_sub_scale_merged_brand_stacked, weights_sheet,req_cols, category, pillar, equity_dt, feat_eng_config) #2. done
#             index_df = pd.concat([index_df, index_df_final1], ignore_index=True)
#             index_df_long = pd.concat([index_df_long, index_df_long1], ignore_index=True)


#     index_df_long.to_csv(output_config["pillar_creation"]["pillars_long_format"],index=False, storage_options = storage_options)
#     index_df.to_csv(output_config["pillar_creation"]["pillars"],index=False, storage_options = storage_options)

#     # index_df = pd.read_csv(output_config["pillar_creation"]["pillars"], storage_options = storage_options)

#     # final_merged_df = pillar_trend_creation(index_df)
#     final_merged_df = pd.DataFrame()
#     for brand in index_df['brand'].unique():
#         brand_data = index_df[index_df['brand'] == brand]
#         for category in brand_data['category'].unique():
#             pillar_cols = [col for col in index_df.columns if col.endswith('_pillar')]
#             for pillar in pillar_cols:
#                 br_cat_df = pillar_trend_creation(index_df, brand, category, pillar, refresh_config) #3. done
#                 final_merged_df = pd.concat([final_merged_df, br_cat_df], ignore_index=True)
#     final_merged_df.to_csv(output_config["trend_pillar"]["trend_pillars"],index=False, storage_options = storage_options)

#     if (refresh_type == "model_refresh") | ((refresh_type == "model_scoring") & (refresh_config["run_importance_model_for_scoring_refresh"] == True)):
#         # results_all_model = pillar_importance_model(final_merged_df, nielsen_rms_data)
#         results_all_model = pd.DataFrame()
#         for brand in final_merged_df.brand.unique():
#             br_df = final_merged_df[final_merged_df['brand'] == brand]
#             for category in br_df['category_new'].unique():
#                 results = pillar_importance_model(final_merged_df, nielsen_rms_data, feat_eng_config) #4. done
#                 results_all_model = pd.concat([results_all_model, results])

#     if (refresh_type == "model_refresh") | ((refresh_type == "model_scoring") & (refresh_config["run_importance_model_for_scoring_refresh"] == True)):
#         results_all_model.to_csv(output_config["importance_model"]["model_results"],index=False, storage_options = storage_options)

#     if ((refresh_type == "model_scoring") & (refresh_config["run_importance_model_for_scoring_refresh"] == False)):
#         results_all_model = pd.read_csv('abfss://restricted-dataoperations@npusdvdatalakesta.dfs.core.windows.net/staging/cmi_brand_hub/score_card_data/output/random_forest2/model_results_all_cat.csv', storage_options = storage_options)

#     # scaled_scores, scaled_scores_long = scaled_scores(index_df)
#     # Initialize processed dataframes list
#     processed_dataframes = []

#     df1, grouped, pillar_columns, columns_to_mean, unique_combinations = scaled_scores_prep(
#         index_df, feat_eng_config, filter_config, refresh_config
#     )

#     # Iterate over unique combinations of date and category
#     for _, group in unique_combinations.iterrows():
#         date = group["date"]
#         category = group['category']

#         # Process a single combination
#         processed_subset = compute_scaled_scores(
#             date, category, df1, grouped, pillar_columns, columns_to_mean, feat_eng_config["scaled_score"]["only_pillars"]
#         ) #5. done

#         # Append the processed subset to the list
#         processed_dataframes.append(processed_subset)

#     # Combine all processed dataframes
#     scaled_scores = pd.concat(processed_dataframes, ignore_index=True)

#     # Melt the result DataFrame
#     columns_to_pivot = [col for col in scaled_scores.columns if col not in ['date', 'brand', 'category', refresh_config["DV"]]]
#     scaled_scores_long = pd.melt(scaled_scores, id_vars=['date', 'brand', 'category'], value_vars=columns_to_pivot, value_name='column_value')

#     scaled_scores.to_csv(output_config["scaled_scores"]["scaled_pillars"],index=False, storage_options = storage_options)
#     scaled_scores_long.to_csv(output_config["scaled_scores"]["scaled_pillars_long_format"],index=False, storage_options = storage_options)

#     index_df_long = pd.read_csv(output_config["pillar_creation"]["pillars_long_format"], storage_options = storage_options)



#     hub_data, pillar_importances, var_map = data_hub_data_format_code(pillars_long_data=index_df_long, weights_sheet=weights_sheet, scaled_scores_long=scaled_scores_long, imp_model_results=results_all_model, idv_list=req_cols)

#     hub_data.to_csv(output_config["pillar_importances"]["hub_data"],index=False, storage_options = storage_options)
#     pillar_importances.to_csv(output_config["pillar_importances"]["pillar_importances"],index=False, storage_options = storage_options)
#     var_map.to_csv(output_config["pillar_importances"]["variable_mapping"],index=False, storage_options = storage_options)

#     scorecard_detailed = hub_data.copy()
#     scorecard_summary = scorecard_summary(scorecard_detailed)

#     hub_data.to_csv(output_config["scorecard"]["detailed"],index=False, storage_options = storage_options)

#     scorecard_summary.to_csv(output_config["scorecard"]["summary"],index=False, storage_options = storage_options)

#     pillar_importances.to_csv(output_config["scorecard"]["pillar_importances"],index=False, storage_options = storage_options)

#     updated_summary, updated_pillar_importances = updated_scorecard_format(scorecard_detailed, processed_harmonized_data, pillar_importances, dashboard_metric_names_mapping, price_class_mapping)

#     print("post modelling 3- staging_output_path:",staging_output_path)
#     print("summary path:",output_config["updated_scorecard"]["updated_summary"])
#     updated_summary.to_csv(output_config["updated_scorecard"]["updated_summary"],index=False, storage_options = storage_options)
#     updated_pillar_importances.to_csv(output_config["updated_scorecard"]["updated_pillar_importances"],index=False, storage_options = storage_options)



# COMMAND ----------

# Initialize processed dataframes list
processed_dataframes = []

# Convert date column to datetime
pillar_data['date'] = pd.to_datetime(pillar_data['date'], format="%Y-%m-%d")

# Define relevant columns for grouping and scaling
pillar_columns = [col for col in pillar_data.columns if col.endswith('_pillar')]
additional_columns = ['date', 'category', refresh_config["dv"], 'brand']

if filter_config["scaled_score"]["only_pillars"]:
    grouped = pillar_data.groupby(['date', 'category'])[pillar_columns].mean().reset_index()
    df1 = pillar_data[additional_columns + pillar_columns]
else:
    columns_to_mean = pillar_data.columns.difference(['date', 'brand', 'category', refresh_config["dv"]]).tolist()
    grouped = pillar_data.groupby(['date', 'category'])[columns_to_mean].mean().reset_index()
    df1 = pillar_data

# Extract unique combinations of date and category
unique_combinations = df1[['date', 'category']].drop_duplicates()

# Iterate over unique combinations of date and category
for _, group in unique_combinations.iterrows():
    date = group["date"]
    category = group['category']

    # Process a single combination
    subset1 = df1[(df1['date'] == date) & (df1['category'] == category)].copy()
    subset = grouped[(grouped['date'] == date) & (grouped['category'] == category)]

    for column_name in (pillar_columns if only_pillars else columns_to_mean):
        new_column_name = f"{column_name}_scores"
        average = subset[column_name].values[0]  # Assumes there's at least one row in subset
        subset1[new_column_name] = (subset1[column_name] - average) * 100 + 100

    # Append the processed subset to the list
    processed_dataframes.append(subset1)

# Combine all processed dataframes
scaled_scores = pd.concat(processed_dataframes, ignore_index=True)

# Melt the result DataFrame
columns_to_pivot = [col for col in scaled_scores.columns if col not in ['date', 'brand', 'category', refresh_config["dv"]]]
scaled_scores_long = pd.melt(scaled_scores, id_vars=['date', 'brand', 'category'], value_vars=columns_to_pivot, value_name='column_value')

# COMMAND ----------


