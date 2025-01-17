# Databricks notebook source
# MAGIC %md
# MAGIC before cleaning(v1)

# COMMAND ----------

# MAGIC %run ./configuration_function

# COMMAND ----------

# def modelling(input_config,output_config,mapping_config,refresh_config,filter_config,storage_options):
#     print(run_config['refresh_type'])
#     print(run_config['input_scoring_date'])
#     eq_sub_scale_merged_brand = pd.read_csv(output_config["data_prep"]["eq_sub_scale"], storage_options = storage_options)

#     modeling_data = pd.read_csv(output_config["data_prep"]["modeling_data"], storage_options = storage_options)

#     output_file_path = f"{static_output_path}/idv_list.csv"

#     req_cols = pd.read_csv(output_file_path, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))


#     # Activate automatic conversion
#     pandas2ri.activate()

#     # Convert to R DataFrame
#     eq_sub_scale_merged_brand_r_df = pandas2ri.py2rpy(eq_sub_scale_merged_brand)

#     # Set default CRAN mirror
#     robjects.r('options(repos = c(CRAN="https://cran.rstudio.com/"))')

#     robjects.r('library(data.table)')  # Load the data.table library

#     req_cols_r_df = pandas2ri.py2rpy(req_cols)
#     # Pass the config to R as individual values
#     robjects.r.assign('ALL_CATEGORY_PILLARS', robjects.StrVector(model_config["cfa"]["pillars"]['all_category_pillars']))
#     robjects.r.assign('BY_CATEGORY_PILLARS', robjects.StrVector(model_config["cfa"]["pillars"]['by_category_pillars']))
#     robjects.r.assign('std_lv', model_config["cfa"]["std_lv"])
#     robjects.r.assign('check_gradient', model_config["cfa"]["check_gradient"])
#     robjects.r.assign('standardized_', model_config["cfa"]["standardized_"])
#     robjects.r.assign('fit_measures', model_config["cfa"]["fit_measures"])
#     robjects.r.assign('refresh_type', run_config["refresh_type"])
#     robjects.r.assign('sample_seeds', robjects.IntVector(model_config["cfa"]["sample_seeds"]))


#     # Assign the R DataFrame to the R environment
#     # robjects.r.assign('eq_sub_scale_merged_brand_r2', eq_sub_scale_merged_brand_r)
#     robjects.globalenv['eq_sub_scale_merged_brand_r_df'] = eq_sub_scale_merged_brand_r_df
#     robjects.globalenv['req_cols_r_df'] = req_cols_r_df
#     robjects.r('eq_sub_scale_merged_brand_r_dt <- as.data.table(eq_sub_scale_merged_brand_r_df)')  # Convert to data.table
#     robjects.r('req_cols_r_df <- as.data.frame(req_cols_r_df)')
#     # robjects.r('setDT(eq_sub_scale_merged_brand_r_dt)  # Ensure it is a data.table')

#     robjects.r('''
#     print(unique(eq_sub_scale_merged_brand_r_dt$date))
#     # Clean and convert to IDate
#     eq_sub_scale_merged_brand_r_dt[, date := as.IDate(date, format = "%Y-%m-%d")]

#     # Handle conversion errors (replace invalid dates with NA)
#     eq_sub_scale_merged_brand_r_dt[, date := suppressWarnings(as.IDate(date, format = "%Y-%m-%d"))]

#     eq_sub_scale_merged_brand_r_dt[, date := as.IDate(date)]
#     excluded_cols <- c("Brand","New_Brand","Category","date")
#     for (col in names(eq_sub_scale_merged_brand_r_dt)) {
#     if (!(col %in% excluded_cols)) {
#         # Replace NaN with NA
#         eq_sub_scale_merged_brand_r_dt[is.nan(get(col)), (col) := NA]

#         # Convert to numeric if necessary
#         eq_sub_scale_merged_brand_r_dt[, (col) := as.numeric(get(col))]
#     }
#     }
#     for (col in names(req_cols_r_df)) {
#     req_cols_r_df[[col]][is.nan(req_cols_r_df[[col]])] <- NA
#     }
#     ''')

#     # R cell with function
#     robjects.r('''
#     required_packages <- c("lavaan", "dplyr", "data.table", "semTools", "magrittr", "tidyr")

#     # Check for missing packages and install them
#     missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
#     if(length(missing_packages)) install.packages(missing_packages)

#     # Load libraries
#     library(lavaan)        # For Confirmatory Factor Analysis (CFA)
#     library(dplyr)         # For data manipulation (mutate, select, etc.)
#     library(tidyr)
#     library(data.table)    # For efficient data manipulation (copy, as.data.frame)
#     library(semTools)      # For SEM diagnostics (e.g., standardized solution)
#     library(magrittr)      # For piping operations (%>%)


#     cfa = function(fa_str,scaled_data,s){

#     fit_ <- sem(fa_str, data=scaled_data,std.lv=std_lv,check.gradient=check_gradient)
#     summary(fit_, standardized=standardized_, fit.measures=fit_measures)

#     # store fit measures - tli, cfi, rmsea
#     cfa_fit_indices <- as.data.frame(fitMeasures(fit_))
#     cfa_fit_indices$fit_measures <- row.names(cfa_fit_indices)
#     row.names(cfa_fit_indices) <- NULL
#     names(cfa_fit_indices) <- c("value","fitmeasure")
#     cfa_fit_indices <- cfa_fit_indices[cfa_fit_indices$fitmeasure %in% c("cfi","tli","rmsea"),]
#     cfa_fit_indices$value <- as.numeric(cfa_fit_indices$value)

#     cfa_fit_indices_t <- as.data.frame(t(cfa_fit_indices))
#     colnames(cfa_fit_indices_t) <- cfa_fit_indices$fitmeasure
#     cfa_fit_indices_t <- cfa_fit_indices_t["value",]

#     # Extract parameter estimates
#     cfa_estimates <- as.data.frame(standardizedsolution(fit_))

#     # cfa summary table
#     cfa_summary <- cbind(cfa_estimates, cfa_fit_indices_t)

#     # Store the results
#     cfa_summary$factor_str = fa_str

#     cfa_summary$Seed = s

#     # sample_data_ = rbind(sample_data_, eq_sub)
#     #fit_summary_ = rbind(fit_summary_, cfa_summary)

#     return(cfa_summary)
#     }
#     cfa_weights_r_func1 <- function(eq_sub_scale_merged_brand,req_cols) {
#     # print(R.version.string)
#     category_list_pillar <- unique(eq_sub_scale_merged_brand$Category)

#     ###------ CFA - all category ---- ###

#     sample_data_ = data.frame()
#     fit_indices_ = data.frame()
#     estimates_ = data.frame()
#     fit_summary_all_cat = data.frame()
#     corr_pillar_ = data.frame()
#     error_messages = c()

#     ## Selecting columns that needs to be taken for all categories together
#     req_cols_all_cat <- req_cols[req_cols$Select == 'Y', ]

#     ##taking stacked brand results
#     eq_sub_scale_merged_all_category_filtered_base <- eq_sub_scale_merged_brand[eq_sub_scale_merged_brand$New_Brand == "Stacked Brand",]
#     # ind_factor_list_all_cat <- unique(req_cols_all_cat$Equity Pillar)
#     # print('ind_factor_list_all_cat:',ind_factor_list_all_cat)

#     # Create the samples
#     eq_sub_scale_merged_all_category_filtered_list <- list()

#     equity_dt_seed_all_cat <- as.data.frame(eq_sub_scale_merged_all_category_filtered_base)


#     for (i in sample_seeds) {
#     # For different seeds
#     set.seed(i)

#     # Sample the data based on the sampled indicies
#     eq_sub_scale_merged_all_category_filtered_list[[i]]  <- equity_dt_seed_all_cat[sample(nrow(equity_dt_seed_all_cat),0.95*nrow(equity_dt_seed_all_cat)),]
#     }

#     for (s in sample_seeds)
#     {
#     print(paste("Seed: ",s))
#     eq_sub_scale_merged_all_category_filtered <- eq_sub_scale_merged_all_category_filtered_list[[s]]

#     complete_null_columns_all_cat <- colnames(eq_sub_scale_merged_all_category_filtered)[colSums(is.na(eq_sub_scale_merged_all_category_filtered)) == nrow(eq_sub_scale_merged_all_category_filtered)]
#     eq_sub_scale_merged_all_category_filtered <- select(eq_sub_scale_merged_all_category_filtered,-contains(complete_null_columns_all_cat))


#     eq_sub_scale_merged_all_category_filtered <- eq_sub_scale_merged_all_category_filtered %>% mutate_if(is.numeric, ~replace_na(.,mean(., na.rm = TRUE)))


#     all_cat_pillars_list <- ALL_CATEGORY_PILLARS
#     all_cat_pillars_list <- gsub('_pillar', '', all_cat_pillars_list)

#     #for (pillar_ in ind_factor_list_all_cat){
#     for (pillar_ in all_cat_pillars_list){
#         # pillar_ = "ratings_reviews"

#         print(pillar_)
#         cfa_fit_indices = data.frame()
#         cfa_estimates = data.frame()

#         l_pillar <- pillar_

#         # print("pillar_values_unique:",unique(req_cols_all_cat$`Equity Pillar`))
#         # print(colnames(req_cols_all_cat))
#         pillar_metrics <- unique(req_cols_all_cat[req_cols_all_cat$`Equity Pillar` == l_pillar,]$idv_for_model_corrected)
#         # print("pillar_metrics:",pillar_metrics)
#         l_n_pillar_metrics <- length(pillar_metrics)

#         # Intersection with metrics that are actually only available
#         pillar_a_metrics <- pillar_metrics[pillar_metrics %in% names(eq_sub_scale_merged_all_category_filtered)]
#         l_pillar_a_metrics <- pillar_metrics[!pillar_metrics %in% names(eq_sub_scale_merged_all_category_filtered)]
#         l_n_pillar_a_metrics <- length(pillar_a_metrics)

#         # Dynamic way of writing to factor structure
#         fa_str <- paste0(l_pillar,"_pillar =~ ",paste(pillar_a_metrics, collapse = "+"))

#         # Correlation of available metrics
#         corr_pillar <- as.data.frame(as.table(cor(eq_sub_scale_merged_all_category_filtered[pillar_a_metrics])))
#         corr_pillar$Pillar <- pillar_
#         corr_pillar_ = rbind(corr_pillar_, corr_pillar)

#         tryCatch({
#         # print("all_cat_fa_str :",fa_str)
#         fit_summary_c = cfa(fa_str,eq_sub_scale_merged_all_category_filtered,s)
#         if (nrow(fit_summary_all_cat) == 0) {
#         fit_summary_all_cat <- fit_summary_c  # Initialize if empty
#         } else {
#             fit_summary_all_cat <- rbind(fit_summary_all_cat, fit_summary_c)  # Append if not empty
#         }
#         },
#         error = function(e) {
#         # Capture and store the error message
#         #error_messages[[n_factors]] <- conditionMessage(e)
#         warning(paste0("Error in CFA in factor:",pillar_))
#         message(paste0("Error in CFA in factor:",pillar_))

#         }
#         )
#     }
#     }

#     fit_summary_all_cat$Brands = "Stacked Brand"
#     fit_summary_all_cat$Category = "ALL CATEGORY"

#     corr_pillar_all_cat <- copy(corr_pillar_)
#     corr_pillar_all_cat$Brands = "Stacked Brand"
#     corr_pillar_all_cat$Category = "ALL CATEGORY"

#     ### --- CFA -by category ---###

#     fit_summary_all_brands = data.frame()
#     corr_pillar_all_brands = data.frame()
#     for(category_ in category_list_pillar){

#     tryCatch({
#         #category_ = "CAT FOOD"

#         req_cols1 <- req_cols[req_cols$Select == 'Y' & req_cols$product_category_idv == category_,]
#         eq_sub_scale_merged_category_filtered <- eq_sub_scale_merged_brand[eq_sub_scale_merged_brand$Category == category_,]
#         brand_list_pillar <- "Stacked Brand"

#         for (brand in brand_list_pillar){

#         eq_sub_scale_merged_brand_category_filtered_base <- eq_sub_scale_merged_category_filtered[eq_sub_scale_merged_category_filtered$New_Brand == brand,]

#         complete_null_columns_by_cat <- colnames(eq_sub_scale_merged_brand_category_filtered_base)[colSums(is.na(eq_sub_scale_merged_brand_category_filtered_base)) == nrow(eq_sub_scale_merged_brand_category_filtered_base)]
#         eq_sub_scale_merged_brand_category_filtered_base <- select(eq_sub_scale_merged_brand_category_filtered_base,-contains(complete_null_columns_by_cat))

#         eq_sub_scale_merged_brand_category_filtered_base <- eq_sub_scale_merged_brand_category_filtered_base %>% mutate_if(is.numeric, ~replace_na(.,mean(., na.rm = TRUE)))

#         # Access each factor structure and apply cfa
#         sample_data_ = data.frame()
#         fit_indices_ = data.frame()
#         estimates_ = data.frame()
#         fit_summary_ = data.frame()
#         corr_pillar_ = data.frame()
#         error_messages = c()

#         ind_factor_list <- unique(req_cols1$`Equity Pillar`)

#         by_cat_pillars_list <- BY_CATEGORY_PILLARS
#         by_cat_pillars_list <- gsub('_pillar', '', by_cat_pillars_list)

#         # Create the samples
#         eq_sub_scale_merged_brand_category_filtered_list <- list()

#         # Ensure main data frame is data table
#         equity_dt_seed <- as.data.frame(eq_sub_scale_merged_brand_category_filtered_base)

#         # Loop through each seeds
#         for (i in sample_seeds) {
#             # For different seeds
#             set.seed(i)

#             # Sample the data based on the sampled indicies
#             eq_sub_scale_merged_brand_category_filtered_list[[i]]  <- equity_dt_seed[sample(nrow(equity_dt_seed),0.95*nrow(equity_dt_seed)),]
#         }

#         for (s in sample_seeds)
#         {
#             print(paste("seed:",s))
#             eq_sub_scale_merged_brand_category_filtered <- eq_sub_scale_merged_brand_category_filtered_list[[s]]

#             for (pillar_ in by_cat_pillars_list){
#             # for (pillar_ in list('product_feedback')) {
#                 fa_str <- ""
#                 pillar_a_metrics <- list()

#                 print(paste('Pillar_name:', str(pillar_)))

#                 # Filter pillar metrics
#                 pillar_metrics <- unique(req_cols1$idv_for_model_corrected[
#                     (req_cols1$`Equity Pillar` == pillar_) &
#                     (req_cols1$product_category_idv == category_ | req_cols1$product_category_idv == 'All')
#                 ])
#                 print("pillar_metrics")
#                 print(pillar_metrics)

#                 pillar_a_metrics <- pillar_metrics[pillar_metrics %in% names(eq_sub_scale_merged_brand_category_filtered)]
#                 print("pillar_a_metrics")
#                 print(pillar_a_metrics)

#                 if (length(pillar_a_metrics) > 0) {
#                     fa_str <- paste0(pillar_, "_pillar =~ ", paste(pillar_a_metrics, collapse = "+"))
#                     print("fa_str")
#                     print(fa_str)
#                 } else {
#                     message(paste('No metrics available for pillar:', pillar_))
#                     next  # Skip to the next iteration
#                 }

#                 tryCatch({
#                     fit_summary_ <- cfa(fa_str, eq_sub_scale_merged_brand_category_filtered,s)

#                     # Investigate variance issues
#                     print(varTable(fit_summary_))

#                     # Append results
#                     fit_summary_b <- copy(fit_summary_)
#                     fit_summary_b$Brands <- brand
#                     fit_summary_b$Category <- category_
#                     # fit_summary_all_brands <- rbind(fit_summary_all_brands, fit_summary_b)
#                     if (nrow(fit_summary_all_brands) == 0) {
#                     fit_summary_all_brands <- fit_summary_b  # Initialize if empty
#                     } else {
#                         fit_summary_all_brands <- rbind(fit_summary_all_brands, fit_summary_b)  # Append if not empty
#                     }
#                     # Handle correlation data
#                     corr_pillar <- as.data.frame(as.table(cor(eq_sub_scale_merged_brand_category_filtered[pillar_a_metrics])))
#                     corr_pillar$Pillar <- pillar_
#                     corr_pillar_ <- rbind(corr_pillar_, corr_pillar)

#                     corr_pillar_b <- copy(corr_pillar_)
#                     corr_pillar_b$Brands <- brand
#                     corr_pillar_b$Category <- category_
#                     # corr_pillar_all_brands <- rbind(corr_pillar_all_brands, corr_pillar_b)

#                     if (nrow(corr_pillar_all_brands) == 0) {
#                     corr_pillar_all_brands <- corr_pillar_b  # Initialize if empty
#                     } else {
#                         corr_pillar_all_brands <- rbind(corr_pillar_all_brands, corr_pillar_b)  # Append if not empty
#                     }
#                 }, error = function(e) {
#                     message(paste('Error in CFA fit for pillar:', pillar_))
#                     message(conditionMessage(e))  # Detailed error message
#                     next
#                 })
#             }
#         }


#         }
#     },
#     error = function(e) {
#         # Capture and store the error message
#         warning(paste0("Error in CFA in factor:",pillar_,brand,category_," seed: ",s))
#         message(paste0("Error in CFA in factor:",pillar_,brand,category_," seed: ",s))

#     }
#     )
#     #
#     }

#     return(list(fit_summary_all_cat = fit_summary_all_cat, fit_summary_all_brands = fit_summary_all_brands))
#     }
#     ''')

#     # Call the R function

#     robjects.r('''
#             if(refresh_type == "full") {
#             cfa_fit_results <- cfa_weights_r_func1(eq_sub_scale_merged_brand_r_dt,req_cols_r_df)
#             fit_summary_all_cat <- cfa_fit_results$fit_summary_all_cat
#             fit_summary_all_brands <- cfa_fit_results$fit_summary_all_brands
#             }
#             ''')

#     # Convert result back to pandas DataFrame
#     if run_config["refresh_type"] == "full":
#         cfa_fit_results_py = pandas2ri.rpy2py(robjects.globalenv['cfa_fit_results'])
#         fit_summary_all_cat_py = pandas2ri.rpy2py(robjects.globalenv['fit_summary_all_cat'])
#         fit_summary_all_brands_py = pandas2ri.rpy2py(robjects.globalenv['fit_summary_all_brands'])

#     if run_config["refresh_type"] == "full":
#         fit_summary_all_cat_py.to_csv(output_config["cfa"]["model_results_all_category"], index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))
#         fit_summary_all_brands_py.to_csv(output_config["cfa"]["model_results_by_category"],index=False, **({'storage_options': {'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}} if run_config["platform"] == "databricks" else {}))


#     def metric_to_pillar_model_weights():

#         start = time.time()
#         attr_df1 = modeling_data.copy()
#         # attr_df1 = pd.read_csv(DATA_PATH,index_col=0)
#         # attr_df1
#         attr_df1['date'] = pd.to_datetime(attr_df1['date'], format="%Y-%m-%d")
#         attr_df1 = attr_df1.sort_values(by='date', ascending=True)
#         # attr_df1['month_rt'] = attr_df1['date'].dt.month
#         # attr_df1['year_rt'] = attr_df1['date'].dt.year

#         # Assuming df is your DataFrame
#         attr_df1.rename(columns={'brand': 'brand_rt', 'category': 'category_rt', 'market_share': 'market_share_total_sales_rt'}, inplace=True)

#         ##Framework starts
#         if model_config["weights_model"]["log_convert_DV"]:
#             dep_var = 'log_'+ model_config["weights_model"]["DV"]
#         else:
#             dep_var = model_config["weights_model"]["DV"]

#         # trial = model_config["random_forest1"]["Trial"]

#         logdf = pd.DataFrame(columns = ['Brand' , 'Category', 'Nrows', 'Success','e','Error'])


#         ##############################################
#         # 3. Load necessary data
#         # (Needs to be configured to read from data lake/ data hub)
#         ##############################################

#         # Harmonized data (Provided by data hub)

#         attr_df = attr_df1.copy()
#         if model_config["weights_model"]["drop_mean_attributes"]:
#             attr_df = attr_df[attr_df.columns.drop(list(attr_df.filter(regex='_mean')))]

#         if model_config["weights_model"]["drop_rank1st_attributes"]:
#             attr_df = attr_df[attr_df.columns.drop(list(attr_df.filter(regex='_rank_1st')))]
#         all_pillar_results =pd.DataFrame()
#         corr_all_results =pd.DataFrame()

#         ## Predefined list of IDVs (Created within DS team)

#         idv_sel_cols = req_cols.copy()
#         idv_sel_cols.drop(columns=["Unnamed: 0"],inplace=True)

#         for pillar in model_config["weights_model"]["pillars_list"]:
#             print(pillar)
#             Metric_Group_list =[]
#             Metric_Group_list.append(pillar)
#             temp_title = pillar
#             pillar_name = pillar
#             output_file_name = pillar
#             idv_sel_cols = req_cols.copy()
#         ## IDV columns to select
#             idv_sel_cols = idv_sel_cols[idv_sel_cols['Equity Pillar'].isin( Metric_Group_list)]
#             #pillar_name = "advocacy"

#             ##############################################
#             # 4. Data cleaning
#             # (Convert to single function that performs same data cleaning operations on any new file)
#             ##############################################

#             # Replacing spaces with underscore and removing ' - ' in column names
#             attr_df.columns = attr_df.columns.str.replace(' - ','_')
#             attr_df.columns = attr_df.columns.str.replace(' ','_')
#             # removing case sensitivity
#             attr_df.columns = attr_df.columns.str.lower()

#             ## Correcting column names for the dataframe to run Framework
#             attr_df.rename(columns={'brand_rt':'brand','category_rt':'c_category','market_share_total_sales_rt':'market_share_total_sales'},inplace=True)

#             attr_df['brand'].sort_values(ascending=True).value_counts()


#             ## Filtering data for single brand
#             dataframes = []
#             matched_cols=[]
#             idv_sel_cols_list=[]

#             results_all_model = pd.DataFrame()

#             actual_vs_predicted =pd.DataFrame()

#             error_brands=[]

#             results_df_corr = pd.DataFrame()
#             corr_all_br =pd.DataFrame()


#             for brand_name in attr_df['brand'].unique():
#                 brand_df = attr_df.loc[attr_df['brand']==brand_name]
#                 idv_sel_cols_list=[]
#                 idv_sel_cols_merge = pd.DataFrame()
#                 idv_sel_cols_merge_exclude = pd.DataFrame()
#                 matched_cols=[]
#                 chosen_idvs = []
#                 brand_prc_cls_df = pd.DataFrame()
#                 brand_req_df = pd.DataFrame()
#                 idvs = pd.DataFrame()
#                 idvs_scaled = pd.DataFrame()
#                 results_all_model_RF = pd.DataFrame()
#                 results_all_model_XGB = pd.DataFrame()
#                 results_all_model_RF_feat = pd.DataFrame()
#                 results_all_model_corr = pd.DataFrame()
#                 results_all_model_brute = pd.DataFrame()
#                 for category_name in brand_df['c_category'].unique():
#                     if category_name != 'CORPORATE':
#                         try:
#                             brand_prc_cls_df = brand_df.loc[brand_df['c_category']==category_name]
#                             brand_req_df=brand_prc_cls_df.copy()
#                             if brand_req_df[model_config["weights_model"]["DV"]].sum() <= 0:
#                                 error_brands.append(brand_name+"-"+category_name)
#                                 continue
#                             print("Brand: ", brand_name)
#                             print("Category: ", category_name)

#                             ## Drop the unwanted columns for modeling
#                             columns_log_list = list(brand_req_df.columns)
#                             columns_log_list.remove(model_config["weights_model"]["DV"])
#                             columns_log_list.remove('c_category')
#                             columns_log_list.remove('brand')
#                             columns_log_list.remove('date')

#                             columns_log_df = pd.DataFrame(columns_log_list,columns=["Column_names"])
#                             columns_log_df['Brand'] = brand_name
#                             columns_log_df['Category'] = category_name
#                             columns_log_df['Reasons_to_drop'] = np.nan

#                             idv_sel_cols1 = idv_sel_cols[idv_sel_cols['product_category_idv'] == category_name]

#                             idv_sel_cols1  = idv_sel_cols1[idv_sel_cols1 ['Select'] == 'Y']

#                             idv_sel_cols_list = list(idv_sel_cols1['idv_for_model_corrected'].unique())
#                             if model_config["weights_model"]["Cols_force_sel"] == True:
#                                 idv_sel_cols_list = model_config["weights_model"]["Force_cols_net"]


#                             # Replacing spaces with underscore and removing ' - ' in column names
#                             idv_sel_cols_list = [x.replace(' - ','_') for x in idv_sel_cols_list]
#                             idv_sel_cols_list = [x.replace(' ','_') for x in idv_sel_cols_list]
#                             # removing case sensitivity
#                             idv_sel_cols_list = [x.lower() for x in idv_sel_cols_list]

#                             for col1 in idv_sel_cols_list:
#                                 if col1 in list(brand_req_df.columns):
#                                     matched_cols.append(col1)


#                             idv_sel_cols_list = list(set(matched_cols))

#                             if model_config["weights_model"]["drop_mean_attributes"]:
#                                 idv_sel_cols_list =  [i for i in idv_sel_cols_list if not ('_mean' in i )]
#                                 for col in list(columns_log_df.loc[pd.isnull(columns_log_df['Reasons_to_drop'])==True,'Column_names']):
#                                     if col not in idv_sel_cols_list:
#                                         columns_log_df.loc[columns_log_df['Column_names']==col,'Reasons_to_drop'] = "Dropped for mean column"

#                             if model_config["weights_model"]["drop_rank1st_attributes"]:
#                                 idv_sel_cols_list = [i for i in idv_sel_cols_list if not ('_rank_1st' in i )]
#                                 for col in list(columns_log_df.loc[pd.isnull(columns_log_df['Reasons_to_drop'])==True,'Column_names']):
#                                     if col not in idv_sel_cols_list:
#                                         columns_log_df.loc[columns_log_df['Column_names']==col,'Reasons_to_drop'] = "Dropped for Rank column"

#                             #actual_df = brand_req_df.copy()
#                             brand_req_df1 = brand_req_df[idv_sel_cols_list+[model_config["weights_model"]["DV"]]+[model_config["weights_model"]["Temp"]]]
#                             brand_req_df1
#                             filtered_df=pd.DataFrame()
#                             null_columns =[]
#                             filtered_df = brand_req_df1[brand_req_df1['new_brand'] == 'Stacked Brand']
#                             # Get column names with all NaN values
#                             null_columns = filtered_df.columns[filtered_df.isnull().all()]

#                             # Remove the completely null columns
#                             df = filtered_df.drop(columns=null_columns)
#                             column_names = df.columns.tolist()
#                             # Convert the lists to sets
#                             set1 = set(idv_sel_cols_list )
#                             set2 = set(column_names)

#                             common_elements = set1.intersection(set2)
#                             #final_idvs
#                             final_idvs = list(common_elements)
#                             final_idvs
#                             brand_req_df_2 = brand_req_df[final_idvs+[model_config["weights_model"]["DV"]]+[model_config["weights_model"]["Temp"]]]
#                             brand_req_df_2
#                             brand_req_df = brand_req_df_2[brand_req_df_2['new_brand'] == brand_name]
#                             brand_req_df = brand_req_df[final_idvs+[model_config["weights_model"]["DV"]]]
#                             brand_req_df

#                             brand_req_df = brand_req_df.reset_index(drop=True)
#                             length =len(brand_req_df)
#                             print("length",length)
#                             if len(brand_req_df) >20:

#                                 ## Removing columns that have more than 50% missing values

#                                 df_null = pd.DataFrame(brand_req_df.isnull().sum(),columns=(['Null count']))
#                                 df_null = df_null.sort_values(by='Null count',ascending=False)

#                                 df_null[df_null['Null count'] >= brand_req_df.shape[0]*0.5]
#                                 brand_req_df.drop(columns=list(df_null[df_null['Null count'] >= brand_req_df.shape[0]*0.5].index),inplace=True)


#                                 for col in list(columns_log_df.loc[pd.isnull(columns_log_df['Reasons_to_drop'])==True,'Column_names']):
#                                     if col not in list(brand_req_df.columns):
#                                         columns_log_df.loc[columns_log_df['Column_names']==col,'Reasons_to_drop'] = "Dropped for > 50 % null"


#                                 print("len(set(brand_req_df.columns) - {model_config['weights_model']['DV']})",len(set(brand_req_df.columns) -{model_config["weights_model"]["DV"]}))

#                                 if len(set(brand_req_df.columns) - {model_config["weights_model"]["DV"]}) > 0:
#                                     print(len(set(brand_req_df.columns) - {model_config["weights_model"]["DV"]}))
#                                     ## Keeping out first 3 records (Null values created by lag operation)
#                                     if model_config["weights_model"]["is_lag_considered"] == True:
#                                         brand_req_df = brand_req_df.iloc[3:,:].reset_index(drop=True)

#                                     brand_req_df.isnull().sum().sort_values(ascending=False)

#                                     ## Replacing null values with average column value -- seems incorrect

#                                     brand_req_df =brand_req_df.fillna(brand_req_df.mean())

#                                     brand_req_df.isnull().sum().sort_values(ascending=False)

#                                     # https://timeseriesreasoning.com/contents/akaike-information-criterion/
#                                     # Regressing lags with base feature to pick only best lag feature based on AIC
#                                     # chosen_idvs = choose_best_lag_feature1(brand_req_df.drop(DV, 1), idv_sel_cols)

#                                     # Selecting best correlated feature with DV (including base features)
#                                     if model_config["weights_model"]["is_lag_considered"] == True:
#                                         chosen_idvs = choose_best_lag_feature(brand_req_df, idv_sel_cols, model_config["weights_model"]["DV"])
#                                         for col in list(columns_log_df.loc[pd.isnull(columns_log_df['Reasons_to_drop'])==True,'Column_names']):
#                                             if col not in list(chosen_idvs):
#                                                 columns_log_df.loc[columns_log_df['Column_names']==col,'Reasons_to_drop'] = "Dropped for best lag"


#                                     else:
#                                         chosen_idvs = list(brand_req_df.columns)
#                                         chosen_idvs.remove(model_config["weights_model"]["DV"]) ## if only ACV is needed and drop price in the idv list
#                                     print("chosen_idvs : ",list(chosen_idvs))
#                                     ##############################################
#                                     # 6. Creating modeling data:
#                                     ##############################################

#                                     # modeldf = brand_req_df[features+["Price_Class_Market_Share"]]

#                                     modeldf = brand_req_df
#                                     model_DV = model_config["weights_model"]["DV"]
#                                     modeldf.shape

#                                     # Creating modeling data and target

#                                     idvs = modeldf.drop(model_DV, 1)       # feature matrix
#                                     dv = modeldf[model_DV]

#                                     # Taking log if Base EQ is target
#                                     if model_config["weights_model"]["log_convert_DV"]:
#                                         dv = np.log1p(modeldf[model_DV])


#                                     if model_config["weights_model"]["standardize"]:
#                                         mmscaler = MinMaxScaler() # Do even before feature selection
#                                         idvs_scaled = pd.DataFrame(mmscaler.fit_transform(idvs), columns = idvs.columns)
#                                         idvs = idvs_scaled.copy()
#                 #                         idvs = pd.DataFrame(mmscaler.fit_transform(idvs), columns = idvs.columns)

#                                     feat_at_each_iter = []
#                                     per_at_each_iter = []
#                                     coeff_at_each_iter = []
#                                     subset_at_each_iter = pd.DataFrame()

#                                     ##############################################
#                                     # 5.2 Feature selection
#                                     # Run forward seq feature selection using mlxtend SFS library - Ridge estimator and custom scoring after taking exponential
#                                     # Run for 15 different seeds - features that come consistent across the seeds are selected
#                                     ##############################################
#                                     print("Length of Chosen IDVS:",len(chosen_idvs))

#                                     ## Creating a list of final columns for modeling
#                                     cols_to_select = sorted(list(set(chosen_idvs)))

#                                     if len(cols_to_select) >= 2:
#                                         for i in [2]:
#                                             # i = 2
#                                             print(i)
#                                             if model_config["weights_model"]["Time_series_split"] == True:
#                                                 train_x = idvs[cols_to_select].iloc[:42,:]
#                                                 train_y = dv[:42]
#                                                 test_x = idvs[cols_to_select].iloc[42:,:]
#                                                 test_y = dv[42:]
#                                             if model_config["weights_model"]["Random_seed_split"] == True:
#                                                 train_x, test_x, train_y, test_y = train_test_split(idvs[cols_to_select],dv,test_size=6, random_state=i, shuffle=True)
#                                             train_x_all =  idvs[cols_to_select]
#                                             train_y_all = dv


#                                             ## Checking the counter Intuitive
#                                             ## Code for restricting the direction of coefficient of features as per business constraints


#                                             if model_config["weights_model"]["P_N_check"] == True:
#                                                 corr_df_pn = pd.DataFrame()
#                                                 corr_df_pn = pd.DataFrame(train_x.corrwith(train_y))
#                                                 corr_df_pn['Brand'] = brand_name                               #Comment if col not needed
#                                                 corr_df_pn['category'] = category_name                            #Comment if col not needed

#                                                 corr_df_pn['idv_for_model_corrected'] = corr_df_pn.index
#                                                 corr_df_pn.reset_index(drop = True, inplace = True)
#                                                 corr_df_pn=corr_df_pn.rename(columns = {0:'Correlation'})
#                                                 corr_df_pn['corr_abs'] = abs(corr_df_pn['Correlation'])   #Comment if col not needed
#                                                 corr_df_pn.sort_values(by=['corr_abs'],ascending=False,inplace=True) #Comment if col not needed
#                                                 corr_df_pn.reset_index(drop = True, inplace = True)               # Comment if col not needed

#                                                 results_df_corr = pd.concat([results_df_corr,corr_df_pn],axis=0) # Comment if col not needed

#                                                 cols_to_sel_df = pd.DataFrame({"idv_for_model_corrected":cols_to_select},index=None)
#                                                 M1_df = pd.merge(idv_sel_cols_merge_exclude,cols_to_sel_df,on=['idv_for_model_corrected'],how='inner')
#                                                 idv_sel_P_N = pd.merge(M1_df,corr_df_pn,on=['idv_for_model_corrected'],how='inner')

#                                                 idv_sel_P_N_Match = idv_sel_P_N[~((idv_sel_P_N['Correlation'] < -model_config["weights_model"]["Counter_Intuitive_cut_off"]) & (idv_sel_P_N['Intutive']=='P')) |((idv_sel_P_N['Correlation'] > model_config["weights_model"]["Counter_Intuitive_cut_off"]) & (idv_sel_P_N['Intutive']=='N'))].reset_index(drop=True)

#                                                 Count_intuitive_cols_filtered = sorted(list(set(idv_sel_P_N_Match['idv_for_model_corrected'])))
#                                                 train_x = train_x[Count_intuitive_cols_filtered]
#                                                 test_x = test_x[Count_intuitive_cols_filtered]
#                                                 train_x_all =  idvs[Count_intuitive_cols_filtered]

#                                             if model_config["weights_model"]["Corr_file_generation"] == True :
#                                                 if model_config["weights_model"]["P_N_check"] == True:
#                                                     train_x_corr =  idvs[Count_intuitive_cols_filtered]
#                                                 else:
#                                                     train_x_corr =  idvs[cols_to_select]
#                                                 train_y_corr = dv
#                                                 corr_raw_df = pd.DataFrame(train_x_corr.corrwith(train_y_corr))
#                                                 corr_raw_df['Brand'] = brand_name                               #Comment if col not needed
#                                                 corr_raw_df['category'] = category_name
#                                                 corr_raw_df['idvs_for_model'] = corr_raw_df.index
#                                                 corr_raw_df.reset_index(drop = True, inplace = True)
#                                                 corr_raw_df=corr_raw_df.rename(columns = {0:'Correlation with DV'})
#                                                 corr_raw_df['corr_abs'] = corr_raw_df['Correlation with DV'].abs()   #Comment if col not needed
#                                                 corr_raw_df.sort_values(by=['corr_abs'],ascending=False,inplace=True) #Comment if col not needed
#                                                 corr_raw_df.reset_index(drop = True, inplace = True)

#                                             if model_config["weights_model"]["PCA_Transform"] == True:
#                                                 pca = PCA(n_components = 0.95)
#                                                 PCA_train = pca.fit_transform(train_x)
#                                                 PCA_comp = []
#                                                 for i in range(0,pca.n_components_,1):
#                                                     PCA_comp.append("PCA"+str(i))
#                                                 train_x = pd.DataFrame(PCA_train,columns=PCA_comp,index=None)
#                                                 PCA_test = pca.transform(test_x)
#                                                 test_x = pd.DataFrame(PCA_test,columns=PCA_comp,index=None)
#                                                 PCA_train_all = pca.transform(train_x_all)
#                                                 train_x_all = pd.DataFrame(PCA_train_all,columns=PCA_comp,index=None)


#                                             print("Chosen_lags:",chosen_idvs)

#                                             X_test_hold = test_x.copy()
#                                             y_test_hold = test_y.copy()


#                                             X_train = train_x.copy()
#                                             y_train = train_y.copy()

#                                             feat_importance = pd.DataFrame()
#                                             feat_df=pd.DataFrame()
#                                             if (run_config["weights_models"]["RandomForest"]["run"] == True):
#                                                 param_grid_rf = {"max_depth" : model_config["weights_model"]["hyperparameters"]["RandomForest"]["grid_search"]["max_depth"],"n_estimators" : model_config["weights_model"]["hyperparameters"]["RandomForest"]["grid_search"]["n_estimators"], "max_features" : model_config["weights_model"]["hyperparameters"]["RandomForest"]["grid_search"]['max_features'], "random_state" : model_config["weights_model"]["hyperparameters"]["RandomForest"]["grid_search"]["random_state"]}

#                                                 rf1=RandomForestRegressor(random_state=model_config["weights_model"]["hyperparameters"]["RandomForest"]["random_state"])
#                                                 search_rf = GridSearchCV(rf1, param_grid_rf,cv=model_config["weights_model"]["cross_validation_number"],scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train, y_train)
#                                                 rf=RandomForestRegressor(n_estimators  = search_rf.best_params_["n_estimators"],max_depth = search_rf.best_params_["max_depth"],random_state=model_config["weights_model"]["hyperparameters"]["RandomForest"]["random_state"])
#                                                 rf.fit(X_train,y_train)
#                                                 features = list(X_train.columns)
#                                                 f_i = list(zip(features,rf.feature_importances_))
#                                                 f_i.sort(key = lambda x : x[1],reverse=True)

#                                                 rfe = RFECV(rf,cv=model_config["weights_model"]["cross_validation_number"],scoring='neg_mean_absolute_percentage_error')
#                                                 rfe.fit(X_train,y_train)
#                                                 selected_features = list(np.array(features)[rfe.get_support()])
#                                                 print(selected_features)
#                                                 feat_importance = pd.DataFrame(f_i,columns=['Features','Feature Importance'])
#                                                 feat_importance.set_index('Features',inplace=True)
#                                                 feat_importance = feat_importance.iloc[:20,:]
#                                                 print(feat_importance)
#                                                 best_features = list(feat_importance.index)

#                                                 explainer = shap.TreeExplainer(rf)
#                                                 shap_values = explainer.shap_values(X_train)
#                                                 feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train.columns)).mean(),columns=['shap values'])
#                                                 print("Random Forest shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                                 feat_df = feat_df.sort_values(by='shap values',ascending=False)


#                                                 y_pred_test = rf.predict(X_test_hold)
#                                                 y_pred_train = rf.predict(X_train)
#                                                 y_pred_all = rf.predict(train_x_all[list(X_train.columns)])

#                                                 feat_importance = feat_importance.reset_index().rename(columns={'Feature Importance':'Feature Importance/coefficient'})
#                                                 feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                                 mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                                 mse_train = metrics.mean_squared_error(y_train, y_pred_train)

#                                                 rmse_train = np.sqrt(mse_train) #mse**(0.5)
#                                                 r2_train = metrics.r2_score(y_train, y_pred_train)
#                                                 mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                                 feat_df

#                                                 results_all_model_RF = pd.concat([feat_importance,feat_df], axis=1)
#                                                 results_all_model_RF['Model'] = "Random Forest"
#                                                 results_all_model_RF['Brand'] = brand_name
#                                                 results_all_model_RF['Category'] = category_name
#                                                 results_all_model_RF['pillar'] = pillar_name

#                                                 results_all_model_RF['Latest DV'] = dv.values[-1]

#                                                 results_all_model_RF['R2_Score_Train'] = r2_train
#                                                 results_all_model_RF['MAPE_Train'] = mape_train
#                                                 results_all_model_RF['R2_score_fold'] = cross_val_score(rf,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='r2').mean()
#                                                 results_all_model_RF['MAPE_fold'] = cross_val_score(rf,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                                 results_all_model_RF['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                                 results_all_model_RF['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                                 results_all_model_RF['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                                 results_all_model_RF['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)


#                                                 results_all_model_RF['Trial name'] = temp_title
#                                                 results_all_model_RF['Best_Params_Gridsearchcv']=str(search_rf.best_params_)

#                                                 results_all_model = pd.concat([results_all_model, results_all_model_RF],axis=0)

#                                                 actual_vs_predicted_RF = pd.DataFrame()

#                                                 actual_vs_predicted_RF['Actual'] = train_y_all
#                                                 actual_vs_predicted_RF['Predicted'] = y_pred_all
#                                                 actual_vs_predicted_RF['Brand'] = brand_name
#                                                 actual_vs_predicted_RF['Category'] = category_name
#                                                 actual_vs_predicted_RF['pillar'] = pillar_name
#                                                 actual_vs_predicted_RF['Model'] = "Random Forest"
#                                                 actual_vs_predicted_RF['Trial name'] =temp_title
#                                                 actual_vs_predicted = pd.concat([actual_vs_predicted,actual_vs_predicted_RF],axis=0)

#                                                 title=(brand_name+"-"+category_name+"-"+"Random_Forest"+"_"+temp_title)
#                                                 print(os.listdir())

#                                             feat_importance = pd.DataFrame()
#                                             feat_df=pd.DataFrame()
#                                             if run_config["weights_models"]["XGBoost"]["run"] == True:
#                                                 param_grid = {"max_depth":model_config["weights_model"]["hyperparameters"]["XGBoost"]["grid_search"]["max_depth"],"n_estimators": model_config["weights_model"]["hyperparameters"]["XGBoost"]["grid_search"]["n_estimators"],"learning_rate": model_config["weights_model"]["hyperparameters"]["XGBoost"]["grid_search"]["learning_rate"],"random_state":model_config["weights_model"]["hyperparameters"]["XGBoost"]["grid_search"]["random_state"]}
#                                                 regressor=xgb.XGBRegressor(eval_metric='mape',random_state=model_config["weights_model"]["hyperparameters"]["XGBoost"]["random_state"])
#                                                 search_xgb = GridSearchCV(regressor, param_grid,cv=model_config["weights_model"]["cross_validation_number"],scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train, y_train)
#                                                 print("The best hyperparameters are ",search_xgb.best_params_)

#                                                 regressor=xgb.XGBRegressor(learning_rate = search_xgb.best_params_["learning_rate"],
#                                                             n_estimators  = search_xgb.best_params_["n_estimators"],
#                                                             max_depth     = search_xgb.best_params_["max_depth"],
#                                                             eval_metric='mape',random_state=model_config["weights_model"]["hyperparameters"]["XGBoost"]["random_state"])

#                                                 regressor.fit(X_train, y_train)
#                                                 dict_result = regressor.get_booster().get_score(importance_type='gain')
#                                                 print("Feature importance XGBoost",pd.DataFrame(dict_result.items(),columns=['Feature','gain']).sort_values(by ='gain',ascending=False))
#                                                 feat_importance = pd.DataFrame(dict_result.items(),columns=['Feature','gain']).sort_values(by ='gain',ascending=False)
#                                                 explainer = shap.TreeExplainer(regressor)
#                                                 shap_values = explainer.shap_values(X_train)
#                                                 feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train.columns)).mean(),columns=['shap values'])
#                                                 print("XGBoost shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                                 feat_df = feat_df.sort_values(by='shap values',ascending=False)

#                                                 y_pred_test = regressor.predict(X_test_hold)
#                                                 y_pred_train = regressor.predict(X_train)
#                                                 y_pred_all = regressor.predict(train_x_all[list(X_train.columns)])

#                                                 feat_importance = feat_importance.reset_index().rename(columns={'Feature':'Features','gain':'Feature Importance/coefficient'})
#                                                 feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                                 mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                                 mse_train = metrics.mean_squared_error(y_train, y_pred_train)
#                                                 rmse_train = np.sqrt(mse_train) #mse**(0.5)
#                                                 r2_train = metrics.r2_score(y_train, y_pred_train)
#                                                 mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                                 results_all_model_XGB = pd.concat([feat_importance,feat_df], axis=1)
#                                                 results_all_model_XGB['Model'] = "XGBoost"
#                                                 results_all_model_XGB['Brand'] = brand_name
#                                                 results_all_model_XGB['Category'] = category_name
#                                                 results_all_model_XGB['Latest DV'] = dv.values[-1]

#                                                 results_all_model_XGB['R2_Score_Train'] = r2_train
#                                                 results_all_model_XGB['MAPE_Train'] = mape_train
#                                                 results_all_model_XGB['R2_score_fold'] = cross_val_score(regressor,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='r2').mean()
#                                                 results_all_model_XGB['MAPE_fold'] = cross_val_score(regressor,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                                 results_all_model_XGB['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                                 results_all_model_XGB['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                                 results_all_model_XGB['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                                 results_all_model_XGB['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)


#                                                 results_all_model_XGB['Trial name'] = temp_title
#                                                 results_all_model_XGB['Best_Params_Gridsearchcv']=str(search_xgb.best_params_)
#                                                 results_all_model = pd.concat([results_all_model, results_all_model_XGB],axis=0)

#                                                 actual_vs_predicted_XGB = pd.DataFrame()

#                                                 actual_vs_predicted_XGB['Actual'] = train_y_all
#                                                 actual_vs_predicted_XGB['Predicted'] = y_pred_all
#                                                 actual_vs_predicted_XGB['Brand'] = brand_name
#                                                 actual_vs_predicted_XGB['Category'] = category_name
#                                                 actual_vs_predicted_XGB['Model'] = "XGBoost"
#                                                 actual_vs_predicted_XGB['Trial name'] =temp_title
#                                                 actual_vs_predicted = pd.concat([actual_vs_predicted,actual_vs_predicted_XGB],axis=0)

#                                                 title=(brand_name+"-"+category_name+"-"+"XGBOOST"+"_"+temp_title)

#                                             feat_importance = pd.DataFrame()
#                                             feat_df=pd.DataFrame()
#                                             if run_config["weights_models"]["RF_Ridge"]["run"] ==True:

#                                                     rf.fit(X_train,y_train)
#                                                     features = list(X_train.columns)
#                                                     f_i = list(zip(features,rf.feature_importances_))
#                                                     f_i.sort(key = lambda x : x[1],reverse=True)

#                                                     rfe = RFECV(rf,cv=model_config["weights_model"]["cross_validation_number"],scoring="neg_mean_absolute_percentage_error")
#                                                     rfe.fit(X_train,y_train)
#                                                     selected_features = list(np.array(features)[rfe.get_support()])
#                                                     print(selected_features)
#                                                     feat_importance = pd.DataFrame(f_i,columns=['Features','Feature Importance'])
#                                                     feat_importance.set_index('Features',inplace=True)
#                                                     feat_importance = feat_importance.iloc[:20,:]
#                                                     print(feat_importance)
#                                                     best_features = list(feat_importance.index)
#                                                     print(best_features)

#                                                     n_feat = len(best_features)

#                                                     param_rf_ridge = {'alpha':model_config["weights_model"]["hyperparameters"]["RF_Ridge"]["grid_search"]["alpha"],"random_state":model_config["weights_model"]["hyperparameters"]["RF_Ridge"]["grid_search"]["random_state"]}
#                                                     ridg1=Ridge(positive=model_config["weights_model"]["hyperparameters"]["RF_Ridge"]["positive"],random_state=model_config["weights_model"]["hyperparameters"]["RF_Ridge"]["random_state"])
#                                                     search_rf_ridge = GridSearchCV(ridg1,param_rf_ridge,cv=model_config["weights_model"]["cross_validation_number"],scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train[best_features], y_train)
#                                                     fit_model_RF_Ridge=Ridge(alpha=search_rf_ridge.best_params_["alpha"],random_state=42).fit(X_train[best_features], y_train)
#                                                     stat_df = pd.DataFrame()
#                                                     params = np.append(fit_model_RF_Ridge.intercept_,fit_model_RF_Ridge.coef_)
#                                                     stat_df["coefficients"] = list(params)
#                                                     features = ['intercept']+best_features
#                                                     stat_df.insert(0,"features", features)
#                                                     feat_import = stat_df.sort_values(by='coefficients',ascending=False)
#                                                     print(feat_import)

#                                                     explainer = shap.LinearExplainer(fit_model_RF_Ridge,X_train[best_features])
#                                                     shap_values = explainer.shap_values(X_train[best_features])
#                                                     feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train[best_features].columns)).mean(),columns=['shap values'])
#                                                     print("RF for ridfe shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                                     feat_df = feat_df.sort_values(by='shap values',ascending=False)

#                                                     y_pred_train = fit_model_RF_Ridge.predict(X_train[best_features])
#                                                     y_pred_test = fit_model_RF_Ridge.predict(X_test_hold[best_features])
#                                                     y_pred_all = fit_model_RF_Ridge.predict(train_x_all[best_features])

#                                                     feat_importance = feat_importance.reset_index().rename(columns={'Feature Importance':'Feature Importance/coefficient'})
#                                                     feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                                     mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                                     mse_train = metrics.mean_squared_error(y_train, y_pred_train)
#                                                     rmse_train = np.sqrt(mse_train) #mse**(0.5)
#                                                     r2_train = metrics.r2_score(y_train, y_pred_train)
#                                                     mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                                     results_all_model_RF_feat = pd.concat([feat_importance,feat_df], axis=1)
#                                                     results_all_model_RF_feat['Model'] = "Random Forest for Feature Selection"
#                                                     results_all_model_RF_feat['Brand'] = brand_name
#                                                     results_all_model_RF_feat['Category'] = category_name
#                                                     results_all_model_RF_feat['Latest DV'] = dv.values[-1]

#                                                     results_all_model_RF_feat['R2_Score_Train'] = r2_train
#                                                     results_all_model_RF_feat['MAPE_Train'] = mape_train
#                                                     results_all_model_RF_feat['R2_score_fold'] = cross_val_score(fit_model_RF_Ridge,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='r2').mean()
#                                                     results_all_model_RF_feat['MAPE_fold'] = cross_val_score(fit_model_RF_Ridge,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                                     results_all_model_RF_feat['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                                     results_all_model_RF_feat['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                                     results_all_model_RF_feat['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                                     results_all_model_RF_feat['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)

#                                                     results_all_model_RF_feat['Trial name'] = temp_title
#                                                     results_all_model_RF_feat['Best_Params_Gridsearchcv']=str(search_rf_ridge.best_params_)
#                                                     results_all_model = pd.concat([results_all_model, results_all_model_RF_feat],axis=0)

#                                                     actual_vs_predicted_RF_feat = pd.DataFrame()

#                                                     actual_vs_predicted_RF_feat['Actual'] = train_y_all
#                                                     actual_vs_predicted_RF_feat['Predicted'] = y_pred_all
#                                                     actual_vs_predicted_RF_feat['Brand'] = brand_name
#                                                     actual_vs_predicted_RF_feat['Category'] = category_name
#                                                     actual_vs_predicted_RF_feat['Model'] = "Random Forest for Feature Selection"
#                                                     actual_vs_predicted_RF_feat['Trial name'] =temp_title
#                                                     actual_vs_predicted = pd.concat([actual_vs_predicted,actual_vs_predicted_RF_feat],axis=0)

#                                                     title=(brand_name+"-"+category_name+"-"+"RF_Feature"+"_"+temp_title)

#                                             feat_importance = pd.DataFrame()
#                                             feat_df=pd.DataFrame()
#                                             try:
#                                                 if run_config["weights_models"]["Corr_ridge"]["run"] == True:
#                                                         corr_df = pd.DataFrame()
#                                                         corr_df_sub = pd.DataFrame(X_train.corrwith(y_train))

#                                                         corr_df_sub['Metric1'] = corr_df_sub.index
#                                                         corr_df_sub.reset_index(drop = True, inplace = True)
#                                                         corr_df_sub=corr_df_sub.rename(columns = {0:'Correlation'})
#                                                         print("corr_df")

#                                                         corr_df_sub['corr_abs'] = abs(corr_df_sub['Correlation'])
#                                                         corr_df_sub.sort_values(by=['corr_abs'],ascending=False,inplace=True)
#                                                         corr_df_sub.reset_index(drop = True, inplace = True)
#                                                         corr_df_sub = corr_df_sub[corr_df_sub['corr_abs'] > 0.4]
#                                                         print(corr_df_sub)

#                                                         best_features = list(corr_df_sub['Metric1'])

#                                                         n_feat = len(best_features)

#                                                         param_corr_ridge = {'alpha':model_config["weights_model"]["hyperparameters"]["Corr_ridge"]["grid_search"]["alpha"],"random_state":model_config["weights_model"]["hyperparameters"]["Corr_ridge"]["grid_search"]["random_state"]}
#                                                         ridg2=Ridge(positive=model_config["weights_model"]["hyperparameters"]["Corr_ridge"]["positive"],random_state=model_config["weights_model"]["hyperparameters"]["Corr_ridge"]["random_state"])
#                                                         search_corr = GridSearchCV(ridg2,param_corr_ridge,cv=model_config["weights_model"]["cross_validation_number"],scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train[best_features], y_train)
#                                                         fit_model_corr=Ridge(alpha=search_corr.best_params_["alpha"],random_state=model_config["weights_model"]["hyperparameters"]["Corr_ridge"]["random_state"]).fit(X_train[best_features], y_train)

#                                                         stat_df = pd.DataFrame()
#                                                         params = np.append(fit_model_corr.intercept_,fit_model_corr.coef_)
#                                                         stat_df["coefficients"] = list(params)
#                                                         features = ['intercept']+best_features
#                                                         stat_df.insert(0,"features", features)
#                                                         feat_import = stat_df.sort_values(by='coefficients',ascending=False)
#                                                         print(feat_import)

#                                                         explainer = shap.LinearExplainer(fit_model_corr,X_train[best_features])
#                                                         shap_values = explainer.shap_values(X_train[best_features])
#                                                         feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train[best_features].columns)).mean(),columns=['shap values'])
#                                                         print("corr for shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                                         feat_df = feat_df.sort_values(by='shap values',ascending=False)

#                                                         y_pred_train = fit_model_corr.predict(X_train[best_features])
#                                                         y_pred_test = fit_model_corr.predict(X_test_hold[best_features])
#                                                         y_pred_all = fit_model_corr.predict(train_x_all[best_features])

#                                                         feat_importance = feat_import.reset_index().rename(columns={'features':'Features','coefficients':'Feature Importance/coefficient'})
#                                                         feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})

#                                                         mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                                         mse_train = metrics.mean_squared_error(y_train, y_pred_train)
#                                                         rmse_train = np.sqrt(mse_train)
#                                                         r2_train = metrics.r2_score(y_train, y_pred_train)
#                                                         mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                                         results_all_model_corr = pd.concat([feat_importance,feat_df], axis=1)
#                                                         results_all_model_corr['Model'] = "Correlation for feature"
#                                                         results_all_model_corr['Brand'] = brand_name
#                                                         results_all_model_corr['Category'] = category_name
#                                                         results_all_model_corr['Latest DV'] = dv.values[-1]

#                                                         results_all_model_corr['R2_Score_Train'] = r2_train
#                                                         results_all_model_corr['MAPE_Train'] = mape_train
#                                                         results_all_model_corr['R2_score_fold'] = cross_val_score(fit_model_corr,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='r2').mean()
#                                                         results_all_model_corr['MAPE_fold'] = cross_val_score(fit_model_corr,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                                         results_all_model_corr['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                                         results_all_model_corr['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                                         results_all_model_corr['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                                         results_all_model_corr['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)

#                                                         results_all_model_corr['Trial name'] = temp_title
#                                                         results_all_model_corr['Best_Params_Gridsearchcv']=str(search_corr.best_params_)
#                                                         results_all_model = pd.concat([results_all_model, results_all_model_corr],axis=0)

#                                                         actual_vs_predicted_corr = pd.DataFrame()

#                                                         actual_vs_predicted_corr['Actual'] = train_y_all
#                                                         actual_vs_predicted_corr['Predicted'] = y_pred_all
#                                                         actual_vs_predicted_corr['Brand'] = brand_name
#                                                         actual_vs_predicted_corr['Category'] = category_name
#                                                         actual_vs_predicted_corr['Model'] = "Correlation for feature"
#                                                         actual_vs_predicted_corr['Trial name'] =temp_title
#                                                         actual_vs_predicted = pd.concat([actual_vs_predicted,actual_vs_predicted_corr],axis=0)

#                                                         title=(brand_name+"-"+category_name+"-"+"Correlation"+"_"+temp_title)

#                                             except Exception as e1:
#                                                 print(e1)
#                                             feat_importance = pd.DataFrame()
#                                             feat_df=pd.DataFrame()
#                                             if run_config["weights_models"]["Brute_force"]["run"] == True:
#                                                     param_rf_ridge = {'alpha':model_config["weights_model"]["hyperparameters"]["Brute_force"]["grid_search"]["alpha"],"random_state":model_config["weights_model"]["hyperparameters"]["Brute_force"]["grid_search"]["random_state"]}
#                                                     ridg3=Ridge(positive=model_config["weights_model"]["hyperparameters"]["Brute_force"]["positive"],random_state=model_config["weights_model"]["hyperparameters"]["Brute_force"]["random_state"])
#                                                     search_brute = GridSearchCV(ridg3,param_rf_ridge,cv=model_config["weights_model"]["cross_validation_number"],scoring=['r2','neg_mean_absolute_percentage_error'],refit='neg_mean_absolute_percentage_error').fit(X_train, y_train)
#                                                     fit_model_brute=Ridge(alpha=search_brute.best_params_["alpha"],random_state=model_config["weights_model"]["hyperparameters"]["Brute_force"]["random_state"]).fit(X_train, y_train)


#                                                     stat_df = pd.DataFrame()
#                                                     params = np.append(fit_model_brute.intercept_,fit_model_brute.coef_)
#                                                     stat_df["coefficients"] = list(params)
#                                                     features = ['intercept']+list(X_train.columns)
#                                                     stat_df.insert(0,"features", features)
#                                                     feat_import = stat_df.sort_values(by='coefficients',ascending=False)
#                                                     print(feat_import)

#                                                     explainer = shap.LinearExplainer(fit_model_brute,X_train)
#                                                     shap_values = explainer.shap_values(X_train)
#                                                     feat_df = pd.DataFrame(np.abs(pd.DataFrame(shap_values,columns=X_train.columns)).mean(),columns=['shap values'])
#                                                     print("Brute force shap importance",feat_df.sort_values(by='shap values',ascending=False))
#                                                     feat_df = feat_df.sort_values(by='shap values',ascending=False)

#                                                     y_pred_train = fit_model_brute.predict(X_train)
#                                                     y_pred_test = fit_model_brute.predict(X_test_hold)
#                                                     y_pred_all = fit_model_brute.predict(train_x_all[list(X_train.columns)])


#                                                     feat_importance = feat_import.reset_index().rename(columns={'features':'Features','coefficients':'Feature Importance/coefficient'})
#                                                     feat_df = feat_df.reset_index().rename(columns={'index':'Shap Features'})


#                                                     mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
#                                                     mse_train = metrics.mean_squared_error(y_train, y_pred_train)
#                                                     rmse_train = np.sqrt(mse_train) #mse**(0.5)
#                                                     r2_train = metrics.r2_score(y_train, y_pred_train)
#                                                     mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

#                                                     results_all_model_brute = pd.concat([feat_importance,feat_df], axis=1)
#                                                     results_all_model_brute['Model'] = "Brute Force Model"
#                                                     results_all_model_brute['Brand'] = brand_name
#                                                     results_all_model_brute['Category'] = category_name
#                                                     results_all_model_brute['Latest DV'] = dv.values[-1]

#                                                     results_all_model_brute['R2_Score_Train'] = r2_train
#                                                     results_all_model_brute['MAPE_Train'] = mape_train
#                                                     results_all_model_brute['R2_score_fold'] = cross_val_score(fit_model_brute,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='r2').mean()
#                                                     results_all_model_brute['MAPE_fold'] = cross_val_score(fit_model_brute,X_train,y_train,cv=model_config["weights_model"]["cross_validation_number"],scoring='neg_mean_absolute_percentage_error').mean() * -1
#                                                     results_all_model_brute['R2_score_Hold_out'] = metrics.r2_score(y_test_hold, y_pred_test)
#                                                     results_all_model_brute['MAPE_Hold_out'] = mean_absolute_percentage_error(y_test_hold, y_pred_test)
#                                                     results_all_model_brute['R2_score_all'] = metrics.r2_score(train_y_all, y_pred_all)
#                                                     results_all_model_brute['MAPE_all'] = mean_absolute_percentage_error(train_y_all, y_pred_all)


#                                                     results_all_model_brute['Trial name'] = temp_title
#                                                     results_all_model_brute['Best_Params_Gridsearchcv']=str(search_brute.best_params_)
#                                                     results_all_model = pd.concat([results_all_model, results_all_model_brute],axis=0)

#                                                     actual_vs_predicted_brute=pd.DataFrame()

#                                                     actual_vs_predicted_brute['Actual'] = train_y_all
#                                                     actual_vs_predicted_brute['Predicted'] = y_pred_all
#                                                     actual_vs_predicted_brute['Brand'] = brand_name
#                                                     actual_vs_predicted_brute['Category'] = category_name
#                                                     actual_vs_predicted_brute['Model'] = "Brute Force Model"
#                                                     actual_vs_predicted_brute['Trial name'] =temp_title
#                                                     actual_vs_predicted = pd.concat([actual_vs_predicted,actual_vs_predicted_brute],axis=0)

#                                                     title=(brand_name+"-"+category_name+"-"+"Brute_Force"+"_"+temp_title)
#                         except Exception as e:
#                             print("Exception: ",e)
#                             _, _, exc_tb = sys.exc_info()
#                             # pdb.set_trace()
#                             error = [exc_tb.tb_lineno, os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]]
#                             brand_prc_cls_df = brand_df.loc[brand_df['c_category']==category_name]
#                             brand_req_df=brand_prc_cls_df
#                             logdf.loc[len(logdf)] = [brand_name, category_name, len(brand_prc_cls_df), 'N', e, error]
#                             pass


#             end = time.time()
#             Exec_time = end - start
#         #   print(Exec_time)
#             results_all_model = results_all_model.reset_index(drop=True)
#             all_pillar_results  = all_pillar_results.append(results_all_model, ignore_index=True)
#             all_pillar_results = all_pillar_results.drop_duplicates().reset_index(drop=True)

#             corr_all_br = corr_all_br.reset_index(drop=True)
#             corr_all_results  = corr_all_results.append(corr_all_br, ignore_index=True)
#             corr_all_results = corr_all_results.drop_duplicates().reset_index(drop=True)
#         return all_pillar_results, corr_all_results

#     if run_config["refresh_type"] == "full":
#         all_pillar_results, corr_all_results = metric_to_pillar_model_weights()

#     if run_config["refresh_type"] == "full":
#         all_pillar_results.to_csv(output_config["weights_model"]["model_results"],index=False, , storage_options = storage_options)

#         corr_all_results.to_csv(output_config["weights_model"]["corr_results"],index=False, , storage_options = storage_options)

# COMMAND ----------

# MAGIC %md
# MAGIC after cleaning(v3)

# COMMAND ----------
from library_installation import *


def cfa(
    output_config,
    refresh_config,
    feat_eng_config,
    eq_sub_scale_merged_brand,
    req_cols,
    storage_options,
    refresh_type,
):
    """CFA function for compute the estimated for each metrics corresponding to the dependent variable.
    Entire function composed of R programming apis in python using the library rpy2.
    Args:
        input_config (dict): Dictionary containing details of the path of input files
        refresh_config (dict): Dictionary containing details of refresh type
        feat_eng_config (dict): Dictionary containing details of hyperparameters for the models
        eq_sub_scale_merged_brand (DataFrame): Data used for CFA
        req_cols (List): Required column filtered from the idv_list
        storage_options (dict/None): Dictionary containing details of storage options for databricks else None
        refresh_type (str): refresh type for the data

    """

    # Activate automatic conversion
    pandas2ri.activate()

    # Convert to R DataFrame
    eq_sub_scale_merged_brand_r_df = pandas2ri.py2rpy(
        eq_sub_scale_merged_brand
    )

    # Set default CRAN mirror
    robjects.r('options(repos = c(CRAN="https://cran.rstudio.com/"))')

    robjects.r("library(data.table)")  # Load the data.table library

    req_cols_r_df = pandas2ri.py2rpy(req_cols)
    # Pass the config to R as individual values
    robjects.r.assign(
        "ALL_CATEGORY_PILLARS",
        robjects.StrVector(
            feat_eng_config["cfa"]["pillars"]["all_category_pillars"]
        ),
    )
    robjects.r.assign(
        "BY_CATEGORY_PILLARS",
        robjects.StrVector(
            feat_eng_config["cfa"]["pillars"]["by_category_pillars"]
        ),
    )
    robjects.r.assign("std_lv", feat_eng_config["cfa"]["std_lv"])
    robjects.r.assign(
        "check_gradient", feat_eng_config["cfa"]["check_gradient"]
    )
    robjects.r.assign("standardized_", feat_eng_config["cfa"]["standardized_"])
    robjects.r.assign("fit_measures", feat_eng_config["cfa"]["fit_measures"])
    robjects.r.assign("refresh_type", refresh_type)
    robjects.r.assign(
        "sample_seeds",
        robjects.IntVector(feat_eng_config["cfa"]["sample_seeds"]),
    )

    # Assign the R DataFrame to the R environment
    # robjects.r.assign('eq_sub_scale_merged_brand_r2', eq_sub_scale_merged_brand_r)
    robjects.globalenv["eq_sub_scale_merged_brand_r_df"] = (
        eq_sub_scale_merged_brand_r_df
    )
    robjects.globalenv["req_cols_r_df"] = req_cols_r_df
    robjects.r(
        "eq_sub_scale_merged_brand_r_dt <- as.data.table(eq_sub_scale_merged_brand_r_df)"
    )  # Convert to data.table
    robjects.r("req_cols_r_df <- as.data.frame(req_cols_r_df)")
    # robjects.r('setDT(eq_sub_scale_merged_brand_r_dt)  # Ensure it is a data.table')

    robjects.r(
        """
    print(unique(eq_sub_scale_merged_brand_r_dt$date))
    # Clean and convert to IDate
    eq_sub_scale_merged_brand_r_dt[, date := as.IDate(date, format = "%Y-%m-%d")]

    # Handle conversion errors (replace invalid dates with NA)
    eq_sub_scale_merged_brand_r_dt[, date := suppressWarnings(as.IDate(date, format = "%Y-%m-%d"))]

    eq_sub_scale_merged_brand_r_dt[, date := as.IDate(date)]
    excluded_cols <- c("Brand","New_Brand","Category","date")
    # excluded_cols <- c("brand","new_brand","category","date")
    for (col in names(eq_sub_scale_merged_brand_r_dt)) {
    if (!(col %in% excluded_cols)) {
        # Replace NaN with NA
        eq_sub_scale_merged_brand_r_dt[is.nan(get(col)), (col) := NA]

        # Convert to numeric if necessary
        eq_sub_scale_merged_brand_r_dt[, (col) := as.numeric(get(col))]
    }
    }
    for (col in names(req_cols_r_df)) {
    req_cols_r_df[[col]][is.nan(req_cols_r_df[[col]])] <- NA
    }
    """
    )

    # R cell with function
    robjects.r(
        """
    # List of required packages
    required_packages <- c("dplyr", "data.table", "semTools", "magrittr", "tidyr")

    # Check for missing packages
    missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

    # Install missing packages
    if (length(missing_packages) > 0) {
    install.packages(missing_packages, dependencies = TRUE)
    } else {
    message("All required packages are already installed.")
    }

    # Ensure devtools is installed
    if (!requireNamespace("devtools", quietly = TRUE)) {
        install.packages("devtools")
    }

    # Install specific version of lavaan
    devtools::install_version("lavaan", version = "0.6-17", repos = "http://cran.us.r-project.org")


    # Load libraries
    library(lavaan)        # For Confirmatory Factor Analysis (CFA)
    library(dplyr)         # For data manipulation (mutate, select, etc.)
    library(tidyr)
    library(data.table)    # For efficient data manipulation (copy, as.data.frame)
    library(semTools)      # For SEM diagnostics (e.g., standardized solution)
    library(magrittr)      # For piping operations (%>%)




    cfa = function(fa_str,scaled_data,s){

    fit_ <- sem(fa_str, data=scaled_data,std.lv=std_lv,check.gradient=check_gradient)
    summary(fit_, standardized=standardized_, fit.measures=fit_measures)

    # store fit measures - tli, cfi, rmsea
    cfa_fit_indices <- as.data.frame(fitMeasures(fit_))
    cfa_fit_indices$fit_measures <- row.names(cfa_fit_indices)
    row.names(cfa_fit_indices) <- NULL
    names(cfa_fit_indices) <- c("value","fitmeasure")
    cfa_fit_indices <- cfa_fit_indices[cfa_fit_indices$fitmeasure %in% c("cfi","tli","rmsea"),]
    cfa_fit_indices$value <- as.numeric(cfa_fit_indices$value)

    cfa_fit_indices_t <- as.data.frame(t(cfa_fit_indices))
    colnames(cfa_fit_indices_t) <- cfa_fit_indices$fitmeasure
    cfa_fit_indices_t <- cfa_fit_indices_t["value",]

    # Extract parameter estimates
    cfa_estimates <- as.data.frame(standardizedsolution(fit_))

    # cfa summary table
    cfa_summary <- cbind(cfa_estimates, cfa_fit_indices_t)

    # Store the results
    cfa_summary$factor_str = fa_str

    cfa_summary$Seed = s

    # sample_data_ = rbind(sample_data_, eq_sub)
    #fit_summary_ = rbind(fit_summary_, cfa_summary)

    return(cfa_summary)
    }
    cfa_weights_r_func1 <- function(eq_sub_scale_merged_brand,req_cols) {
    # print(R.version.string)
    category_list_pillar <- unique(eq_sub_scale_merged_brand$Category)

    ###------ CFA - all category ---- ###

    sample_data_ = data.frame()
    fit_indices_ = data.frame()
    estimates_ = data.frame()
    fit_summary_all_cat = data.frame()
    corr_pillar_ = data.frame()
    error_messages = c()

    ## Selecting columns that needs to be taken for all categories together
    req_cols_all_cat <- req_cols[req_cols$Select == 'Y', ]

    ##taking stacked brand results
    eq_sub_scale_merged_all_category_filtered_base <- eq_sub_scale_merged_brand[eq_sub_scale_merged_brand$New_Brand == "Stacked Brand",]
    # ind_factor_list_all_cat <- unique(req_cols_all_cat$Equity Pillar)
    # print('ind_factor_list_all_cat:',ind_factor_list_all_cat)

    # Create the samples
    eq_sub_scale_merged_all_category_filtered_list <- list()

    equity_dt_seed_all_cat <- as.data.frame(eq_sub_scale_merged_all_category_filtered_base)


    for (i in sample_seeds) {
    # For different seeds
    set.seed(i)

    # Sample the data based on the sampled indicies
    eq_sub_scale_merged_all_category_filtered_list[[i]]  <- equity_dt_seed_all_cat[sample(nrow(equity_dt_seed_all_cat),0.95*nrow(equity_dt_seed_all_cat)),]
    }

    for (s in sample_seeds)
    {
    print(paste("Seed: ",s))
    eq_sub_scale_merged_all_category_filtered <- eq_sub_scale_merged_all_category_filtered_list[[s]]

    complete_null_columns_all_cat <- colnames(eq_sub_scale_merged_all_category_filtered)[colSums(is.na(eq_sub_scale_merged_all_category_filtered)) == nrow(eq_sub_scale_merged_all_category_filtered)]
    eq_sub_scale_merged_all_category_filtered <- select(eq_sub_scale_merged_all_category_filtered,-contains(complete_null_columns_all_cat))


    eq_sub_scale_merged_all_category_filtered <- eq_sub_scale_merged_all_category_filtered %>% mutate_if(is.numeric, ~replace_na(.,mean(., na.rm = TRUE)))


    all_cat_pillars_list <- ALL_CATEGORY_PILLARS
    all_cat_pillars_list <- gsub('_pillar', '', all_cat_pillars_list)

    #for (pillar_ in ind_factor_list_all_cat){
    for (pillar_ in all_cat_pillars_list){
        # pillar_ = "ratings_reviews"

        print(pillar_)
        cfa_fit_indices = data.frame()
        cfa_estimates = data.frame()

        l_pillar <- pillar_
        # print(str(req_cols_all_cat))
        # print("pillar_values_unique:",unique(req_cols_all_cat$`Equity Pillar`))
        print(req_cols_all_cat[req_cols_all_cat$`Equity Pillar` == l_pillar,]$idv_for_model_corrected)
        # print(colnames(req_cols_all_cat))
        pillar_metrics <- unique(req_cols_all_cat[req_cols_all_cat$`Equity Pillar` == l_pillar,]$idv_for_model_corrected)
        # print(req_cols_all_cat)
        # print(colnames(req_cols_all_cat))
        # print("pillar_metrics:",pillar_metrics)
        l_n_pillar_metrics <- length(pillar_metrics)

        # Intersection with metrics that are actually only available
        pillar_a_metrics <- pillar_metrics[pillar_metrics %in% names(eq_sub_scale_merged_all_category_filtered)]
        l_pillar_a_metrics <- pillar_metrics[!pillar_metrics %in% names(eq_sub_scale_merged_all_category_filtered)]
        l_n_pillar_a_metrics <- length(pillar_a_metrics)

        # Dynamic way of writing to factor structure
        fa_str <- paste0(l_pillar,"_pillar =~ ",paste(pillar_a_metrics, collapse = "+"))

        # Correlation of available metrics
        corr_pillar <- as.data.frame(as.table(cor(eq_sub_scale_merged_all_category_filtered[pillar_a_metrics])))
        corr_pillar$Pillar <- pillar_
        corr_pillar_ = rbind(corr_pillar_, corr_pillar)

        tryCatch({
        # print("all_cat_fa_str :",fa_str)
        fit_summary_c = cfa(fa_str,eq_sub_scale_merged_all_category_filtered,s)
        if (nrow(fit_summary_all_cat) == 0) {
        fit_summary_all_cat <- fit_summary_c  # Initialize if empty
        } else {
            fit_summary_all_cat <- rbind(fit_summary_all_cat, fit_summary_c)  # Append if not empty
        }
        },
        error = function(e) {
        # Capture and store the error message
        #error_messages[[n_factors]] <- conditionMessage(e)
        warning(paste0("Error in CFA in factor:",pillar_))
        message(paste0("Error in CFA in factor:",pillar_))

        }
        )
    }
    }

    fit_summary_all_cat$Brands = "Stacked Brand"
    fit_summary_all_cat$Category = "ALL CATEGORY"

    corr_pillar_all_cat <- copy(corr_pillar_)
    corr_pillar_all_cat$Brands = "Stacked Brand"
    corr_pillar_all_cat$Category = "ALL CATEGORY"

    ### --- CFA -by category ---###

    fit_summary_all_brands = data.frame()
    corr_pillar_all_brands = data.frame()
    for(category_ in category_list_pillar){

    tryCatch({
        #category_ = "CAT FOOD"

        req_cols1 <- req_cols[req_cols$Select == 'Y' & req_cols$product_category_idv == category_,]
        eq_sub_scale_merged_category_filtered <- eq_sub_scale_merged_brand[eq_sub_scale_merged_brand$Category == category_,]
        brand_list_pillar <- "Stacked Brand"

        for (brand in brand_list_pillar){

        eq_sub_scale_merged_brand_category_filtered_base <- eq_sub_scale_merged_category_filtered[eq_sub_scale_merged_category_filtered$New_Brand == brand,]

        complete_null_columns_by_cat <- colnames(eq_sub_scale_merged_brand_category_filtered_base)[colSums(is.na(eq_sub_scale_merged_brand_category_filtered_base)) == nrow(eq_sub_scale_merged_brand_category_filtered_base)]
        eq_sub_scale_merged_brand_category_filtered_base <- select(eq_sub_scale_merged_brand_category_filtered_base,-contains(complete_null_columns_by_cat))

        eq_sub_scale_merged_brand_category_filtered_base <- eq_sub_scale_merged_brand_category_filtered_base %>% mutate_if(is.numeric, ~replace_na(.,mean(., na.rm = TRUE)))

        # Access each factor structure and apply cfa
        sample_data_ = data.frame()
        fit_indices_ = data.frame()
        estimates_ = data.frame()
        fit_summary_ = data.frame()
        corr_pillar_ = data.frame()
        error_messages = c()

        ind_factor_list <- unique(req_cols1$`Equity Pillar`)

        by_cat_pillars_list <- BY_CATEGORY_PILLARS
        by_cat_pillars_list <- gsub('_pillar', '', by_cat_pillars_list)

        # Create the samples
        eq_sub_scale_merged_brand_category_filtered_list <- list()

        # Ensure main data frame is data table
        equity_dt_seed <- as.data.frame(eq_sub_scale_merged_brand_category_filtered_base)

        # Loop through each seeds
        for (i in sample_seeds) {
            # For different seeds
            set.seed(i)

            # Sample the data based on the sampled indicies
            eq_sub_scale_merged_brand_category_filtered_list[[i]]  <- equity_dt_seed[sample(nrow(equity_dt_seed),0.95*nrow(equity_dt_seed)),]
        }

        for (s in sample_seeds)
        {
            print(paste("seed:",s))
            eq_sub_scale_merged_brand_category_filtered <- eq_sub_scale_merged_brand_category_filtered_list[[s]]

            for (pillar_ in by_cat_pillars_list){
            # for (pillar_ in list('product_feedback')) {
                fa_str <- ""
                pillar_a_metrics <- list()

                print(paste('Pillar_name:', str(pillar_)))

                # Filter pillar metrics
                pillar_metrics <- unique(req_cols1$idv_for_model_corrected[
                    (req_cols1$`Equity Pillar` == pillar_) &
                    (req_cols1$product_category_idv == category_ | req_cols1$product_category_idv == 'All')
                ])
                print("pillar_metrics")
                print(pillar_metrics)

                pillar_a_metrics <- pillar_metrics[pillar_metrics %in% names(eq_sub_scale_merged_brand_category_filtered)]
                print("pillar_a_metrics")
                print(pillar_a_metrics)

                if (length(pillar_a_metrics) > 0) {
                    fa_str <- paste0(pillar_, "_pillar =~ ", paste(pillar_a_metrics, collapse = "+"))
                    print("fa_str")
                    print(fa_str)
                } else {
                    message(paste('No metrics available for pillar:', pillar_))
                    next  # Skip to the next iteration
                }

                tryCatch({
                    fit_summary_ <- cfa(fa_str, eq_sub_scale_merged_brand_category_filtered,s)

                    # Investigate variance issues
                    print(varTable(fit_summary_))

                    # Append results
                    fit_summary_b <- copy(fit_summary_)
                    fit_summary_b$Brands <- brand
                    fit_summary_b$Category <- category_
                    # fit_summary_all_brands <- rbind(fit_summary_all_brands, fit_summary_b)
                    if (nrow(fit_summary_all_brands) == 0) {
                    fit_summary_all_brands <- fit_summary_b  # Initialize if empty
                    } else {
                        fit_summary_all_brands <- rbind(fit_summary_all_brands, fit_summary_b)  # Append if not empty
                    }
                    # Handle correlation data
                    corr_pillar <- as.data.frame(as.table(cor(eq_sub_scale_merged_brand_category_filtered[pillar_a_metrics])))
                    corr_pillar$Pillar <- pillar_
                    corr_pillar_ <- rbind(corr_pillar_, corr_pillar)

                    corr_pillar_b <- copy(corr_pillar_)
                    corr_pillar_b$Brands <- brand
                    corr_pillar_b$Category <- category_
                    # corr_pillar_all_brands <- rbind(corr_pillar_all_brands, corr_pillar_b)

                    if (nrow(corr_pillar_all_brands) == 0) {
                    corr_pillar_all_brands <- corr_pillar_b  # Initialize if empty
                    } else {
                        corr_pillar_all_brands <- rbind(corr_pillar_all_brands, corr_pillar_b)  # Append if not empty
                    }
                }, error = function(e) {
                    message(paste('Error in CFA fit for pillar:', pillar_))
                    message(conditionMessage(e))  # Detailed error message
                    next
                })
            }
        }


        }
    },
    error = function(e) {
        # Capture and store the error message
        warning(paste0("Error in CFA in factor:",pillar_,brand,category_," seed: ",s))
        message(paste0("Error in CFA in factor:",pillar_,brand,category_," seed: ",s))

    }
    )
    #
    }

    return(list(fit_summary_all_cat = fit_summary_all_cat, fit_summary_all_brands = fit_summary_all_brands))
    }
    """
    )

    # Call the R function

    robjects.r(
        """
            # if(refresh_type == "model_refresh") {
            cfa_fit_results <- cfa_weights_r_func1(eq_sub_scale_merged_brand_r_dt,req_cols_r_df)
            fit_summary_all_cat <- cfa_fit_results$fit_summary_all_cat
            fit_summary_all_brands <- cfa_fit_results$fit_summary_all_brands
            # }
            """
    )

    # Convert result back to pandas DataFrame
    # cfa_fit_results_py = pandas2ri.rpy2py(robjects.globalenv['cfa_fit_results'])
    fit_summary_all_cat_py = pandas2ri.rpy2py(
        robjects.globalenv["fit_summary_all_cat"]
    )
    fit_summary_all_brands_py = pandas2ri.rpy2py(
        robjects.globalenv["fit_summary_all_brands"]
    )

    return fit_summary_all_cat_py, fit_summary_all_brands_py


# COMMAND ----------


def run_model_type(
    model_type,
    feat_eng_config,
    refresh_config,
    X_train,
    y_train,
    X_test_hold,
    y_test_hold,
    train_x_all,
    train_y_all,
    dv,
    brand_name,
    category_name,
    pillar_name,
    temp_title,
):
    """
    Function to handle both RandomForest and XGBoost models.
    Arguments:
    - model_type: Type of model to run ('RandomForest' or 'XGBoost')
    - model_config: Configuration dictionary for model hyperparameters
    - run_config: Run configuration dictionary (which models to run)
    - X_train: Training feature set
    - y_train: Training target set
    - X_test_hold: Test holdout feature set
    - train_x_all: Full training feature set
    - train_y_all: Full training target set
    - dv: Dependent variable (used for the 'Latest DV')
    - brand_name: Name of the brand
    - category_name: Name of the category
    - pillar_name: Name of the pillar
    - temp_title: Trial name for the model run
    """
    # Initialize the model type
    if model_type == "RandomForest":
        model_class = RandomForestRegressor
    elif model_type == "XGBoost":
        model_class = xgb.XGBRegressor
    # param_grid = {
    #     "max_depth": feat_eng_config["weights_model"]["hyperparameters"][model_type]["grid_search"]["max_depth"],
    #     "n_estimators": feat_eng_config["weights_model"]["hyperparameters"][model_type]["grid_search"]["n_estimators"],
    #     "learning_rate": feat_eng_config["weights_model"]["hyperparameters"][model_type]["grid_search"]["learning_rate"],
    #     "random_state": feat_eng_config["weights_model"]["hyperparameters"][model_type]["grid_search"]["random_state"]
    # }
    param_grid = {
        key: feat_eng_config["weights_model"]["hyperparameters"][model_type][
            "grid_search"
        ][key]
        for key in feat_eng_config["weights_model"]["hyperparameters"][
            model_type
        ]["grid_search"]
        if key != "eval_metrics"  # Exclude eval_metrics if not needed
    }

    # Perform grid search for hyperparameter tuning
    regressor = model_class(
        random_state=feat_eng_config["weights_model"]["hyperparameters"][
            model_type
        ]["random_state"]
    )
    search = GridSearchCV(
        regressor,
        param_grid,
        cv=feat_eng_config["weights_model"]["cross_validation_number"],
        scoring=["r2", "neg_mean_absolute_percentage_error"],
        refit="neg_mean_absolute_percentage_error",
    )
    search.fit(X_train, y_train)

    print(
        f"The best hyperparameters for {model_type} are {search.best_params_}"
    )

    # Set up the model with the best parameters
    regressor = model_class(**search.best_params_)
    regressor.fit(X_train, y_train)

    # Feature importance and SHAP values
    if model_type == "RandomForest":
        features = list(X_train.columns)
        f_i = list(zip(features, regressor.feature_importances_))
        f_i.sort(key=lambda x: x[1], reverse=True)

        rfe = RFECV(
            regressor,
            cv=feat_eng_config["weights_model"]["cross_validation_number"],
            scoring="neg_mean_absolute_percentage_error",
        )
        rfe.fit(X_train, y_train)
        selected_features = list(np.array(features)[rfe.get_support()])
        print(selected_features)

        feat_importance = pd.DataFrame(
            f_i, columns=["Features", "Feature Importance"]
        )
        feat_importance.set_index("Features", inplace=True)
        feat_importance = feat_importance.iloc[:20, :]
        print(feat_importance)

    elif model_type == "XGBoost":
        dict_result = regressor.get_booster().get_score(importance_type="gain")
        print(
            f"Feature importance for {model_type}",
            pd.DataFrame(
                dict_result.items(), columns=["Feature", "gain"]
            ).sort_values(by="gain", ascending=False),
        )

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_train)
    feat_df = pd.DataFrame(
        np.abs(pd.DataFrame(shap_values, columns=X_train.columns)).mean(),
        columns=["shap values"],
    )
    print(
        f"{model_type} shap importance",
        feat_df.sort_values(by="shap values", ascending=False),
    )

    # Predictions
    y_pred_test = regressor.predict(X_test_hold)
    y_pred_train = regressor.predict(X_train)
    y_pred_all = regressor.predict(train_x_all[list(X_train.columns)])

    feat_importance = feat_importance.reset_index().rename(
        columns={"Feature Importance": "Feature Importance/coefficient"}
    )
    feat_df = feat_df.reset_index().rename(columns={"index": "Shap Features"})

    # Metrics
    mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
    mse_train = metrics.mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = metrics.r2_score(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

    results_all_model = pd.concat([feat_importance, feat_df], axis=1)
    results_all_model["Model"] = model_type
    results_all_model["Brand"] = brand_name
    results_all_model["Category"] = category_name
    results_all_model["pillar"] = pillar_name
    results_all_model["Latest DV"] = dv.values[-1]
    results_all_model["R2_Score_Train"] = r2_train
    results_all_model["MAPE_Train"] = mape_train
    results_all_model["R2_score_fold"] = cross_val_score(
        regressor,
        X_train,
        y_train,
        cv=feat_eng_config["weights_model"]["cross_validation_number"],
        scoring="r2",
    ).mean()
    results_all_model["MAPE_fold"] = (
        cross_val_score(
            regressor,
            X_train,
            y_train,
            cv=feat_eng_config["weights_model"]["cross_validation_number"],
            scoring="neg_mean_absolute_percentage_error",
        ).mean()
        * -1
    )
    results_all_model["R2_score_Hold_out"] = metrics.r2_score(
        y_test_hold, y_pred_test
    )
    results_all_model["MAPE_Hold_out"] = mean_absolute_percentage_error(
        y_test_hold, y_pred_test
    )
    results_all_model["R2_score_all"] = metrics.r2_score(
        train_y_all, y_pred_all
    )
    results_all_model["MAPE_all"] = mean_absolute_percentage_error(
        train_y_all, y_pred_all
    )
    results_all_model["Trial name"] = temp_title
    results_all_model["Best_Params_Gridsearchcv"] = str(search.best_params_)

    # results_all_model = pd.concat([results_all_model, results_all_model], axis=0)

    actual_vs_predicted = pd.DataFrame()
    actual_vs_predicted["Actual"] = train_y_all
    actual_vs_predicted["Predicted"] = y_pred_all
    actual_vs_predicted["Brand"] = brand_name
    actual_vs_predicted["Category"] = category_name
    actual_vs_predicted["pillar"] = pillar_name
    actual_vs_predicted["Model"] = model_type
    actual_vs_predicted["Trial name"] = temp_title
    actual_vs_predicted = pd.concat(
        [actual_vs_predicted, actual_vs_predicted], axis=0
    )

    title = f"{brand_name}-{category_name}-{model_type}_{temp_title}"
    print(os.listdir())

    return results_all_model, actual_vs_predicted


# COMMAND ----------


def metric_weights_model(
    pillar,
    brand_name,
    category_name,
    modeling_data,
    feat_eng_config,
    req_cols,
    refresh_config,
):
    """Model data preparation and model training selection and selecting variables from idv_list

    Args:
        pillar (str): The pillar that we are considering for training
        brand_name (str): Brand name for the training
        category_name (str): category of the brand to train
        modeling_data (DataFrame): Data for the training
        feat_eng_config (dict): Feature engineering config
        req_cols (DataFrame): idv data
        refresh_config (dict): refresh config from configuration_function.py

    Returns:
        _type_: _description_
    """
    attr_df1 = modeling_data.copy()
    attr_df1["date"] = pd.to_datetime(attr_df1["date"], format="%Y-%m-%d")
    attr_df1 = attr_df1.sort_values(by="date", ascending=True)
    # attr_df1.rename(columns={'brand': 'brand_rt', 'category': 'category_rt', 'market_share': 'market_share_total_sales_rt'}, inplace=True)
    attr_df = attr_df1.copy()
    if feat_eng_config["weights_model"]["drop_mean_attributes"]:
        attr_df = attr_df[
            attr_df.columns.drop(list(attr_df.filter(regex="_mean")))
        ]

    if feat_eng_config["weights_model"]["drop_rank1st_attributes"]:
        attr_df = attr_df[
            attr_df.columns.drop(list(attr_df.filter(regex="_rank_1st")))
        ]

    all_pillar_results = pd.DataFrame()
    corr_all_results = pd.DataFrame()
    idv_sel_cols = req_cols.copy()
    idv_sel_cols.drop(columns=["Unnamed: 0"], inplace=True)
    attr_df.columns = attr_df.columns.str.replace(" - ", "_")
    attr_df.columns = attr_df.columns.str.replace(" ", "_")
    attr_df.columns = attr_df.columns.str.lower()

    ## Correcting column names for the dataframe to run Framework
    attr_df.rename(
        columns={
            "brand_rt": "brand",
            "category": "c_category",
            "category_rt": "c_category",
            "market_share_total_sales_rt": "market_share_total_sales",
            "market_share": "market_share_total_sales",
        },
        inplace=True,
    )
    print(pillar)
    Metric_Group_list = []
    Metric_Group_list.append(pillar)
    temp_title = pillar
    pillar_name = pillar
    output_file_name = pillar
    idv_sel_cols = req_cols.copy()
    idv_sel_cols = idv_sel_cols[
        idv_sel_cols["Equity Pillar"].isin(Metric_Group_list)
    ]
    dataframes = []
    matched_cols = []
    idv_sel_cols_list = []
    results_all_model = pd.DataFrame()
    actual_vs_predicted = pd.DataFrame()
    error_brands = []
    results_df_corr = pd.DataFrame()
    corr_all_br = pd.DataFrame()
    idv_sel_cols_list = []
    idv_sel_cols_merge = pd.DataFrame()
    idv_sel_cols_merge_exclude = pd.DataFrame()
    matched_cols = []
    chosen_idvs = []
    brand_prc_cls_df = pd.DataFrame()
    brand_req_df = pd.DataFrame()
    idvs = pd.DataFrame()
    idvs_scaled = pd.DataFrame()
    results_all_model_RF = pd.DataFrame()
    results_all_model_XGB = pd.DataFrame()
    brand_prc_cls_df = attr_df.loc[
        (attr_df["brand"] == brand_name)
        & (attr_df["c_category"] == category_name)
    ]
    brand_req_df = brand_prc_cls_df.copy()
    if brand_req_df[feat_eng_config["weights_model"]["DV"]].sum() <= 0:
        error_brands.append(brand_name + "-" + category_name)

    print("Brand: ", brand_name)
    print("Category: ", category_name)

    ## Drop the unwanted columns for modeling
    columns_log_list = list(brand_req_df.columns)
    columns_log_list.remove(feat_eng_config["weights_model"]["DV"])
    columns_log_list.remove("c_category")
    columns_log_list.remove("brand")
    columns_log_list.remove("date")

    columns_log_df = pd.DataFrame(columns_log_list, columns=["Column_names"])
    columns_log_df["Brand"] = brand_name
    columns_log_df["Category"] = category_name
    columns_log_df["Reasons_to_drop"] = np.nan

    idv_sel_cols1 = idv_sel_cols[
        idv_sel_cols["product_category_idv"] == category_name
    ]

    idv_sel_cols1 = idv_sel_cols1[idv_sel_cols1["Select"] == "Y"]

    idv_sel_cols_list = list(idv_sel_cols1["idv_for_model_corrected"].unique())
    if feat_eng_config["weights_model"]["Cols_force_sel"] == True:
        idv_sel_cols_list = feat_eng_config["weights_model"]["Force_cols_net"]

    # Replacing spaces with underscore and removing ' - ' in column names
    idv_sel_cols_list = [x.replace(" - ", "_") for x in idv_sel_cols_list]
    idv_sel_cols_list = [x.replace(" ", "_") for x in idv_sel_cols_list]
    # removing case sensitivity
    idv_sel_cols_list = [x.lower() for x in idv_sel_cols_list]

    for col1 in idv_sel_cols_list:
        if col1 in list(brand_req_df.columns):
            matched_cols.append(col1)
    idv_sel_cols_list = list(set(matched_cols))

    if feat_eng_config["weights_model"]["drop_mean_attributes"]:
        idv_sel_cols_list = [
            i for i in idv_sel_cols_list if not ("_mean" in i)
        ]
        for col in list(
            columns_log_df.loc[
                pd.isnull(columns_log_df["Reasons_to_drop"]) == True,
                "Column_names",
            ]
        ):
            if col not in idv_sel_cols_list:
                columns_log_df.loc[
                    columns_log_df["Column_names"] == col, "Reasons_to_drop"
                ] = "Dropped for mean column"

    if feat_eng_config["weights_model"]["drop_rank1st_attributes"]:
        idv_sel_cols_list = [
            i for i in idv_sel_cols_list if not ("_rank_1st" in i)
        ]
        for col in list(
            columns_log_df.loc[
                pd.isnull(columns_log_df["Reasons_to_drop"]) == True,
                "Column_names",
            ]
        ):
            if col not in idv_sel_cols_list:
                columns_log_df.loc[
                    columns_log_df["Column_names"] == col, "Reasons_to_drop"
                ] = "Dropped for Rank column"

    # actual_df = brand_req_df.copy()
    brand_req_df1 = brand_req_df[
        idv_sel_cols_list
        + [feat_eng_config["weights_model"]["DV"]]
        + [feat_eng_config["weights_model"]["Temp"]]
    ]
    brand_req_df1
    filtered_df = pd.DataFrame()
    null_columns = []
    filtered_df = brand_req_df1[brand_req_df1["new_brand"] == "Stacked Brand"]
    # Get column names with all NaN values
    null_columns = filtered_df.columns[filtered_df.isnull().all()]

    # Remove the completely null columns
    df = filtered_df.drop(columns=null_columns)
    column_names = df.columns.tolist()
    # Convert the lists to sets
    set1 = set(idv_sel_cols_list)
    set2 = set(column_names)

    common_elements = set1.intersection(set2)
    # final_idvs
    final_idvs = list(common_elements)
    final_idvs
    brand_req_df_2 = brand_req_df[
        final_idvs
        + [feat_eng_config["weights_model"]["DV"]]
        + [feat_eng_config["weights_model"]["Temp"]]
    ]
    brand_req_df_2
    brand_req_df = brand_req_df_2[brand_req_df_2["new_brand"] == brand_name]
    brand_req_df = brand_req_df[
        final_idvs + [feat_eng_config["weights_model"]["DV"]]
    ]
    brand_req_df

    brand_req_df = brand_req_df.reset_index(drop=True)
    length = len(brand_req_df)
    print("length", length)
    if len(brand_req_df) > 20:

        df_null = pd.DataFrame(
            brand_req_df.isnull().sum(), columns=(["Null count"])
        )
        df_null = df_null.sort_values(by="Null count", ascending=False)

        df_null[df_null["Null count"] >= brand_req_df.shape[0] * 0.5]
        brand_req_df.drop(
            columns=list(
                df_null[
                    df_null["Null count"] >= brand_req_df.shape[0] * 0.5
                ].index
            ),
            inplace=True,
        )

        for col in list(
            columns_log_df.loc[
                pd.isnull(columns_log_df["Reasons_to_drop"]) == True,
                "Column_names",
            ]
        ):
            if col not in list(brand_req_df.columns):
                columns_log_df.loc[
                    columns_log_df["Column_names"] == col, "Reasons_to_drop"
                ] = "Dropped for > 50 % null"

        print(
            "len(set(brand_req_df.columns) - {feat_eng_config['weights_model']['DV']})",
            len(
                set(brand_req_df.columns)
                - {feat_eng_config["weights_model"]["DV"]}
            ),
        )

        if (
            len(
                set(brand_req_df.columns)
                - {feat_eng_config["weights_model"]["DV"]}
            )
            > 0
        ):
            print(
                len(
                    set(brand_req_df.columns)
                    - {feat_eng_config["weights_model"]["DV"]}
                )
            )

            if feat_eng_config["weights_model"]["is_lag_considered"] == True:
                brand_req_df = brand_req_df.iloc[3:, :].reset_index(drop=True)

            brand_req_df.isnull().sum().sort_values(ascending=False)

            brand_req_df = brand_req_df.fillna(brand_req_df.mean())

            brand_req_df.isnull().sum().sort_values(ascending=False)

            if feat_eng_config["weights_model"]["is_lag_considered"] == True:
                chosen_idvs = choose_best_lag_feature(
                    brand_req_df,
                    idv_sel_cols,
                    feat_eng_config["weights_model"]["DV"],
                )
                for col in list(
                    columns_log_df.loc[
                        pd.isnull(columns_log_df["Reasons_to_drop"]) == True,
                        "Column_names",
                    ]
                ):
                    if col not in list(chosen_idvs):
                        columns_log_df.loc[
                            columns_log_df["Column_names"] == col,
                            "Reasons_to_drop",
                        ] = "Dropped for best lag"

            else:
                chosen_idvs = list(brand_req_df.columns)
                chosen_idvs.remove(
                    feat_eng_config["weights_model"]["DV"]
                )  ## if only ACV is needed and drop price in the idv list
            print("chosen_idvs : ", list(chosen_idvs))

            modeldf = brand_req_df
            model_DV = feat_eng_config["weights_model"]["DV"]
            modeldf.shape

            # Creating modeling data and target
            print("parameter:", model_DV)
            idvs = modeldf.drop(model_DV, axis=1)  # feature matrix
            dv = modeldf[model_DV]

            # Taking log if Base EQ is target
            if feat_eng_config["weights_model"]["log_convert_DV"]:
                dv = np.log1p(modeldf[model_DV])

            if feat_eng_config["weights_model"]["standardize"]:
                mmscaler = MinMaxScaler()  # Do even before feature selection
                idvs_scaled = pd.DataFrame(
                    mmscaler.fit_transform(idvs), columns=idvs.columns
                )
                idvs = idvs_scaled.copy()
                # idvs = pd.DataFrame(mmscaler.fit_transform(idvs), columns = idvs.columns)

            feat_at_each_iter = []
            per_at_each_iter = []
            coeff_at_each_iter = []
            subset_at_each_iter = pd.DataFrame()

            print("Length of Chosen IDVS:", len(chosen_idvs))

            ## Creating a list of final columns for modeling
            cols_to_select = sorted(list(set(chosen_idvs)))

            if len(cols_to_select) >= 2:
                for i in [2]:
                    # i = 2
                    print(i)
                    if (
                        feat_eng_config["weights_model"]["Time_series_split"]
                        == True
                    ):
                        train_x = idvs[cols_to_select].iloc[:42, :]
                        train_y = dv[:42]
                        test_x = idvs[cols_to_select].iloc[42:, :]
                        test_y = dv[42:]
                    if (
                        feat_eng_config["weights_model"]["Random_seed_split"]
                        == True
                    ):
                        train_x, test_x, train_y, test_y = train_test_split(
                            idvs[cols_to_select],
                            dv,
                            test_size=6,
                            random_state=i,
                            shuffle=True,
                        )
                    train_x_all = idvs[cols_to_select]
                    train_y_all = dv

                    if feat_eng_config["weights_model"]["P_N_check"] == True:
                        corr_df_pn = pd.DataFrame()
                        corr_df_pn = pd.DataFrame(train_x.corrwith(train_y))
                        corr_df_pn["Brand"] = (
                            brand_name  # Comment if col not needed
                        )
                        corr_df_pn["category"] = (
                            category_name  # Comment if col not needed
                        )

                        corr_df_pn["idv_for_model_corrected"] = (
                            corr_df_pn.index
                        )
                        corr_df_pn.reset_index(drop=True, inplace=True)
                        corr_df_pn = corr_df_pn.rename(
                            columns={0: "Correlation"}
                        )
                        corr_df_pn["corr_abs"] = abs(
                            corr_df_pn["Correlation"]
                        )  # Comment if col not needed
                        corr_df_pn.sort_values(
                            by=["corr_abs"], ascending=False, inplace=True
                        )  # Comment if col not needed
                        corr_df_pn.reset_index(
                            drop=True, inplace=True
                        )  # Comment if col not needed

                        results_df_corr = pd.concat(
                            [results_df_corr, corr_df_pn], axis=0
                        )  # Comment if col not needed

                        cols_to_sel_df = pd.DataFrame(
                            {"idv_for_model_corrected": cols_to_select},
                            index=None,
                        )
                        M1_df = pd.merge(
                            idv_sel_cols_merge_exclude,
                            cols_to_sel_df,
                            on=["idv_for_model_corrected"],
                            how="inner",
                        )
                        idv_sel_P_N = pd.merge(
                            M1_df,
                            corr_df_pn,
                            on=["idv_for_model_corrected"],
                            how="inner",
                        )

                        idv_sel_P_N_Match = idv_sel_P_N[
                            ~(
                                (
                                    idv_sel_P_N["Correlation"]
                                    < -feat_eng_config["weights_model"][
                                        "Counter_Intuitive_cut_off"
                                    ]
                                )
                                & (idv_sel_P_N["Intutive"] == "P")
                            )
                            | (
                                (
                                    idv_sel_P_N["Correlation"]
                                    > feat_eng_config["weights_model"][
                                        "Counter_Intuitive_cut_off"
                                    ]
                                )
                                & (idv_sel_P_N["Intutive"] == "N")
                            )
                        ].reset_index(drop=True)

                        Count_intuitive_cols_filtered = sorted(
                            list(
                                set(
                                    idv_sel_P_N_Match[
                                        "idv_for_model_corrected"
                                    ]
                                )
                            )
                        )
                        train_x = train_x[Count_intuitive_cols_filtered]
                        test_x = test_x[Count_intuitive_cols_filtered]
                        train_x_all = idvs[Count_intuitive_cols_filtered]

                    if (
                        feat_eng_config["weights_model"][
                            "Corr_file_generation"
                        ]
                        == True
                    ):
                        if (
                            feat_eng_config["weights_model"]["P_N_check"]
                            == True
                        ):
                            train_x_corr = idvs[Count_intuitive_cols_filtered]
                        else:
                            train_x_corr = idvs[cols_to_select]
                        train_y_corr = dv
                        corr_raw_df = pd.DataFrame(
                            train_x_corr.corrwith(train_y_corr)
                        )
                        corr_raw_df["Brand"] = (
                            brand_name  # Comment if col not needed
                        )
                        corr_raw_df["category"] = category_name
                        corr_raw_df["idvs_for_model"] = corr_raw_df.index
                        corr_raw_df.reset_index(drop=True, inplace=True)
                        corr_raw_df = corr_raw_df.rename(
                            columns={0: "Correlation with DV"}
                        )
                        corr_raw_df["corr_abs"] = corr_raw_df[
                            "Correlation with DV"
                        ].abs()  # Comment if col not needed
                        corr_raw_df.sort_values(
                            by=["corr_abs"], ascending=False, inplace=True
                        )  # Comment if col not needed
                        corr_raw_df.reset_index(drop=True, inplace=True)

                    if (
                        feat_eng_config["weights_model"]["PCA_Transform"]
                        == True
                    ):
                        pca = PCA(n_components=0.95)
                        PCA_train = pca.fit_transform(train_x)
                        PCA_comp = []
                        for i in range(0, pca.n_components_, 1):
                            PCA_comp.append("PCA" + str(i))
                        train_x = pd.DataFrame(
                            PCA_train, columns=PCA_comp, index=None
                        )
                        PCA_test = pca.transform(test_x)
                        test_x = pd.DataFrame(
                            PCA_test, columns=PCA_comp, index=None
                        )
                        PCA_train_all = pca.transform(train_x_all)
                        train_x_all = pd.DataFrame(
                            PCA_train_all, columns=PCA_comp, index=None
                        )

                    print("Chosen_lags:", chosen_idvs)

                    X_test_hold = test_x.copy()
                    y_test_hold = test_y.copy()

                    X_train = train_x.copy()
                    y_train = train_y.copy()

                    feat_importance = pd.DataFrame()
                    feat_df = pd.DataFrame()

                    if (
                        refresh_config["weights_models"]["RandomForest"]["run"]
                        == True
                    ):
                        results_all_model, actual_vs_predicted = (
                            run_model_type(
                                "RandomForest",
                                feat_eng_config,
                                refresh_config,
                                X_train,
                                y_train,
                                X_test_hold,
                                y_test_hold,
                                train_x_all,
                                train_y_all,
                                dv,
                                brand_name,
                                category_name,
                                pillar_name,
                                temp_title,
                            )
                        )
                        # Process results_rf and actual_rf as needed

                    if (
                        refresh_config["weights_models"]["XGBoost"]["run"]
                        == True
                    ):
                        results_all_model, actual_vs_predicted = (
                            run_model_type(
                                "XGBoost",
                                feat_eng_config,
                                refresh_config,
                                X_train,
                                y_train,
                                X_test_hold,
                                y_test_hold,
                                train_x_all,
                                train_y_all,
                                dv,
                                brand_name,
                                category_name,
                                pillar_name,
                                temp_title,
                            )
                        )

    return results_all_model, actual_vs_predicted


# COMMAND ----------


def modelling(
    input_config,
    output_config,
    mapping_config,
    refresh_config,
    feat_eng_config,
    filter_config,
    storage_options,
    refresh_type,
):
    """Main function for modelling for creating cfa result and ML model shap value against
    the dependent variables and storing all the result to corresponding path mentioned in the config

    Args:
        input_config (dict): Dictionary containing details of the path of input files
        output_config (dict): Dictionary containing details of the path of output files
        mapping_config (dict): Dictionary containing details of the path of mapping files
        refresh_config (dict): Dictionary containing details of refresh type
        feat_eng_config (dict): Dictionary containing details of hyperparameters for the models
        filter_config (dict): Dictionary containing details of required filters for the data preprocessing
        storage_options (dict): Dictionary containing details of storage options for databricks else None
        refresh_type (str): refresh type for the data



    """
    # input
    eq_sub_scale_merged_brand = pd.read_csv(
        output_config["data_prep"]["eq_sub_scale"],
        storage_options=storage_options,
    )
    modeling_data = pd.read_csv(
        output_config["data_prep"]["modeling_data"],
        storage_options=storage_options,
    )
    eq_sub_scale_merged_brand["date"] = pd.to_datetime(
        eq_sub_scale_merged_brand["date"]
    )
    modeling_data["date"] = pd.to_datetime(modeling_data["date"])
    eq_sub_scale_merged_brand.rename(
        columns={
            "brand": "Brand",
            "category": "Category",
            "new_brand": "New_Brand",
        },
        inplace=True,
    )
    req_cols = pd.read_csv(
        mapping_config["idv_list"], storage_options=storage_options
    )
    brand_list = pd.read_csv(
        mapping_config["brand_list"], storage_options=storage_options
    )
    brand_list_dict = (
        brand_list.groupby("brand_group_expanded")["category"]
        .apply(list)
        .to_dict()
    )
    # cfa (need to bring loops outside)
    fit_summary_all_cat_py, fit_summary_all_brands_py = cfa(
        output_config,
        refresh_config,
        feat_eng_config,
        eq_sub_scale_merged_brand,
        req_cols,
        storage_options,
        refresh_type,
    )

    fit_summary_all_cat_py.to_csv(
        output_config["cfa"]["model_results_all_category"],
        index=False,
        storage_options=storage_options,
    )
    fit_summary_all_brands_py.to_csv(
        output_config["cfa"]["model_results_by_category"],
        index=False,
        storage_options=storage_options,
    )
    # rf1
    all_pillar_results = pd.DataFrame()
    for pillar in feat_eng_config["weights_model"]["pillars_list"]:
        for brand_name in list(brand_list_dict.keys()):
            for category_name in brand_list_dict[brand_name]:

                results_all_model, actual_vs_predicted = metric_weights_model(
                    pillar,
                    brand_name,
                    category_name,
                    modeling_data,
                    feat_eng_config,
                    req_cols,
                    refresh_config,
                )

                results_all_model = results_all_model.reset_index(drop=True)
                print("using concat")
                all_pillar_results = pd.concat(
                    [all_pillar_results, results_all_model],
                    axis=0,
                    ignore_index=True,
                )
                all_pillar_results = (
                    all_pillar_results.drop_duplicates().reset_index(drop=True)
                )

    all_pillar_results.to_csv(
        output_config["weights_model"]["model_results"],
        index=False,
        storage_options=storage_options,
    )


# COMMAND ----------

# MAGIC %md
# MAGIC R - 4.2.3 ,
# MAGIC lavaan - 0.617
# MAGIC
# MAGIC python - 3.9.19 ,
# MAGIC sklearn - 1.4.2 ,
# MAGIC shap - 0.45.1
