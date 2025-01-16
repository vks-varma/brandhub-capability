# Databricks notebook source
# MAGIC %md
# MAGIC **Pre Validation Code**

# COMMAND ----------

# MAGIC %run ./configuration_function

from configuration_function import *

# COMMAND ----------
from library_installation import *


def nielsen_rms_data_extraction(dbs_sql_hostname,dbs_sql_http_path,dbs_sql_token, market_description = 'Total US Pet Retail Plus', time_granularity = refresh_config["time_granularity"]):

    conn = sql.connect(server_hostname=dbs_sql_hostname, http_path=dbs_sql_http_path, access_token=dbs_sql_token)
    cursor = conn.cursor()

    query = f'''
        select C_VENDOR_EXPANDED as vendor, C_BRAND_GROUP_EXPANDED as brand_group_expanded, C_CATEGORY as category, Week_Ending as date,sum(dollars) as s_dollars, sum(base_dollars) as s_base_dollars, sum(Pounds) as s_pounds, sum(base_Pounds) as s_base_pounds,sum(Pct_ACV*dollars) as RMS_ACV_Selling_num, sum(dollars) as RMS_ACV_Selling_den, sum(Units) as s_units, sum(Base_Units) as s_base_units from nielsen.rms_market_facts_raw_no_walmart_view as MF
        left join nielsen.rms_periods_view as P on P.Period_Key=MF.Period_Key
        left join nielsen.rms_markets_view as M on M.Market_Key=MF.Market_Key
        left join nielsen.rms_products_view as Pr on Pr.PRODUCT_KEY=MF.PRODUCT_KEY
        where (Period_Weeks=1) and (Market_Description= '{market_description}') group by C_VENDOR_EXPANDED, C_BRAND_GROUP_EXPANDED,C_CATEGORY, Week_Ending
    '''

    # cursor.execute(query, {'market_description': market_description})
    cursor.execute(query)

    columns = [column[0] for column in cursor.description]
    rms_facts = pd.DataFrame(cursor.fetchall(), columns=columns)

    cursor.close()
    conn.close()

    rms_facts['date'] = pd.to_datetime(rms_facts['date'], format="%Y-%m-%d")

    rms_facts['RMS_ACV_Selling_wt'] = rms_facts['RMS_ACV_Selling_num'] / rms_facts['RMS_ACV_Selling_den']

    # Step 1: Filter for "BLUE BUFFALO" vendor
    blue_buffalo_rms = rms_facts[rms_facts['vendor'] == 'BLUE BUFFALO']

    # Step 2: Group by 'vendor', 'category', and 'date' and aggregate the relevant metrics (sales, eq, units)
    aggregated_bb_rms = blue_buffalo_rms.groupby(['vendor', 'category', 'date'], as_index=False).agg({
        's_dollars': 'sum',
        's_pounds': 'sum',
        's_units': 'sum',
        'RMS_ACV_Selling_num': 'sum',
        'RMS_ACV_Selling_den': 'sum',
        's_base_dollars': 'sum',
        's_base_pounds': 'sum',
        's_base_units': 'sum'
    })

    # Step 3: Create a new 'brand_group_expanded' column and set it to "BLUE BUFFALO"
    aggregated_bb_rms['brand_group_expanded'] = 'BLUE BUFFALO'

    # Step 4: Filter out the original "BLUE BUFFALO" rows from the original DataFrame
    non_blue_buffalo_rms = rms_facts[rms_facts['brand_group_expanded'] != 'BLUE BUFFALO']

    # Step 5: Combine the aggregated data with the rest of the data
    rms_facts = pd.concat([non_blue_buffalo_rms, aggregated_bb_rms], ignore_index=True)

    rms_facts["month"] = rms_facts['date'].dt.month
    rms_facts["year"] = rms_facts['date'].dt.year

    # Perform groupby and aggregations
    if time_granularity == "monthly":
        rms_facts["month"] = rms_facts['date'].dt.month
        rms_facts["year"] = rms_facts['date'].dt.year
        group_vars = ['vendor', 'brand_group_expanded', 'category', "year", "month"]
    elif time_granularity == "weekly":
        group_vars = ['vendor', 'brand_group_expanded', 'category', 'date']

    rms_facts_renamed = rms_facts.groupby(group_vars).agg(
        total_sales=('s_dollars', 'sum'),
        s_base_dollars=('s_base_dollars', 'sum'),
        eq_volume=('s_pounds', 'sum'),
        s_base_pounds=('s_base_pounds', 'sum'),
        acv_selling=('RMS_ACV_Selling_wt', 'mean'),
        total_units=('s_units', 'sum'),
        s_base_units=('s_base_units', 'sum'),
        a_dollars=('s_dollars', 'mean'),
        a_base_dollars=('s_base_dollars', 'mean'),
        a_pounds=('s_pounds', 'mean'),
        a_base_pounds=('s_base_pounds', 'mean'),
        a_units=('s_units', 'mean'),
        a_base_units=('s_base_units', 'mean')
    ).reset_index()

    if time_granularity == "monthly":
        # Helper function to calculate days in a month
        def days_in_month(year, month):
            return calendar.monthrange(year, month)[1]

        # Assuming 'year' and 'month' are columns in the DataFrame
        rms_facts_renamed['daysinmonth'] = rms_facts_renamed.apply(lambda row: days_in_month(int(row['year']), int(row['month'])), axis=1)

        # Convert monthly values to weekly values by multiplying by (daysinmonth / 7)
        rms_facts_renamed['total_sales'] = rms_facts_renamed['a_dollars'] * (rms_facts_renamed['daysinmonth'] / 7)
        rms_facts_renamed['s_base_dollars'] = rms_facts_renamed['a_base_dollars'] * (rms_facts_renamed['daysinmonth'] / 7)
        rms_facts_renamed['s_base_pounds'] = rms_facts_renamed['a_base_pounds'] * (rms_facts_renamed['daysinmonth'] / 7)
        rms_facts_renamed['eq_volume'] = rms_facts_renamed['a_pounds'] * (rms_facts_renamed['daysinmonth'] / 7)
        rms_facts_renamed['s_base_units'] = rms_facts_renamed['a_base_units'] * (rms_facts_renamed['daysinmonth'] / 7)
        rms_facts_renamed['total_units'] = rms_facts_renamed['a_units'] * (rms_facts_renamed['daysinmonth'] / 7)
        rms_facts_renamed['date'] = pd.to_datetime(rms_facts_renamed[['year', 'month']].assign(day=1))
        rms_facts_renamed.drop(columns=["month","year","daysinmonth"] ,inplace=True)
        col_to_move = rms_facts_renamed.pop('date')  # Remove the column from its original position
        rms_facts_renamed.insert(3, 'date', col_to_move)  # Insert it at the desired index

    rms_facts_renamed.drop(columns=["a_dollars","a_base_dollars","a_pounds","a_base_pounds","a_units","a_base_units"], inplace=True)

    # Round off values for each column separately
    rms_facts_renamed = rms_facts_renamed.round({'total_sales': 0, 'a_base_dollars': 0, 'eq_volume': 0, 'a_base_pounds': 0, 'acv_selling': 2, 'total_units': 0, 'total_base_units': 0})

    # Sorting the results
    rms_facts_renamed = rms_facts_renamed.sort_values(by=['vendor', 'brand_group_expanded', 'category', 'date'])

    rms_facts_renamed["average_price"] = rms_facts_renamed["total_sales"]/rms_facts_renamed["total_units"]

    # First, group by 'date' and 'CATEGORY_RT' to compute the sum of 'total_sales' for each group
    rms_facts_renamed['Category_total_sales'] = rms_facts_renamed.groupby(['date', 'category'])['total_sales'].transform('sum')

    # Then, compute the market share by dividing 'total_sales' by 'Category_total_sales'
    rms_facts_renamed['Market_share_total_sales'] = rms_facts_renamed['total_sales'] / rms_facts_renamed['Category_total_sales']
    return rms_facts_renamed


# COMMAND ----------

def create_percentage_columns(df):
    # Identify columns with patterns for 'ratings_*_count' and specific 'ratings_reviews_*_star_ratings' format
    rating_count_columns = [col for col in df.columns if col.startswith('ratings_') and col.endswith('_count')]
    star_rating_pattern = r"^ratings_reviews_[1-5]_star_ratings$"
    star_rating_columns = [col for col in df.columns if re.match(star_rating_pattern, col)]
    print('ratings_reviews_review_count' in df.columns.tolist())
    # Combine the column lists
    columns_to_process = rating_count_columns + star_rating_columns

    # Calculate percentage columns
    for col in columns_to_process:
        df[f"{col}_percentage"] = df[col] / df['ratings_reviews_review_count']

    return df

# COMMAND ----------

# def create_inverse_metrics(df, inverse_logic_df):
#     df_copy = df.copy()
#     inverse_logic_df = inverse_logic_df.reset_index(drop=True)

#     for _, row in inverse_logic_df.iterrows():
#         operation = row['operation']
#         new_col_name = row['new_name']

#         # Handle subtraction operation
#         if operation == 'sub':
#             value = float(row['values'])
#             raw_col = row['raw_cols']
#             df_copy[new_col_name] = value - df_copy[raw_col]

#         # Handle addition operation
#         elif operation == 'add':
#             cols_to_add = row['values'].split('+')
#             df_copy[new_col_name] = df_copy[cols_to_add].sum(axis=1)

#     return df_copy

# COMMAND ----------

def create_inverse_metrics(df, inverse_logic_df):
    df_copy = df.copy()
    inverse_logic_df = inverse_logic_df.reset_index(drop=True)

    for _, row in inverse_logic_df.iterrows():
        operation = row['operation']
        new_col_name = row['new_name']

        # Handle subtraction operation
        if operation == 'sub':
            value = float(row['values'])
            raw_col = row['raw_cols']
            # Ensure subtraction result is NaN if all involved values are NaN
            df_copy[new_col_name] = np.where(df_copy[raw_col].isnull(), np.nan, value - df_copy[raw_col])

        # Handle addition operation
        elif operation == 'add':
            cols_to_add = row['values'].split('+')
            # Ensure addition result is NaN if all involved values are NaN
            df_copy[new_col_name] = df_copy[cols_to_add].apply(lambda x: x.sum() if not x.isnull().all() else np.nan, axis=1)

    return df_copy


# COMMAND ----------

def create_share_metrics(df):
    # Ensure column names are lowercase and create a copy for stateless operation
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Initialize final DataFrame
    df_final = pd.DataFrame()

    # Iterate over unique dates and categories using groupby for efficient grouping
    for (date_val, category_name), cat_df in df.groupby(['date', 'category']):
        print(f"Date: {date_val}, Category: {category_name}")

        # Calculate sums for category-level metrics
        category_sum_search_volume = cat_df['search_searchvolume'].sum()
        category_sum_review_count = cat_df['ratings_reviews_review_count'].sum()

        # For each brand within the category on the given date, calculate shares
        for brand_name, br_cat_df in cat_df.groupby('brand_group_expanded'):
            print(f"Brand: {brand_name}")

            # Calculate share metrics within the temporary brand DataFrame
            br_cat_df = br_cat_df.assign(
                search_searchvolume_share=br_cat_df['search_searchvolume'] / category_sum_search_volume,
                ratings_reviews_review_count_share=br_cat_df['ratings_reviews_review_count'] / category_sum_review_count
            )

            # Concatenate each brand-level result into the final DataFrame
            df_final = pd.concat([df_final, br_cat_df], ignore_index=True)

    return df_final

# COMMAND ----------

def load_and_preprocess_data(file_path, date_cols=None, drop_cols=None, sales_or_harmonized=None):
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(file_path, storage_options= storage_options)
    #df = pd.read_csv("/dbfs/FileStore/shared_uploads/mohammad.fazil@purina.nestle.com/#harmonized_data_q2_w_rms_panel_28_8_24_v2_bl.csv")

    if drop_cols:
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Create 'date' column only if sales_or_harmonized is "sales"
    if sales_or_harmonized == "sales" and 'Year_RT' in df.columns and 'Month_RT' in df.columns:
        df['date'] = pd.to_datetime(df['Year_RT'].astype(str) + '-' + df['Month_RT'].astype(str).str.zfill(2) + '-01')

    if date_cols:
        for date_col in date_cols:
            df[date_col] = pd.to_datetime(df[date_col])

    return df

# COMMAND ----------

def rename_trends_category(df):
    # Define brand lists
    dog_treats_brands = {"BEGGIN'", "BUSY", "DENTALIFE", "GREENIES", "MILK-BONE", "PUP-PERONI", "ZUKE'S"}
    cat_treats_brands = {"DENTALIFE", "GREENIES", "TEMPTATIONS"}

    # Update categories based on brand and current category
    df.loc[df["brand_group_expanded"].isin(dog_treats_brands) & (df["category"] == "DOG FOOD"), "category"] = "DOG TREATS ONLY"
    df.loc[df["brand_group_expanded"].isin(cat_treats_brands) & (df["category"] == "CAT FOOD"), "category"] = "CAT TREATS ONLY"

    return df

# COMMAND ----------

def add_idv_and_brand_lists(df_grouped, data_sufficiency_check=False):
    """Add IDV and brand list information to the grouped DataFrame."""
    # Load the IDV list
    # idv_list = pd.read_csv("/dbfs/FileStore/shared_uploads/mohammad.fazil@purina.nestle.com/idv_list_v7_8_10_11_2023_iteration6_new.csv")
    idv_list = pd.read_csv(mapping_config["idv_list"], storage_options=storage_options)

    idv_list = idv_list[idv_list['Select'] == "Y"].drop(['Unnamed: 0', 'metric_name', 'Vendor Metric Group', 'Group', 'Common',
                                                        'Blank brand count', 'Blank brands', 'Comments', 'min', 'max', 'Neg Flag'], axis=1)

    df_grouped = pd.merge(df_grouped, idv_list, left_on=['category', 'variable'], right_on=['product_category_idv', 'idv_for_model_corrected'], how='left')
    df_grouped = df_grouped.drop(['idv_for_model_corrected', 'Keep/ Remove', 'Select Null', 'product_category_idv', 'data_source'], axis=1)

    # Add data source
    df_grouped['data_source'] = df_grouped['variable'].str.split("_").str[0].replace({"neilsen": "neilsen_panel", "ratings": "ratings_reviews"})

    rms_metrics = ['rms_acv_selling_rt', 'total_sales_rt', 'equalized_volume_rt', 'total_units_rt', 'average_price_rt', 'market_share_total_sales_rt']

    df_grouped.loc[df_grouped['variable'].isin(rms_metrics), 'data_source'] = 'RMS'

    # Update category for specific brands
    dog_treats_brands = ["BEGGIN'", "BUSY", "DENTALIFE", "GREENIES", "MILK-BONE", "PUP-PERONI", "ZUKE'S"]
    cat_treats_brands = ["DENTALIFE", "GREENIES", "TEMPTATIONS"]

    df_grouped['category'] = np.where(
        (df_grouped['brand_group_expanded'].isin(dog_treats_brands)) & (df_grouped['category'] == "DOG FOOD"), "DOG TREATS ONLY",
        np.where(
            (df_grouped['brand_group_expanded'].isin(cat_treats_brands)) & (df_grouped['category'] == "CAT FOOD"), "CAT TREATS ONLY",
            df_grouped['category']
        )
    )

    # Load the brand list and merge
    # brand_list = pd.read_csv("/dbfs/FileStore/shared_uploads/mohammad.fazil@purina.nestle.com/brand_select_list_v8.csv") #dbfs
    brand_list = pd.read_csv(mapping_config["brand_list"], storage_options=storage_options) #adls

    brand_list = brand_list.rename(columns={"brand_group_expanded": "brand_bl", "category": "category_bl"})
    df_grouped_bl = pd.merge(df_grouped, brand_list, left_on=['brand_group_expanded', 'category'], right_on=['brand_bl', 'category_bl'], how='left')

    df_grouped_bl['brand_list'] = np.where(df_grouped_bl['brand_bl'].notna(), "Y", "N")
    df_grouped_bl['idv_list'] = df_grouped_bl['Select'].fillna("N")
    if data_sufficiency_check==False:
        df_grouped_bl = df_grouped_bl.drop(["Select", "brand_bl", "category_bl"], axis=1)

    return df_grouped_bl

# COMMAND ----------

def harmonized_data_comparison(old_data_path, new_data_path, sales_or_harmonized, start_date="2022-01-01", end_date="2024-06-01"):
    """
    Compares two datasets and returns a comparison report with difference percentages.

    Parameters:
    old_data_path (string): Old data's CSV file path.
    new_data_path (string): New data's CSV file path.
    sales_or_harmonized (string): Enter "sales" or "harmonized"

    Returns:
    DataFrame: Comparison report with difference percentages.
    """

    if sales_or_harmonized == "harmonized":
        # Define drop columns and load old and new data
        drop_cols = ['Unnamed: 0', 'month_h', 'year_h', 'year_rt', 'month_rt', 'brand_rt', 'category_rt',
                    'a_pounds', 'rms_acv_selling_rt', 'rms_a_units', 'total_sales_rt', 'equalized_volume_rt',
                    'total_units_rt', 'average_price_rt', 'market_share_total_sales_rt', 's_units_sold_1010',
                    's_pounds_facts_1010', 'date.1']

        old_hd = load_and_preprocess_data(old_data_path, date_cols=['date'], drop_cols=drop_cols, sales_or_harmonized="harmonized")
        new_hd = load_and_preprocess_data(new_data_path, date_cols=['date'], drop_cols=drop_cols, sales_or_harmonized="harmonized")
        #old_hd ['date'] = pd.to_datetime(df_old['date'])
        #new_hd['date'] = pd.to_datetime(df_new['date'])
        panel_rename_df = pd.read_csv(mapping_config["panel_rename_mapping"], storage_options=storage_options)
        old_hd = old_hd.rename(columns=dict(panel_rename_df))

        # Find the common timeframe
        start_date = max(old_hd['date'].min(), new_hd['date'].min())
        end_date = min(old_hd['date'].max(), new_hd['date'].max())
        old_hd = old_hd[(old_hd['date'] >= start_date) & (old_hd['date'] <= end_date)]
        new_hd = new_hd[(new_hd['date'] >= start_date) & (new_hd['date'] <= end_date)]

        # Filter by date range
        #old_hd = old_hd[(old_hd['date'] >= start_date) & (old_hd['date'] <= end_date)]
        #new_hd = new_hd[(new_hd['date'] >= start_date) & (new_hd['date'] <= end_date)]

        # Melt both DataFrames
        value_vars = [col for col in new_hd.columns if col not in ["Unnamed: 0.1","brand_group_expanded", "category", "date","directions_mean_is_designed_to_meet_my_cat’s_unique_health_needs",
            "directions_rank_1st_is_designed_to_meet_my_cat’s_unique_health_needs",
            "directions_mean_brand_is_doing_the_right_thing_for_the_planet,_people_and_pets",
            "directions_mean_helps_keep_my_dog_occupied/distracted",
            "directions_rank_1st_brand_is_doing_the_right_thing_for_the_planet,_people_and_pets",
            "directions_rank_1st_helps_keep_my_dog_occupied/distracted",
            "neilsen_panel_volume_lbs", "neilsen_panel_voume_lbs"]]
        value_vars1 = [col for col in old_hd.columns if col not in ["Unnamed: 0.1","brand_group_expanded", "category", "date","directions_mean_is_designed_to_meet_my_cat’s_unique_health_needs",
            "directions_rank_1st_is_designed_to_meet_my_cat’s_unique_health_needs",
            "directions_mean_brand_is_doing_the_right_thing_for_the_planet,_people_and_pets",
            "directions_mean_helps_keep_my_dog_occupied/distracted",
            "directions_rank_1st_brand_is_doing_the_right_thing_for_the_planet,_people_and_pets",
            "directions_rank_1st_helps_keep_my_dog_occupied/distracted",
            "neilsen_panel_volume_lbs", "neilsen_panel_voume_lbs"]]

        new_hd_lf = new_hd.melt(id_vars=["brand_group_expanded", "category", "date"], value_vars=value_vars, var_name='variable', value_name='value_new')
        old_hd_lf = old_hd.melt(id_vars=["brand_group_expanded", "category", "date"], value_vars=value_vars1, var_name='variable', value_name='value_old')
        print("new_hd_lf head:", new_hd_lf.head())
        print("old_hd_lf head:", old_hd_lf.head())
        new_hd_lf['value_new'] = pd.to_numeric(new_hd_lf['value_new'], errors='coerce')
        old_hd_lf['value_old'] = pd.to_numeric(old_hd_lf['value_old'], errors='coerce')

        # Merge and calculate percentage difference
        df_joined = pd.merge(
            old_hd_lf, new_hd_lf,
            on=["brand_group_expanded", "category", "date", "variable"]
        )

        # Calculate percentage difference
        df_joined['difference%'] = (
            (df_joined['value_new'] - df_joined['value_old']) / df_joined['value_old']
        ) * 100

        # Merge and calculate percentage difference
        #df_joined = pd.merge(old_hd_lf, new_hd_lf, on=["brand_group_expanded", "category", "date", "variable"])
        #df_joined['difference%'] = (df_joined['value_new'] - df_joined['value_old']) / df_joined['value_old'] * 100

        # Group by and calculate average percentage difference
        df_grouped = df_joined.groupby(["brand_group_expanded", "category", "variable", "date","value_new", "value_old"]).agg({'difference%': 'mean'}).reset_index()
        df_grouped = df_grouped.rename(columns={'difference%': 'avg_difference%'})

        # Add IDV and brand list information
        df_grouped = add_idv_and_brand_lists(df_grouped)

    elif sales_or_harmonized == "sales":
        # Load and preprocess sales data
        old_sales = load_and_preprocess_data(old_data_path, drop_cols=['Unnamed: 0'], sales_or_harmonized="sales")
        new_sales = load_and_preprocess_data(new_data_path, drop_cols=['Unnamed: 0'], sales_or_harmonized="sales")

        # Melt data
        value_vars = ['RMS_ACV_Selling', 'total_sales', 'total_units', 'equalized_volume', 'Average_Price', 'Market_share_total_sales']
        old_sales_lf = pd.melt(old_sales, id_vars=["BRAND_RT", "CATEGORY_RT", "date"], value_vars=value_vars, var_name='variable', value_name='value_old')
        new_sales_lf = pd.melt(new_sales, id_vars=["BRAND_RT", "CATEGORY_RT", "date"], value_vars=value_vars, var_name='variable', value_name='value_new')

        # Merge and calculate percentage difference
        sales_merged = pd.merge(old_sales_lf, new_sales_lf, on=["BRAND_RT", "CATEGORY_RT", "date", "variable"])

        # Debugging: Print column names
        print("Columns in sales_merged DataFrame:", sales_merged.columns)

        # Ensure the correct column names are used
        if 'value_new' in sales_merged.columns and 'value_old' in sales_merged.columns:
            sales_merged['difference%'] = (sales_merged['value_new'] - sales_merged['value_old']) / sales_merged['value_old'] * 100
        else:
            raise KeyError("Expected columns 'value_new' or 'value_old' are not present in the merged DataFrame")

        # Group by and calculate average percentage difference
        sales_grouped = sales_merged.groupby(["BRAND_RT", "CATEGORY_RT", "variable"]).agg({'difference%': 'mean'}).reset_index()
        sales_grouped = sales_grouped.rename(columns={'difference%': 'avg_difference%'})

        return sales_grouped

    return df_grouped


# COMMAND ----------

def generate_filtered_df(data,sales_or_harmonized,output_path,storage_options):

    if sales_or_harmonized == "harmonized":
        #ata['brand'] = data['brand_group_expanded'] + " " + data['category']
        filtered_data = data[(data['idv_list'] == 'Y') &(data['brand_list'] == 'Y')]
        filtered_data =filtered_data.rename(columns={'brand_group_expanded': 'brand' })
        print(filtered_data.columns)
        filtered_data['brand'] = filtered_data['brand'] + " " + filtered_data['category'].str.split().str[0]
        deviation_df = filtered_data[(filtered_data['avg_difference%'] > 5) |
                                        (filtered_data['avg_difference%'] < -5)].copy()
        deviation_df=  deviation_df[['brand', 'date','variable','value_new','value_old', 'avg_difference%']]
        #filtered_data = data[(data['idv_list'] == 'Y') &(data['brand_list'] == 'Y') &
        # (data['avg_difference%'] > 5)][['brand', 'date','variable','value_new','value_old' 'avg_difference%']]
        deviation_df['avg_difference%'] = deviation_df['avg_difference%'].round(1).astype(str) + '%'


    elif sales_or_harmonized == "sales":
        data['brand'] = data['BRAND_RT'] + " " + data['CATEGORY_RT']
        filtered_data = data[(data['avg_difference%'] > 5)][['brand', 'variable', 'avg_difference%']]
        filtered_data['brand'] = filtered_data['brand'].replace({' TREATS ONLY': '', ' LITTER': '', ' FOOD': ''}, regex=True)
    #filtered_data['avg_difference%'] = filtered_data['avg_difference%'].round(2)
    deviation_df.rename(columns={
    'brand': 'Brand',
    'date': 'Date',
    'variable': 'Variable',
    'value_old': 'Previous Refresh',
    'value_new': 'Current Refresh',
    'avg_difference%': '%Deviation'
        }, inplace=True)
    # Calculate summary details
    brands_with_deviation = deviation_df['Brand'].nunique()
    brand_new = filtered_data['brand'].unique()
    total_brands=len(brand_new)
    brands_without_deviation = total_brands - brands_with_deviation
    success_percentage = round((brands_without_deviation / total_brands) * 100, 2)
    fail_percentage = round((brands_with_deviation / total_brands) * 100, 2)
    # summary_data = {
    #     "Total Brands": [total_brands],
    #     "Brands without Deviation": [brands_without_deviation],
    #     "Brands with Deviation": [brands_with_deviation],
    #     "% of Success": [f"{success_percentage}%"],
    #     "% of Fail": [f"{fail_percentage}%"]
    #     }
    # summary_df = pd.DataFrame(summary_data)

    # Generate HTML report
    current_date = datetime.now().strftime("%Y-%m-%d")
    brands_with_deviation_list = deviation_df['Brand'].unique()
    brands_with_deviation_text = ", ".join(brands_with_deviation_list)
    # brands_list = filtered_data['brand'].unique().tolist()
    # summary_data = {
    #     'Count of brands with deviation >5%': [brands_with_deviation_list],
    #     'List of brands': [", ".join(brands_list)]
    # }
    # summary_df = pd.DataFrame(summary_data)

    html_content = f"""
    <html>
        <head>
            <title>Pre Validation Report</title>
        </head>
        <body>
            <h1>Pre Validation Report</h1>
            <p><strong>Date:</strong> {current_date}</p>
            <p><strong>Number of brands with deviation >5%:</strong> {brands_with_deviation}</p>


            <p><strong>Brands with Deviation:</strong> {brands_with_deviation_text}</p>

            <h2>Deviation Summary</h2>
            {deviation_df.to_html(index=False, border=1)}
        </body>
    </html>
    """


    dir_path = '/'.join(output_path.split('/')[:-1])  # Extract directory path
    #   fs = fsspec.filesystem("abfs", **storage_options)

    if not fs.exists(dir_path):
        fs.mkdirs(dir_path)
        print(f"Directory {dir_path} created.")

    # Save the HTML file
    with fs.open(output_path, 'w') as file:
        file.write(html_content)

    print(f"Comparison report saved as {output_path}")

    # Combine summary and filtered data for HTML export

    return filtered_data


# COMMAND ----------

def harmonized_data_sufficiency(data_path, start_date="2019-08-01", end_date="2024-06-01"):
    """
    Compares two datasets and returns a comparison report with difference percentages.

    Parameters:
    old_data_path (string): Old data's CSV file path.
    new_data_path (string): New data's CSV file path.
    sales_or_harmonized (string): Enter "sales" or "harmonized"

    Returns:
    DataFrame: Comparison report with difference percentages.
    """
    # Define drop columns and load old and new data
    drop_cols = ['Unnamed: 0', 'month_h', 'year_h', 'year_rt', 'month_rt', 'brand_rt', 'category_rt',
                    'a_pounds', 'rms_a_units', 's_units_sold_1010', 's_pounds_facts_1010', 'date.1']


    hd = load_and_preprocess_data(data_path, date_cols=['date'], drop_cols=drop_cols, sales_or_harmonized="harmonized")

    # Filter by date range
    hd = hd[(hd['date'] >= start_date) & (hd['date'] <= end_date)]

    # Melt both DataFrames
    value_vars = [col for col in hd.columns if col not in ["brand_group_expanded", "category", "date",
        "directions_mean_is_designed_to_meet_my_cat’s_unique_health_needs",
        "directions_rank_1st_is_designed_to_meet_my_cat’s_unique_health_needs",
        "directions_mean_brand_is_doing_the_right_thing_for_the_planet,_people_and_pets",
        "directions_mean_helps_keep_my_dog_occupied/distracted",
        "directions_rank_1st_brand_is_doing_the_right_thing_for_the_planet,_people_and_pets",
        "directions_rank_1st_helps_keep_my_dog_occupied/distracted",
        "neilsen_panel_volume_lbs", "neilsen_panel_voume_lbs"]]

    hd_lf = hd.melt(id_vars=["brand_group_expanded", "category", "date"], value_vars=value_vars)

    # Add IDV and brand list information
    hd_idv_bl_lf = add_idv_and_brand_lists(hd_lf, data_sufficiency_check=True)

    # Group by brand, category, and variable
    data_availability_hd = hd_idv_bl_lf.groupby(["brand_group_expanded", "category", "variable", "data_source","brand_list","idv_list"]).agg(
        non_null_count = pd.NamedAgg(column="value", aggfunc=lambda x: x.notnull().sum()),  # Count of non-null data points
        total_count = pd.NamedAgg(column="value", aggfunc="count"),  # Total count of data points
        availability_percentage = pd.NamedAgg(column="value", aggfunc=lambda x: x.notnull().mean() * 100),  # Percentage of non-null values
        start_date = pd.NamedAgg(column="date", aggfunc=lambda x: x.loc[x.notnull()].min()),  # Min date where value is not null
        end_date = pd.NamedAgg(column="date", aggfunc=lambda x: x.loc[x.notnull()].max())  # Max date where value is not null
    ).reset_index()

    return data_availability_hd

# COMMAND ----------

def harmonized_data_extraction(time_granularity = refresh_config["time_granularity"]):
    if time_granularity == "weekly":
        harmonized_df = spark.sql(f"select * from brand_hub_gold.harmonized_data_weekly_view")
    elif time_granularity == "monthly":
        harmonized_df = spark.sql(f"select * from brand_hub_gold.harmonized_data_view")
    harmonized_df_corrected_metric_name = harmonized_df.withColumn("metric_name_new",concat_ws("_",'data_source','metric_type','metric_name','user_segment'))
    harmonized_df_corrected_metric_name = harmonized_df_corrected_metric_name.withColumn("metric_name_new",lower("metric_name_new")).withColumn("metric_name_new",translate('metric_name_new', " ", "_"))

    harmonized_df_corrected_metric_name = harmonized_df_corrected_metric_name.groupBy('brand_group_expanded','category','date').pivot('metric_name_new').avg("metric_value")

    harmonized_df_corrected_metric_name = harmonized_df_corrected_metric_name.orderBy(['brand_group_expanded','category','date'])
    harmonized_df_corrected_metric_name = harmonized_df_corrected_metric_name.toPandas()
    return harmonized_df_corrected_metric_name

# COMMAND ----------

def harmonized_data_prep(input_config, output_config, mapping_config, refresh_config, storage_options):

    col_name_rename_df = pd.read_csv(mapping_config["metrics_rename_mapping"], storage_options=storage_options)
    inverse_logic_df = pd.read_csv(mapping_config["inverse_logic_mapping"], storage_options=storage_options)

    # if spark.conf.get("spark.databricks.clusterUsageTags.clusterName", "Cluster name not found").startswith("NPUS-PR-"):
    if platform_type == 'databricks':
        if spark.conf.get("spark.databricks.clusterUsageTags.clusterName", "Cluster name not found").startswith("NPUS-PR"):
            nielsen_rms_data = nielsen_rms_data_extraction(dbs_sql_hostname,dbs_sql_http_path,dbs_sql_token)
        else:
            # nielsen_rms_data = pd.read_csv(input_config["current_sales_data"], storage_options=storage_options)
            nielsen_rms_data = pd.read_csv("abfss://restricted-dataoperations@npusdvdatalakesta.dfs.core.windows.net/staging/cmi_brand_hub/fazil/rms_tenten_monthly_19_9_24_blue.csv", storage_options=storage_options)
            nielsen_rms_data.columns = nielsen_rms_data.columns.str.lower()
            # Ensure the correct column names are used
            nielsen_rms_data['date'] = pd.to_datetime(
                nielsen_rms_data[['year_rt', 'month_rt']].rename(columns={'year_rt': 'year', 'month_rt': 'month'}).assign(day=1)
            )

        # if refresh_config["time_granularity"] == "weekly":
        #     nielsen_rms_data['date'] = nielsen_rms_data['date'].str.split('T').str[0]
        #     nielsen_rms_data['date'] = pd.to_datetime(nielsen_rms_data['date'], utc=False)
        #     # nielsen_rms_data["average_price"] = nielsen_rms_data["total_sales"]/nielsen_rms_data["total_units"]
        #     # nielsen_rms_data.rename(columns={
        #     #         'brand_rt': 'brand_group_expanded',
        #     #         'category_rt': 'category',
        #     #         'week_ending': 'date'
        #     #     }, inplace=True)
        # else:
        # nielsen_rms_data['date'] = nielsen_rms_data['date'].str.split('T').str[0]
            nielsen_rms_data["date"] = pd.to_datetime(nielsen_rms_data["date"], utc=False)
            nielsen_rms_data.rename(columns={"eq_volume" : "equalized_volume","brand_rt":"brand_group_expanded","category_rt":"category"}, inplace=True)
            nielsen_rms_data["average_price"] = nielsen_rms_data["total_sales"]/nielsen_rms_data["total_units"]

    # if refresh_config["platform"] == "databricks":
    #     harmonized_df = harmonized_data_extraction(time_granularity = refresh_config["time_granularity"])
    # elif refresh_config["platform"] == "local":
    #     harmonized_df = pd.read_csv(input_config["harmonized_data"])
    else:
        nielsen_rms_data = pd.read_csv(input_config['current_sales_data'], storage_options=storage_options)
        nielsen_rms_data.columns = nielsen_rms_data.columns.str.lower()
        # Ensure the correct column names are used
        nielsen_rms_data['date'] = pd.to_datetime(
        nielsen_rms_data[['year_rt', 'month_rt']].rename(columns={'year_rt': 'year', 'month_rt': 'month'}).assign(day=1))


        nielsen_rms_data["date"] = pd.to_datetime(nielsen_rms_data["date"], utc=False)
        nielsen_rms_data.rename(columns={"eq_volume" : "equalized_volume","brand_rt":"brand_group_expanded","category_rt":"category"}, inplace=True)
        nielsen_rms_data["average_price"] = nielsen_rms_data["total_sales"]/nielsen_rms_data["total_units"]


    harmonized_df = pd.read_csv(input_config['current_harmonized_data'], storage_options=storage_options)
    harmonized_df.rename(columns={"Date":"date"},inplace=True)
    harmonized_df['date'] = pd.to_datetime(harmonized_df['date'], utc=False)

    harmonized_df.to_csv(f"{output_config['raw_input_data']}", index=False,storage_options=storage_options)

    req_cols = pd.read_csv(mapping_config["idv_list"], storage_options=storage_options)

    brand_category_to_run = pd.read_csv(mapping_config["brand_list"], storage_options=storage_options)

    # dashboard_metric_names_mapping = pd.read_excel(mapping_config["dashboard_metric_names_mapping"],storage_options=storage_options)

    # price_class_mapping = pd.read_csv(mapping_config["price_class_mapping"], storage_options=storage_options)


    # col_name_rename_df = col_name_rename_df.dropna(subset='New column name mapping',how='any')
    col_name_rename_df = col_name_rename_df.dropna(subset='DataHub Name',how='any')

    # rms_harmonized_data = pd.merge(harmonized_df, nielsen_rms_data, on=['brand_group_expanded','category','date'],how = 'left')
    rms_harmonized_data = harmonized_df

    df = rms_harmonized_data
    print("columns_presents:", 'ratings_reviews_review' in df.columns.tolist())
    print("columns_presents:", 'ratings_reviews_review_' in df.columns.tolist())
    df = df.rename(columns=dict(col_name_rename_df.values))
    print("columns_presents:", 'ratings_reviews_review_count' in df.columns.tolist())
    # df = df.rename(columns=dict(zip(col_name_rename_df['From prod harmonized view'],col_name_rename_df['New column name mapping'])))


    df = create_percentage_columns(df)

    df = create_inverse_metrics(df,inverse_logic_df)

    df = create_share_metrics(df)

    df = rename_trends_category(df)

    processed_harmonized_data = df.copy()
    # processed_harmonized_data= pd.read_csv(input_config["harmonized_data"], storage_options=storage_options)
    #display(processed_harmonized_data)

    #df=processed_harmonized_data.copy()

    processed_harmonized_data.to_csv(f"{output_config['processed_input_data']}", index=False, storage_options=storage_options)

    nielsen_rms_data.to_csv(f"{output_config['processed_sales_data']}", index=False, storage_options=storage_options)

    return processed_harmonized_data

# COMMAND ----------

def pre_validation(input_config,output_config,mapping_config,refresh_config,storage_options):

    processed_harmonized_data =harmonized_data_prep(input_config,output_config,mapping_config,refresh_config, storage_options)
    print("harmonized_data saved...")
    # data_comp_harmonized=harmonized_data_comparison(old_data_path = input_config["prev_harmonized_data"], new_data_path = input_config["current_harmonized_data"], sales_or_harmonized = "harmonized")

    # data_comp_sales=harmonized_data_comparison(old_data_path = f"abfss://{dataoperations_name}@{account_name}.dfs.core.windows.net/staging/cmi_brand_hub/fazil/pre_validation_data/old_refresh_sample/data/sales_data.csv", new_data_path = f"abfss://{dataoperations_name}@{account_name}.dfs.core.windows.net/staging/cmi_brand_hub/fazil/pre_validation_data/new_refresh_sample/data/sales_data.csv", sales_or_harmonized = "sales")

    # generate_filtered_df(data_comp_harmonized, sales_or_harmonized = "harmonized",output_path=output_config["updated_scorecard"]["pre_validation_report"], storage_options = storage_options)


    # data_sufficiency_df = harmonized_data_sufficiency(data_path = f"abfss://{dataoperations_name}@{account_name}.dfs.core.windows.net/staging/cmi_brand_hub/fazil/pre_validation_data/new_refresh_sample/data/harmonized_data.csv")

    # data_sufficiency_df.to_csv(f"/pre_validation_data_sufficiency.csv", index=False, storage_options = storage_options)

# COMMAND ----------

# pre_validation(input_config,output_config,mapping_config,refresh_config,storage_options)
