# Databricks notebook source
def post_validation(input_config,output_config):
    def scaled_scores_data_comparison(old_data_path, new_data_path, output_path, storage_options):
        """
        Compares two datasets and generates a post-validation report.
        
        Parameters:
        old_data_path (string): Old data's CSV file path.
        new_data_path (string): New data's CSV file path.
        output_path (string): Output path for the comparison report.
        storage_options (dict): Storage options for accessing the files.
        
        Returns:
        summary_df (DataFrame): Summary of brands with and without deviations.
        deviation_df (DataFrame): Details of deviations per brand and pillar.
        """
        # Read data
        df_old = pd.read_csv(old_data_path, storage_options=storage_options)
        df_new = pd.read_csv(new_data_path, storage_options=storage_options)
        df_old['date'] = pd.to_datetime(df_old['date'])
        df_new['date'] = pd.to_datetime(df_new['date'])
        
        # Filter metric_type == True
        df_old = df_old[df_old["metric_type"] == True]
        df_new = df_new[df_new["metric_type"] == True]
        
        # Compare brands
        brand_old = df_old['brand'].unique()
        brand_new = df_new['brand'].unique()
        new_brands_added = set(brand_new) - set(brand_old)
        if new_brands_added:
            print("New brand(s) added to the current refresh:", ", ".join(new_brands_added))

        
        # Find the common timeframe
        start_date = max(df_old['date'].min(), df_new['date'].min())
        end_date = min(df_old['date'].max(), df_new['date'].max())
        df_old_common = df_old[(df_old['date'] >= start_date) & (df_old['date'] <= end_date)]
        df_new_common = df_new[(df_new['date'] >= start_date) & (df_new['date'] <= end_date)]
        
        # Rename columns for merging
        df_old_common.rename(columns={'scaled_scores': 'old_score'}, inplace=True)
        df_new_common.rename(columns={'scaled_scores': 'new_score'}, inplace=True)
        
        # Merge dataframes for comparison
        df = df_new_common.merge(df_old_common, on=['date', 'brand', 'category', 'equity_pillar'])
        df['difference'] = df['new_score'] - df['old_score']
        df['difference_percentage'] = (df['difference'] / df['old_score']) * 100
        df['brand'] = df['brand'] + " " + df['category'].str.split().str[0]
        #brand_old = df_old['brand'].nunique()
        brand_new = df['brand'].unique()
        df = df[['brand', 'date', 'equity_pillar', 'old_score', 'new_score', 'difference_percentage']]
        
        # Filter for significant deviations (> 5%)
        #deviation_df = df[df['difference_percentage'].abs() > 5].copy()
        # Filter for significant deviations (> 5% or < -5%)
        deviation_df = df[(df['difference_percentage'] > 5) | (df['difference_percentage'] < -5)].copy()
        deviation_df['difference_percentage'] = deviation_df['difference_percentage'].round(1).astype(str) + '%'
        
        # Rename columns for the report
        deviation_df.rename(columns={
            'brand': 'Brand',
            'date': 'Date',
            'equity_pillar': 'Pillar',
            'old_score': 'Previous Refresh',
            'new_score': 'Current Refresh',
            'difference_percentage': '%Deviation'
        }, inplace=True)
        
        # Summary statistics
        total_brands = len(brand_new)
        brands_with_deviation = deviation_df['Brand'].nunique()
        brands_without_deviation = total_brands - brands_with_deviation
        success_percentage = round((brands_without_deviation / total_brands) * 100, 2)
        fail_percentage = round((brands_with_deviation / total_brands) * 100, 2)
        summary_data = {
            "Total Brands": [total_brands],
            "Brands without Deviation": [brands_without_deviation],
            "Brands with Deviation": [brands_with_deviation],
            "% of Success": [f"{success_percentage}%"],
            "% of Fail": [f"{fail_percentage}%"]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Generate HTML report
        current_date = datetime.now().strftime("%Y-%m-%d")
        brands_with_deviation_list = deviation_df['Brand'].unique()
        brands_with_deviation_text = ", ".join(brands_with_deviation_list)
        
        html_content = f"""
        <html>
            <head>
                <title>Post Validation Report</title>
            </head>
            <body>
                <h1>Post Validation Report</h1>
                <p><strong>Date:</strong> {current_date}</p>
                
                <h2>Brands Summary</h2>
                {summary_df.to_html(index=False, border=1)}
                
                <p><strong>Brands with Deviation:</strong> {brands_with_deviation_text}</p>
                
                <h2>Deviation Summary</h2>
                {deviation_df.to_html(index=False, border=1)}
            </body>
        </html>
        """
        
        # Ensure directory exists
        #d# Ensure directory exists
        dir_path = '/'.join(output_path.split('/')[:-1])  # Extract directory path
    #   fs = fsspec.filesystem("abfs", **storage_options)
        
        if not fs.exists(dir_path):
            fs.mkdirs(dir_path)
            print(f"Directory {dir_path} created.")
        
        # Save the HTML file
        with fs.open(output_path, 'w') as file:
            file.write(html_content)
        
        print(f"Comparison report saved as {output_path}")
        
        return summary_df
    print("post val 1- staging_output_path:",staging_output_path)
    scaled_scores_data_comparison(
        old_data_path=input_config["old_scores_data"],
        new_data_path = output_config["updated_scorecard"]["updated_summary"],
        output_path = output_config["updated_scorecard"]["post_validation_report"],
        # new_data_path=f"abfss://{dataoperations_name}@{account_name}.dfs.core.windows.net/staging/cmi_brand_hub/scorecard_refresh/2024-11-01/scorecard/brand_health_scorecard_summary_updated.csv",
        # output_path=f"abfss://{dataoperations_name}@{account_name}.dfs.core.windows.net/staging/cmi_brand_hub/scorecard_refresh/2024-11-01/scorecard/scaled_scores_data_comparison_eq.html",
        storage_options={'tenant_id': tenant_id, 'client_id': client_id, 'client_secret': client_secret}
    )
    output_path = output_config["updated_scorecard"]["post_validation_report"]
    # output_path=f"abfss://{dataoperations_name}@{account_name}.dfs.core.windows.net/staging/cmi_brand_hub/scorecard_refresh/2024-11-01/scorecard/scaled_scores_data_comparison_eq.html"
    html_file_path=output_path

    # Read the HTML file
    #fs = fsspec.filesystem("abfs", **storage_options)
    # with fs.open(html_file_path, "r") as f:
    #     html_content = f.read()

    # # Render the HTML content in Databricks
    # displayHTML(html_content)

    # local_file_path ='/Workspace/app/brandhub/data_modelling/pipeline_v1_with_func/post_validation_modeling_report.html'

    # # Open the HTML file from Azure Blob Storage and save it to the local path
    # with fs.open(html_file_path, "r") as f:
    #     html_content = f.read()

    # # Write the content to a local file
    # with open(local_file_path, "w") as local_file:
    #     local_file.write(html_content)

    # Display the local file path for downloading
    # local_file_path
