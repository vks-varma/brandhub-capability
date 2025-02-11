# Introduction to BHC Capability

## Overview
BHC Capability is a data-driven framework designed to analyze and model hierarchical data structures such as **Category-Brand**, **Country-Vendor-Category-Brand**, and other business dimensions. The project utilizes **harmonized data** and an **IDV list** containing features that map to specific pillars, enabling advanced data processing and trend analysis.

## Data Structure
The project operates on a hierarchical data model, where data is categorized at multiple levels, such as:
- **Category → Brand**
- **Country → Vendor → Category → Brand**

## Pillar Creation Process
To generate meaningful insights, we create **pillars** by applying transformations and modeling techniques on the harmonized dataset. The process follows these key steps:

### 1. **Feature Mapping to Pillars**
- The IDV list contains various features that are mapped to predefined pillars.
- Feature transformation methods include:
  - **Custom Scaling**
  - **Standard Scaling**
  - **Min-Max Scaling**
  - **Imputation** for handling missing values

### 2. **Weightage Calculation**
- We utilize **Confirmatory Factor Analysis (CFA)** and **Random Forest SHAP values** to determine the importance of each feature in relation to its pillar.
- These weightages are merged to derive a final weight for each metric within the pillar structure.

### 3. **Pillar Creation**
- Using the derived weightages, we construct the **pillar values** from the harmonized dataset.
- The processed pillar data is then used for further modeling.

## Importance Modeling
Once the pillars are created, we conduct an **importance analysis** using historical trend data:
- We analyze past pillar data trends.
- A **hierarchical importance model** is built for each level in the hierarchy.

## Conclusion
BHC Capability provides a structured approach to understanding the relationships between features, pillars, and hierarchical business structures. By leveraging **CFA, SHAP values, and importance modeling**, it enables data-driven decision-making and deeper insights into key business metrics.


### Steps to Build HTML Documentation

1. **Navigate to the Documentation Directory**
   Open a terminal or command prompt and change to the root directory of your Sphinx documentation (where the `Makefile` or `make.bat` is located).

   ```sh
   cd /path/to/your/docs
   ```

2. **Run the Build Command**
   If you are using Linux or macOS, run:

   ```sh
   make html
   ```

   If you are using Windows, run:

   ```sh
   .\make.bat html
   ```

3. **View the Generated Documentation**
   Once the build process is complete, open the generated HTML documentation by navigating to:

   ```
   build/html/index.html
   ```

   You can open it in a web browser to view the documentation.

### Troubleshooting

- If there are errors, ensure all dependencies are installed:

  ```sh
  pip install -r requirements.txt
  ```

- If the documentation does not update correctly, try cleaning the build directory before rebuilding:

  ```sh
  make clean
  make html
  ```

This process should successfully generate and update your Sphinx documentation. 🚀