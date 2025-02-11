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

