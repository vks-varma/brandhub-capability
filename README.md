# brandhub-capability

 %md
 **Guidelines for Writing Production Grade for BHC/MMX**

 To ensure consistency and maintain best practices, please follow these steps when working on brand hub tasks:
 (you will better understand these, once you have looked at latest BHC codes)

 Possible Conditions
   - time_granularity = 'weekly' or 'monthly'
   - refresh_type = 'model_scoring' or 'model_refresh'

 1. **Time Granularity**
    - Always set `time_granularity` to either `monthly` or `weekly`.
    - Use lowercase letters with underscores as separators (e.g., `monthly`, `weekly`).

 2. **File Configuration**
    - Declare all files created in a previous stage and used in the next stage under `input_config`.
    - Any common files that remain unchanged (e.g., mappings) should be declared under `mapping_config`.
    - Place all output files under `output_config`.
    - Pass `storage_options` as an additional layer for file handling.

 3. **Refresh Configuration**
    - Place all refresh-related configurations under `refresh_config` (e.g., `start_date`, `end_date`, `time_granularity`, `scoring`, `modelling`).

 4. **Feature Engineering and Additional Configurations**
    - If additional configurations are needed (e.g., for data preparation, feature engineering, or modelling), feel free to add them.
    - Append new configurations at the end of the flow, maintaining the order: `input_config`, `output_config`, `mapping_config`, `refresh_config`, and `feat_eng_config`.

 5. **Code Structure**
    - Ensure that functions do not exceed 100–150 lines of code.
    - Avoid using loops within functions whenever possible; use vectorization techniques instead.
    - For running operations across multiple entities (e.g., brands or categories), use loops to call the function externally rather than embedding loops within the function.

 6. **Intermediate and Final Data Guidelines**
    - Ensure the following structure for data:
      - All keys should be placed on the left, ordered hierarchically (e.g., `vendor x brand x sub_brand x category`).
      - Independent variables (IDVs) should be on the right.
      - Date column should be seperate these (e.g., `brand x category x date x eq_volume x amazon_spends ...`).

 7. **Key or Group Variables**
    - For tasks involving model iterations (e.g., running a model for `brand x category x pillar`), create a key or `group_var`.
    - Pass this variable into the code to enable automatic configuration for future iterations.
    - If implementing this is overly complex, it can be skipped.

 By following these guidelines, we can ensure clarity, consistency, and scalability in our processes.

