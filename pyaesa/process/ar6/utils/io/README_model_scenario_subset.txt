Model-scenario subset guide for deterministic_ar6_cc(...)

Files written by process_ar6(...) in this folder:
- model_scenario_subset__template.csv
- README_model_scenario_subset.txt

How to create a subset file:
1. Open model_scenario_subset__template.csv.
2. Keep only the rows you want to retain.
3. Save the filtered file in this same folder as:
   model_scenario_subset__<your_version_name>.csv
4. Pass subset_version="<your_version_name>" into deterministic_ar6_cc(...).

Required columns in the subset CSV:
- model
- scenario

Helper columns available in the template:
- category
- ssp_scenario

How category and SSP selection interact with the subset file:
- The subset CSV is a model-scenario whitelist.
- You may keep rows covering several categories and several SSP scenarios in
  the same subset file.
- Later AR6 CC calls may still pass category=... and ssp_scenario=....
- The runtime keeps only rows that satisfy all requested filters together:
  - the requested AR6 variable family
  - the requested category filter, if any
  - the requested SSP filter, if any
  - membership in the subset model-scenario whitelist
- The category and ssp_scenario columns in the template are provided to help
  manual filtering and review. They do not add extra required columns to the
  subset contract beyond the model-scenario keys.

The subset file is read from the matching process_ar6 folder for the same
study period and harmonization scope.
