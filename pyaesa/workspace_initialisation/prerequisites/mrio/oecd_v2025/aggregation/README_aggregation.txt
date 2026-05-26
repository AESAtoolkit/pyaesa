OECD aggregation prerequisite guide
================================

Purpose
-------

This folder stores packaged OECD aggregation templates used by:
- process_mrio(...)
- deterministic_asocc(...)
- deterministic_io_lca(...)
- downstream aCC and ASR workflows that reuse processed MRIOs in a custom classification

This workspace folder stores the OECD MRIO aggregation and disaggregation CSVs available to the current
project under:
- `data_raw/mrio/oecd_v2025/aggregation/`

The normal user workflow is to copy MRIO aggregation and disaggregation CSVs in this workspace
folder, then call package functions with the matching
`agg_reg`, `agg_sec`, and `agg_version` arguments.

Available packaged templates
----------------------------

Region aggregation template and packaged classification versions:
- agg_reg_template.csv
- agg_reg_eu27.csv
- agg_reg_world.csv

Sector aggregation templates:
- agg_sec_template.csv

How custom MRIO aggregation and disaggregation works
-------------------------------------------

1. Copy the relevant template CSV.
2. Rename it to:
   - agg_reg_<name>.csv for regional MRIO aggregation and disaggregation
   - agg_sec_<name>.csv for sector MRIO aggregation and disaggregation
3. Fill the `aggregated_mrio` column with the target MRIO label.
   - To keep a native label unchanged, repeat the original label.
   - To aggregate labels, assign several original labels to the same target label.
   - To disaggregate one original label into several target labels, add a
     `weight` column and repeat the original label once for each target label.
     Weights must be finite, non negative, and sum to 1 for each original label.
4. Use the same `<name>` as `agg_version` in package calls.

Public argument contract
------------------------

- agg_reg=True enables regional MRIO aggregation and disaggregation
- agg_sec=True enables sector MRIO aggregation and disaggregation
- agg_version="<name>" selects the custom MRIO aggregation and disaggregation CSV stem

If both region and sector mappings are enabled, both mapping CSVs should use
the same `<name>` token.

Canonical template columns
--------------------------

Region template:
- original_classification_full_name
- original_classification
- aggregated_mrio

Sector template:
- original_classification_full_name
- original_classification
- aggregated_mrio
