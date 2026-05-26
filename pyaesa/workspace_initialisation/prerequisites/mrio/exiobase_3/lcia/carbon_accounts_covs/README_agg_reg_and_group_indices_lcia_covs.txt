agg_reg and group_indices LCIA CoV guide
========================================

LCIA uncertainty uses the same region and s_p sector labels written in result
tables.

agg_reg scope: processed MRIO region labels
-------------------------------------------

When process_mrio(...) uses ``agg_reg=True`` and those outputs are later used
with LCIA uncertainty, provide a matching region CoV file in this folder.

File name:
- reg_cbca_covs_agg_<agg_version>.csv

Required columns:
- exio_code: MRIO region label after agg_reg MRIO aggregation and disaggregation
- cov: coefficient of variation used by LCIA uncertainty

The exio_code labels must match the aggregated_mrio labels in the matching
data_raw/mrio/<source>/aggregation/agg_reg_<agg_version>.csv file and must
include World. If an agg_reg label is unchanged from the original EXIOBASE
region list, copy its CoV from reg_cbca_covs.csv. If an agg_reg label combines
or disaggregates regions, explicitly choose or compute the CoV.

group_indices scope: public output grouping labels
--------------------------------------------------

When a public function uses group_indices=True with LCIA uncertainty, the CoV
keys must match the combined labels in the function output domain.

When group_indices=True, provide CoV entries keyed by the combined output labels
because the public result row is the selected labels summed after the function
calculation has been performed.

For region CoVs:
- If agg_reg=False, add combined region output labels to
  reg_cbca_covs_group_indices.csv.
- If agg_reg=True, add combined region output labels in the custom
  MRIO aggregation and disaggregation to
  reg_cbca_covs_agg_<agg_version>_group_indices.csv.

A combined label is the selected output labels sorted and joined with comma and
space. For example:
- FR, US
- Electricity, Steel

For sector CoVs, keep the Rodrigues sector CoV codes from sec_cbca_covs.csv.
In the public function sector_cov_mapping argument, use the output s_p labels as
keys. With agg_sec=True, those keys are the sector labels after agg_sec
MRIO aggregation and disaggregation. With group_indices=True, those keys are the combined
sector output labels in the function output domain.

Example sector mapping syntax:
- If group_indices=True and the selected output s_p labels are A and B, use
  sector_cov_mapping={"A, B": "Electricity"} when Electricity is the sector
  CoV code selected from sec_cbca_covs.csv.
