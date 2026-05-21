Grouped and aggregate LCIA CoV guide
====================================

LCIA uncertainty uses the same region and s_p sector labels written in result
tables.

Region grouping
---------------

When process_mrio(...) uses region grouping and the grouped outputs are later
used with LCIA uncertainty, provide a matching region CoV file in this folder.

File name:
- reg_cbca_covs_group_<group_version>.csv

Required columns:
- exio_code: grouped MRIO region label
- cov: coefficient of variation used by LCIA uncertainty

The exio_code labels must match the grouped_mrio labels in the matching
data_raw/mrio/<source>/grouping/group_reg_<group_version>.csv file and must
include World. If a grouped label is unchanged from the original EXIOBASE
region list, copy its CoV from reg_cbca_covs.csv. If a grouped label aggregates
several regions, explicitly choose or compute the CoV.

Aggregate indices
-----------------

When a public function uses aggreg_indices=True with LCIA uncertainty, the CoV
keys must match the aggregate labels in the function output domain.

For region CoVs:
- If group_reg=False, add aggregate region labels to
  reg_cbca_covs_aggreg_indices.csv.
- If group_reg=True, add aggregate grouped region labels to
  reg_cbca_covs_group_<group_version>_aggreg_indices.csv.

An aggregate label is the selected output labels sorted and joined with comma
and space, for example:
- FR, US
- Electricity, Steel

For sector CoVs, keep the Rodrigues sector CoV codes from sec_cbca_covs.csv.
In the public function sector_cov_mapping argument, use the output s_p labels as
keys. With group_sec=True or aggreg_indices=True, those keys are the grouped
or aggregate sector labels in the function output domain.

Example sector mapping syntax:
- If aggreg_indices=True and the selected output s_p labels are A and B, use
  sector_cov_mapping={"A, B": "Electricity"} when Electricity is the sector
  CoV code selected from sec_cbca_covs.csv.
