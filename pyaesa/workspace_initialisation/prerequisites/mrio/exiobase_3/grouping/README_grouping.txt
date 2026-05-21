EXIOBASE grouping prerequisite guide
===================================

Purpose
-------

This folder stores packaged MRIO grouping templates used by:
- process_mrio(...)
- deterministic_asocc(...)
- deterministic_io_lca(...)
- downstream grouped aCC and ASR workflows that reuse grouped processed MRIOs

This workspace folder stores the EXIOBASE grouping CSVs available to the
current project under:
- `data_raw/mrio/exiobase_3/grouping/`

The normal user workflow is to copy grouping CSVs in this workspace
folder, then call package functions with the matching
`group_reg`, `group_sec`, and `group_version` arguments.

Available packaged templates
----------------------------

Region grouping template and packaged grouping versions in this folder:
- group_reg_template.csv
- group_reg_eu27.csv
- group_reg_world.csv

Sector grouping templates and examples depend on the EXIOBASE table system:
- ixi/group_sec_template.csv
- ixi/group_sec_elec.csv
- ixi/group_sec_oecd_d.csv
- pxp/group_sec_template.csv

The ixi/group_sec_elec.csv example groups electricity sectors together.
The ixi/group_sec_oecd_d.csv example groups electricity, gas, and water
sectors to match OECD ICIO sector D resolution.

Packaged EXIOBASE sector definition guide
-----------------------------------------

The parent EXIOBASE prerequisite folder also includes:
- `data_raw/mrio/exiobase_3/sector_classification.xlsx`

Use that detailed EXIOBASE guide when selecting the sector that matches a
study objective or when preparing sector grouping CSVs:
- sheet `Product descriptions` gives detailed EXIOBASE product definitions

How custom grouping works
-------------------------

1. Copy the relevant template CSV.
2. Rename it to:
   - group_reg_<name>.csv for regional grouping
   - group_sec_<name>.csv for sector grouping
3. Fill the `grouped_mrio` column with the target grouped label.
4. Use the same `<name>` as `group_version` in package calls.

Public argument contract
------------------------

- group_reg=True enables regional grouping
- group_sec=True enables sector grouping
- group_version="<name>" selects the custom grouping CSV stem

If both region and sector grouping are enabled, both grouping CSVs should use
the same `<name>` token.

Canonical template columns
--------------------------

Region template:
- original_classification_full_name
- original_classification
- grouped_mrio

Sector template:
- original_classification
- grouped_mrio
