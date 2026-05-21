Add custom static carrying capacities
=====================================

Purpose
-------

This folder stores static carrying capacity CSVs used by:
- deterministic_acc(...)
- deterministic_asr(...)
- external LCA validation against bundled `(impact, impact_unit)` pairs

This workspace folder stores the static carrying capacity CSVs available to
the current project under `data_raw/carrying_capacities/`.

This guide is only needed when adding carrying capacity data for a custom
LCIA method or custom impact vocabulary beyond the files shipped with the
package. Users who run package shipped methods can use the bundled CSV files
directly and do not need to edit this folder.

The custom method workflow is:
- work in that copied user side folder
- add or edit the CSV there
- call package functions with the selected `lcia_method` name

What to copy for a custom method
--------------------------------

Add one new CSV named:
- <lcia_method>_cc_steady_state.csv

Use exactly one of the packaged model files as the starting structure:
- name_lcia_cc_steady_state_template.csv
- name_lcia_cc_steady_state_planetary_boundary_template.csv

Canonical column contracts
--------------------------

Standard schema:
- impact_full_name
- impact
- impact_unit
- optional_global_normalisation_factor
- min_cc
- max_cc

Planetary boundary schema:
- Planetary boundary
- Control variable
- impact
- impact_unit
- min_cc
- max_cc

The package normalizes both schemas to the same internal runtime contract.

When to use each schema
-----------------------

Use the standard schema when a plain impact vocabulary is sufficient.

Use the planetary boundary schema only when the packaged file should preserve
explicit `Planetary boundary` and `Control variable` labels for contributor
and user facing traceability.

Custom method workflow
----------------------

User workflow:
1. Run `set_workspace(...)` for the target workspace.
2. Go to `data_raw/carrying_capacities/` in that workspace.
3. Copy one template CSV there.
4. Rename it to `<lcia_method>_cc_steady_state.csv`.
5. Fill `impact`, `impact_unit`, and carrying capacity values.
6. Use the same `lcia_method` name in package calls.

Optional contributor route:
1. If the new file should become available to all users by default, submit it
   to `pyaesa` through a pull request.
2. Add a matching characterization matrix in:
   `prerequisites/mrio/exiobase_3/lcia/characterization_factors_matrices/`
3. If the method needs historical responsibility allocation, add a
   responsibility period table in:
   `prerequisites/mrio/exiobase_3/lcia/responsibility_periods/`.
