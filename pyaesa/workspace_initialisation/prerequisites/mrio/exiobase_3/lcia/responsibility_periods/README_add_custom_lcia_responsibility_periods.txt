Add custom EXIOBASE LCIA responsibility periods
===============================================

Purpose
-------

This folder stores optional responsibility period tables used only for
historical responsibility allocation methods.

This workspace folder stores the optional EXIOBASE responsibility period
tables available to the current project under:
- `data_raw/mrio/exiobase_3/lcia/responsibility_periods/`

This guide is only needed when adding a responsibility period table for a
custom LCIA method beyond the files shipped with the package. Users who run
package shipped historical responsibility methods can use the bundled CSV files
directly and do not need to edit this folder.

The custom method workflow is to copy one CSV template in this workspace
folder and then use the same `lcia_method` name in package calls.

What to copy for a custom method
--------------------------------

Add one CSV per method:
- <lcia_method>_rps.csv

Use exactly one of the packaged model files as the starting structure:
- name_lcia_rps_template.csv
- name_lcia_rps_planetary_boundary_template.csv

Canonical column contract
-------------------------

Standard schema:
- impact_full_name
- impact_parent
- impact
- responsibility_period_years

Planetary boundary schema:
- Planetary boundary
- Control variable
- impact_parent
- impact
- impact_duration_years
- responsibility_period_years
- source
- comment

When to use each schema
-----------------------

Use the standard schema when the method only needs a generic
`impact_full_name` label.

Use the planetary boundary schema only when the responsibility table should
preserve explicit `Planetary boundary` and `Control variable` vocabulary plus
supporting duration or citation notes in the packaged file.

Guidance
--------

- Reuse the same `impact_parent` and `impact` structure as the paired
  characterization matrix.
- Define the actual responsibility period on each concrete `impact` row.
- Keep the file stem equal to the public `lcia_method` token.
- This table does not define `impact_unit`; unit ownership stays in the
  characterization matrix and bundled static carrying capacity CSV.
- Optional metadata columns such as `impact_duration_years`, `source`, and
  `comment` may also be added to the standard schema when they help document
  one method's scientific interpretation.

When a custom responsibility period file is needed
-------------------------------------------------

Add an RPs file only when the workflow should support historical responsibility
allocation. Methods that never use that allocation route do not need an
`_rps.csv` file.

Custom method workflow
----------------------

User workflow:
1. Run `set_workspace(...)` for the target workspace.
2. Go to `data_raw/mrio/exiobase_3/lcia/responsibility_periods/`.
3. Copy one template CSV there.
4. Rename it to `<lcia_method>_rps.csv`.
5. Fill the responsibility period rows.
6. Use the same `lcia_method` name in package calls.

Optional contributor route:
1. If the new method should become available to all users by default, submit
   the paired characterization matrix and this responsibility period CSV to
   `pyaesa` through a pull request.
2. Keep the impact structure aligned across both files.
