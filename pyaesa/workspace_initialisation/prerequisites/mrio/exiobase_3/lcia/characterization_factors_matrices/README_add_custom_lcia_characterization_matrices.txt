Add custom EXIOBASE LCIA characterization matrices
==================================================

Purpose
-------

This folder stores EXIOBASE LCIA characterization matrices used by:

- `process_mrio(...)`
- `deterministic_io_lca(...)`
- LCIA based aSoCC methods that depend on processed MRIO characterization

This workspace folder stores the EXIOBASE LCIA characterization matrices
available to the current project under:

- `data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/<exiobase_version>/`

This guide is only needed when adding an LCIA characterization matrix for a
custom LCIA method beyond the files shipped with the package. Users who run
package shipped LCIA methods can use the bundled CSV files directly and do not
need to edit this folder.

Version folders are `exiobase_3112`, `exiobase_3102`, and `exiobase_396`.
Each contains `pb_lcia.csv`, `ef_3.1.csv`, `ef_3.1_14cat.csv`,
`gwp100_ef31.csv`, and `gwp100_ipcc.csv`.
Templates and this guide remain in the parent folder.

The custom method workflow is to copy a template into the required version
folder and use the same `lcia_method` name in package calls. Each requested
source version requires its own matrix.

What to copy for a custom method
--------------------------------

Add one CSV per method in each required source version folder:

- `<lcia_method>.csv`

Use exactly one of the packaged model files as the starting structure:

- `name_lcia_template.csv`
- `name_lcia_planetary_boundary_template.csv`

Canonical column contract
-------------------------

Standard schema:

- `impact_full_name`
- `impact_parent`
- `impact`
- `impact_unit`
- `stressor`
- `extension`
- `stressor_unit`
- `factor`

Planetary boundary schema:

- `Planetary boundary`
- `Control variable`
- `impact_parent`
- `impact`
- `impact_unit`
- `stressor`
- `extension`
- `stressor_unit`
- `factor`

When to use each schema
-----------------------

Use the standard schema when the method only needs a generic
`impact_full_name` label.

Use the planetary boundary schema only when the characterization matrix should
preserve explicit `Planetary boundary` and `Control variable` vocabulary for
user facing and contributor facing traceability.

Guidance
--------

- Keep the file name stem equal to the public `lcia_method` token.
- Match every `stressor`, `extension`, and `stressor_unit` to the selected
  EXIOBASE version exactly. Do not rename or combine inventory rows to bypass
  incompatibility. Validate factors, units, source citations, and overlapping
  inventory accounts before adding rows.
- Responsibility periods and carrying capacities are shared by method, outside
  the source version folders; use consistent impact names and units.
- Use `impact_parent` for the parent category and `impact` for the concrete
  characterized output.
- When no split is needed, write the same label in both `impact_parent` and
  `impact`.
- Keep `impact` and `impact_parent` aligned with any later
  `<lcia_method>_rps.csv` file so historical responsibility workflows stay
  scientifically consistent.

Custom method workflow
----------------------

User workflow:

1. Run `set_workspace(...)` for the target workspace.
2. Go to `data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/`.
3. Copy one template CSV into the selected version folder, for example
   `exiobase_3112/`.
4. Rename it to `<lcia_method>.csv`.
5. Fill the characterization rows.
6. Use the same `lcia_method` name in package calls.

Optional contributor route:

1. If the new method should become available to all users by default, submit
   the characterization matrix to `pyaesa` through a pull request.
2. Add the matching bundled static carrying capacity CSV when the method should
   support denominator workflows.
3. Add the responsibility period table when the method should support
   historical responsibility allocation.
4. Update the process data tutorial and affected public docstrings. Add
   deterministic file based package tests for source version selection,
   flow names, units, duplicate mappings, factors, and missing extensions.
   Run the package coverage, lint, and type gates listed in AGENTS.md.
