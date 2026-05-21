Add custom EXIOBASE LCIA characterization matrices
==================================================

Purpose
-------

This folder stores EXIOBASE LCIA characterization matrices used by:
- process_mrio(...)
- deterministic_io_lca(...)
- LCIA based aSoCC methods that depend on processed MRIO characterization

This workspace folder stores the EXIOBASE LCIA characterization matrices
available to the current project under:
- `data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/`

This guide is only needed when adding an LCIA characterization matrix for a
custom LCIA method beyond the files shipped with the package. Users who run
package shipped LCIA methods can use the bundled CSV files directly and do not
need to edit this folder.

The custom method workflow is to copy the CSV template in this workspace
folder and then use the same `lcia_method` name in package calls.

What to copy for a custom method
--------------------------------

Add one CSV per method:
- <lcia_method>.csv

Use exactly one of the packaged model files as the starting structure:
- name_lcia_template.csv
- name_lcia_planetary_boundary_template.csv

Canonical column contract
-------------------------

Standard schema:
- impact_full_name
- impact_parent
- impact
- impact_unit
- stressor
- extension
- stressor_unit
- factor

Planetary boundary schema:
- Planetary boundary
- Control variable
- impact_parent
- impact
- impact_unit
- stressor
- extension
- stressor_unit
- factor

When to use each schema
-----------------------

Use the standard schema when the method only needs a generic
`impact_full_name` label.

Use the planetary boundary schema only when the characterization matrix should
preserve explicit `Planetary boundary` and `Control variable` vocabulary for
user facing and contributor facing traceability.

Guidance
---------------

- Keep the file name stem equal to the public `lcia_method` token.
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
3. Copy one template CSV there.
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
4. Update the docs if the method introduces a new public contract.
