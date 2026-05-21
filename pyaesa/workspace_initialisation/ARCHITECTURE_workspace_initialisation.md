# Architecture: Workspace Setup (`pyaesa/workspace_initialisation/`)

## Purpose

The `pyaesa.workspace_initialisation` package owns the workspace setup boundary
centered on `set_workspace(...)`. It creates or reuses the user side workspace
repository root, imports prerequisite files, and records the active
workspace repository root for the active Python session.

This document is for external Python contributors. It describes the public
setup surface, workspace state ownership, prerequisite import contract, and the
rules that keep later package calls independent from hidden setup side effects.

## Public Surface

The package level public API is exported through `pyaesa.__init__`.

| Public function | Owner module | Responsibility |
| --- | --- | --- |
| `set_workspace(...)` | `pyaesa/workspace_initialisation/set_workspace.py` | Create or reuse the workspace repository root, import prerequisites, and set the active workspace repository root. |

`set_workspace(...)` is the only public workspace setup function. Internal
workspace helpers must remain under `workspace_initialisation/` or shared runtime
helpers.

## Responsibility Boundary

`pyaesa.workspace_initialisation` owns:

| Area | Canonical owner |
| --- | --- |
| Public setup orchestration | `set_workspace.py` |
| User top path to repository root resolution | `workspace.py` |
| Active session repository root | `workspace.py` |
| Project output root naming | `workspace.py` |
| Prerequisite import | `packaged_prerequisites.py` |
| Packaged prerequisite assets | `prerequisites/` |
| Methodological note package assets | `prerequisites/methodological_notes/` |

`pyaesa.workspace_initialisation` does not own:

| Area | Owner |
| --- | --- |
| Raw source downloads | `pyaesa/download/` |
| Processed data generation | `pyaesa/process/` |
| Deterministic or uncertainty scientific outputs | Respective scientific packages |
| Generic file parent creation helpers | `pyaesa/shared/runtime/io/filesystem.py` |
| Family specific output routing under `<project_name>/` | Respective family path owner modules |

Do not add download, processing, or scientific execution logic to workspace
setup. It only prepares workspace prerequisites and records the active
workspace repository root.

## Package Layout

| Path | Role |
| --- | --- |
| `set_workspace.py` | Public `set_workspace(...)` coordinator. |
| `workspace.py` | Repository root resolution, active session root state, and `<project_name>/` root contract. |
| `packaged_prerequisites.py` | Copy package resource prerequisites into the workspace `data_raw/` tree. |
| `<repo_root>/data_raw/summary.log` | User facing setup guidance log written by `set_workspace(...)`. |
| `prerequisites/mrio/` | Grouping templates, region matching tables, sector classification, LCIA characterization factors, LCIA responsibility period files, and local README guides for grouping or custom LCIA inputs. |
| `prerequisites/carrying_capacities/` | Static carrying capacity prerequisite CSVs and the local README guide for adding custom carrying capacity files. |
| `prerequisites/methodological_notes/` | Detailed methodological PDFs, recommended citation guide, quick functional unit and allocation method guide, and allocation paths figure copied to `data_raw/methodological_notes/`. |

`workspace_initialisation` path helpers own only the workspace root and
project output root. Family packages own their own branch paths below those
roots.

## Workspace Setup Flow

`set_workspace(...)` follows one canonical sequence:

1. Resolve the user supplied `top_path` to `<top_path>/pyaesa`.
2. Create the repository root when missing.
3. Copy package resource prerequisites into `<repo_root>/data_raw/`.
4. Write `<repo_root>/data_raw/summary.log` with setup guidance information
   for methodological references, editable guides, CSV templates, CoV inputs,
   the EXIOBASE sector definition guide, and the GitHub tutorial notebooks.
5. Store the active workspace repository root in session state only after
   prerequisite import succeeds.
6. Print the full setup guidance information when prerequisite files were copied,
   refreshed, or the summary log was updated. Print only the summary log path
   when the workspace files and summary log are unchanged.

If prerequisite import fails, the active workspace repository root is not
updated. This prevents later package calls from using a partially initialized
workspace.
When `refresh=False` and all prerequisite files already exist,
`set_workspace(...)` still records the active workspace repository root and prints the
`data_raw/summary.log` path.

## Workspace State Contract

`workspace.py` owns the only in memory setup state:

| Function | Contract |
| --- | --- |
| `resolve_repo_root(top_path)` | Return the absolute `<top_path>/pyaesa` repository root and reject blank string paths. |
| `set_default_repo_root(repo_root)` | Store the active workspace repository root for the session. |
| `get_default_repo_root()` | Return the active workspace repository root or fail before setup. |
| `clear_default_repo_root()` | Clear the active workspace repository root for tests. |
| `project_outputs_root(project_name=...)` | Return `<repo_root>/<project_name>` and reject blank project names. |

The active workspace repository root is intentionally process local. It is not
persisted to a global config file. Users call `set_workspace(...)` in each
Python session or notebook before package functions that require a workspace.

Public user facing functions in other packages should call
`get_default_repo_root()` or a family path helper that delegates to it. They
should not duplicate workspace state.

## Prerequisite Import Contract

`packaged_prerequisites.py` copies files from the installed package resource
tree into the workspace `data_raw/` tree. Methodological notes are packaged
under the prerequisite resource tree and copied to `data_raw/methodological_notes/`.

Prerequisite families:

| Family | Workspace purpose |
| --- | --- |
| `prerequisites/mrio/exiobase_3/grouping/` | EXIOBASE region and sector grouping templates. |
| `prerequisites/mrio/exiobase_3/reg_matching/` | EXIOBASE region matching tables for WB and SSP data. |
| `prerequisites/mrio/exiobase_3/lcia/characterization_factors_matrices/` | EXIOBASE LCIA characterization factors and the local README guide for adding custom characterization matrices beyond package shipped methods. |
| `prerequisites/mrio/exiobase_3/lcia/responsibility_periods/` | LCIA responsibility period files and the local README guide for adding custom responsibility period tables beyond package shipped methods. |
| `prerequisites/mrio/exiobase_3/lcia/carbon_accounts_covs/` | Carbon consumption based accounts coefficients of variation (CoV) CSVs for LCIA based allocation methods and IO-LCA LCIA result uncertainty. |
| `prerequisites/mrio/oecd_v2025/grouping/` | OECD region and sector grouping templates. |
| `prerequisites/mrio/oecd_v2025/reg_matching/` | OECD region matching tables for WB and SSP data. |
| `prerequisites/carrying_capacities/` | Static carrying capacity tables, LCIA carrying capacity metadata, and the custom carrying capacity guide. |
| `prerequisites/methodological_notes/` | Detailed methodological PDFs, recommended citations, the quick functional unit and allocation method guide, and the allocation paths figure copied to `data_raw/methodological_notes/`. |

Import rules:

| `refresh` | Behavior |
| --- | --- |
| `False` | Copy prerequisite files that are missing and preserve user edited existing files. |
| `True` | Overwrite existing prerequisite files in the workspace. |

The import owner preserves the package resource folder hierarchy beneath
`data_raw/`. Methodological assets are written beneath
`data_raw/methodological_notes/` because their package resources live under
`prerequisites/methodological_notes/`.
If a destination file path is occupied by a directory, import fails because the
workspace cannot represent the prerequisite file.
The import owner returns whether any file was copied or refreshed. The public
coordinator combines that result with the summary log write result to decide
whether to print full setup guidance or only the persisted summary log path.

## Path Ownership

Path ownership is intentionally narrow:

| Path responsibility | Owner |
| --- | --- |
| `<top_path>/pyaesa` repository root | `workspace_initialisation/workspace.py` |
| `<repo_root>/data_raw/` prerequisite import target | `workspace_initialisation/packaged_prerequisites.py` |
| `<repo_root>/data_raw/summary.log` setup guidance log | `workspace_initialisation/set_workspace.py` |
| `<repo_root>/<project_name>/` root | `workspace_initialisation/workspace.py` |
| Raw dataset subpaths | `pyaesa/download/**/paths.py` or equivalent family path owner |
| Processed dataset subpaths | `pyaesa/process/**/paths.py` or equivalent family path owner |
| Deterministic and uncertainty output subpaths | Family runtime path owner modules |

Do not construct family specific raw, processed, deterministic, or Monte Carlo
paths inside `workspace_initialisation`.

## Adding Prerequisites

When adding a package resource prerequisite:

1. Place the file under the correct `prerequisites/` family folder.
2. Add or update a local README guide next to files users may edit.
3. Update the consumer path owner or loader in the package that reads the file.
4. Add tests under `tests/package/workspace_initialisation/` for import
   behavior when the prerequisite structure changes.
5. Add downstream tests for the package that consumes the prerequisite.
6. Update user docs or tutorials when users need to edit or inspect the file.

Do not add prerequisite file discovery in downstream scientific code when a path
owner or loader should own the contract.

When adding a methodological reference file, place the package resource copy
under `prerequisites/methodological_notes/` and include the file extension in
packaging metadata. Keep the human readable source copy in the matching
repository documentation or tutorial location when the file is also maintained
outside the package resource tree.

## Refresh Semantics

`set_workspace(refresh=True)` overwrites prerequisite files only. It does
not delete raw downloads, processed data, deterministic outputs, uncertainty
outputs, figures, or user project output folders.

Family refresh arguments belong to their public family functions:

| Refresh target | Public function |
| --- | --- |
| Raw downloads | `download_mrio(...)`, `download_pop_gdp(...)`, `download_ar6(...)` |
| Processed data | `process_mrio(...)`, `process_pop_gdp(...)`, `process_ar6(...)` |
| Deterministic scientific outputs | Family deterministic functions |
| Monte Carlo outputs | Family uncertainty functions |

Keep these refresh scopes separate.

## Testing And Quality Gates

Package tests for workspace setup live under:

`tests/package/workspace_initialisation/`

Workspace setup tests must cover:

1. Repository root resolution.
2. Active session root storage and clearing.
3. Prerequisite import with `refresh=False`.
4. Prerequisite overwrite with `refresh=True`.
5. Failure when a destination file path is occupied by a directory.
6. Project output root naming and blank project name rejection.
7. Methodological note and citation guide prerequisite placement.
8. Startup guidance logging to `data_raw/summary.log`.
9. Full startup guidance printing when files are copied, refreshed, or the
   summary log is updated.
10. Summary log path printing when a complete workspace is reused unchanged.

Downstream package tests may use initialized dummy repositories, but they should
not retest `set_workspace(...)` internals unless the setup contract itself is
being changed.

For touched workspace setup owners, run:

1. `python -m ruff check <touched paths>`.
2. `python -m ruff format --check <touched paths>`.
3. `python -m pyright <touched package paths>`.
4. Targeted `pytest` with `--cov=<touched owner> --cov-branch`.

Keep touched owners at 100 percent line and branch coverage.
