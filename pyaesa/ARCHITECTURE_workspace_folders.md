# Architecture: Workspace Folders

## Purpose

This note describes workspace folder ownership for the package. The public
surface writes raw data, processed data, deterministic analytical outputs,
uncertainty run outputs, figures, logs, and external input scaffolds.

## Workspace Root

`set_workspace(...)` creates or reuses the workspace root and imports
packaged prerequisite files used by later public functions. The root contains:

- `data_raw/`
- `data_processed/`
- `<project_name>/`
- workspace metadata files owned by `pyaesa/workspace_initialisation/`

Folder ownership is split by runtime family:

| Workspace area | Canonical owner |
| --- | --- |
| Workspace root and packaged prerequisites under `data_raw/` | `pyaesa/workspace_initialisation/` |
| Raw source payloads under `data_raw/` | `pyaesa/download/` |
| Processed runtime assets under `data_processed/` | `pyaesa/process/` |
| Deterministic and uncertainty scientific outputs under `<project_name>/` | The owning deterministic or uncertainty family package |
| External input scaffolds under `<project_name>/A_lca/external_lca/` and `<project_name>/B1_asocc/external_asocc/` | `pyaesa/external_inputs/` |

Refresh behavior is scoped to the public function that exposes the refresh
argument. Refresh cleanup must delete only that function's own output scope
after active file and log handlers for that scope are closed.

MRIO refresh is an expensive replacement of raw or processed MRIO
scopes. It must be run only when the caller explicitly requests
`download_mrio(..., refresh=True)` or `process_mrio(..., refresh=True)`.

## Deterministic Output Roots

Active deterministic families write under:

- `<project_name>/B1_asocc/`
- `<project_name>/B2_acc/`
- `<project_name>/C_asr/`
- `<project_name>/A_lca/io_lca/`
- `<project_name>/A_lca/external_lca/`
- `data_processed/ar6/<processed_scope>/ar6_cc/<cc_scope>/deterministic/`
  for deterministic dynamic AR6 carrying capacity tables

Each deterministic family owns its own path helper module. Shared helpers may
provide neutral filename parsing and table IO, but they do not own family
specific output layout.

AR6 CC uncertainty writes beside the deterministic AR6 CC scope under
`data_processed/ar6/<processed_scope>/ar6_cc/<cc_scope>/monte_carlo/<run_id>/`.
Downstream aCC and ASR uncertainty functions consume those paths through the
AR6 CC uncertainty manifest contract.

## External Input Roots

`prepare_external_inputs(...)` creates project scoped external input folders,
README guidance, and runnable CSV examples:

- `<project_name>/B1_asocc/external_asocc/deterministic/`
- `<project_name>/B1_asocc/external_asocc/monte_carlo/`
- `<project_name>/B1_asocc/external_asocc/templates/`
- `<project_name>/A_lca/external_lca/deterministic/`
- `<project_name>/A_lca/external_lca/monte_carlo/`
- `<project_name>/A_lca/external_lca/templates/`

External LCA figure folders are created by the external LCA figure renderer
when figures are requested.

## Testing And Quality Gates

Workspace folder behavior is tested through package tests for project
initialisation, download, process, deterministic families, uncertainty
families, external inputs, and shared path helpers under `tests/package/`.

For touched folder ownership code, run scoped `ruff`, `pyright`, and targeted
package tests with 100 percent line and branch coverage for the touched owners.
