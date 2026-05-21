# Architecture: External Inputs (`external_inputs/`)

## Purpose

`pyaesa/external_inputs/` owns project scoped external aSoCC and external LCA
input preparation, contracts, staged file loading, and package code reused by
downstream deterministic families.

It is the canonical internal owner for:

- the public `prepare_external_inputs(...)` function
- external aSoCC selector contracts and staged input loading
- external LCA staged input loading and figure rendering
- project scoped README guidance and runnable example preparation

## Public Surface

| Surface | Owner | Contract |
| --- | --- | --- |
| `prepare_external_inputs(...)` | `prepare_external_inputs.py` | Prepare project scoped scaffolds from `project_name`. |
| External aSoCC staged inputs | `asocc/` | External aSoCC contracts, deterministic loading, Monte Carlo loading, downstream aSoCC share loading, and aSoCC guidance and example staging. |
| External LCA staged inputs | `lca/` | External LCA deterministic loading, Monte Carlo source loading for ASR uncertainty, figure rendering, and LCA guidance and example staging. |

## Responsibility Boundary

`external_inputs/` owns only project scoped external input behavior.

It does:

- resolve the project scoped external input scaffold reused by downstream
  deterministic calls
- write the preparation `prepare_external_inputs_log/scope_manifest.json` and
  `prepare_external_inputs_log/summary.log` for the public scaffold scope
- define the exact external file contracts for aSoCC and LCA inputs
- load staged deterministic external inputs
- load staged external aSoCC Monte Carlo rows for uncertainty consumers
- load staged external LCA Monte Carlo rows for ASR uncertainty consumers
- provide project scoped README guidance and runnable example preparation
- preserve existing user staged files during scaffold preparation and write
  packaged assets only when the target file is missing

It does not:

- own native deterministic aSoCC computation
- own downstream aCC or ASR scientific logic
- own uncertainty runtime infrastructure

## Internal Organization

- `prepare_external_inputs.py`
  - public scaffold preparation coordinator and preparation summary or
    manifest writer
- `asocc/schema/contracts.py`
  - external aSoCC selector and collision rules
- `asocc/schema/file_specs.py`
  - external aSoCC filename, SSP, and year assignment contracts
- `asocc/deterministic/files.py`
  - deterministic project scoped external aSoCC loading
- `asocc/monte_carlo/files.py`
  - canonical external aSoCC Monte Carlo owner for source resolution,
    validation, run inventory, compact first storage selection, selected year
    scoping, and manifest identity payloads
- `asocc/monte_carlo/matrix.py`
  - internal compact folder, Arrow, Parquet, and pickle numeric matrix
    materialization helper used only through `asocc/monte_carlo/files.py`
- `asocc/monte_carlo/matrix_inventory.py`
  - internal chunked inventory validation helpers used only through
    `asocc/monte_carlo/files.py`
- `asocc/deterministic/downstream_shares.py`
  - shared downstream aSoCC share preparation for external aSoCC from the
    canonical published source label passed by downstream owners
- `asocc/schema/row_schema.py`
  - external aSoCC canonical public row shaping
- `asocc/templates/templates.py`
  - external aSoCC README guidance and runnable example staging
- `lca/io.py`
  - external LCA filename grammar, file discovery, year assignment, and
    normalized staged row loading
- `lca/paths.py`
  - external LCA root, deterministic, Monte Carlo, template, and deterministic
    figure path ownership. Deterministic external LCA figures live under the
    deterministic external LCA subtree and carry the external LCA version in
    the file stem.
- `lca/deterministic.py`
  - deterministic external LCA loading
- `lca/monte_carlo.py`
  - external LCA Monte Carlo loading, compact first storage selection, and run
    inventory validation used by ASR uncertainty
- `shared/`
  - schema inspection, projected table reads, Arrow wide unpivoting, shared
    external Monte Carlo identity lookup, and compact run matrix loading used
    by both external input families
- `lca/figures.py`
  - external LCA figure rendering
- `lca/templates.py`
  - external LCA README guidance and runnable example staging
- `templates.py`
  - shared packaged asset copier used by both external input families

## Testing And Quality Gates

Mandatory package tests covering this boundary live under:

| Scope | Tests |
| --- | --- |
| External input public preparation | `tests/package/external_inputs/` |
| External LCA loading and figures | `tests/package/external_inputs/lca/` |
| ASR uncertainty external LCA consumption | `tests/package/asr/` |
| Downstream consumers | Targeted tests when staged file contracts change. |

For touched external input owners, run scoped `ruff`, scoped `pyright`, and
targeted package tests with 100 percent line and branch coverage for the
touched owners. Tests should exercise staged file contracts through real local
files rather than private patching.
