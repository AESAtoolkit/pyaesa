# Architecture: Shared Package Primitives (`shared/`)

## Purpose

`shared/` owns package wide internal primitives that are reused across more
than one scientific family.

This package is the canonical home for contributor facing shared ownership such
as:

- runtime messaging, metadata, filesystem, and branch state primitives
- tabular contracts and shape utilities
- selector normalization
- LCIA prerequisite and bundled schema contracts
- shared plotting and figure primitives

The cross family runtime scope and signature contract that decides branch
ownership, reuse, refresh, and project scope behavior
is documented in
`pyaesa/shared/ARCHITECTURE_runtime_scope_and_signature_contracts.md`.

## Public Surface

There is no user facing public API under `shared/`.

All modules here are internal package owners. Internal imports should target
the concrete owner modules rather than package re export layers.

## Responsibility Boundary

`shared/` is responsible for package wide primitives only. It must not own:

- one family scientific logic
- composite aCC/ASR only family layers
- public entrypoint orchestration

Those responsibilities belong in their owning scientific packages.

## Internal Organization

The shared package is organized by explicit ownership subtrees:

- `runtime/` for package wide runtime primitives
- `tabular/` for shared tabular format and shape contracts
- `selectors/` for shared selector normalization
- `lcia/` for shared LCIA prerequisite and schema contracts
- `figures/` for shared plotting and deterministic figure primitives
- `uncertainty_assessment/` for family neutral uncertainty runtime primitives

Concrete canonical owners include:

- `runtime/metadata/contracts.py` for shared artifact name tokens such as
  `scope_manifest.json`
- `runtime/manifest_contract.py` for canonical manifest JSON value
  normalization, stable digest keys, and path list serialization
- `runtime/io/file_identity.py` for file size and checksum identity payloads used
  by deterministic and uncertainty manifests
- `runtime/reuse/contracts.py` for deterministic persisted scope compatibility
  checks shared by aSoCC and IO-LCA consumers
- `runtime/scenario/partitions.py` for SSP partition filename parsing
- `tabular/table_io.py` for small shared table read/write helpers
- `lcia/file_owned_tables.py` for LCIA method owned table path helpers
- `lcia/static_cc.py` for static carrying capacity CSV loading and bound
  validation used by aCC, ASR, and shared aCC/ASR request normalization
- `lcia/uncertainty_keys.py` for LCIA shared random key construction used
  by family uncertainty code
- `figures/contracts.py` for shared figure request validation, including
  `validate_figure_dpi(...)`
- `figures/request_validation.py` for public figure option and format request
  normalization
- `uncertainty_assessment/io/formats.py` for uncertainty output format
  validation
- `uncertainty_assessment/monte_carlo/runs.py` for fixed run batch plans and
  batch random stream ownership
- `uncertainty_assessment/io/tables.py` for compact CSV and Parquet uncertainty
  table IO
- `uncertainty_assessment/io/summary_kernels.py` for exact summary statistic
  kernels over bounded float64 blocks
- `uncertainty_assessment/io/public_summary.py` for bounded exact public
  summary scans from compact and sparse uncertainty run artifacts
- `uncertainty_assessment/run_state/manifest.py` for canonical uncertainty run
  state
- `uncertainty_assessment/sobol/plan.py` for family neutral Sobol public
  parameter normalization and fixed Sobol method constants recorded in run
  metadata
- `uncertainty_assessment/sobol/design.py` for balanced Sobol or Saltelli design
  construction and chunk planning
- `uncertainty_assessment/sobol/accumulator.py` for centered S1 and ST
  estimator accumulation over evaluated Sobol design rows
- `uncertainty_assessment/sobol/diagnostics.py` for finite sample Sobol
  diagnostics and source summary convergence checks
- `uncertainty_assessment/sobol/summary.py` for row level Sobol index tables
  and variance weighted source summaries
- `uncertainty_assessment/sobol/summary_levels.py` for selector level source
  summaries and family supplied invariant axis expansion
- `uncertainty_assessment/sobol/reporting.py` for common Sobol README writing
  entry point and method metadata payloads
- `uncertainty_assessment/sobol/readme_text.py` for shared Sobol README
  interpretation, convergence, parameter, estimator, reference, and text
  wrapping
- `uncertainty_assessment/sobol/runner.py` for fixed and convergence Sobol
  execution loops that remain independent of family scientific source logic

The detailed uncertainty runtime contract is documented in
`pyaesa/shared/uncertainty_assessment/ARCHITECTURE_uncertainty_assessment.md`.

Additional subfolders are appropriate when they improve
contributor navigation and make ownership clearer.

## Runtime Contracts

- `shared/` must be the only owner tree for package wide non public shared
  primitives
- `runtime/reporting/` owns family neutral display mechanics only; scientific
  event text remains owned by the calling family
- public runtime messages and returned nonpath summary lines must be wrapped at
  100 visible characters through `runtime/text.py`
- transient progress printers must be cleaned up on success and failure before
  a public function returns control to the user
- no compatibility wrappers or duplicate implementations should remain once a
  canonical shared owner has been established
- helpers that are reused only inside one scientific family should stay in that
  family instead of being promoted here
- `shared/uncertainty_assessment/` must stay family neutral; aSoCC, aCC, ASR,
  AR6 CC, IO-LCA, and external input scientific source logic belongs in the
  owning family packages

## Testing And Quality Gates

- Package tests for shared primitives live under `tests/package/shared/`.
- Behavior used through a family public entry point is also covered in that
  family suite under `tests/package/`.
- For touched shared owners, run scoped `ruff`, scoped `pyright`, and targeted
  package tests with line and branch coverage for the touched owner.
- When a shared contract changes accepted public behavior in a consuming
  family, update the consuming family architecture note and public docstring in
  the same change set.
- Shared uncertainty owners must remain family neutral in tests as well as code:
  test fixtures may supply family data shapes, but shared tests must not import
  private family runtime helpers to create the behavior under test.


