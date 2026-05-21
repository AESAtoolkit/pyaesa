# Architecture: Runtime Scope And Signature Contracts

## Purpose

This note describes shared runtime helpers that normalize deterministic scope
identity, persisted scope compatibility, filesystem writes, scenario partition
metadata, and family neutral uncertainty run state across families.

## Active Owners

- `runtime/metadata/contracts.py`
  - shared artifact name constants such as `scope_manifest.json`
- `runtime/reuse/contracts.py`
  - deterministic aSoCC persisted scope compatibility
  - deterministic IO-LCA persisted scope compatibility
  - selector payload normalization used by deterministic scope lookup
- `runtime/scenario/partitions.py`
  - SSP partition filename parsing for deterministic and shared table readers
- `runtime/io/persisted_paths.py`
  - existing deterministic table path discovery helpers
- `runtime/io/filesystem.py`
  - scoped filesystem helpers used by deterministic writers
- `runtime/reporting/status.py`
  - transient status reporting for long deterministic operations
- `runtime/reporting/progress.py`
  - the family neutral `YearProgressPrinter` owner for persistent and
    transient console messages
- `runtime/reporting/run_progress.py`
  - run scale and Sobol scale transient progress for uncertainty runs
- `runtime/reporting/figure_progress.py`
  - bounded transient figure generation status lines
- `runtime/text.py`
  - shared wrapping and compacting helpers for public runtime text
- `uncertainty_assessment/`
  - family neutral Monte Carlo requests, compact run IO, exact summaries,
    run manifests, completed run reuse, convergence state, and Sobol planning

## Responsibility Boundary

Shared runtime helpers own package wide deterministic identity primitives.
They do not own one family scientific logic, public function orchestration, or
family specific uncertainty source evaluation.

## Runtime Reporting Contract

Shared reporting owns reusable display mechanics only. Family code owns the
scientific event being reported and passes already meaningful user text to the
shared helpers.

| Contract | Owner | Required behavior |
| --- | --- | --- |
| Transient lines | `runtime/reporting/progress.py` | Replace the active line and clear it with `finish()` before return or failure. |
| Run and Sobol progress | `runtime/reporting/run_progress.py` | Keep one reusable transient line per family phase and allow repeated `finish()` calls. |
| Figure progress | `runtime/reporting/figure_progress.py` | Render one bounded `generating figure N/M` line and clear it after rendering. |
| Public runtime text | `runtime/text.py` | Wrap nonpath user visible lines at 100 characters and compact transient labels. |
| Returned uncertainty summaries | `uncertainty_assessment/run_state/report.py` | Keep nonpath lines wrapped and leave filesystem paths unwrapped. |

The 100 character visible line rule applies to printed messages, returned
summaries, generated `.txt` files, and workbook README cells. Public
notebook markdown source lines are exempt because notebook rendering wraps user
facing prose. `source_methods.csv` remains a structured CSV artifact: cell text
and physical CSV rows are not wrapped unless a schema change explicitly splits
those notes into additional fields.

Family runners must create progress printers only where cleanup is guaranteed
or must clear them in the same exception scope that starts the progress line.
Composite functions that auto run upstream functions must clear each upstream
transient line before starting the downstream phase. Persisted summaries must
not expose internal status tokens when a user facing phrase exists.

## Testing And Quality Gates

Package tests for active shared runtime behavior live under
`tests/package/shared/` and through the deterministic family tests that consume
these helpers.

Required validation for touched runtime scope and signature owners:

- `python -m ruff check <touched shared runtime paths> <touched tests>`
- `python -m ruff format --check <touched shared runtime paths> <touched tests>`
- `python -m pyright <touched shared runtime paths>`
- targeted package tests with line and branch coverage for the touched owner

When a runtime signature contract changes reuse, refresh, or persisted scope
identity behavior, validate at least one consuming public entry point through
its package tests and update the owning family architecture note.
