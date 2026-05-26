# Architecture: Shared Uncertainty Assessment

## Purpose

`pyaesa/shared/uncertainty_assessment/` owns family neutral runtime primitives
used by public uncertainty functions.

It provides reusable mechanics for run planning, random streams, output table
formats, summaries, run manifests, completed run reuse, and Sobol
variance decomposition. Scientific source evaluation stays in the family
package that calls these primitives.

## Public Surface

There is no user facing public API in this package.

Public functions such as `uncertainty_asocc(...)`,
`uncertainty_ar6_cc(...)`, `uncertainty_acc(...)`,
`uncertainty_io_lca(...)`, and `uncertainty_asr(...)` may consume these
internal owners through their family orchestration layers.

## Responsibility Boundary

This package owns:

| Owner | Contract |
| --- | --- |
| `io/formats.py` | Public uncertainty output format validation and filename suffixes. |
| `request/core.py` | Family neutral Monte Carlo request normalization and uncertainty run batch sizing from family supplied memory blocks. Omitted or empty `mc_parameters` select convergence defaults; fixed run requests activate the `fixed` block and deactivate the `convergence` block. |
| `../runtime/memory.py` | Shared package runtime memory budget and row budget helpers used by uncertainty and deterministic owners. |
| `monte_carlo/runs.py` | Package allocated run ids, deterministic seeds, and fixed run batch plans. |
| `monte_carlo/random_streams.py` | Deterministic uniform random streams keyed by source name and run index. |
| `request/shared_u.py` | Deterministic shared uniform values from caller supplied shared keys. |
| `request/sources.py` | Active uncertainty source normalization independent of scientific source logic. |
| `io/tables.py` | Complete uncertainty table reading, writing, and column inspection. |
| `io/csv_fragments.py` | Compressed CSV fragment streams and CSV render byte estimates for generated run artifacts. |
| `io/run_artifacts.py` | Public run artifact metadata, fragment naming, interval indexes, and artifact README lines. |
| `io/run_writers.py` | Compact run matrix writers, sparse selected row writers, sparse row records, and run row byte estimates. |
| `io/downstream_run_outputs.py` | Family neutral downstream public run table writing, public summary scans, and Monte Carlo convergence loops for formulas that consume upstream run tables. |
| `evaluation/summary_groups.py` | Public row grouping, compact value collapse, sparse selected row memberships, and sparse per run group means for downstream uncertainty summaries and convergence. |
| `io/summary_kernels.py` | Shared summary statistic kernels for bounded float64 value blocks. |
| `io/public_summary.py` | Bounded summary and ASR frequency dispatch for compact and sparse public run artifacts. |
| `io/sparse_public_summary.py` | Sparse public summary replay through bounded temporary bucket files. |
| `run_state/manifest.py` | Run state, request key, and manifest payload construction. |
| `run_state/report.py` and `run_state/report_*.py` | Public uncertainty report assembly, argument extraction, output root discovery, dependency phase sections, dependency message propagation, and summary log writing. Generic phase labels and generic value formatting remain in `pyaesa/shared/runtime/reporting/`. |
| `run_state/branch_sets.py` | Mixed branch run id reuse, branch set manifests, branch scope manifest lists, and public branch set reports for composite uncertainty calls. |
| `run_state/figure_artifacts.py` | Figure artifact paths, figure request signatures, and rerender checks for complete uncertainty runs. |
| `run_state/runs.py` | Completed run discovery, appendable run selection, and requested Sobol reuse status checks for complete runs. |
| `run_state/sobol_artifacts.py` | Manifest persistence for completed runs that receive requested Sobol artifacts after initial Monte Carlo output generation. |
| `orchestration.py` | Family neutral convergence component figure timing, reusable progress labels, public output root discovery, and phase index entry construction. |
| `monte_carlo/convergence.py` | Family neutral checkpoint cursors, streaming cumulative mean accumulators, and relative stability comparison. |
| `sobol/plan.py` | Public Sobol parameter normalization and fixed package method constants. |
| `sobol/design.py` | Balanced Sobol or Saltelli design construction and chunk planning. |
| `sobol/accumulator.py` | Centered S1 and ST estimator accumulation over evaluated Sobol design rows. |
| `sobol/diagnostics.py` | Sobol convergence status and finite sample diagnostics. |
| `sobol/summary.py` | Generic row level Sobol index tables and variance weighted source summaries. |
| `sobol/summary_levels.py` | Generic selector level summaries and caller supplied invariant axis expansion. |
| `sobol/reporting.py` | Common Sobol method metadata payloads and README file writing entry point. |
| `sobol/readme_text.py` | Shared Sobol README interpretation, confidence, convergence, parameter, estimator, reference, and wrapping text. |
| `sobol/runner.py` | Fixed and convergence Sobol execution loops, including deterministic bootstrap confidence resampling for S1 and ST estimates. |

This package does not own:

| Responsibility | Owning layer |
| --- | --- |
| Scientific evaluation of source units | Family uncertainty package. |
| Family source activation rules | Family uncertainty package. |
| Public row selector columns | Family uncertainty package. |
| Value column names such as aSoCC output columns | Family uncertainty package. |
| Source method log schemas | Family uncertainty package. |
| LCIA, MRIO, method, or reference year rules | Owning domain package. |
| External input scientific contracts | `pyaesa/external_inputs/` plus the consuming family package. |

Shared uncertainty code must not import or encode aSoCC, aCC, ASR, AR6 CC,
IO-LCA, LCIA, MRIO, allocation method taxonomy, deterministic family paths, or
public family row columns.

## Family Integration Contract

A family uncertainty owner supplies the scientific parts of a run:

| Family supplied item | Used by shared owners |
| --- | --- |
| Normalized request scope | Manifest and completed run compatibility. |
| Active source names | Random stream keys and Sobol source dimensions. |
| Public output identity | Run tables, summary tables, and Sobol index rows. |
| Output value arrays | Run writers, summary, and Sobol estimators. |
| Downstream run iterators and summary callbacks | Shared downstream run writer. |
| Value column name | Sparse run row writing and reading. |
| CSV dtype contract for public identity columns | Family table readers. |
| Source method log rows | Family source method writer. |
| Sobol evaluator callback | Shared Sobol runner. |
| Sobol source summary levels and family notes | Shared Sobol summary and README writers. |

Every uncertainty family persists one `logs/scope_manifest.json` per Monte
Carlo run folder. The manifest has one public call payload: `function`,
`arguments`, `execution`, `reuse`, `artifacts`, and `provenance`. The shared
manifest owner derives compatibility from normalized `arguments`, active
sources, prerequisite compatibility, Monte Carlo parameters, Sobol parameters,
and the output contract. Family owners supply family scientific payloads;
shared runtime code does not inspect family row schemas.

Public Monte Carlo run artifacts follow one shared fragment dataset contract.
For `csv_compact` output, the artifact path is a dataset directory containing
compressed `part-*.csv.zst` CSV fragments. For Parquet output, the artifact
path is a dataset directory containing `part-*.parquet` fragments. Each run
artifact has a required interval index beside it using the
`*.run_intervals.<suffix>` stem. The interval index records the batch, run
range, artifact row range, and fragment name needed for bounded scans. Public
readers use it to load requested run windows without concatenating or scanning
unrelated batches. Manifests record the artifact kind, fragment pattern, and
interval index path for each run artifact.

Composite uncertainty calls that request more than one carrying capacity branch
write one branch set manifest at the family Monte Carlo root. The branch set
manifest records `branch_scope_manifests`, `branch_run_roots`, and a public
output branch list. Each branch run keeps its own complete uncertainty scope
manifest and artifacts under the branch token root. Single branch calls use the
same branch token root directly and do not need a branch set manifest.

Figure reuse is scoped by the containing Monte Carlo run folder and a figure
request signature stored under `artifacts.figure_request`. The signature covers
the completed run count, compatibility key, public arguments, active sources,
public output payload, figure options, and figure format. A reused uncertainty
run renders figures only when the requested visual contract differs from the
stored signature or a required figure artifact is missing.

Family source method writers own `source_methods.csv` schemas. Those CSVs are
structured audit artifacts, so note cells and physical CSV rows are not wrapped
by the shared 100 character text rule. Printed summaries, returned reports,
README files and workbook README cells remain wrapped by their
owning runtime text helpers.

The shared Sobol runner receives a family evaluator that maps unit interval
source values to public output values. It does not know how each source changes
the scientific model.

## Data Flow

Monte Carlo runs use this package for request normalization, run plan creation,
random stream construction, public run table writing, bounded public
summary scans, streaming mean convergence diagnostics, run manifest
persistence, shared orchestration messages, and public phase index entry
helpers. Runtime memory budget detection is owned by
`pyaesa/shared/runtime/memory.py` so deterministic and uncertainty code use the
same system memory policy without importing each other. Family code owns
prerequisite loading, source specific sample application, final component
figure calls, and public phase ordering.

Sobol runs use this package for normalized Sobol parameters, balanced design
chunks, S1 and ST accumulation, bootstrap confidence precision, convergence
status, generic output tables, and README method text. Family code owns selected
output scoping, source evaluation, selector level policy, and family specific
notes.

## Testing And Quality Gates

Shared uncertainty tests live under
`tests/package/shared/uncertainty_assessment/`.

Contributor changes must also run the package tests for every family that
consumes the touched shared owner. Touched owners must retain 100 percent line
and branch coverage. Tests should exercise behavior reachable through public
functions or realistic file flows; branches that cannot be reached through
those routes should be deleted with their tests.
