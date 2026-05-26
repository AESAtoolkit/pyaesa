# Architecture: IO-LCA (`io_lca/`)

## Purpose

`io_lca/` owns pyaesa owned deterministic input output life cycle
assessment from processed MRIO assets and the pyaesa owned IO-LCA
uncertainty leaf for LCIA value uncertainty.

## Public Surface

| Public function | Canonical owner | Contract |
| --- | --- | --- |
| `deterministic_io_lca(...)` | `deterministic_io_lca.py` | Writes deterministic IO-LCA outputs. |
| `uncertainty_io_lca(...)` | `uncertainty_io_lca.py` | Writes IO-LCA Monte Carlo outputs. |

## Responsibility Boundary

`io_lca/` owns:

| Responsibility | Owner |
| --- | --- |
| Deterministic IO-LCA orchestration | `deterministic_io_lca.py`, `orchestration/` |
| Deterministic IO-LCA result and metadata paths | `data/paths.py`, `data/metadata.py` |
| Deterministic IO-LCA table writing and reading | `data/`, `orchestration/io/method_writes.py` |
| Common IO-LCA figure policy | `figures/common.py` |
| Deterministic IO-LCA figure planning and rendering | `orchestration/figure_generation.py`, `plot/` |
| IO-LCA uncertainty public request normalization | `uncertainty/request/normalization.py` |
| Deterministic prerequisite resolution for IO-LCA uncertainty | `uncertainty/runtime/prerequisites.py` |
| LCIA public row sampling | `uncertainty/evaluation/sampling.py` |
| IO-LCA source unit evaluation for downstream ASR Sobol | `uncertainty/sobol/evaluator.py` |
| IO-LCA uncertainty run artifacts and manifests | `uncertainty/io/run_outputs.py`, `uncertainty/io/manifest_payloads.py`, `uncertainty/io/paths.py` |
| IO-LCA uncertainty source method log and README text | `uncertainty/io/source_methods.py` |
| IO-LCA uncertainty figure planning, row reading, rendering, and exact reuse rendering | `uncertainty/figures/` |
| IO-LCA uncertainty public orchestration | `uncertainty/runner.py` |

`io_lca/` does not own:

| Responsibility | Owning package |
| --- | --- |
| MRIO downloading or processing | `pyaesa/download/`, `pyaesa/process/` |
| External LCA file grammar | `pyaesa/external_inputs/` |
| ASR ratio logic | `pyaesa/asr/` |
| Monte Carlo runtime mechanics | `pyaesa/shared/uncertainty_assessment/` |
| Sobol design, accumulation, diagnostics, summaries, and common README text | `pyaesa/shared/uncertainty_assessment/` |
| LCIA CoV lookup and random variable keys | `pyaesa/shared/lcia/` |

Shared uncertainty code is family neutral. It receives IO-LCA row identities,
numeric output arrays, and family metadata from `io_lca/uncertainty/`; it must
not import IO-LCA deterministic paths, MRIO selectors, or LCIA file contracts.

## IO-LCA Uncertainty Contract

The active IO-LCA uncertainty source is `lcia_uncertainty`. It samples
deterministic IO-LCA LCIA values with:

`lower = value * (1 - cov_value); upper = value * (1 + cov_value);
sampled_value = lower + u_shared * (upper - lower)`

Public rows are sampled in their persisted deterministic selector domain.
Rows in a custom MRIO aggregation and disaggregation use that
classification's selector labels. Rows produced with `group_indices=True` keep
selector columns and use the full combined output selector label as the LCIA
uncertainty key.

| Deterministic output scope | Uncertainty driver rule |
| --- | --- |
| Direct L1 country rows | Use the selected `r_f` or `r_p` country CoV. |
| Custom classification region L1 rows | Use the region CoV from `reg_cbca_covs_agg_<agg_version>.csv`. |
| `group_indices=True` L1 rows | Use the full combined output selector label from `reg_cbca_covs_group_indices.csv` or `reg_cbca_covs_agg_<agg_version>_group_indices.csv`. |
| Direct L2 sector rows | Map public `s_p` labels to bundled sector CoV codes. |
| Custom classification sector L2 rows | Map the public `s_p` label through `sector_cov_mapping`. |
| `group_indices=True` L2 rows | Map the full combined output `s_p` label through `sector_cov_mapping`. |

The LCIA `u_shared` key is built by `pyaesa/shared/lcia/uncertainty_keys.py`.
It is scoped by project, source, aggregation scope, driver kind, and driver key.
It excludes studied year, LCIA method, impact category, and public row id so
the same uncertainty driver is linked across those outputs within a run.

`uncertainty_io_lca(...)` exposes Monte Carlo uncertainty only. It does not
accept public Sobol parameters and does not write Sobol result files. The
`uncertainty/sobol/evaluator.py` module is an IO-LCA owned source unit
evaluator used by downstream ASR Sobol so ASR can evaluate IO-LCA source
variation without importing IO-LCA sampling internals.

`refresh=True` for `uncertainty_io_lca(...)` clears the resolved IO-LCA Monte
Carlo branch and refreshes the resolved deterministic IO-LCA upstream branch.
Processed MRIO prerequisites are read and are not refreshed by IO-LCA.

## Internal Organization

| Folder | Contract |
| --- | --- |
| `compute/` | Deterministic IO-LCA numerical kernels. |
| `contracts/` | Deterministic IO-LCA functional unit and selector contracts. |
| `data/` | Deterministic paths, metadata, loaders, and writers. |
| `figures/` | Common deterministic and uncertainty IO-LCA figure policy, including LCIA impact ordering, selector tokens, and impact panel layout. |
| `orchestration/` | Deterministic mode runners, figure support, and result writing. |
| `plot/` | Deterministic IO-LCA figure writers. |
| `uncertainty/` | IO-LCA uncertainty runner with responsibility subfolders for request normalization, runtime prerequisites, evaluation, IO run materialization, manifest payloads, figures, and downstream Sobol evaluation. |

## Output And Reuse

Deterministic outputs are written under
`A_lca/io_lca/<source>__<version>/deterministic/`. IO-LCA Monte Carlo outputs
are written under
`A_lca/io_lca/<source>__<version>/monte_carlo/<run_id>/`.

Deterministic IO-LCA writes `logs/scope_manifest.json` as the discovery point
for completed public call scopes. The manifest owns `function`, `arguments`,
`execution`, `reuse`, `artifacts`, and `provenance` for each persisted scope,
including completed outputs and reuse identity. Figure scopes use the same
manifest owner and record figure paths inside the matching scope artifacts.

Required IO-LCA uncertainty artifacts are:

| Artifact | Path owner |
| --- | --- |
| `results/public_row_identity.<ext>` | `uncertainty/io/paths.py` |
| `results/lca_runs.<ext>` | `uncertainty/io/paths.py` |
| `results/summary_stats_runs.<ext>` | `uncertainty/io/paths.py` |
| `results/README.txt` | `uncertainty/io/source_methods.py` |
| `logs/source_methods.csv` | `uncertainty/io/source_methods.py` |
| `logs/scope_manifest.json` | shared manifest owner plus IO-LCA path owner |

When `figures=True`, IO-LCA uncertainty figures are written directly under
`figures/` for exact one year and multi year requests. Single year uncertainty
figures read `lca_runs` with chunked compact run matrix selection and attach
run vectors only for the planned public row ids. Multi year uncertainty
figures read `summary_stats_runs` and do not read run matrices. IO-LCA figure
panels use LCIA metadata impact ordering, one or two column impact layouts, a
zero based y axis, one default color across impact panels, and the resolved
impact unit on each y axis. Native IO-LCA figures are retrospective only.

Completed run reuse is keyed by the normalized public request, active source
parameters, deterministic prerequisite identity, and output format.
When `figures=False`, public IO-LCA functions skip rendering and preserve
existing figure files and figure metadata. Figure cleanup occurs only inside a
requested `figures=True` render for the resolved figure scope.

## Runtime Reporting Contract

| Runtime surface | IO-LCA behavior |
| --- | --- |
| Deterministic progress | Year progress is owned by `orchestration/pipeline/progress.py`. |
| Figure progress | Figure descriptions are bounded before reaching shared transient reporting. |
| Uncertainty progress | `uncertainty/runner.py` owns public orchestration; shared uncertainty progress helpers own Monte Carlo status lines. |
| Figure summaries | Public summaries use `Figures available` and `Figures folder`. |
| Refresh scope | Refresh affects only the resolved deterministic or Monte Carlo IO-LCA scope. |

## Testing And Quality Gates

Package tests for IO-LCA behavior live under `tests/package/io_lca/`.

| Scope | Required tests |
| --- | --- |
| Deterministic IO-LCA | Public deterministic tests and retained owner contracts. |
| IO-LCA uncertainty | `tests/package/io_lca/test_uncertainty_io_lca.py` through public `uncertainty_io_lca(...)` with real fixture files. |
| Shared uncertainty owners touched by IO-LCA | Corresponding tests under `tests/package/shared/uncertainty_assessment/`. |
| Shared LCIA CoV owners | `tests/package/shared/test_lcia_contracts.py`. |

Touched owners must retain 100 percent line and branch coverage. Tests should
exercise behavior reachable through public functions or realistic file flows;
branches that cannot be reached through those routes should be deleted with
their tests.
