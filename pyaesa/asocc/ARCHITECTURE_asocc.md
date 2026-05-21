# Architecture: aSoCC (`pyaesa/asocc/`)

## Purpose

The `pyaesa.asocc` package owns allocated shares of carrying capacities. It
provides deterministic allocation, deterministic disaggregation, method weight
export, figure generation, and the aSoCC Monte Carlo layer used by uncertainty
entry points.

This document is for external Python contributors. It describes the code
architecture, the canonical owners of each responsibility, and the boundaries
that must stay stable when allocation methods or runtime behavior are changed.

## Public Surface

The package level public API is exported through `pyaesa.__init__`. The aSoCC
public functions are:

| Public function | Owner module | Responsibility |
| --- | --- | --- |
| `deterministic_asocc(...)` | `pyaesa/asocc/deterministic_asocc.py` | Public deterministic allocation entry point and deterministic output publisher. |
| `disaggregate_asocc(...)` | `pyaesa/asocc/disaggregate_asocc.py` | Public deterministic transform over existing deterministic aSoCC outputs. |
| `uncertainty_asocc(...)` | `pyaesa/asocc/uncertainty_asocc.py` | Public Monte Carlo aSoCC entry point built from deterministic row identity. Runtime orchestration is owned by `uncertainty/engine/runner.py`. |
| `write_asocc_weight_template(...)` | `pyaesa/asocc/inter_method_weights.py` | Public writer for the editable inter-method probability template. |
| `preview_asocc_weight_tree(...)` | `pyaesa/asocc/inter_method_weights.py` | Public validator and preview renderer for edited inter-method probability trees. |

Non public modules under `pyaesa/asocc/` must not be imported by user code.
When a new top level public function is added intentionally, update
`pyaesa.__init__`, `docs/api.rst`, and this document in the same change set.

## Responsibility Boundary

`pyaesa.asocc` owns:

| Area | Canonical owner |
| --- | --- |
| Public deterministic API | `deterministic_asocc.py` |
| Deterministic runtime orchestration | `orchestration/run_allocate.py`, `orchestration/setup/run_setup.py` |
| Request normalization and public selector validation | `entrypoints/argument_contracts.py`, `runtime/selection/`, `orchestration/setup/request/`, `orchestration/setup/validation/` |
| Method registry and method capability discovery | `methods/registry/registry.py`, `methods/registry/build/build.py`, `methods/registry/queries/` |
| L1 and L2 scientific equations | `methods/compute_l1.py`, `methods/compute_l2.py`, `methods/run_ar.py`, `methods/run_ut.py`, `methods/equations/` |
| Yearly routing and scenario execution | `orchestration/yearly/` |
| Projection and historical reuse | `orchestration/projection/` |
| Public output schemas and write paths | `runtime/output/contracts.py`, `runtime/paths/published.py`, `orchestration/write/` |
| Deterministic path scope and downstream path reuse | `runtime/scope/branch_resolution.py`, `runtime/paths/deterministic.py`, `runtime/paths/published.py`, `disaggregation/paths.py` |
| Deterministic scope metadata and reuse decisions | `io/metadata.py`, `orchestration/setup/reuse/completed_run_policy.py` |
| Deterministic figures | `figures/` |
| Disaggregation | `disaggregation/` |
| aSoCC uncertainty | `uncertainty/` |

`pyaesa.asocc` does not own:

| Area | Owner |
| --- | --- |
| Downloading MRIO, population, GDP, or AR6 data | `pyaesa/download/` |
| Processing raw MRIO files into package ready matrices | `pyaesa/process/` |
| Workspace creation and workspace root selection | `pyaesa/workspace_initialisation/` |
| Downstream aCC and ASR scientific calculations | `pyaesa/acc/`, `pyaesa/asr/` |
| IO-LCA numerator calculations | `pyaesa/io_lca/` |

Cross package changes are allowed only when the public data contract requires
them. Downstream packages must consume the canonical aSoCC public contract
directly.

## Package Layout

| Path | Role |
| --- | --- |
| `data/` | Read processed MRIO, LCIA, population, GDP, source metadata, and unit metadata. |
| `disaggregation/` | Load deterministic aSoCC branches and publish disaggregated outputs. |
| `entrypoints/` | Normalize public arguments before orchestration. |
| `figures/` | Render deterministic figures from published deterministic rows. |
| `inter_method_tools/` | Plan, validate, serialize, and render the inter-method probability tree. |
| `io/` | Metadata models, logging setup, and format specific IO used by aSoCC. |
| `methods/` | Method registry, equation dispatch, and method family equations. |
| `orchestration/` | Setup, yearly execution, projection, recomposition, and writing. |
| `runtime/` | Path contracts, scope identity, selection helpers, and public output contracts. |
| `uncertainty/` | aSoCC Monte Carlo owners that depend on deterministic public row identity. |

The root package keeps public function modules and this architecture note. New
implementation helpers belong in the existing purpose based subpackages. Path
resolution must stay in dedicated path modules such as `runtime/paths/published.py`
and `runtime/paths/deterministic.py`; avoid resolving output paths in scientific
equation code.

## Deterministic Data Flow

The deterministic public call follows one canonical pipeline:

1. `deterministic_asocc(...)` normalizes public arguments and delegates to the
   allocation runner.
2. `_prepare_context(...)` validates source, grouping, FU, selectors, filters,
   LCIA requirements, projection routes, and scope metadata before compute.
3. The completed run policy either returns a complete compatible deterministic
   scope, a missing compute subset, or a clear incompatibility error.
4. `_process_year(...)` loads the required processed inputs for one year and
   dispatches each scenario partition to yearly owners.
5. L1 and L2 method owners compute scientific results and push indexed wide
   frames into scope local state.
6. Write owners convert indexed frames through `OutputSpec` and
   `OutputArtifact`, then persist public wide tables.
7. Metadata owners write one deterministic `logs/scope_manifest.json` file for
   each public call scope.
8. Figure owners render figures only when requested and only from exact figure
   scope identity.

No public call repairs incomplete outputs from an interrupted prior run. A user
can request `refresh=True`, which deletes only the resolved deterministic
aSoCC source scope after active file handlers are closed.

## Scope Identity And Reuse

`logs/scope_manifest.json` is the deterministic discovery point for published
aSoCC tables. The completed run policy reads the scope manifest before compute
starts.

Each per scope manifest has one canonical public payload:

| Field | Contract |
| --- | --- |
| `function` | Public entry point name, `deterministic_asocc`. |
| `arguments` | Normalized public arguments that reproduce the scope. |
| `execution` | Status, requested years, resolved years, completed years, skipped years, and timestamp. |
| `reuse` | Stable identity key for exact scope reuse. |
| `artifacts` | Published output paths, figure paths, and regression stats paths when applicable. |
| `provenance` | Selected methods, functional unit, filters, SSP scope, reference years, and projection route. |

Deterministic output path scope is limited to project output root, published
aSoCC source label, and group version. `AsoccDeterministicPathScope` carries only
those path dimensions. Scientific and request identity dimensions such as
`group_reg`, `group_sec`, `aggreg_indices`, `l1_reg_aggreg`, filters, method
selection, projection settings, and selector axes are persisted in `arguments`
and validated by the completed run policy.

An aSoCC project is functional unit scoped. All deterministic aSoCC outputs
under one `<project_name>/B1_asocc/` tree must use the same
`fu_code`, across all native sources, grouped source versions, and output
sources. A different `fu_code` requires a different `project_name`, unless the
existing aSoCC outputs for that project are manually removed before the call.
Function level `refresh=True` is source scoped; it does not authorize
mixing another FU with deterministic aSoCC outputs that remain elsewhere in the
same project tree.

Deterministic table reuse compares these extendable axes by set relation:

| Extendable axis | Meaning |
| --- | --- |
| `years` | Public studied years. |
| `l2_reuse_years` | Effective historical L2 reuse years, user supplied or route planned. |
| `reference_years_input` | User requested AR reference years. |
| `lcia_methods` | Requested LCIA method set. |
| `ssp_scenario_input` | Requested SSP scenarios. |
| `selected_methods` | Selected L1 methods, L2 one step methods, and L2 in L1 pairs. |

When aSoCC SSP scenario identity is materialized as a row column for figures,
disaggregation, uncertainty, aCC, or ASR, the canonical column name is
`asocc_ssp_scenario`. The public request argument and internal deterministic
year routing variable remain `ssp_scenario`. External aSoCC Monte Carlo staged
files must use `asocc_ssp_scenario`; deterministic external aSoCC staged files
keep SSP scenario ownership in the filename stem and must not provide an
`asocc_ssp_scenario` row column.

Exact identity fields inside one functional unit project are source, studied
filters, grouping, aggregation mode, output format, public schema, and
deterministic value semantics. Non append identity fields must match before
compute or write because they share the same deterministic output folder.
`reg_window` is exact whenever selected routes compute regression outputs,
because it changes fitted coefficients and projected values.

Reuse rules:

| Request relation to persisted scope | Behavior |
| --- | --- |
| Exact or subset on every extendable axis | Reuse existing tables. |
| Exact or strict superset on an extendable axis | Compute missing rows and update metadata. |
| Mixed subset and superset axes | Incompatible with append. Use a distinct scope or refresh. |
| Partial overlap on any extendable axis | Incompatible with append. |

Figure reuse is stricter than table reuse. Figures require exact compute
signature identity, including the LCIA method selector used for rendering. A
subset or superset table scope does not authorize figure reuse. Recomputing a
figure scope replaces prior figures for that figure request.

Deterministic figure generation uses these owners:

| Owner | Responsibility |
| --- | --- |
| `figures/scope_planner.py` | Resolve the exact persisted deterministic table scope and requested SSP figure scopes. |
| `figures/row_reader.py` | Read one persisted table at a time, melt only requested years, attach row owned method identity, and preserve invariant historical rows for SSP expansion. |
| `pyaesa/shared/figures/deterministic_variant_compressor.py` | Select retained deterministic reference year and `l2_reuse_year` variant combinations for deterministic figure display. |
| `figures/product_renderers.py` | Coordinate shared variant styling, transition markers, integer year axes, figure saving, and render multi-method products before per method products. |
| `figures/per_method_renderer.py` | Plan and render method scoped deterministic products from prepared rows. |
| `figures/multi_method_renderer.py` | Plan method comparison products and repeat LCIA generic external rows into each LCIA impact comparison scope. |
| `pyaesa/shared/figures/` | Own reusable deterministic legend grouping, transition marker geometry, and figure scope helpers used by aSoCC figures. |

Figure code must consume persisted deterministic rows and must not recompute
scientific allocation results. Row readers may reshape and scope table rows for
rendering, but scientific values remain owned by deterministic compute and write
owners.

Append execution keeps one compute context per public deterministic call. Pure
year append computes missing public years. Selector axis append computes missing
selector axes. Mixed year plus selector append computes the requested rectangle
because exact missing cell execution would require splitting one public call into
multiple internal compute contexts.

## Method Registry Contract

The method registry is the source of truth for method identity and capability
discovery.

| File | Responsibility |
| --- | --- |
| `methods/registry/specs/l1.py` | Declarative L1 method rows. |
| `methods/registry/specs/l2.py` | Declarative L2 method rows per FU and route form. |
| `methods/registry/specs/all_specs.py` | Combined method spec inventory. |
| `methods/registry/registry.py` | Registry facade and supported family inventory. |
| `methods/registry/build/build.py` | Registry assembly from declarative specs and validation owners. |
| `methods/registry/model/types.py` | Typed method spec model and FU normalization. |
| `methods/registry/queries/queries.py` | Read only registry query facade. |
| `methods/registry/queries/resolve.py` | User label resolution into canonical registry labels. |
| `methods/registry/model/input_requirements.py` | Method family input coverage and enacting metric requirements. |
| `methods/registry/model/family_checks.py` | Registry integrity validation at import time. |

Registry rows are declarative. Scientific behavior belongs in equation and route
owners. Do not encode calculation behavior in selection code or write code.

## Method Tooling And Equation Owners

Inter-method probability tree responsibilities are split by artifact boundary:

| Owner | Responsibility |
| --- | --- |
| `inter_method_tools/tree.py` | Candidate discovery from normalized selectors, inter-method parameter validation, tree node planning, sibling edge validation, and canonical tree CSV paths. |
| `inter_method_tools/tree_artifacts.py` | Editable CSV and guide text writing. |
| `inter_method_tools/tree_figure.py` | Probability tree rendering. |
| `inter_method_weights.py` | Public export and preview orchestration only. |

AR method support code separates scientific runtime from output shape helpers:

| Owner | Responsibility |
| --- | --- |
| `methods/run_ar.py` | AR L1 and L2 runtime dispatch, cache use, and reference year routing. |
| `methods/equations/ar_result_indexing.py` | Impact and reference year index level attachment for computed AR results. |
| `methods/equations/ar_nan_outputs.py` | NaN placeholder result shapes for studied years before an AR reference year. |

L2 preweight multiplication uses one vectorized owner,
`orchestration/yearly/l2/l2_batch_weighting.py`. It owns numeric alignment
plans, grouped aggregation with pandas `sum(min_count=1)` semantics, historical
reuse batching, and AR/UT matrix kernels. It exposes matrix oriented kernels to
yearly compute owners; caller side conversion wrappers are not part of the
runtime contract.

Important registry fields:

| Field | Meaning |
| --- | --- |
| `name` | Canonical public method label. |
| `level` | `L1` or `L2`. |
| `fu_code` | Supported L2 FU for L2 rows, or `None` for L1 rows. |
| `l1_weighting` | Whether an L2 support share is combined with an L1 method. |
| `needs_*` flags | Required processed input families. |
| `indices` | Public identity axes owned by the method output. |
| `l1_kind` | LCIA boundary kind required by an L1 method or by an L2 support route. |
| `l2_weight_axis` | L1 axis used when an L2 support share is multiplied by L1 output. |

## Selection And Pairing

Public selectors are normalized under `runtime/selection/` and
`orchestration/setup/request/selection.py`.

Supported public modes:

| `method_plan` | Meaning |
| --- | --- |
| `default` | Package default methods for the selected FU. |
| `one_step` | Direct L2 methods only. |
| `two_steps` | L2 support shares combined with compatible L1 methods. |
| `pairs` | Explicit `L1::L2` pair selection. |
| `one_step_pairs` | Mix direct one step L2 methods with explicit pairs. |

Pairing rules are registry driven:

| Pair case | Rule |
| --- | --- |
| Neutral L1 with L2 support | Allowed when the L2 support route has no conflicting LCIA kind. |
| LCIA L1 with L2 support | Allowed only when `l1_kind` matches the L2 support requirement. |
| AR L1 with AR L2 same pair | Use the canonical direct AR L2 route. |
| Source `iso3` | L1 only, with no grouping, LCIA, or reference year selector. |

Selection code should decide reachability only. It must not reshape scientific
outputs or correct equation output axes.

## Scientific Equation Ownership

Scientific calculations are owned by method family modules:

| Family | Owner |
| --- | --- |
| L1 dispatch | `methods/compute_l1.py` |
| L2 dispatch | `methods/compute_l2.py` |
| AR family | `methods/run_ar.py`, `methods/equations/ar_*.py` |
| UT family | `methods/run_ut.py`, `methods/equations/ut_*.py` |
| PR and EG family | `methods/equations/pr_*.py`, `methods/equations/eg_*.py` |
| LCIA input shaping | `methods/lcia_inputs.py`, `data/run_lcia.py`, `data/reference_payloads.py` |

Equation functions should be pure for their input frames and route parameters.
They should return indexed frames with the correct scientific identity. Writers
must not infer missing scientific identity from filenames or patch malformed
equation output.

Input corruption should fail at loaders or cleaners. If a loader guarantees an
invariant, do not duplicate the same defensive error later in equation,
recomposition, or writer code.

## Projection, Reuse, Reference Years, And SSP Routing

Projection is planned before yearly computation by
`orchestration/projection/config/` and expanded by setup year planning.

| Route | Owner | Contributor rule |
| --- | --- | --- |
| Regression projection | `orchestration/projection/regression/` | Declare fitted target series, valid fit window, diagnostics rows, and clipping behavior at the regression owner. |
| Historical reuse | `orchestration/projection/reuse/` | Declare effective L2 reuse year and output subfolder ownership. |
| Projection payloads | `orchestration/projection/payload/` | Build route payloads used by yearly computation. |
| Reference years | `data/reference_payloads.py`, AR equation owners | Reference year axes are admitted only for methods that own a reference year dependency. |
| SSP routing | `orchestration/yearly/shared/scenario_routing.py`, `orchestration/yearly/shared/scenario_processing.py` | SSP partitions contain scenario specific rows plus scenario invariant rows repeated into the partition where required. |

Projection diagnostics are emitted only for routes that use regression. Historic
reuse routes must not produce regression diagnostics. Scenario file routing must
derive the expected deterministic stem from route rules before reading.

## Output Publication

Public deterministic tables are wide tables. Identifier columns describe method
and row identity; year columns store values.

Publication contracts:

| Contract | Owner |
| --- | --- |
| Output descriptors | `runtime/output/contracts.py` |
| Public path roots | `runtime/paths/published.py` |
| Deterministic path helpers | `runtime/paths/deterministic.py` |
| Output writing | `orchestration/write/writers/allocations.py` |
| Metadata payload | `orchestration/write/metadata/payload.py` |
| Regression diagnostics writing | `orchestration/write/regression_stats/write.py` |

`OutputSpec` owns file stem, route, scenario token behavior, method columns, and
identifier columns. `OutputArtifact` owns the validated wide DataFrame passed to
the writer. New methods should change output contracts only when their public
identity axes require it.

Supported deterministic output formats remain `csv`, `pickle`, and `parquet`
unless the public API is changed explicitly with matching tests and docs.

## Intermediate Outputs

`intermediate_outputs=True` writes additional enacting metrics, preweights,
propagation metrics, and diagnostics that are useful for method inspection.
These outputs are for user audit only and are not used by downstream package
functions.
`intermediate_outputs=False` preserves final deterministic publication without
writing those additional families.

Intermediate output ownership lives under:

| Area | Owner |
| --- | --- |
| Enacting metric recording | `orchestration/yearly/enacting_metric/` |
| Enacting output writing | `orchestration/write/writers/enacting_metric.py` and `orchestration/write/writers/enacting_metric_units.py` |
| Propagation and preweight writing | `orchestration/write/` and related yearly owners |
| Regression diagnostics | Regression projection owners plus `orchestration/write/regression_stats/write.py` |

Do not make final public table correctness depend on an optional intermediate
output file.

## Disaggregation Boundary

`disaggregate_asocc(...)` reads deterministic aSoCC outputs and writes
disaggregated deterministic outputs. It does not run allocation equations.
Disaggregation publishes final L2 rows only.

Contributors must review `disaggregation/` when a method change affects:

| Change | Reason |
| --- | --- |
| Published method labels | Disaggregation method eligibility may depend on labels. |
| Public identity columns | Disaggregation readers must match deterministic row identity. |
| Source labels or manifests | Disaggregated output reuse depends on scope metadata. |
| L2 public row semantics | Disaggregation publishes final L2 rows only. |

Disaggregation should use deterministic public row identity directly. Do not add
private translation layers for alternate aSoCC shapes.

## Uncertainty Boundary

`uncertainty_asocc(...)` depends on deterministic aSoCC as its canonical row
identity source. Deterministic method changes require uncertainty review when
they alter method discovery, public row identity, projection route identity,
LCIA method behavior, reference year behavior, L2 reuse year behavior, or SSP
partition behavior.

`refresh=True` for `uncertainty_asocc(...)` clears the resolved aSoCC Monte
Carlo branch and refreshes the resolved deterministic aSoCC upstream branch for
the request before recomputing uncertainty outputs.

aSoCC uncertainty source owners are:

| Source | Owner |
| --- | --- |
| LCIA | `uncertainty/sources/lcia.py` |
| Projection | `uncertainty/sources/projection.py` |
| Reference year | `uncertainty/sources/reference_year.py` |
| Inter-MRIO eligibility | `uncertainty/sources/inter_mrio_eligibility.py` |
| Inter-MRIO planning and run evaluation | `uncertainty/sources/inter_mrio.py` |
| Inter-MRIO route reporting | `uncertainty/sources/inter_mrio_reporting.py` |
| Inter-method | `uncertainty/sources/inter_method.py` |

LCIA uncertainty is active only when the resolved public aSoCC row universe
contains pyaesa owned LCIA dependent target rows after external method rows
are excluded. When a request includes LCIA uncertainty but no such target rows
exist, the source scope owner removes LCIA from the active source list before
reuse, manifest construction, source method logging, Monte Carlo evaluation,
and Sobol source evaluation. Source activation decisions are recorded in the
manifest and returned summary; they are not emitted as standalone live status
messages.

For `aggreg_indices=True`, deterministic public rows keep selector columns and
write full aggregate selector labels. LCIA uncertainty uses those labels as the
active CoV keys: L1 country keys are read from `reg_cbca_covs_aggreg_indices.csv`
or `reg_cbca_covs_group_<group_version>_aggreg_indices.csv`, and L2 sector keys
are resolved through `sector_cov_mapping`.

aSoCC uncertainty engine owners are:

| Owner | Responsibility |
| --- | --- |
| `uncertainty/engine/runner.py` | Public `uncertainty_asocc(...)` orchestration. |
| `uncertainty/engine/phase_reporting.py` | Direct aSoCC uncertainty phase printing and `composite_phase_index.json` entries for deterministic aSoCC and aSoCC uncertainty phases. |
| `uncertainty/engine/planning.py` | Row universe, active source, source plan, sampling plan, and memory bounded batch size assembly. |
| `uncertainty/engine/reuse/reuse.py` | aSoCC completed run discovery, Monte Carlo reuse, Sobol reuse, and appendable run selection. |
| `uncertainty/engine/monte_carlo/run_execution.py` | Selection of fixed, convergence, and sparse inter-method execution paths. |
| `uncertainty/engine/monte_carlo/fixed_batches.py` | Fixed compact Monte Carlo run execution. |
| `uncertainty/engine/convergence/convergence.py` | Monte Carlo convergence batch execution. |
| `uncertainty/engine/convergence/state.py` | Convergence replay, transient cache state, and convergence statistic calculation. |
| `uncertainty/engine/monte_carlo/batch_sizing.py` | Memory planning row count for compact and sparse run batches. |
| `uncertainty/engine/inter_method/execution.py` | Inter-method branch execution planning, including external branch plan selection for one sampled method label. |
| `uncertainty/engine/inter_method/sampling.py` | Sparse Monte Carlo and Sobol sampling for inter-method selected branches. |
| `uncertainty/engine/inter_method/identity.py` | Sparse inter-method row identity, public row id mapping, and sparse row concatenation. |
| `uncertainty/engine/sobol/runner.py` | aSoCC Sobol orchestration and active source unit evaluation. |
| `uncertainty/engine/sobol/scope.py` | Selected output year scoping and inter-MRIO alternate plan filtering for Sobol evaluation. |
| `uncertainty/engine/sobol/summary.py` | aSoCC selector summary levels and SSP invariant summary policy. |
| `uncertainty/engine/sobol/reporting.py` | aSoCC specific Sobol README notes and selected year method payload entries. |
| `uncertainty/figures/reuse.py` | Figure rendering for exact reused Monte Carlo runs, with figure paths recorded in the run manifest. |
| `uncertainty/io/source_methods.py` | aSoCC source method log schema and writer. |

Uncertainty code must read deterministic method identity from the registry or
from public deterministic rows. It must not maintain an independent method
taxonomy.

The inter-method tree is owned by `inter_method_tools/` and used by
`write_asocc_weight_template(...)`, `preview_asocc_weight_tree(...)`, and
`uncertainty_asocc(...)`. The tree classifies package and external methods from
the scientific method labels, assigns equal weights by branch hierarchy, and
stores editable sibling edge weights in
`B1_asocc/preview_inter_method_weights/`. The default export writes
`equal_weights.csv`, `README_inter_method_weights.txt`, and a
`probability_tree__equal_weights` figure directly in that folder. Custom
versions use `weights__<version_name>.csv` and
`probability_tree__<version_name>` directly in the same folder. The
`uncertainty_asocc(...)` run local tree under
`figures/inter_method_tree/` uses the same CSV and figure naming policy, with
the active uncertainty figure format and DPI. For L2 scopes, a sharing
principle with both multi step and one step candidates uses `m_s` and `o_s`
branches before splitting weights within each path.

The inter-MRIO source owner evaluates only eligible final non LCIA rows. It
decides alpha interpolation at studied year scope: a year is interpolated only
when every eligible row has a matched alternate row and the eligible rows share
one identical deterministic time route on the main and alternate endpoints.
Skipped years keep the main deterministic values for all eligible rows and are
reported by the same route matching owner used for run evaluation.

aSoCC Sobol uses the family neutral shared Sobol design, accumulator, and
runner owners, plus the shared diagnostics, summary, summary level, and
reporting owners. The aSoCC layer owns only selected output year scoping,
scientific evaluation of active aSoCC source units into public aSoCC values,
and the presentation of selected output aSoCC selectors in
`sobol_source_summary`. Common Sobol README wording and method metadata shape
belong in shared uncertainty code. aSoCC contributes only aSoCC specific notes
such as SSP invariant summary behavior and selected output years. Sobol method
metadata belongs in `README_sobol.txt` and `logs/scope_manifest.json`; aSoCC
Sobol does not write a separate method table.

## External Inputs

External aSoCC and LCA inputs are prepared through package public input
preparation routes, not by direct edits inside `asocc/`. When deterministic
aSoCC accepts an external source label or disaggregated source label, the public
row identity must match the native deterministic table contract for the selected
function family.

`uncertainty_asocc(...)` resolves declared external aSoCC methods before Monte
Carlo compatibility checks. Monte Carlo external files are used directly when a
complete run file set exists. Otherwise matching deterministic external files
are loaded once and repeated over runs. Both external input modes are recorded
in `logs/scope_manifest.json` through `external_inputs`, including selected
file paths, sizes, and SHA-256 checksums. These manifest rows participate in the
Monte Carlo compatibility key so edits to external input files cannot reuse a
completed run whose staged external content digest differs from the current
request.

Any change to external input schemas must update the external template writers,
template explanations, package docs, and tests in the same change set.

## Runtime Reporting Contract

| Runtime surface | aSoCC behavior |
| --- | --- |
| Deterministic progress | Year progress is owned by deterministic orchestration. |
| Monte Carlo progress | `uncertainty/engine/runner.py` owns public phase ordering and shared progress helpers; convergence status is recorded in the manifest and returned summary. |
| Inter-MRIO notes | Route skip messages are emitted by `uncertainty/sources/inter_mrio_reporting.py`. |
| Figure summaries | Public summaries use `Figures available` and `Figures folder`. |
| Refresh scope | Refresh affects only the resolved aSoCC output scope. |

## Contributor Testing Gate

Package tests for aSoCC live under `tests/package/asocc/`. Scientific allocation
validation lives under `tests/allocation_equation_validation/`.

For code changes:

1. Run `python -m ruff check <touched paths>`.
2. Run `python -m ruff format --check <touched paths>`.
3. Run `python -m pyright <touched package paths>`.
4. Run targeted `pytest` with `--cov=<touched owner> --cov-branch`.
5. Keep touched owners at 100 percent line and branch coverage.
6. Run allocation equation validation when method logic, pair logic, FU support,
   route planning, or public row identity changes.

Tests must exercise behavior reachable from public functions or realistic file
flows. Do not keep code branches solely to satisfy tests. If a branch cannot be
reached through normal public behavior and validated fixtures, delete the branch
and delete the test.

## Contributor Rules

1. Make changes at the canonical owner.
2. Keep scientific logic explicit and local to the method or route owner.
3. Do not add compatibility layers for unpublished behavior.
4. Do not add restart recovery, stale metadata repair, or output scanning to
   infer completed deterministic work.
5. Do not duplicate validation already enforced by an upstream owner.
6. Keep path construction in path helper modules.
7. Keep public wide output shape stable unless a public API change is approved.
8. Update `docs/ADDING_METHODS_checklist.md` whenever method addition steps
   change.
