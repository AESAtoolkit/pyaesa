# Architecture: AR6 Carrying Capacity (`ar6_cc/`)

## Purpose

`ar6_cc/` owns dynamic AR6 climate change carrying capacity extraction and
Monte Carlo uncertainty for those carrying capacity trajectories.

Deterministic AR6 CC consumes processed AR6 workbooks. AR6 CC uncertainty
consumes deterministic AR6 CC trajectory tables and delegates family neutral
Monte Carlo runtime concerns to `pyaesa/shared/uncertainty_assessment/`.

## Public Surface

| Public function | Owner | Contract |
| --- | --- | --- |
| `deterministic_ar6_cc(...)` | `pyaesa/ar6_cc/deterministic_ar6_cc.py` | Extract retained AR6 model and scenario carrying capacity trajectories from processed AR6 outputs, write the deterministic trajectory table, and render optional deterministic figures. |
| `uncertainty_ar6_cc(...)` | `pyaesa/ar6_cc/uncertainty_ar6_cc.py` | Run Monte Carlo uncertainty by sampling retained deterministic AR6 CC trajectories and writing shared uncertainty artifacts. |

Both public functions are exported at package level through `pyaesa.__init__`.

## Responsibility Boundary

`ar6_cc/` owns:

| Concern | Canonical owner |
| --- | --- |
| Deterministic AR6 CC selector normalization | `deterministic/runner.py` plus `shared/runtime/signatures.py` |
| Deterministic processed AR6 prerequisite resolution | `deterministic/runner.py` |
| Shared AR6 CC scope paths and deterministic scope tokens | `shared/runtime/paths.py` |
| Deterministic AR6 CC paths | `deterministic/io/paths.py` |
| Deterministic AR6 CC metadata | `deterministic/runtime/metadata.py` |
| Deterministic AR6 CC table IO | `deterministic/io/tables.py` |
| Deterministic AR6 CC figures | `deterministic/figures/render.py` with study and budget panels in `deterministic/figures/period_panels.py` |
| Shared AR6 CC figure titles and category colors | `shared/runtime/figure_titles.py` and `shared/runtime/figure_style.py` |
| AR6 CC uncertainty request normalization | `uncertainty/request/normalization.py` |
| AR6 CC uncertainty deterministic prerequisite loading and row preparation | `uncertainty/runtime/prerequisites.py`, `uncertainty/runtime/rows.py` |
| AR6 CC uncertainty source evaluation | `uncertainty/evaluation/sampling.py` |
| AR6 CC uncertainty summary identity grouping | `uncertainty/evaluation/summary_identity.py` |
| AR6 CC source unit evaluation for downstream aCC Sobol | `uncertainty/sobol/evaluator.py` |
| AR6 CC uncertainty paths | `uncertainty/io/paths.py` |
| AR6 CC uncertainty run artifact dispatch and manifest payloads | `uncertainty/io/run_outputs.py`, `uncertainty/io/manifest_payloads.py`, with period mapping in `uncertainty/io/period_dispatch.py` |
| AR6 CC uncertainty source method and README text | `uncertainty/io/source_methods.py` |
| AR6 CC uncertainty orchestration | `uncertainty/runner.py` |
| AR6 CC uncertainty figures | `uncertainty/figures/`, with exact reuse rendering in `uncertainty/figures/reuse.py` and study or budget panels in `uncertainty/figures/period_panels.py` |

`ar6_cc/` does not own:

| Responsibility | Owning package |
| --- | --- |
| Raw AR6 data download | `pyaesa/download/ar6/` |
| AR6 scenario processing and harmonization | `pyaesa/process/ar6/` |
| Monte Carlo runtime mechanics | `pyaesa/shared/uncertainty_assessment/` |
| aCC and ASR downstream formulas | `pyaesa/acc/` and `pyaesa/asr/` |
| Public Sobol variance decomposition | Owned by aCC and ASR through shared Sobol owners. |

## Deterministic Data Flow

`deterministic_ar6_cc(...)` resolves a consecutive study period from an
explicit year list or `range` and the matching processed AR6 scope. If the
processed workbook is missing, it creates the matching processed scope through
the AR6 processing owner for the same study period and harmonization scope.

The deterministic study period table is the canonical trajectory inventory for
downstream consumers. It is written as:

`data_processed/ar6/<processed_scope>/ar6_cc/<emission_variable_scope>/<category_tokens>__<ssp_tokens>/deterministic/results/ar6_cc.<format>`

When the requested study period ends before 2100, the same deterministic run
also writes:

`data_processed/ar6/<processed_scope>/ar6_cc/<emission_variable_scope>/<category_tokens>__<ssp_tokens>/deterministic/results/ar6_cc_post_study_period.<format>`

The post study table uses the same row identity and contains years after the
study period through 2100. It supports AR6 CC reporting and figures only. aCC
and ASR consume the study period `ar6_cc.<format>` table.

Rows are identified by:

- `cc_model`
- `cc_scenario`
- `cc_category`
- `ssp_scenario`
- `cc_flow`
- `cc_variable`
- `impact_unit`

Output years are stored as wide numeric value columns. `cc_variable` records
the exact processed AR6 variable for each row. `cc_flow` records whether a row
is a downstream carrying capacity denominator (`net_emissions` or
`positive_emissions`) or AR6 CC evidence only (`negative_sequestration`).

The unit is written as a row identity column so downstream aCC and ASR outputs
can carry the carrying capacity unit explicitly.

`emissions_mode="net"` writes only `net_emissions` rows. Gross modes write
`positive_emissions` rows from the harmonized gross variable plus signed
`negative_sequestration` companion rows from the same processed AR6 final sheet
as the selected gross variable. The companion rows are filtered to the exact
model and scenario scope retained by the selected positive emissions variable, so
variable specific AR6 gross sign filtering is preserved in AR6 CC outputs. aCC
and ASR load only the downstream denominator flows and do not propagate
negative sequestration rows into their formulas.

Category and SSP selectors are part of the deterministic filesystem scope. The
selector folder uses the exact normalized requested token sets, for example
`C1-C3__SSP1-SSP5`; non-consecutive tokens are not compressed into implied
ranges. Repeating the same selector reuses the same deterministic table.

## Uncertainty Data Flow

`uncertainty_ar6_cc(...)` accepts `base_ar6_cc_args`, `uncertainty_config`,
`output_format`, `figures`, `figure_options`, `figure_format`, and `refresh`.

The uncertainty owner:

1. normalizes the deterministic selector envelope and the
   `dynamic_ar6_cc_uncertainty` source block
2. calls or reuses `deterministic_ar6_cc(...)` with figures disabled
3. reads the deterministic study period trajectory table and the post study
   companion table when it exists
4. converts retained trajectory pools into compact NumPy arrays across all
   requested study and post study years
5. samples trajectories once per run batch
6. dispatches the same sampled run stream into study period, post study
   period, and cumulative budget artifacts

Uncertainty public row identity is:

- `cc_model`
- `cc_scenario`
- `cc_category`
- `ssp_scenario`
- `cc_flow`
- `cc_variable`
- `impact_unit`
- `year`

`cc_model` and `cc_scenario` are sampled source state columns. They stay in
`public_row_identity` so run rows can be joined back to the selected AR6
trajectory. Exact summaries group them out because public summaries report the
sampled carrying capacity distribution rather than one distribution per
candidate trajectory.

Gross mode uncertainty samples one retained model and scenario trajectory and
materializes all AR6 CC flows belonging to that selected trajectory. This keeps
positive emissions and negative sequestration companion rows tied to the same
selected AR6 pathway.

The AR6 CC uncertainty source block is `dynamic_ar6_cc_uncertainty`.

`uncertainty_ar6_cc(...)` exposes Monte Carlo uncertainty only. It does not
accept public Sobol parameters and does not write Sobol result files. The
`uncertainty/sobol/evaluator.py` module is an AR6 CC owned source unit
evaluator used by downstream aCC Sobol so aCC can evaluate AR6 CC source
variation without importing AR6 CC sampling internals.

`refresh=True` for `uncertainty_ar6_cc(...)` clears the resolved AR6 CC Monte
Carlo branch, refreshes the deterministic AR6 CC upstream branch, and refreshes
the matching `process_ar6(...)` prerequisite through the deterministic owner.

Supported source parameters:

| Parameter | Contract |
| --- | --- |
| `sampling_method` | `"srs"` samples trajectories. `"lhs"` samples model, then scenario. |
| `category_uncertainty` | When true, run one category and group out `cc_category`. |

`cc_runs` uses sparse selected rows with `run_index`, `public_row_id`, and
`cc`. `public_row_identity` is trajectory resolved, so joining `cc_runs` to
`public_row_identity` exposes the selected `cc_category`, `cc_model`,
`cc_scenario`, `ssp_scenario`, `cc_flow`, `cc_variable`, `impact_unit`, and year
for each run row. Post study run artifacts use the same layout and the same
run indices when a post study period exists.
`summary_stats_runs` groups out `cc_model` and `cc_scenario` for all runs. It
also groups out `cc_category` when `category_uncertainty` is true and reports
the integrated category uncertainty distribution by `ssp_scenario`,
`cc_flow`, `cc_variable`, `impact_unit`, and `year`.

The Monte Carlo run folder is:

`data_processed/ar6/<processed_scope>/ar6_cc/<emission_variable_scope>/<category_tokens>__<ssp_tokens>/monte_carlo/<run_id>/`

Required artifacts:

| Artifact | Path |
| --- | --- |
| Public row identity | `results/public_row_identity.<suffix>` |
| Run values | `results/cc_runs.<suffix>` |
| Exact summary | `results/summary_stats_runs.<suffix>` |
| Post study public row identity | `results/post_study_period_public_row_identity.<suffix>` when the deterministic scope has post study years |
| Post study run values | `results/post_study_period_cc_runs.<suffix>` when the deterministic scope has post study years |
| Post study exact summary | `results/post_study_period_summary_stats_runs.<suffix>` when the deterministic scope has post study years |
| Period budget row identity | `results/study_and_post_study_period_budget_row_identity.<suffix>` |
| Period budget run values | `results/study_and_post_study_period_budget_runs.<suffix>` |
| Period budget exact summary | `results/study_and_post_study_period_budget_summary_stats.<suffix>` |
| Result guide | `results/README.txt` |
| Source methods | `logs/source_methods.csv` |
| Run manifest | `logs/scope_manifest.json` |
| Figures | `figures/` when `figures=True` |

Deterministic and uncertainty AR6 CC manifests use the package manifest
contract: `function`, normalized public `arguments`, `execution`, `reuse`,
`artifacts`, and `provenance`. Deterministic reuse keys are derived from the
normalized AR6 CC selector arguments. Uncertainty compatibility keys are
derived by the shared uncertainty manifest owner from arguments, active sources,
Monte Carlo parameters, Sobol parameters, prerequisites, and output format.

Figure paths inherit the processed AR6 CC scope from the run root. That scope
owns emissions mode, emission type, AFOLU selection, subset version, and the
processed year range. AR6 CC figure stems use the SSP token, with
`cat_<category>` appended only for category resolved uncertainty figures.
Impact unit and `cc_flow` are not figure stem tokens. The y axis carries the
impact unit, and gross mode figures keep gross emissions and negative
sequestration in the same SSP or category figure. Pathway panels span the
study period and, when present, post study years through 2100. A vertical
transition marker and light purple shaded region identify the post study period. The
right panel reports cumulative budgets for the study period and post study
period when both exist. Deterministic figures encode gross emissions as solid
lines and negative sequestration as dotted lines. Deterministic negative
sequestration rows are plotted only from the first negative year and are not
rendered when they are zero across the visible period. Uncertainty figures encode
gross emissions and negative sequestration with distinct band colors while
keeping the band geometry legend neutral.

Supported uncertainty output formats are `csv_compact` and `parquet`.

## Runtime Reporting Contract

| Runtime surface | AR6 CC behavior |
| --- | --- |
| Deterministic progress | `deterministic/runner.py` reports pathway processing through shared status lines. |
| Uncertainty progress | `uncertainty/runner.py` reports scope availability; shared uncertainty progress helpers report Monte Carlo convergence status lines. |
| Figure progress | Figure owners use shared bounded figure progress. |
| Figure summaries | Public summaries use `Figures available` and `Figures folder`. |
| Refresh scope | Refresh affects only the resolved AR6 CC deterministic or uncertainty scope. |

## Internal Organization

| Folder | Contract |
| --- | --- |
| `deterministic/` | Deterministic runtime, paths, metadata, table IO, figures, and reports. |
| `shared/` | AR6 CC selector, path, title, and signature helpers shared by deterministic consumers. |
| `uncertainty/` | AR6 CC family uncertainty runner with responsibility subfolders for request normalization, runtime prerequisites, evaluation, IO, and downstream Sobol evaluation. |
| `uncertainty/figures/` | AR6 CC uncertainty figure planning, public table reading, rendering, and figure manifest updates. |

The family code contains only AR6 CC scientific logic and AR6 CC path or
metadata contracts. Shared uncertainty code remains family neutral and must not
import AR6 CC modules.

## Testing And Quality Gates

Package tests for AR6 CC live under `tests/package/ar6_cc/`.

Required gates for this package:

- public deterministic tests through `deterministic_ar6_cc(...)`
- public uncertainty tests through `uncertainty_ar6_cc(...)`
- `csv_compact` and parquet uncertainty output coverage
- fixed and convergence Monte Carlo coverage
- 100 percent line and branch coverage for touched owners
- scoped `ruff check`, `ruff format --check`, and `pyright`

Tests must exercise real public functions or realistic file flows. Branches
that cannot be reached through those routes should be deleted rather than
tested through private patching.
