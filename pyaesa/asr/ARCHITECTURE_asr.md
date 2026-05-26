# Architecture: Absolute Sustainability Ratio (`asr/`)

## Purpose

`asr/` owns absolute sustainability ratio computation. ASR combines an LCA
numerator with an allocated carrying capacity denominator using the public row
semantics and unit conversion contract owned by the deterministic ASR runtime.

## Public Surface

| Public function | Canonical owner | Responsibility |
| --- | --- | --- |
| `deterministic_asr(...)` | `pyaesa/asr/deterministic_asr.py` | Resolve deterministic aCC and LCA prerequisites, align numerator and denominator rows, compute ASR values, and write deterministic tables and figures. |
| `uncertainty_asr(...)` | `pyaesa/asr/uncertainty_asr.py` | Resolve upstream aCC uncertainty and LCA uncertainty inputs, compute ASR run values, write summaries, and run optional Sobol analysis. |

Both functions are exported at package level through `pyaesa.__init__`.

## Responsibility Boundary

`asr/` owns:

- ASR public function orchestration
- LCA numerator route selection for pyaesa owned IO-LCA and external LCA
- ASR numerator denominator row alignment
- ASR unit compatibility and supported unit conversion
- ASR value computation
- deterministic ASR figures
- ASR Monte Carlo run outputs, summaries, manifests, source methods, and
  Sobol family notes

`asr/` consumes but does not own:

- aSoCC allocation methodology
- aCC denominator methodology
- AR6 CC dynamic carrying capacity methodology
- pyaesa owned IO-LCA methodology
- external LCA file grammar and validation
- shared Monte Carlo and Sobol runtime mechanics

Broken upstream table schemas, manifest payloads, or source method contracts
must be fixed in the upstream owner. ASR code must not duplicate late
defensive validation for invariants already guaranteed by those owners.

## Internal Organization

| Path | Role |
| --- | --- |
| `deterministic/` | Deterministic ASR runner, prerequisite resolution, row computation, dynamic cumulative assembly, state, reports, and figures. |
| `shared/` | ASR path helpers, LCA route request normalization, deterministic external LCA helpers, component unit conversion contracts, and ASR figure shared utilities. |
| `uncertainty/` | ASR uncertainty runner with responsibility subfolders for IO, runtime component inputs, checkpoint orchestration, LCA sources, source keys, vectorized run evaluation, summaries, manifests, source methods, figures, and Sobol evaluation. The `uncertainty/evaluation/` package is split by owner: `planning.py` builds the ASR plan from upstream manifests, `alignment.py` owns numerator denominator row alignment, `runs.py` owns yearly ASR run products, `cumulative.py` owns dynamic AR6 full period cumulative ASR identities and values, and `summary.py` owns ASR summary identities and frequency of no-transgression metric collapse. |
| `pyaesa/shared/acc_asr_common/` | Shared aCC and ASR branch expansion, deterministic downstream loading, shared branch identity guards, and request payload helpers. |
| `pyaesa/external_inputs/lca/` | Canonical external LCA filename parsing, templates, deterministic loading, Monte Carlo loading, validation, and manifest identity. |

The deterministic runtime module `deterministic/runtime/compute.py` is the ASR
owner for numerator denominator matching, selector matching axes, unit factor
resolution, and ASR division. ASR uncertainty reuses those row alignment
contracts and adds only run matrix positioning and batch evaluation.

## Data Flow

Deterministic ASR:

1. Normalize the public ASR request.
2. Resolve or run deterministic aCC prerequisites.
3. Resolve pyaesa owned IO-LCA rows or versioned external LCA rows.
4. Match rows by impact, year, retained selector axes, and external LCA SSP
   route when present.
5. Resolve one numerator to denominator unit factor per matched row.
6. Compute `ASR = LCA / aCC` after applying any supported unit conversion.
7. For dynamic AR6 ASR, compute full period cumulative ASR from matched LCA
   and aCC component sums before public ASR tables are written.
8. Write deterministic result tables, metadata, and optional figures. The
   deterministic manifest records `function`, normalized public `arguments`,
   `execution`, `reuse`, `artifacts`, and `provenance`.

Uncertainty ASR:

1. Normalize the public uncertainty request and optional Sobol request.
2. Resolve upstream `uncertainty_acc(...)` through its public run manifest.
3. Resolve LCA numerator values:
   - IO-LCA route uses `uncertainty_io_lca(...)` when LCIA uncertainty is
     active and deterministic IO-LCA rows otherwise.
   - external LCA route first looks for a versioned Monte Carlo file; when no
     Monte Carlo file is present it uses the matching deterministic external
     LCA file for every package run.
4. Build a compact ASR alignment map with aCC positions, LCA positions, and
   unit factors.
5. Evaluate ASR run batches with NumPy arrays. ASR preserves the upstream aCC
   run layout, so sparse aCC selected rows remain sparse ASR selected rows.
   Yearly run products are owned by `uncertainty/evaluation/runs.py`.
   Dynamic AR6 ASR full period cumulative identities and values are owned by
   `uncertainty/evaluation/cumulative.py`; cumulative ASR remains a period
   metric and is not assigned to one studied year or one aSoCC time route.
   Static ASR writes yearly
   ASR and yearly frequency of no-transgression summaries only.
6. Write public row identity, run values, summaries, source methods,
   README, and `scope_manifest.json`. `uncertainty/io/manifest_payloads.py`
   owns ASR manifest payload assembly and output column metadata. The
   uncertainty manifest uses the same top level `function`, `arguments`,
   `execution`, `reuse`, `artifacts`, and `provenance` sections as the shared
   uncertainty contract.
7. Run optional Sobol analysis through the shared Sobol runtime when at least
   two source dimensions are active.

ASR uncertainty source routing has three independent component lanes. The
public `uncertainty_config` accepts `asocc_uncertainty_sources`,
`ar6_cc_uncertainty_sources`, and `io_lca_uncertainty_sources`. ASR assembles the
aSoCC and AR6 CC lanes into the aCC prerequisite request and passes the LCA
lane only to pyaesa owned IO-LCA. This allows deterministic values in any
component while another component is uncertain.

ASR consumes aSoCC SSP scenario identity through `asocc_ssp_scenario`, aSoCC
time route provenance through `asocc_time_route`, dynamic AR6 carrying capacity
identity through `ar6_cc_ssp_scenario`, and external LCA identity through
`lca_ssp_scenario`. The public selector argument `ssp_scenario` remains a
request input, not a downstream aSoCC row identity column.

ASR scenario scoped uncertainty summaries use the shared scenario target row
ownership contract from `pyaesa/shared/runtime/scenario/scoped_rows.py`.
Scenario specific aSoCC or LCA rows own the matching year and sampled source
identity for their target scenario. Scenario invariant rows fill only sampled
identities that have no target specific row for that year. Cumulative dynamic
ASR applies this ownership per studied year before summing LCA and aCC
components over the full period.

When upstream aCC uses dynamic AR6 CC uncertainty, ASR public row identity
keeps aCC selected trajectory columns such as `cc_category`, `cc_model`,
`cc_scenario`, and `ar6_cc_ssp_scenario`. ASR summaries preserve the aCC
collapsed sampled axis behavior so public summaries report the ASR
distribution, not one summary per candidate AR6 trajectory.

ASR owns the downstream source key namespace in
`pyaesa/asr/uncertainty/sources/source_keys.py`. Upstream aCC sources are exposed as
`acc::<upstream_source>`, pyaesa owned IO-LCA sources as
`io_lca::<upstream_source>`, and external LCA Monte Carlo versions as
`external_lca::<version_name>`. The same keys must appear in
`scope_manifest.json`, `source_methods.csv`, and Sobol dimension names.

## External LCA Contract

External LCA inputs are owned by `pyaesa/external_inputs/lca/`.

| Input type | Location | Filename contract | Row contract |
| --- | --- | --- | --- |
| deterministic | `A_lca/external_lca/deterministic/` | `<version_name>__<lcia_method>` or `<version_name>__<lcia_method>__<ssp_token>` | Wide year columns, `impact`, `impact_unit`, and the ASR selector columns. |
| Monte Carlo | `A_lca/external_lca/monte_carlo/` | `<version_name>__<lcia_method>` | Long rows with `run_index`, `year`, `lca_ssp_scenario`, `impact`, `impact_unit`, `value`, and the ASR selector columns. |

The LCIA method is filename owned. External LCA row tables must not provide a
`lcia_method` column.

## Path Ownership

| Scope | Owner |
| --- | --- |
| Deterministic ASR roots | `pyaesa/asr/shared/runtime/paths.py` |
| ASR Monte Carlo roots | `pyaesa/asr/uncertainty/io/paths.py` |
| External LCA storage roots | `pyaesa/external_inputs/lca/paths.py` |
| Shared uncertainty run folders and manifests | `pyaesa/shared/uncertainty_assessment/` |

ASR outputs are scoped by source branch and LCA route under `C_asr/`.
Single branch uncertainty outputs are further scoped by carrying capacity
branch token:

```text
C_asr/<source_token>/<lca_route>/monte_carlo/<branch_token>/<run_id>/
    results/public_row_identity.<ext>
    results/asr_runs.<ext>
    results/summary_stats_runs.<ext>
    results/cumulative_row_identity.<ext>
    results/cumulative_asr_runs.<ext>
    results/cumulative_summary_stats_runs.<ext>
    results/sobol/
    logs/scope_manifest.json
```

The branch token is `static__<lcia_method>` or
`dynamic_ar6__<lcia_method>`. A mixed public request that includes several
static or dynamic carrying capacity branches writes a branch set manifest at
`C_asr/<source_token>/<lca_route>/monte_carlo/<run_id>/logs/scope_manifest.json`.
The branch set manifest records the branch scope manifests and branch run
roots; each branch still stores its complete ASR artifacts under
`monte_carlo/<branch_token>/<run_id>/`.

## Runtime Contracts

- Run scale and Sobol scale ASR computations use vectorized NumPy arrays.
- Pandas is limited to deterministic loading, validation, table assembly, and
  public table IO.
- Deterministic IO-LCA inputs used by ASR must carry the canonical
  `lcia_method` column written by `deterministic_io_lca(...)`.
- pyaesa owned IO-LCA is bounded by processed historical MRIO coverage.
  Future ASR years require external LCA unless a pyaesa owned IO-LCA
  projection contract is introduced.
- ASR convergence owns the downstream stopping rule. pyaesa owned upstream
  aCC and IO-LCA component inventories use the ASR run id and append missing
  run indices in their own run folders as checkpoints increase. Component
  figures are disabled during checkpoints and rendered once after ASR reaches
  its final completed draw count.
- ASR pyaesa owned component input execution is owned by
  `uncertainty/runtime/component_inputs.py`, convergence checkpoint execution
  by `uncertainty/runtime/checkpoints.py`, and request scope normalization by
  `uncertainty/runtime/scope.py`.
- ASR reused figure rendering is owned by
  `uncertainty/figures/reuse.py` and records updated figure paths in the run
  manifest for the reused Monte Carlo folder.
- Compact ASR run tables contain every requested public row id and are used
  by direct public id indexing. Sparse selected row tables contain only selected
  rows and may omit identities outside the selected sparse scope.
- `refresh=True` for `uncertainty_asr(...)` clears the resolved ASR Monte
  Carlo branch and refreshes pyaesa owned upstream branches called by the
  request, including upstream aCC and pyaesa owned IO-LCA branches when
  selected. Staged external LCA input files are loaded and are not deleted.
- Deterministic ASR figure state is owned by `pyaesa/asr/deterministic/state/`;
  shared aCC ASR code must not carry ASR figure state because aCC has no
  deterministic figure surface.
- Dynamic deterministic ASR aCC versus LCA panels consume prerequisite aCC
  output files and the LCA rows selected for the ASR branch. These panels are
  diagnostic figure products and do not define ASR metric outputs.
- Dynamic ASR Global AR6 CC figure rows consume existing deterministic or
  uncertainty AR6 CC prerequisite artifacts through ASR metadata. These rows are
  diagnostic figure products and do not add AR6 CC data to ASR output tables.
  Dynamic ASR figures write paired `__incl_post` and `__excl_post` products so
  users can inspect either the post study AR6 CC extension or only the requested
  study window.
- ASR figure scale is resolved once per deterministic output scope or Monte
  Carlo run folder within each LCIA method figure family. Negative ASR values
  and ASR values below `1e-5` select normal scale; values above `10` select log
  scale only when no negative or small value is visible. Static ASR figures and
  dynamic ASR aCC versus LCA plus ASR rows consume this resolved mode. fNT rows
  and Global AR6 CC rows use their own axis contracts.
- Figure visible text is formatted by the shared figure scientific text
  contract for labels, titles, legends, axes, and panel text. Output table
  values and filenames keep their stored plain text values.
- ASR Sobol consumes
  `pyaesa/acc/uncertainty/evaluation/source_unit_evaluator.py` as the
  aCC owned source unit evaluation contract. It must not import aCC runner
  internals.
- Public functions must not retain module level dataframes, arrays, or hidden
  caches after return.

## Runtime Reporting Contract

| Runtime surface | ASR behavior |
| --- | --- |
| Upstream auto runs | Report aCC first, then IO-LCA for the pyaesa owned IO-LCA route. |
| Progress cleanup | Clear upstream progress before ASR work and before failures return. |
| Subfigures | Forward `subfigures=True` to upstream aCC and LCA prerequisites. aCC then renders deterministic figures for fixed aSoCC or AR6 CC lanes and uncertainty figures for active stochastic lanes. |
| Figure summaries | Public summaries use `Figures available` and `Figures folder`. |
| Refresh scope | Refresh affects the resolved ASR Monte Carlo branch or branch set and deterministic IO-LCA prerequisites only. |

## Testing And Quality Gates

Package tests for ASR live under `tests/package/asr/`.

Required gates for touched ASR code:

1. `python -m ruff check <touched package paths> <touched tests>`
2. `python -m ruff format --check <touched package paths> <touched tests>`
3. `python -m pyright <touched package paths>`
4. `python -m pytest <targeted tests> --cov=<touched owner> --cov-branch --cov-report=term-missing`

Touched ASR owners must keep 100 percent line and branch coverage. Tests must
cover public reachable behavior and real file flows, not impossible private
states.
