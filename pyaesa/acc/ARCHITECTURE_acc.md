# Architecture: Allocated Carrying Capacity (`acc/`)

## Purpose

`acc/` owns allocated carrying capacity computation. The scientific formula is
`aCC = aSoCC * CC`. Deterministic aCC owns public row
semantics, path layout, prerequisite orchestration, and deterministic result
writing. Uncertainty aCC owns Monte Carlo and Sobol evaluation of the same
formula without reading deterministic aCC result tables as runtime inputs.

## Public Surface

| Public function | Owner | Responsibility |
| --- | --- | --- |
| `deterministic_acc(...)` | `deterministic_acc.py` | Write deterministic aCC tables. |
| `uncertainty_acc(...)` | `uncertainty_acc.py` | Write aCC Monte Carlo and Sobol outputs. |

Both public functions are exported at package level through `pyaesa.__init__`.

## Responsibility Boundary

`acc/` owns:

- static and dynamic aCC branch expansion from public aCC arguments
- aCC result path resolution under `B2_acc/`
- deterministic aCC row identity and value writing
- aCC Monte Carlo row identity, source dependent run layout, exact summaries,
  source methods, manifest, and Sobol outputs
- vectorized aCC value evaluation from compact aSoCC and CC component matrices

`acc/` does not own:

- aSoCC deterministic or uncertainty methodology
- AR6 CC deterministic or uncertainty methodology
- shared uncertainty run ids, manifests, run table IO, summaries,
  convergence mechanics, Sobol design, or Sobol estimators
- ASR numerator or ratio logic

## Canonical Data Flow

Deterministic aCC:

1. Resolve branch requests from `base_cc_args`.
2. Provision deterministic aSoCC and, for dynamic branches, deterministic
   AR6 CC prerequisites without refreshing those upstream scopes unless the
   user called the upstream public function directly.
3. aCC prerequisite orchestration is owned by
   `pyaesa/acc/deterministic/runtime/prerequisites.py`.
4. Load each downstream aSoCC share table once through
   `pyaesa/shared/acc_asr_common/deterministic/downstream/`.
   Materialized aSoCC SSP scenario identity is named `asocc_ssp_scenario`.
   Dynamic AR6 carrying capacity SSP identity is named `ar6_cc_ssp_scenario`
   after AR6 CC rows enter aCC.
5. Build static or dynamic aCC rows with vectorized matrix operations.
   Dynamic AR6 carrying capacity values are converted at the aCC boundary to
   the `impact_unit` declared by the bundled static carrying capacity CSV for
   the matched LCIA method and impact.
6. Write one branch local deterministic manifest for the aCC write scope. The
   manifest records `function`, normalized public `arguments`, `execution`,
   `reuse`, `artifacts`, and `provenance`.

Uncertainty aCC:

1. Resolve aSoCC as one typed component input. Active aSoCC uncertainty
   sources or external aSoCC Monte Carlo files run or reuse
   `uncertainty_asocc(...)`; otherwise deterministic aSoCC rows, including
   deterministic external method rows, are loaded as fixed values.
2. For dynamic branches, resolve AR6 CC as one typed component input.
   `dynamic_ar6_cc_uncertainty` runs or reuses `uncertainty_ar6_cc(...)`;
   otherwise deterministic AR6 CC rows are loaded as fixed carrying capacity
   rows.
   `refresh=True` for `uncertainty_acc(...)` clears the resolved aCC Monte
   Carlo branch and refreshes pyaesa owned upstream aSoCC and dynamic AR6 CC
   branches called by the request.
3. Build compact aCC branch maps from component public row identities.
   aCC consumes upstream aSoCC scenario identity through
   `asocc_ssp_scenario` and upstream aSoCC route provenance through
   `asocc_time_route`; raw AR6 CC `ssp_scenario` is renamed to
   `ar6_cc_ssp_scenario` at the aCC boundary.
   For one dynamic AR6 SSP target, scenario specific aSoCC rows own the
   matching year and sampled source identity. Scenario invariant aSoCC rows
   fill only sampled identities that have no target specific aSoCC row for
   that year.
   Dynamic AR6 CC identities and values are converted to the bundled static
   carrying capacity unit before aCC public row identities, run values, and
   summaries are written.
4. Evaluate aCC run batches with NumPy arrays. Fixed deterministic aSoCC
   values are broadcast across requested run indices by the aCC input owner.
   Compact upstream aSoCC runs stay compact when carrying capacity values are
   deterministic or compact.
   Sparse upstream aSoCC selected rows and sparse dynamic AR6 CC selected rows
   stay sparse in aCC and are expanded through sorted row maps without
   constructing a dense aCC matrix.
   pyaesa owned component input execution is owned by
   `uncertainty/runtime/component_inputs.py`, convergence checkpoint append
   orchestration by `uncertainty/runtime/checkpoints.py`, and request scope
   normalization by `uncertainty/runtime/scope.py`.
   `uncertainty/io/run_outputs.py` is the aCC owner for shared downstream
   run writer. Family neutral run table writes, exact summary cache writes,
   and Monte Carlo convergence mechanics are owned by
   `pyaesa/shared/uncertainty_assessment/io/downstream_run_outputs.py`.
5. Write aCC uncertainty artifacts under `monte_carlo/<run_id>/`. aCC summary
   statistics collapse sampled upstream aSoCC axes in the same way as aSoCC
   uncertainty summaries. `uncertainty/io/manifest_payloads.py` is the canonical
   owner for aCC manifest compatibility payloads and public output column
   metadata.
6. When requested, run aCC Sobol by reusing shared Sobol design and family
   neutral accumulators while aCC owns only formula evaluation and aCC source
   summary grouping.

## Paths And Artifacts

aCC path and public artifact ownership is split by scope:

| Scope | Canonical owner |
| --- | --- |
| Shared aCC family root and source token resolution | `pyaesa/acc/shared/runtime/paths.py` |
| Deterministic branch roots, result paths, and branch metadata paths | `pyaesa/acc/deterministic/runtime/paths.py` |
| Monte Carlo run roots, result files, Sobol files, and log files | `pyaesa/acc/uncertainty/io/paths.py` |
| aCC uncertainty manifest compatibility payloads and public output metadata | `pyaesa/acc/uncertainty/io/manifest_payloads.py` |
| aCC pyaesa owned component inputs and convergence checkpoints | `pyaesa/acc/uncertainty/runtime/component_inputs.py`, `pyaesa/acc/uncertainty/runtime/checkpoints.py` |
| aCC exact reused figure rendering | `pyaesa/acc/uncertainty/figures/reuse.py` |
| Family neutral completed run reuse and appendable run selection | `pyaesa/shared/uncertainty_assessment/run_state/runs.py` |

The deterministic branch root is keyed only by the source token, carrying
capacity family, and carrying capacity source: `static__<lcia_method>` or
`dynamic_ar6__<lcia_method>`. Row selectors such as `cc_bound`, dynamic AR6
category, dynamic AR6 SSP, model, scenario, and studied year coverage are row
or manifest coverage axes. They are not aCC path helper arguments.

Deterministic branch outputs:

```text
B2_acc/<source_token>/deterministic/
    static__<lcia_method>/
    dynamic_ar6__<lcia_method>/
```

Uncertainty outputs:

```text
B2_acc/<source_token>/monte_carlo/<run_id>/
    results/public_row_identity.<ext>
    results/acc_runs.<ext>
    results/summary_stats_runs.<ext>
    results/README.txt
    results/sobol/sobol_indices.<ext>
    results/sobol/sobol_source_summary.<ext>
    results/sobol/README_sobol.txt
    logs/source_methods.csv
    logs/scope_manifest.json
```

`acc_runs` is compact only when every upstream run input can be represented
as a compact fixed row matrix. Sparse selected aSoCC rows or sparse dynamic
AR6 CC rows produce sparse aCC run rows with `run_index`, `public_row_id`,
and `acc`. Dynamic AR6 CC source state columns such as `cc_category`,
`cc_model`, `cc_scenario`, and `ar6_cc_ssp_scenario` remain in
`public_row_identity` so selected trajectories are auditable. aCC summaries
group sampled AR6 source state out in the same way as AR6 CC summaries.
The upstream aSoCC `asocc_time_route` column remains in aCC public identity so
downstream ASR and figure renderers can derive retrospective to prospective
boundaries from the same route provenance as aSoCC.

Sobol files are written only when Sobol is requested and at least two source
dimensions are active.

## Shared Owners

aCC ASR shared deterministic downstream aSoCC share loading, selector handling,
scenario transition metadata, branch identity guards, and tabular IO live under
`pyaesa/shared/acc_asr_common/`.

Shared uncertainty owners live under `pyaesa/shared/uncertainty_assessment/`.
aCC uncertainty must use those shared owners for request normalization, run
reuse, compact table IO, exact summaries, convergence, Sobol design, Sobol
estimators, Sobol diagnostics, and README or manifest method payloads.

Static carrying capacity CSV loading and bound validation are owned by
`pyaesa/shared/lcia/static_cc.py` because deterministic aCC, uncertainty aCC,
and ASR request validation consume the same LCIA carrying capacity prerequisite
contract.

`pyaesa/acc/uncertainty/request/normalization.py` owns aCC source routing from
the shared aCC request into upstream aSoCC uncertainty and dynamic AR6 CC
uncertainty. aCC uncertainty evaluation is split by responsibility:
`evaluation/planning.py` builds one run plan, `evaluation/branches.py` aligns
aSoCC public rows to static or dynamic carrying capacity rows,
`evaluation/runs.py` evaluates compact and sparse run values, and
`evaluation/summary.py` owns aCC source aware summary grouping. Shared public
row grouping arithmetic lives under
`pyaesa/shared/uncertainty_assessment/evaluation/`.
`pyaesa/acc/uncertainty/evaluation/source_unit_evaluator.py` owns aCC unit
interval source evaluation for aCC Sobol and downstream ASR Sobol. Downstream
consumers may consume that evaluator contract when they need to evaluate the
same aCC source dimensions directly; they must not import aCC runner private
helpers.

aCC source method logs keep upstream source names in the same namespace as
aCC manifests: aSoCC sources are recorded as `asocc::<source_name>` and
dynamic AR6 CC sources as `ar6_cc::<source_name>`.

## Runtime Reporting Contract

| Runtime surface | aCC behavior |
| --- | --- |
| Upstream auto runs | Report aSoCC first, then dynamic AR6 CC for dynamic branches. |
| Progress cleanup | Clear upstream progress before aCC work and before failures return. |
| Subfigures | Render deterministic prerequisite figures for fixed lanes and uncertainty figures for active stochastic lanes. |
| Figure summaries | Public summaries use `Figures available` and `Figures folder`. |
| Refresh scope | Refresh affects only the aCC Monte Carlo scope. |

## Testing And Quality Gates

Package tests for aCC live under `tests/package/acc/`. Downstream ASR tests
under `tests/package/asr/` also exercise aCC deterministic contracts.

Required validation for aCC changes:

- `python -m ruff check <touched aCC and shared paths> <touched tests>`
- `python -m ruff format --check <touched aCC and shared paths> <touched tests>`
- `python -m pyright <touched aCC and shared paths>`
- targeted package tests with line and branch coverage for touched owners
- public smoke evidence through `deterministic_acc(...)` or
  `uncertainty_acc(...)` for the changed runtime path

Touched aCC owners must keep 100 percent line and branch coverage.
