# Architecture: Process (`pyaesa/process/`)

## Purpose

The `pyaesa.process` package converts raw pyaesa owned inputs into processed
runtime assets used by deterministic and uncertainty workflows. It consumes
`data_raw/` files created or prepared by `set_workspace(...)` and
`pyaesa.download`, then writes versioned processed files under `data_processed/`.

This document is for external Python contributors. It describes the public
processing surface, canonical processing owners, metadata boundaries, and the
rules that keep processed assets reproducible for scientific workflows.

## Public Surface

The package level public API is exported through `pyaesa.__init__`.

| Public function | Owner module | Responsibility |
| --- | --- | --- |
| `process_mrio(...)` | `pyaesa/process/mrios/process_mrio.py` | Public MRIO processing entry point. Runtime orchestration is owned by `mrios/utils/pipeline/runner.py`. |
| `process_pop_gdp(...)` | `pyaesa/process/pop_gdp/process_pop_gdp.py` | Harmonize raw population and GDP inputs into processed historical and SSP tables. |
| `process_ar6(...)` | `pyaesa/process/ar6/process_ar6.py` | Build processed AR6 climate pathway workbooks, logs, and optional figures. |

Processing functions are user facing and must have complete Google style
docstrings when touched.

## Responsibility Boundary

`pyaesa.process` owns:

| Area | Canonical owner |
| --- | --- |
| Processed MRIO public entry point | `mrios/process_mrio.py` |
| Processed MRIO runtime orchestration | `mrios/utils/pipeline/runner.py` |
| Processed MRIO paths and metadata | `mrios/utils/io/paths.py`, `mrios/utils/io/metadata.py` |
| EXIOBASE and OECD parsing | `mrios/utils/parsers/` |
| MRIO aggregation validation | `mrios/utils/aggregation/`, `mrios/utils/pipeline/aggregation_validation.py` |
| MRIO year processing contracts | `mrios/utils/pipeline/` |
| LCIA characterization prerequisite paths | `pyaesa/shared/lcia/paths.py` |
| Raw correction application | `mrios/utils/raw_corrections/` |
| UNCASExt MRIO metric generation | `mrios/utils/uncasext_metrics/` |
| Processed population GDP paths and metadata | `pop_gdp/io/paths.py`, `pop_gdp/io/metadata.py` |
| Population GDP harmonization | `pop_gdp/sources/wb.py`, `pop_gdp/sources/ssp.py`, `pop_gdp/pipeline/finalize.py`, `pop_gdp/pipeline/parent_aggregation.py`, `pop_gdp/pipeline/fill_log.py` |
| AR6 processing orchestration | `ar6/process_ar6.py`, `ar6/utils/pipeline/process_runner.py` |
| AR6 processed paths and metadata | `ar6/utils/io/paths.py`, `ar6/utils/io/metadata.py` |
| AR6 workbooks, template files, reports, and text outputs | `ar6/utils/io/writers.py`, `ar6/utils/io/reports.py`, `ar6/utils/io/report_summaries.py`, `ar6/utils/io/text_outputs.py` |
| AR6 preprocessing, derived variables, and harmonization | `ar6/utils/pipeline/preprocessing.py`, `ar6/utils/pipeline/derived_variables.py`, `ar6/utils/pipeline/harmonization.py` |
| AR6 CO2 reconstruction and row issue logging | `ar6/utils/pipeline/co2_reconstruction.py`, `ar6/utils/pipeline/processing_modes.py` |
| AR6 process scoped figures | `ar6/utils/figures/` |

`pyaesa.process` does not own:

| Area | Owner |
| --- | --- |
| Raw source acquisition | `pyaesa/download/` |
| Workspace creation and packaged prerequisite import | `pyaesa/workspace_initialisation/` |
| Deterministic scientific formulas | Respective scientific packages |
| Monte Carlo or Sobol execution | Family uncertainty packages and `pyaesa/shared/uncertainty_assessment/` |
| User edited external input schemas | `pyaesa/external_inputs/` |

Do not add deterministic scientific outputs to processing modules. Processing
creates reusable input assets; scientific result publication belongs to the
family runtime packages.

## Package Layout

| Path | Role |
| --- | --- |
| `mrios/` | Public MRIO processing entry point. |
| `mrios/utils/io/` | Processed MRIO paths, saved directory naming, metadata, and clipping log paths. |
| `mrios/utils/parsers/` | Source parsers and EXIOBASE characterization. |
| `mrios/utils/aggregation/` | Aggregation and disaggregation file loading and validation helpers. |
| `mrios/utils/pipeline/` | MRIO processing runtime orchestration, contracts, setup, persistence, matrix operations, and LCIA tracking. |
| `mrios/utils/raw_corrections/` | Maintainer owned raw correction generation and runtime application. |
| `mrios/utils/uncasext_metrics/` | Processed enacting metrics and utility propagation metrics used by aSoCC. |
| `pop_gdp/` | Population GDP processing, matching, parent aggregation, fill logs, and metadata. |
| `ar6/` | AR6 processing entry point, pipeline, processed output writers, reports, and process figures. |

Path construction must stay in path owner modules. Do not build processed paths
inside equation owners, downstream deterministic functions, or tests when a path
helper exists.

## Processed Data Flow

Processing functions follow one common pattern:

1. Public function validates public arguments.
2. Processing owner resolves raw inputs and workspace paths.
3. Existing processed metadata and files are checked before work starts.
4. Refresh clears only the selected processed scope owned by the public
   function before rebuilding.
5. Processed outputs and metadata are written atomically at the family scope.
6. Public function returns its documented public result for both fresh work and
   compatible reuse. `process_ar6(...)` returns `ProcessReportAR6` and writes
   `summary.log` for both cases. MRIO and population GDP process functions keep
   their documented result contracts.

Processing functions must not keep hidden module level runtime state across
calls. Persistent state belongs in documented processed files and metadata.

## MRIO Processing Contract

`process_mrio(...)` consumes raw MRIO assets and writes processed year scoped
directories.

Supported public source keys are shared with `download_mrio(...)` through
`pyaesa/download/mrios/utils/source_registry.py`.

Important public options:

| Argument | Processing contract |
| --- | --- |
| `source` | Supported MRIO source key. |
| `years` | Year selector normalized by the shared MRIO year selector. |
| `refresh` | Recompute selected years in the processed MRIO scope. |
| `lcia_method` | EXIOBASE LCIA characterization methods; unsupported for OECD ICIO. |
| `agg_reg`, `agg_sec`, `agg_version` | Region and sector MRIO aggregation and disaggregation controls. |
| `keep_intermediate_uncasext` | Keep post clip core and extension payloads. |
| `pymrio_calc_all` | Keep full PyMRIO `calc_all` preclip core and extension payloads. |

MRIO processing writes:

| Output family | Owner |
| --- | --- |
| Minimal core pickles needed by package metrics | `pyaesa/process/mrios/utils/pipeline/persistence.py` |
| Utility propagation metrics | `mrios/utils/uncasext_metrics/utility_propagation_metrics.py` |
| Enacting metrics | `mrios/utils/uncasext_metrics/enacting_metric.py` |
| LCIA characterized payloads for EXIOBASE | `pyaesa/process/mrios/utils/parsers/exio_characterization.py`, `pyaesa/process/mrios/utils/pipeline/persistence.py` |
| Processed metadata | `mrios/utils/io/metadata.py` |
| Clipping and raw correction logs | `pyaesa/process/mrios/utils/uncasext_metrics/enacting_metric_clip_log.py`, `pyaesa/process/mrios/utils/raw_corrections/runtime.py` |

Default MRIO processing keeps the minimal package runtime assets. Optional
intermediate modes may keep additional matrices for inspection, but downstream
deterministic correctness must not depend on optional intermediate payloads.

### MRIO Aggregation

Aggregation uses packaged prerequisite CSVs under `data_raw/`. `agg_version`
selects the aggregation folder. Aggregation validation must happen before year
processing begins.

Contributor rules:

1. Aggregation path ownership stays in process and project prerequisite path
   helpers.
2. Aggregated flows, aggregated `G`, enacting metrics, and LCIA payloads must use the
   same aggregated product basis.
3. Aggregated full PyMRIO output is allowed only through `pymrio_calc_all=True`.
4. Aggregation metadata must describe region and sector labels used by downstream
   deterministic functions.

### MRIO Raw Corrections

Raw corrections are maintainer owned and documented separately in:

`pyaesa/process/mrios/utils/raw_corrections/ARCHITECTURE_raw_corrections.md`

Runtime processing may apply precomputed correction rows before aggregation and
characterization. Do not add ad hoc correction logic inside parsers or
downstream scientific packages.

### MRIO LCIA Characterization

EXIOBASE LCIA characterization uses prerequisite characterization factors and
responsibility period files under `data_raw/`. LCIA processing must record:

LCIA characterization matrix paths are owned by `pyaesa/shared/lcia/paths.py`.
MRIO processing consumes that path owner through
`mrios/utils/io/paths.py::_get_characterization_matrix_path(...)`; it does not
own a separate characterization directory contract.

| Metadata | Purpose |
| --- | --- |
| Applied LCIA methods | Downstream aSoCC LCIA method availability. |
| Missing LCIA methods or impacts | Clear reason for skipped LCIA payloads. |
| LCIA units | Public deterministic output unit metadata. |
| Characterization status | Downstream setup validation. |

OECD ICIO does not support `lcia_method`; public processing must fail early when
LCIA is requested for OECD.

## Population And GDP Processing Contract

`process_pop_gdp(...)` consumes raw World Bank, IMF Taiwan, and SSP files and
writes harmonized processed tables.

| Dataset | Owner | Purpose |
| --- | --- | --- |
| Historical WB plus IMF Taiwan | `pop_gdp/sources/wb.py` | Historical population and GDP table. |
| SSP | `pop_gdp/sources/ssp.py` | Future population and GDP table. |
| MRIO matching | `pop_gdp/io/paths.py`, `pop_gdp/process_pop_gdp.py` | Map population GDP regions to EXIOBASE and OECD region labels. |
| Parent aggregation and filling | `pop_gdp/pipeline/parent_aggregation.py`, `pop_gdp/pipeline/fill_log.py`, `pop_gdp/pipeline/finalize.py` | Complete harmonized rows and record fill decisions. |

Public switches `past_years` and `future_years` select which processed tables
are built. Existing processed files are reused when metadata covers the required
year range and `refresh=False`.

`refresh=True` clears only the selected processed dataset scope before
rebuilding: the processed CSV, its metadata JSON, and the WB fill log when the
historical WB dataset is selected.

Population GDP processing does not choose aSoCC SSP scenarios. It prepares the
processed tables later selected by deterministic aSoCC and AR6 workflows.

## AR6 Processing Contract

`process_ar6(...)` consumes raw AR6 climate input files and writes a processed
scope for a selected study period and harmonization mode.

Important public options:

| Argument | Processing contract |
| --- | --- |
| `years` | Consecutive study period expressed as a year list or `range(start, stop)`. |
| `figures` | Generate process scoped diagnostic figures and figure guides. |
| `harmonization` | Choose harmonized or retained original pathway output. |
| `harmonization_method` | Harmonization method. Supported value is `offset`. |
| `refresh` | Rebuild the selected processed AR6 scope. |
| Figure options | Apply only when `figures=True`. |

AR6 processing owners:

| Area | Owner |
| --- | --- |
| Study period validation | `ar6/utils/pipeline/study_period.py` |
| Raw input requirements | `ar6/utils/pipeline/raw_inputs.py` |
| Preprocessing identity and metadata validation | `ar6/utils/pipeline/preprocessing.py` |
| CO2 reconstruction filter | `ar6/utils/pipeline/co2_reconstruction.py` |
| Derived variable construction and row issue reasons | `ar6/utils/pipeline/derived_variables.py` |
| Harmonization | `ar6/utils/pipeline/harmonization.py` |
| Processing modes and output building | `ar6/utils/pipeline/processing_modes.py` |
| Process runner | `ar6/utils/pipeline/process_runner.py` |
| Process paths and metadata | `ar6/utils/io/paths.py`, `ar6/utils/io/metadata.py` |
| Workbooks and model-scenario template outputs | `ar6/utils/io/writers.py` |
| User reports and variable coverage summaries | `ar6/utils/io/reports.py`, `ar6/utils/io/report_summaries.py` |
| Log text outputs and figure sampling guides | `ar6/utils/io/text_outputs.py` |
| Process scoped figures | `ar6/utils/figures/` |

The retained AR6 domain is the requested category selector within C1 through C8,
SSP family 1 to 5, and pathways whose historical and future vetting fields both
equal `Pass`. The default category selector is C1 to C4. The AR6 variable
contract is owned by `pyaesa/download/ar6/utils/config.py`. Processing first
keeps model and scenario pairs whose raw `Emissions|CO2` row has the requested
study start year and year 2100, then applies the CO2 decomposition
reconstruction check and builds pre harmonization rows from the raw variable
list: package net variables, WO AFOLU net variables, `Carbon Sequestration|Total`, and
`Carbon Sequestration|Subtotal_seq`. The pre harmonization owner removes
model and scenario pairs whose required raw CO2 coverage is incomplete, whose
cumulative CO2 reconstruction error does not pass the reference threshold, or
whose carbon sequestration variables contain negative values, and reports those
exclusions through the AR6 row issue log.

Harmonization is applied only to the four package net variables. Gross and
gross alternative emissions variables are computed after net harmonization by
adding the matching sequestration companion rows. The final processed workbook
stores the twelve positive emissions variables plus
`Carbon Sequestration|Total` and `Carbon Sequestration|Subtotal_seq`. The same
derived variable owner applies the final gross emissions sign check per
model, scenario, and variable row. A negative value in one gross row removes only
that gross output row; it does not remove net rows, other gross variables, or
the corresponding sequestration companion rows for that model and scenario when
they pass their own checks. Those exclusions are reported through the AR6 row
issue log. `ORIGINAL_AR6` remains a traceability sheet with raw and pre
harmonization derived rows.

AR6 process figures are owned by `ar6/utils/figures/`. Figure owners read only
persisted processing outputs and logs, write figure files and companion guides
inside the process scope, and keep cleanup limited to the requested process
figure scope. `figure_guides.py` owns the local figure guide text, figure
renderers own plot content, and the process metadata writer records the
resolved figure request.

`ProcessReportAR6` summarizes variable retention as retained after AR6
processing filters. The process runner returns this report and writes the same
text to `logs/summary.log` for fresh processing, figure generation, and
compatible reuse. Variable summary reason text uses the recorded row issue log
`drop_reason` values when retained model and scenario pairs differ from the
available model and scenario pairs. The persisted CSV remains the detailed
source for `drop_stage`, `drop_reason`, model, scenario, variable, retained
variable, SSP family, and category.

The processed AR6 scope lives under a dedicated category scoped `process_ar6/`
subtree below the processed AR6 root. The root scope name is derived from the
study period and harmonization state, for example
`2019-2060_harmonization_offset` or `2019-2060_no_harmonization`. The selected
category scope is the child folder below `process_ar6/`, for example `C1-C4`,
`C3`, or `C5-C8`.

AR6 path helpers resolve paths only. `ar6/utils/pipeline/process_runner.py`
owns directory materialization and refresh cleanup for the selected processed,
log, and figure scope.

Downstream dynamic carrying capacity functions may reuse this processed scope.
They must not recompute AR6 preprocessing internally when the public processing
owner can provide the required scope, and they consume the same retained
filtering summary facts from the reused `ProcessReportAR6` as from a fresh
processing run.

## Refresh And Reuse

Refresh behavior is family local:

| Family | Refresh rule |
| --- | --- |
| MRIO | Recompute selected processed years and update processed metadata. |
| Population GDP | Rebuild selected processed historical or SSP tables. |
| AR6 | Rebuild the selected study period and harmonization scope, plus figures when `figures=True` (default). |

Reuse decisions must use family metadata and expected processed files. Do not
scan downstream deterministic outputs to infer processing completeness.

## Adding A Processed Asset

For a new processed asset, define:

1. Raw input source and raw path owner.
2. Processed path owner.
3. Metadata payload and coverage fields.
4. Processing owner and numeric method.
5. Unit metadata when downstream outputs expose units.
6. Downstream readers that consume the asset.
7. Tests with local raw fixtures.

For MRIO assets, also update source configs, persistence checks, year metadata,
and aSoCC input loaders when downstream deterministic workflows need the asset.
For AR6 and population GDP assets, update the processing runner and downstream
selectors in the same change set.

## Testing And Quality Gates

Package tests for processing code live under:

| Family | Tests |
| --- | --- |
| MRIO | `tests/package/process/mrio/` |
| Population GDP | `tests/package/process/pop_gdp/` |
| AR6 | `tests/package/process/ar6/` |

Processing tests must be deterministic:

1. No real network calls.
2. Use local raw fixtures that exercise normal processing readers.
3. Cover metadata writes and reuse behavior.
4. Cover refresh only inside the processed family scope.
5. Avoid tests that force private states unreachable through public functions.

For touched process owners, run:

1. `python -m ruff check <touched paths>`.
2. `python -m ruff format --check <touched paths>`.
3. `python -m pyright <touched package paths>`.
4. Targeted `pytest` with `--cov=<touched owner> --cov-branch`.

Keep touched owners at 100 percent line and branch coverage.
