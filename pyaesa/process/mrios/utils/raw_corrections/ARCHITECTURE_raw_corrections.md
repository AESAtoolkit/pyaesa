# Raw Corrections Architecture

## Purpose

`raw_corrections/` owns source-specific EXIOBASE correction precomputation and
runtime application before MRIO aggregation and LCIA characterization. Supported
sources are EXIOBASE 3.10.2 and 3.11.2, each in ixi and pxp form.

## Public Surface

This package has no package-level public function. `process_mrio(...)` consumes its
precomputed tables through the year orchestrator.

| Owner | Responsibility |
| --- | --- |
| `build_corrected_values.py` | Executable maintainer module. |
| `maintainer_cli.py` | Workspace, source and output-directory arguments; generation progress. |
| `workspace_data.py` | Source-version selection, raw archive access, bundled prerequisite access and call-local caches. |
| `exio_corrected_values.py` | Shared donor-intensity and OLS calculations, source-specific correction scopes and CSV generation. |
| `basis.py` | Positive factor-input predictor and package-owned corrected-value directory. |
| `runtime.py` | Supported source keys, year-row loading, raw `F` replacement and correction descriptions. |
| `reporting.py` | Process-report totals, classification-scoped logs and metadata payloads. |

## Responsibility Boundary

Maintainer generation reads original raw environmental `F` tables and positive
categories of `factor_inputs/F`. It writes package-owned corrected values in
`corrected_values/<source>_raw_corrected_values.csv`. Runtime loads only the
selected source and year; it requires neither donor archives nor a complete
historical series in the user's workspace.

Runtime replaces environmental production-account `F` cells in memory. Raw ZIP
archives and direct final-demand `F_Y` accounts remain unchanged. Regression or
donor fitting belongs exclusively to maintainer generation.

## Internal Organization And Runtime Contracts

### Source-Specific Scientific Scopes

| Source version | Country and scope | Calculation and year boundary |
| --- | --- | --- |
| 3.10.2 | Taiwan, `land/Forest` | China sector-wise forest intensity per positive factor-input sum. |
| 3.10.2 | Malta, `land/Forest` | Pooled EU27 intensity, excluding Malta; target sectors require a positive predictor. |
| 3.10.2 | Switzerland, two phosphorus-to-water stressors | Sector-wise OLS levels for 2021-2022, fitted on 1995-2020. |
| 3.10.2 | Luxembourg, PB-LCIA freshwater-consumption stressors | Sector-wise OLS levels for 2020, fitted on 1995-2022 excluding 2020. |
| 3.11.2 | Taiwan and eligible Malta sectors, `Naturally generating forest - Non primary forest` and `Planted forest` | The same country-specific donor-intensity rules, applied separately to each native forest stressor for 1995-2023. |

The bundled 3.10.2 tables determine the exact corrected source/year/sector cells.
The 3.11.2 tables contain 5,012 ixi rows (58 Taiwan and 4,954 Malta) and 58 pxp
rows (Taiwan). Malta pxp forestry sectors lack a positive predictor throughout
the land-data period; their native values are retained. These tables contain no
Swiss phosphorus or Luxembourg water regression rows: the corresponding 3.10.2
anomalies are absent from the 3.11.2 raw accounts.

Both forest stressors are zero for Taiwan in all 1995-2023 ixi and pxp archives;
Malta ixi also has zero forest accounts throughout that period. The native Malta
pxp accounts contain forest values.

Eligibility requires positive donor forest values and positive target value
added. A nonpositive donor denominator in an eligible sector is an error. A
nonpositive target predictor produces a generation diagnostic and no correction
row. The predictor sums positive factor-input categories separately before
summing across categories; it is not the clipped signed total used by every
downstream metric.

EXIOBASE 3.11.2 has no `land` extension for 2024. Correction generation stops at
2023.

### Maintainer Generation And Path Ownership

Run from the package checkout, for example:

```powershell
python -m pyaesa.process.mrios.utils.raw_corrections.build_corrected_values `
  --workspace-root <workspace-root> `
  --source exiobase_3112_ixi --source exiobase_3112_pxp
```

`--source` can be repeated; omission selects all supported correction sources.
`--out-dir` selects a review destination instead of the package resource folder.
The selected source archives must exist under the supplied workspace root.

`workspace_data.py` consumes the download path owner's
`download/mrios/utils/paths.py::_resolve_source_layout(...)` contract. Archives
are read from `<workspace-root>/data_raw/mrio/exiobase_3/exiobase_<version>/full_<system>`;
there is no additional `pyaesa` directory beneath the workspace root. Source
identity and year bounds come from `download/mrios/utils/source_registry.py`.
Bundled EU27 mappings and version-specific PB-LCIA matrices define donor regions
and freshwater stressors. Caches belong to one `_WorkspaceData` instance and are
not retained across generation calls.

### Runtime Data Flow And Reuse

1. The source year is parsed.
2. `runtime.py` loads precomputed rows for that exact source and year.
3. Corrected values replace the designated raw environmental `F` cells.
4. Region/sector aggregation, characterization and matrix calculation consume
   the corrected accounts.
5. `reporting.py` writes correction logs and metadata.

### Logs And Metadata

The process path owner `mrios/utils/io/paths.py` resolves one shared CSV per
source/classification:

`data_processed/mrio/<source>/<classification>/logs/raw_corrected_values_log.csv`.

Rows include their year, source value, corrected value, delta, method and reason.
The log is appended during processing; independent classifications own separate
logs. `ProcessReportMRIO` groups scopes across processed years. Each metadata year
entry records `raw_corrected_values.row_count`, `log_path` and `summary_lines`.
Years with no applied rows have no correction payload.

## Testing And Quality Gates

| Package tests | Contract |
| --- | --- |
| `tests/package/process/mrio/test_raw_corrections.py` | Generation from real small ZIP fixtures, source-specific scopes, donor/OLS behavior, runtime replacement and unchanged `F_Y`. |
| `tests/package/process/mrio/test_exio3112.py` | Both public 3.11.2 systems, corrected LCIA values, correction counts, metadata, missing 2024 land and output reuse. |
| `tests/package/process/mrio/test_process_mrio.py` | Correction reporting and processing orchestration. |
| `tests/package/process/mrio/test_year_entry_runtime_env_metadata.py` | Metadata payload structure. |

Reusable archive scaffolds are in `tests/package/helpers/raw_corrections.py` and
`tests/package/helpers/data_processing_dummy.py`. Tests use local files and small
real PyMRIO datasets without real downloads or internal control-flow patching.
