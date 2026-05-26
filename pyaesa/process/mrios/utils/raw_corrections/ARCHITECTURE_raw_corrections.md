# Raw Corrections Architecture

## Purpose

`raw_corrections/` owns the EXIOBASE 3.10.2 raw corrected values workflow used
by `process_mrio(...)`.

It covers:

- maintainer side generation of precomputed raw corrected values tables
- runtime loading of those precomputed values
- runtime application to parsed raw environmental `F` tables before aggregation
  and characterization

## Public Surface

This package has no package level public user function.

Maintainer owned module:

- `build_corrected_values.py`: maintainer entrypoint for generating the raw
  corrected values tables in advance
- `maintainer_cli.py`: argument parsing and execution logic used by the
  maintainer entrypoint

Runtime owned modules:

- `runtime.py`: loads and applies precomputed raw corrected values during
  `process_mrio(...)`
- `basis.py`: computes the positive factor input predictor basis used during
  maintainer generation
- `exio_3102_corrected_values.py`: source specific EXIOBASE 3.10.2 generation logic

## Responsibility Boundary

`raw_corrections/`:

- generates repository owned raw corrected values tables for EXIOBASE 3.10.2
- applies those values to parsed raw EXIO tables in memory
- keeps runtime independent from the user local availability of full historical
  EXIOBASE year ranges

It does not:

- expose a public package API
- rewrite EXIO archives on disk
- recompute donor ratios or regressions during user runtime
- modify `F_Y` for the correction scope

The values are generated in advance by maintainers so runtime correction still
works when users process only a subset of years.

## Internal Organization And Runtime Contract

### Maintainer Generation

`build_corrected_values.py` is repository owned maintainer code. It is not part
of normal user runtime. It owns the executable module path; `maintainer_cli.py`
owns the reusable argument parsing and generation call.

It runs:

```bash
python -m pyaesa.process.mrios.utils.raw_corrections.build_corrected_values `
  --workspace-root <workspace root>
```

and writes:

- `corrected_values/exiobase_3102_ixi_raw_corrected_values.csv`
- `corrected_values/exiobase_3102_pxp_raw_corrected_values.csv`

The generation logic uses:

- raw environmental extension `F`
- positive categories of raw `factor_inputs/F`, summed by `(region, sector)`
- source specific donor or OLS level rules implemented in
  `exio_3102_corrected_values.py`

### Runtime Application

During `process_mrio(...)`:

1. raw EXIO year is parsed
2. precomputed raw corrected values for that source and year are loaded
3. corrected values are applied to parsed raw environmental `F`
4. aggregation and characterization continue normally

This keeps downstream `PBA`, `CBA`, and processed UNCASExt metrics aligned to
the same corrected source values.

### User Visible Reporting

Runtime does not print per year progress messages for corrections.

The final `ProcessReportMRIO` summary adds one aggregated block when corrections
were applied, listing:

- affected year or year range
- region
- extension
- stressor family
- method used
- correction reason

Per processed year log:

- `<saved_dir>/logs/raw_corrected_values.csv`

Metadata year entry:

- `raw_corrected_values.row_count`
- `raw_corrected_values.log_path`
- `raw_corrected_values.summary_lines`

## Testing And Quality Gates

Package tests for this scope live under:

- `tests/package/process/mrio/test_raw_corrections.py`
- `tests/package/process/mrio/test_process_mrio.py`
- `tests/package/process/mrio/test_year_entry_runtime_env_metadata.py`

These tests cover:

- maintainer generation entrypoint behavior
- runtime loading and application
- user visible report wording
- metadata and log payload structure

Required validation for touched raw correction owners:

- `python -m ruff check <touched raw correction paths> <touched tests>`
- `python -m ruff format --check <touched raw correction paths> <touched tests>`
- `python -m pyright <touched raw correction paths>`
- targeted package tests with line and branch coverage for the touched owner

Maintainer generation changes must also regenerate the corrected value CSVs in
a workspace and record the command and resulting file identity in
the handoff notes. Runtime application changes must be validated through
`process_mrio(...)` package tests without MRIO refresh.
