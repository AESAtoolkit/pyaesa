"""Reference year uncertainty for final aSoCC rows."""

import numpy as np
import pandas as pd

from pyaesa.asocc.uncertainty.inputs.deterministic_rows import ASOCC_VALUE_COLUMN
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import LoadedAsoccFinalRows
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.uncertainty_assessment.monte_carlo.random_streams import uniform_by_run_index
from pyaesa.asocc.uncertainty.io.source_methods import SourceMethodRow
from pyaesa.asocc.uncertainty.sources.names import REFERENCE_YEAR_SOURCE

REFERENCE_YEAR_RANDOM_STREAM = "asocc.reference_year.reference_year"


def admissible_reference_year_rows(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose reference year can represent the studied year."""
    if "reference_year" not in frame.columns:
        return frame
    rows = frame.copy()
    reference_year = pd.Series(
        pd.to_numeric(pd.Series(rows.loc[:, "reference_year"], copy=False)),
        index=rows.index,
    )
    year = pd.Series(
        pd.to_numeric(pd.Series(rows.loc[:, "year"], copy=False), errors="raise"),
        index=rows.index,
    )
    return rows.loc[reference_year.isna() | reference_year.le(year)].reset_index(drop=True)


def reference_year_uncertainty_has_targets(*, rows: pd.DataFrame) -> bool:
    """Return whether selected rows expose at least two admissible reference year candidates."""
    if "reference_year" not in rows.columns:
        return False
    candidates = admissible_reference_year_rows(frame=rows)
    reference_year: pd.Series = pd.Series(
        pd.to_numeric(
            pd.Series(candidates.loc[:, "reference_year"], copy=False),
            errors="raise",
        ),
        copy=False,
    )
    return bool(reference_year.dropna().nunique() >= 2)


def collapse_reference_year_public_template(*, template: pd.DataFrame) -> pd.DataFrame:
    """Return the public template after collapsing sampled reference year candidates."""
    rows = admissible_reference_year_rows(frame=template).reset_index(drop=True)
    identity_columns = [
        column for column in rows.columns if column not in {ASOCC_VALUE_COLUMN, "reference_year"}
    ]
    return rows.drop(columns=["reference_year"]).drop_duplicates(
        identity_columns,
        ignore_index=True,
    )


def apply_reference_year_uncertainty_to_matrix(
    *,
    template: pd.DataFrame,
    values: np.ndarray,
    batch: RunBatch,
    unit_values: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sample reference year candidates directly into the compact value matrix."""
    if "reference_year" not in template.columns:
        return template, values
    source_rows = template.copy()
    source_rows["_input_position"] = np.arange(len(source_rows), dtype=np.int64)
    rows = admissible_reference_year_rows(frame=source_rows).reset_index(drop=True)
    rows["reference_year"] = pd.to_numeric(
        pd.Series(rows.loc[:, "reference_year"], copy=False),
        errors="raise",
    )
    rows["year"] = pd.Series(
        pd.to_numeric(pd.Series(rows.loc[:, "year"], copy=False), errors="raise"),
        index=rows.index,
    ).astype("int64")
    identity_columns = [
        column
        for column in rows.columns
        if column not in {ASOCC_VALUE_COLUMN, "reference_year", "_input_position"}
    ]
    output_template = collapse_reference_year_public_template(
        template=rows.drop(columns=["_input_position"])
    )
    output_positions = output_template.loc[:, identity_columns].assign(
        _output_position=np.arange(len(output_template), dtype=np.int64)
    )
    indexed = rows.merge(output_positions, on=identity_columns, how="left", sort=False)
    output = np.empty((batch.n_runs, len(output_template)), dtype=np.float64)
    invariant = indexed.loc[indexed["reference_year"].isna()]
    if not invariant.empty:
        output[:, invariant["_output_position"].to_numpy(dtype=np.int64)] = values[
            :,
            invariant["_input_position"].to_numpy(dtype=np.int64),
        ]
    candidates = indexed.loc[indexed["reference_year"].notna()].copy()
    if not candidates.empty:
        _sample_candidate_values(
            output=output,
            values=values,
            candidates=candidates,
            batch=batch,
            unit_values=unit_values,
        )
    return output_template, output


def _sample_candidate_values(
    *,
    output: np.ndarray,
    values: np.ndarray,
    candidates: pd.DataFrame,
    batch: RunBatch,
    unit_values: np.ndarray | None,
) -> None:
    reference_years = np.array(
        sorted({int(reference_year) for reference_year in candidates["reference_year"].tolist()}),
        dtype=np.int64,
    )
    run_indices = batch.run_indices()
    raw = _sample_reference_years(
        options=reference_years,
        run_indices=run_indices,
        stream_name=REFERENCE_YEAR_RANDOM_STREAM,
        unit_values=unit_values,
    )
    for year, year_rows in candidates.groupby("year", dropna=False, sort=False):
        year_int = int(float(str(year)))
        compatible = reference_years[reference_years <= year_int]
        selected = raw.copy()
        replace_mask = selected > year_int
        if bool(replace_mask.any()):
            selected[replace_mask] = _sample_reference_years(
                options=compatible,
                run_indices=run_indices[replace_mask],
                stream_name=f"{REFERENCE_YEAR_RANDOM_STREAM}.{year_int}",
                unit_values=None if unit_values is None else unit_values[replace_mask],
            )
        for reference_year, reference_rows in year_rows.groupby(
            "reference_year",
            dropna=False,
            sort=False,
        ):
            run_positions = np.flatnonzero(selected == int(float(str(reference_year))))
            if run_positions.size:
                output_positions = reference_rows["_output_position"].to_numpy(dtype=np.int64)
                input_positions = reference_rows["_input_position"].to_numpy(dtype=np.int64)
                output[np.ix_(run_positions, output_positions)] = values[
                    np.ix_(run_positions, input_positions)
                ]


def _sample_reference_years(
    *,
    options: np.ndarray,
    run_indices: np.ndarray,
    stream_name: str,
    unit_values: np.ndarray | None = None,
) -> np.ndarray:
    uniform = (
        np.asarray(unit_values, dtype=np.float64)
        if unit_values is not None
        else uniform_by_run_index(stream_name=stream_name, run_indices=run_indices)
    )
    index = np.floor(uniform * len(options)).astype(np.int64)
    return options[index]


def reference_year_source_method_row(*, loaded: LoadedAsoccFinalRows) -> SourceMethodRow:
    """Return the compact scientific log row for reference year uncertainty."""
    return SourceMethodRow(
        source_component="asocc",
        source_name=REFERENCE_YEAR_SOURCE,
        scope=str(loaded.base_asocc_args["fu_code"]),
        applied_bucket=loaded.final_bucket,
        year_min=min(loaded.requested_years),
        year_max=max(loaded.requested_years),
        distribution="discrete uniform over admissible deterministic reference years",
        shared_random_variable="run_index",
        formula=(
            "sampled row = deterministic final aSoCC candidate selected from reference_year <= year"
        ),
        notes=(
            "The compact public identity groups across sampled reference years; run matrices "
            "store the resulting aSoCC values."
        ),
    )
