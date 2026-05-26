"""Write enacting metric outputs."""

from pathlib import Path
from typing import Callable

import pandas as pd

from ....io.metadata import EnactingMetricKey
from ....runtime.output.contracts import IdentifierSchema, OutputArtifact, contract_year_columns
from ....runtime.paths.published import _get_enacting_metric_output_path, _owning_fu_level_for_code
from ...yearly.shared.scenario_routing import (
    is_regression_projection_year,
    regression_projection_subfolder_for_context,
)
from pyaesa.asocc.orchestration.write.writers.enacting_metric_units import (
    resolve_enacting_metric_unit,
)
from pyaesa.asocc.orchestration.write.tables.wide_validation import (
    assert_no_duplicate_columns,
    validate_wide_frame,
)


def _reset_metric_index_strict(df: pd.DataFrame) -> pd.DataFrame:
    """Strict reset for enacting metric assembly only."""
    if isinstance(df.index, pd.MultiIndex):
        raw_names = list(df.index.names)
    else:
        raw_names = [df.index.name]
    if any(name is None for name in raw_names):
        raise ValueError(
            f"Cannot reset enacting metric index with unnamed levels. index_names={raw_names}"
        )
    index_names = [str(name) for name in raw_names]
    collisions = set(index_names) & {str(col) for col in df.columns}
    if collisions:
        raise ValueError(
            "Cannot reset enacting metric index because index names collide with "
            f"existing columns: {sorted(collisions)}"
        )
    return df.reset_index()


def _build_enacting_metric_output(
    *,
    context,
    state,
    key: EnactingMetricKey,
    year_map: dict[int, pd.Series],
    l1_source: str | None,
    projection_subfolder: str | None,
) -> tuple[OutputArtifact, Path]:
    """Build typed enacting metric artifact and destination path."""
    df = _build_enacting_metric_frame(
        context=context,
        key=key,
        year_map=year_map,
        mrio_default_monetary_unit=state.mrio_default_monetary_unit,
        mrio_units=state.mrio_units,
        lcia_units=state.lcia_units,
    )
    level = state.enacting_metric_levels[key]
    out_path = _get_enacting_metric_output_path(
        proj_base=context.proj_base,
        source=str(l1_source if level == "level_1" and l1_source is not None else context.source),
        agg_version=context.agg_version,
        level=level,
        key_metric=key.metric,
        key_method=key.lcia_method,
        key_scenario=key.ssp_scenario,
        output_format=context.output_format,
        lcia_sub=projection_subfolder,
        owning_fu_level=_owning_fu_level_for_code(fu_code=context.fu_code),
    )
    year_cols = contract_year_columns(context)
    id_cols = tuple(str(c) for c in df.columns if str(c) not in year_cols)
    schema = IdentifierSchema(columns=id_cols, year_columns=year_cols)
    artifact = OutputArtifact(
        schema=schema,
        data_wide=validate_wide_frame(df, schema),
    )
    return artifact, out_path


def _build_enacting_metric_frame(
    *,
    context,
    key: EnactingMetricKey,
    year_map: dict[int, pd.Series],
    mrio_default_monetary_unit: str | None,
    mrio_units: dict[str, str],
    lcia_units: dict[str, pd.Series],
) -> pd.DataFrame:
    """Build one enacting metric wide DataFrame."""
    year_series = {str(int(y)): s for y, s in year_map.items()}
    df = pd.concat(year_series, axis=1)
    df = _reset_metric_index_strict(df)
    unit_col = resolve_enacting_metric_unit(
        context=context,
        key=key,
        year_map=year_map,
        mrio_default_monetary_unit=mrio_default_monetary_unit,
        mrio_units=mrio_units,
        lcia_units=lcia_units,
        df=df,
    )
    if isinstance(unit_col, pd.Series):
        df.insert(0, "unit", unit_col.reindex(df.index))
    else:
        df.insert(0, "unit", unit_col)
    assert_no_duplicate_columns(df, where=f"enacting metric {key}")
    return df


def _uses_regression_projection_metric(metric: str) -> bool:
    """Return whether an enacting metric is directly projection regressed."""
    normalized = str(metric).lower()
    return (
        normalized.startswith("fd_") or normalized.startswith("gva_") or normalized.startswith("x_")
    )


def _split_year_map_for_output(
    *,
    context,
    key: EnactingMetricKey,
    year_map: dict[int, pd.Series],
) -> list[tuple[str | None, dict[int, pd.Series]]]:
    """Split one enacting metric year map by projection subfolder routing."""
    projection_context = context.projection_context
    if projection_context is None or not projection_context.enabled:
        return [(None, year_map)]
    if projection_context.mode != "regression":
        return [(None, year_map)]
    if not _uses_regression_projection_metric(key.metric):
        return [(None, year_map)]
    projected = {
        int(year): series
        for year, series in year_map.items()
        if is_regression_projection_year(context=context, year=int(year))
    }
    historical = {
        int(year): series for year, series in year_map.items() if int(year) not in projected
    }
    out: list[tuple[str | None, dict[int, pd.Series]]] = []
    if historical:
        out.append((None, historical))
    if projected:
        out.append(
            (
                regression_projection_subfolder_for_context(context=context),
                projected,
            )
        )
    return out


def count_enacting_metric_output_targets(*, context, state) -> int:
    """Return number of enacting metric output files to write in current flush."""
    if not bool(getattr(context, "intermediate_outputs", True)):
        return 0
    total = 0
    for key, year_map in state.enacting_metric_inputs.items():
        if not year_map:
            continue
        total += len(
            _split_year_map_for_output(
                context=context,
                key=key,
                year_map=year_map,
            )
        )
    return int(total)


def _write_enacting_metric_outputs(
    *,
    context,
    state,
    refresh_effective: bool,
    l1_source: str | None,
    write_result_artifact: Callable[..., None],
) -> None:
    """Write enacting metrics outputs."""
    if not bool(getattr(context, "intermediate_outputs", True)):
        return
    for key, year_map in state.enacting_metric_inputs.items():
        if not year_map:
            continue
        for projection_subfolder, scoped_year_map in _split_year_map_for_output(
            context=context,
            key=key,
            year_map=year_map,
        ):
            artifact, out_path = _build_enacting_metric_output(
                context=context,
                state=state,
                key=key,
                year_map=scoped_year_map,
                l1_source=l1_source,
                projection_subfolder=projection_subfolder,
            )
            write_result_artifact(
                context=context,
                artifact=artifact,
                out_path=out_path,
                refresh_effective=refresh_effective,
                output_format=context.output_format,
                state=state,
            )
