"""Allocation artifact builders and write loops."""

from pathlib import Path

import pandas as pd

from ....runtime.output.contracts import (
    IdentifierSchema,
    OutputArtifact,
    OutputSpec,
    contract_year_columns,
    persisted_method_columns_for_output_spec,
)
from ....runtime.paths.published import (
    _get_asocc_l1_dir,
    _get_asocc_l2_dir,
    _owning_fu_level_for_code,
)
from ...projection.config.config import required_projection_years
from pyaesa.asocc.orchestration.write.tables.allocation_frame import prepare_allocation_frame
from pyaesa.asocc.orchestration.write.writers.progress import tick_write_progress
from pyaesa.asocc.orchestration.write.tables.wide_table_io import upsert_wide_table
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)


def _write_artifact(
    *,
    artifact: OutputArtifact,
    wide_path: Path,
    refresh: bool,
    output_format: str,
) -> bool:
    """Persist one artifact directly to deterministic wide output table."""
    changed = upsert_wide_table(
        path=wide_path,
        frame=artifact.data_wide,
        schema=artifact.schema,
        refresh=refresh,
        output_format=output_format,
    )
    return changed


def _build_allocation_artifact(
    *,
    output_spec: OutputSpec,
    df: pd.DataFrame,
    context,
    output_years: list[int],
) -> OutputArtifact:
    """Build one typed wide allocation artifact for L1 or L2 outputs."""
    method_columns = persisted_method_columns_for_output_spec(output_spec)
    identifier_columns = tuple(
        column for column in output_spec.identifier_columns if column not in method_columns
    )
    metadata_columns = tuple(
        column
        for column in (ASOCC_SSP_SCENARIO_COLUMN, ASOCC_TIME_ROUTE_PUBLIC_COLUMN)
        if column in df.columns
    )
    identifier_columns = (
        *metadata_columns,
        *(column for column in identifier_columns if column not in metadata_columns),
    )
    schema = IdentifierSchema(
        columns=(
            *method_columns,
            *identifier_columns,
        ),
        year_columns=(
            tuple(str(int(year)) for year in output_years) or contract_year_columns(context)
        ),
    )
    return OutputArtifact(
        schema=schema,
        data_wide=df,
    )


def _output_years_for_output_spec(*, context, output_spec: OutputSpec) -> list[int]:
    """Return the canonical persisted year contract for one output artifact."""
    years = sorted({int(year) for year in context.persisted_years})
    projection_context = getattr(context, "projection_context", None)
    if (
        (output_spec.route.bucket or "l2_vs_global") != "l2_in_l1"
        or projection_context is None
        or not bool(getattr(projection_context, "enabled", False))
        or output_spec.l2_method is None
    ):
        return years
    route_for_method = projection_context.route_for_l2_method
    if route_for_method(output_spec.l2_method) != "historical_reuse":
        return years
    # Historical reuse support tables are written from the deterministic owner
    # years needed to build future l2_vs_global rows. For adjusted UT routes in
    # regression mode that includes the regression fit window and the
    # selected L2 reuse years.
    support_years = required_projection_years(projection_context=projection_context)
    return sorted({*years, *(int(year) for year in support_years)})


def write_result_artifact(
    *,
    context,
    artifact: OutputArtifact,
    out_path: Path,
    refresh_effective: bool,
    output_format: str,
    state,
) -> None:
    """Write one artifact and update run output trackers."""
    out_path_str = str(out_path)
    existed_before = out_path.exists()
    changed = _write_artifact(
        artifact=artifact,
        wide_path=out_path,
        refresh=refresh_effective,
        output_format=output_format,
    )
    if not out_path.exists():
        # Keep write progress aligned with target counting even when the
        # current artifact produces an empty batch and no file is written.
        tick_write_progress(context=context, state=state)
        return

    if changed:
        # Track created/updated separately for end of run summary reporting.
        if existed_before:
            if (
                out_path_str not in state.output_files_created
                and out_path_str not in state.output_files_updated
            ):
                state.output_files_updated.append(out_path_str)
        else:
            state.output_files_created = list(
                dict.fromkeys([*state.output_files_created, out_path_str])
            )
            state.output_files_updated = [
                path for path in state.output_files_updated if path != out_path_str
            ]
        if out_path_str not in state.outputs_written:
            state.outputs_written.append(out_path_str)
    if out_path_str not in state.outputs_all:
        state.outputs_all.append(out_path_str)
    tick_write_progress(context=context, state=state)


def build_l1_artifact(
    *,
    output_spec: OutputSpec,
    df: pd.DataFrame,
    context,
) -> OutputArtifact:
    """Build typed wide output artifact for one L1 output."""
    return _build_allocation_artifact(
        output_spec=output_spec,
        df=df,
        context=context,
        output_years=_output_years_for_output_spec(context=context, output_spec=output_spec),
    )


def build_l2_artifact(
    *,
    output_spec: OutputSpec,
    df: pd.DataFrame,
    context,
) -> OutputArtifact:
    """Build typed wide output artifact for one L2 output."""
    return _build_allocation_artifact(
        output_spec=output_spec,
        df=df,
        context=context,
        output_years=_output_years_for_output_spec(context=context, output_spec=output_spec),
    )


def _collect_output_frames(
    results_by_scenario: dict[object, dict[OutputSpec, list[pd.DataFrame]]],
) -> dict[OutputSpec, list[pd.DataFrame]]:
    """Collect branch output frames by final public output specification."""
    collected: dict[OutputSpec, list[pd.DataFrame]] = {}
    for results in results_by_scenario.values():
        for output_spec, frames in results.items():
            collected.setdefault(output_spec, []).extend(frames)
    return collected


def count_l1_output_targets(*, state) -> int:
    """Return number of final L1 allocation tables written in the current flush."""
    return len(_collect_output_frames(state.l1_results_by_ssp_scenario))


def count_l2_output_targets(*, context, state) -> int:
    """Return number of final L2 allocation tables written in the current flush."""
    write_intermediate = bool(getattr(context, "intermediate_outputs", True))
    total = 0
    for output_spec in _collect_output_frames(state.l2_results_by_ssp_scenario):
        bucket = output_spec.route.bucket or "l2_vs_global"
        if not write_intermediate and bucket == "utility_propagation_contrib":
            continue
        total += 1
    return total


def write_l1_outputs(
    *,
    context,
    state,
    refresh_effective: bool,
    l1_source: str | None,
) -> None:
    """Write all L1 allocation outputs for all scenarios."""
    base = _get_asocc_l1_dir(
        proj_base=context.proj_base,
        source=str(l1_source if l1_source is not None else context.source),
        group_version=context.group_version,
        lcia_sub=None,
        owning_fu_level=_owning_fu_level_for_code(fu_code=context.fu_code),
    )
    for output_spec, frames in _collect_output_frames(state.l1_results_by_ssp_scenario).items():
        output_years = _output_years_for_output_spec(context=context, output_spec=output_spec)
        df = prepare_allocation_frame(
            output_spec=output_spec,
            frames=frames,
            filters=context.filters,
            aggreg_indices=context.aggreg_indices,
            persisted_years=output_years,
        )
        artifact = build_l1_artifact(
            output_spec=output_spec,
            df=df,
            context=context,
        )
        write_result_artifact(
            context=context,
            artifact=artifact,
            out_path=base / output_spec.file_name_for_format(context.output_format),
            refresh_effective=refresh_effective,
            output_format=context.output_format,
            state=state,
        )


def write_l2_outputs(
    *,
    context,
    state,
    refresh_effective: bool,
    allowed_buckets: set[str] | None = None,
) -> None:
    """Write all L2 allocation outputs for all scenarios."""
    write_intermediate = bool(getattr(context, "intermediate_outputs", True))
    output_source = context.output_source
    for output_spec, frames in _collect_output_frames(state.l2_results_by_ssp_scenario).items():
        bucket = output_spec.route.bucket or "l2_vs_global"
        if allowed_buckets is not None and bucket not in allowed_buckets:
            continue
        if not write_intermediate and bucket == "utility_propagation_contrib":
            continue
        output_years = _output_years_for_output_spec(context=context, output_spec=output_spec)
        df = prepare_allocation_frame(
            output_spec=output_spec,
            frames=frames,
            filters=context.filters,
            aggreg_indices=context.aggreg_indices,
            persisted_years=output_years,
        )
        base = _get_asocc_l2_dir(
            proj_base=context.proj_base,
            source=output_source,
            group_version=context.group_version,
            bucket=bucket,
            lcia_sub=output_spec.route.projection_subfolder,
        )
        artifact = build_l2_artifact(
            output_spec=output_spec,
            df=df,
            context=context,
        )
        write_result_artifact(
            context=context,
            artifact=artifact,
            out_path=base / output_spec.file_name_for_format(context.output_format),
            refresh_effective=refresh_effective,
            output_format=context.output_format,
            state=state,
        )
