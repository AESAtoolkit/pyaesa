"""LCA numerator input resolution for ASR uncertainty."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.external_inputs.lca.deterministic import (
    load_external_lca_deterministic_rows,
    load_external_lca_deterministic_rows_from_paths,
)
from pyaesa.external_inputs.lca.figures import (
    render_external_lca_deterministic_figures_from_rows,
    render_external_lca_uncertainty_figures_from_source,
)
from pyaesa.external_inputs.lca.monte_carlo import (
    ExternalLCAMonteCarloSource,
    external_lca_values_for_runs,
    external_lca_values_for_units,
    load_external_lca_monte_carlo_source,
    load_external_lca_monte_carlo_source_from_path,
)
from pyaesa.external_inputs.lca.paths import (
    external_lca_deterministic_dir,
    external_lca_monte_carlo_dir,
    external_lca_root,
)
from pyaesa.io_lca.data.contracts import IO_LCA_FAMILY
from pyaesa.io_lca.uncertainty.runtime.prerequisites import (
    load_deterministic_public_rows,
    prepare_io_lca_deterministic_prerequisite,
)
from pyaesa.io_lca.uncertainty.request.normalization import normalize_io_lca_uncertainty_request
from pyaesa.io_lca.uncertainty.figures.reuse import (
    render_reusable_io_lca_figures_if_requested,
)
from pyaesa.io_lca.uncertainty.runner import run_uncertainty_io_lca_component
from pyaesa.shared.figures.contracts import SELECTOR_COLUMNS
from pyaesa.shared.figures.request_validation import normalize_figure_format
from pyaesa.shared.lcia.uncertainty_source import LCIA_SOURCE
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_A_LCA,
    phase_ready_detail,
    phase_reused_detail,
)
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import RunProgressPrinter
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.scenario.columns import EXT_LCA_SSP_SCENARIO_COLUMN
from pyaesa.shared.selectors.time_selectors import normalize_requested_years
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import (
    ComponentInput,
    fixed_inventory_mc_parameters,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import iter_compact_run_matrix
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table
from pyaesa.shared.uncertainty_assessment.orchestration import (
    manifest_output_root,
    output_root_from_path,
)

from pyaesa.asr.uncertainty.runtime.models import LCAUncertaintyInput
from pyaesa.asr.uncertainty.sources.source_keys import (
    external_lca_source_name,
    io_lca_source_name,
)


def resolve_lca_uncertainty_component_input(
    *,
    proj_base: Path,
    source_label: str,
    lca_type: str,
    lca_version_name: str | None,
    base_allocate_args: dict[str, Any],
    lcia_methods: list[str],
    uncertainty_config: dict[str, Any],
    output_format: str,
    refresh: bool,
    phase: PhasePrinter,
    status: StatusSink,
    progress: RunProgressPrinter | None = None,
    component_inventory: dict[str, Any] | None = None,
    show_progress: bool = True,
    figures: bool = False,
    figure_format: dict[str, Any] | None = None,
    run_id: str | None = None,
    component_session: Any | None = None,
    finalize_component_inventory: bool = False,
    figure_run_count: int | None = None,
) -> ComponentInput[LCAUncertaintyInput]:
    """Resolve LCA input and local session for ASR component checkpoints."""
    if lca_type == IO_LCA_FAMILY:
        return _io_lca_input(
            source_label=source_label,
            base_allocate_args=base_allocate_args,
            lcia_methods=lcia_methods,
            uncertainty_config=uncertainty_config,
            output_format=output_format,
            refresh=refresh,
            phase=phase,
            component_inventory=cast(dict[str, Any], component_inventory),
            show_progress=show_progress,
            figures=figures,
            figure_format=figure_format,
            run_id=run_id,
            progress=progress,
            component_session=component_session,
            finalize_component_inventory=finalize_component_inventory,
        )
    phase.announce(PHASE_A_LCA, f"external_lca version={lca_version_name}")
    phase.status("Loading external LCA inputs", owner="external_lca")
    external = _external_lca_input(
        proj_base=proj_base,
        source_label=source_label,
        lca_version_name=cast(str, lca_version_name),
        base_allocate_args=base_allocate_args,
        lcia_methods=lcia_methods,
        figures=figures,
        figure_format=figure_format,
        status=status,
        figure_run_count=figure_run_count,
    )
    phase.complete(
        phase_ready_detail(
            scope_name=f"LCA external {cast(str, lca_version_name)}",
            output_root=external.phase_output_root,
        )
    )
    return ComponentInput(input=external, session=None)


def lca_values_for_runs(*, lca_input: LCAUncertaintyInput, run_indices: np.ndarray) -> np.ndarray:
    """Return LCA numerator values for package run indices."""
    if lca_input.run_values_for_runs is not None:
        requested = np.asarray(run_indices, dtype=np.int64)
        if (
            requested.size
            and lca_input.run_inventory_size is not None
            and int(requested.max()) >= int(lca_input.run_inventory_size)
        ):
            raise ValueError(
                "LCA run inventory was exhausted before ASR Monte Carlo convergence was "
                "reached. Provide more LCA runs or run a fixed run request within the "
                "available inventory. "
                f"Requested maximum run_index={int(requested.max())}; "
                f"available run count={int(lca_input.run_inventory_size)}."
            )
        return lca_input.run_values_for_runs(requested)
    return np.broadcast_to(
        cast(np.ndarray, lca_input.fixed_values),
        (len(run_indices), len(lca_input.identity)),
    )


def render_lca_subfigures_from_input(
    *,
    lca_input: LCAUncertaintyInput,
    base_allocate_args: dict[str, Any],
    lcia_methods: list[str],
    lca_version_name: str | None,
    lca_config: dict[str, Any],
    figure_format: dict[str, Any] | None,
    status: StatusSink,
    completed_runs: int,
) -> None:
    """Render persisted LCA subfigures after the parent ASR run is final."""
    if lca_input.lca_type == IO_LCA_FAMILY:
        request = normalize_io_lca_uncertainty_request(
            base_io_lca_args=base_io_lca_args_from_allocate_args(
                base_allocate_args={**base_allocate_args, "lcia_method": lcia_methods}
            ),
            lcia_parameters=cast(dict[str, Any], lca_config.get(LCIA_SOURCE, {})),
        )
        if lca_input.manifest is not None:
            render_reusable_io_lca_figures_if_requested(
                manifest=lca_input.manifest,
                request=request,
                figures=True,
                figure_options=None,
                figure_format=figure_format,
                status=status,
            )
            return
        prepare_io_lca_deterministic_prerequisite(
            request=request,
            refresh=False,
            figures=True,
            figure_format=figure_format,
            status=status,
        )
        return
    version_name = cast(str, lca_version_name)
    years = normalize_requested_years(base_allocate_args["years"])
    selected_methods = {str(method) for method in lcia_methods}
    project_base = cast(Path, lca_input.phase_output_root).parents[1]
    for item in lca_input.external_inputs:
        lcia_method = str(item["lcia_method"])
        if lcia_method not in selected_methods:
            continue
        paths = tuple(Path(str(path)) for path in item.get("paths", ()))
        if item.get("type") == "external_lca_monte_carlo":
            for path in paths:
                source = load_external_lca_monte_carlo_source_from_path(
                    path=path,
                    version_name=version_name,
                    lcia_method=lcia_method,
                    years=years,
                    base_allocate_args=base_allocate_args,
                )
                _render_external_lca_uncertainty_subfigures(
                    proj_base=project_base,
                    source=source,
                    figure_format=figure_format,
                    status=status,
                    completed_runs=completed_runs,
                )
        elif item.get("type") == "external_lca_deterministic":
            rows = _asr_external_deterministic_rows(
                rows=load_external_lca_deterministic_rows_from_paths(
                    paths=paths,
                    lcia_method=lcia_method,
                    years=years,
                    base_allocate_args=base_allocate_args,
                ),
                lcia_method=lcia_method,
            )
            _render_external_lca_deterministic_subfigures(
                proj_base=project_base,
                lca_version_name=version_name,
                lcia_method=lcia_method,
                rows=rows,
                figure_format=figure_format,
                status=status,
            )


def _io_lca_input(
    *,
    source_label: str,
    base_allocate_args: dict[str, Any],
    lcia_methods: list[str],
    uncertainty_config: dict[str, Any],
    output_format: str,
    refresh: bool,
    phase: PhasePrinter,
    component_inventory: dict[str, Any],
    show_progress: bool,
    progress: RunProgressPrinter | None,
    figures: bool,
    figure_format: dict[str, Any] | None,
    run_id: str | None,
    component_session: Any | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentInput[LCAUncertaintyInput]:
    base_io_lca_args = base_io_lca_args_from_allocate_args(
        base_allocate_args={**base_allocate_args, "lcia_method": lcia_methods}
    )
    lcia_active = lcia_uncertainty_source_active(uncertainty_config)
    if lcia_active:
        run = run_uncertainty_io_lca_component(
            base_io_lca_args=base_io_lca_args,
            uncertainty_config=_io_lca_uncertainty_config(
                config=uncertainty_config,
                component_inventory=component_inventory,
            ),
            output_format=output_format,
            figures=figures,
            figure_options=None,
            figure_format=figure_format,
            refresh=refresh,
            phase=phase,
            component_inventory=component_inventory,
            run_id=run_id,
            show_progress=show_progress,
            progress=progress,
            component_session=component_session,
            finalize_component_inventory=finalize_component_inventory,
        )
        manifest = run.report.manifest
        artifacts = cast(dict[str, Any], manifest.artifacts)
        identity = read_uncertainty_table(
            path=Path(artifacts["public_row_identity"]),
            output_format=output_format,
        )
        source_methods = pd.read_csv(artifacts["source_methods"])
        return ComponentInput(
            input=LCAUncertaintyInput(
                identity=identity,
                fixed_values=None,
                manifest=manifest,
                phase_function="uncertainty_io_lca",
                phase_reuse_status=run.report.reuse_status,
                phase_output_root=manifest_output_root(manifest),
                external_inputs=(),
                source_method_rows=source_methods,
                active_sources=tuple(
                    io_lca_source_name(source) for source in manifest.active_sources
                ),
                lca_type=IO_LCA_FAMILY,
                run_values_for_runs=_public_lca_run_value_provider(
                    runs_path=Path(artifacts["lca_runs"]),
                    output_format=output_format,
                    column_count=len(identity),
                    run_count=int(manifest.completed_runs),
                ),
                run_inventory_size=int(manifest.completed_runs),
            ),
            session=run.session,
        )
    request = normalize_io_lca_uncertainty_request(
        base_io_lca_args=base_io_lca_args,
        lcia_parameters={},
    )
    phase.announce(PHASE_A_LCA, "uncertainty_io_lca")
    prerequisite = prepare_io_lca_deterministic_prerequisite(
        request=request,
        refresh=refresh,
        figures=figures,
        figure_format=figure_format,
        status=phase,
    )
    detail_builder = (
        phase_reused_detail if prerequisite.reuse_status == "reused_exact" else phase_ready_detail
    )
    phase.complete(
        detail_builder(
            scope_name="LCA deterministic",
            output_root=output_root_from_path(prerequisite.metadata_path),
        )
    )
    phase.status("Loading deterministic IO-LCA outputs", owner="deterministic_io_lca")
    rows = load_deterministic_public_rows(request=request, scope=prerequisite)
    identity = _identity_from_lca_rows(rows=rows)
    return ComponentInput(
        input=LCAUncertaintyInput(
            identity=identity,
            fixed_values=rows["lca_value"].to_numpy(dtype=np.float64),
            manifest=None,
            phase_function="deterministic_io_lca",
            phase_reuse_status=prerequisite.reuse_status,
            phase_output_root=output_root_from_path(prerequisite.metadata_path),
            external_inputs=(
                {
                    "type": "io_lca_deterministic",
                    "source": source_label,
                    "metadata_path": str(prerequisite.metadata_path),
                    "reuse_status": prerequisite.reuse_status,
                    "output_format": prerequisite.output_format,
                },
            ),
            source_method_rows=pd.DataFrame(),
            active_sources=(),
            lca_type=IO_LCA_FAMILY,
        ),
        session=None,
    )


def _external_lca_input(
    *,
    proj_base: Path,
    source_label: str,
    lca_version_name: str,
    base_allocate_args: dict[str, Any],
    lcia_methods: list[str],
    figures: bool,
    figure_format: dict[str, Any] | None,
    status: StatusSink,
    figure_run_count: int | None,
) -> LCAUncertaintyInput:
    methods = [str(value) for value in lcia_methods]
    years = normalize_requested_years(base_allocate_args["years"])
    identities: list[pd.DataFrame] = []
    value_blocks: list[np.ndarray | ExternalLCAMonteCarloSource] = []
    mc_sources: list[ExternalLCAMonteCarloSource] = []
    external_inputs: list[dict[str, Any]] = []
    for method in methods:
        status.show(f"[external_lca] Loading external LCA input for {method}")
        mc_source = load_external_lca_monte_carlo_source(
            proj_base=proj_base,
            version_name=lca_version_name,
            lcia_method=method,
            years=years,
            base_allocate_args=base_allocate_args,
        )
        if mc_source is not None:
            figure_paths: list[Path] = []
            if figures:
                figure_paths = _render_external_lca_uncertainty_subfigures(
                    proj_base=proj_base,
                    source=mc_source,
                    figure_format=figure_format,
                    status=status,
                    completed_runs=figure_run_count,
                )
            identities.append(mc_source.identity.drop(columns=["public_row_id"]))
            mc_sources.append(mc_source)
            value_blocks.append(mc_source)
            external_inputs.append(
                {
                    "type": "external_lca_monte_carlo",
                    "reuse_status": "computed",
                    "version_name": lca_version_name,
                    "lcia_method": method,
                    "output_root": str(external_lca_root(project_base=proj_base)),
                    "paths": [str(path) for path in mc_source.paths],
                    "figures_available": len(figure_paths) if figures else None,
                }
            )
            continue
        deterministic, deterministic_paths = _external_deterministic_rows(
            proj_base=proj_base,
            lca_version_name=lca_version_name,
            lcia_method=method,
            years=years,
            base_allocate_args=base_allocate_args,
        )
        figure_paths = []
        if figures:
            figure_paths = _render_external_lca_deterministic_subfigures(
                proj_base=proj_base,
                lca_version_name=lca_version_name,
                lcia_method=method,
                rows=deterministic,
                figure_format=figure_format,
                status=status,
            )
        identities.append(
            _identity_from_lca_rows(rows=deterministic).drop(columns=["public_row_id"])
        )
        value_blocks.append(deterministic["lca_value"].to_numpy(dtype=np.float64))
        external_inputs.append(
            {
                "type": "external_lca_deterministic",
                "reuse_status": "computed",
                "version_name": lca_version_name,
                "lcia_method": method,
                "output_root": str(external_lca_root(project_base=proj_base)),
                "paths": [str(path) for path in deterministic_paths],
                "figures_available": len(figure_paths) if figures else None,
            }
        )
    identity = pd.concat(identities, ignore_index=True)
    identity.insert(0, "public_row_id", np.arange(len(identity), dtype=np.int64))
    fixed_values = _external_fixed_values(value_blocks=value_blocks)
    run_values_for_runs = (
        _external_lca_run_value_provider(
            identity=identity,
            value_blocks=value_blocks,
            version_name=lca_version_name,
        )
        if mc_sources
        else None
    )
    run_values_for_units = (
        _external_lca_unit_value_provider(identity=identity, value_blocks=value_blocks)
        if mc_sources
        else None
    )
    return LCAUncertaintyInput(
        identity=identity,
        fixed_values=None if mc_sources else fixed_values,
        manifest=None,
        phase_function="external_lca",
        phase_reuse_status="computed",
        phase_output_root=external_lca_root(project_base=proj_base),
        external_inputs=tuple(external_inputs),
        source_method_rows=_external_source_methods(
            version_name=lca_version_name,
            mc_sources=tuple(mc_sources),
        ),
        active_sources=((external_lca_source_name(lca_version_name),) if mc_sources else ()),
        lca_type="external",
        run_values_for_runs=run_values_for_runs,
        run_values_for_units=run_values_for_units,
        run_inventory_size=(
            min(len(source.run_indices) for source in mc_sources) if mc_sources else None
        ),
    )


def _render_external_lca_uncertainty_subfigures(
    *,
    proj_base: Path,
    source: ExternalLCAMonteCarloSource,
    figure_format: dict[str, Any] | None,
    status: StatusSink | None,
    completed_runs: int | None,
) -> list[Path]:
    figure = normalize_figure_format(figure_format)
    return render_external_lca_uncertainty_figures_from_source(
        proj_base=proj_base,
        source=source,
        output_format=str(figure["format"]),
        dpi=int(figure["dpi"]),
        completed_runs=completed_runs,
        status=status,
    )


def _render_external_lca_deterministic_subfigures(
    *,
    proj_base: Path,
    lca_version_name: str,
    lcia_method: str,
    rows: pd.DataFrame,
    figure_format: dict[str, Any] | None,
    status: StatusSink | None,
) -> list[Path]:
    figure = normalize_figure_format(figure_format)
    return render_external_lca_deterministic_figures_from_rows(
        proj_base=proj_base,
        version_name=lca_version_name,
        lcia_method=lcia_method,
        rows=rows,
        value_column="lca_value",
        output_format=str(figure["format"]),
        dpi=int(figure["dpi"]),
        status=status,
    )


def _external_fixed_values(
    *,
    value_blocks: list[np.ndarray | ExternalLCAMonteCarloSource],
) -> np.ndarray | None:
    mc_sources = [block for block in value_blocks if isinstance(block, ExternalLCAMonteCarloSource)]
    if mc_sources:
        return None
    arrays = [cast(np.ndarray, block) for block in value_blocks]
    return np.concatenate(arrays)


def _external_lca_run_value_provider(
    *,
    identity: pd.DataFrame,
    value_blocks: list[np.ndarray | ExternalLCAMonteCarloSource],
    version_name: str,
):
    run_count = min(
        len(block.run_indices)
        for block in value_blocks
        if isinstance(block, ExternalLCAMonteCarloSource)
    )
    external_cache: dict[int, np.ndarray] = {}
    external_cache_stop: dict[int, int] = {}

    def provider(run_indices: np.ndarray) -> np.ndarray:
        requested = np.asarray(run_indices, dtype=np.int64)
        if requested.size == 0:
            return np.empty((0, len(identity)), dtype=np.float64)
        if int(requested.max()) >= int(run_count):
            raise ValueError(
                "External LCA Monte Carlo run inventory was exhausted before ASR Monte Carlo "
                "convergence was reached. "
                f"version_name='{version_name}', available run_index range 0 to "
                f"{run_count - 1}, first missing run_index={run_count}. Provide more "
                "external LCA Monte Carlo runs or run a fixed request within the available "
                "external inventory."
            )
        values = np.empty((len(requested), len(identity)), dtype=np.float64)
        start = 0
        target_stop = int(requested.max()) + 1
        for block_index, block in enumerate(value_blocks):
            if isinstance(block, ExternalLCAMonteCarloSource):
                stop = start + len(block.identity)
                cached_stop = int(external_cache_stop.get(block_index, 0))
                if target_stop > cached_stop:
                    prefetch_stop = min(
                        int(run_count),
                        max(target_stop, cached_stop * 2, target_stop + len(requested)),
                    )
                    new_values = external_lca_values_for_runs(
                        source=block,
                        run_indices=np.arange(cached_stop, prefetch_stop, dtype=np.int64),
                    )
                    cached = external_cache.get(block_index)
                    if cached is None:
                        external_cache[block_index] = new_values
                    else:
                        extended = np.empty(
                            (prefetch_stop, cached.shape[1]),
                            dtype=np.float64,
                        )
                        extended[:cached_stop] = cached
                        extended[cached_stop:prefetch_stop] = new_values
                        external_cache[block_index] = extended
                    external_cache_stop[block_index] = prefetch_stop
                values[:, start:stop] = external_cache[block_index][requested]
            else:
                base = cast(np.ndarray, block)
                stop = start + len(base)
                values[:, start:stop] = base[None, :]
            start = stop
        return values

    return provider


def _external_lca_unit_value_provider(
    *,
    identity: pd.DataFrame,
    value_blocks: list[np.ndarray | ExternalLCAMonteCarloSource],
):
    def provider(unit_values: np.ndarray) -> np.ndarray:
        units = np.asarray(unit_values, dtype=np.float64)
        values = np.empty((len(units), len(identity)), dtype=np.float64)
        start = 0
        for block in value_blocks:
            if isinstance(block, ExternalLCAMonteCarloSource):
                stop = start + len(block.identity)
                values[:, start:stop] = external_lca_values_for_units(
                    source=block,
                    unit_values=units,
                )
            else:
                base = cast(np.ndarray, block)
                stop = start + len(base)
                values[:, start:stop] = base[None, :]
            start = stop
        return values

    return provider


def _public_lca_run_value_provider(
    *,
    runs_path: Path,
    output_format: str,
    column_count: int,
    run_count: int,
):
    def provider(run_indices: np.ndarray) -> np.ndarray:
        requested = np.asarray(run_indices, dtype=np.int64)
        if requested.size == 0:
            return np.empty((0, int(column_count)), dtype=np.float64)
        if int(requested.max()) >= int(run_count):
            raise ValueError(
                "LCA run inventory was exhausted before ASR Monte Carlo convergence was "
                "reached. Provide more LCA runs or run a fixed run request within the "
                "available inventory. "
                f"Requested maximum run_index={int(requested.max())}; "
                f"available run count={int(run_count)}."
            )
        start = int(requested.min())
        stop = int(requested.max()) + 1
        pieces = [
            (chunk_runs, values)
            for chunk_runs, values in iter_compact_run_matrix(
                path=runs_path,
                output_format=output_format,
                column_count=int(column_count),
                start_run_index=start,
                stop_run_index=stop,
                max_rows_per_chunk=len(requested),
            )
        ]
        source_values = np.vstack([values for _chunk_runs, values in pieces])
        return source_values[requested - start]

    return provider


def _external_deterministic_rows(
    *,
    proj_base: Path,
    lca_version_name: str,
    lcia_method: str,
    years: list[int],
    base_allocate_args: dict[str, Any],
) -> tuple[pd.DataFrame, tuple[Path, ...]]:
    rows, paths = load_external_lca_deterministic_rows(
        proj_base=proj_base,
        version_name=lca_version_name,
        lcia_method=lcia_method,
        years=years,
        ssp_scenario_options_by_year=None,
        base_allocate_args=base_allocate_args,
    )
    if rows is None:
        deterministic_dir = external_lca_deterministic_dir(project_base=proj_base)
        monte_carlo_dir = external_lca_monte_carlo_dir(project_base=proj_base)
        expected_stem = f"{lca_version_name}__{lcia_method}"
        raise FileNotFoundError(
            "uncertainty_asr could not find external LCA input files for "
            f"version_name='{lca_version_name}' and lcia_method='{lcia_method}'. "
            f"Expected Monte Carlo stem '{expected_stem}' under {monte_carlo_dir}, "
            f"or deterministic stem '{expected_stem}' under {deterministic_dir}."
        )
    return _asr_external_deterministic_rows(rows=rows, lcia_method=lcia_method), paths


def _asr_external_deterministic_rows(*, rows: pd.DataFrame, lcia_method: str) -> pd.DataFrame:
    out = rows.rename(columns={"value": "lca_value"}).copy()
    out["lcia_method"] = str(lcia_method)
    return out


def _identity_from_lca_rows(*, rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["lcia_method", "year", "impact", "impact_unit"]
    columns.extend(column for column in SELECTOR_COLUMNS if column in rows.columns)
    if EXT_LCA_SSP_SCENARIO_COLUMN in rows.columns:
        columns.append(EXT_LCA_SSP_SCENARIO_COLUMN)
    extras = [column for column in rows.columns if column not in {*columns, "lca_value", "value"}]
    out = rows.loc[:, [*columns, *extras]].copy().reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    out.insert(0, "public_row_id", np.arange(len(out), dtype=np.int64))
    return out


def _external_source_methods(
    *,
    version_name: str,
    mc_sources: tuple[ExternalLCAMonteCarloSource, ...],
) -> pd.DataFrame:
    if not mc_sources:
        return pd.DataFrame()
    return pd.DataFrame.from_records(
        [
            {
                "source_component": "external_lca",
                "source_name": external_lca_source_name(version_name),
                "lcia_method": source.lcia_method,
                "run_inventory_size": int(len(source.run_indices)),
                "formula": "ASR numerator uses user supplied external LCA run values.",
            }
            for source in mc_sources
        ]
    )


def base_io_lca_args_from_allocate_args(*, base_allocate_args: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "project_name",
        "source",
        "agg_reg",
        "agg_sec",
        "agg_version",
        "years",
        "lcia_method",
        "fu_code",
        "r_f",
        "r_c",
        "r_p",
        "s_p",
        "group_indices",
    )
    return {key: base_allocate_args[key] for key in keys}


def _io_lca_uncertainty_config(
    *,
    config: dict[str, Any],
    component_inventory: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mc_parameters": fixed_inventory_mc_parameters(
            target_runs=int(component_inventory["target_runs"])
        ),
        LCIA_SOURCE: config[LCIA_SOURCE],
    }


def lcia_uncertainty_source_active(config: dict[str, Any]) -> bool:
    value = config.get(LCIA_SOURCE)
    return value is True or isinstance(value, dict)
