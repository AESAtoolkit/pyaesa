"""Deterministic IO-LCA prerequisite ownership for uncertainty."""

from pathlib import Path
from typing import cast

import pandas as pd

from pyaesa.io_lca.data.loaders import load_io_lca_method_table
from pyaesa.io_lca.data.metadata import (
    get_scope,
    load_scope_manifest,
    require_scope_signature,
)
from pyaesa.io_lca.data.paths import (
    IOLCAPaths,
    io_metadata_path_for_source,
    main_results_path,
    resolve_io_lca_paths,
)
from pyaesa.io_lca.deterministic_io_lca import deterministic_io_lca
from pyaesa.io_lca.orchestration.figure_support import (
    done_and_skipped_lcia_years,
    validate_lcia_method_coverage,
)
from pyaesa.io_lca.orchestration.pipeline.run_signatures import (
    build_io_lca_signature,
    table_extension_for_output,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import build_compatibility_key
from pyaesa.shared.runtime.reporting.status import StatusSink

from .models import IOLCADeterministicScope, IOLCAUncertaintyRequest


def prepare_io_lca_deterministic_prerequisite(
    *,
    request: IOLCAUncertaintyRequest,
    refresh: bool,
    figures: bool = False,
    figure_format: dict[str, object] | None = None,
    status: StatusSink | None = None,
) -> IOLCADeterministicScope:
    """Run or reuse deterministic IO-LCA and return the resolved main result scope."""
    deterministic_args = {
        **request.deterministic_args,
        "figures": figures,
        "figure_format": figure_format,
        "refresh": refresh,
        "_status": status,
    }
    report = deterministic_io_lca(**deterministic_args)
    paths = resolve_io_lca_paths(
        project_name=request.project_name,
        agg_reg=request.agg_reg,
        agg_sec=request.agg_sec,
        agg_version=request.agg_version,
    )
    metadata_path = io_metadata_path_for_source(paths=paths, source=request.source)
    payload = load_scope_manifest(path=metadata_path, function_name="deterministic_io_lca")
    output_format = "csv"
    signature = build_io_lca_signature(
        project_name=request.project_name,
        source=request.source,
        agg_reg=request.agg_reg,
        agg_sec=request.agg_sec,
        agg_version=request.agg_version,
        years=request.years,
        methods=request.lcia_methods,
        fu_code=request.fu_spec.fu_code,
        filters={
            key: value for key, value in request.filters.items() if key != "studied_indices_tag"
        },
        upstream_analysis=False,
        upstream_stages=3,
        group_indices=request.group_indices,
        output_format=output_format,
    )
    _scope_key, scope = get_scope(payload=payload, signature=signature)
    scope = cast(dict, scope)
    completed: dict[str, tuple[int, ...]] = {}
    for method in request.lcia_methods:
        validate_lcia_method_coverage(io_scope=scope, lcia_method=method, years=request.years)
        done, _skipped = done_and_skipped_lcia_years(scope=scope, lcia_method=method)
        completed[method] = tuple(sorted(int(year) for year in done if int(year) in request.years))
    signature = require_scope_signature(scope=scope)
    deterministic_paths = tuple(
        str(
            _method_path(
                paths=paths,
                source=request.source,
                output_format=output_format,
                lcia_method=method,
            )
        )
        for method in request.lcia_methods
    )
    return IOLCADeterministicScope(
        paths=paths,
        source=request.source,
        metadata_path=metadata_path,
        scope_key=build_compatibility_key(signature),
        output_format=output_format,
        completed_years_by_method=completed,
        deterministic_paths=deterministic_paths,
        reuse_status=report.reuse_status,
    )


def load_deterministic_public_rows(
    *,
    request: IOLCAUncertaintyRequest,
    scope: IOLCADeterministicScope,
) -> pd.DataFrame:
    """Load deterministic IO-LCA main result rows for the uncertainty public surface."""
    rows = [
        _read_method_rows(
            request=request,
            scope=scope,
            lcia_method=method,
        )
        for method in request.lcia_methods
        if scope.completed_years_by_method.get(method)
    ]
    if not rows:
        raise ValueError(
            "uncertainty_io_lca found no deterministic IO-LCA rows for the requested "
            "LCIA methods and years. The deterministic prerequisite skipped every "
            "requested method year; check deterministic_io_lca scope_manifest.json "
            "or request years with available processed MRIO LCIA data."
        )
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


def _read_method_rows(
    *,
    request: IOLCAUncertaintyRequest,
    scope: IOLCADeterministicScope,
    lcia_method: str,
):
    path = _method_path(
        paths=scope.paths,
        source=scope.source,
        output_format=scope.output_format,
        lcia_method=lcia_method,
    )
    frame = load_io_lca_method_table(path=path)
    frame = frame.loc[
        frame["year"].astype(int).isin(scope.completed_years_by_method[lcia_method])
    ].copy()
    frame["year"] = frame["year"].astype(int)
    frame["lcia_method"] = str(lcia_method)
    frame["lca_value"] = frame["lca_value"].astype(float)
    return frame


def _method_path(
    *,
    paths: IOLCAPaths,
    source: str,
    output_format: str,
    lcia_method: str,
) -> Path:
    return main_results_path(
        paths=paths,
        source=source,
        lcia_method=lcia_method,
        extension=table_extension_for_output(output_format),
    )
