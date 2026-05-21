"""Deterministic AR6 CC prerequisite ownership for uncertainty."""

from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.ar6_cc.deterministic_ar6_cc import deterministic_ar6_cc
from pyaesa.ar6_cc.deterministic.io.tables import read_cc_output
from pyaesa.ar6_cc.deterministic.io.paths import (
    get_cc_metadata_path,
    get_cc_scope_dir,
)
from pyaesa.ar6_cc.deterministic.runtime.metadata import (
    load_run_metadata,
)
from pyaesa.shared.runtime.reporting.status import StatusSink

from .models import AR6CCDeterministicScope, AR6CCUncertaintyRequest


def prepare_ar6_cc_deterministic_prerequisite(
    *,
    request: AR6CCUncertaintyRequest,
    refresh: bool,
    figures: bool = False,
    figure_format: dict[str, Any] | None = None,
    status: StatusSink | None = None,
) -> AR6CCDeterministicScope:
    """Run or reuse deterministic AR6 CC and return the resolved scope."""
    deterministic_args = {
        **request.deterministic_args,
        "figures": figures,
        "figure_format": figure_format,
        "refresh": refresh,
        "_status": status,
    }
    report = deterministic_ar6_cc(**deterministic_args)
    cc_dir = get_cc_scope_dir(
        request.study_period,
        harmonization=request.harmonization,
        harmonization_method=request.harmonization_method,
        emission_type=request.emission_type,
        include_afolu=request.include_afolu,
        emissions_mode=request.emissions_mode,
        subset_version=request.subset_version,
        category=request.category,
        ssp_scenario=request.ssp_scenario,
    )
    metadata_path = get_cc_metadata_path(cc_dir=cc_dir)
    metadata = cast(dict[str, Any], load_run_metadata(metadata_path))
    execution = cast(dict[str, Any], metadata["execution"])
    reuse = cast(dict[str, Any], metadata["reuse"])
    artifacts = cast(dict[str, Any], metadata["artifacts"])
    provenance = cast(dict[str, Any], metadata["provenance"])
    return AR6CCDeterministicScope(
        metadata_path=metadata_path,
        reuse_status=report.reuse_status,
        output_file=Path(cast(str, artifacts["output_file"])),
        post_study_output_file=(
            None
            if artifacts.get("post_study_output_file") is None
            else Path(cast(str, artifacts["post_study_output_file"]))
        ),
        output_format="csv",
        scope_key=cast(str, reuse["write_scope_key"]),
        emission_type=request.emission_type,
        include_afolu=request.include_afolu,
        variable=cast(str, provenance["variable"]),
        emissions_mode=cast(str, provenance["emissions_mode"]),
        categories=tuple(cast(list[str], provenance["cc_categories"])),
        ssp_scenarios=tuple(cast(list[str], provenance["ssp_scenarios"])),
        subset_version=request.subset_version,
        pathway_counts=tuple(cast(list[dict[str, object]], execution["pathway_counts"])),
        missing_pathway_combinations=tuple(
            cast(list[dict[str, object]], execution["missing_pathway_combinations"])
        ),
        process_ar6=cast(dict[str, object], provenance["process_ar6"]),
    )


def load_deterministic_ar6_cc_rows(
    *,
    request: AR6CCUncertaintyRequest,
    scope: AR6CCDeterministicScope,
) -> pd.DataFrame:
    """Load deterministic AR6 CC trajectory rows for the requested uncertainty scope."""
    frame = read_cc_output(output_file=scope.output_file, output_format=scope.output_format)
    return _filter_deterministic_rows(request=request, frame=frame)


def load_deterministic_ar6_cc_post_study_rows(
    *,
    request: AR6CCUncertaintyRequest,
    scope: AR6CCDeterministicScope,
) -> pd.DataFrame | None:
    """Load deterministic AR6 CC post study rows for uncertainty figures."""
    if scope.post_study_output_file is None:
        return None
    frame = read_cc_output(
        output_file=scope.post_study_output_file,
        output_format=scope.output_format,
    )
    return _filter_deterministic_rows(request=request, frame=frame)


def _filter_deterministic_rows(
    *,
    request: AR6CCUncertaintyRequest,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return deterministic rows inside the requested AR6 CC public selectors."""
    rows = frame.copy()
    rows = rows.loc[
        rows["cc_category"].astype(str).isin(request.category)
        & rows["ssp_scenario"].astype(str).isin(request.ssp_scenario)
    ].copy()
    return rows.reset_index(drop=True)
