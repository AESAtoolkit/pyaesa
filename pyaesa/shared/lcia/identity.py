"""Scientific input identities for characterization and dependent result reuse."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd

from pyaesa.shared.lcia.paths import (
    characterization_matrix_path,
    responsibility_periods_csv_path,
    static_cc_csv_path,
)
from pyaesa.shared.lcia.prerequisite_tables import (
    clean_characterization_matrix_frame,
    clean_responsibility_period_frame,
)
from pyaesa.shared.lcia.static_cc import read_static_cc
from pyaesa.shared.runtime.io.file_identity import table_sha256


def characterization_frame_sha256(frame: pd.DataFrame) -> str:
    """Fingerprint validated characterization factors, mappings, categories and units."""
    return table_sha256(
        frame,
        labels=("extension", "stressor", "stressor_unit", "impact", "impact_parent", "impact_unit"),
        numbers=("factor",),
    )


def characterization_sha256(*, source: str, lcia_method: str) -> str:
    """Fingerprint the active characterization values; ignore citation and comment fields."""
    path = characterization_matrix_path(source=source, lcia_method=lcia_method)
    return characterization_frame_sha256(
        clean_characterization_matrix_frame(
            frame=pd.read_csv(path, float_precision="round_trip"), path=path
        )
    )


def responsibility_period_sha256(*, source: str, lcia_method: str) -> str:
    """Fingerprint category mappings and responsibility periods, excluding annotations."""
    path = responsibility_periods_csv_path(source=source, lcia_method=lcia_method)
    return table_sha256(
        clean_responsibility_period_frame(
            frame=pd.read_csv(path, float_precision="round_trip"), path=path
        ),
        labels=("impact", "impact_parent"),
        numbers=("responsibility_period_years",),
    )


def carrying_capacity_sha256(*, lcia_method: str) -> str:
    """Fingerprint impact identities, units and capacity bounds, excluding annotations."""
    return table_sha256(
        read_static_cc(static_cc_csv_path(lcia_method=lcia_method)).reindex(
            columns=["impact", "impact_unit", "min_cc", "max_cc"]
        ),
        labels=("impact", "impact_unit"),
        numbers=("min_cc", "max_cc"),
    )


def require_matching_scientific_inputs(
    *,
    persisted: Mapping[str, Any] | None,
    current: Mapping[str, str],
    context: str,
) -> None:
    """Reject reuse when input identities are absent or differ.

    Extra persisted inputs are allowed for a request that selects a subset of
    an existing result. Missing identities cannot establish scientific reuse.
    """
    recorded = persisted or {}
    mismatches = [key for key, digest in current.items() if recorded.get(key) != digest]
    if mismatches:
        raise ValueError(
            f"Scientific inputs for {context} are changed or unverified. "
            f"Missing or different SHA256 identities: {sorted(mismatches)[:10]}. "
            "Existing results cannot be reused. Use a fresh workspace or output scope. "
            "Existing files were preserved."
        )


def processed_lcia_inputs(
    *,
    source: str,
    metadata: Mapping[str, Any],
    years: Sequence[int],
    methods: Sequence[str],
    domain: str = "original_class",
) -> dict[str, str]:
    """Validate processed factors and return selected identities for a run signature."""
    current = {
        method: characterization_sha256(source=source, lcia_method=method) for method in methods
    }
    entries = cast(Mapping[str, Mapping[str, Any]], metadata.get("years", {}))
    identities: dict[str, str] = {}
    for year in sorted(set(years)):
        entry = entries.get(str(year), {})
        require_matching_scientific_inputs(
            persisted=entry.get("characterization_inputs"),
            current=current,
            context=f"processed MRIO source={source}, year={year}, domain={domain}",
        )
        identities.update(
            {f"characterization/{domain}/{method}": digest for method, digest in current.items()}
        )
    identities.update(
        {
            f"responsibility_periods/{method}": responsibility_period_sha256(
                source=source, lcia_method=method
            )
            for method in methods
        }
    )
    return identities
