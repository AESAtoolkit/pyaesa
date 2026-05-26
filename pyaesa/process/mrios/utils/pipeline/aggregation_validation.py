"""Validation ownership for MRIO process orchestration."""

from pathlib import Path
from typing import Any, Optional

from pyaesa.process.mrios.utils.aggregation.aggregation import (
    build_aggregation_spec,
)


def validate_metadata_aggregation(
    *,
    metadata: dict[str, Any],
    version_tag: str,
    aggregation_payload: dict[str, Any],
    agg_reg: bool,
    agg_sec: bool,
    agg_reg_df,
    agg_sec_df,
    agg_reg_path: Optional[Path],
    agg_sec_path: Optional[Path],
    metadata_path: Optional[Path] = None,
) -> None:
    """Validate aggregation metadata consistency against requested aggregation inputs."""
    existing_tag = metadata.get("version_tag")
    metadata_label = f" Metadata={metadata_path}." if metadata_path is not None else ""
    if existing_tag and existing_tag != version_tag:
        raise ValueError(
            "Processed MRIO metadata was written for a different matrix scope. "
            f"Stored version_tag={existing_tag!r}; requested version_tag={version_tag!r}."
            f"{metadata_label} Use a different agg_version or refresh this processed MRIO scope."
        )
    existing_aggregation = metadata.get("aggregation", {})
    if existing_aggregation:
        for key, value in aggregation_payload.items():
            if existing_aggregation.get(key) != value:
                raise ValueError(
                    "Processed MRIO aggregation metadata is incompatible with the requested "
                    "MRIO aggregation and disaggregation files. "
                    f"Requested aggregation={aggregation_payload}; stored "
                    f"aggregation={existing_aggregation}.{metadata_label} Use a different "
                    "agg_version or refresh this processed MRIO scope."
                )

    labels = metadata.get("labels", {})
    if (agg_reg or agg_sec) and not labels and metadata.get("years"):
        raise ValueError(
            "Processed MRIO aggregation metadata is missing canonical region and sector labels. "
            f"{metadata_label} Refresh this processed MRIO scope before "
            "reusing aggregated matrices."
        )
    if agg_reg and labels.get("regions_original"):
        reg_spec = build_aggregation_spec(
            labels["regions_original"],
            agg_reg_df,
            label_kind="region",
            csv_path=agg_reg_path or "agg_reg",
        )
        reg_used = list(reg_spec.aggregated_labels)
        if labels.get("regions_used") and reg_used != labels["regions_used"]:
            raise ValueError(
                "Region aggregation mapping does not match the stored processed MRIO metadata. "
                f"Mapping CSV={agg_reg_path}. Stored regions_used={labels['regions_used'][:10]}; "
                f"requested regions_used={reg_used[:10]}.{metadata_label} Use a different "
                "agg_version or refresh this processed MRIO scope."
            )
    if agg_sec and labels.get("sectors_original"):
        sec_spec = build_aggregation_spec(
            labels["sectors_original"],
            agg_sec_df,
            label_kind="sector",
            csv_path=agg_sec_path or "agg_sec",
        )
        sec_used = list(sec_spec.aggregated_labels)
        if labels.get("sectors_used") and sec_used != labels["sectors_used"]:
            raise ValueError(
                "Sector MRIO aggregation and disaggregation mapping does not match the "
                "stored processed MRIO metadata. "
                f"Mapping CSV={agg_sec_path}. Stored sectors_used={labels['sectors_used'][:10]}; "
                f"requested sectors_used={sec_used[:10]}.{metadata_label} Use a different "
                "agg_version or refresh this processed MRIO scope."
            )
