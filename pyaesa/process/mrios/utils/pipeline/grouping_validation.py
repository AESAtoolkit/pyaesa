"""Validation ownership for MRIO process orchestration."""

from pathlib import Path
from typing import Any, Optional

from pyaesa.process.mrios.utils.grouping.grouping import (
    build_agg_vector,
    unique_in_order,
)


def validate_metadata_grouping(
    *,
    metadata: dict[str, Any],
    version_tag: str,
    grouping_payload: dict[str, Any],
    group_reg: bool,
    group_sec: bool,
    group_reg_df,
    group_sec_df,
    group_reg_path: Optional[Path],
    group_sec_path: Optional[Path],
    metadata_path: Optional[Path] = None,
) -> None:
    """Validate grouping metadata consistency against requested grouping inputs."""
    existing_tag = metadata.get("version_tag")
    metadata_label = f" Metadata={metadata_path}." if metadata_path is not None else ""
    if existing_tag and existing_tag != version_tag:
        raise ValueError(
            "Processed MRIO metadata was written for a different matrix scope. "
            f"Stored version_tag={existing_tag!r}; requested version_tag={version_tag!r}."
            f"{metadata_label} Use a different group_version or refresh this processed MRIO scope."
        )
    existing_grouping = metadata.get("grouping", {})
    if existing_grouping:
        for key, value in grouping_payload.items():
            if existing_grouping.get(key) != value:
                raise ValueError(
                    "Processed MRIO grouping metadata is incompatible with the requested "
                    f"grouping files. Requested grouping={grouping_payload}; stored "
                    f"grouping={existing_grouping}.{metadata_label} Use a different "
                    "group_version or refresh this processed MRIO scope."
                )

    labels = metadata.get("labels", {})
    if (group_reg or group_sec) and not labels and metadata.get("years"):
        raise ValueError(
            "Processed MRIO grouping metadata is missing canonical region and sector labels. "
            f"{metadata_label} Refresh this processed MRIO scope before reusing grouped matrices."
        )
    if group_reg and labels.get("regions_original"):
        reg_vec = build_agg_vector(
            labels["regions_original"],
            group_reg_df,
            label_kind="region",
            csv_path=group_reg_path or "group_reg",
        )
        reg_used = unique_in_order(reg_vec)
        if labels.get("regions_used") and reg_used != labels["regions_used"]:
            raise ValueError(
                "Region grouping mapping does not match the stored processed MRIO metadata. "
                f"Mapping CSV={group_reg_path}. Stored regions_used={labels['regions_used'][:10]}; "
                f"requested regions_used={reg_used[:10]}.{metadata_label} Use a different "
                "group_version or refresh this processed MRIO scope."
            )
    if group_sec and labels.get("sectors_original"):
        sec_vec = build_agg_vector(
            labels["sectors_original"],
            group_sec_df,
            label_kind="sector",
            csv_path=group_sec_path or "group_sec",
        )
        sec_used = unique_in_order(sec_vec)
        if labels.get("sectors_used") and sec_used != labels["sectors_used"]:
            raise ValueError(
                "Sector grouping mapping does not match the stored processed MRIO metadata. "
                f"Mapping CSV={group_sec_path}. Stored sectors_used={labels['sectors_used'][:10]}; "
                f"requested sectors_used={sec_used[:10]}.{metadata_label} Use a different "
                "group_version or refresh this processed MRIO scope."
            )
