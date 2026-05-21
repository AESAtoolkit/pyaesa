"""Processed MRIO metadata payload accessors used by the pipeline."""

from typing import cast


def extract_root_extension_payload(
    year_meta: dict[str, object] | None,
) -> dict[str, object] | None:
    """Return root extension payload from one year metadata entry."""
    if year_meta is None:
        return None
    return cast(dict[str, object], year_meta["extensions"])


def extract_preclip_extension_payload(
    year_meta: dict[str, object] | None,
) -> dict[str, object] | None:
    """Return preclip extension payload from one year metadata entry."""
    if year_meta is None:
        return None
    raw_preclip = year_meta.get("preclip")
    if raw_preclip is None:
        return None
    return cast(dict[str, object], cast(dict[str, object], raw_preclip)["extensions"])
