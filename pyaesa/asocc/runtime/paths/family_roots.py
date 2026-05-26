"""Family-local aSoCC namespace and allocation root ownership."""

from pathlib import Path

from pyaesa.download.mrios.utils.source_registry import list_mrio_source_keys
from pyaesa.shared.runtime.io.family_root_names import ASOCC_ROOT_DIRNAME

_NATIVE_SOURCE_KEYS = {"iso3", *(key.lower() for key in list_mrio_source_keys())}


def _is_non_native_output_source_label(*, source: str) -> bool:
    """Return whether one published aSoCC source label is not a native input source."""
    return str(source).strip().lower() not in _NATIVE_SOURCE_KEYS


def is_native_asocc_source(*, source: str) -> bool:
    """Return whether one aSoCC source label is a native package source key."""
    return not _is_non_native_output_source_label(source=source)


def effective_agg_version_for_source(*, source: str, agg_version: str | None) -> str | None:
    """Return the published aggregation token owned by one aSoCC source label."""
    if _is_non_native_output_source_label(source=source):
        return None
    version_token = str(agg_version).strip() if agg_version is not None else ""
    return version_token or None


def effective_agg_flags_for_source(
    *,
    source: str,
    agg_reg: bool | None,
    agg_sec: bool | None,
) -> tuple[bool, bool]:
    """Return the published aggregation flags owned by one aSoCC source label."""
    if _is_non_native_output_source_label(source=source):
        return False, False
    return bool(agg_reg), bool(agg_sec)


def asocc_source_version_token(*, source: str, agg_version: str | None) -> str:
    """Return the aSoCC family local source/version token."""
    source_clean = str(source).strip()
    version_token = effective_agg_version_for_source(
        source=source,
        agg_version=agg_version,
    )
    if version_token is None:
        return source_clean
    return f"{source_clean}__{version_token}"


def _get_asocc_root(*, proj_base: Path) -> Path:
    """Return the branch local aSoCC family root."""
    return Path(proj_base) / ASOCC_ROOT_DIRNAME
