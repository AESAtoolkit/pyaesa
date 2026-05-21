"""Shared path ownership for the dynamic AR6 CC family."""

from pathlib import Path
import re

from pyaesa.process.ar6.utils.io.paths import get_processed_scope_dir

from .signatures import normalize_cc_category, normalize_cc_ssp_scenario

_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_piece(value: str) -> str:
    """Return a filesystem-safe token."""
    text = _SAFE_TOKEN_RE.sub("_", str(value).strip())
    return text.strip("._-") or "item"


def ar6_variable_tag(*, emission_type: str, include_afolu: bool, emissions_mode: str) -> str:
    """Return the compact AR6 variable folder token for one CC family."""
    emission_type_token = _sanitize_piece(emission_type)
    afolu = "with_afolu" if include_afolu else "wo_afolu"
    mode = _sanitize_piece(emissions_mode)
    return f"{mode}_{emission_type_token}_{afolu}"


def afolu_tag(*, include_afolu: bool) -> str:
    """Return the AFOLU selection tag."""
    return "with_afolu" if include_afolu else "wo_afolu"


def cc_scope_dir_name(
    *,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str = "gross_alt",
    subset_version: str | None,
) -> str:
    """Return the final AR6 CC scope directory token."""
    base = ar6_variable_tag(
        emission_type=emission_type,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode,
    )
    if subset_version is None:
        return base
    return f"{base}__sub_{_sanitize_piece(subset_version)}"


def cc_selector_dir_name(
    *,
    category: str | list[str] | None,
    ssp_scenario: str | list[str] | None,
) -> str:
    """Return the category and SSP selector folder token for one AR6 CC scope."""
    categories = "-".join(_sanitize_piece(value) for value in normalize_cc_category(category))
    ssps = "-".join(_sanitize_piece(value) for value in normalize_cc_ssp_scenario(ssp_scenario))
    return f"{categories}__{ssps}"


def get_cc_family_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
    emission_type: str = "kyoto_gases",
    include_afolu: bool = False,
    emissions_mode: str = "gross_alt",
    subset_version: str | None = None,
) -> Path:
    """Return the shared processed dynamic AR6 CC family directory for one AR6 scope."""
    ar6_dir = get_processed_scope_dir(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
    )
    return (
        ar6_dir
        / "ar6_cc"
        / cc_scope_dir_name(
            emission_type=emission_type,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode,
            subset_version=subset_version,
        )
    )
