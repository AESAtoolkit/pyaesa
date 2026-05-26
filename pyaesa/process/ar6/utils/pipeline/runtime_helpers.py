"""Small runtime ownership for the public AR6 processing entrypoint."""

from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.download.ar6.utils.config import (
    DEFAULT_DATABASE,
    DEFAULT_SSPS,
    DEFAULT_VARIABLES_OUTPUT,
    DEFAULT_VARIABLES_RELEVANT,
    normalize_ar6_categories,
)

DEFAULT_HARMONIZATION_METHOD = "offset"
HARMONIZATION_METHOD_OPTIONS = ("offset",)


def validate_harmonization_method(*, harmonization: bool, harmonization_method: str) -> str:
    """Validate and normalize the harmonization method selector."""
    if not harmonization:
        return DEFAULT_HARMONIZATION_METHOD
    normalized_harmonization_method = str(harmonization_method).strip()
    if normalized_harmonization_method not in HARMONIZATION_METHOD_OPTIONS:
        raise ValueError(
            f"'harmonization_method' must be one of: {', '.join(HARMONIZATION_METHOD_OPTIONS)}."
        )
    return normalized_harmonization_method


def process_signature(
    study_period: list[int],
    harmonization: bool,
    harmonization_method: str,
    category: str | list[str] | None = None,
) -> dict[str, object]:
    """Return the persisted process signature for one AR6 run."""
    signature = {
        "database": DEFAULT_DATABASE,
        "categories": normalize_ar6_categories(category),
        "ssps": [int(value) for value in DEFAULT_SSPS],
        "variables_relevant": list(DEFAULT_VARIABLES_RELEVANT),
        "variables_output": list(DEFAULT_VARIABLES_OUTPUT),
        "processing_contract": "ar6_v2_net_harmonized_gross_companions",
        "study_period": [int(study_period[0]), int(study_period[1])],
        "harmonization": bool(harmonization),
    }
    if harmonization:
        signature["harmonization_method"] = str(harmonization_method)
    return signature


def show_stage(status: StatusSink | None, message: str) -> None:
    """Render one replaceable process stage line when work is active."""
    if status is not None:
        status.show(f"[process_ar6] {message}")
