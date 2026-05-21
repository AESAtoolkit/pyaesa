"""Shared scenario token contracts for staged external input filenames."""

from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens, ssp_partition_token


def external_file_ssp_token(value: object, *, family_label: str) -> str:
    """Return one lowercase SSP token for staged external filenames."""
    try:
        return ssp_partition_token(value, context=f"{family_label} filename SSP token")
    except ValueError as exc:
        raise ValueError(f"{family_label} SSP token '{value}' is invalid.") from exc


def external_row_ssp_token(value: object, *, family_label: str) -> str:
    """Return one row SSP label from a staged external filename suffix."""
    text = str(value).strip()
    file_token = external_file_ssp_token(text, family_label=family_label)
    if text != file_token:
        raise ValueError(f"{family_label} filenames must use lowercase SSP tokens such as 'ssp2'.")
    return normalize_ssp_tokens([text])[0]
