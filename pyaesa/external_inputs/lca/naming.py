"""Filename tokens for staged external LCA inputs."""

import re

_VERSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def normalize_external_lca_version_name(value: object, *, argument_name: str) -> str:
    """Return one external LCA version token used in filenames and output paths."""
    if not isinstance(value, str):
        raise ValueError(f"{argument_name} must be a non empty string.")
    text = value.strip()
    if not text:
        raise ValueError(f"{argument_name} must be a non empty string.")
    if "__" in text or _VERSION_NAME_RE.fullmatch(text) is None:
        raise ValueError(
            f"{argument_name} must be a filename token using letters, numbers, '.', '_', "
            "or '-', and must not contain '__'."
        )
    return text
