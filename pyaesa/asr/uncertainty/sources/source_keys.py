"""Canonical ASR uncertainty source keys."""

ACC_SOURCE_PREFIX = "acc::"
IO_LCA_SOURCE_PREFIX = "io_lca::"
EXTERNAL_LCA_SOURCE_PREFIX = "external_lca::"


def acc_source_name(source_name: str) -> str:
    """Return the ASR source key for one upstream aCC source."""
    return f"{ACC_SOURCE_PREFIX}{str(source_name).strip()}"


def io_lca_source_name(source_name: str) -> str:
    """Return the ASR source key for one upstream IO-LCA source."""
    return f"{IO_LCA_SOURCE_PREFIX}{str(source_name).strip()}"


def external_lca_source_name(version_name: str) -> str:
    """Return the ASR source key for one external LCA Monte Carlo version."""
    return f"{EXTERNAL_LCA_SOURCE_PREFIX}{str(version_name).strip()}"
