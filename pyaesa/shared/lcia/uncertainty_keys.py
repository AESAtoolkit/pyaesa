"""LCIA uncertainty shared random variable keys."""

from pyaesa.shared.uncertainty_assessment.request.shared_u import stable_json_key


def build_lcia_shared_u_key(
    *,
    project_name: str,
    source: str,
    group_reg: bool,
    group_sec: bool,
    group_version: str | None,
    driver_kind: str,
    driver_key: str,
) -> str:
    """Return the canonical LCIA shared random variable key."""
    return stable_json_key(
        payload={
            "family": "lcia_shared_u_v1",
            "project_name": str(project_name),
            "source": str(source),
            "group_reg": bool(group_reg),
            "group_sec": bool(group_sec),
            "group_version": None if group_version is None else str(group_version),
            "driver_kind": str(driver_kind),
            "driver_key": str(driver_key),
        }
    )
