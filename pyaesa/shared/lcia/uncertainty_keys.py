"""LCIA uncertainty shared random variable keys."""

from pyaesa.shared.uncertainty_assessment.request.shared_u import stable_json_key


def build_lcia_shared_u_key(
    *,
    project_name: str,
    source: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
    driver_kind: str,
    driver_key: str,
) -> str:
    """Return the canonical LCIA shared random variable key."""
    return stable_json_key(
        payload={
            "family": "lcia_shared_u_v1",
            "project_name": str(project_name),
            "source": str(source),
            "agg_reg": bool(agg_reg),
            "agg_sec": bool(agg_sec),
            "agg_version": None if agg_version is None else str(agg_version),
            "driver_kind": str(driver_kind),
            "driver_key": str(driver_key),
        }
    )
