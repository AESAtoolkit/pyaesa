"""Shared dummy repository preparation helpers for ASR package tests."""

import json

import pandas as pd

from pyaesa import prepare_external_inputs
from pyaesa.external_inputs.lca.paths import external_lca_deterministic_dir
from pyaesa.process.mrios.utils.io.paths import _get_metadata_path
from pyaesa.shared.lcia.paths import responsibility_periods_csv_path, static_cc_csv_path
from tests.package.helpers.ar6_dummy_repo import build_ar6_dummy_repo


def prepare_static_asr_io_lca_repo(
    allocation_dummy_repo,
    *,
    source: str,
    lcia_method: str,
    impact_parent: str,
    impact_unit: str,
) -> None:
    """Align one allocation dummy repository to the deterministic ASR IO-LCA contract."""
    metadata_path = _get_metadata_path(source, matrix_version=None)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    years_payload = payload.get("years")
    if not isinstance(years_payload, dict):
        raise ValueError("Dummy MRIO metadata must expose a 'years' mapping for ASR tests.")
    for year_entry in years_payload.values():
        if not isinstance(year_entry, dict):
            raise ValueError("Dummy MRIO metadata year entries must be mappings.")
        units_payload = year_entry.get("enacting_metrics", {}).get("units", {})
        lcia_units = units_payload.get("lcia_by_method")
        if not isinstance(lcia_units, dict):
            raise ValueError("Dummy MRIO metadata must expose 'lcia_by_method' for ASR tests.")
        lcia_units[lcia_method] = {impact_parent: impact_unit}
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "impact": "climate_child",
                "impact_parent": impact_parent,
                "responsibility_period_years": 2,
            }
        ]
    ).to_csv(
        responsibility_periods_csv_path(source=source, lcia_method=lcia_method),
        index=False,
    )


def prepare_static_asr_pb_lcia_repo(
    allocation_dummy_repo,
    *,
    source: str,
    years: list[int] | None = None,
    impacts: list[str] | None = None,
) -> None:
    """Stage full PB LCIA IO-LCA support with units matching static CC inputs."""
    cc_path = static_cc_csv_path(lcia_method="pb_lcia")
    cc_frame = pd.read_csv(cc_path)
    if impacts is not None:
        cc_frame = cc_frame.loc[cc_frame["impact"].astype(str).isin(impacts)].reset_index(drop=True)
        cc_frame.to_csv(cc_path, index=False)
    impact_units = dict(
        zip(
            cc_frame["impact"].astype(str),
            cc_frame["impact_unit"].astype(str),
            strict=True,
        )
    )
    allocation_dummy_repo.write_lcia_support(
        source=source,
        matrix_version=None,
        lcia_method="pb_lcia",
        available_years=years,
        impacts=list(impact_units),
        impact_parents={impact: impact for impact in impact_units},
    )
    metadata_path = _get_metadata_path(source, matrix_version=None)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    for year_entry in payload["years"].values():
        year_entry["enacting_metrics"]["units"]["lcia_by_method"]["pb_lcia"] = impact_units
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_dynamic_asr_io_lca_repo(
    allocation_dummy_repo,
    *,
    source: str,
    lcia_method: str,
    impact_parent: str,
    impact_unit: str,
    historical_years: list[int],
    scenario_years: list[int],
) -> None:
    """Extend one allocation dummy repository for dynamic ASR with IO-LCA."""
    build_ar6_dummy_repo(allocation_dummy_repo.repo_root)
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=historical_years,
        scenario_years=scenario_years,
    )
    allocation_dummy_repo.write_mrio_metadata(
        source=source,
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=historical_years,
    )
    allocation_dummy_repo.write_mrio_history(
        source=source,
        matrix_version=None,
        years=historical_years,
    )
    allocation_dummy_repo.write_lcia_support(
        source=source,
        matrix_version=None,
        lcia_method=lcia_method,
        available_years=historical_years,
    )
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source=source,
        lcia_method=lcia_method,
        impact_parent=impact_parent,
        impact_unit=impact_unit,
    )


def prepare_static_asr_external_lca_repo(
    allocation_dummy_repo,
    *,
    project_name: str,
    source: str,
    lcia_method: str,
    impact: str,
    impact_unit: str,
    include_deterministic: bool = False,
    version_name: str = "supplier_v1",
    years: list[int] | None = None,
) -> None:
    """Stage project scoped external LCA inputs for ASR tests."""
    del allocation_dummy_repo
    del source
    report = prepare_external_inputs(project_name=project_name)
    requested_years = [2005] if years is None else [int(year) for year in years]
    base_rows = [
        {
            "r_p": "FR",
            "s_p": "D",
            "impact": impact,
            "impact_unit": impact_unit,
            **{str(year): 1.0 + index * 0.1 for index, year in enumerate(requested_years)},
        },
        {
            "r_p": "FR",
            "s_p": "X",
            "impact": impact,
            "impact_unit": impact_unit,
            **{str(year): 2.0 + index * 0.1 for index, year in enumerate(requested_years)},
        },
        {
            "r_p": "US",
            "s_p": "D",
            "impact": impact,
            "impact_unit": impact_unit,
            **{str(year): 3.0 + index * 0.1 for index, year in enumerate(requested_years)},
        },
        {
            "r_p": "US",
            "s_p": "X",
            "impact": impact,
            "impact_unit": impact_unit,
            **{str(year): 4.0 + index * 0.1 for index, year in enumerate(requested_years)},
        },
    ]
    if include_deterministic:
        cc_path = static_cc_csv_path(lcia_method=lcia_method)
        if not cc_path.exists():
            cc_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "impact_full_name": [impact],
                    "impact": [impact],
                    "impact_unit": [impact_unit],
                    "min_cc": [100.0],
                    "max_cc": [150.0],
                }
            ).to_csv(cc_path, index=False)
        deterministic_path = (
            external_lca_deterministic_dir(project_base=report.project_root)
            / f"{version_name}__{lcia_method}.csv"
        )
        deterministic_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(base_rows).to_csv(deterministic_path, index=False)
