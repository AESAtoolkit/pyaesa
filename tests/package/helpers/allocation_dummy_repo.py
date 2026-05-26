"""Shared dummy repository helpers for asocc package tests."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Literal

import pandas as pd

from pyaesa import set_workspace
from pyaesa.asocc.disaggregation.branch_context import _build_selector_request
from pyaesa.asocc.disaggregation.models import RunSelector
from pyaesa.asocc.io.metadata import _build_run_metadata, _save_run_metadata
from pyaesa.asocc.data.paths import _get_mrio_year_dir
from pyaesa.asocc.runtime.paths.deterministic import _get_allocate_run_metadata_path
from pyaesa.asocc.runtime.paths.published import (
    _get_asocc_l1_dir,
    _get_asocc_l2_dir,
    _get_enacting_metric_dir,
)
from pyaesa.asocc.orchestration.setup.run_setup import _prepare_context
from pyaesa.process.mrios.utils.io.paths import _get_agg_map_path, _get_metadata_path
from pyaesa.workspace_initialisation.workspace import clear_default_repo_root, get_default_repo_root
from pyaesa.shared.lcia.paths import responsibility_periods_csv_path


@dataclass(frozen=True)
class AllocationDummyRepo:
    """Reusable on disk fixture describing one minimal allocation repository."""

    top_path: Path
    repo_root: Path
    historical_years: tuple[int, ...] = (2005, 2006)
    future_year: int = 2030

    @property
    def all_years(self) -> list[int]:
        """Return all dummy years available across WB/SSP fixtures."""
        return [*self.historical_years, self.future_year]

    def _metadata_path(self, *, source: str, matrix_version: str | None) -> Path:
        """Return processed MRIO metadata path for one source/domain."""
        return _get_metadata_path(source, matrix_version=matrix_version)

    def _read_mrio_metadata(
        self,
        *,
        source: str,
        matrix_version: str | None,
    ) -> dict:
        """Read one processed MRIO metadata payload."""
        path = self._metadata_path(source=source, matrix_version=matrix_version)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_mrio_metadata_payload(
        self,
        *,
        source: str,
        matrix_version: str | None,
        payload: dict,
    ) -> None:
        """Persist one processed MRIO metadata payload."""
        path = self._metadata_path(source=source, matrix_version=matrix_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _regions_for(self, *, source: str, matrix_version: str | None) -> list[str]:
        """Return region labels declared in metadata for one source/domain."""
        metadata = self._read_mrio_metadata(source=source, matrix_version=matrix_version)
        labels = metadata.get("labels", {})
        regions = labels.get("regions_used")
        if not isinstance(regions, list) or not regions:
            raise ValueError("Dummy MRIO metadata must declare non-empty labels.regions_used.")
        return [str(value) for value in regions]

    def _sectors_for(self, *, source: str, matrix_version: str | None) -> list[str]:
        """Return sector labels declared in metadata for one source/domain."""
        metadata = self._read_mrio_metadata(source=source, matrix_version=matrix_version)
        labels = metadata.get("labels", {})
        sectors = labels.get("sectors_used")
        if not isinstance(sectors, list) or not sectors:
            raise ValueError("Dummy MRIO metadata must declare non-empty labels.sectors_used.")
        return [str(value) for value in sectors]

    def write_mrio_metadata(
        self,
        *,
        source: str,
        matrix_version: str | None,
        sectors_used: list[str],
        regions_used: list[str],
        years: list[int] | None = None,
    ) -> None:
        """Write one minimal processed MRIO metadata payload."""
        effective_years = (
            list(self.historical_years) if years is None else [int(year) for year in years]
        )
        payload = {
            "source": source,
            "version_tag": "original_classification"
            if matrix_version is None
            else f"custom_classification_{matrix_version}",
            "labels": {
                "sectors_used": list(sectors_used),
                "regions_used": list(regions_used),
            },
            "years": {
                str(int(year)): {
                    "core": ["A", "L"],
                    "extensions": {},
                    "enacting_metrics": {
                        "units": {
                            "mrio_default_monetary": "M EUR",
                            "mrio_by_metric": {
                                "fd_rf": "M EUR",
                                "gva_rp": "M EUR",
                                "fd_rp_sp_rf": "M EUR",
                                "fd_rp_sp": "M EUR",
                                "fd_rf_sp": "M EUR",
                                "gva_rp_sp": "M EUR",
                                "x_to_rc": "M EUR",
                            },
                            "lcia_by_method": {},
                        }
                    },
                }
                for year in effective_years
            },
        }
        self._write_mrio_metadata_payload(
            source=source,
            matrix_version=matrix_version,
            payload=payload,
        )

    def set_lcia_methods(
        self,
        *,
        source: str,
        matrix_version: str | None,
        methods: list[str],
        available_years_by_method: dict[str, list[int]] | None = None,
        units_by_method: dict[str, str] | None = None,
    ) -> None:
        """Add LCIA availability and unit metadata to processed MRIO metadata."""
        payload = self._read_mrio_metadata(source=source, matrix_version=matrix_version)
        years_payload = payload.get("years")
        if not isinstance(years_payload, dict):
            raise ValueError("Dummy MRIO metadata must contain a 'years' mapping.")
        for year_key, year_entry in years_payload.items():
            if not isinstance(year_entry, dict):
                raise ValueError(f"Year entry '{year_key}' must be a mapping.")
            year_entry.setdefault("lcia_status", {})
            lcia_status = year_entry["lcia_status"]
            if not isinstance(lcia_status, dict):
                raise ValueError(f"Year entry '{year_key}' has invalid 'lcia_status'.")
            enacting_metrics = year_entry.setdefault("enacting_metrics", {})
            if not isinstance(enacting_metrics, dict):
                raise ValueError(f"Year entry '{year_key}' has invalid 'enacting_metrics'.")
            units = enacting_metrics.setdefault("units", {})
            if not isinstance(units, dict):
                raise ValueError(
                    f"Year entry '{year_key}' has invalid enacting metric units payload."
                )
            lcia_units = units.setdefault("lcia_by_method", {})
            if not isinstance(lcia_units, dict):
                raise ValueError(f"Year entry '{year_key}' has invalid lcia_by_method payload.")
            year_value = int(year_key)
            for lcia_method in methods:
                available_years = (
                    set(int(item) for item in available_years_by_method.get(lcia_method, []))
                    if available_years_by_method is not None
                    else set(int(item) for item in years_payload.keys())
                )
                available = year_value in available_years
                lcia_status[lcia_method] = {
                    "available": available,
                    "reason": None if available else "Dummy LCIA unavailable",
                }
                lcia_units[lcia_method] = {
                    "climate_parent": str(
                        (units_by_method or {}).get(lcia_method, "kg CO2-eq / year")
                    )
                }
        self._write_mrio_metadata_payload(
            source=source,
            matrix_version=matrix_version,
            payload=payload,
        )

    def write_mrio_year_payloads(
        self,
        *,
        source: str,
        matrix_version: str | None,
        year: int,
        lcia_methods: list[str] | None = None,
        lcia_impacts_by_method: dict[str, list[str]] | None = None,
    ) -> Path:
        """Write one complete processed MRIO year directory for allocation tests."""
        regions = self._regions_for(source=source, matrix_version=matrix_version)
        sectors = self._sectors_for(source=source, matrix_version=matrix_version)
        if len(regions) < 2 or len(sectors) < 2:
            raise ValueError("Dummy MRIO payloads require at least two regions and two sectors.")
        r_1, r_2 = regions[:2]
        s_1, s_2 = sectors[:2]
        saved_dir = _get_mrio_year_dir(
            source=source,
            year=year,
            agg_version=matrix_version,
        )
        level_1 = saved_dir / "enacting_metrics" / "level_1"
        level_2 = saved_dir / "enacting_metrics" / "level_2"
        utility_dir = saved_dir / "utility_propag_uncasext"
        level_1.mkdir(parents=True, exist_ok=True)
        level_2.mkdir(parents=True, exist_ok=True)
        utility_dir.mkdir(parents=True, exist_ok=True)

        fd_rf = pd.Series(
            [float(year - 2000), float(year - 1998)],
            index=pd.Index([r_1, r_2], name="r_f"),
        )
        gva_rp = pd.Series(
            [float(year - 1999), float(year - 1997)],
            index=pd.Index([r_1, r_2], name="r_p"),
        )
        fd_rp_sp_rf = pd.DataFrame(
            [
                [1.0 + year / 1000.0, 2.0 + year / 1000.0],
                [3.0 + year / 1000.0, 4.0 + year / 1000.0],
                [5.0 + year / 1000.0, 6.0 + year / 1000.0],
                [7.0 + year / 1000.0, 8.0 + year / 1000.0],
            ],
            index=pd.MultiIndex.from_tuples(
                [(r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2)],
                names=["r_p", "s_p"],
            ),
            columns=pd.Index([r_1, r_2], name="r_f"),
        )
        fd_rp_sp = pd.Series(
            [11.0, 12.0, 13.0, 14.0],
            index=pd.MultiIndex.from_tuples(
                [(r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2)],
                names=["r_p", "s_p"],
            ),
        )
        fd_rf_sp = pd.Series(
            [9.0, 10.0, 11.0, 12.0],
            index=pd.MultiIndex.from_tuples(
                [(r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2)],
                names=["r_f", "s_p"],
            ),
        )
        gva_rp_sp = pd.Series(
            [6.0, 7.0, 8.0, 9.0],
            index=pd.MultiIndex.from_tuples(
                [(r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2)],
                names=["r_p", "s_p"],
            ),
        )
        x_to_rc = pd.DataFrame(
            [
                [2.0, 1.0],
                [3.0, 2.0],
                [4.0, 3.0],
                [5.0, 4.0],
            ],
            index=pd.MultiIndex.from_tuples(
                [(r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2)],
                names=["r_p", "s_p"],
            ),
            columns=pd.Index([r_1, r_2], name="r_c"),
        )
        kappa = pd.DataFrame(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    (r_1, r_1, s_1),
                    (r_1, r_1, s_2),
                    (r_1, r_2, s_1),
                    (r_1, r_2, s_2),
                    (r_2, r_1, s_1),
                    (r_2, r_1, s_2),
                    (r_2, r_2, s_1),
                    (r_2, r_2, s_2),
                ],
                names=["r_c", "r_p", "s_p"],
            ),
            columns=pd.Index([r_1, r_2], name="r_f"),
        )
        omega_reg = pd.DataFrame(
            [
                [0.55, 0.45, 0.40, 0.60],
                [0.45, 0.55, 0.60, 0.40],
            ],
            index=pd.Index([r_1, r_2], name="r_u"),
            columns=pd.MultiIndex.from_tuples(
                [(r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2)],
                names=["r_p", "s_p"],
            ),
        )
        fd_rf.to_pickle(level_1 / "fd_rf.pickle")
        gva_rp.to_pickle(level_1 / "gva_rp.pickle")
        fd_rp_sp_rf.to_pickle(level_2 / "fd_rp_sp_rf.pickle")
        fd_rp_sp.to_pickle(level_2 / "fd_rp_sp.pickle")
        fd_rf_sp.to_pickle(level_2 / "fd_rf_sp.pickle")
        gva_rp_sp.to_pickle(level_2 / "gva_rp_sp.pickle")
        x_to_rc.to_pickle(utility_dir / "x_to_rc.pickle")
        kappa.to_pickle(utility_dir / "kappa.pickle")
        omega_reg.to_pickle(utility_dir / "omega_reg.pickle")

        for lcia_method in lcia_methods or []:
            self._write_lcia_year_payloads(
                source=source,
                matrix_version=matrix_version,
                year=year,
                lcia_method=lcia_method,
                impacts=(lcia_impacts_by_method or {}).get(lcia_method),
            )
        return saved_dir

    def _write_lcia_year_payloads(
        self,
        *,
        source: str,
        matrix_version: str | None,
        year: int,
        lcia_method: str,
        impacts: list[str] | None = None,
    ) -> None:
        """Write one LCIA payload set for one MRIO year."""
        regions = self._regions_for(source=source, matrix_version=matrix_version)
        sectors = self._sectors_for(source=source, matrix_version=matrix_version)
        r_1, r_2 = regions[:2]
        s_1, s_2 = sectors[:2]
        saved_dir = _get_mrio_year_dir(
            source=source,
            year=year,
            agg_version=matrix_version,
        )
        impact_values = list(impacts or ["climate_child"])
        level_1 = saved_dir / "enacting_metrics" / "level_1" / lcia_method
        level_2 = saved_dir / "enacting_metrics" / "level_2" / lcia_method
        level_1.mkdir(parents=True, exist_ok=True)
        level_2.mkdir(parents=True, exist_ok=True)
        impact_index = pd.Index(impact_values, name="impact")

        e_cba_fd_reg = pd.DataFrame(
            [
                [
                    float(index + 1) * (10.0 + year / 1000.0),
                    float(index + 1) * (20.0 + year / 1000.0),
                ]
                for index, _impact in enumerate(impact_values)
            ],
            index=impact_index,
            columns=pd.Index([r_1, r_2], name="r_f"),
        )
        e_pba_reg = pd.DataFrame(
            [
                [
                    float(index + 1) * (6.0 + year / 1000.0),
                    float(index + 1) * (8.0 + year / 1000.0),
                ]
                for index, _impact in enumerate(impact_values)
            ],
            index=impact_index,
            columns=pd.Index([r_1, r_2], name="r_p"),
        )
        e_rp_sp = pd.DataFrame(
            [
                [float(index + 1) * value for value in (3.0, 4.0, 5.0, 6.0)]
                for index, _impact in enumerate(impact_values)
            ],
            index=impact_index,
            columns=pd.MultiIndex.from_tuples(
                [(r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2)],
                names=["r_p", "s_p"],
            ),
        )
        e_rf_sp = pd.DataFrame(
            [
                [float(index + 1) * value for value in (2.0, 3.0, 4.0, 5.0)]
                for index, _impact in enumerate(impact_values)
            ],
            index=impact_index,
            columns=pd.MultiIndex.from_tuples(
                [(r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2)],
                names=["r_f", "s_p"],
            ),
        )
        e_rc_sp = pd.DataFrame(
            [
                [float(index + 1) * value for value in (2.5, 3.5, 4.5, 5.5)]
                for index, _impact in enumerate(impact_values)
            ],
            index=impact_index,
            columns=pd.MultiIndex.from_tuples(
                [(r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2)],
                names=["r_c", "s_p"],
            ),
        )
        e_rp_sp_rf = pd.DataFrame(
            [
                [float(index + 1) * first, float(index + 1) * second]
                for index, _impact in enumerate(impact_values)
                for first, second in ((1.0, 1.5), (2.0, 2.5), (3.0, 3.5), (4.0, 4.5))
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    (impact, region, sector)
                    for impact in impact_values
                    for region, sector in ((r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2))
                ],
                names=["impact", "r_p", "s_p"],
            ),
            columns=pd.Index([r_1, r_2], name="r_f"),
        )
        e_rp_sp_rc = pd.DataFrame(
            [
                [float(index + 1) * first, float(index + 1) * second]
                for index, _impact in enumerate(impact_values)
                for first, second in ((1.1, 1.6), (2.1, 2.6), (3.1, 3.6), (4.1, 4.6))
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    (impact, region, sector)
                    for impact in impact_values
                    for region, sector in ((r_1, s_1), (r_1, s_2), (r_2, s_1), (r_2, s_2))
                ],
                names=["impact", "r_p", "s_p"],
            ),
            columns=pd.Index([r_1, r_2], name="r_c"),
        )
        e_cba_fd_reg.to_pickle(level_1 / "e_cba_fd_reg.pickle")
        e_pba_reg.to_pickle(level_1 / "e_pba_reg.pickle")
        e_rp_sp.to_pickle(level_2 / "e_pba_rp_sp.pickle")
        e_rp_sp.to_pickle(level_2 / "e_cba_fd_rp_sp.pickle")
        e_rp_sp.to_pickle(level_2 / "e_cba_td_rp_sp.pickle")
        e_rp_sp_rf.to_pickle(level_2 / "e_cba_fd_rp_sp_rf.pickle")
        e_rp_sp_rc.to_pickle(level_2 / "e_cba_td_rp_sp_rc.pickle")
        e_rf_sp.to_pickle(level_2 / "e_cba_fd_rf_sp.pickle")
        e_rc_sp.to_pickle(level_2 / "e_cba_td_rc_sp.pickle")

    def write_lcia_support(
        self,
        *,
        source: str,
        matrix_version: str | None,
        lcia_method: str,
        available_years: list[int] | None = None,
        impacts: list[str] | None = None,
        impact_parents: dict[str, str] | None = None,
    ) -> None:
        """Write LCIA metadata, RPS CSV, and MRIO payloads for one method."""
        effective_years = (
            list(self.historical_years)
            if available_years is None
            else [int(year) for year in available_years]
        )
        effective_impacts = list(impacts or ["climate_child"])
        parent_by_impact = {
            impact: (impact_parents or {}).get(impact, "climate_parent")
            for impact in effective_impacts
        }
        self.set_lcia_methods(
            source=source,
            matrix_version=matrix_version,
            methods=[lcia_method],
            available_years_by_method={lcia_method: effective_years},
        )
        metadata = self._read_mrio_metadata(source=source, matrix_version=matrix_version)
        for year_entry in metadata["years"].values():
            lcia_units = year_entry["enacting_metrics"]["units"]["lcia_by_method"]
            lcia_units[lcia_method] = {
                parent: "kg CO2-eq / year" for parent in sorted(set(parent_by_impact.values()))
            }
        self._write_mrio_metadata_payload(
            source=source,
            matrix_version=matrix_version,
            payload=metadata,
        )
        rps_path = responsibility_periods_csv_path(source=source, lcia_method=lcia_method)
        rps_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "impact": impact,
                    "impact_parent": parent_by_impact[impact],
                    "responsibility_period_years": 2,
                }
                for impact in effective_impacts
            ]
        ).to_csv(rps_path, index=False)
        for year in effective_years:
            self.write_mrio_year_payloads(
                source=source,
                matrix_version=matrix_version,
                year=year,
            )
            self._write_lcia_year_payloads(
                source=source,
                matrix_version=matrix_version,
                year=year,
                lcia_method=lcia_method,
                impacts=effective_impacts,
            )

    def write_mrio_history(
        self,
        *,
        source: str,
        matrix_version: str | None,
        years: list[int] | None = None,
        lcia_methods: list[str] | None = None,
    ) -> None:
        """Write all requested historical processed MRIO year payloads."""
        for year in list(self.historical_years) if years is None else [int(y) for y in years]:
            self.write_mrio_year_payloads(
                source=source,
                matrix_version=matrix_version,
                year=year,
                lcia_methods=lcia_methods,
            )

    def set_processed_pop_gdp_years(
        self,
        *,
        historical_years: list[int],
        scenario_years: list[int] | None = None,
    ) -> None:
        """Rewrite processed WB/SSP fixtures over explicit year coverage."""
        _write_processed_pop_gdp_with_years(
            repo_root=self.repo_root,
            historical_years=[int(year) for year in historical_years],
            scenario_years=(
                [self.future_year]
                if scenario_years is None
                else [int(year) for year in scenario_years]
            ),
        )

    def write_agg_map(
        self,
        *,
        source: str,
        kind: Literal["reg", "sec"],
        agg_version: str,
        mapping: dict[str, str],
    ) -> Path:
        """Write one minimal aggregation map CSV and return its path."""
        path = _get_agg_map_path(
            source,
            kind=kind,
            agg_version=agg_version,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"original_classification": original, "aggregated_mrio": aggregated}
                for original, aggregated in mapping.items()
            ]
        ).to_csv(path, index=False)
        return path

    def write_scope_metadata(
        self,
        *,
        context,
        completed_years: list[int],
        outputs: list[Path],
    ) -> Path:
        """Write compute_asocc run metadata for one already resolved scope."""
        output_source = context.output_source_label or context.source
        meta_path = _get_allocate_run_metadata_path(
            context.proj_base,
            source=output_source,
            agg_version=context.agg_version,
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        scope_payload = _build_run_metadata(
            requested_years=list(context.requested_years),
            resolved_years=[int(year) for year in completed_years],
            selected_methods=dict(context.selected_methods),
            fu_code=str(context.fu_code),
            studied_indices_tag=str(context.studied_indices_tag),
            skipped_years={},
            outputs=[str(path) for path in outputs],
            signature=context.run_signature,
        )
        scope_payload["execution"]["completed_years"] = [int(year) for year in completed_years]
        scope_payload["provenance"]["ssp_scenarios"] = list(context.ssp_scenario_options)
        _save_run_metadata(meta_path, scope_payload)
        return meta_path

    def write_l2_table(
        self,
        *,
        proj_base: Path,
        source_label: str,
        bucket: str,
        method_name: str,
        frame: pd.DataFrame,
        output_format: str = "csv",
    ) -> Path:
        """Write one allocation L2 table in the expected bucket path."""
        base = _get_asocc_l2_dir(
            proj_base=proj_base,
            source=source_label,
            agg_version=None,
            bucket=bucket,
            lcia_sub=None,
        )
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{method_name}.{output_format}"
        if output_format == "csv":
            frame.to_csv(path, index=False)
        elif output_format == "pickle":
            frame.to_pickle(path)
        else:
            raise ValueError(f"Unsupported dummy output format '{output_format}'.")
        return path

    def write_l1_output(
        self,
        *,
        proj_base: Path,
        source_label: str,
        method_name: str,
        content: str = "x",
    ) -> Path:
        """Write one dummy L1 output file."""
        path = (
            _get_asocc_l1_dir(
                proj_base=proj_base,
                source=source_label,
                agg_version=None,
                lcia_sub=None,
            )
            / f"l1_{method_name}.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_enacting_metric_file(
        self,
        *,
        proj_base: Path,
        source_label: str,
        level: str,
        stem: str,
        content: str = "x",
    ) -> Path:
        """Write one dummy enacting metric file."""
        path = (
            _get_enacting_metric_dir(
                proj_base=proj_base,
                source=source_label,
                agg_version=None,
                level=level,
                lcia_sub=None,
            )
            / f"{stem}.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def prepare_selector_context(
        self,
        *,
        selector: RunSelector,
        base_allocate_args: dict,
        l1_methods: list[str] | None,
        combined_methods: list[tuple[str, str]] | None,
        one_step_methods: list[str] | None,
        l1_reg_aggreg: str,
        output_summed: bool,
        variant_tag: str,
        output_source_label: str | None = None,
    ):
        """Resolve one selector request to its allocation context."""
        request = _build_selector_request(
            selector=selector,
            base_allocate_args=base_allocate_args,
            l1_methods=l1_methods,
            combined_methods=combined_methods,
            one_step_methods=one_step_methods,
            l1_reg_aggreg=l1_reg_aggreg,
            group_indices=output_summed,
            variant_tag=variant_tag,
            output_source_label=output_source_label,
        )
        context, _, _ = _prepare_context(request=request)
        return context


def _write_processed_pop_gdp(repo_root: Path) -> None:
    """Write one small processed WB/SSP fixture used by allocation tests."""
    _write_processed_pop_gdp_with_years(
        repo_root=repo_root,
        historical_years=[2005, 2006],
        scenario_years=[2030],
    )


def _write_processed_pop_gdp_with_years(
    *,
    repo_root: Path,
    historical_years: list[int],
    scenario_years: list[int],
) -> None:
    """Write one processed WB/SSP fixture over explicit year ranges."""
    processed_dir = repo_root / "data_processed" / "pop_gdp"
    processed_dir.mkdir(parents=True, exist_ok=True)
    historical_years = sorted({int(year) for year in historical_years})
    scenario_years = sorted({int(year) for year in scenario_years})
    wb_rows: list[dict[str, object]] = []
    ssp_rows: list[dict[str, object]] = []
    base_rows = [
        ("France", "FRA", "FR", "FR", "Population", "Persons", 10.0),
        ("United States", "USA", "US", "US", "Population", "Persons", 20.0),
        ("France", "FRA", "FR", "FR", "GDP|PPP", "USD_2021/yr", 100.0),
        ("United States", "USA", "US", "US", "GDP|PPP", "USD_2021/yr", 200.0),
    ]
    for full_name, iso3, oecd, exio, variable, unit, base_value in base_rows:
        wb_row: dict[str, object] = {
            "wb_full_name": full_name,
            "iso3_code": iso3,
            "oecd_code": oecd,
            "exio_code": exio,
            "variable": variable,
            "unit": unit,
        }
        for year in historical_years:
            wb_row[str(year)] = base_value + float(year - historical_years[0])
        wb_rows.append(wb_row)

        ssp_row: dict[str, object] = {
            "ssp_scenario": "SSP2",
            "iso3_code": iso3,
            "oecd_code": oecd,
            "exio_code": exio,
            "variable": variable,
            "unit": unit,
        }
        for year in scenario_years:
            ssp_row[str(year)] = base_value + float(year - historical_years[0])
        ssp_rows.append(ssp_row)

    wb = pd.DataFrame(wb_rows)
    ssp = pd.DataFrame(ssp_rows)
    wb.to_csv(processed_dir / "wb_processed.csv", index=False)
    ssp.to_csv(processed_dir / "ssp_processed.csv", index=False)


def build_allocation_dummy_repo(top_path: Path) -> AllocationDummyRepo:
    """Create one reusable minimal repository for asocc tests."""
    set_workspace(top_path, refresh=True)
    repo_root = get_default_repo_root()
    _write_processed_pop_gdp(repo_root)
    repo = AllocationDummyRepo(top_path=Path(top_path), repo_root=repo_root)
    exio_baseline_years = list(range(1995, 2007))
    repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
    )
    repo.write_mrio_metadata(
        source="exiobase_396_ixi",
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=exio_baseline_years,
    )
    repo.write_agg_map(
        source="oecd_v2025",
        kind="reg",
        agg_version="demo_reg",
        mapping={"FR": "EU", "US": "NAM"},
    )
    repo.write_agg_map(
        source="exiobase_396_ixi",
        kind="reg",
        agg_version="demo_reg",
        mapping={"FR": "EU", "US": "NAM"},
    )
    repo.write_mrio_history(source="oecd_v2025", matrix_version=None)
    repo.write_mrio_history(
        source="exiobase_396_ixi",
        matrix_version=None,
        years=exio_baseline_years,
    )
    repo.write_lcia_support(
        source="oecd_v2025",
        matrix_version=None,
        lcia_method="gwp100_lcia",
    )
    repo.write_lcia_support(
        source="exiobase_396_ixi",
        matrix_version=None,
        lcia_method="gwp100_lcia",
        available_years=exio_baseline_years,
    )
    return repo


def clone_allocation_dummy_repo(
    template_repo: AllocationDummyRepo,
    *,
    top_path: Path,
) -> AllocationDummyRepo:
    """Clone one prepared allocation dummy repo into a fresh active workspace."""
    target_top_path = Path(top_path)
    target_repo_root = target_top_path / "pyaesa"
    if target_repo_root.exists():
        shutil.rmtree(target_repo_root)
    shutil.copytree(template_repo.repo_root, target_repo_root)
    clear_default_repo_root()
    set_workspace(target_top_path, refresh=False)
    return AllocationDummyRepo(
        top_path=target_top_path,
        repo_root=get_default_repo_root(),
        historical_years=template_repo.historical_years,
        future_year=template_repo.future_year,
    )
