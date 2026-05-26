"""Shared dummy processed MRIO repository for IO-LCA package tests."""

from dataclasses import dataclass
from collections.abc import Callable
import json
from pathlib import Path

import pandas as pd

from pyaesa import set_workspace
from pyaesa.process.mrios.utils.io.paths import _get_metadata_path, _get_year_saved_dir
from pyaesa.workspace_initialisation.workspace import get_default_repo_root
from pyaesa.shared.lcia.paths import responsibility_periods_csv_path

_PB_LCIA_IMPACTS = ["AAL", "BI FD GHG"]
_PB_LCIA_PARENT_BY_IMPACT = {
    "AAL": "AAL",
    "BI FD GHG": "BI FD",
}


@dataclass(frozen=True)
class IOLCADummyRepo:
    """Shared on disk fixture describing one minimal IO-LCA ready repository."""

    top_path: Path
    repo_root: Path
    source: str = "exiobase_396_ixi"
    lcia_method: str = "pb_lcia"
    available_year: int = 2019
    unavailable_year: int = 2020
    available_years: tuple[int, ...] = (2019,)
    unavailable_years: tuple[int, ...] = (2020,)
    impacts: tuple[str, ...] = tuple(_PB_LCIA_IMPACTS)

    @property
    def years(self) -> list[int]:
        return [*self.available_years, *self.unavailable_years]


def _write_processed_pop_gdp(repo_root: Path) -> None:
    processed_dir = repo_root / "data_processed" / "pop_gdp"
    processed_dir.mkdir(parents=True, exist_ok=True)
    wb = pd.DataFrame(
        {
            "variable": ["population", "population"],
            "exio_code": ["FR", "DE"],
            "2019": [1.0, 1.0],
            "2020": [1.0, 1.0],
        }
    )
    ssp = pd.DataFrame(
        {
            "scenario": ["SSP2", "SSP2"],
            "variable": ["gdp", "gdp"],
            "exio_code": ["FR", "DE"],
            "2019": [1.0, 1.0],
            "2020": [1.0, 1.0],
        }
    )
    wb.to_csv(processed_dir / "wb_processed.csv", index=False)
    ssp.to_csv(processed_dir / "ssp_processed.csv", index=False)


def _write_lcia_rps(
    *,
    source: str,
    lcia_method: str,
    impacts: list[str],
    parent_by_impact: dict[str, str],
) -> None:
    rps_path = responsibility_periods_csv_path(source=source, lcia_method=lcia_method)
    rps_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Planetary boundary": ["boundary"] * len(impacts),
            "Control variable": ["control"] * len(impacts),
            "impact": impacts,
            "impact_parent": [parent_by_impact[impact] for impact in impacts],
            "impact_duration_years": [1] * len(impacts),
            "responsibility_period_years": [1] * len(impacts),
            "source": ["dummy"] * len(impacts),
            "comment": [""] * len(impacts),
        }
    ).to_csv(rps_path, index=False)


def _write_processed_mrio_metadata(
    source: str,
    lcia_method: str,
    *,
    parent_by_impact: dict[str, str],
    available_years: list[int],
    unavailable_years: list[int],
) -> None:
    years: dict[str, dict[str, object]] = {}
    for year in [*available_years, *unavailable_years]:
        available = int(year) in set(available_years)
        years[str(int(year))] = {
            "core": ["A", "L"],
            "extensions": {
                lcia_method: {
                    "available": available,
                    **({} if available else {"reason": "extension missing"}),
                }
            },
            "lcia_status": {
                lcia_method: {
                    "available": available,
                    **({} if available else {"reason": "extension missing"}),
                }
            },
            "enacting_metrics": {
                "units": {
                    "mrio_default_monetary": "EUR",
                    "mrio_by_metric": {
                        "fd_rp_sp_rf": "EUR",
                        "x_to_rc": "EUR",
                    },
                    "lcia_by_method": {
                        lcia_method: {
                            parent: "kg" for parent in sorted(set(parent_by_impact.values()))
                        }
                    },
                }
            },
        }
    metadata = {
        "source": source,
        "version_tag": "original_classification",
        "labels": {"sectors_used": ["A", "B"]},
        "years": years,
    }
    metadata_path = _get_metadata_path(source, matrix_version=None)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_year_payloads(source: str, lcia_method: str, year: int, *, impacts: list[str]) -> None:
    saved_dir = _get_year_saved_dir(source, year, matrix_version=None)
    (saved_dir / "enacting_metrics" / "level_1" / lcia_method).mkdir(parents=True, exist_ok=True)
    (saved_dir / "enacting_metrics" / "level_2" / lcia_method).mkdir(parents=True, exist_ok=True)
    (saved_dir / "enacting_metrics" / "level_2").mkdir(parents=True, exist_ok=True)
    (saved_dir / "extensions" / lcia_method).mkdir(parents=True, exist_ok=True)
    (saved_dir / "utility_propag_uncasext").mkdir(parents=True, exist_ok=True)

    products = pd.MultiIndex.from_tuples([("FR", "A"), ("DE", "B")], names=["r_p", "s_p"])
    impact_index = pd.Index(impacts, name="impact")

    pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=products, columns=products).to_pickle(
        saved_dir / "A.pickle"
    )
    pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=products, columns=products).to_pickle(
        saved_dir / "L.pickle"
    )
    extension_values = _rows_with_defaults(
        defaults=[[1.0, 2.0], [0.5, 1.5]],
        row_count=len(impacts),
        fallback=lambda index: [1.0 + float(index), 2.0 + float(index)],
    )
    pd.DataFrame(extension_values, index=impact_index, columns=products).to_pickle(
        saved_dir / "extensions" / lcia_method / "S.pickle"
    )
    pd.DataFrame(
        [[10.0, 0.0], [0.0, 20.0]],
        index=products,
        columns=pd.Index(["FR", "DE"], name="r_f"),
    ).to_pickle(saved_dir / "enacting_metrics" / "level_2" / "fd_rp_sp_rf.pickle")
    pd.DataFrame(
        [[10.0], [20.0]],
        index=products,
        columns=pd.Index(["RC"], name="r_c"),
    ).to_pickle(saved_dir / "utility_propag_uncasext" / "x_to_rc.pickle")
    fy_values = _rows_with_defaults(
        defaults=[[1.0, 3.0], [4.0, 2.0]],
        row_count=len(impacts),
        fallback=lambda index: [1.0 + float(index), 3.0 + float(index)],
    )
    pd.DataFrame(
        fy_values,
        index=impact_index,
        columns=pd.Index(["FR", "DE"], name="region"),
    ).to_pickle(saved_dir / "enacting_metrics" / "level_1" / lcia_method / "F_Y.pickle")
    l1_values = [
        [(s_fr * 10.0) + fy_fr, (s_de * 20.0) + fy_de]
        for (s_fr, s_de), (fy_fr, fy_de) in zip(extension_values, fy_values, strict=True)
    ]
    pd.DataFrame(
        l1_values,
        index=impact_index,
        columns=pd.Index(["FR", "DE"], name="r_f"),
    ).to_pickle(saved_dir / "enacting_metrics" / "level_1" / lcia_method / "e_cba_fd_reg.pickle")
    pd.DataFrame(
        l1_values,
        index=impact_index,
        columns=pd.Index(["FR", "DE"], name="r_p"),
    ).to_pickle(saved_dir / "enacting_metrics" / "level_1" / lcia_method / "e_pba_reg.pickle")
    l2_columns = pd.MultiIndex.from_tuples(
        [("FR", "A"), ("FR", "B"), ("DE", "A"), ("DE", "B")],
        names=["r_c", "s_p"],
    )
    pd.DataFrame(
        [
            [
                5.0 + float(index),
                7.0 + float(index),
                11.0 + float(index),
                13.0 + float(index),
            ]
            for index, _impact in enumerate(impacts)
        ],
        index=impact_index,
        columns=l2_columns,
    ).to_pickle(saved_dir / "enacting_metrics" / "level_2" / lcia_method / "e_cba_td_rc_sp.pickle")


def _rows_with_defaults(
    *,
    defaults: list[list[float]],
    row_count: int,
    fallback: Callable[[int], list[float]],
) -> list[list[float]]:
    """Return deterministic matrix rows while preserving historic fixture values."""
    rows: list[list[float]] = []
    for index in range(row_count):
        if index < len(defaults):
            rows.append(defaults[index])
        else:
            rows.append(fallback(index))
    return rows


def build_io_lca_dummy_repo(
    top_path: Path,
    *,
    impacts: list[str] | None = None,
    parent_by_impact: dict[str, str] | None = None,
    available_years: list[int] | None = None,
    unavailable_years: list[int] | None = None,
) -> IOLCADummyRepo:
    """Create one minimal workspace repository that supports real IO-LCA runs."""

    set_workspace(top_path, refresh=True)
    repo_root = get_default_repo_root()
    source = "exiobase_396_ixi"
    lcia_method = "pb_lcia"
    effective_impacts = list(impacts or _PB_LCIA_IMPACTS)
    effective_parent_by_impact = {
        impact: (parent_by_impact or _PB_LCIA_PARENT_BY_IMPACT).get(impact, impact)
        for impact in effective_impacts
    }
    effective_available_years = (
        [2019] if available_years is None else [int(year) for year in available_years]
    )
    effective_unavailable_years = (
        [2020] if unavailable_years is None else [int(year) for year in unavailable_years]
    )
    _write_processed_pop_gdp(repo_root)
    _write_lcia_rps(
        source=source,
        lcia_method=lcia_method,
        impacts=effective_impacts,
        parent_by_impact=effective_parent_by_impact,
    )
    _write_processed_mrio_metadata(
        source,
        lcia_method,
        parent_by_impact=effective_parent_by_impact,
        available_years=effective_available_years,
        unavailable_years=effective_unavailable_years,
    )
    for year in effective_available_years:
        _write_year_payloads(source, lcia_method, int(year), impacts=effective_impacts)
    return IOLCADummyRepo(
        top_path=Path(top_path),
        repo_root=repo_root,
        available_year=effective_available_years[0],
        unavailable_year=effective_unavailable_years[0] if effective_unavailable_years else 2020,
        available_years=tuple(effective_available_years),
        unavailable_years=tuple(effective_unavailable_years),
        impacts=tuple(effective_impacts),
    )


def add_io_lca_dummy_method(
    repo: IOLCADummyRepo,
    *,
    lcia_method: str,
    impacts: list[str],
    parent_by_impact: dict[str, str],
) -> None:
    """Add one LCIA method to an existing IO-LCA dummy repository."""
    method = str(lcia_method)
    _write_lcia_rps(
        source=repo.source,
        lcia_method=method,
        impacts=impacts,
        parent_by_impact=parent_by_impact,
    )
    metadata_path = _get_metadata_path(repo.source, matrix_version=None)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parent_units = {parent: "kg" for parent in sorted(set(parent_by_impact.values()))}
    available = {int(year) for year in repo.available_years}
    for raw_year, year_payload in metadata["years"].items():
        year = int(raw_year)
        method_status: dict[str, object] = {"available": year in available}
        if year not in available:
            method_status["reason"] = "extension missing"
        year_payload["extensions"][method] = dict(method_status)
        year_payload["lcia_status"][method] = dict(method_status)
        year_payload["enacting_metrics"]["units"]["lcia_by_method"][method] = dict(parent_units)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for year in repo.available_years:
        _write_year_payloads(repo.source, method, int(year), impacts=impacts)
