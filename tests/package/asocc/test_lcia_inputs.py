from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest

from pyaesa import set_workspace
from pyaesa.asocc.methods import lcia_inputs as inputs_mod
from pyaesa.shared.lcia.paths import responsibility_periods_csv_path


def _write_rps_csv(
    *,
    source: str,
    lcia_method: str,
    frame: pd.DataFrame,
) -> Path:
    path = responsibility_periods_csv_path(source=source, lcia_method=lcia_method)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_rps_loading_and_cf_mapping_cover_success_and_failures(tmp_path: Path) -> None:
    set_workspace(tmp_path / "workspace", refresh=True)

    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        inputs_mod.read_rps_frame(missing_path)

    source = "oecd_v2025"
    lcia_method = "gwp100_lcia"
    valid_path = _write_rps_csv(
        source=source,
        lcia_method=lcia_method,
        frame=pd.DataFrame(
            [
                {"impact": "AAL", "impact_parent": "AAL"},
                {"impact": "BI FD GHG", "impact_parent": "BI FD"},
            ]
        ),
    )
    loaded = inputs_mod.load_rps_frame(source=source, lcia_method=lcia_method)
    assert loaded.attrs["source_csv"] == str(valid_path)

    mapping = inputs_mod.load_impact_parent_mapping(source=source, lcia_method=lcia_method)
    assert mapping.to_dict() == {"AAL": "AAL", "BI FD GHG": "BI FD"}
    assert mapping.attrs["source_csv"] == str(valid_path)

    _write_rps_csv(
        source=source,
        lcia_method=lcia_method,
        frame=pd.DataFrame([{"impact": "AAL"}]),
    )
    with pytest.raises(ValueError):
        inputs_mod.load_impact_parent_mapping(source=source, lcia_method=lcia_method)

    _write_rps_csv(
        source=source,
        lcia_method=lcia_method,
        frame=pd.DataFrame([{"impact": "AAL", "impact_parent": None}]),
    )
    with pytest.raises(ValueError):
        inputs_mod.load_impact_parent_mapping(source=source, lcia_method=lcia_method)

    _write_rps_csv(
        source=source,
        lcia_method=lcia_method,
        frame=pd.DataFrame(
            [
                {"impact": "AAL", "impact_parent": "P1"},
                {"impact": "AAL", "impact_parent": "P2"},
            ]
        ),
    )
    with pytest.raises(ValueError):
        inputs_mod.load_impact_parent_mapping(source=source, lcia_method=lcia_method)


def test_lcia_normalization_contracts_cover_all_reachable_paths() -> None:
    assert inputs_mod.normalize_lcia_methods(None) is None
    assert inputs_mod.normalize_lcia_methods("gwp100_lcia") == ["gwp100_lcia"]
    assert inputs_mod.normalize_lcia_methods([" gwp100_lcia ", "", "pb_lcia"]) == [
        "gwp100_lcia",
        "pb_lcia",
    ]
    with pytest.raises(ValueError):
        inputs_mod.normalize_lcia_methods(["gwp100_lcia", "gwp100_lcia"])


def test_lcia_parent_aggregation_contracts_cover_success_and_failures() -> None:
    mapping = pd.Series({"AAL child": "AAL", "BI FD GHG": "BI FD"}, name="impact_parent")
    mapping.attrs["source_csv"] = "demo.csv"

    plain = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.Index(["x", "y"], name="r_p"))
    assert inputs_mod.aggregate_frame_to_parent(plain, mapping).equals(plain)

    single = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.Index(["AAL child", "BI FD GHG"], name="impact"),
    )
    aggregated_single = inputs_mod.aggregate_frame_to_parent(single, mapping)
    assert aggregated_single.index.tolist() == ["AAL", "BI FD"]

    multi = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [("AAL child", "FR"), ("AAL child", "DE")],
            names=["impact", "r_p"],
        ),
    )
    aggregated_multi = inputs_mod.aggregate_frame_to_parent(multi, mapping)
    assert aggregated_multi.index.names == ["impact", "r_p"]
    assert aggregated_multi.loc[("AAL", "FR"), "value"] == pytest.approx(1.0)

    with pytest.raises(ValueError):
        inputs_mod.aggregate_frame_to_parent(
            pd.DataFrame(
                {"value": [1.0]},
                index=pd.Index(["missing"], name="impact"),
            ),
            mapping,
        )

    aggregated = inputs_mod.aggregate_lcia_to_parent(
        {"e_cba_fd_reg": single},
        mapping,
    )
    assert list(aggregated) == ["e_cba_fd_reg"]


def test_load_pr_hr_timeseries_covers_store_selection_and_cache_init(tmp_path: Path) -> None:
    set_workspace(tmp_path / "workspace", refresh=True)
    source = "oecd_v2025"
    lcia_method = "gwp100_lcia"
    _write_rps_csv(
        source=source,
        lcia_method=lcia_method,
        frame=pd.DataFrame(
            [
                {"impact": "AAL", "impact_parent": "AAL"},
                {"impact": "BI FD GHG", "impact_parent": "BI FD"},
            ]
        ),
    )

    state = SimpleNamespace(
        lcia_timeseries={},
        lcia_timeseries_original={},
        rps_by_method={},
        cf_by_method={},
    )

    inputs_mod.initialize_pr_hr_timeseries(
        source=source,
        state=state,
        lcia_methods=[lcia_method],
        selected_l1=["AR(E^{CBA_FD})"],
        store="grouped",
    )
    assert state.lcia_timeseries == {}
    assert state.rps_by_method == {}

    inputs_mod.initialize_pr_hr_timeseries(
        source=source,
        state=state,
        lcia_methods=None,
        selected_l1=["PR-HR(Ecap,cum)"],
        store="grouped",
    )
    assert state.lcia_timeseries == {}

    inputs_mod.initialize_pr_hr_timeseries(
        source=source,
        state=state,
        lcia_methods=[lcia_method],
        selected_l1=["PR-HR(Ecap,cum)"],
        store="grouped",
    )
    assert state.lcia_timeseries[lcia_method] == {"CBA_FD": {}, "PBA": {}}
    assert lcia_method in state.rps_by_method
    assert lcia_method in state.cf_by_method

    inputs_mod.initialize_pr_hr_timeseries(
        source=source,
        state=state,
        lcia_methods=[lcia_method],
        selected_l1=["PR-HR(Ecap,cum)"],
        store="original",
    )
    assert state.lcia_timeseries_original[lcia_method] == {"CBA_FD": {}, "PBA": {}}
