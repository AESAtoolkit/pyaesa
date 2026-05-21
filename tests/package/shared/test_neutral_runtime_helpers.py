from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pyaesa.shared.lcia.file_owned_tables import (
    expected_lcia_method_table_paths,
    lcia_method_from_table_path,
    lcia_method_partition_path,
    resolved_lcia_method_table_paths,
)
from pyaesa.shared.runtime.reuse.contracts import (
    asocc_signature_matches_request,
    io_lca_signature_compatible,
    normalize_selector_payload,
)
from pyaesa.shared.runtime.scenario.partitions import (
    scenario_partition_glob_pattern,
    scenario_partition_path,
    scenario_partition_token_from_path,
    trailing_scenario_partition_token,
)
from pyaesa.shared.tabular.table_io import partitioned_output_paths, read_table, write_table


class _Signature:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def as_dict(self) -> dict[str, Any]:
        return self._payload


class _Scope:
    def __init__(self, signature: dict[str, Any], ssp_scenarios: list[str] | None = None) -> None:
        self._signature = _Signature(signature)
        self._ssp_scenarios = list(ssp_scenarios or [])

    @property
    def compute_signature(self) -> _Signature:
        return self._signature

    @property
    def ssp_scenarios(self) -> list[str]:
        return self._ssp_scenarios


def test_table_io_and_scenario_partitions_cover_supported_formats(tmp_path: Path) -> None:
    frame = pd.DataFrame({"region": ["FR"], "value": [1.0]})
    csv_path = tmp_path / "table.csv"
    parquet_path = tmp_path / "table.parquet"
    pickle_path = tmp_path / "table.pickle"

    assert read_table(path=tmp_path / "missing.csv").empty
    for path in (csv_path, parquet_path, pickle_path):
        write_table(path=path, frame=frame)
        assert read_table(path=path).equals(frame)

    bad_read_path = tmp_path / "bad.xlsx"
    bad_read_path.write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError):
        read_table(path=bad_read_path)
    with pytest.raises(ValueError):
        write_table(path=tmp_path / "bad.xlsx", frame=frame)

    assert scenario_partition_glob_pattern(base_path=csv_path) == "table__ssp*.csv"
    assert scenario_partition_path(base_path=csv_path, token="SSP 2").name == "table__ssp2.csv"
    assert scenario_partition_path(base_path=csv_path, token="2").name == "table__ssp2.csv"
    assert scenario_partition_path(base_path=csv_path, token="<n>").name == "table__ssp<n>.csv"
    with pytest.raises(ValueError):
        scenario_partition_path(base_path=csv_path, token="scenario2")

    valid_partition = tmp_path / "table__ssp2.csv"
    invalid_partition = tmp_path / "table__ssp_bad.csv"
    missing_base = tmp_path / "missing_table.csv"
    missing_partition = tmp_path / "missing_table__ssp2.csv"
    unrelated = tmp_path / "unrelated__ssp2.csv"
    write_table(path=valid_partition, frame=frame)
    write_table(path=invalid_partition, frame=frame)
    write_table(path=missing_partition, frame=frame)
    write_table(path=unrelated, frame=frame)

    assert scenario_partition_token_from_path(base_path=csv_path, path=csv_path) is None
    assert scenario_partition_token_from_path(base_path=csv_path, path=valid_partition) == "ssp2"
    assert scenario_partition_token_from_path(base_path=csv_path, path=invalid_partition) is None
    assert scenario_partition_token_from_path(base_path=csv_path, path=unrelated) is None
    assert trailing_scenario_partition_token(path=valid_partition) == "ssp2"
    assert trailing_scenario_partition_token(path=invalid_partition) is None
    assert partitioned_output_paths(base_path=csv_path) == [csv_path, valid_partition]
    assert partitioned_output_paths(base_path=missing_base) == [missing_partition]


def test_lcia_file_owned_table_paths_cover_method_and_scenario_partitions(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "lca.csv"
    method_path = tmp_path / "lca__pb_lcia.csv"
    method_scenario_path = tmp_path / "lca__pb_lcia__ssp2.csv"
    base_scenario_path = tmp_path / "lca__ssp2.csv"
    blank_method_path = tmp_path / "lca__.csv"
    unrelated_path = tmp_path / "lca_extra.csv"
    frame = pd.DataFrame({"value": [1.0]})

    for path in (
        base_path,
        method_path,
        method_scenario_path,
        base_scenario_path,
        blank_method_path,
        unrelated_path,
    ):
        write_table(path=path, frame=frame)

    assert lcia_method_partition_path(base_path=base_path, lcia_method=None) == base_path
    assert lcia_method_partition_path(base_path=base_path, lcia_method="  ") == base_path
    assert lcia_method_partition_path(base_path=base_path, lcia_method="pb_lcia") == method_path
    assert lcia_method_from_table_path(path=base_path, file_stem="lca") is None
    assert lcia_method_from_table_path(path=base_scenario_path, file_stem="lca") is None
    assert lcia_method_from_table_path(path=blank_method_path, file_stem="lca") is None
    assert lcia_method_from_table_path(path=method_scenario_path, file_stem="lca") == "pb_lcia"
    with pytest.raises(ValueError):
        lcia_method_from_table_path(path=tmp_path / "other.csv", file_stem="lca")

    assert expected_lcia_method_table_paths(
        base_path=base_path,
        lcia_methods=[" pb_lcia ", "", "pb_lcia", "other"],
    ) == [tmp_path / "lca__other.csv", method_path]
    assert resolved_lcia_method_table_paths(base_path=base_path) == [
        base_path,
        base_scenario_path,
        method_path,
        method_scenario_path,
    ]


def _base_asocc_signature() -> dict[str, Any]:
    return {
        "source": "exiobase",
        "group_version": "g",
        "group_reg": True,
        "group_sec": False,
        "fu_code": "L2.c.b",
        "studied_indices_tag": "tag",
        "l1_reg_aggreg": False,
        "variant_tag": {"items": ["a", "b"]},
        "aggreg_indices": True,
        "projection_mode": "historical_reuse",
        "reg_window": [2000, 2010],
        "lcia_methods": ["m1", "m2"],
        "reference_years_input": [2019, 2020],
        "l2_reuse_years": [2010, 2015],
        "ssp_scenario_input": None,
        "selected_methods": {"bucket": ["a", "b"]},
    }


def test_reuse_contracts_cover_asocc_signature_matching() -> None:
    candidate = _base_asocc_signature()
    requested = {
        **_base_asocc_signature(),
        "lcia_methods": "m1",
        "reference_years_input": 2019,
        "l2_reuse_years": [2010],
        "ssp_scenario_input": 2,
        "selected_methods": {"bucket": ["a"]},
    }
    assert normalize_selector_payload(None) == {}
    assert normalize_selector_payload(
        {" r_c ": ["FR", ""], "blank": "", "scalar": 1, " ": "x"}
    ) == {
        "r_c": ("FR",),
        "blank": (),
        "scalar": ("1",),
    }
    with pytest.raises(ValueError):
        normalize_selector_payload(["r_c"])

    assert asocc_signature_matches_request(
        requested_signature=requested,
        scope=_Scope(candidate, ssp_scenarios=[]),
        run_ssp_scenarios=["SSP2"],
    )

    for key, value in (
        ("source", "other"),
        ("reg_window", [2001, 2010]),
        ("lcia_methods", "missing"),
        ("reference_years_input", 2030),
        ("l2_reuse_years", 2030),
        ("ssp_scenario_input", "SSP3"),
    ):
        invalid = {**requested, key: value}
        assert not asocc_signature_matches_request(
            requested_signature=invalid,
            scope=_Scope(candidate, ssp_scenarios=[]),
            run_ssp_scenarios=["SSP2"],
        )

    assert not asocc_signature_matches_request(
        requested_signature={**requested, "selected_methods": {"bucket": ["missing"]}},
        scope=_Scope(candidate, ssp_scenarios=[]),
        run_ssp_scenarios=["SSP2"],
    )
    assert not asocc_signature_matches_request(
        requested_signature=requested,
        scope=_Scope({**candidate, "selected_methods": ["bad"]}, ssp_scenarios=["SSP2"]),
        run_ssp_scenarios=None,
    )
    assert asocc_signature_matches_request(
        requested_signature={**requested, "ssp_scenario_input": ["SSP2"]},
        scope=_Scope({**candidate, "ssp_scenario_input": ["SSP2"]}),
        run_ssp_scenarios=None,
    )
    assert asocc_signature_matches_request(
        requested_signature={
            **requested,
            "reg_window": None,
            "lcia_methods": None,
            "reference_years_input": None,
            "l2_reuse_years": None,
            "selected_methods": None,
        },
        scope=_Scope(candidate, ssp_scenarios=["SSP2"]),
        run_ssp_scenarios=None,
    )
    assert not asocc_signature_matches_request(
        requested_signature=requested,
        scope=_Scope({**candidate, "lcia_methods": None}, ssp_scenarios=["SSP2"]),
        run_ssp_scenarios=None,
    )


def test_reuse_contracts_cover_io_lca_signature_matching() -> None:
    signature = {
        "project_name": "demo",
        "source": "exiobase",
        "group_reg": True,
        "group_sec": False,
        "group_version": "g",
        "fu_code": "LCA",
        "aggreg_indices": True,
        "output_format": "csv",
        "years": [2019, 2020, 2021],
        "lcia_methods": ["m1", "m2", "m3"],
        "selectors": {"r_c": ["FR", "DE"]},
    }
    kwargs = {
        "signature": signature,
        "project_name": "demo",
        "source": "exiobase",
        "group_reg": True,
        "group_sec": False,
        "group_version": "g",
        "fu_code": "LCA",
        "aggreg_indices": True,
        "output_format": "csv",
        "requested_years": {2019, 2020},
        "requested_methods": {"m1"},
        "requested_selectors": {"r_c": ("FR",)},
    }
    assert io_lca_signature_compatible(**kwargs) == (True, (1, 2))
    assert not io_lca_signature_compatible(**{**kwargs, "source": "other"})[0]
    assert not io_lca_signature_compatible(**{**kwargs, "requested_years": {2030}})[0]
    assert not io_lca_signature_compatible(**{**kwargs, "requested_methods": {"missing"}})[0]
    assert not io_lca_signature_compatible(
        **{**kwargs, "requested_selectors": {"r_c": ("missing",)}}
    )[0]
