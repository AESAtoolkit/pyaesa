from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa.shared.runtime.io import filesystem as filesystem_mod
from pyaesa.shared.runtime.io import persisted_paths as persisted_mod
from pyaesa.shared.runtime.reuse import derived_state as derived_mod
from pyaesa.shared.runtime.scenario import file_routing as routing_mod
from pyaesa.shared.selectors import time_selectors as selector_mod
from pyaesa.shared.tabular import empty_rows as empty_rows_mod
from pyaesa.shared.tabular import scalars as scalar_mod
from pyaesa.shared.tabular import table_io as table_io_mod
from pyaesa.shared.tabular import wide_tables as wide_mod


def test_normalize_persisted_paths_contracts() -> None:
    existing_path = Path("demo.csv")
    assert persisted_mod.normalize_persisted_paths(raw_paths=None) == []
    assert persisted_mod.normalize_persisted_paths(
        raw_paths=[existing_path, " nested/demo.pickle "],
    ) == [existing_path, Path("nested/demo.pickle")]


def test_scoped_existing_table_paths_filters_and_errors(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    csv_path = root / "a.csv"
    pickle_path = root / "nested" / "b.pickle"
    pickle_path.parent.mkdir()
    pd.DataFrame({"value": [1.0]}).to_csv(csv_path, index=False)
    pd.DataFrame({"value": [2.0]}).to_pickle(pickle_path)

    ordered = persisted_mod.scoped_existing_table_paths(
        raw_paths=[pickle_path, csv_path],
        root=root,
        field_name="tables",
    )
    assert ordered == sorted([csv_path.resolve(), pickle_path.resolve()])

    outside_path = tmp_path / "outside.csv"
    outside_path.write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        persisted_mod.scoped_existing_table_paths(
            raw_paths=[outside_path],
            root=root,
            field_name="tables",
        )

    unsupported_path = root / "bad.txt"
    unsupported_path.write_text("demo", encoding="utf-8")
    with pytest.raises(ValueError):
        persisted_mod.scoped_existing_table_paths(
            raw_paths=[unsupported_path],
            root=root,
            field_name="tables",
        )

    with pytest.raises(ValueError):
        persisted_mod.scoped_existing_table_paths(
            raw_paths=[root / "missing.csv"],
            root=root,
            field_name="tables",
        )

    with pytest.raises(ValueError):
        persisted_mod.scoped_existing_table_paths(
            raw_paths=[csv_path, csv_path],
            root=root,
            field_name="tables",
        )


def test_time_selector_contracts_cover_supported_and_invalid_inputs() -> None:
    assert selector_mod.normalize_optional_year_selector(None, name="years") is None
    assert selector_mod.normalize_optional_year_selector(2020, name="years") == [2020]
    assert selector_mod.normalize_optional_year_selector(range(2020, 2023), name="years") == [
        2020,
        2021,
        2022,
    ]
    assert selector_mod.normalize_optional_year_selector([2022, 2021, 2021], name="years") == [
        2021,
        2022,
    ]

    with pytest.raises(ValueError):
        selector_mod.normalize_optional_year_selector(cast(Any, (2020, 2021)), name="years")

    assert selector_mod.normalize_optional_reg_window_selector(None) is None
    assert selector_mod.normalize_optional_reg_window_selector(range(2020, 2022)) == [
        2020,
        2021,
    ]
    assert selector_mod.normalize_optional_reg_window_selector([2020, 2021]) == [2020, 2021]

    with pytest.raises(ValueError):
        selector_mod.normalize_optional_reg_window_selector(cast(Any, 2020))

    assert selector_mod.normalize_time_selector_mapping(None) is None
    assert selector_mod.normalize_time_selector_mapping(
        {
            "years": range(2020, 2022),
            "reference_years": [2018, 2019, 2018],
            "l2_reuse_years": 2020,
            "reg_window": range(2019, 2021),
            "other": "keep",
        }
    ) == {
        "years": [2020, 2021],
        "reference_years": [2018, 2019],
        "l2_reuse_years": [2020],
        "reg_window": [2019, 2020],
        "other": "keep",
    }

    with pytest.raises(ValueError):
        selector_mod.normalize_time_selector_mapping({"years": (2020, 2021)})


def test_read_table_and_wide_table_contracts(tmp_path: Path) -> None:
    mixed_year_frame = pd.DataFrame(
        {
            "r_p": ["FR", "US"],
            "detail": ["keep", None],
            "2020": [1.0, 2.0],
            2021: [3.0, 4.0],
            "bad_year": [5.0, 6.0],
        }
    )
    base_frame = pd.DataFrame(
        {
            "r_p": ["FR", "US"],
            "detail": ["keep", None],
            "2020": [1.0, 2.0],
            "2021": [3.0, 4.0],
            "bad_year": [5.0, 6.0],
        }
    )
    csv_path = tmp_path / "table.csv"
    pickle_path = tmp_path / "table.pickle"
    parquet_path = tmp_path / "table.parquet"
    bad_path = tmp_path / "table.txt"
    base_frame.to_csv(csv_path, index=False)
    base_frame.to_pickle(pickle_path)
    base_frame.to_parquet(parquet_path, index=False)
    bad_path.write_text("demo", encoding="utf-8")

    csv_frame = table_io_mod.read_table(path=csv_path)
    pickle_frame = table_io_mod.read_table(path=pickle_path)
    parquet_frame = table_io_mod.read_table(path=parquet_path)
    assert list(csv_frame.columns) == list(base_frame.columns)
    assert list(pickle_frame.columns) == list(base_frame.columns)
    assert list(parquet_frame.columns) == list(base_frame.columns)

    with pytest.raises(ValueError):
        table_io_mod.read_table(path=bad_path)

    detected_years = wide_mod.detect_year_columns(mixed_year_frame)
    assert detected_years == ["2020", "2021"]
    assert wide_mod.requested_year_columns(
        mixed_year_frame,
        requested_years=[2021, 2022],
    ) == ["2021"]
    assert wide_mod.id_columns(
        base_frame,
        year_columns=detected_years,
        ignored_columns={"detail"},
    ) == ["r_p", "bad_year"]

    all_years, requested_columns, metadata_columns = wide_mod.planned_melt_columns(
        base_frame,
        requested_years=[2020, 2021],
        ignored_columns={"detail"},
    )
    assert all_years == ["2020", "2021"]
    assert requested_columns == ["2020", "2021"]
    assert metadata_columns == ["r_p", "bad_year"]

    melted = wide_mod.melt_requested_year_value_rows(
        base_frame,
        requested_years=[2020, 2021],
        ignored_columns={"detail"},
    )
    assert set(melted.columns) == {"r_p", "bad_year", "year", "value"}
    assert set(melted["year"]) == {"2020", "2021"}

    with pytest.raises(ValueError):
        wide_mod.validate_complete_wide_year_values(
            pd.DataFrame({"r_p": ["FR"], "2020": [1.0], "2021": [pd.NA]}),
            year_columns=["2020", "2021"],
            where="demo table",
        )

    empty_melt = wide_mod.melt_requested_year_value_rows(
        base_frame,
        requested_years=[2010],
        ignored_columns={"detail"},
    )
    assert list(empty_melt.columns) == ["r_p", "bad_year", "year", "value"]
    assert empty_melt.empty

    labeled = wide_mod.row_label(
        row=pd.Series({"r_p": "FR", "detail": pd.NA}),
        columns=["r_p", "detail"],
        default_prefix="row",
    )
    assert labeled == "r_p=FR"
    default_labeled = wide_mod.row_label(
        row=pd.Series({"r_p": pd.NA}, name=2),
        columns=["r_p"],
        default_prefix="row",
    )
    assert default_labeled == "row 3"

    with pytest.raises(TypeError):
        wide_mod._is_missing_scalar(["not", "scalar"])  # noqa: SLF001
    assert wide_mod.row_identity_key(row=pd.Series({"r_p": "FR"}), columns=[], ordinal=2) == (
        "__row__",
        "2",
    )
    assert wide_mod.row_identity_key(
        row=pd.Series({"r_p": "FR"}),
        columns=["r_p"],
        ordinal=0,
    ) == ("FR", "__dup_0")


def test_filesystem_atomic_writer_cleans_temp_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "demo.txt"
    seen: list[Path] = []

    def _failing_writer(tmp_file: Path) -> None:
        seen.append(tmp_file)
        tmp_file.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        filesystem_mod.write_via_atomic_temp(target, writer=_failing_writer)

    assert seen
    assert not seen[0].exists()
    assert not target.exists()

    target_without_tmp = tmp_path / "nested" / "demo_no_tmp.txt"

    def _fail_before_write(_tmp_file: Path) -> None:
        raise RuntimeError("boom_without_tmp")

    with pytest.raises(RuntimeError):
        filesystem_mod.write_via_atomic_temp(target_without_tmp, writer=_fail_before_write)

    assert not target_without_tmp.exists()

    target_deleted_tmp = tmp_path / "nested" / "demo_deleted_tmp.txt"

    def _delete_then_fail(tmp_file: Path) -> None:
        tmp_file.unlink()
        raise RuntimeError("boom_deleted_tmp")

    with pytest.raises(RuntimeError):
        filesystem_mod.write_via_atomic_temp(target_deleted_tmp, writer=_delete_then_fail)

    assert not target_deleted_tmp.exists()


def test_scenario_file_routing_contracts(tmp_path: Path) -> None:
    historical = routing_mod.ScenarioTaggedFileSpec(
        path=tmp_path / "historical.csv",
        scenario=None,
        years=(2018, 2019),
    )
    ssp2 = routing_mod.ScenarioTaggedFileSpec(
        path=tmp_path / "ssp2.csv",
        scenario="SSP2",
        years=(2030, 2040),
    )
    ssp3 = routing_mod.ScenarioTaggedFileSpec(
        path=tmp_path / "ssp3.csv",
        scenario="SSP3",
        years=(2030, 2040),
    )

    assert routing_mod.allowed_scenarios_for_year(
        year=2030,
        ssp_scenario_options_by_year={2030: [None, "SSP2"]},
    ) == {"SSP2"}
    assert (
        routing_mod.allowed_scenarios_for_year(year=2030, ssp_scenario_options_by_year=None)
        == set()
    )

    routing_mod.validate_scenario_inventory(
        specs=(historical, ssp2, ssp3),
        family_label="external files",
        item_label="demo",
    )

    with pytest.raises(ValueError):
        routing_mod.validate_scenario_inventory(
            specs=(ssp2, routing_mod.ScenarioTaggedFileSpec(ssp2.path, "SSP2", (2030,))),
            family_label="external files",
            item_label="demo",
        )

    with pytest.raises(ValueError):
        routing_mod.validate_scenario_inventory(
            specs=(ssp2, routing_mod.ScenarioTaggedFileSpec(ssp3.path, "SSP3", (2030,))),
            family_label="external files",
            item_label="demo",
        )

    with pytest.raises(ValueError):
        routing_mod.validate_scenario_inventory(
            specs=(historical, routing_mod.ScenarioTaggedFileSpec(ssp2.path, "SSP2", (2019, 2030))),
            family_label="external files",
            item_label="demo",
        )

    assignments = routing_mod.resolve_year_assignments(
        specs=(historical, ssp2, ssp3),
        years=[2018, 2030, 2040],
        ssp_scenario_options_by_year={2030: ["SSP2"], 2040: ["SSP2"]},
        family_label="external files",
        item_label="demo",
        expected_stems=["historical", "SSP2", "SSP3"],
    )
    assert assignments == {
        historical.path: [2018],
        ssp2.path: [2030, 2040],
    }

    with pytest.raises(ValueError):
        routing_mod.resolve_year_assignments(
            specs=(historical,),
            years=[2050],
            ssp_scenario_options_by_year=None,
            family_label="external files",
            item_label="demo",
        )

    with pytest.raises(ValueError):
        routing_mod.resolve_year_assignments(
            specs=(ssp2, ssp3),
            years=[2030],
            ssp_scenario_options_by_year={2030: ["SSP4"]},
            family_label="external files",
            item_label="demo",
        )


def test_derived_state_persistence_contracts(tmp_path: Path) -> None:
    file_one = tmp_path / "one.csv"
    file_two = tmp_path / "two.csv"
    file_one.write_text("value\n1\n", encoding="utf-8")
    file_two.write_text("value\n2\n", encoding="utf-8")
    payload: dict[str, object] = {}

    derived_mod.set_request_state(
        payload=payload,
        state_key="figures",
        request_signature={"dpi": 150},
        paths=[file_two, file_one, file_one],
        compute_signature={"source": "demo"},
        extra={"note": "kept"},
    )
    stored_block = derived_mod._state_block(payload, state_key="figures")
    assert stored_block is not None
    assert stored_block["note"] == "kept"
    assert stored_block["paths"] == sorted([str(file_one), str(file_two)])
    assert derived_mod._paths_exist([file_one, file_two]) is True
    assert derived_mod._paths_exist([]) is False

    assert derived_mod.request_state_matches(
        payload=payload,
        state_key="figures",
        request_signature={"dpi": 150},
        compute_signature={"source": "demo"},
    )
    assert not derived_mod.request_state_matches(
        payload=payload,
        state_key="figures",
        request_signature={"dpi": 300},
    )
    assert not derived_mod.request_state_matches(
        payload=payload,
        state_key="figures",
        request_signature={"dpi": 150},
        compute_signature={"source": "other"},
    )

    file_two.unlink()
    assert not derived_mod.request_state_matches(
        payload=payload,
        state_key="figures",
        request_signature={"dpi": 150},
        compute_signature={"source": "demo"},
    )


def test_scalar_and_empty_row_contracts(tmp_path: Path) -> None:
    assert scalar_mod.canonical_scalar_text(pd.NA) == "missing"
    assert scalar_mod.canonical_scalar_text("  demo  ") == "demo"
    assert scalar_mod.is_display_missing(None) is True
    assert scalar_mod.is_display_missing(float("nan")) is True
    assert scalar_mod.is_display_missing(" NaT ") is True
    assert scalar_mod.is_display_missing("demo") is False
    assert scalar_mod.display_scalar(2) == "2"
    assert scalar_mod.display_scalar(2.0) == "2"
    assert scalar_mod.display_scalar(2.5) == "2.5"
    assert scalar_mod.display_scalar(" 3.0 ") == "3"
    assert scalar_mod.display_scalar(" 3.50 ") == "3.50"
    assert scalar_mod.display_scalar(" demo ") == "demo"
    assert scalar_mod.display_scalar(" ") is None
    assert scalar_mod.sanitize_token(" A/B demo* ") == "A_B_demo"
    assert scalar_mod.sanitize_token(pd.NA) == "missing"

    frame = pd.DataFrame(
        {
            "text": ["demo", "   ", None],
            "value": [1.0, pd.NA, pd.NA],
        }
    )
    cleaned = empty_rows_mod.drop_fully_empty_rows(frame=frame)
    assert cleaned.to_dict(orient="records") == [{"text": "demo", "value": 1.0}]

    string_frame = pd.DataFrame(
        {
            "text": pd.Series(["demo", "   ", None], dtype="string"),
            "value": [1.0, pd.NA, pd.NA],
        }
    )
    string_cleaned = empty_rows_mod.drop_fully_empty_rows(frame=string_frame)
    assert string_cleaned.to_dict(orient="records") == [{"text": "demo", "value": 1.0}]

    already_clean = pd.DataFrame({"text": ["demo"], "value": [1.0]})
    untouched = empty_rows_mod.drop_fully_empty_rows(frame=already_clean)
    assert untouched.equals(already_clean)
    assert untouched is not already_clean

    empty = pd.DataFrame(columns=["text", "value"])
    empty_cleaned = empty_rows_mod.drop_fully_empty_rows(frame=empty)
    assert empty_cleaned.empty
    assert empty_cleaned is not empty
