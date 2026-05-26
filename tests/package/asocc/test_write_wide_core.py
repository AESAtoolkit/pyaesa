from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pyaesa.asocc.orchestration.write.tables.allocation_frame import prepare_allocation_frame
from pyaesa.asocc.orchestration.write.tables.wide_merge import _safe_cell_equal, merge_wide_frames
from pyaesa.asocc.orchestration.write.tables.wide_table_io import upsert_wide_table
from pyaesa.asocc.orchestration.write.tables.wide_validation import (
    assert_no_duplicate_columns,
    normalize_existing_wide_table,
    validate_wide_frame,
)
from pyaesa.asocc.runtime.output.contracts import (
    IdentifierSchema,
    OutputRoute,
    OutputSpec,
)


def _spec() -> OutputSpec:
    return OutputSpec(
        l1_l2_method="UT(FD)",
        l2_method="UT(FD)",
        l1_method=None,
        file_stem="table_l2_asocc",
        route=OutputRoute(
            level="L2",
            bucket="l2_vs_global",
            source="oecd_v2025",
            grouped_mode=False,
            variant_tag=None,
            ssp_scenario=None,
            lcia_method=None,
            projection_subfolder=None,
        ),
        scenario_dependent=False,
        identifier_columns=("r_p", "s_p"),
    )


class _EqRaises:
    def __eq__(self, _other):
        raise TypeError("boom")


class _BoolRaises:
    def __bool__(self):
        raise TypeError("boom")


class _EqArray:
    def __init__(self, value):
        self._value = value

    def __eq__(self, _other):
        return self._value


class _EqScalar:
    def __init__(self, value):
        self._value = value

    def __eq__(self, _other):
        return self._value


def test_safe_cell_equal_handles_scalar_and_edge_cases() -> None:
    import numpy as np

    assert _safe_cell_equal(1, 1) is True
    assert _safe_cell_equal(1, 2) is False
    assert _safe_cell_equal(pd.NA, pd.NA) is False
    assert _safe_cell_equal(_EqRaises(), 1) is False
    assert _safe_cell_equal(_EqArray(np.array([True, False], dtype=object)), 1) is False
    assert _safe_cell_equal(_EqArray(np.array([pd.NA], dtype=object)), 1) is False
    assert _safe_cell_equal(_EqArray(np.array([_BoolRaises()], dtype=object)), 1) is False
    assert _safe_cell_equal(_EqScalar(_BoolRaises()), 1) is False


def test_merge_wide_frames_empty_returns_schema_columns() -> None:
    out = merge_wide_frames(
        frames=[],
        identifier_columns=("r_p", "s_p"),
        year_columns=("2020", "2021"),
        where="x",
    )
    assert list(out.columns) == ["r_p", "s_p", "2020", "2021"]
    assert out.empty


def test_merge_wide_frames_internal_hot_path_and_duplicate_coalesce() -> None:
    idx = pd.MultiIndex.from_tuples([("FR", "A"), ("FR", "A")], names=["r_p", "s_p"])
    frame_1 = pd.DataFrame({2020: [1.0, 1.0]}, index=idx)
    frame_2 = pd.DataFrame({2021: [2.0, 2.0]}, index=idx)

    out = merge_wide_frames(
        frames=[frame_1, frame_2],
        identifier_columns=("r_p", "s_p"),
        year_columns=("2020", "2021"),
        where="x",
    )
    assert out.shape == (1, 4)
    assert out.loc[0, "2020"] == 1.0
    assert out.loc[0, "2021"] == 2.0


def test_merge_wide_frames_hot_path_float_and_python_int_year_labels() -> None:
    frame_float = pd.DataFrame({2020.0: ["1.0"]}, index=pd.Index(["FR"], name="r_p"))
    frame_int = pd.DataFrame(
        [[2.0]],
        index=pd.Index(["FR"], name="r_p"),
        columns=pd.Index([int(2021)], dtype=object),
    )
    out = merge_wide_frames(
        frames=[frame_float, frame_int],
        identifier_columns=("r_p",),
        year_columns=("2020", "2021"),
        where="hot-year-labels",
    )
    assert list(out.columns) == ["r_p", "2020", "2021"]
    assert float(out.loc[0, "2020"]) == 1.0
    assert float(out.loc[0, "2021"]) == 2.0


def test_merge_wide_frames_hot_path_coerces_object_dtype_year_series() -> None:
    frame = pd.DataFrame({"2020": ["1.0"]}, index=pd.Index(["FR"], name="r_p"))
    out = merge_wide_frames(
        frames=[frame],
        identifier_columns=("r_p",),
        year_columns=("2020",),
        where="hot-object",
    )
    assert float(out.loc[0, "2020"]) == 1.0

    mixed_object = pd.DataFrame(
        {
            "2020": pd.Series(
                [1, "2"],
                index=pd.Index(["FR", "US"], name="r_p"),
                dtype=object,
            )
        }
    )
    mixed_out = merge_wide_frames(
        frames=[mixed_object],
        identifier_columns=("r_p",),
        year_columns=("2020",),
        where="hot-object-mixed",
    )
    assert mixed_out["2020"].tolist() == [1.0, 2.0]


def test_merge_wide_frames_hot_duplicate_year_bucket_paths() -> None:
    frame_a = pd.DataFrame({"2020": ["1.0", "1.0"]}, index=pd.Index(["FR", "FR"], name="r_p"))
    frame_b = pd.DataFrame({"2020": ["1.0"]}, index=pd.Index(["FR"], name="r_p"))
    out = merge_wide_frames(
        frames=[frame_a, frame_b],
        identifier_columns=("r_p",),
        year_columns=("2020",),
        where="hot-bucket",
    )
    assert list(out.columns) == ["r_p", "2020"]
    assert out.shape[0] == 1
    assert float(out.loc[0, "2020"]) == 1.0


def test_merge_wide_frames_hot_same_year_disjoint_rows_skip_year_unstack() -> None:
    frame_a = pd.DataFrame({"2020": [1.0]}, index=pd.Index(["FR"], name="r_p"))
    frame_b = pd.DataFrame({"2020": [2.0]}, index=pd.Index(["DE"], name="r_p"))
    out = merge_wide_frames(
        frames=[frame_a, frame_b],
        identifier_columns=("r_p",),
        year_columns=("2020",),
        where="hot-same-year-disjoint",
    )
    assert list(out.columns) == ["r_p", "2020"]
    assert set(out["r_p"]) == {"FR", "DE"}
    assert float(out.loc[out["r_p"] == "FR", "2020"].iloc[0]) == 1.0
    assert float(out.loc[out["r_p"] == "DE", "2020"].iloc[0]) == 2.0


def test_merge_wide_frames_hot_repeated_year_disjoint_rows_unstack() -> None:
    frame_a = pd.DataFrame({"2020": [1.0]}, index=pd.Index(["FR"], name="r_p"))
    frame_b = pd.DataFrame({"2021": [3.0]}, index=pd.Index(["FR"], name="r_p"))
    frame_c = pd.DataFrame({"2020": [2.0]}, index=pd.Index(["DE"], name="r_p"))

    out = merge_wide_frames(
        frames=[frame_a, frame_b, frame_c],
        identifier_columns=("r_p",),
        year_columns=("2020", "2021"),
        where="hot-repeated-year-disjoint",
    )

    assert list(out.columns) == ["r_p", "2020", "2021"]
    assert set(out["r_p"]) == {"FR", "DE"}
    assert float(out.loc[out["r_p"] == "FR", "2020"].iloc[0]) == 1.0
    assert float(out.loc[out["r_p"] == "FR", "2021"].iloc[0]) == 3.0
    assert float(out.loc[out["r_p"] == "DE", "2020"].iloc[0]) == 2.0


def test_merge_wide_frames_hot_multiindex_duplicate_bucket_paths() -> None:
    idx = pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"])
    frame_a = pd.DataFrame({"2020": ["1.0"]}, index=idx)
    frame_b = pd.DataFrame({"2020": ["1.0"]}, index=idx)
    out = merge_wide_frames(
        frames=[frame_a, frame_b],
        identifier_columns=("r_p", "s_p"),
        year_columns=("2020",),
        where="hot-bucket-mi",
    )
    assert list(out.columns) == ["r_p", "s_p", "2020"]
    assert out.shape[0] == 1
    assert float(out.loc[0, "2020"]) == 1.0

    frame_2020 = pd.DataFrame(
        {"2020": [2.0]},
        index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
    )
    frame_2021 = pd.DataFrame(
        {"2021": [3.0]},
        index=pd.MultiIndex.from_tuples([("DE", "B")], names=["r_p", "s_p"]),
    )
    disjoint = merge_wide_frames(
        frames=[frame_2020, frame_2021],
        identifier_columns=("r_p", "s_p"),
        year_columns=("2020", "2021"),
        where="hot-bucket-mi-disjoint",
    )
    assert list(disjoint.columns) == ["r_p", "s_p", "2020", "2021"]
    assert set(zip(disjoint["r_p"], disjoint["s_p"], strict=True)) == {("FR", "A"), ("DE", "B")}


def test_merge_wide_frames_rejects_bad_index_contracts() -> None:
    with pytest.raises(ValueError):
        merge_wide_frames(
            frames=[pd.DataFrame({"r_p": ["FR"], "2020": [1.0]})],
            identifier_columns=("r_p",),
            year_columns=("2020",),
            where="x",
        )

    idx_unnamed = pd.Index(["FR"], name=None)
    with pytest.raises(ValueError):
        merge_wide_frames(
            frames=[pd.DataFrame({"2020": [1.0]}, index=idx_unnamed)],
            identifier_columns=("r_p",),
            year_columns=("2020",),
            where="x",
        )

    idx_mismatch = pd.Index(["FR"], name="region")
    with pytest.raises(ValueError):
        merge_wide_frames(
            frames=[pd.DataFrame({"2020": [1.0]}, index=idx_mismatch)],
            identifier_columns=("r_p",),
            year_columns=("2020",),
            where="x",
        )


def test_merge_wide_frames_rejects_non_year_columns_and_conflicts() -> None:
    idx = pd.Index(["FR"], name="r_p")
    with pytest.raises(ValueError):
        merge_wide_frames(
            frames=[pd.DataFrame({"bad": [1.0]}, index=idx)],
            identifier_columns=("r_p",),
            year_columns=("2020",),
            where="x",
        )

    frame_a = pd.DataFrame({"2020": [1.0]}, index=idx)
    frame_b = pd.DataFrame({"2020": [2.0]}, index=idx)
    with pytest.raises(ValueError):
        merge_wide_frames(
            frames=[frame_a, frame_b],
            identifier_columns=("r_p",),
            year_columns=("2020",),
            where="x",
        )


def test_merge_wide_frames_non_hot_prepare_frame_with_duplicate_index() -> None:
    idx = pd.MultiIndex.from_tuples(
        [("FR", "A"), ("FR", "A"), ("US", "A")],
        names=["r_p", "s_p"],
    )
    frame = pd.DataFrame({2020: ["1.0", "1.0", "2.0"], 2021.0: ["3.0", "3.0", "4.0"]}, index=idx)
    out = merge_wide_frames(
        frames=[frame],
        identifier_columns=("r_p", "s_p"),
        year_columns=("2020", "2021"),
        where="prepare-duplicate",
    )
    assert list(out.columns) == ["r_p", "s_p", "2020", "2021"]
    assert set(out["r_p"]) == {"FR", "US"}

    int_year = merge_wide_frames(
        frames=[
            pd.DataFrame(
                {2020: [1.0]},
                index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
            ),
            pd.DataFrame(
                {2020: [2.0]},
                index=pd.MultiIndex.from_tuples([("US", "A")], names=["r_p", "s_p"]),
            ),
        ],
        identifier_columns=("r_p", "s_p"),
        year_columns=("2020",),
        where="prepare-int-year",
    )
    assert list(int_year.columns) == ["r_p", "s_p", "2020"]
    assert set(int_year["r_p"]) == {"FR", "US"}

    object_int_year = merge_wide_frames(
        frames=[
            pd.DataFrame(
                [[1.0, 2.0]],
                index=pd.Index(["FR"], name="r_p"),
                columns=pd.Index([2020, "2021"], dtype=object),
            )
        ],
        identifier_columns=("r_p",),
        year_columns=("2020", "2021"),
        where="prepare-object-int-year",
    )
    assert list(object_int_year.columns) == ["r_p", "2020", "2021"]


def test_merge_wide_frames_general_merge_paths() -> None:
    frame_1 = pd.DataFrame(
        {2020: ["1.0", "2.0"], 2021.0: [pd.NA, "3.0"]},
        index=pd.Index(["FR", "US"], name="r_p"),
    )
    frame_2 = pd.DataFrame(
        {"2021": ["4.0", "6.0"], "2022": ["5.0", "7.0"]},
        index=pd.Index(["FR", "DE"], name="r_p"),
    )

    out = merge_wide_frames(
        frames=[frame_1, frame_2],
        identifier_columns=("r_p",),
        year_columns=("2020", "2021", "2022"),
        where="merge-general",
    )
    assert list(out.columns) == ["r_p", "2020", "2021", "2022"]
    row_fr = out.loc[out["r_p"] == "FR"].iloc[0]
    row_us = out.loc[out["r_p"] == "US"].iloc[0]
    row_de = out.loc[out["r_p"] == "DE"].iloc[0]
    assert float(row_fr["2020"]) == 1.0
    assert float(row_fr["2021"]) == 4.0
    assert float(row_fr["2022"]) == 5.0
    assert float(row_us["2021"]) == 3.0
    assert float(row_de["2022"]) == 7.0


def test_merge_wide_frames_general_merge_conflict_and_branch_variants() -> None:
    frame_1 = pd.DataFrame({"2020": [1.0], "2021": [2.0]}, index=pd.Index(["FR"], name="r_p"))
    frame_2 = pd.DataFrame({"2020": [9.0], "2021": [8.0]}, index=pd.Index(["FR"], name="r_p"))
    with pytest.raises(ValueError):
        merge_wide_frames(
            frames=[frame_1, frame_2],
            identifier_columns=("r_p",),
            year_columns=("2020", "2021"),
            where="merge-conflict",
        )

    same_value = pd.DataFrame({"2020": [1.0]}, index=pd.Index(["FR"], name="r_p"))
    same_out = merge_wide_frames(
        frames=[same_value, same_value],
        identifier_columns=("r_p",),
        year_columns=("2020",),
        where="overlap-same-value",
    )
    assert same_out.shape == (1, 2)

    no_common = merge_wide_frames(
        frames=[
            pd.DataFrame({"2020": [1.0]}, index=pd.Index(["FR"], name="r_p")),
            pd.DataFrame({"2020": [2.0]}, index=pd.Index(["DE"], name="r_p")),
        ],
        identifier_columns=("r_p",),
        year_columns=("2020",),
        where="overlap-no-common-index",
    )
    assert set(no_common["r_p"]) == {"FR", "DE"}

    add_cols = merge_wide_frames(
        frames=[
            pd.DataFrame({"2020": [1.0], "2021": [2.0]}, index=pd.Index(["FR"], name="r_p")),
            pd.DataFrame({"2022": [3.0]}, index=pd.Index(["DE"], name="r_p")),
        ],
        identifier_columns=("r_p",),
        year_columns=("2020", "2021", "2022"),
        where="general-add-cols",
    )
    assert set(add_cols["r_p"]) == {"FR", "DE"}

    no_overlap_cols = merge_wide_frames(
        frames=[
            pd.DataFrame({"2020": [1.0], "2022": [3.0]}, index=pd.Index(["FR"], name="r_p")),
            pd.DataFrame({"2021": [2.0]}, index=pd.Index(["FR"], name="r_p")),
            pd.DataFrame({"2021": [2.0]}, index=pd.Index(["FR"], name="r_p")),
        ],
        identifier_columns=("r_p",),
        year_columns=("2020", "2021", "2022"),
        where="general-loop-no-overlap",
    )
    assert float(no_overlap_cols.loc[0, "2021"]) == 2.0
    disjoint_add_cols = merge_wide_frames(
        frames=[
            pd.DataFrame({"2020": [1.0]}, index=pd.Index(["FR"], name="r_p")),
            pd.DataFrame({"2021": [2.0]}, index=pd.Index(["DE"], name="r_p")),
        ],
        identifier_columns=("r_p",),
        year_columns=("2020", "2021"),
        where="general-disjoint-add-cols",
    )
    assert set(disjoint_add_cols["r_p"]) == {"FR", "DE"}

    disjoint_overlap = merge_wide_frames(
        frames=[
            pd.DataFrame(
                {"2020": [1.0], "2021": [2.0]},
                index=pd.Index(["FR"], name="r_p"),
            ),
            pd.DataFrame(
                {"2021": [3.0], "2022": [4.0]},
                index=pd.Index(["DE"], name="r_p"),
            ),
        ],
        identifier_columns=("r_p",),
        year_columns=("2020", "2021", "2022"),
        where="general-disjoint-overlap",
    )
    assert set(disjoint_overlap["r_p"]) == {"FR", "DE"}


def test_merge_wide_frames_returns_identifier_only_when_all_prepared_frames_empty() -> None:
    out = merge_wide_frames(
        frames=[pd.DataFrame(index=pd.Index(["FR"], name="r_p"))],
        identifier_columns=("r_p",),
        year_columns=(),
        where="merge-empty-prepared",
    )
    assert list(out.columns) == ["r_p"]
    assert out.empty


def test_normalize_existing_wide_table_and_upsert_existing_paths(tmp_path: Path) -> None:
    normalized = normalize_existing_wide_table(
        pd.DataFrame(
            {
                "l2_method": ["m"],
                "r_p": ["FR"],
                2020: [1.0],
                2021.0: [2.0],
            }
        )
    )
    assert list(normalized.columns) == ["l2_method", "r_p", "2020", "2021"]
    assert float(normalized.loc[0, "2020"]) == 1.0
    assert float(normalized.loc[0, "2021"]) == 2.0

    with pytest.raises(ValueError):
        normalize_existing_wide_table(pd.DataFrame({"l2_method": ["m"], "r_p": ["FR"]}))

    path = tmp_path / "cached_existing.csv"
    base_schema = IdentifierSchema(columns=("l2_method", "r_p"), year_columns=("2020",))
    first_batch = pd.DataFrame({"l2_method": ["m"], "r_p": ["FR"], "2020": [1.0]})
    assert (
        upsert_wide_table(
            path=path,
            frame=first_batch,
            schema=base_schema,
            refresh=True,
            output_format="csv",
        )
        is True
    )

    extended_schema = IdentifierSchema(
        columns=("l2_method", "r_p", "reference_year"),
        year_columns=("2021",),
    )
    second_batch = pd.DataFrame(
        {"l2_method": ["m"], "r_p": ["FR"], "reference_year": [2018], "2021": [2.0]}
    )
    assert (
        upsert_wide_table(
            path=path,
            frame=second_batch,
            schema=extended_schema,
            refresh=False,
            output_format="csv",
        )
        is True
    )
    written = pd.read_csv(path)
    assert list(written.columns) == ["l2_method", "r_p", "reference_year", "2020", "2021"]
    assert written.shape == (2, 5)
    cached_row = written.loc[written["reference_year"].isna()].reset_index(drop=True)
    batch_row = written.loc[written["reference_year"].notna()].reset_index(drop=True)
    assert float(cached_row.loc[0, "2020"]) == 1.0
    assert pd.isna(cached_row.loc[0, "2021"])
    assert pd.isna(cached_row.loc[0, "reference_year"])
    assert pd.isna(batch_row.loc[0, "2020"])
    assert float(batch_row.loc[0, "2021"]) == 2.0
    assert float(batch_row.loc[0, "reference_year"]) == 2018.0


def test_assert() -> None:
    dup = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError):
        assert_no_duplicate_columns(dup, where="x")


def test_validate_wide_frame_success_and_error_paths() -> None:
    schema = IdentifierSchema(columns=("l2_method", "r_p"), year_columns=("2020", "2021"))
    ok = pd.DataFrame({"l2_method": ["m"], "r_p": ["FR"], 2020: [1.0], 2021.0: [2.0]})
    out = validate_wide_frame(ok, schema)
    assert list(out.columns) == ["l2_method", "r_p", "2020", "2021"]

    with pytest.raises(ValueError):
        validate_wide_frame(pd.DataFrame({"r_p": ["FR"], "2020": [1.0]}), schema)

    with pytest.raises(ValueError):
        validate_wide_frame(
            pd.DataFrame({"l2_method": ["m"], "r_p": ["FR"], "Y2020": [1.0]}),
            schema,
        )

    with pytest.raises(ValueError):
        validate_wide_frame(
            pd.DataFrame({"l2_method": ["m"], "r_p": ["FR"], "1990": [1.0]}),
            schema,
        )

    dup = pd.DataFrame({"l2_method": ["m", "m"], "r_p": ["FR", "FR"], "2020": [1.0, 2.0]})
    with pytest.raises(ValueError):
        validate_wide_frame(dup, schema, enforce_year_contract=False)


def test_upsert_wide_table_pickle_and_parquet_paths(tmp_path: Path) -> None:
    schema = IdentifierSchema(columns=("l2_method", "r_p"), year_columns=("2020", "2021"))
    csv_path = tmp_path / "wide.csv"
    pickle_path = tmp_path / "wide.pkl"
    parquet_path = tmp_path / "wide.parquet"

    batch_2020 = pd.DataFrame({"l2_method": ["m"], "r_p": ["FR"], "2020": [1.0]})
    batch_2021 = pd.DataFrame({"l2_method": ["m"], "r_p": ["FR"], "2021": [2.0]})

    assert (
        upsert_wide_table(
            path=csv_path,
            frame=pd.DataFrame(columns=["l2_method", "r_p", "2020"]),
            schema=schema,
            refresh=False,
            output_format="csv",
        )
        is False
    )

    assert (
        upsert_wide_table(
            path=csv_path,
            frame=batch_2020,
            schema=schema,
            refresh=True,
            output_format="csv",
        )
        is True
    )
    assert (
        upsert_wide_table(
            path=csv_path,
            frame=batch_2020,
            schema=schema,
            refresh=False,
            output_format="csv",
        )
        is False
    )
    assert (
        upsert_wide_table(
            path=csv_path,
            frame=batch_2021,
            schema=schema,
            refresh=False,
            output_format="csv",
        )
        is True
    )
    assert list(pd.read_csv(csv_path).columns) == ["l2_method", "r_p", "2020", "2021"]

    assert (
        upsert_wide_table(
            path=pickle_path,
            frame=batch_2020,
            schema=schema,
            refresh=True,
            output_format="pickle",
        )
        is True
    )
    assert (
        upsert_wide_table(
            path=pickle_path,
            frame=pd.DataFrame({"l2_method": ["m"], "r_p": ["DE"], "2020": [3.0]}),
            schema=schema,
            refresh=False,
            output_format="pickle",
        )
        is True
    )
    written_pickle = pd.read_pickle(pickle_path)
    assert set(written_pickle["r_p"]) == {"FR", "DE"}

    pytest.importorskip("pyarrow")
    assert (
        upsert_wide_table(
            path=parquet_path,
            frame=batch_2020,
            schema=schema,
            refresh=True,
            output_format="parquet",
        )
        is True
    )
    assert (
        upsert_wide_table(
            path=parquet_path,
            frame=batch_2021,
            schema=schema,
            refresh=False,
            output_format="parquet",
        )
        is True
    )
    written_parquet = pd.read_parquet(parquet_path)
    assert list(written_parquet.columns) == ["l2_method", "r_p", "2020", "2021"]


def test_upsert_wide_table_writer_preserves_comma_labels_blank_values_and_precision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quoted.csv"
    schema = IdentifierSchema(columns=("l2_method", "r_p"), year_columns=("2020", "2021"))
    frame = pd.DataFrame(
        {
            "l2_method": ["PR-HR(Ecap,cum)"],
            "r_p": ["FR"],
            "2020": [1.23456789012345],
            "2021": [np.nan],
        }
    )

    assert (
        upsert_wide_table(
            path=path,
            frame=frame,
            schema=schema,
            refresh=True,
            output_format="csv",
        )
        is True
    )

    raw = path.read_text(encoding="utf-8")
    expected_value = format(1.23456789012345, ".17g")
    assert '"PR-HR(Ecap,cum)"' in raw
    assert expected_value in raw
    assert raw.endswith(",\n")
    written = pd.read_csv(path)
    assert written.loc[0, "l2_method"] == "PR-HR(Ecap,cum)"
    assert float(written.loc[0, "2020"]) == pytest.approx(1.23456789012345)
    assert pd.isna(written.loc[0, "2021"])


def test_upsert_wide_table_updated_value_for_existing_year_writes(tmp_path: Path) -> None:
    path = tmp_path / "updated.csv"
    schema = IdentifierSchema(columns=("l2_method", "r_p"), year_columns=("2020",))
    frame1 = pd.DataFrame({"l2_method": ["m"], "r_p": ["FR"], "2020": [1.0]})
    frame2 = pd.DataFrame({"l2_method": ["m"], "r_p": ["FR"], "2020": [2.0]})
    upsert_wide_table(path=path, frame=frame1, schema=schema, refresh=True, output_format="csv")

    result = upsert_wide_table(
        path=path, frame=frame2, schema=schema, refresh=False, output_format="csv"
    )
    assert result is True
    assert float(pd.read_csv(path).loc[0, "2020"]) == 2.0


def test_upsert_wide_table_preserves_existing_row_identity_columns(tmp_path: Path) -> None:
    path = tmp_path / "reference_year_union.csv"
    schema_with_reference = IdentifierSchema(
        columns=("l2_method", "r_p", "reference_year"),
        year_columns=("2020",),
    )
    schema_without_reference = IdentifierSchema(
        columns=("l2_method", "r_p"),
        year_columns=("2021",),
    )
    upsert_wide_table(
        path=path,
        frame=pd.DataFrame(
            {
                "l2_method": ["m"],
                "r_p": ["FR"],
                "reference_year": [2018],
                "2020": [1.0],
            }
        ),
        schema=schema_with_reference,
        refresh=True,
        output_format="csv",
    )

    changed = upsert_wide_table(
        path=path,
        frame=pd.DataFrame(
            {
                "l2_method": ["m"],
                "r_p": ["FR"],
                "2021": [2.0],
            }
        ),
        schema=schema_without_reference,
        refresh=False,
        output_format="csv",
    )

    written = pd.read_csv(path)
    assert changed is True
    assert list(written.columns) == ["l2_method", "r_p", "reference_year", "2020", "2021"]
    assert written.shape == (2, 5)
    with_reference = written.loc[written["reference_year"].notna()].reset_index(drop=True)
    without_reference = written.loc[written["reference_year"].isna()].reset_index(drop=True)
    assert float(with_reference.loc[0, "2020"]) == 1.0
    assert pd.isna(with_reference.loc[0, "2021"])
    assert float(with_reference.loc[0, "reference_year"]) == 2018.0
    assert pd.isna(without_reference.loc[0, "2020"])
    assert float(without_reference.loc[0, "2021"]) == 2.0


def test_prepare_allocation_frame_paths() -> None:
    spec = _spec()
    idx = pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"])
    frame = pd.DataFrame({2020: [1.0, 2.0]}, index=idx)

    out = prepare_allocation_frame(
        output_spec=spec,
        frames=[frame],
        filters={"r_p": ["FR", "US"], "s_p": ["A"], "r_c": None, "r_f": None},
        group_indices=True,
        persisted_years=[2020],
    )
    assert list(out.columns) == [
        "l1_l2_method",
        "l2_method",
        "asocc_ssp_scenario",
        "asocc_time_route",
        "r_p",
        "s_p",
        "2020",
    ]
    assert out.shape[0] == 1

    passthrough_out = prepare_allocation_frame(
        output_spec=spec,
        frames=[
            pd.DataFrame(
                {2020: [1.0]},
                index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
            )
        ],
        filters={"r_p": ["FR"], "s_p": ["A"], "r_c": None, "r_f": None},
        group_indices=True,
        persisted_years=[2020],
    )
    assert passthrough_out.shape == (1, 7)
    assert passthrough_out.loc[0, "r_p"] == "FR"
    assert passthrough_out.loc[0, "s_p"] == "A"
    assert pd.isna(passthrough_out.loc[0, "asocc_ssp_scenario"])
    assert passthrough_out.loc[0, "asocc_time_route"] == "historical"
    assert float(passthrough_out.loc[0, "2020"]) == 1.0

    wide_idx_values = [f"R{i:02d}" for i in range(11)]
    wide_grouped_out = prepare_allocation_frame(
        output_spec=spec,
        frames=[
            pd.DataFrame(
                {2020: list(range(1, 12))},
                index=pd.MultiIndex.from_tuples(
                    [(region, "A") for region in wide_idx_values],
                    names=["r_p", "s_p"],
                ),
            )
        ],
        filters={"r_p": wide_idx_values, "s_p": ["A"], "r_c": None, "r_f": None},
        group_indices=True,
        persisted_years=[2020],
    )
    assert wide_grouped_out.shape == (1, 7)
    assert wide_grouped_out.loc[0, "r_p"] == ", ".join(wide_idx_values)
    assert wide_grouped_out.loc[0, "s_p"] == "A"
    assert float(wide_grouped_out.loc[0, "2020"]) == float(sum(range(1, 12)))

    empty_out = prepare_allocation_frame(
        output_spec=spec,
        frames=[pd.DataFrame(index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]))],
        filters={
            "r_p": ["FR"],
            "s_p": ["A"],
            "r_c": ["US"],
            "r_f": None,
            "unknown_key": ["x"],
        },
        group_indices=False,
        persisted_years=[],
    )
    assert list(empty_out.columns) == [
        "r_p",
        "s_p",
    ]
    assert empty_out.empty


def test_prepare_allocation_frame_prefixed_index_normalization_and_empty_row_drop() -> None:
    spec = _spec()
    prefixed_index = pd.MultiIndex.from_tuples(
        [
            ("UT(FD)", "UT(FD)", None, "FR", "A"),
            ("UT(FD)", "UT(FD)", None, "US", "A"),
        ],
        names=["l1_l2_method", "l2_method", "l1_method", "r_p", "s_p"],
    )
    frame = pd.DataFrame({2020: [1.0, pd.NA]}, index=prefixed_index)

    out = prepare_allocation_frame(
        output_spec=spec,
        frames=[frame],
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None},
        group_indices=False,
        persisted_years=[2020],
    )

    assert out.shape == (1, 7)
    assert out.loc[0, "r_p"] == "FR"
    assert pd.isna(out.loc[0, "asocc_ssp_scenario"])
    assert out.loc[0, "asocc_time_route"] == "historical"
    assert float(out.loc[0, "2020"]) == 1.0


def test_prepare_allocation_frame_prefixed_single_identifier_index_normalizes_name() -> None:
    spec = OutputSpec(
        l1_l2_method="L1.a",
        l2_method="L1.a",
        l1_method=None,
        file_stem="table_l1_asocc",
        route=OutputRoute(
            level="L1",
            bucket="level_1",
            source="oecd_v2025",
            grouped_mode=False,
            variant_tag=None,
            ssp_scenario=None,
            lcia_method=None,
            projection_subfolder=None,
        ),
        scenario_dependent=False,
        identifier_columns=("r_p",),
    )
    prefixed_index = pd.MultiIndex.from_tuples(
        [("L1.a", "L1.a", None, "FR")],
        names=["l1_l2_method", "l2_method", "l1_method", "r_p"],
    )
    frame = pd.DataFrame({2020: [2.0]}, index=prefixed_index)

    out = prepare_allocation_frame(
        output_spec=spec,
        frames=[frame],
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None},
        group_indices=False,
        persisted_years=[2020],
    )

    assert list(out.columns) == [
        "l1_l2_method",
        "l2_method",
        "asocc_ssp_scenario",
        "asocc_time_route",
        "r_p",
        "2020",
    ]
    assert out.loc[0, "r_p"] == "FR"
    assert float(out.loc[0, "2020"]) == 2.0


def test_prepare_allocation_frame_keeps_all_method_columns_when_present() -> None:
    spec = OutputSpec(
        l1_l2_method="EG(Pop)_UT(FD)",
        l2_method="UT(FD)",
        l1_method="EG(Pop)",
        file_stem="table_l2_asocc",
        route=OutputRoute(
            level="L2",
            bucket="l2_vs_global",
            source="oecd_v2025",
            grouped_mode=False,
            variant_tag=None,
            ssp_scenario=None,
            lcia_method=None,
            projection_subfolder=None,
        ),
        scenario_dependent=False,
        identifier_columns=("r_p",),
    )
    frame = pd.DataFrame(
        {2020: [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("EG(Pop)_UT(FD)", "UT(FD)", "EG(Pop)", "FR")],
            names=["l1_l2_method", "l2_method", "l1_method", "r_p"],
        ),
    )

    out = prepare_allocation_frame(
        output_spec=spec,
        frames=[frame],
        filters={"r_p": ["FR"], "s_p": None, "r_c": None, "r_f": None},
        group_indices=False,
        persisted_years=[2020],
    )

    assert list(out.columns) == [
        "l1_l2_method",
        "l1_method",
        "l2_method",
        "asocc_ssp_scenario",
        "asocc_time_route",
        "r_p",
        "2020",
    ]
    assert out.loc[0, "l1_method"] == "EG(Pop)"
    assert out.loc[0, "l2_method"] == "UT(FD)"


def test_prepare_allocation_frame_surfaces_noncanonical_index_contracts() -> None:
    spec = _spec()

    with pytest.raises(ValueError):
        prepare_allocation_frame(
            output_spec=spec,
            frames=[pd.DataFrame({"r_p": ["FR"], "s_p": ["A"], "2020": [1.0]})],
            filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None},
            group_indices=False,
            persisted_years=[2020],
        )

    unnamed_index = pd.MultiIndex.from_tuples(
        [("UT(FD)", "UT(FD)", None, None, "FR", "A")],
        names=["l1_l2_method", None, "l1_method", "lcia_method", "r_p", "s_p"],
    )
    with pytest.raises(ValueError):
        prepare_allocation_frame(
            output_spec=spec,
            frames=[pd.DataFrame({2020: [1.0]}, index=unnamed_index)],
            filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None},
            group_indices=False,
            persisted_years=[2020],
        )

    with pytest.raises(ValueError):
        prepare_allocation_frame(
            output_spec=spec,
            frames=[pd.DataFrame({2020: [1.0]}, index=pd.Index(["FR"], name="r_p"))],
            filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None},
            group_indices=False,
            persisted_years=[2020],
        )
