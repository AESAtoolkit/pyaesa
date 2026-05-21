import pandas as pd
from pyaesa.shared.figures import variant_selection as variant_mod


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    return [
        dict(zip(columns, row, strict=True))
        for row in frame[columns].itertuples(index=False, name=None)
    ]


def test_variant_selection_contracts_cover_grouping_and_empty_paths() -> None:
    frame = pd.DataFrame(
        {
            "method": ["A", "A", "B"],
            "region": ["EU", "EU", "EU"],
            "reference_year": [2020, 2030, 2030],
            "value": [1.0, 2.0, 3.0],
        }
    )
    assert variant_mod.compression_base_columns(
        frame,
        variant_columns=("reference_year",),
        ignored_columns={"value"},
    ) == ["method"]
    assert variant_mod.base_group_key_from_row(
        pd.Series({"method": " A ", "region": None}),
        base_columns=["method", "region"],
    ) == ("A", "missing")

    empty_main, empty_compressions = variant_mod.split_variant_frames(
        frame=pd.DataFrame(),
        requested_years=[2030],
    )
    assert empty_main.empty
    assert empty_compressions == tuple()

    no_value_frame = pd.DataFrame({"year": [2030], "method": ["A"]})
    no_value_main, no_value_compressions = variant_mod.split_variant_frames(
        frame=no_value_frame,
        requested_years=[2030],
    )
    assert no_value_main.to_dict(orient="records") == [{"year": 2030, "method": "A"}]
    assert no_value_compressions == tuple()

    out_of_scope = pd.DataFrame(
        [{"method": "A", "year": 2020, "reference_year": 2010, "value": 1.0}]
    )
    passthrough_main, passthrough_compressions = variant_mod.split_variant_frames(
        frame=out_of_scope,
        requested_years=[2030],
    )
    assert passthrough_main.equals(out_of_scope)
    assert passthrough_compressions == tuple()


def test_variant_selection_compresses_single_variant_groups() -> None:
    frame = pd.DataFrame(
        [
            {"method": "A", "region": "EU", "year": 2030, "reference_year": 2020, "value": 10.0},
            {"method": "A", "region": "EU", "year": 2030, "reference_year": 2030, "value": 15.0},
            {"method": "A", "region": "EU", "year": 2030, "reference_year": 2040, "value": 30.0},
            {"method": "A", "region": "EU", "year": 2035, "reference_year": 2020, "value": 11.0},
            {"method": "A", "region": "EU", "year": 2035, "reference_year": 2030, "value": 16.0},
            {"method": "A", "region": "EU", "year": 2035, "reference_year": 2040, "value": 31.0},
            {"method": "B", "region": "EU", "year": 2030, "reference_year": 2030, "value": 8.0},
            {"method": "B", "region": "EU", "year": 2035, "reference_year": 2030, "value": 9.0},
        ]
    )

    main_frame, compressions = variant_mod.split_variant_frames(
        frame=frame,
        requested_years=[2030, 2035],
        variant_columns=("reference_year",),
    )

    assert _records(main_frame, ["method", "year", "reference_year", "value"]) == [
        {"method": "A", "year": 2030, "reference_year": 2020.0, "value": 10.0},
        {"method": "A", "year": 2030, "reference_year": 2040.0, "value": 30.0},
        {"method": "A", "year": 2035, "reference_year": 2020.0, "value": 11.0},
        {"method": "A", "year": 2035, "reference_year": 2040.0, "value": 31.0},
        {"method": "B", "year": 2030, "reference_year": 2030.0, "value": 8.0},
        {"method": "B", "year": 2035, "reference_year": 2030.0, "value": 9.0},
    ]
    assert compressions == (
        variant_mod.VariantCompression(
            column="reference_year",
            kept_values=(2020.0, 2040.0),
            filtered=True,
            base_key=("A",),
        ),
    )
    assert variant_mod.variant_footer_note(compressions, average_over_years=False)


def test_variant_selection_compresses_multi_variant_groups_and_footer_note() -> None:
    frame = pd.DataFrame(
        [
            {
                "method": "A",
                "region": "EU",
                "year": 2030,
                "reference_year": 2020,
                "l2_reuse_year": 2030,
                "value": 10.0,
            },
            {
                "method": "A",
                "region": "EU",
                "year": 2030,
                "reference_year": 2025,
                "l2_reuse_year": 2035,
                "value": 15.0,
            },
            {
                "method": "A",
                "region": "EU",
                "year": 2030,
                "reference_year": 2030,
                "l2_reuse_year": 2040,
                "value": 30.0,
            },
            {
                "method": "A",
                "region": "EU",
                "year": 2035,
                "reference_year": 2020,
                "l2_reuse_year": 2030,
                "value": 11.0,
            },
            {
                "method": "A",
                "region": "EU",
                "year": 2035,
                "reference_year": 2025,
                "l2_reuse_year": 2035,
                "value": 16.0,
            },
            {
                "method": "A",
                "region": "EU",
                "year": 2035,
                "reference_year": 2030,
                "l2_reuse_year": 2040,
                "value": 31.0,
            },
        ]
    )

    main_frame, compressions = variant_mod.split_variant_frames(
        frame=frame,
        requested_years=[2030, 2035],
        variant_columns=("reference_year", "l2_reuse_year"),
    )

    assert _records(
        main_frame,
        ["method", "year", "reference_year", "l2_reuse_year", "value"],
    ) == [
        {
            "method": "A",
            "year": 2030,
            "reference_year": 2020.0,
            "l2_reuse_year": 2030.0,
            "value": 10.0,
        },
        {
            "method": "A",
            "year": 2030,
            "reference_year": 2030.0,
            "l2_reuse_year": 2040.0,
            "value": 30.0,
        },
        {
            "method": "A",
            "year": 2035,
            "reference_year": 2020.0,
            "l2_reuse_year": 2030.0,
            "value": 11.0,
        },
        {
            "method": "A",
            "year": 2035,
            "reference_year": 2030.0,
            "l2_reuse_year": 2040.0,
            "value": 31.0,
        },
    ]
    assert compressions == (
        variant_mod.VariantCompression(
            column="reference_year",
            kept_values=(2020.0, 2030.0),
            filtered=True,
            base_key=tuple(),
        ),
        variant_mod.VariantCompression(
            column="l2_reuse_year",
            kept_values=(2030.0, 2040.0),
            filtered=True,
            base_key=tuple(),
        ),
    )
    assert variant_mod.variant_footer_note(compressions, average_over_years=True)
    assert variant_mod.variant_footer_note(
        compressions,
        average_over_years=True,
        display_aliases={"l2_reuse_year": "l2_reuse_year"},
    )


def test_variant_footer_note_and_scope_text_cover_remaining_branches() -> None:
    assert variant_mod.variant_footer_note(tuple(), average_over_years=False) is None
    assert variant_mod._variant_scope_text({"reference_year"})
    assert variant_mod._variant_scope_text({"l2_reuse_year"})
    assert variant_mod._variant_scope_text({"method"}) is None

    compressions = (
        variant_mod.VariantCompression(
            column="reference_year",
            kept_values=(2020, 2030),
            filtered=False,
            base_key=("A",),
        ),
    )
    assert variant_mod.variant_footer_note(compressions, average_over_years=True)
    invalid_compressions = (
        variant_mod.VariantCompression(
            column="method_variant",
            kept_values=("A", "B"),
            filtered=True,
            base_key=("A",),
        ),
    )
    assert variant_mod.variant_footer_note(invalid_compressions, average_over_years=False) is None


def test_variant_selection_without_year_column_and_two_value_single_variant() -> None:
    no_year_frame = pd.DataFrame(
        [
            {"method": "A", "reference_year": 2020, "value": 1.0},
            {"method": "A", "reference_year": 2030, "value": 3.0},
            {"method": "B", "reference_year": 2020, "value": 2.0},
        ]
    )
    main_frame, compressions = variant_mod.split_variant_frames(
        frame=no_year_frame,
        requested_years=[2030],
        variant_columns=("reference_year",),
    )

    assert main_frame.equals(no_year_frame)
    assert compressions == (
        variant_mod.VariantCompression(
            column="reference_year",
            kept_values=(2020, 2030),
            filtered=False,
            base_key=("A",),
        ),
    )


def test_variant_selection_public_contracts_cover_null_single_and_partial_combo_paths() -> None:
    frame = pd.DataFrame({"method": ["A"], "value": [1.0]})
    main_frame, compressions = variant_mod.split_variant_frames(
        frame=frame,
        requested_years=[2030],
        variant_columns=("reference_year",),
    )
    assert main_frame.equals(frame)
    assert compressions == tuple()

    same_variant_frame = pd.DataFrame(
        {
            "method": ["A", "A"],
            "reference_year": [2020, 2020],
            "value": [1.0, 2.0],
        }
    )
    same_variant, compressions = variant_mod.split_variant_frames(
        frame=same_variant_frame,
        requested_years=[2030],
        variant_columns=("reference_year",),
    )
    assert same_variant.equals(same_variant_frame)
    assert compressions == tuple()

    null_variant_frame = pd.DataFrame(
        {
            "method": ["A"],
            "reference_year": [None],
            "value": [1.0],
        }
    )
    null_variant, compressions = variant_mod.split_variant_frames(
        frame=null_variant_frame,
        requested_years=[2030],
        variant_columns=("reference_year",),
    )
    assert null_variant.equals(null_variant_frame)
    assert compressions == tuple()

    repeated_variant_frame = pd.DataFrame(
        {
            "method": ["A", "A"],
            "reference_year": [2020, 2020],
            "value": [1.0, 3.0],
        }
    )
    repeated_variant, compressions = variant_mod.split_variant_frames(
        frame=repeated_variant_frame,
        requested_years=[2030],
        variant_columns=("reference_year",),
    )
    assert repeated_variant.equals(repeated_variant_frame)
    assert compressions == tuple()

    two_value_variant_frame = pd.DataFrame(
        {
            "method": ["A", "A"],
            "reference_year": [2020, 2030],
            "value": [1.0, 3.0],
        }
    )
    kept_frame, single_compressions = variant_mod.split_variant_frames(
        frame=two_value_variant_frame,
        requested_years=[2030],
        variant_columns=("reference_year",),
    )
    assert kept_frame.equals(two_value_variant_frame)
    assert single_compressions == (
        variant_mod.VariantCompression(
            column="reference_year",
            kept_values=(2020, 2030),
            filtered=False,
            base_key=tuple(),
        ),
    )

    incomplete_combo_frame = pd.DataFrame(
        {
            "method": ["A", "A"],
            "reference_year": [2020, None],
            "l2_reuse_year": [None, 2030],
            "value": [1.0, 2.0],
        }
    )
    incomplete_combo, compressions = variant_mod.split_variant_frames(
        frame=incomplete_combo_frame,
        variant_columns=("reference_year", "l2_reuse_year"),
        requested_years=[2030],
    )
    assert incomplete_combo.equals(incomplete_combo_frame)
    assert compressions == tuple()


def test_variant_selection_ignores_series_label_when_grouping_method_families() -> None:
    frame = pd.DataFrame(
        [
            {
                "l1_l2_method": "AR(E^{CBA_TD})",
                "series_label": "AR(E^{CBA_TD}), ref_year=1995",
                "year": 2022,
                "reference_year": 1995,
                "value": 0.20,
            },
            {
                "l1_l2_method": "AR(E^{CBA_TD})",
                "series_label": "AR(E^{CBA_TD}), ref_year=1995",
                "year": 2023,
                "reference_year": 1995,
                "value": 0.21,
            },
            {
                "l1_l2_method": "AR(E^{CBA_TD})",
                "series_label": "AR(E^{CBA_TD}), ref_year=2022",
                "year": 2022,
                "reference_year": 2022,
                "value": 0.55,
            },
            {
                "l1_l2_method": "AR(E^{CBA_TD})",
                "series_label": "AR(E^{CBA_TD}), ref_year=2022",
                "year": 2023,
                "reference_year": 2022,
                "value": 0.56,
            },
        ]
    )

    main_frame, compressions = variant_mod.split_variant_frames(
        frame=frame,
        requested_years=[2022, 2023],
        variant_columns=("reference_year",),
    )

    assert main_frame.equals(frame)
    assert compressions == (
        variant_mod.VariantCompression(
            column="reference_year",
            kept_values=(1995, 2022),
            filtered=False,
            base_key=tuple(),
        ),
    )

    single_combo_frame = pd.DataFrame(
        {
            "method": ["A"],
            "reference_year": [2020],
            "l2_reuse_year": [2030],
            "value": [1.0],
        }
    )
    single_combo, single_combo_compressions = variant_mod.split_variant_frames(
        frame=single_combo_frame,
        variant_columns=("reference_year", "l2_reuse_year"),
        requested_years=[2030],
    )
    assert single_combo.equals(single_combo_frame)
    assert single_combo_compressions == tuple()

    two_combo_frame = pd.DataFrame(
        {
            "method": ["A", "A"],
            "reference_year": [2020, 2030],
            "l2_reuse_year": [2030, 2040],
            "value": [1.0, 3.0],
        }
    )
    filtered_frame, compressions = variant_mod.split_variant_frames(
        frame=two_combo_frame,
        variant_columns=("reference_year", "l2_reuse_year"),
        requested_years=[2030],
    )
    assert filtered_frame.equals(two_combo_frame)
    assert compressions == (
        variant_mod.VariantCompression(
            column="reference_year",
            kept_values=(2020, 2030),
            filtered=False,
            base_key=tuple(),
        ),
        variant_mod.VariantCompression(
            column="l2_reuse_year",
            kept_values=(2030, 2040),
            filtered=False,
            base_key=tuple(),
        ),
    )
